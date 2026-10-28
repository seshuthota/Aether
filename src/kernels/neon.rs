//! ARM NEON (AArch64 Advanced SIMD) f32 kernels.
//!
//! A NEON register holds 4 f32 lanes (`float32x4_t`). `vld1q_f32` loads 4
//! consecutive floats, `vfmaq_f32(acc, a, b)` computes `acc + a*b` on all 4
//! lanes in one fused instruction (one rounding instead of two), and
//! `vaddvq_f32` adds the 4 lanes of a register together.
//!
//! The matrix-vector product is decode's main cost. It is written here in
//! three steps, each benchmarked by `examples/gemv_bench.rs`; the engine uses
//! the last:
//!
//! 1. [`gemv_one_acc`]: 4 lanes, but every FMA of a row adds into the same
//!    register, so each must wait for the previous one (FMA latency, ~4
//!    cycles). Four times the scalar loop's work per step, same chain.
//! 2. [`gemv_row_4acc`]: four independent accumulators per row, so four FMAs
//!    are in flight. Combined only at the end of the row.
//! 3. [`gemv`]: four rows at once with two accumulators each (8 chains). Each
//!    loaded `x` vector feeds four rows, so loads per FMA drop from 2 to 1.25,
//!    and there is enough independent work to keep memory busy.
//!
//! Summing in a different order than the scalar loop (and with FMA's single
//! rounding) gives slightly different bits; results are checked against
//! numpy with a tolerance, and stories15M against llama2.c by tolerance and
//! by identical greedy tokens.
//!
//! Every function here requires `x.len() == n` and `w.len() == out.len() * n`
//! (checked) and handles any `n`, including sizes that are not multiples of
//! 4 or 8: those tails are where SIMD bugs live.

use std::arch::aarch64::*;

fn check(out: &[f32], x: &[f32], w: &[f32]) -> usize {
    assert_eq!(w.len(), out.len() * x.len(), "W must be out.len() x x.len()");
    x.len()
}

/// Step 1: one 4-lane accumulator per row.
pub fn gemv_one_acc(out: &mut [f32], x: &[f32], w: &[f32]) {
    let n = check(out, x, w);
    let n4 = n / 4 * 4;
    for (o, row) in out.iter_mut().zip(w.chunks_exact(n)) {
        // SAFETY: every load reads 4 floats starting at j < n4 <= n, inside
        // `row` and `x` (both length n).
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for j in (0..n4).step_by(4) {
                acc = vfmaq_f32(acc, vld1q_f32(row.as_ptr().add(j)), vld1q_f32(x.as_ptr().add(j)));
            }
            let mut s = vaddvq_f32(acc);
            for j in n4..n {
                s += row[j] * x[j];
            }
            *o = s;
        }
    }
}

/// Dot product of one row with `x`: four accumulators over 16-float steps,
/// then one accumulator over the remaining 4-float steps, then scalar.
///
/// # Safety
/// `row` and `x` must both point to `n` readable floats.
#[inline(always)]
unsafe fn dot_row(row: *const f32, x: *const f32, n: usize) -> f32 {
    unsafe {
        let mut a = [vdupq_n_f32(0.0); 4];
        let n16 = n / 16 * 16;
        for j in (0..n16).step_by(16) {
            for (k, acc) in a.iter_mut().enumerate() {
                *acc = vfmaq_f32(*acc, vld1q_f32(row.add(j + 4 * k)), vld1q_f32(x.add(j + 4 * k)));
            }
        }
        let n4 = n / 4 * 4;
        for j in (n16..n4).step_by(4) {
            a[0] = vfmaq_f32(a[0], vld1q_f32(row.add(j)), vld1q_f32(x.add(j)));
        }
        let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])));
        for j in n4..n {
            s += *row.add(j) * *x.add(j);
        }
        s
    }
}

/// Step 2: four independent accumulators per row.
pub fn gemv_row_4acc(out: &mut [f32], x: &[f32], w: &[f32]) {
    let n = check(out, x, w);
    for (o, row) in out.iter_mut().zip(w.chunks_exact(n)) {
        // SAFETY: `row` and `x` both hold n floats.
        *o = unsafe { dot_row(row.as_ptr(), x.as_ptr(), n) };
    }
}

/// Step 3 (the engine's GEMV): four rows at a time, two accumulators per row.
pub fn gemv(out: &mut [f32], x: &[f32], w: &[f32]) {
    let n = check(out, x, w);
    let d = out.len();
    let d4 = d / 4 * 4;
    let (n8, n4) = (n / 8 * 8, n / 4 * 4);
    let xp = x.as_ptr();
    for i in (0..d4).step_by(4) {
        // SAFETY: rows i..i+4 exist (i + 4 <= d4 <= d), so each row pointer
        // has n floats; every vector load reads 4 floats at j + 4 <= n.
        unsafe {
            let r = [0, 1, 2, 3].map(|k| w.as_ptr().add((i + k) * n));
            let mut lo = [vdupq_n_f32(0.0); 4];
            let mut hi = [vdupq_n_f32(0.0); 4];
            for j in (0..n8).step_by(8) {
                let (x0, x1) = (vld1q_f32(xp.add(j)), vld1q_f32(xp.add(j + 4)));
                for k in 0..4 {
                    lo[k] = vfmaq_f32(lo[k], vld1q_f32(r[k].add(j)), x0);
                    hi[k] = vfmaq_f32(hi[k], vld1q_f32(r[k].add(j + 4)), x1);
                }
            }
            if n4 > n8 {
                let x0 = vld1q_f32(xp.add(n8));
                for k in 0..4 {
                    lo[k] = vfmaq_f32(lo[k], vld1q_f32(r[k].add(n8)), x0);
                }
            }
            for k in 0..4 {
                let mut s = vaddvq_f32(vaddq_f32(lo[k], hi[k]));
                for j in n4..n {
                    s += *r[k].add(j) * *xp.add(j);
                }
                out[i + k] = s;
            }
        }
    }
    for (i, o) in out.iter_mut().enumerate().skip(d4) {
        // SAFETY: row i holds n floats, x holds n floats.
        *o = unsafe { dot_row(w.as_ptr().add(i * n), xp, n) };
    }
}

/// Batched product for prefill (GEMM): `out[t] = W · x[t]` for `t` tokens.
/// `x` is `[t, n]`, `out` is `[t, d]`, `W` is `[d, n]`, all row-major.
///
/// Decode multiplies W by one vector, so every weight is used once per load
/// (bandwidth-bound). With T vectors each weight can be reused T times, so
/// the work becomes compute-bound if the kernel reuses loads well. The core
/// is a 4-row × 4-token tile: per 4-float step it loads 4 weight vectors and
/// 4 token vectors (8 loads) and does 16 FMAs, one per (row, token) pair,
/// each into its own accumulator (16 registers). Rows and tokens outside
/// whole tiles fall back to the GEMV paths.
///
/// Loop order: for each block of 4 weight rows (4·n floats, a few KB to
/// ~80 KB, small enough to stay in L1/L2), sweep all token blocks, so each
/// weight row is fetched from memory once per call rather than once per
/// token.
pub fn gemm(out: &mut [f32], x: &[f32], w: &[f32], t: usize) {
    let n = x.len() / t.max(1);
    assert_eq!(x.len(), t * n, "x must be t x n");
    let d = out.len() / t.max(1);
    assert_eq!(out.len(), t * d, "out must be t x d");
    assert_eq!(w.len(), d * n, "W must be d x n");
    // SAFETY: `out` holds t·d floats and nothing else can access it.
    unsafe { gemm_rows(out.as_mut_ptr(), d, 0..d, x, w, t) }
}

/// [`gemm`] restricted to output rows `rows` (all tokens): writes
/// `out[tt·d + i]` for `i` in `rows`. Threads given disjoint row ranges
/// can run it on the same `out` at once.
///
/// # Safety
/// `out` must be valid for `t·d` floats, `x` must be `t × n` and `w`
/// `d × n` with `rows.end <= d`, and no other thread may access the
/// elements this call writes while it runs.
pub unsafe fn gemm_rows(out: *mut f32, d: usize, rows: std::ops::Range<usize>, x: &[f32], w: &[f32], t: usize) {
    let n = x.len() / t.max(1);
    assert!(rows.end <= d && w.len() == d * n);
    let (r0, k) = (rows.start, rows.len());
    let (k4, t4, n4) = (k / 4 * 4, t / 4 * 4, n / 4 * 4);
    let (wp, xp) = (w.as_ptr(), x.as_ptr());

    for i in (r0..r0 + k4).step_by(4) {
        for tt in (0..t4).step_by(4) {
            // SAFETY: rows i..i+4 < rows.end <= d and tokens tt..tt+4 <= t
            // exist; every vector load reads 4 floats at j + 4 <= n4 <= n;
            // the writes are inside out's t·d floats, in this call's rows.
            unsafe {
                let wr = [0, 1, 2, 3].map(|k| wp.add((i + k) * n));
                let xr = [0, 1, 2, 3].map(|k| xp.add((tt + k) * n));
                let mut acc = [[vdupq_n_f32(0.0); 4]; 4]; // acc[row][token]
                for j in (0..n4).step_by(4) {
                    let wv = wr.map(|p| vld1q_f32(p.add(j)));
                    let xv = xr.map(|p| vld1q_f32(p.add(j)));
                    for r in 0..4 {
                        for c in 0..4 {
                            acc[r][c] = vfmaq_f32(acc[r][c], wv[r], xv[c]);
                        }
                    }
                }
                for r in 0..4 {
                    for c in 0..4 {
                        let mut s = vaddvq_f32(acc[r][c]);
                        for j in n4..n {
                            s += *wr[r].add(j) * *xr[c].add(j);
                        }
                        *out.add((tt + c) * d + i + r) = s;
                    }
                }
            }
        }
    }
    // Tokens outside whole tiles: one GEMV each over the tiled rows (their
    // outputs are contiguous within that token's row of `out`).
    for tt in t4..t {
        // SAFETY: the k4 floats from out[tt·d + r0] are this call's rows.
        let o = unsafe { std::slice::from_raw_parts_mut(out.add(tt * d + r0), k4) };
        gemv(o, &x[tt * n..(tt + 1) * n], &w[r0 * n..(r0 + k4) * n]);
    }
    // Rows outside whole tiles: one dot product per (row, token).
    for i in r0 + k4..rows.end {
        for tt in 0..t {
            // SAFETY: row i and token tt each hold n floats; the write is in
            // this call's rows.
            unsafe { *out.add(tt * d + i) = dot_row(wp.add(i * n), xp.add(tt * n), n) };
        }
    }
}

/// Attention of one query head over positions `0..=pos`; same contract as
/// [`super::scalar::attention_head`]. It is [`attention_group`] with one
/// head, so the two give the same bits.
#[inline(never)] // keep a separate symbol so profiles attribute time per kernel
pub fn attention_head<'a>(
    out: &mut [f32],
    q: &[f32],
    keys: impl Fn(usize) -> &'a [f32],
    values: impl Fn(usize) -> &'a [f32],
    pos: usize,
    scores: &mut [f32],
) {
    attention_group(out, q, q.len(), keys, values, pos, scores)
}

/// Positions per block in [`group_fixed`]: 32 keys of 64 floats are 8 KB,
/// so a block stays in L1 while every head of the group uses it.
const POS_BLOCK: usize = 32;

/// [`attention_group`] for head size `4·V`, with the vectors of one head
/// in registers: the query while its scores are computed (four positions at
/// a time, two accumulators each, the four sums reduced together by two
/// pairwise adds), and the output while the values are summed (each
/// multiply-add then loads only the value). Positions go in blocks of
/// [`POS_BLOCK`], heads inside a block.
///
/// # Safety
/// `q` and `out` must hold g·4V floats, `scores` g·(pos + 1), and every
/// `keys(t)` / `values(t)` 4V floats (checked by the caller).
#[inline(always)]
unsafe fn group_fixed<'a, const V: usize>(
    out: &mut [f32],
    q: &[f32],
    keys: &impl Fn(usize) -> &'a [f32],
    values: &impl Fn(usize) -> &'a [f32],
    n: usize,
    scores: &mut [f32],
) {
    let (hs, g) = (4 * V, q.len() / (4 * V));
    let scale = 1.0 / (hs as f32).sqrt();
    // SAFETY (whole body): every pointer below indexes q / out (g·hs
    // floats), a key or value row (hs floats) or scores (g·n floats) within
    // the bounds the caller guarantees.
    unsafe {
        let load = |p: *const f32| -> [float32x4_t; V] { std::array::from_fn(|j| vld1q_f32(p.add(4 * j))) };
        for tb in (0..n).step_by(POS_BLOCK) {
            let te = (tb + POS_BLOCK).min(n);
            for h in 0..g {
                let qv = load(q.as_ptr().add(h * hs));
                let sh = scores.as_mut_ptr().add(h * n);
                let mut t = tb;
                while t + 4 <= te {
                    let mut acc = [[vdupq_n_f32(0.0); 2]; 4];
                    for (p, a) in acc.iter_mut().enumerate() {
                        let k = keys(t + p).as_ptr();
                        for j in 0..V {
                            a[j % 2] = vfmaq_f32(a[j % 2], vld1q_f32(k.add(4 * j)), qv[j]);
                        }
                    }
                    let s = acc.map(|a| vaddq_f32(a[0], a[1]));
                    let r = vpaddq_f32(vpaddq_f32(s[0], s[1]), vpaddq_f32(s[2], s[3]));
                    vst1q_f32(sh.add(t), vmulq_n_f32(r, scale));
                    t += 4;
                }
                for t in t..te {
                    let k = keys(t).as_ptr();
                    let mut a = [vdupq_n_f32(0.0); 2];
                    for j in 0..V {
                        a[j % 2] = vfmaq_f32(a[j % 2], vld1q_f32(k.add(4 * j)), qv[j]);
                    }
                    *sh.add(t) = vaddvq_f32(vaddq_f32(a[0], a[1])) * scale;
                }
            }
        }
        for s in scores[..g * n].chunks_exact_mut(n) {
            softmax(s);
        }
        for tb in (0..n).step_by(POS_BLOCK) {
            let te = (tb + POS_BLOCK).min(n);
            for h in 0..g {
                let op = out.as_mut_ptr().add(h * hs);
                let mut o = if tb == 0 { [vdupq_n_f32(0.0); V] } else { load(op) };
                for t in tb..te {
                    let (a, v) = (*scores.get_unchecked(h * n + t), values(t).as_ptr());
                    for (j, oj) in o.iter_mut().enumerate() {
                        *oj = vfmaq_n_f32(*oj, vld1q_f32(v.add(4 * j)), a);
                    }
                }
                for (j, oj) in o.iter().enumerate() {
                    vst1q_f32(op.add(4 * j), *oj);
                }
            }
        }
    }
}

/// `row`, after checking it holds `hs` floats.
#[inline(always)]
fn checked(row: &[f32], hs: usize) -> &[f32] {
    assert_eq!(row.len(), hs);
    row
}

/// [`attention_head`] for the `g` query heads that read the same keys and
/// values (grouped-query attention: Qwen2.5-0.5B has 7 per K/V head). `q`
/// and `out` hold the g heads one after another (g·hs floats), `scores`
/// at least g·(pos + 1). Each key and value row is loaded once and used by
/// all g heads while it sits in L1, instead of being read again from L2
/// for every head. Head sizes 64 and 48 (every model here) take
/// [`group_fixed`]; others a general loop of [`dot_row`]s. A head's result
/// does not depend on the other heads of its group, so it matches
/// [`attention_head`] bit for bit.
#[inline(never)] // keep a separate symbol so profiles attribute time per kernel
pub fn attention_group<'a>(
    out: &mut [f32],
    q: &[f32],
    hs: usize,
    keys: impl Fn(usize) -> &'a [f32],
    values: impl Fn(usize) -> &'a [f32],
    pos: usize,
    scores: &mut [f32],
) {
    let g = q.len() / hs;
    assert!(g * hs == q.len() && out.len() == q.len());
    let n = pos + 1;
    assert!(scores.len() >= g * n);
    if hs == 64 || hs == 48 {
        let (k, v) = (|t: usize| checked(keys(t), hs), |t: usize| checked(values(t), hs));
        // SAFETY: lengths checked above and in `checked`.
        unsafe {
            if hs == 64 { group_fixed::<16>(out, q, &k, &v, n, scores) } else { group_fixed::<12>(out, q, &k, &v, n, scores) }
        }
        return;
    }
    let scores = &mut scores[..g * n];
    let scale = 1.0 / (hs as f32).sqrt();
    for t in 0..n {
        let k = keys(t);
        assert_eq!(k.len(), hs);
        for h in 0..g {
            // SAFETY: k holds hs floats (checked) and q holds g·hs, so head
            // h's query q[h·hs..(h+1)·hs] is in bounds.
            scores[h * n + t] = unsafe { dot_row(k.as_ptr(), q.as_ptr().add(h * hs), hs) } * scale;
        }
    }
    for s in scores.chunks_exact_mut(n) {
        softmax(s);
    }
    out.fill(0.0);
    let hs4 = hs / 4 * 4;
    for t in 0..n {
        let v = values(t);
        assert_eq!(v.len(), hs);
        for (h, oh) in out.chunks_exact_mut(hs).enumerate() {
            let a = scores[h * n + t];
            let op = oh.as_mut_ptr();
            // SAFETY: oh and v hold hs floats (checked); vector accesses
            // stay below hs4 <= hs.
            unsafe {
                let av = vdupq_n_f32(a);
                for j in (0..hs4).step_by(4) {
                    vst1q_f32(op.add(j), vfmaq_f32(vld1q_f32(op.add(j)), av, vld1q_f32(v.as_ptr().add(j))));
                }
            }
            for j in hs4..hs {
                oh[j] += a * v[j];
            }
        }
    }
}

/// `e^x` on four lanes, for `x <= 0`. Writes `x = n·ln2 + r` with
/// `|r| <= ln2/2`, takes `e^r` from a degree-5 polynomial (the coefficients
/// of ARM's optimized-routines `v_expf`) and multiplies by `2^n`, built
/// directly in the exponent bits. Max relative error 1.9e-7 on [-87, 0]
/// (numpy, against float64 `exp`; numpy's own float32 `exp`: 2.0e-7).
/// Below -87, where `e^x` leaves f32's normal range, the result is 0.
#[inline(always)]
fn exp_neg(x: float32x4_t) -> float32x4_t {
    let c = f32::from_bits;
    let (inv_ln2, ln2_hi, ln2_lo) = (c(0x3fb8_aa3b), c(0x3f31_7200), c(0x35bf_be8e));
    let (c0, c1, c2, c3, c4) = (c(0x3c07_2010), c(0x3d2b_9f17), c(0x3e2a_af33), c(0x3eff_fedb), c(0x3f7f_fff6));
    // SAFETY: register-only NEON ops.
    unsafe {
        let n = vrndnq_f32(vmulq_n_f32(x, inv_ln2));
        let r = vfmaq_n_f32(vfmaq_n_f32(x, n, -ln2_hi), n, -ln2_lo);
        // 2^n: n + 127 in the exponent field (n >= -126 for x >= -87).
        let scale = vreinterpretq_f32_s32(vaddq_s32(vshlq_n_s32::<23>(vcvtq_s32_f32(n)), vdupq_n_s32(0x3f80_0000)));
        // e^r - 1 = c4·r + c3·r² + c2·r³ + c1·r⁴ + c0·r⁵, as two halves.
        let r2 = vmulq_f32(r, r);
        let hi = vfmaq_f32(vfmaq_n_f32(vdupq_n_f32(c3), r, c2), vfmaq_n_f32(vdupq_n_f32(c1), r, c0), r2);
        let poly = vfmaq_f32(vmulq_n_f32(r, c4), hi, r2);
        let y = vfmaq_f32(scale, poly, scale);
        vbslq_f32(vcltq_f32(x, vdupq_n_f32(-87.0)), vdupq_n_f32(0.0), y)
    }
}

/// SwiGLU in place, four lanes at a time (see [`super::scalar::swiglu`]).
/// `sigmoid(v)` uses only `e^(−|v|)`, which [`exp_neg`] covers: it is
/// `1 / (1 + e)` for `v >= 0` and `e / (1 + e)` for `v < 0`. The last 0–3
/// values go through the scalar version.
#[inline(never)] // keep a separate symbol so profiles attribute time per kernel
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    let n4 = gate.len() / 4 * 4;
    // SAFETY: vector accesses stay below n4 <= both lengths.
    unsafe {
        let (g, u) = (gate.as_mut_ptr(), up.as_ptr());
        let one = vdupq_n_f32(1.0);
        for i in (0..n4).step_by(4) {
            let v = vld1q_f32(g.add(i));
            let e = exp_neg(vnegq_f32(vabsq_f32(v)));
            let num = vbslq_f32(vcgeq_f32(v, vdupq_n_f32(0.0)), one, e);
            let sig = vdivq_f32(num, vaddq_f32(one, e));
            vst1q_f32(g.add(i), vmulq_f32(vmulq_f32(v, sig), vld1q_f32(u.add(i))));
        }
    }
    super::scalar::swiglu(&mut gate[n4..], &up[n4..]);
}

/// Softmax in place, four lanes at a time: max, then `e^(x - max)` with
/// [`exp_neg`] and their sum, then a division by the sum. The last 0-3
/// values go through the same vector code, padded with -inf (whose `e^x` is
/// 0, so the sum is unchanged).
pub fn softmax(x: &mut [f32]) {
    let n4 = x.len() / 4 * 4;
    let tail = |x: &[f32]| {
        let mut t = [f32::NEG_INFINITY; 4];
        t[..x.len() - n4].copy_from_slice(&x[n4..]);
        t
    };
    // SAFETY: vector accesses stay below n4 <= x.len(); the tail goes
    // through a local 4-float array.
    unsafe {
        let p = x.as_mut_ptr();
        let mut m = vld1q_f32(tail(x).as_ptr());
        for i in (0..n4).step_by(4) {
            m = vmaxq_f32(m, vld1q_f32(p.add(i)));
        }
        let max = vdupq_n_f32(vmaxvq_f32(m));
        let mut sum = vdupq_n_f32(0.0);
        for i in (0..n4).step_by(4) {
            let e = exp_neg(vsubq_f32(vld1q_f32(p.add(i)), max));
            vst1q_f32(p.add(i), e);
            sum = vaddq_f32(sum, e);
        }
        let mut t = tail(x);
        let e = exp_neg(vsubq_f32(vld1q_f32(t.as_ptr()), max));
        sum = vaddq_f32(sum, e);
        let sum = vdupq_n_f32(vaddvq_f32(sum));
        for i in (0..n4).step_by(4) {
            vst1q_f32(p.add(i), vdivq_f32(vld1q_f32(p.add(i)), sum));
        }
        vst1q_f32(t.as_mut_ptr(), vdivq_f32(e, sum));
        let k = x.len() - n4;
        x[n4..].copy_from_slice(&t[..k]);
    }
}
