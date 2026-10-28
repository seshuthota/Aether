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
/// [`super::scalar::attention_head`].
///
/// The scalar version's score loop is one long chain of dependent adds per
/// key (the compiler may not reorder float sums, so it cannot vectorize it):
/// at depth 512 that chain dominated Qwen2.5's decode step. Here each score
/// is [`dot_row`] (four independent 4-lane chains), and the weighted sum of
/// values adds 4 lanes of `out` per FMA. Softmax stays scalar (`exp`).
#[inline(never)] // keep a separate symbol so profiles attribute time per kernel
pub fn attention_head<'a>(
    out: &mut [f32],
    q: &[f32],
    keys: impl Fn(usize) -> &'a [f32],
    values: impl Fn(usize) -> &'a [f32],
    pos: usize,
    scores: &mut [f32],
) {
    let hs = q.len();
    assert_eq!(out.len(), hs);
    let scores = &mut scores[..=pos];
    let scale = 1.0 / (hs as f32).sqrt();
    for (t, s) in scores.iter_mut().enumerate() {
        let k = keys(t);
        assert_eq!(k.len(), hs);
        // SAFETY: k and q both hold hs floats (checked).
        *s = unsafe { dot_row(k.as_ptr(), q.as_ptr(), hs) } * scale;
    }
    super::scalar::softmax(scores);
    out.fill(0.0);
    let hs4 = hs / 4 * 4;
    let op = out.as_mut_ptr();
    for (t, &a) in scores.iter().enumerate() {
        let v = values(t);
        assert_eq!(v.len(), hs);
        // SAFETY: out and v hold hs floats (checked); vector accesses stay
        // below hs4 <= hs.
        unsafe {
            let av = vdupq_n_f32(a);
            for j in (0..hs4).step_by(4) {
                vst1q_f32(op.add(j), vfmaq_f32(vld1q_f32(op.add(j)), av, vld1q_f32(v.as_ptr().add(j))));
            }
        }
        for j in hs4..hs {
            out[j] += a * v[j];
        }
    }
}

/// [`attention_head`] for the `g` query heads that read the same keys and
/// values (grouped-query attention: Qwen2.5-0.5B has 7 per K/V head). `q`
/// and `out` hold the g heads one after another (g·hs floats), `scores`
/// at least g·(pos + 1). Each key and value row is loaded once and used by
/// all g heads while it sits in L1, instead of being read again from L2
/// for every head. Each head's arithmetic is that of [`attention_head`],
/// in the same order, so the results are bit-identical.
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
        super::scalar::softmax(s);
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
