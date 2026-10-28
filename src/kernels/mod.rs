//! The engine's op functions. Model code calls these and never touches
//! memory layouts or SIMD itself; faster versions (NEON, threads, quantized)
//! replace the bodies here without changing the callers.
//!
//! On AArch64 the hot ops dispatch to `neon`; elsewhere (the x86 host that
//! runs `cargo test` without a phone) they fall back to `scalar`.

#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "aarch64")]
pub mod neon_quant;
pub mod scalar;

use crate::pool::{SharedOut, ThreadPool};
use crate::quant::{BlockQ8_0, QK};
use crate::weight::Weight;
pub use scalar::{add_inplace, argmax, rmsnorm, rmsnorm_inplace, rope_adjacent, rope_halves, softmax, swiglu};

/// `out = W · x`, `W` row-major `[out.len(), x.len()]`.
#[inline]
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    neon::gemv(out, x, w);
    #[cfg(not(target_arch = "aarch64"))]
    scalar::matmul(out, x, w);
}

/// Batched `out[t] = W · x[t]` for `t` tokens (prefill). `x` is `[t, n]`,
/// `out` is `[t, d]`, `W` is `[d, n]`.
#[inline]
pub fn matmul_batch(out: &mut [f32], x: &[f32], w: &[f32], t: usize) {
    #[cfg(target_arch = "aarch64")]
    neon::gemm(out, x, w, t);
    #[cfg(not(target_arch = "aarch64"))]
    scalar::matmul_batch(out, x, w, t);
}

/// Attention of one query head over positions `0..=pos` (see
/// [`scalar::attention_head`] for the contract).
#[inline]
pub fn attention_head<'a>(
    out: &mut [f32],
    q: &[f32],
    keys: impl Fn(usize) -> &'a [f32],
    values: impl Fn(usize) -> &'a [f32],
    pos: usize,
    scores: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    neon::attention_head(out, q, keys, values, pos, scores);
    #[cfg(not(target_arch = "aarch64"))]
    scalar::attention_head(out, q, keys, values, pos, scores);
}

/// [`attention_head`] for consecutive query heads that share one K/V head:
/// `q` and `out` hold them one after another (hs floats each), `scores`
/// needs one row of `pos + 1` per head. Same results as calling
/// [`attention_head`] once per head.
#[inline]
pub fn attention_group<'a>(
    out: &mut [f32],
    q: &[f32],
    hs: usize,
    keys: impl Fn(usize) -> &'a [f32],
    values: impl Fn(usize) -> &'a [f32],
    pos: usize,
    scores: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    neon::attention_group(out, q, hs, keys, values, pos, scores);
    #[cfg(not(target_arch = "aarch64"))]
    for (oh, qh) in out.chunks_exact_mut(hs).zip(q.chunks_exact(hs)) {
        scalar::attention_head(oh, qh, &keys, &values, pos, scores);
    }
}

/// Output rows per block when splitting across threads: whole 4-row tiles,
/// and 16 floats = one 64-byte line, so threads never write the same line.
const ROW_BLOCK: usize = 16;

/// [`matmul`] split by output rows across `pool` (or on this thread when
/// `pool` is `None` or has one thread). Each row's sum is computed exactly
/// as in the single-threaded kernel, so the result is bit-identical.
pub fn matmul_on(pool: Option<&ThreadPool>, out: &mut [f32], x: &[f32], w: &[f32]) {
    let Some(pool) = pool.filter(|p| p.threads() > 1) else { return matmul(out, x, w) };
    let n = x.len();
    assert_eq!(w.len(), out.len() * n, "W must be out.len() x x.len()");
    let o = SharedOut(out.as_mut_ptr());
    let block = crate::pool::block_size(out.len(), pool.threads(), ROW_BLOCK);
    pool.for_blocks(out.len(), block, &|r| {
        // SAFETY: `for_blocks` hands each block of rows to exactly one thread.
        let mine = unsafe { std::slice::from_raw_parts_mut(o.ptr().add(r.start), r.len()) };
        matmul(mine, x, &w[r.start * n..r.end * n]);
    });
}

/// [`matmul_batch`] split by output rows (all tokens per thread) across
/// `pool`. Bit-identical to the single-threaded call.
pub fn matmul_batch_on(pool: Option<&ThreadPool>, out: &mut [f32], x: &[f32], w: &[f32], t: usize) {
    let Some(pool) = pool.filter(|p| p.threads() > 1) else { return matmul_batch(out, x, w, t) };
    let (n, d) = (x.len() / t.max(1), out.len() / t.max(1));
    assert_eq!((x.len(), out.len(), w.len()), (t * n, t * d, d * n), "shapes");
    let o = SharedOut(out.as_mut_ptr());
    pool.for_blocks(d, crate::pool::block_size(d, pool.threads(), ROW_BLOCK), &|r| {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: shapes checked above; each block of rows goes to one thread.
        unsafe {
            neon::gemm_rows(o.ptr(), d, r, x, w, t)
        };
        #[cfg(not(target_arch = "aarch64"))]
        for tt in 0..t {
            // SAFETY: token tt's outputs for these rows are contiguous and
            // belong to this thread only.
            let mine = unsafe { std::slice::from_raw_parts_mut(o.ptr().add(tt * d + r.start), r.len()) };
            scalar::matmul(mine, &x[tt * n..(tt + 1) * n], &w[r.start * n..r.end * n]);
        }
    });
}

/// Quantizes one activation vector (a multiple of 32 long) to Q8_0 for the
/// integer dot-product kernels.
fn quantize_activation(x: &[f32]) -> Vec<BlockQ8_0> {
    let mut q = vec![BlockQ8_0 { d: 0, qs: [0; QK] }; x.len() / QK];
    #[cfg(target_arch = "aarch64")]
    neon_quant::quantize_q8_0(x, &mut q);
    #[cfg(not(target_arch = "aarch64"))]
    crate::quant::quantize_q8_0(x, &mut q);
    q
}

/// One row's dot product with one quantized activation (any CPU).
#[inline]
fn dot1(w: &Weight, i: usize, x: &[BlockQ8_0]) -> f32 {
    let nb = x.len();
    match w {
        #[cfg(target_arch = "aarch64")]
        Weight::Q8_0(b) => neon_quant::dot_q8_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(target_arch = "aarch64")]
        Weight::Q4_0(b) => neon_quant::dot_q4_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(target_arch = "aarch64")]
        Weight::Q5_0(b) => neon_quant::dot_q5_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(target_arch = "aarch64")]
        Weight::Q4_K(b) => neon_quant::dot_q4_k_q8_0(&b[i * nb / 8..(i + 1) * nb / 8], x),
        #[cfg(target_arch = "aarch64")]
        Weight::Q6_K(b) => neon_quant::dot_q6_k_q8_0(&b[i * nb / 8..(i + 1) * nb / 8], x),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q8_0(b) => crate::quant::scalar::dot_q8_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q4_0(b) => crate::quant::scalar::dot_q4_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q5_0(b) => crate::quant::scalar::dot_q5_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q4_K(b) => crate::quant::scalar::dot_q4_k_q8_0(&b[i * nb / 8..(i + 1) * nb / 8], x),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q6_K(b) => crate::quant::scalar::dot_q6_k_q8_0(&b[i * nb / 8..(i + 1) * nb / 8], x),
        Weight::F32(_) => unreachable!(),
        Weight::Q8_0x4(_) | Weight::Q4_0x4(_) | Weight::Q5_0x4(_) => unreachable!("repacked rows go four at a time"),
    }
}

/// True when the 4-row integer kernels can run (AArch64 with `sdot`).
fn has_dot4() -> bool {
    #[cfg(target_arch = "aarch64")]
    return neon_quant::has_dotprod();
    #[cfg(not(target_arch = "aarch64"))]
    return false;
}

/// Rows `i..i+4` · one quantized activation. Only called when
/// [`has_dot4`] is true; for repacked weights `i` is a multiple of 4 (rows
/// are handed out in blocks of [`ROW_BLOCK`]).
#[inline]
#[allow(unused_variables)]
fn dot4(w: &Weight, i: usize, x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        let nb = x.len();
        let rows = |k: usize| (i + k) * nb..(i + k + 1) * nb;
        // SAFETY: callers check has_dot4 (dotprod present); every row slice
        // has nb blocks.
        unsafe {
            match w {
                Weight::Q8_0(b) => neon_quant::dot4_q8_0_q8_0([0, 1, 2, 3].map(|k| &b[rows(k)]), x, xd),
                Weight::Q4_0(b) => neon_quant::dot4_q4_0_q8_0([0, 1, 2, 3].map(|k| &b[rows(k)]), x, xd),
                Weight::Q5_0(b) => neon_quant::dot4_q5_0_q8_0([0, 1, 2, 3].map(|k| &b[rows(k)]), x, xd),
                Weight::Q4_K(b) => neon_quant::dot4_q4_k_q8_0([0, 1, 2, 3].map(|k| &b[(i + k) * nb / 8..(i + k + 1) * nb / 8]), x, xd),
                Weight::Q6_K(b) => neon_quant::dot4_q6_k_q8_0([0, 1, 2, 3].map(|k| &b[(i + k) * nb / 8..(i + k + 1) * nb / 8]), x, xd),
                Weight::Q8_0x4(b) => neon_quant::gemv4_q8_0x4(&b[group(i, nb)], x, xd),
                Weight::Q4_0x4(b) => neon_quant::gemv4_q4_0x4(&b[group(i, nb)], x, xd),
                Weight::Q5_0x4(b) => neon_quant::gemv4_q5_0x4(&b[group(i, nb)], x, xd),
                Weight::F32(_) => unreachable!(),
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    unreachable!("dot4 needs the AArch64 sdot kernels")
}

/// True when the `smmla` kernels can run (i8mm present).
#[cfg(target_arch = "aarch64")]
fn has_mmla() -> bool {
    neon_quant::has_i8mm()
}

/// Whether models should repack their quantized matrices four rows at a
/// time (see [`Weight::repack`]): yes where the `smmla` kernels run, which
/// then need no interleaving in their loops (decode uses the matching
/// `sdot` GEMV).
pub fn repack_weights() -> bool {
    #[cfg(target_arch = "aarch64")]
    return has_mmla() && has_dot4();
    #[cfg(not(target_arch = "aarch64"))]
    return false;
}

/// The blocks of the repacked group holding rows `i..i+4` (`i` a multiple
/// of 4), rows of `nb` blocks.
#[cfg(target_arch = "aarch64")]
fn group(i: usize, nb: usize) -> std::ops::Range<usize> {
    assert!(i.is_multiple_of(4), "repacked rows start at a multiple of 4");
    i / 4 * nb..(i / 4 + 1) * nb
}

/// Rows `rows` · the tokens of pairs `ps` (see [`neon_quant::mmla4`]). Only
/// called when [`has_mmla`] is true.
#[cfg(target_arch = "aarch64")]
#[inline]
fn mmla4(w: &Weight, rows: [usize; 4], ps: [usize; 2], xp: &[neon_quant::PairQ8], nb: usize) -> [[f32; 4]; 4] {
    let x = ps.map(|p| &xp[p * nb..(p + 1) * nb]);
    // SAFETY: callers check has_mmla (i8mm present); every row and pair
    // slice has nb blocks.
    unsafe {
        match w {
            Weight::Q8_0(b) => neon_quant::mmla4_q8_0(rows.map(|i| &b[i * nb..(i + 1) * nb]), x),
            Weight::Q4_0(b) => neon_quant::mmla4_q4_0(rows.map(|i| &b[i * nb..(i + 1) * nb]), x),
            Weight::Q5_0(b) => neon_quant::mmla4_q5_0(rows.map(|i| &b[i * nb..(i + 1) * nb]), x),
            Weight::Q4_K(_) | Weight::Q6_K(_) => unreachable!("no smmla kernel for K-quants (see has_mmla_kernel)"),
            Weight::Q8_0x4(b) => neon_quant::mmla4_q8_0x4(&b[group4(rows, nb)], x),
            Weight::Q4_0x4(b) => neon_quant::mmla4_q4_0x4(&b[group4(rows, nb)], x),
            Weight::Q5_0x4(b) => neon_quant::mmla4_q5_0x4(&b[group4(rows, nb)], x),
            Weight::F32(_) => unreachable!(),
        }
    }
}

/// Whether [`mmla4`] handles `w`'s format (K-quants go through
/// [`quant_rows`] instead).
#[cfg(target_arch = "aarch64")]
fn has_mmla_kernel(w: &Weight) -> bool {
    !matches!(w, Weight::Q4_K(_) | Weight::Q6_K(_))
}

/// [`group`] for four row indices, which must be one whole group.
#[cfg(target_arch = "aarch64")]
fn group4(rows: [usize; 4], nb: usize) -> std::ops::Range<usize> {
    assert!(rows == [0, 1, 2, 3].map(|k| rows[0] + k), "repacked rows go four at a time");
    group(rows[0], nb)
}

/// [`quant_rows`] with `smmla`: rows four at a time, and for each group of
/// four rows (which then stays in L1) all tokens four at a time, from
/// activations interleaved into token pairs once per matmul (`xp`). Rows
/// and pairs past the end are replaced by the last one; each output's value
/// does not depend on its tile partners, so every element comes out the
/// same however the rows are split across threads.
///
/// # Safety
/// As for [`quant_rows`], with `xp` the pairs of all `t` tokens.
#[cfg(target_arch = "aarch64")]
unsafe fn quant_rows_mmla(out: SharedOut, d: usize, r: std::ops::Range<usize>, w: &Weight, xp: &[neon_quant::PairQ8], t: usize) {
    let pairs = t.div_ceil(2);
    let nb = xp.len() / pairs;
    for i in r.clone().step_by(4) {
        let rows = [0, 1, 2, 3].map(|k| (i + k).min(r.end - 1));
        for p in (0..pairs).step_by(2) {
            let ps = [p, (p + 1).min(pairs - 1)];
            let v = mmla4(w, rows, ps, xp, nb);
            for (tile, v) in v.iter().enumerate() {
                let (rp, tp) = (tile / 2, tile % 2);
                for (k, v) in v.iter().enumerate() {
                    let (row, tok) = (rows[2 * rp + k / 2], (2 * ps[tp] + k % 2).min(t - 1));
                    // SAFETY: element (tok, row) lies in this call's rows;
                    // a duplicate row or token writes the same value again.
                    unsafe { *out.ptr().add(tok * d + row) = *v };
                }
            }
        }
    }
}

/// `out[i] = row_i(W) · x` for quantized W, rows `r` of all `t` tokens,
/// with the activations already quantized (`xq` is `[t, nb]` blocks, `xd`
/// their scales as f32). Rows go four at a time where possible (each
/// activation block loaded once for four rows); each group of weight rows
/// stays in L1 across tokens. With i8mm and more than one token,
/// [`linear_batch_on`] runs the `smmla` version instead.
///
/// # Safety
/// `out` must be valid for `t·d` floats and no other thread may write
/// rows `r` of it during the call.
unsafe fn quant_rows(out: SharedOut, d: usize, r: std::ops::Range<usize>, w: &Weight, xq: &[BlockQ8_0], xd: &[f32], t: usize) {
    let nb = xq.len() / t;
    let fast = has_dot4();
    let mut i = r.start;
    while i < r.end {
        let step = if fast && i + 4 <= r.end { 4 } else { 1 };
        for tt in 0..t {
            let (x, s) = (&xq[tt * nb..(tt + 1) * nb], &xd[tt * nb..(tt + 1) * nb]);
            if step == 4 {
                let v = dot4(w, i, x, s);
                // SAFETY: elements (tt, i..i+4) lie in this call's rows.
                (0..4).for_each(|k| unsafe { *out.ptr().add(tt * d + i + k) = v[k] });
            } else {
                // SAFETY: element (tt, i) lies in this call's rows.
                unsafe { *out.ptr().add(tt * d + i) = dot1(w, i, x) };
            }
        }
        i += step;
    }
}

/// `out[t] = W · x[t]` for `t` tokens and any storage format of W, split by
/// rows across `pool`. Quantized formats quantize the activations to Q8_0
/// once, then use integer dot products (llama.cpp's approach).
pub fn linear_batch_on(pool: Option<&ThreadPool>, out: &mut [f32], x: &[f32], w: &Weight, t: usize) {
    let (n, d) = (x.len() / t.max(1), out.len() / t.max(1));
    let Weight::F32(f) = w else {
        assert_eq!((x.len(), out.len()), (t * n, t * d), "shapes");
        let xq: Vec<BlockQ8_0> = x.chunks_exact(n).flat_map(quantize_activation).collect();
        // The activation's scales as f32, converted once rather than per row.
        let xd: Vec<f32> = xq.iter().map(|b| crate::quant::f16_to_f32(b.d)).collect();
        let o = SharedOut(out.as_mut_ptr());
        #[cfg(target_arch = "aarch64")]
        if t > 1 && has_mmla() && has_mmla_kernel(w) {
            let xp = neon_quant::interleave_pairs(&xq, &xd, t);
            match pool.filter(|p| p.threads() > 1) {
                // SAFETY: each thread writes only its own rows of `out`.
                Some(p) => p.for_blocks(d, crate::pool::block_size(d, p.threads(), ROW_BLOCK), &|r| unsafe { quant_rows_mmla(o, d, r, w, &xp, t) }),
                // SAFETY: single thread, all rows.
                None => unsafe { quant_rows_mmla(o, d, 0..d, w, &xp, t) },
            }
            return;
        }
        match pool.filter(|p| p.threads() > 1) {
            // SAFETY: each thread writes only its own rows of `out`.
            Some(p) => p.for_blocks(d, crate::pool::block_size(d, p.threads(), ROW_BLOCK), &|r| unsafe { quant_rows(o, d, r, w, &xq, &xd, t) }),
            // SAFETY: single thread, all rows.
            None => unsafe { quant_rows(o, d, 0..d, w, &xq, &xd, t) },
        }
        return;
    };
    if t == 1 { matmul_on(pool, out, x, f) } else { matmul_batch_on(pool, out, x, f, t) }
}

/// `out = W · x` for any storage format of W.
pub fn linear_on(pool: Option<&ThreadPool>, out: &mut [f32], x: &[f32], w: &Weight) {
    linear_batch_on(pool, out, x, w, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{golden, max_errors, read_npy};
    use std::path::Path;

    /// Every `testdata/kernels/<prefix>*` case directory.
    fn cases(prefix: &str) -> Vec<std::path::PathBuf> {
        let Some(root) = golden("kernels") else { return Vec::new() };
        let mut v: Vec<_> = std::fs::read_dir(root)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with(prefix))
            .collect();
        v.sort();
        v
    }

    fn arr(dir: &Path, name: &str) -> Vec<f32> {
        read_npy(&dir.join(format!("{name}.npy"))).1
    }

    /// Runs `f` on every matmul case and checks it against numpy (float64
    /// reference). Error grows with the row length n (n rounded additions),
    /// so the bound is relative to the output's scale and to sqrt(n).
    fn check_matmul(label: &str, f: impl Fn(&mut [f32], &[f32], &[f32])) {
        let mut worst: f32 = 0.0;
        let cs = cases("matmul_");
        for dir in &cs {
            let (w, x, want) = (arr(dir, "w"), arr(dir, "x"), arr(dir, "out"));
            let mut out = vec![f32::NAN; want.len()];
            f(&mut out, &x, &w);
            let (_, rel) = max_errors(&out, &want);
            let bound = 1e-6 * (x.len() as f32).sqrt().max(4.0);
            assert!(rel <= bound, "{label} {}: rel {rel:.2e} > {bound:.2e}", dir.display());
            worst = worst.max(rel);
        }
        println!("{label}: {} cases vs numpy, worst relative error {worst:.2e}", cs.len());
    }

    #[test]
    fn matmul_matches_numpy() {
        check_matmul("matmul (dispatched)", matmul);
        check_matmul("scalar::matmul", scalar::matmul);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_gemv_steps_match_numpy() {
        check_matmul("neon::gemv_one_acc", neon::gemv_one_acc);
        check_matmul("neon::gemv_row_4acc", neon::gemv_row_4acc);
        check_matmul("neon::gemv", neon::gemv);
    }

    /// GEMM cases: x [t, n], w [d, n], out [t, d] = x · wᵀ.
    #[test]
    fn matmul_batch_matches_numpy() {
        let cs = cases("gemm_");
        let mut worst: f32 = 0.0;
        for dir in &cs {
            let (w, (xs, x), want) = (arr(dir, "w"), read_npy(&dir.join("x.npy")), arr(dir, "out"));
            let mut out = vec![f32::NAN; want.len()];
            matmul_batch(&mut out, &x, &w, xs[0]);
            let (_, rel) = max_errors(&out, &want);
            let bound = 1e-6 * (xs[1] as f32).sqrt().max(4.0);
            assert!(rel <= bound, "{}: rel {rel:.2e} > {bound:.2e}", dir.display());
            worst = worst.max(rel);
        }
        println!("matmul_batch: {} cases vs numpy, worst relative error {worst:.2e}", cs.len());
    }

    /// Row-split threading must not change a single bit: each output row is
    /// still one thread's sum in the same order. Checked on the golden
    /// shapes, 1 to 8 threads.
    #[test]
    fn threaded_matmuls_are_bit_identical() {
        std::thread::spawn(|| {
            let online = crate::sys::affinity::online_cpus().unwrap();
            for threads in [2, 3, 8] {
                let pool = ThreadPool::new(&online[..threads.min(online.len())]).unwrap();
                for dir in cases("matmul_").iter().chain(&cases("gemm_")) {
                    let (w, (xs, x)) = (arr(dir, "w"), read_npy(&dir.join("x.npy")));
                    let t = if xs.len() == 2 { xs[0] } else { 1 };
                    let d = w.len() / (x.len() / t);
                    let (mut one, mut many) = (vec![0.0; t * d], vec![1.0; t * d]);
                    matmul_batch(&mut one, &x, &w, t);
                    matmul_batch_on(Some(&pool), &mut many, &x, &w, t);
                    assert_eq!(one, many, "{} with {threads} threads", dir.display());
                    if t == 1 {
                        let mut gemv = vec![2.0; d];
                        matmul_on(Some(&pool), &mut gemv, &x, &w);
                        let mut single = vec![0.0; d];
                        matmul(&mut single, &x, &w);
                        assert_eq!(single, gemv, "{} with {threads} threads", dir.display());
                    }
                }
            }
        })
        .join()
        .unwrap();
    }

    /// The model's quantized path (`linear_on`: activation quantized to
    /// Q8_0, 4-row integer kernels plus single-row tails, threaded) against
    /// the "b" reference built from llama.cpp's quantizers. Row counts 1, 3,
    /// 5, 17 and 130 exercise the tails.
    #[test]
    fn quantized_linear_matches_reference() {
        let Some(root) = golden("quant") else { return };
        std::thread::spawn(move || {
            let online = crate::sys::affinity::online_cpus().unwrap();
            let pool = ThreadPool::new(&online[..3.min(online.len())]).unwrap();
            for dir in std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()) {
                let ((ws, w), x) = (read_npy(&dir.join("w.npy")), arr(&dir, "x"));
                for (format, name) in [(crate::weight::Format::Q8_0, "q8_0"), (crate::weight::Format::Q4_0, "q4_0"), (crate::weight::Format::Q5_0, "q5_0")] {
                    let mut buf = crate::aligned::AlignedBuf::zeros(w.len());
                    buf.copy_from_slice(&w);
                    let copy = buf.clone_buf();
                    let q = Weight::from(buf).convert(ws[1], format);
                    let want = arr(&dir, &format!("out_{name}_b"));
                    // Row-major, and repacked four rows at a time where the
                    // CPU uses that layout.
                    let repacked = repack_weights().then(|| Weight::from(copy).convert(ws[1], format).repack(ws[1]));
                    for q in std::iter::once(&q).chain(repacked.as_ref()) {
                        for p in [None, Some(&pool)] {
                            let mut out = vec![f32::NAN; ws[0]];
                            linear_on(p, &mut out, &x, q);
                            let (_, rel) = max_errors(&out, &want);
                            assert!(rel < 1e-6 * (ws[1] as f32).sqrt().max(4.0), "{} {name} {:?}: rel {rel:.2e}", dir.display(), q.layout());
                        }
                    }
                }
            }
        })
        .join()
        .unwrap();
    }

    /// The batched quantized path (prefill: `t` tokens, each activation
    /// quantized to Q8_0 on its own) against the same "b" reference.
    /// Odd row and token counts exercise every tile's tails; 1 to 3
    /// threads must give the same bits as one.
    #[test]
    fn quantized_linear_batch_matches_reference() {
        let Some(root) = golden("quant_batch") else { return };
        std::thread::spawn(move || {
            let online = crate::sys::affinity::online_cpus().unwrap();
            let pool = ThreadPool::new(&online[..3.min(online.len())]).unwrap();
            for dir in std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()) {
                let ((ws, w), (xs, x)) = (read_npy(&dir.join("w.npy")), read_npy(&dir.join("x.npy")));
                let t = xs[0];
                for (format, name) in [(crate::weight::Format::Q8_0, "q8_0"), (crate::weight::Format::Q4_0, "q4_0"), (crate::weight::Format::Q5_0, "q5_0")] {
                    let mut buf = crate::aligned::AlignedBuf::zeros(w.len());
                    buf.copy_from_slice(&w);
                    let copy = buf.clone_buf();
                    let q = Weight::from(buf).convert(ws[1], format);
                    let want = arr(&dir, &format!("out_{name}_b"));
                    let mut one = vec![f32::NAN; t * ws[0]];
                    linear_batch_on(None, &mut one, &x, &q, t);
                    let (_, rel) = max_errors(&one, &want);
                    assert!(rel < 1e-6 * (ws[1] as f32).sqrt().max(4.0), "{} {name}: rel {rel:.2e}", dir.display());
                    let mut many = vec![f32::NAN; t * ws[0]];
                    linear_batch_on(Some(&pool), &mut many, &x, &q, t);
                    assert_eq!(one, many, "{} {name}: threads changed the result", dir.display());
                    // Repacked four rows at a time (where the CPU uses it):
                    // the same `smmla` sums, so the same bits.
                    if repack_weights() {
                        let r = Weight::from(copy).convert(ws[1], format).repack(ws[1]);
                        for p in [None, Some(&pool)] {
                            let mut out = vec![f32::NAN; t * ws[0]];
                            linear_batch_on(p, &mut out, &x, &r, t);
                            assert_eq!(one, out, "{} {name} {:?}: repacking changed the result", dir.display(), r.layout());
                        }
                    }
                }
            }
        })
        .join()
        .unwrap();
    }

    /// K-quant matrices (random blocks, decoded by gguf) through the
    /// model's path: one token at a time and all `t` at once, on one and
    /// three threads, against the float64 "b" reference.
    #[test]
    fn k_quant_linear_matches_reference() {
        use crate::quant::blocks_from_bytes;
        use crate::weight::Store;
        let Some(root) = golden("quant_k") else { return };
        std::thread::spawn(move || {
            let online = crate::sys::affinity::online_cpus().unwrap();
            let pool = ThreadPool::new(&online[..3.min(online.len())]).unwrap();
            for dir in std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()) {
                let (xs, x) = read_npy(&dir.join("x.npy"));
                let (t, n) = (xs[0], xs[1]);
                let bytes = |name: &str| std::fs::read(dir.join(format!("w_{name}.bin"))).unwrap();
                for (name, w) in [("q4_k", Weight::Q4_K(Store::Owned(blocks_from_bytes(&bytes("q4_k"))))), ("q6_k", Weight::Q6_K(Store::Owned(blocks_from_bytes(&bytes("q6_k")))))] {
                    let want = arr(&dir, &format!("out_{name}_b"));
                    let d = want.len() / t;
                    let bound = 1e-6 * (n as f32).sqrt();
                    for p in [None, Some(&pool)] {
                        let mut all = vec![f32::NAN; t * d];
                        linear_batch_on(p, &mut all, &x, &w, t);
                        let (_, rel) = max_errors(&all, &want);
                        assert!(rel < bound, "{} {name} batch: rel {rel:.2e}", dir.display());
                        for tt in 0..t {
                            let mut one = vec![f32::NAN; d];
                            linear_on(p, &mut one, &x[tt * n..(tt + 1) * n], &w);
                            let (_, rel) = max_errors(&one, &want[tt * d..(tt + 1) * d]);
                            assert!(rel < bound, "{} {name} token {tt}: rel {rel:.2e}", dir.display());
                        }
                    }
                }
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn rmsnorm_softmax_swiglu_match_numpy() {
        for dir in cases("rmsnorm_") {
            let (x, wt, want) = (arr(&dir, "x"), arr(&dir, "weight"), arr(&dir, "out"));
            let mut out = vec![0.0; x.len()];
            rmsnorm(&mut out, &x, &wt, 1e-5);
            let (_, rel) = max_errors(&out, &want);
            assert!(rel < 1e-6, "{}: rel {rel:.2e}", dir.display());
            let mut inplace = x.clone();
            rmsnorm_inplace(&mut inplace, &wt, 1e-5);
            assert_eq!(inplace, out);
        }
        for dir in cases("softmax_") {
            let (mut x, want) = (arr(&dir, "x"), arr(&dir, "out"));
            let n = x.len() as f32;
            softmax(&mut x);
            // The normalising sum adds n rounded terms: bound grows with sqrt(n).
            let (_, rel) = max_errors(&x, &want);
            assert!(rel < 2e-6 * n.sqrt().max(4.0), "{}: rel {rel:.2e}", dir.display());
        }
        for dir in cases("swiglu_") {
            let (mut g, u, want) = (arr(&dir, "gate"), arr(&dir, "up"), arr(&dir, "out"));
            swiglu(&mut g, &u);
            let (_, rel) = max_errors(&g, &want);
            assert!(rel < 1e-6, "{}: rel {rel:.2e}", dir.display());
        }
    }

    /// One head over n positions (`attention_<n>x<hs>` cases), scalar and
    /// dispatched. The output is a weighted average of values, so its error
    /// is that of the scores (dot products of length hs) plus the softmax
    /// sum over n terms.
    #[test]
    fn attention_matches_numpy() {
        let cs = cases("attention_");
        let mut worst: f32 = 0.0;
        for dir in &cs {
            let (q, k, v, want) = (arr(dir, "q"), arr(dir, "k"), arr(dir, "v"), arr(dir, "out"));
            let hs = q.len();
            let n = k.len() / hs;
            let bound = 2e-6 * ((hs as f32).sqrt() + (n as f32).sqrt()).max(4.0);
            type Head = for<'a> fn(&mut [f32], &[f32], &dyn Fn(usize) -> &'a [f32], &dyn Fn(usize) -> &'a [f32], usize, &mut [f32]);
            let fs: [(&str, Head); 2] = [
                ("dispatched", |o, q, k, v, p, s| attention_head(o, q, k, v, p, s)),
                ("scalar", |o, q, k, v, p, s| scalar::attention_head(o, q, k, v, p, s)),
            ];
            for (label, f) in fs {
                let (mut out, mut scores) = (vec![f32::NAN; hs], vec![f32::NAN; n + 3]);
                f(&mut out, &q, &|t| &k[t * hs..(t + 1) * hs], &|t| &v[t * hs..(t + 1) * hs], n - 1, &mut scores);
                let (_, rel) = max_errors(&out, &want);
                assert!(rel <= bound, "{label} {}: rel {rel:.2e} > {bound:.2e}", dir.display());
                worst = worst.max(rel);
            }
        }
        println!("attention: {} cases vs numpy, worst relative error {worst:.2e}", cs.len());
    }

    /// The grouped kernel on the same cases: three heads with the query
    /// scaled by 1, -0.5 and 2 (numpy's answer is for the first; the others
    /// are checked to equal the per-head kernel bit for bit, which is the
    /// grouped kernel's contract). Two heads in a row must not share
    /// scores.
    #[test]
    fn attention_group_matches_numpy() {
        for dir in &cases("attention_") {
            let (q, k, v, want) = (arr(dir, "q"), arr(dir, "k"), arr(dir, "v"), arr(dir, "out"));
            let hs = q.len();
            let n = k.len() / hs;
            let bound = 2e-6 * ((hs as f32).sqrt() + (n as f32).sqrt()).max(4.0);
            let qg: Vec<f32> = [1.0f32, -0.5, 2.0].iter().flat_map(|&a| q.iter().map(move |x| x * a)).collect();
            let (keys, values) = (|t: usize| &k[t * hs..(t + 1) * hs], |t: usize| &v[t * hs..(t + 1) * hs]);
            let (mut out, mut scores) = (vec![f32::NAN; 3 * hs], vec![f32::NAN; 3 * n + 5]);
            attention_group(&mut out, &qg, hs, keys, values, n - 1, &mut scores);
            let (_, rel) = max_errors(&out[..hs], &want);
            assert!(rel <= bound, "{}: rel {rel:.2e} > {bound:.2e}", dir.display());
            for h in 0..3 {
                let (mut one, mut s1) = (vec![f32::NAN; hs], vec![f32::NAN; n]);
                attention_head(&mut one, &qg[h * hs..(h + 1) * hs], keys, values, n - 1, &mut s1);
                assert!(one.iter().zip(&out[h * hs..]).all(|(a, b)| a.to_bits() == b.to_bits()), "{} head {h}", dir.display());
            }
        }
    }

    /// The angle `pos * freq` is rounded to f32 before sin/cos (as llama2.c
    /// does), and an f32 near `pos` is only accurate to about pos·ε, so the
    /// bound grows with the position: ~6e-5 at 255, ~1e-3 at 4000.
    #[test]
    fn rope_matches_numpy() {
        for dir in cases("rope_") {
            let (mut x, want) = (arr(&dir, "x"), arr(&dir, "out"));
            let name = dir.file_name().unwrap().to_str().unwrap().to_string();
            let halves = name.starts_with("rope_halves_");
            let shape = name.trim_start_matches("rope_halves_").trim_start_matches("rope_");
            let hs: usize = shape.split(['x', '_']).nth(1).unwrap().parse().unwrap();
            let pos: usize = name.rsplit("_p").next().unwrap().parse().unwrap();
            if halves {
                rope_halves(&mut x, pos, hs, 10000.0);
            } else {
                rope_adjacent(&mut x, pos, hs, 10000.0);
            }
            let (_, rel) = max_errors(&x, &want);
            let bound = 2e-6 + 2.0 * pos as f32 * f32::EPSILON;
            assert!(rel < bound, "{name}: rel {rel:.2e}");
        }
    }
}
