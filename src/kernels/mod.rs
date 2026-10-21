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

use crate::pool::ThreadPool;
use crate::quant::{BlockQ8_0, QK};
use crate::weight::Weight;
pub use scalar::{add_inplace, argmax, attention_head, rmsnorm, rmsnorm_inplace, rope_adjacent, rope_halves, softmax, swiglu};

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

/// A raw pointer that may be shared between the pool's threads. Each thread
/// turns it back into a slice of its own, disjoint range.
#[derive(Clone, Copy)]
struct SharedOut(*mut f32);
// SAFETY: only used to give threads disjoint ranges of one buffer (see the
// callers); no element is ever accessed by two threads.
unsafe impl Send for SharedOut {}
unsafe impl Sync for SharedOut {}

impl SharedOut {
    /// A method (not `.0`) so closures capture the whole `Sync` wrapper,
    /// not just its raw-pointer field, which is not `Sync`.
    fn ptr(self) -> *mut f32 {
        self.0
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
    pool.for_ranges(out.len(), ROW_BLOCK, &|r| {
        // SAFETY: `for_ranges` gives each thread a disjoint range of rows.
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
    pool.for_ranges(d, ROW_BLOCK, &|r| {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: shapes checked above; rows are disjoint per thread.
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
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q8_0(b) => crate::quant::scalar::dot_q8_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q4_0(b) => crate::quant::scalar::dot_q4_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        Weight::F32(_) => unreachable!(),
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
/// [`has_dot4`] is true.
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
                Weight::F32(_) => unreachable!(),
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    unreachable!("dot4 needs the AArch64 sdot kernels")
}

/// `out[i] = row_i(W) · x` for quantized W, rows `r` of all `t` tokens,
/// with the activations already quantized (`xq` is `[t, nb]` blocks, `xd`
/// their scales as f32). Rows go four at a time where possible (each
/// activation block loaded once for four rows); each group of weight rows
/// stays in L1 across tokens.
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
        match pool.filter(|p| p.threads() > 1) {
            // SAFETY: each thread writes only its own rows of `out`.
            Some(p) => p.for_ranges(d, ROW_BLOCK, &|r| unsafe { quant_rows(o, d, r, w, &xq, &xd, t) }),
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
                for (format, name) in [(crate::weight::Format::Q8_0, "q8_0"), (crate::weight::Format::Q4_0, "q4_0")] {
                    let mut buf = crate::aligned::AlignedBuf::zeros(w.len());
                    buf.copy_from_slice(&w);
                    let q = Weight::from(buf).convert(ws[1], format);
                    let want = arr(&dir, &format!("out_{name}_b"));
                    for p in [None, Some(&pool)] {
                        let mut out = vec![f32::NAN; ws[0]];
                        linear_on(p, &mut out, &x, &q);
                        let (_, rel) = max_errors(&out, &want);
                        assert!(rel < 1e-6 * (ws[1] as f32).sqrt().max(4.0), "{} {name}: rel {rel:.2e}", dir.display());
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
