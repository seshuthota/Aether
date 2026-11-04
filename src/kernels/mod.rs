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
#[cfg(feature = "hexagon")]
pub mod hexagon;
#[cfg(feature = "vulkan")]
pub mod vulkan;

use crate::pool::{SharedOut, ThreadPool};
use crate::quant::{BlockQ8_0, BlockQ8_K, QK, QK_K};
use crate::weight::Weight;
pub use scalar::{add_inplace, argmax, rmsnorm, rmsnorm_inplace, rope_adjacent, rope_halves, rope_halves_pairs, softmax};

/// RMSNorm each head of `x` with the same gain (LFM3's QK-norm).
pub fn rmsnorm_heads(x: &mut [f32], weight: &[f32], head_size: usize, eps: f32) {
    assert_eq!(weight.len(), head_size);
    for head in x.chunks_exact_mut(head_size) {
        rmsnorm_inplace(head, weight, eps);
    }
}

/// Multiply head `h` of `x` by `sigmoid(gates[h])` (LFM3's output gate).
pub fn gate_heads(x: &mut [f32], gates: &[f32], head_size: usize) {
    for (head, &g) in x.chunks_exact_mut(head_size).zip(gates) {
        let s = 1.0 / (1.0 + (-g).exp());
        for v in head {
            *v *= s;
        }
    }
}

/// SwiGLU in place, `gate_i = silu(gate_i) * up_i` (see [`scalar::swiglu`]);
/// NEON with a polynomial `exp` on aarch64.
#[inline]
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    neon::swiglu(gate, up);
    #[cfg(not(target_arch = "aarch64"))]
    scalar::swiglu(gate, up);
}

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

/// [`attention_group`] over only the last `window` positions ending at
/// `pos` (a sliding-window layer). `scores` needs `min(window, pos + 1)`
/// entries. A full layer passes `usize::MAX`.
pub fn attention_group_window<'a>(
    out: &mut [f32],
    q: &[f32],
    hs: usize,
    keys: impl Fn(usize) -> &'a [f32],
    values: impl Fn(usize) -> &'a [f32],
    pos: usize,
    window: usize,
    scores: &mut [f32],
) {
    let start = (pos + 1).saturating_sub(window);
    let n = pos + 1 - start;
    attention_group(out, q, hs, |t| keys(start + t), |t| values(start + t), n - 1, scores);
}

/// [`attention_head`] with a sliding window.
#[allow(clippy::too_many_arguments)]
pub fn attention_head_window<'a>(
    out: &mut [f32],
    q: &[f32],
    keys: impl Fn(usize) -> &'a [f32],
    values: impl Fn(usize) -> &'a [f32],
    pos: usize,
    window: usize,
    scores: &mut [f32],
) {
    attention_group_window(out, q, q.len(), keys, values, pos, window, scores);
}

/// One depthwise causal short-conv step: `out[ch] = Σ_{k < width-1}
/// state[k][ch] · weight[ch][k] + u[ch] · weight[ch][width-1]`, then the
/// state shifts one row and `u` becomes its last row.
pub fn short_conv_step(u: &[f32], weight: &[f32], state: &mut [f32], dim: usize, width: usize, out: &mut [f32]) {
    assert_eq!((u.len(), out.len(), weight.len()), (dim, dim, dim * width));
    assert!(width >= 1 && width <= 7);
    assert_eq!(state.len(), 6 * dim);
    for ch in 0..dim {
        let mut acc = 0.0;
        for k in 0..width - 1 {
            acc += state[k * dim + ch] * weight[ch * width + k];
        }
        acc += u[ch] * weight[ch * width + width - 1];
        out[ch] = acc;
    }
    state.copy_within(dim..6 * dim, 0);
    state[5 * dim..].copy_from_slice(u);
}

/// [`short_conv_step`] for `t` consecutive tokens.
pub fn short_conv_rows(u: &[f32], weight: &[f32], state: &mut [f32], dim: usize, width: usize, out: &mut [f32], t: usize) {
    assert_eq!((u.len(), out.len()), (t * dim, t * dim));
    for i in 0..t {
        short_conv_step(&u[i * dim..(i + 1) * dim], weight, state, dim, width, &mut out[i * dim..(i + 1) * dim]);
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

/// Quantizes `t` activation vectors (`x` is `[t, n]`, n a multiple of 32)
/// to Q8_0 for the integer dot-product kernels, one token per piece across
/// `pool`.
fn quantize_activations(pool: Option<&ThreadPool>, x: &[f32], t: usize) -> Vec<BlockQ8_0> {
    let n = x.len() / t;
    let mut q = vec![BlockQ8_0 { d: 0, qs: [0; QK] }; x.len() / QK];
    crate::pool::par_chunks_mut(pool, &mut q, n / QK, &|i, q| {
        let x = &x[i * n..(i + 1) * n];
        #[cfg(target_arch = "aarch64")]
        neon_quant::quantize_q8_0(x, q);
        #[cfg(not(target_arch = "aarch64"))]
        crate::quant::quantize_q8_0(x, q);
    });
    q
}

/// True for the formats whose dot products take Q8_K activations: all
/// K-quants, row-major or repacked.
fn needs_q8_k(w: &Weight) -> bool {
    matches!(w, Weight::Q4_K(_) | Weight::Q6_K(_) | Weight::Q4_Kx4(_) | Weight::Q6_Kx4(_))
}

/// Quantizes `t` activation vectors to Q8_K (256 values per block), one
/// token per piece across `pool`. K-quant matrices use these: one f32 scale
/// per super-block lets their sub-block scales stay in int32.
fn quantize_q8_k_activations(pool: Option<&ThreadPool>, x: &[f32], t: usize) -> Vec<BlockQ8_K> {
    let n = x.len() / t;
    assert!(n.is_multiple_of(QK_K), "a K-quant activation is whole super-blocks");
    let mut q = vec![BlockQ8_K { d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16] }; x.len() / QK_K];
    crate::pool::par_chunks_mut(pool, &mut q, n / QK_K, &|i, q| {
        crate::quant::quantize_q8_k(&x[i * n..(i + 1) * n], q);
    });
    q
}

/// One row's dot product with one quantized activation (any CPU).
#[inline]
fn dot1(w: &Weight, i: usize, x: &[BlockQ8_0], xk: Option<&[BlockQ8_K]>) -> f32 {
    let nb = x.len();
    let k = || xk.expect("a K-quant row needs Q8_K activations");
    match w {
        #[cfg(target_arch = "aarch64")]
        Weight::Q8_0(b) => neon_quant::dot_q8_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(target_arch = "aarch64")]
        Weight::Q4_0(b) => neon_quant::dot_q4_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(target_arch = "aarch64")]
        Weight::Q5_0(b) => neon_quant::dot_q5_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(target_arch = "aarch64")]
        Weight::Q4_K(b) => neon_quant::dot_q4_k_q8_k(&b[i * nb / 8..(i + 1) * nb / 8], k()),
        #[cfg(target_arch = "aarch64")]
        Weight::Q6_K(b) => neon_quant::dot_q6_k_q8_k(&b[i * nb / 8..(i + 1) * nb / 8], k()),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q8_0(b) => crate::quant::scalar::dot_q8_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q4_0(b) => crate::quant::scalar::dot_q4_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q5_0(b) => crate::quant::scalar::dot_q5_0_q8_0(&b[i * nb..(i + 1) * nb], x),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q4_K(b) => crate::quant::scalar::dot_q4_k_q8_k(&b[i * nb / 8..(i + 1) * nb / 8], k()),
        #[cfg(not(target_arch = "aarch64"))]
        Weight::Q6_K(b) => crate::quant::scalar::dot_q6_k_q8_k(&b[i * nb / 8..(i + 1) * nb / 8], k()),
        Weight::F32(_) => unreachable!(),
        Weight::Q8_0x4(_) | Weight::Q4_0x4(_) | Weight::Q5_0x4(_) | Weight::Q4_Kx4(_) | Weight::Q6_Kx4(_) => unreachable!("repacked rows go four at a time"),
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
fn dot4(w: &Weight, i: usize, x: &[BlockQ8_0], xd: &[f32], xk: Option<&[BlockQ8_K]>) -> [f32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        let nb = x.len();
        let rows = |k: usize| (i + k) * nb..(i + k + 1) * nb;
        let k = || xk.expect("a K-quant row needs Q8_K activations");
        // SAFETY: callers check has_dot4 (dotprod present); every row slice
        // has nb blocks.
        unsafe {
            match w {
                Weight::Q8_0(b) => neon_quant::dot4_q8_0_q8_0([0, 1, 2, 3].map(|k| &b[rows(k)]), x, xd),
                Weight::Q4_0(b) => neon_quant::dot4_q4_0_q8_0([0, 1, 2, 3].map(|k| &b[rows(k)]), x, xd),
                Weight::Q5_0(b) => neon_quant::dot4_q5_0_q8_0([0, 1, 2, 3].map(|k| &b[rows(k)]), x, xd),
                Weight::Q4_K(b) => neon_quant::dot4_q4_k_q8_k([0, 1, 2, 3].map(|k| &b[(i + k) * nb / 8..(i + k + 1) * nb / 8]), k()),
                Weight::Q6_K(b) => neon_quant::dot4_q6_k_q8_k([0, 1, 2, 3].map(|k| &b[(i + k) * nb / 8..(i + k + 1) * nb / 8]), k()),
                Weight::Q8_0x4(b) => neon_quant::gemv4_q8_0x4(&b[group(i, nb)], x, xd),
                Weight::Q4_0x4(b) => neon_quant::gemv4_q4_0x4(&b[group(i, nb)], x, xd),
                Weight::Q5_0x4(b) => neon_quant::gemv4_q5_0x4(&b[group(i, nb)], x, xd),
                Weight::Q4_Kx4(b) => neon_quant::dot4_q4_k_q8_k_x4(&b[group(i, nb / 8)], k()),
                Weight::Q6_Kx4(b) => neon_quant::dot4_q6_k_q8_k_x4(&b[group(i, nb / 8)], k()),
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
fn mmla4<const P: usize>(w: &Weight, rows: [usize; 4], ps: [usize; P], xp: &[neon_quant::PairQ8], nb: usize) -> neon_quant::Tiles<P> {
    let x = ps.map(|p| &xp[p * nb..(p + 1) * nb]);
    // SAFETY: callers check has_mmla (i8mm present); every row and pair
    // slice has nb blocks.
    unsafe {
        match w {
            Weight::Q8_0(b) => neon_quant::mmla4_q8_0(rows.map(|i| &b[i * nb..(i + 1) * nb]), x),
            Weight::Q4_0(b) => neon_quant::mmla4_q4_0(rows.map(|i| &b[i * nb..(i + 1) * nb]), x),
            Weight::Q5_0(b) => neon_quant::mmla4_q5_0(rows.map(|i| &b[i * nb..(i + 1) * nb]), x),
            Weight::Q4_K(_) | Weight::Q6_K(_) | Weight::Q4_Kx4(_) | Weight::Q6_Kx4(_) => unreachable!("K-quants are unpacked first (see unpack_k4)"),
            Weight::Q8_0x4(b) => neon_quant::mmla4_q8_0x4(&b[group4(rows, nb)], x),
            Weight::Q4_0x4(b) => neon_quant::mmla4_q4_0x4(&b[group4(rows, nb)], x),
            Weight::Q5_0x4(b) => neon_quant::mmla4_q5_0x4(&b[group4(rows, nb)], x),
            Weight::F32(_) => unreachable!(),
        }
    }
}

/// Whether `w` is a K-quant: [`quant_rows_mmla`] unpacks those four rows
/// at a time ([`unpack_k4`]) instead of calling [`mmla4`].
#[cfg(target_arch = "aarch64")]
fn is_k_quant(w: &Weight) -> bool {
    needs_q8_k(w)
}

/// Rows `rows` of a K-quant matrix (rows of `nb` 32-value blocks), unpacked
/// for [`neon_quant::mmla4_k`]: once per matmul for each group of four
/// rows, then used for every token.
#[cfg(target_arch = "aarch64")]
fn unpack_k4(w: &Weight, rows: [usize; 4], nb: usize) -> Vec<neon_quant::KBlock4> {
    use crate::quant::{BlockQ4_K, BlockQ6_K, QK_K, f16_to_f32, q4_k_scales_mins};
    use std::arch::aarch64::{int8x16_t, vget_high_s8, vget_low_s8, vst1_s8};
    // One super-block of row r into its 8 blocks: `sub(j)` gives block j's
    // values and its two half scales and bias.
    #[inline(always)]
    fn fill(o: &mut [neon_quant::KBlock4], r: usize, sub: impl Fn(usize) -> ([int8x16_t; 2], (f32, f32, f32))) {
        for (j, o) in o.iter_mut().enumerate() {
            let v;
            (v, (o.sa[r], o.sb[r], o.bias[r])) = sub(j);
            // Chunk c (8 values) of row r goes to bytes 32c + 8r.
            // SAFETY: 3 · 32 + 8r + 8 <= 128 bytes of qs (r < 4).
            unsafe {
                let q = o.qs.as_mut_ptr();
                vst1_s8(q.add(8 * r), vget_low_s8(v[0]));
                vst1_s8(q.add(32 + 8 * r), vget_high_s8(v[0]));
                vst1_s8(q.add(64 + 8 * r), vget_low_s8(v[1]));
                vst1_s8(q.add(96 + 8 * r), vget_high_s8(v[1]));
            }
        }
    }
    let nk = nb / (QK_K / QK);
    let mut out = vec![neon_quant::KBlock4 { qs: [0; 4 * QK], sa: [0.0; 4], sb: [0.0; 4], bias: [0.0; 4] }; nb];
    for (r, &row) in rows.iter().enumerate() {
        for (k, o) in out.as_chunks_mut::<{ QK_K / QK }>().0.iter_mut().enumerate() {
            match w {
                Weight::Q4_K(b) => {
                    let blk = &b[row * nk + k];
                    let (d, dmin) = (f16_to_f32(blk.d), f16_to_f32(blk.dmin));
                    let (sc, m) = q4_k_scales_mins(&blk.scales);
                    fill(o, r, |j| (neon_quant::q4_k_sub(blk, j), (d * sc[j] as f32, d * sc[j] as f32, -(dmin * m[j] as f32))));
                }
                Weight::Q6_K(b) => {
                    let blk = &b[row * nk + k];
                    let d = f16_to_f32(blk.d);
                    fill(o, r, |j| (neon_quant::q6_k_sub(blk, j), (d * blk.scales[2 * j] as f32, d * blk.scales[2 * j + 1] as f32, 0.0)));
                }
                Weight::Q4_Kx4(b) => {
                    let g = &b[rows[0] / 4 * nk + k];
                    let mut blk = BlockQ4_K { d: g.d[r], dmin: g.dmin[r], scales: [0; 12], qs: [0; 128] };
                    for c in 0..4 {
                        for v in 0..8 {
                            let at = if v < 4 { 4 * v } else { 16 + 4 * (v - 4) };
                            blk.qs[32 * c + at..32 * c + at + 4].copy_from_slice(&g.qs[128 * c + 16 * v + 4 * r..][..4]);
                        }
                    }
                    let (d, dmin) = (f16_to_f32(g.d[r]), f16_to_f32(g.dmin[r]));
                    fill(o, r, |j| (neon_quant::q4_k_sub(&blk, j), (d * g.scales[j * 4 + r] as f32, d * g.scales[j * 4 + r] as f32, -(dmin * g.mins[j * 4 + r] as f32))));
                }
                Weight::Q6_Kx4(b) => {
                    let g = &b[rows[0] / 4 * nk + k];
                    let mut blk = BlockQ6_K { ql: [0; 128], qh: [0; 64], scales: [0; 16], d: g.d[r] };
                    for n in 0..2 {
                        for v in 0..16 {
                            blk.ql[64 * n + 4 * v..64 * n + 4 * v + 4].copy_from_slice(&g.ql[n * 256 + 16 * v + 4 * r..][..4]);
                        }
                        for v in 0..8 {
                            blk.qh[32 * n + 4 * v..32 * n + 4 * v + 4].copy_from_slice(&g.qh[n * 128 + 16 * v + 4 * r..][..4]);
                        }
                    }
                    for s in 0..16 {
                        blk.scales[s] = g.scales[s * 4 + r];
                    }
                    let d = f16_to_f32(g.d[r]);
                    fill(o, r, |j| (neon_quant::q6_k_sub(&blk, j), (d * blk.scales[2 * j] as f32, d * blk.scales[2 * j + 1] as f32, 0.0)));
                }
                _ => unreachable!("unpack_k4 is for K-quants"),
            }
        }
    }
    out
}

/// [`group`] for four row indices, which must be one whole group.
#[cfg(target_arch = "aarch64")]
fn group4(rows: [usize; 4], nb: usize) -> std::ops::Range<usize> {
    assert!(rows == [0, 1, 2, 3].map(|k| rows[0] + k), "repacked rows go four at a time");
    group(rows[0], nb)
}

/// [`quant_rows`] with `smmla`: rows four at a time, and for each group of
/// four rows (which then stays in L1) all tokens 2·NP at a time and the
/// last 1–3 pairs in one narrower tile, from activations interleaved into
/// token pairs once per matmul (`xp`). Rows past the end are replaced by the
/// last one (an odd last token's pair holds it twice); each output's value
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
        let kb = is_k_quant(w).then(|| unpack_k4(w, rows, nb));
        let kb = kb.as_deref();
        // Full tiles of NP pairs, then one tile as wide as what is left.
        let mut p = 0;
        // SAFETY (all four): forwarded from the caller.
        unsafe {
            while p + neon_quant::NP <= pairs {
                tile::<{ neon_quant::NP }>(out, d, w, kb, rows, p, xp, nb, t);
                p += neon_quant::NP;
            }
            match pairs - p {
                1 => tile::<1>(out, d, w, kb, rows, p, xp, nb, t),
                2 => tile::<2>(out, d, w, kb, rows, p, xp, nb, t),
                3 => tile::<3>(out, d, w, kb, rows, p, xp, nb, t),
                _ => {}
            }
        }
    }
}

/// One tile of [`quant_rows_mmla`]: rows `rows` × token pairs `p..p + P`
/// (`kb`: the rows unpacked, for K-quants), written to `out`.
///
/// # Safety
/// As for [`quant_rows_mmla`]; `p + P <= pairs`.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn tile<const P: usize>(out: SharedOut, d: usize, w: &Weight, kb: Option<&[neon_quant::KBlock4]>, rows: [usize; 4], p: usize, xp: &[neon_quant::PairQ8], nb: usize, t: usize) {
    let ps: [usize; P] = std::array::from_fn(|k| p + k);
    let v = match kb {
        // SAFETY: callers check has_mmla; every pair slice has nb entries,
        // as kb does.
        Some(kb) => unsafe { neon_quant::mmla4_k(kb, ps.map(|p| &xp[p * nb..(p + 1) * nb])) },
        None => mmla4(w, rows, ps, xp, nb),
    };
    for (rp, tp, v) in v.iter().enumerate().flat_map(|(rp, row)| row.iter().enumerate().map(move |(tp, v)| (rp, tp, v))) {
        for (k, v) in v.iter().enumerate() {
            let (row, tok) = (rows[2 * rp + k / 2], (2 * ps[tp] + k % 2).min(t - 1));
            // SAFETY: element (tok, row) lies in the caller's rows; a
            // duplicate row or token writes the same value again.
            unsafe { *out.ptr().add(tok * d + row) = *v };
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
unsafe fn quant_rows(out: SharedOut, d: usize, r: std::ops::Range<usize>, w: &Weight, xq: &[BlockQ8_0], xd: &[f32], xk: Option<&[BlockQ8_K]>, t: usize) {
    let nb = xq.len() / t;
    let nbk = nb / (QK_K / QK);
    let fast = has_dot4();
    let mut i = r.start;
    while i < r.end {
        let step = if fast && i + 4 <= r.end { 4 } else { 1 };
        for tt in 0..t {
            let (x, s) = (&xq[tt * nb..(tt + 1) * nb], &xd[tt * nb..(tt + 1) * nb]);
            let k = xk.map(|v| &v[tt * nbk..(tt + 1) * nbk]);
            if step == 4 {
                let v = dot4(w, i, x, s, k);
                // SAFETY: elements (tt, i..i+4) lie in this call's rows.
                (0..4).for_each(|k| unsafe { *out.ptr().add(tt * d + i + k) = v[k] });
            } else {
                // SAFETY: element (tt, i) lies in this call's rows.
                unsafe { *out.ptr().add(tt * d + i) = dot1(w, i, x, k) };
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
        let xq = quantize_activations(pool, x, t);
        // The activation's scales as f32, converted once rather than per row.
        let xd: Vec<f32> = xq.iter().map(|b| crate::quant::f16_to_f32(b.d)).collect();
        let o = SharedOut(out.as_mut_ptr());
        #[cfg(target_arch = "aarch64")]
        if t > 1 && has_mmla() {
            let xp = neon_quant::interleave_pairs(pool, &xq, &xd, t);
            match pool.filter(|p| p.threads() > 1) {
                // SAFETY: each thread writes only its own rows of `out`.
                Some(p) => p.for_blocks(d, crate::pool::block_size(d, p.threads(), ROW_BLOCK), &|r| unsafe { quant_rows_mmla(o, d, r, w, &xp, t) }),
                // SAFETY: single thread, all rows.
                None => unsafe { quant_rows_mmla(o, d, 0..d, w, &xp, t) },
            }
            return;
        }
        let xk = needs_q8_k(w).then(|| quantize_q8_k_activations(pool, x, t));
        match pool.filter(|p| p.threads() > 1) {
            // SAFETY: each thread writes only its own rows of `out`.
            Some(p) => p.for_blocks(d, crate::pool::block_size(d, p.threads(), ROW_BLOCK), &|r| unsafe { quant_rows(o, d, r, w, &xq, &xd, xk.as_deref(), t) }),
            // SAFETY: single thread, all rows.
            None => unsafe { quant_rows(o, d, 0..d, w, &xq, &xd, xk.as_deref(), t) },
        }
        return;
    };
    if t == 1 { matmul_on(pool, out, x, f) } else { matmul_batch_on(pool, out, x, f, t) }
}

/// `outs[k] = ws[k] · x` for `t` tokens and several matrices that read the
/// same `x` (q, k, v; gate, up), as one job for the pool: `x` is quantized
/// (and, for `smmla`, interleaved) once and the threads take blocks of rows
/// from all the matrices, so the small ones (k and v have a few hundred
/// rows) cost no hand-off of their own and leave no thread idle at their
/// end. Each row is computed exactly as [`linear_batch_on`] computes it.
pub fn linear_many_batch_on(pool: Option<&ThreadPool>, outs: &mut [&mut [f32]], x: &[f32], ws: &[&Weight], t: usize) {
    assert_eq!(outs.len(), ws.len(), "one output per matrix");
    let p = pool.filter(|p| p.threads() > 1);
    if p.is_none() || ws.iter().any(|w| matches!(w, Weight::F32(_))) {
        for (o, w) in outs.iter_mut().zip(ws) {
            linear_batch_on(pool, o, x, w, t);
        }
        return;
    }
    let p = p.unwrap();
    let xq = quantize_activations(pool, x, t);
    let xd: Vec<f32> = xq.iter().map(|b| crate::quant::f16_to_f32(b.d)).collect();
    #[cfg(target_arch = "aarch64")]
    let xp = (t > 1 && has_mmla()).then(|| neon_quant::interleave_pairs(pool, &xq, &xd, t));
    #[cfg(target_arch = "aarch64")]
    let xk = (xp.is_none() && ws.iter().any(|w| needs_q8_k(w))).then(|| quantize_q8_k_activations(pool, x, t));
    #[cfg(not(target_arch = "aarch64"))]
    let xk = ws.iter().any(|w| needs_q8_k(w)).then(|| quantize_q8_k_activations(pool, x, t));
    let total: usize = outs.iter().map(|o| o.len() / t).sum();
    let block = crate::pool::block_size(total, p.threads(), ROW_BLOCK);
    // (matrix, rows) pieces, each at most `block` rows, aligned within its
    // own matrix.
    let pieces: Vec<(usize, std::ops::Range<usize>)> = outs
        .iter()
        .enumerate()
        .flat_map(|(k, o)| {
            let d = o.len() / t;
            (0..d).step_by(block).map(move |s| (k, s..(s + block).min(d)))
        })
        .collect();
    let os: Vec<(SharedOut, usize)> = outs.iter_mut().map(|o| (SharedOut(o.as_mut_ptr()), o.len() / t)).collect();
    p.for_blocks(pieces.len(), 1, &|r| {
        for (k, rows) in pieces[r].iter().cloned() {
            let (o, d) = os[k];
            #[cfg(target_arch = "aarch64")]
            if let Some(xp) = &xp {
                // SAFETY: each piece's rows belong to one output and every
                // piece is handed to exactly one thread; has_mmla checked.
                unsafe { quant_rows_mmla(o, d, rows, ws[k], xp, t) };
                continue;
            }
            // SAFETY: as above.
            unsafe { quant_rows(o, d, rows, ws[k], &xq, &xd, xk.as_deref(), t) }
        }
    });
}

/// [`linear_many_batch_on`] for one token (decode).
pub fn linear_many_on(pool: Option<&ThreadPool>, outs: &mut [&mut [f32]], x: &[f32], ws: &[&Weight]) {
    linear_many_batch_on(pool, outs, x, ws, 1)
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

    /// The one-head `attention_<n>x<hs>` cases (not the GPU prompt's
    /// `attention_causal_*` ones, which share the prefix).
    fn attention_cases() -> Vec<std::path::PathBuf> {
        cases("attention_")
            .into_iter()
            .filter(|p| !p.file_name().unwrap().to_str().unwrap().starts_with("attention_causal_"))
            .collect()
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
    /// model's path: one token at a time (Q8_K, `sdot`) and all `t` at once
    /// (Q8_0 with `smmla` on rows unpacked per group, where i8mm exists;
    /// Q8_K otherwise), on one and three threads. The decode reference is
    /// always the float64 "k" one; the batched reference is "b" only on the
    /// `smmla` path.
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
                #[cfg(target_arch = "aarch64")]
                let prefill_q8_0 = t > 1 && has_mmla();
                #[cfg(not(target_arch = "aarch64"))]
                let prefill_q8_0 = false;
                let bytes = |name: &str| std::fs::read(dir.join(format!("w_{name}.bin"))).unwrap();
                for (name, w) in [("q4_k", Weight::Q4_K(Store::Owned(blocks_from_bytes(&bytes("q4_k"))))), ("q6_k", Weight::Q6_K(Store::Owned(blocks_from_bytes(&bytes("q6_k")))))] {
                    let want = arr(&dir, &format!("out_{name}_{}", if prefill_q8_0 { "b" } else { "k" }));
                    let want_one = arr(&dir, &format!("out_{name}_k"));
                    let d = want.len() / t;
                    let bound = 1e-6 * (n as f32).sqrt();
                    let mut first = None;
                    for p in [None, Some(&pool)] {
                        let mut all = vec![f32::NAN; t * d];
                        linear_batch_on(p, &mut all, &x, &w, t);
                        let (_, rel) = max_errors(&all, &want);
                        assert!(rel < bound, "{} {name} batch: rel {rel:.2e}", dir.display());
                        // Threads split rows; every output is computed the
                        // same way whichever thread has it.
                        assert!(first.get_or_insert_with(|| all.clone()) == &all, "{} {name}: threads changed the result", dir.display());
                        for tt in 0..t {
                            let mut one = vec![f32::NAN; d];
                            linear_on(p, &mut one, &x[tt * n..(tt + 1) * n], &w);
                            let (_, rel) = max_errors(&one, &want_one[tt * d..(tt + 1) * d]);
                            assert!(rel < bound, "{} {name} token {tt}: rel {rel:.2e}", dir.display());
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    if d.is_multiple_of(4) {
                        let r = match &w {
                            Weight::Q4_K(b) => Weight::Q4_Kx4(crate::quant::repack_q4_k(b, d)),
                            Weight::Q6_K(b) => Weight::Q6_Kx4(crate::quant::repack_q6_k(b, d)),
                            _ => unreachable!(),
                        };
                        for p in [None, Some(&pool)] {
                            let mut all = vec![f32::NAN; t * d];
                            linear_batch_on(p, &mut all, &x, &r, t);
                            let (_, rel) = max_errors(&all, &want);
                            assert!(rel < bound, "{} {name} x4 batch: rel {rel:.2e}", dir.display());
                            for tt in 0..t {
                                let mut one = vec![f32::NAN; d];
                                linear_on(p, &mut one, &x[tt * n..(tt + 1) * n], &r);
                                let (_, rel) = max_errors(&one, &want_one[tt * d..(tt + 1) * d]);
                                assert!(rel < bound, "{} {name} x4 token {tt}: rel {rel:.2e}", dir.display());
                            }
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
        let cs = attention_cases();
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

    /// Flash-decoding on the same cases (head sizes 64 and 48): the context
    /// cut into parts of 1, 7 and 64 positions, each part computed with
    /// unnormalised weights for two query heads (q and −0.5·q) and the parts
    /// merged, against numpy's answer for q.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn attention_parts_merge_to_numpy() {
        let mut checked = 0;
        for dir in &attention_cases() {
            let (q, k, v, want) = (arr(dir, "q"), arr(dir, "k"), arr(dir, "v"), arr(dir, "out"));
            let hs = q.len();
            if hs != 64 && hs != 48 {
                continue;
            }
            let n = k.len() / hs;
            let bound = 2e-6 * ((hs as f32).sqrt() + (n as f32).sqrt()).max(4.0);
            let qg: Vec<f32> = [1.0f32, -0.5].iter().flat_map(|&a| q.iter().map(move |x| x * a)).collect();
            for chunk in [1, 7, 64] {
                let parts: Vec<Vec<f32>> = (0..n)
                    .step_by(chunk)
                    .map(|t0| {
                        let len = chunk.min(n - t0);
                        let (mut part, mut scores) = (vec![f32::NAN; 2 * hs + 4], vec![f32::NAN; 2 * len]);
                        let (keys, values) = (|t: usize| &k[(t0 + t) * hs..(t0 + t + 1) * hs], |t: usize| &v[(t0 + t) * hs..(t0 + t + 1) * hs]);
                        neon::attention_group_part(&mut part, &qg, hs, keys, values, len, &mut scores);
                        part
                    })
                    .collect();
                let ps: Vec<(&[f32], f32, f32)> = parts.iter().map(|p| (&p[..hs], p[2 * hs], p[2 * hs + 1])).collect();
                let mut out = vec![f32::NAN; hs];
                neon::merge_parts(&mut out, &ps);
                let (_, rel) = max_errors(&out, &want);
                assert!(rel <= bound, "{} chunk {chunk}: rel {rel:.2e} > {bound:.2e}", dir.display());
                checked += 1;
            }
        }
        println!("attention parts: {checked} (case, chunk) pairs vs numpy");
    }

    /// The grouped kernel on the same cases: three heads with the query
    /// scaled by 1, -0.5 and 2 (numpy's answer is for the first; the others
    /// are checked to equal the per-head kernel bit for bit, which is the
    /// grouped kernel's contract). Two heads in a row must not share
    /// scores.
    #[test]
    fn attention_group_matches_numpy() {
        for dir in &attention_cases() {
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

    #[test]
    fn short_conv_matches_numpy() {
        for dir in cases("shortconv_") {
            let (u, w, st, want, end) = (arr(&dir, "u"), arr(&dir, "weight"), arr(&dir, "state"), arr(&dir, "out"), arr(&dir, "state_out"));
            let dim = st.len() / 6;
            let width = w.len() / dim;
            let t = u.len() / dim;
            let mut state = st.clone();
            let mut out = vec![f32::NAN; u.len()];
            short_conv_rows(&u, &w, &mut state, dim, width, &mut out, t);
            let (_, rel) = max_errors(&out, &want);
            assert!(rel < 1e-5, "{}: out rel {rel:.2e}", dir.display());
            assert_eq!(state, end, "{}: final state", dir.display());
        }
    }

    #[test]
    fn rmsnorm_heads_matches_numpy() {
        for dir in cases("headnorm_") {
            let (mut x, weight, want) = (arr(&dir, "x"), arr(&dir, "weight"), arr(&dir, "out"));
            let hs = weight.len();
            rmsnorm_heads(&mut x, &weight, hs, 1e-5);
            let (_, rel) = max_errors(&x, &want);
            assert!(rel < 1e-5, "{}: rel {rel:.2e}", dir.display());
        }
    }

    #[test]
    fn gate_matches_numpy() {
        for dir in cases("gate_") {
            let (mut x, gates, want) = (arr(&dir, "x"), arr(&dir, "gates"), arr(&dir, "out"));
            let heads = gates.len();
            let hs = x.len() / heads;
            gate_heads(&mut x, &gates, hs);
            let (_, rel) = max_errors(&x, &want);
            assert!(rel < 1e-6, "{}: rel {rel:.2e}", dir.display());
        }
    }

    #[test]
    fn rope_partial_matches_numpy() {
        for dir in cases("partial_rope_") {
            let (mut x, want) = (arr(&dir, "x"), arr(&dir, "out"));
            let name = dir.file_name().unwrap().to_str().unwrap().to_string();
            let body = name.trim_start_matches("partial_rope_");
            let (shape, rest) = body.split_once("_p").unwrap();
            let (pos, pairs) = rest.split_once("_r").unwrap();
            let hs: usize = shape.split('x').nth(1).unwrap().parse().unwrap();
            let pos: usize = pos.parse().unwrap();
            let pairs: usize = pairs.parse().unwrap();
            rope_halves_pairs(&mut x, pos, hs, 1e6, pairs);
            let (_, rel) = max_errors(&x, &want);
            let bound = 2e-6 + 2.0 * pos as f32 * f32::EPSILON;
            assert!(rel < bound, "{name}: rel {rel:.2e}");
        }
    }

    #[test]
    fn attention_window_matches_numpy() {
        for dir in cases("window_attention_") {
            let (q, k, v, window, want) = (arr(&dir, "q"), arr(&dir, "k"), arr(&dir, "v"), arr(&dir, "window"), arr(&dir, "out"));
            let hs = q.len();
            let n = k.len() / hs;
            let name = dir.file_name().unwrap().to_str().unwrap().to_string();
            let pos: usize = name.rsplit("_p").next().unwrap().split('_').next().unwrap().parse().unwrap();
            let window = window[0] as usize;
            let (mut out, mut scores) = (vec![f32::NAN; hs], vec![f32::NAN; n + 3]);
            attention_head_window(&mut out, &q, |t| &k[t * hs..(t + 1) * hs], |t| &v[t * hs..(t + 1) * hs], pos, window, &mut scores);
            let (_, rel) = max_errors(&out, &want);
            assert!(rel < 1e-5, "{}: rel {rel:.2e}", dir.display());
        }
    }
}
