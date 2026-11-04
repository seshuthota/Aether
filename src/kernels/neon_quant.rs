//! NEON dot products of one quantized row with an activation vector.
//!
//! (a) kernels widen the int8 weights to f32 (int8 → int16 → int32 → f32,
//! three steps per 8 values) and FMA with the f32 activation. (b) kernels
//! take the activation already quantized to Q8_0 and use `sdot`
//! (`vdotq_s32`, the `dotprod` extension): one instruction multiplies 16
//! int8 pairs and adds them in groups of 4 into 4 int32 lanes. A 32-weight
//! block is then two `sdot`s, one int→float conversion and one FMA by the
//! block's combined scale `d_w · d_x`.
//!
//! Blocks are processed two at a time into two accumulators so consecutive
//! iterations do not wait on each other (the same idea as the f32 GEMV).
//!
//! For prefill (several tokens), the i8mm kernels at the end use `smmla`:
//! two rows × two tokens per instruction, in tiles of four rows × four
//! tokens.

use crate::quant::{BlockQ4_0, BlockQ4_0x4, BlockQ4_K, BlockQ4_Kx4, BlockQ5_0, BlockQ5_0x4, BlockQ6_K, BlockQ6_Kx4, BlockQ8_0, BlockQ8_0x4, BlockQ8_K, QK, f16_to_f32, q4_k_scales_mins};
use std::arch::aarch64::*;

/// 8 int8 → two float32x4.
#[inline(always)]
fn widen8(v: int8x8_t) -> (float32x4_t, float32x4_t) {
    // SAFETY: register-only NEON ops; NEON is part of the AArch64 baseline.
    unsafe {
        let w = vmovl_s8(v);
        (vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))))
    }
}

/// FMA of 16 int8 weights (as f32) with 16 floats at `x`.
///
/// # Safety
/// `x` must point to 16 readable floats.
#[inline(always)]
unsafe fn fma16(acc: float32x4_t, q: int8x16_t, x: *const f32) -> float32x4_t {
    unsafe {
        let (a, b) = widen8(vget_low_s8(q));
        let (c, d) = widen8(vget_high_s8(q));
        let mut s = vfmaq_f32(acc, a, vld1q_f32(x));
        s = vfmaq_f32(s, b, vld1q_f32(x.add(4)));
        s = vfmaq_f32(s, c, vld1q_f32(x.add(8)));
        vfmaq_f32(s, d, vld1q_f32(x.add(12)))
    }
}

/// Q4_0 nibbles of one block → its 32 weights as int8 in −8..=7:
/// (elements 0..16, elements 16..32).
#[inline(always)]
fn unpack_q4(qs: uint8x16_t) -> (int8x16_t, int8x16_t) {
    // SAFETY: register-only NEON ops; NEON is part of the AArch64 baseline.
    unsafe {
        let eight = vdupq_n_s8(8);
        let lo = vreinterpretq_s8_u8(vandq_u8(qs, vdupq_n_u8(0x0F)));
        let hi = vreinterpretq_s8_u8(vshrq_n_u8(qs, 4));
        (vsubq_s8(lo, eight), vsubq_s8(hi, eight))
    }
}

/// One Q5_0 block → its 32 weights as int8 in −16..=15: (elements 0..16,
/// elements 16..32). The fifth bits: each byte of `qh` is copied to eight
/// lanes (a table lookup), and lane k tests bit k % 8 of its byte.
#[inline(always)]
fn unpack_q5(w: &BlockQ5_0) -> (int8x16_t, int8x16_t) {
    const IDX: [[u8; 16]; 2] = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]];
    const BIT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    // SAFETY: 16 bytes of nibbles per block and 16 per table row; the rest
    // are register-only NEON ops.
    unsafe {
        let qs = vld1q_u8(w.qs.as_ptr());
        let h = vreinterpretq_u8_u32(vdupq_n_u32(u32::from_le_bytes(w.qh)));
        let (bit, five) = (vld1q_u8(BIT.as_ptr()), vdupq_n_u8(0x10));
        let fifth = |i: usize| vandq_u8(vtstq_u8(vqtbl1q_u8(h, vld1q_u8(IDX[i].as_ptr())), bit), five);
        let sixteen = vdupq_n_s8(16);
        let lo = vorrq_u8(vandq_u8(qs, vdupq_n_u8(0x0F)), fifth(0));
        let hi = vorrq_u8(vshrq_n_u8(qs, 4), fifth(1));
        (vsubq_s8(vreinterpretq_s8_u8(lo), sixteen), vsubq_s8(vreinterpretq_s8_u8(hi), sixteen))
    }
}

/// (a) Q8_0 row · f32 x.
pub fn dot_q8_0_f32(row: &[BlockQ8_0], x: &[f32]) -> f32 {
    assert_eq!(x.len(), row.len() * QK);
    // SAFETY: register-only NEON op (AArch64 baseline).
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    for (blk, xs) in row.iter().zip(x.as_chunks::<QK>().0) {
        // SAFETY: each block holds 32 int8s; `xs` holds 32 floats.
        unsafe {
            let q0 = vld1q_s8(blk.qs.as_ptr());
            let q1 = vld1q_s8(blk.qs.as_ptr().add(16));
            let s = fma16(fma16(vdupq_n_f32(0.0), q0, xs.as_ptr()), q1, xs.as_ptr().add(16));
            acc = vfmaq_n_f32(acc, s, f16_to_f32(blk.d));
        }
    }
    // SAFETY: register-only NEON op.
    unsafe { vaddvq_f32(acc) }
}

/// (a) Q4_0 row · f32 x.
pub fn dot_q4_0_f32(row: &[BlockQ4_0], x: &[f32]) -> f32 {
    assert_eq!(x.len(), row.len() * QK);
    // SAFETY: register-only NEON op (AArch64 baseline).
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    for (blk, xs) in row.iter().zip(x.as_chunks::<QK>().0) {
        // SAFETY: 16 bytes of nibbles per block; `xs` holds 32 floats.
        unsafe {
            let (lo, hi) = unpack_q4(vld1q_u8(blk.qs.as_ptr()));
            let s = fma16(fma16(vdupq_n_f32(0.0), lo, xs.as_ptr()), hi, xs.as_ptr().add(16));
            acc = vfmaq_n_f32(acc, s, f16_to_f32(blk.d));
        }
    }
    // SAFETY: register-only NEON op.
    unsafe { vaddvq_f32(acc) }
}

/// Quantizes `x` (a multiple of 32 long) into Q8_0 blocks, byte for byte
/// as [`crate::quant::quantize_q8_0`] (llama.cpp's reference quantizer):
/// the max of |x| is exact, the scale is computed the same way, and
/// `vcvtaq` rounds halves away from zero like C's `roundf`. Eight vector
/// loads per block instead of 32 scalar steps; it runs on one thread before
/// every quantized matmul is shared out, so the others wait for it.
pub fn quantize_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    assert_eq!(x.len(), out.len() * QK, "length must be blocks × 32");
    for (blk, v) in out.iter_mut().zip(x.as_chunks::<QK>().0) {
        // SAFETY: each chunk holds 32 floats (8 loads of 4); each block 32
        // int8s (2 stores of 16).
        unsafe {
            let p = v.as_ptr();
            let a: [float32x4_t; 8] = std::array::from_fn(|k| vld1q_f32(p.add(4 * k)));
            let amax = vmaxvq_f32(a.iter().fold(vdupq_n_f32(0.0), |m, &a| vmaxq_f32(m, vabsq_f32(a))));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            blk.d = crate::quant::f32_to_f16(d);
            let q = a.map(|a| vcvtaq_s32_f32(vmulq_n_f32(a, id)));
            let h = [0, 1, 2, 3].map(|k| vcombine_s16(vmovn_s32(q[2 * k]), vmovn_s32(q[2 * k + 1])));
            vst1q_s8(blk.qs.as_mut_ptr(), vcombine_s8(vmovn_s16(h[0]), vmovn_s16(h[1])));
            vst1q_s8(blk.qs.as_mut_ptr().add(16), vcombine_s8(vmovn_s16(h[2]), vmovn_s16(h[3])));
        }
    }
}

/// True when this CPU has the `sdot` instruction (checked once by std).
pub fn has_dotprod() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

/// (b) Q8_0 row · Q8_0 x with `sdot`. Falls back to the scalar kernel on
/// CPUs without `dotprod`.
pub fn dot_q8_0_q8_0(row: &[BlockQ8_0], x: &[BlockQ8_0]) -> f32 {
    assert_eq!(row.len(), x.len());
    if !has_dotprod() {
        return crate::quant::scalar::dot_q8_0_q8_0(row, x);
    }
    // SAFETY: dotprod is present (checked above).
    unsafe { dot_q8_0_q8_0_sdot(row, x) }
}

#[target_feature(enable = "dotprod")]
unsafe fn dot_q8_0_q8_0_sdot(row: &[BlockQ8_0], x: &[BlockQ8_0]) -> f32 {
    let (mut acc0, mut acc1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let block = |w: &BlockQ8_0, v: &BlockQ8_0| -> (float32x4_t, f32) {
        // SAFETY: each block holds 32 int8s.
        unsafe {
            let z = vdupq_n_s32(0);
            let d = vdotq_s32(vdotq_s32(z, vld1q_s8(w.qs.as_ptr()), vld1q_s8(v.qs.as_ptr())), vld1q_s8(w.qs.as_ptr().add(16)), vld1q_s8(v.qs.as_ptr().add(16)));
            (vcvtq_f32_s32(d), f16_to_f32(w.d) * f16_to_f32(v.d))
        }
    };
    let pairs = row.len() / 2;
    for i in 0..pairs {
        let (s0, d0) = block(&row[2 * i], &x[2 * i]);
        let (s1, d1) = block(&row[2 * i + 1], &x[2 * i + 1]);
        acc0 = vfmaq_n_f32(acc0, s0, d0);
        acc1 = vfmaq_n_f32(acc1, s1, d1);
    }
    if row.len() % 2 == 1 {
        let (s, d) = block(&row[row.len() - 1], &x[row.len() - 1]);
        acc0 = vfmaq_n_f32(acc0, s, d);
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

/// (b) Q4_0 row · Q8_0 x with `sdot`.
pub fn dot_q4_0_q8_0(row: &[BlockQ4_0], x: &[BlockQ8_0]) -> f32 {
    assert_eq!(row.len(), x.len());
    if !has_dotprod() {
        return crate::quant::scalar::dot_q4_0_q8_0(row, x);
    }
    // SAFETY: dotprod is present (checked above).
    unsafe { dot_q4_0_q8_0_sdot(row, x) }
}

#[target_feature(enable = "dotprod")]
unsafe fn dot_q4_0_q8_0_sdot(row: &[BlockQ4_0], x: &[BlockQ8_0]) -> f32 {
    let (mut acc0, mut acc1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let block = |w: &BlockQ4_0, v: &BlockQ8_0| -> (float32x4_t, f32) {
        // SAFETY: 16 bytes of nibbles and 32 int8s per block.
        unsafe {
            let (lo, hi) = unpack_q4(vld1q_u8(w.qs.as_ptr()));
            let z = vdupq_n_s32(0);
            let d = vdotq_s32(vdotq_s32(z, lo, vld1q_s8(v.qs.as_ptr())), hi, vld1q_s8(v.qs.as_ptr().add(16)));
            (vcvtq_f32_s32(d), f16_to_f32(w.d) * f16_to_f32(v.d))
        }
    };
    let pairs = row.len() / 2;
    for i in 0..pairs {
        let (s0, d0) = block(&row[2 * i], &x[2 * i]);
        let (s1, d1) = block(&row[2 * i + 1], &x[2 * i + 1]);
        acc0 = vfmaq_n_f32(acc0, s0, d0);
        acc1 = vfmaq_n_f32(acc1, s1, d1);
    }
    if row.len() % 2 == 1 {
        let (s, d) = block(&row[row.len() - 1], &x[row.len() - 1]);
        acc0 = vfmaq_n_f32(acc0, s, d);
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

/// (b) Q5_0 row · Q8_0 x with `sdot`.
pub fn dot_q5_0_q8_0(row: &[BlockQ5_0], x: &[BlockQ8_0]) -> f32 {
    assert_eq!(row.len(), x.len());
    if !has_dotprod() {
        return crate::quant::scalar::dot_q5_0_q8_0(row, x);
    }
    // SAFETY: dotprod is present (checked above).
    unsafe { dot_q5_0_q8_0_sdot(row, x) }
}

#[target_feature(enable = "dotprod")]
unsafe fn dot_q5_0_q8_0_sdot(row: &[BlockQ5_0], x: &[BlockQ8_0]) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    for (w, v) in row.iter().zip(x) {
        // SAFETY: 32 int8s per activation block.
        unsafe {
            let (lo, hi) = unpack_q5(w);
            let z = vdupq_n_s32(0);
            let d = vdotq_s32(vdotq_s32(z, lo, vld1q_s8(v.qs.as_ptr())), hi, vld1q_s8(v.qs.as_ptr().add(16)));
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(d), f16_to_f32(w.d) * f16_to_f32(v.d));
        }
    }
    vaddvq_f32(acc)
}

/// Four f16 values (raw bits) → four f32 with one `fcvtl` instruction
/// (base ARMv8 NEON; no fp16 extension needed).
#[inline(always)]
fn f16x4_to_f32(h: [u16; 4]) -> float32x4_t {
    let r: float32x4_t;
    // SAFETY: a register-to-register conversion; touches no memory.
    unsafe {
        let v = vld1_u16(h.as_ptr());
        std::arch::asm!("fcvtl {0:v}.4s, {1:v}.4h", out(vreg) r, in(vreg) v, options(pure, nomem, nostack));
    }
    r
}

/// Folds one block's four integer dot products into the four row
/// accumulators, each scaled by its lane of `scales`.
#[inline(always)]
fn fold4(acc: &mut [float32x4_t; 4], d: [int32x4_t; 4], scales: float32x4_t) {
    // SAFETY: register-only NEON ops.
    unsafe {
        acc[0] = vfmaq_laneq_f32::<0>(acc[0], vcvtq_f32_s32(d[0]), scales);
        acc[1] = vfmaq_laneq_f32::<1>(acc[1], vcvtq_f32_s32(d[1]), scales);
        acc[2] = vfmaq_laneq_f32::<2>(acc[2], vcvtq_f32_s32(d[2]), scales);
        acc[3] = vfmaq_laneq_f32::<3>(acc[3], vcvtq_f32_s32(d[3]), scales);
    }
}

/// Four Q8_0 rows · one Q8_0 activation at once (style b, `sdot`). `xd`
/// holds the activation's block scales already converted to f32. Each
/// activation block is loaded once for the four rows; the four rows' scales
/// are converted together; each row's 32-wide integer dot is two `sdot`s.
///
/// # Safety
/// The CPU must have `dotprod`; every row must have `x.len()` blocks.
#[target_feature(enable = "dotprod")]
pub unsafe fn dot4_q8_0_q8_0(rows: [&[BlockQ8_0]; 4], x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    let mut acc = [vdupq_n_f32(0.0); 4];
    for (b, (xb, &dx)) in x.iter().zip(xd).enumerate() {
        // SAFETY: each block holds 32 int8s; rows have x.len() blocks.
        unsafe {
            let (xl, xh) = (vld1q_s8(xb.qs.as_ptr()), vld1q_s8(xb.qs.as_ptr().add(16)));
            let z = vdupq_n_s32(0);
            let d = rows.map(|r| {
                let w = r.get_unchecked(b);
                vdotq_s32(vdotq_s32(z, vld1q_s8(w.qs.as_ptr()), xl), vld1q_s8(w.qs.as_ptr().add(16)), xh)
            });
            let scales = vmulq_n_f32(f16x4_to_f32(rows.map(|r| r.get_unchecked(b).d)), dx);
            fold4(&mut acc, d, scales);
        }
    }
    acc.map(|a| vaddvq_f32(a))
}

/// [`dot4_q8_0_q8_0`] for Q4_0 rows.
///
/// # Safety
/// The CPU must have `dotprod`; every row must have `x.len()` blocks.
#[target_feature(enable = "dotprod")]
pub unsafe fn dot4_q4_0_q8_0(rows: [&[BlockQ4_0]; 4], x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    let mut acc = [vdupq_n_f32(0.0); 4];
    for (b, (xb, &dx)) in x.iter().zip(xd).enumerate() {
        // SAFETY: 16 bytes of nibbles / 32 int8s per block; rows have
        // x.len() blocks.
        unsafe {
            let (xl, xh) = (vld1q_s8(xb.qs.as_ptr()), vld1q_s8(xb.qs.as_ptr().add(16)));
            let z = vdupq_n_s32(0);
            let d = rows.map(|r| {
                let (lo, hi) = unpack_q4(vld1q_u8(r.get_unchecked(b).qs.as_ptr()));
                vdotq_s32(vdotq_s32(z, lo, xl), hi, xh)
            });
            let scales = vmulq_n_f32(f16x4_to_f32(rows.map(|r| r.get_unchecked(b).d)), dx);
            fold4(&mut acc, d, scales);
        }
    }
    acc.map(|a| vaddvq_f32(a))
}

/// [`dot4_q8_0_q8_0`] for Q5_0 rows.
///
/// # Safety
/// The CPU must have `dotprod`; every row must have `x.len()` blocks.
#[target_feature(enable = "dotprod")]
pub unsafe fn dot4_q5_0_q8_0(rows: [&[BlockQ5_0]; 4], x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    let mut acc = [vdupq_n_f32(0.0); 4];
    for (b, (xb, &dx)) in x.iter().zip(xd).enumerate() {
        // SAFETY: 32 int8s per activation block; rows have x.len() blocks.
        unsafe {
            let (xl, xh) = (vld1q_s8(xb.qs.as_ptr()), vld1q_s8(xb.qs.as_ptr().add(16)));
            let z = vdupq_n_s32(0);
            let d = rows.map(|r| {
                let (lo, hi) = unpack_q5(r.get_unchecked(b));
                vdotq_s32(vdotq_s32(z, lo, xl), hi, xh)
            });
            let scales = vmulq_n_f32(f16x4_to_f32(rows.map(|r| r.get_unchecked(b).d)), dx);
            fold4(&mut acc, d, scales);
        }
    }
    acc.map(|a| vaddvq_f32(a))
}

// --- K-quants with Q8_K (`sdot`) ---------------------------------------
//
// One Q8_K activation block covers a whole 256-weight super-block: a single
// f32 scale and the sums of each 16 values. A Q4_K sub-block's scale and min
// therefore apply in int32, and the float work is one convert and two FMAs
// per super-block per row instead of one per 32 values.

/// One Q8_K activation block: 256 int8 values as 16 vectors, the f32 scale,
/// and the sums of each 16 values.
struct XKK {
    v: [[int8x16_t; 2]; 8],
    d: f32,
    bsums: [i16; 16],
}

#[inline(always)]
fn load_xkk(x: &BlockQ8_K) -> XKK {
    // SAFETY: the block holds 256 int8s.
    unsafe {
        let v: [[int8x16_t; 2]; 8] = std::array::from_fn(|j| {
            let q = x.qs.as_ptr().add(32 * j);
            [vld1q_s8(q), vld1q_s8(q.add(16))]
        });
        XKK { v, d: x.d, bsums: x.bsums }
    }
}

/// One Q4_K super-block · `x`, added into `acc`. The eight sub-blocks'
/// `sdot` sums are scaled by their 6-bit scales in int32; the min term is
/// `bsums · mins` with `vmlal_s16`; the result converts once. The scales
/// and mins are decoded all at once ([`q4_k_scales_mins`]).
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn q4_k_block_q8_k(w: &BlockQ4_K, x: &XKK, acc: &mut float32x4_t) {
    let (sc, m) = q4_k_scales_mins(&w.scales);
    // SAFETY: 128 bytes of nibbles (offsets below 128); register-only ops.
    unsafe {
        let m16 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(m.as_ptr())));
        let (m4, z) = (vdupq_n_u8(0x0F), vdupq_n_s32(0));
        // Each sub-block's two `sdot`s stay in int32; `sc[j]` scales them.
        let mut sumi = z;
        for c in 0..4 {
            let (a, b) = (vld1q_u8(w.qs.as_ptr().add(32 * c)), vld1q_u8(w.qs.as_ptr().add(32 * c + 16)));
            let (j0, j1) = (2 * c, 2 * c + 1);
            let s0 = vdotq_s32(vdotq_s32(z, vreinterpretq_s8_u8(vandq_u8(a, m4)), x.v[j0][0]), vreinterpretq_s8_u8(vandq_u8(b, m4)), x.v[j0][1]);
            let s1 = vdotq_s32(vdotq_s32(z, vreinterpretq_s8_u8(vshrq_n_u8(a, 4)), x.v[j1][0]), vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), x.v[j1][1]);
            sumi = vmlaq_n_s32(sumi, s0, sc[j0] as i32);
            sumi = vmlaq_n_s32(sumi, s1, sc[j1] as i32);
        }
        // bsums[2j] + bsums[2j + 1] is the sum of sub-block j's 32 values.
        let b32 = vpaddq_s16(vld1q_s16(x.bsums.as_ptr()), vld1q_s16(x.bsums.as_ptr().add(8)));
        let mut sumi_mins = z;
        sumi_mins = vmlal_s16(sumi_mins, vget_low_s16(b32), vget_low_s16(m16));
        sumi_mins = vmlal_s16(sumi_mins, vget_high_s16(b32), vget_high_s16(m16));
        let (d, dmin) = (vdupq_n_f32(f16_to_f32(w.d) * x.d), vdupq_n_f32(f16_to_f32(w.dmin) * x.d));
        *acc = vfmaq_f32(*acc, vcvtq_f32_s32(sumi), d);
        *acc = vfmsq_f32(*acc, vcvtq_f32_s32(sumi_mins), dmin);
    }
}

/// One Q6_K super-block · `x`, added into `acc`.
///
/// Each value is built as the int8 4·(q − 32): `q − 32` in 6-bit two's
/// complement is `q` with bit 5 flipped, so the two high bits (from `qh`,
/// with every pair's upper bit flipped once by XOR 0xAA) go to bits 6..7 and
/// `sri #2` inserts the nibble, held at bits 4..7, below them. Three
/// instructions per vector; the 4 comes out in the float scale. The 16
/// sub-block sums are scaled by their int8 scales in int32 and converted
/// once per super-block.
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn q6_k_block_q8_k(w: &BlockQ6_K, x: &XKK, acc: &mut float32x4_t) {
    let d = f16_to_f32(w.d) * 0.25 * x.d;
    let mut sumi = vdupq_n_s32(0);
    for n in 0..2 {
        // SAFETY: ql holds 128 bytes and qh 64; n < 2 keeps every 16-byte
        // load inside them (64n + 48 + 16 <= 128, 32n + 16 + 16 <= 64).
        unsafe {
            let ql = w.ql.as_ptr().add(64 * n);
            let flip = vdupq_n_u8(0xAA);
            let qh = [veorq_u8(vld1q_u8(w.qh.as_ptr().add(32 * n)), flip), veorq_u8(vld1q_u8(w.qh.as_ptr().add(32 * n + 16)), flip)];
            for k in 0..4 {
                let j = 4 * n + k;
                let lo = |i: usize| {
                    let v = vld1q_u8(ql.add(32 * (k % 2) + 16 * i));
                    if k < 2 { vshlq_n_u8::<4>(v) } else { vandq_u8(v, vdupq_n_u8(0xF0)) }
                };
                // Bits 2k, 2k + 1 of qh to bits 6, 7.
                let q = |i: usize| vreinterpretq_s8_u8(sri2(vshlq_u8(qh[i], vdupq_n_s8(6 - 2 * k as i8)), lo(i)));
                let z = vdupq_n_s32(0);
                let s = vmulq_n_s32(vdotq_s32(z, q(0), x.v[j][0]), w.scales[2 * j] as i32);
                let s = vmlaq_n_s32(s, vdotq_s32(z, q(1), x.v[j][1]), w.scales[2 * j + 1] as i32);
                sumi = vaddq_s32(sumi, s);
            }
        }
    }
    *acc = vfmaq_n_f32(*acc, vcvtq_f32_s32(sumi), d);
}

/// (b) Q4_K row · Q8_K x with `sdot`.
pub fn dot_q4_k_q8_k(row: &[BlockQ4_K], x: &[BlockQ8_K]) -> f32 {
    assert_eq!(row.len(), x.len());
    if !has_dotprod() {
        return crate::quant::scalar::dot_q4_k_q8_k(row, x);
    }
    // SAFETY: dotprod is present (checked above); one block per super-block.
    unsafe { dot4_q4_k_q8_k([row; 4], x)[0] }
}

/// (b) Q6_K row · Q8_K x with `sdot`.
pub fn dot_q6_k_q8_k(row: &[BlockQ6_K], x: &[BlockQ8_K]) -> f32 {
    assert_eq!(row.len(), x.len());
    if !has_dotprod() {
        return crate::quant::scalar::dot_q6_k_q8_k(row, x);
    }
    // SAFETY: as in dot_q4_k_q8_k.
    unsafe { dot4_q6_k_q8_k([row; 4], x)[0] }
}

/// Four Q4_K rows · one Q8_K activation.
///
/// # Safety
/// The CPU must have `dotprod`; every row must have `x.len()` blocks.
#[target_feature(enable = "dotprod")]
pub unsafe fn dot4_q4_k_q8_k(rows: [&[BlockQ4_K]; 4], x: &[BlockQ8_K]) -> [f32; 4] {
    let mut acc = [vdupq_n_f32(0.0); 4];
    for (b, xb) in x.iter().enumerate() {
        // SAFETY: every row has x.len() blocks; dotprod checked by the caller.
        unsafe {
            let xk = load_xkk(xb);
            for r in 0..4 {
                q4_k_block_q8_k(rows[r].get_unchecked(b), &xk, &mut acc[r]);
            }
        }
    }
    acc.map(|a| vaddvq_f32(a))
}

/// Four Q6_K rows · one Q8_K activation.
///
/// # Safety
/// As for [`dot4_q4_k_q8_k`].
#[target_feature(enable = "dotprod")]
pub unsafe fn dot4_q6_k_q8_k(rows: [&[BlockQ6_K]; 4], x: &[BlockQ8_K]) -> [f32; 4] {
    let mut acc = [vdupq_n_f32(0.0); 4];
    for (b, xb) in x.iter().enumerate() {
        // SAFETY: every row has x.len() blocks; dotprod checked by the caller.
        unsafe {
            let xk = load_xkk(xb);
            for r in 0..4 {
                q6_k_block_q8_k(rows[r].get_unchecked(b), &xk, &mut acc[r]);
            }
        }
    }
    acc.map(|a| vaddvq_f32(a))
}

/// Four activation bytes, broadcast to all four lanes: one `sdot` then
/// multiplies each of four rows by the same four values.
#[inline(always)]
unsafe fn dup4(p: *const i8) -> int8x16_t {
    // SAFETY: p points at four readable bytes; the offsets are 4-aligned.
    unsafe { vreinterpretq_s8_s32(vld1q_dup_s32(p as *const i32)) }
}

/// Four `u8` values → int32x4 (the repacked K scales and mins).
#[inline(always)]
unsafe fn u8x4_to_s32(p: &[u8]) -> int32x4_t {
    // SAFETY: p holds four bytes.
    unsafe { vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p.as_ptr()))))) }
}

/// Four `i8` values → int32x4 (Q6_K's sub-block scales).
#[inline(always)]
unsafe fn i8x4_to_s32(p: &[i8]) -> int32x4_t {
    // SAFETY: p holds four bytes.
    unsafe { vmovl_s16(vget_low_s16(vmovl_s8(vld1_s8(p.as_ptr())))) }
}

/// Four repacked Q4_K rows · one Q8_K activation. Each `sdot` lane is one
/// row, so the result needs no horizontal reduction: the four lanes of `acc`
/// are the four rows.
///
/// # Safety
/// The CPU must have `dotprod`; `group` and `x` must have the same length.
#[target_feature(enable = "dotprod")]
pub unsafe fn dot4_q4_k_q8_k_x4(group: &[BlockQ4_Kx4], x: &[BlockQ8_K]) -> [f32; 4] {
    assert_eq!(group.len(), x.len());
    let mut acc = vdupq_n_f32(0.0);
    for (g, xb) in group.iter().zip(x) {
        // SAFETY: group and x have the same length; register-only ops.
        unsafe {
            let (m4, z) = (vdupq_n_u8(0x0F), vdupq_n_s32(0));
            let mut sumi = z;
            for c in 0..4 {
                let (j0, j1) = (2 * c, 2 * c + 1);
                let (sc0, sc1) = (u8x4_to_s32(&g.scales[j0 * 4..]), u8x4_to_s32(&g.scales[j1 * 4..]));
                let (mut s0, mut s1) = (z, z);
                for v in 0..8 {
                    let at = if v < 4 { 4 * v } else { 16 + 4 * (v - 4) };
                    let q = vld1q_u8(g.qs.as_ptr().add(128 * c + 16 * v));
                    let a0 = dup4(xb.qs.as_ptr().add(32 * j0 + at));
                    let a1 = dup4(xb.qs.as_ptr().add(32 * j1 + at));
                    s0 = vdotq_s32(s0, vreinterpretq_s8_u8(vandq_u8(q, m4)), a0);
                    s1 = vdotq_s32(s1, vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), a1);
                }
                sumi = vmlaq_s32(sumi, s0, sc0);
                sumi = vmlaq_s32(sumi, s1, sc1);
            }
            let mut minsum = z;
            for j in 0..8 {
                let b = xb.bsums[2 * j] as i32 + xb.bsums[2 * j + 1] as i32;
                minsum = vmlaq_n_s32(minsum, u8x4_to_s32(&g.mins[j * 4..]), b);
            }
            let d = vmulq_n_f32(f16x4_to_f32(g.d), xb.d);
            let dmin = vmulq_n_f32(f16x4_to_f32(g.dmin), xb.d);
            acc = vfmaq_f32(acc, vcvtq_f32_s32(sumi), d);
            acc = vfmsq_f32(acc, vcvtq_f32_s32(minsum), dmin);
        }
    }
    let mut out = [0.0; 4];
    // SAFETY: out holds four floats.
    unsafe { vst1q_f32(out.as_mut_ptr(), acc) };
    out
}

/// Four repacked Q6_K rows · one Q8_K activation. As the Q4_K version, each
/// `sdot` lane is one row.
///
/// # Safety
/// The CPU must have `dotprod`; `group` and `x` must have the same length.
#[target_feature(enable = "dotprod")]
pub unsafe fn dot4_q6_k_q8_k_x4(group: &[BlockQ6_Kx4], x: &[BlockQ8_K]) -> [f32; 4] {
    assert_eq!(group.len(), x.len());
    let mut acc = vdupq_n_f32(0.0);
    for (g, xb) in group.iter().zip(x) {
        // SAFETY: group and x have the same length; register-only ops.
        unsafe {
            let mut sumi = vdupq_n_s32(0);
            for s in 0..16 {
                let (n, t) = (s / 8, s % 8);
                let v0 = 4 * (t % 2) + 8 * ((t / 2) % 2);
                let (high, k) = (t >= 4, t / 2);
                let sc = i8x4_to_s32(&g.scales[s * 4..]);
                let mut s16 = vdupq_n_s32(0);
                for j in 0..4 {
                    let ql = vld1q_u8(g.ql.as_ptr().add(n * 256 + 16 * (v0 + j)));
                    let qh = vld1q_u8(g.qh.as_ptr().add(n * 128 + 16 * ((v0 % 8) + j)));
                    let lo = if high { vandq_u8(ql, vdupq_n_u8(0xF0)) } else { vshlq_n_u8::<4>(ql) };
                    let q = sri2(vshlq_u8(qh, vdupq_n_s8(6 - 2 * k as i8)), lo);
                    let a = dup4(xb.qs.as_ptr().add(16 * s + 4 * j));
                    s16 = vdotq_s32(s16, vreinterpretq_s8_u8(q), a);
                }
                sumi = vmlaq_s32(sumi, s16, sc);
            }
            let d = vmulq_n_f32(f16x4_to_f32(g.d), 0.25);
            let d = vmulq_n_f32(d, xb.d);
            acc = vfmaq_f32(acc, vcvtq_f32_s32(sumi), d);
        }
    }
    let mut out = [0.0; 4];
    // SAFETY: out holds four floats.
    unsafe { vst1q_f32(out.as_mut_ptr(), acc) };
    out
}

// --- i8mm: `smmla`, two rows × two tokens per instruction ---------------
//
// `smmla d, a, b` treats each 16-byte register as a 2×8 int8 matrix (bytes
// 0..8 are row 0, 8..16 row 1) and adds a · bᵀ, a 2×2 int32 matrix, to d:
//
//     a = [w0 k0..8 | w1 k0..8]      d.lane0 += w0·x0    d.lane1 += w0·x1
//     b = [x0 k0..8 | x1 k0..8]      d.lane2 += w1·x0    d.lane3 += w1·x1
//
// So each weight value loaded serves two tokens, where `sdot` serves one.
// A 32-value block of two rows × two tokens is four `smmla`s, once the rows
// (and the tokens) are interleaved 8 bytes at a time.

/// True when this CPU has the i8mm extension (`smmla`).
pub fn has_i8mm() -> bool {
    std::arch::is_aarch64_feature_detected!("i8mm")
}

/// `d + a · bᵀ` for 2×8 int8 matrices `a`, `b` (see above). Written with
/// `asm!` because the `vmmlaq_s32` intrinsic is not stable on Rust 1.99.
/// Only call it from a `#[target_feature(enable = "i8mm")]` function, so the
/// assembler accepts the instruction once inlined.
#[inline(always)]
fn smmla(d: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = d;
    // SAFETY: a register-only multiply-accumulate; touches no memory. The
    // caller runs only where i8mm is present.
    unsafe { std::arch::asm!("smmla {0:v}.4s, {1:v}.16b, {2:v}.16b", inout(vreg) r, in(vreg) a, in(vreg) b, options(pure, nomem, nostack)) };
    r
}

/// Interleaves two 16-byte vectors 8 bytes at a time:
/// `([a0..8 | b0..8], [a8..16 | b8..16])`.
#[inline(always)]
fn zip8(a: int8x16_t, b: int8x16_t) -> (int8x16_t, int8x16_t) {
    // SAFETY: register-only NEON ops.
    unsafe {
        let (a, b) = (vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b));
        (vreinterpretq_s8_s64(vzip1q_s64(a, b)), vreinterpretq_s8_s64(vzip2q_s64(a, b)))
    }
}


/// The activations of two tokens, one 32-value block, arranged for `smmla`:
/// 8 values of the first token, then the same 8 of the second, four times
/// (`[x0 0..8 | x1 0..8 | x0 8..16 | x1 8..16 | …]`), and the scales
/// `[d0, d1, d0, d1]` as f32 to match the lanes of an `smmla` result. `s`
/// holds each token's Σ q · d in the same lanes (the K-quant kernels'
/// bias terms multiply it).
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct PairQ8 {
    pub qs: [i8; 2 * QK],
    pub d: [f32; 4],
    pub s: [f32; 4],
}

/// Q8_0 activations of `t` tokens (`xq` is `[t, nb]` blocks, `xd` their
/// scales as f32) → `[ceil(t/2), nb]` [`PairQ8`]s: tokens 2p and 2p + 1 in
/// pair p, an odd last token paired with itself. Done once per matmul, so
/// the kernel never interleaves activations itself.
pub fn interleave_pairs(pool: Option<&crate::pool::ThreadPool>, xq: &[BlockQ8_0], xd: &[f32], t: usize) -> Vec<PairQ8> {
    let nb = xq.len() / t;
    let mut out = vec![PairQ8 { qs: [0; 2 * QK], d: [0.0; 4], s: [0.0; 4] }; t.div_ceil(2) * nb];
    crate::pool::par_chunks_mut(pool, &mut out, nb, &|p, out| {
        let (t0, t1) = (2 * p, (2 * p + 1).min(t - 1));
        for (b, o) in out.iter_mut().enumerate() {
            let (x0, x1) = (&xq[t0 * nb + b].qs, &xq[t1 * nb + b].qs);
            let mut qs = [0; 2 * QK];
            for k in 0..QK / 8 {
                qs[16 * k..16 * k + 8].copy_from_slice(&x0[8 * k..8 * k + 8]);
                qs[16 * k + 8..16 * k + 16].copy_from_slice(&x1[8 * k..8 * k + 8]);
            }
            let (d0, d1) = (xd[t0 * nb + b], xd[t1 * nb + b]);
            let sum = |x: &[i8; QK], d: f32| x.iter().map(|&q| q as i32).sum::<i32>() as f32 * d;
            let (s0, s1) = (sum(x0, d0), sum(x1, d1));
            *o = PairQ8 { qs, d: [d0, d1, d0, d1], s: [s0, s1, s0, s1] };
        }
    });
    out
}

/// Token pairs per `smmla` tile in prefill: a tile is 4 rows × 2·NP tokens.
/// Batches of one to three pairs (lookup decoding's verification passes) use
/// tiles of P = 1, 2 or 3 pairs instead, so they do not pay for eight tokens.
pub const NP: usize = 4;

/// The result of a tile of P token pairs: `[row pair][token pair]` →
/// `[r0·t0, r0·t1, r1·t0, r1·t1]`.
pub type Tiles<const P: usize> = [[[f32; 4]; P]; 2];

/// Accumulators `[rp][tp]` → [`Tiles`].
#[inline(always)]
fn store_tiles<const P: usize>(acc: [[float32x4_t; P]; 2]) -> Tiles<P> {
    let mut out = [[[0.0; 4]; P]; 2];
    for (o, a) in out.iter_mut().flatten().zip(acc.iter().flatten()) {
        // SAFETY: each `o` holds 4 floats.
        unsafe { vst1q_f32(o.as_mut_ptr(), *a) };
    }
    out
}

/// One block column of four weight rows, ready for `smmla`: for each row
/// pair (rows 0,1 and rows 2,3) its four 8-value chunks as `[row a chunk |
/// row b chunk]`, and the four rows' f16 scales.
type Rows4 = ([[int8x16_t; 4]; 2], [u16; 4]);

/// Row-major rows (each as values 0..16 and 16..32) → [`Rows4`]: interleaves
/// rows 0,1 and rows 2,3 eight bytes at a time.
#[inline(always)]
fn interleave4(w: [[int8x16_t; 2]; 4], d: [u16; 4]) -> Rows4 {
    let ((a0, a1), (a2, a3)) = (zip8(w[0][0], w[1][0]), zip8(w[0][1], w[1][1]));
    let ((c0, c1), (c2, c3)) = (zip8(w[2][0], w[3][0]), zip8(w[2][1], w[3][1]));
    ([[a0, a1, a2, a3], [c0, c1, c2, c3]], d)
}

/// int32 sums → f32, undoing a ×2^F the weights came in (F = 4 for
/// repacked Q4_0, 3 for repacked Q5_0, 0 otherwise) exactly, by a
/// fixed-point conversion with F fraction bits.
#[inline(always)]
fn to_f32<const F: i32>(s: int32x4_t) -> float32x4_t {
    // SAFETY: register-only NEON ops.
    unsafe {
        match F {
            4 => vcvtq_n_f32_s32::<4>(s),
            3 => vcvtq_n_f32_s32::<3>(s),
            _ => vcvtq_f32_s32(s),
        }
    }
}

/// Four weight rows · the 2·[`NP`] tokens of pairs `x`, over `nb` blocks,
/// where `load(b)` gives block column `b` of the rows as [`Rows4`]. Per
/// block: 8·NP `smmla`s; the weights' loads, unpacking and scales are
/// shared by all NP pairs.
///
/// Returns 2×2 tiles, `[rp][tp]` for row pair `rp` and token pair `tp`,
/// each `[r0·t0, r0·t1, r1·t0, r1·t1]`. Every output has its own lane,
/// integer sums and float additions, so its value does not depend on the
/// rows or tokens it shares a tile with (passing the same row or pair twice
/// is how callers handle the edges), nor on whether the rows were repacked.
///
/// # Safety
/// The CPU must have i8mm; `load` must be valid for `0..nb` and both pair
/// slices must have `nb` entries.
#[target_feature(enable = "i8mm")]
unsafe fn mmla4<const F: i32, const P: usize>(nb: usize, x: [&[PairQ8]; P], load: impl Fn(usize) -> Rows4) -> Tiles<P> {
    // SAFETY: register-only NEON ops (AArch64 baseline).
    let (mut acc, z) = unsafe { ([[vdupq_n_f32(0.0); P]; 2], vdupq_n_s32(0)) };
    for b in 0..nb {
        let (w, d) = load(b);
        // SAFETY: the caller guarantees every pair slice has nb entries;
        // each PairQ8 holds 64 int8s and 4 floats.
        unsafe {
            // [d0, d1, d2, d3] → [d0, d0, d1, d1] and [d2, d2, d3, d3].
            let dw = f16x4_to_f32(d);
            let dw = [vzip1q_f32(dw, dw), vzip2q_f32(dw, dw)];
            for tp in 0..P {
                let p = x[tp].get_unchecked(b);
                let q = p.qs.as_ptr();
                let xv = [vld1q_s8(q), vld1q_s8(q.add(16)), vld1q_s8(q.add(32)), vld1q_s8(q.add(48))];
                let dx = vld1q_f32(p.d.as_ptr());
                for rp in 0..2 {
                    // One chain of four per tile: the tiles (and the next
                    // block's) are independent, which is enough to keep
                    // the `smmla` units busy without extra adds.
                    let s = smmla(smmla(smmla(smmla(z, w[rp][0], xv[0]), w[rp][1], xv[1]), w[rp][2], xv[2]), w[rp][3], xv[3]);
                    acc[rp][tp] = vfmaq_f32(acc[rp][tp], to_f32::<F>(s), vmulq_f32(dw[rp], dx));
                }
            }
        }
    }
    store_tiles(acc)
}

/// Q8_0 rows as 32 int8s (values 0..16, 16..32).
#[inline(always)]
fn load_q8_0(w: &BlockQ8_0) -> [int8x16_t; 2] {
    // SAFETY: each block holds 32 int8s.
    unsafe { [vld1q_s8(w.qs.as_ptr()), vld1q_s8(w.qs.as_ptr().add(16))] }
}

/// Repacked Q8_0: the 16-byte loads are already row pairs.
#[inline(always)]
fn load_q8_0x4(w: &BlockQ8_0x4) -> Rows4 {
    let q = w.qs.as_ptr();
    // SAFETY: each block holds 128 int8s; offsets reach at most 112 + 16.
    let v = |o: usize| unsafe { vld1q_s8(q.add(o)) };
    ([[v(0), v(32), v(64), v(96)], [v(16), v(48), v(80), v(112)]], w.d)
}

/// Repacked Q4_0 (nibbles XOR 8): a shift left by 4 gives the low nibbles
/// ×16 as signed int8, a mask the high ones; no subtraction of 8.
#[inline(always)]
fn load_q4_0x4(w: &BlockQ4_0x4) -> Rows4 {
    let q = w.qs.as_ptr();
    // SAFETY: each block holds 64 bytes; register-only ops after the loads.
    unsafe {
        let (lo, hi) = (|v: uint8x16_t| vreinterpretq_s8_u8(vshlq_n_u8::<4>(v)), |v: uint8x16_t| vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0xF0))));
        // Half h, row pair rp at byte 32h + 16rp: low nibbles are chunk h,
        // high nibbles chunk h + 2.
        let rp = |o: usize| {
            let (h0, h1) = (vld1q_u8(q.add(o)), vld1q_u8(q.add(32 + o)));
            [lo(h0), lo(h1), hi(h0), hi(h1)]
        };
        ([rp(0), rp(16)], w.d)
    }
}

/// `sri a, b, #1`: bit 7 of `a`, bits 0..7 of `b` shifted right by one into
/// bits 0..6. `asm!` because LLVM rewrites `vsriq_n_u8` into the and/or it
/// equals, two instructions instead of one.
#[inline(always)]
fn sri1(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut r = a;
    // SAFETY: a register-to-register instruction; touches no memory.
    unsafe { std::arch::asm!("sri {0:v}.16b, {1:v}.16b, #1", inout(vreg) r, in(vreg) b, options(pure, nomem, nostack)) };
    r
}

/// `sri a, b, #2`: bits 6..7 of `a`, `b` shifted right by two into bits
/// 0..5 (see [`sri1`]).
#[inline(always)]
fn sri2(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let mut r = a;
    // SAFETY: a register-to-register instruction; touches no memory.
    unsafe { std::arch::asm!("sri {0:v}.16b, {1:v}.16b, #2", inout(vreg) r, in(vreg) b, options(pure, nomem, nostack)) };
    r
}

/// Repacked Q5_0 (see [`BlockQ5_0x4`]): each vector is its nibbles moved to
/// bits 3..7 and its inverted fifth bit inserted as bit 7, i.e. 8·(q − 16).
#[inline(always)]
fn load_q5_0x4(w: &BlockQ5_0x4) -> Rows4 {
    let q = w.qs.as_ptr();
    // SAFETY: 64 bytes of nibbles and 16 of fifth bits per block;
    // register-only ops after the loads.
    unsafe {
        let h = vld1q_u8(w.qh.as_ptr());
        // Each nibble to bits 4..7 (as for Q4_0), then `sri #1` shifts it
        // to bits 3..6 under bit 7 of the other operand: bit i of h, moved
        // there. Three instructions per vector.
        let lo = |v: uint8x16_t| vshlq_n_u8::<4>(v);
        let hi = |v: uint8x16_t| vandq_u8(v, vdupq_n_u8(0xF0));
        let fifth = |i: i8, x: uint8x16_t| vreinterpretq_s8_u8(sri1(vshlq_u8(h, vdupq_n_s8(7 - i)), x));
        let rp = |o: usize, base: i8| {
            let (h0, h1) = (vld1q_u8(q.add(o)), vld1q_u8(q.add(32 + o)));
            [fifth(base, lo(h0)), fifth(base + 1, lo(h1)), fifth(base + 2, hi(h0)), fifth(base + 3, hi(h1))]
        };
        ([rp(0, 0), rp(16, 4)], w.d)
    }
}

/// Four row-major Q8_0 rows × four tokens with `smmla`; see [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], every row with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q8_0<const P: usize>(rows: [&[BlockQ8_0]; 4], x: [&[PairQ8]; P]) -> Tiles<P> {
    // SAFETY: forwarded from the caller (b < nb = every row's length).
    unsafe {
        mmla4::<0, P>(x[0].len(), x, |b| {
            let r = rows.map(|r| r.get_unchecked(b));
            interleave4(r.map(load_q8_0), r.map(|r| r.d))
        })
    }
}

/// Four row-major Q4_0 rows × four tokens with `smmla`: each row's nibbles
/// are unpacked once for all four tokens. See [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], every row with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q4_0<const P: usize>(rows: [&[BlockQ4_0]; 4], x: [&[PairQ8]; P]) -> Tiles<P> {
    // SAFETY: forwarded from the caller (b < nb = every row's length);
    // 16 bytes of nibbles per block.
    unsafe {
        mmla4::<0, P>(x[0].len(), x, |b| {
            let r = rows.map(|r| r.get_unchecked(b));
            interleave4(r.map(|r| <[int8x16_t; 2]>::from(unpack_q4(vld1q_u8(r.qs.as_ptr())))), r.map(|r| r.d))
        })
    }
}

/// Four row-major Q5_0 rows × four tokens with `smmla`. See [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], every row with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q5_0<const P: usize>(rows: [&[BlockQ5_0]; 4], x: [&[PairQ8]; P]) -> Tiles<P> {
    // SAFETY: forwarded from the caller (b < nb = every row's length).
    unsafe {
        mmla4::<0, P>(x[0].len(), x, |b| {
            let r = rows.map(|r| r.get_unchecked(b));
            interleave4(r.map(|r| <[int8x16_t; 2]>::from(unpack_q5(r))), r.map(|r| r.d))
        })
    }
}

/// One repacked group of four Q8_0 rows × four tokens: no interleaving in
/// the loop. See [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], `group` with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q8_0x4<const P: usize>(group: &[BlockQ8_0x4], x: [&[PairQ8]; P]) -> Tiles<P> {
    // SAFETY: forwarded from the caller (b < nb = group.len()).
    unsafe { mmla4::<0, P>(x[0].len(), x, |b| load_q8_0x4(group.get_unchecked(b))) }
}

/// One repacked group of four Q4_0 rows × four tokens. See [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], `group` with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q4_0x4<const P: usize>(group: &[BlockQ4_0x4], x: [&[PairQ8]; P]) -> Tiles<P> {
    // SAFETY: forwarded from the caller (b < nb = group.len()).
    unsafe { mmla4::<4, P>(x[0].len(), x, |b| load_q4_0x4(group.get_unchecked(b))) }
}

/// One repacked group of four Q5_0 rows × four tokens. See [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], `group` with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q5_0x4<const P: usize>(group: &[BlockQ5_0x4], x: [&[PairQ8]; P]) -> Tiles<P> {
    // SAFETY: forwarded from the caller (b < nb = group.len()).
    unsafe { mmla4::<3, P>(x[0].len(), x, |b| load_q5_0x4(group.get_unchecked(b))) }
}

/// Sub-block `j` (values 32j..32j + 32) of a Q4_K super-block as int8
/// (0..15), in two vectors of 16.
#[inline(always)]
pub fn q4_k_sub(b: &BlockQ4_K, j: usize) -> [int8x16_t; 2] {
    // SAFETY: 128 bytes of nibbles; j < 8 keeps 32·(j / 2) + 32 <= 128.
    unsafe {
        let q = b.qs.as_ptr().add(32 * (j / 2));
        let f = |v: uint8x16_t| vreinterpretq_s8_u8(if j.is_multiple_of(2) { vandq_u8(v, vdupq_n_u8(0x0F)) } else { vshrq_n_u8::<4>(v) });
        [f(vld1q_u8(q)), f(vld1q_u8(q.add(16)))]
    }
}

/// Sub-block `j` of a Q6_K super-block as int8 q − 32, in two vectors of 16.
#[inline(always)]
pub fn q6_k_sub(b: &BlockQ6_K, j: usize) -> [int8x16_t; 2] {
    let (n, k) = (j / 4, j % 4);
    // SAFETY: ql holds 128 bytes and qh 64; j < 8 keeps every 16-byte load
    // inside them (as in `q6_k_block`).
    unsafe {
        let (ql, qh) = (b.ql.as_ptr().add(64 * n + 32 * (k % 2)), b.qh.as_ptr().add(32 * n));
        let f = |i: usize| {
            let v = vld1q_u8(ql.add(16 * i));
            let lo = if k < 2 { vandq_u8(v, vdupq_n_u8(0x0F)) } else { vshrq_n_u8::<4>(v) };
            let hi = vshlq_n_u8::<4>(vandq_u8(vshlq_u8(vld1q_u8(qh.add(16 * i)), vdupq_n_s8(-2 * k as i8)), vdupq_n_u8(3)));
            vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo, hi)), vdupq_n_s8(32))
        };
        [f(0), f(1)]
    }
}

/// One 32-value block of four rows of a K-quant matrix, unpacked once per
/// matmul for [`mmla4_k`]: the values as int8 in [`BlockQ8_0x4`]'s layout
/// (Q4_K: q in 0..=15; Q6_K: q − 32), each row's f32 scale for values
/// 0..16 (`sa`) and 16..32 (`sb`) (one Q4_K sub-block, two Q6_K ones), and
/// each row's coefficient on the activation's Σx (`bias`: Q4_K's −dmin·m,
/// 0 for Q6_K).
#[derive(Clone, Copy)]
pub struct KBlock4 {
    pub qs: [i8; 4 * QK],
    pub sa: [f32; 4],
    pub sb: [f32; 4],
    pub bias: [f32; 4],
}

/// Four K-quant rows (unpacked, `g`) × the tokens of pairs `x`, with
/// `smmla`; the result is laid out as in [`mmla4`]. Per block
/// each tile has two chains of two `smmla`s, one per 16-value half, each
/// scaled by its half's row scales, plus the bias times Σx.
///
/// # Safety
/// The CPU must have i8mm; both pair slices must have `g.len()` entries.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_k<const P: usize>(g: &[KBlock4], x: [&[PairQ8]; P]) -> Tiles<P> {
    // SAFETY: register-only NEON ops (AArch64 baseline).
    let (mut acc, z) = unsafe { ([[vdupq_n_f32(0.0); P]; 2], vdupq_n_s32(0)) };
    for (b, w) in g.iter().enumerate() {
        // SAFETY: each block holds 128 int8s and 3 × 4 floats; the caller
        // guarantees both pair slices have g.len() entries.
        unsafe {
            let q = w.qs.as_ptr();
            let v = |o: usize| vld1q_s8(q.add(o));
            let wv = [[v(0), v(32), v(64), v(96)], [v(16), v(48), v(80), v(112)]];
            // [r0, r1, r2, r3] → [r0, r0, r1, r1] and [r2, r2, r3, r3].
            let dup = |a: &[f32; 4]| {
                let f = vld1q_f32(a.as_ptr());
                [vzip1q_f32(f, f), vzip2q_f32(f, f)]
            };
            let (sa, sb, bias) = (dup(&w.sa), dup(&w.sb), dup(&w.bias));
            for tp in 0..P {
                let p = x[tp].get_unchecked(b);
                let xq = p.qs.as_ptr();
                let xv = [vld1q_s8(xq), vld1q_s8(xq.add(16)), vld1q_s8(xq.add(32)), vld1q_s8(xq.add(48))];
                let (dx, sx) = (vld1q_f32(p.d.as_ptr()), vld1q_f32(p.s.as_ptr()));
                for rp in 0..2 {
                    let lo = smmla(smmla(z, wv[rp][0], xv[0]), wv[rp][1], xv[1]);
                    let hi = smmla(smmla(z, wv[rp][2], xv[2]), wv[rp][3], xv[3]);
                    let a = &mut acc[rp][tp];
                    *a = vfmaq_f32(*a, vcvtq_f32_s32(lo), vmulq_f32(sa[rp], dx));
                    *a = vfmaq_f32(*a, vcvtq_f32_s32(hi), vmulq_f32(sb[rp], dx));
                    *a = vfmaq_f32(*a, bias[rp], sx);
                }
            }
        }
    }
    store_tiles(acc)
}

/// One repacked group of four rows · one Q8_0 token (decode), with `sdot`:
/// per block 8 `sdot`s, as llama.cpp's `gemv_q4_0_4x8_q8_0`. Each token
/// chunk is broadcast to both halves of a register, so `sdot` against a row
/// pair gives `[a lo, a hi, b lo, b hi]` partial sums; a pairwise add of the
/// two row pairs' results gives the four rows.
///
/// # Safety
/// The CPU must have `dotprod`; `load` must be valid for `0..x.len()`, and
/// `xd` as long as `x`.
#[target_feature(enable = "dotprod")]
unsafe fn gemv4<const F: i32>(x: &[BlockQ8_0], xd: &[f32], load: impl Fn(usize) -> Rows4) -> [f32; 4] {
    let (mut acc, z) = ([vdupq_n_f32(0.0); 2], vdupq_n_s32(0));
    // One block into `acc`.
    let block = |b: usize, acc: &mut float32x4_t| {
        let (w, d) = load(b);
        // SAFETY: b < x.len() = xd.len(); each block holds 32 int8s.
        unsafe {
            let xb = x.get_unchecked(b);
            let (x01, x23) = (vreinterpretq_s64_s8(vld1q_s8(xb.qs.as_ptr())), vreinterpretq_s64_s8(vld1q_s8(xb.qs.as_ptr().add(16))));
            let a = [
                vreinterpretq_s8_s64(vdupq_laneq_s64::<0>(x01)),
                vreinterpretq_s8_s64(vdupq_laneq_s64::<1>(x01)),
                vreinterpretq_s8_s64(vdupq_laneq_s64::<0>(x23)),
                vreinterpretq_s8_s64(vdupq_laneq_s64::<1>(x23)),
            ];
            let r = w.map(|w| vdotq_s32(vdotq_s32(vdotq_s32(vdotq_s32(z, w[0], a[0]), w[1], a[1]), w[2], a[2]), w[3], a[3]));
            let s = vpaddq_s32(r[0], r[1]);
            let scales = vmulq_n_f32(f16x4_to_f32(d), *xd.get_unchecked(b));
            *acc = vfmaq_f32(*acc, to_f32::<F>(s), scales);
        }
    };
    // Even and odd blocks into separate accumulators, so consecutive FMAs
    // do not wait on each other; two blocks per iteration, so both stay in
    // registers (indexed by `b & 1` they went through the stack).
    let mut b = 0;
    while b + 2 <= x.len() {
        let [a0, a1] = &mut acc;
        block(b, a0);
        block(b + 1, a1);
        b += 2;
    }
    if b < x.len() {
        block(b, &mut acc[0]);
    }
    let mut out = [0.0; 4];
    // SAFETY: `out` holds 4 floats.
    unsafe { vst1q_f32(out.as_mut_ptr(), vaddq_f32(acc[0], acc[1])) };
    out
}

/// One repacked group of four Q8_0 rows · one Q8_0 token; see [`gemv4`].
///
/// # Safety
/// The CPU must have `dotprod`; `group`, `x` and `xd` the same length.
#[target_feature(enable = "dotprod")]
pub unsafe fn gemv4_q8_0x4(group: &[BlockQ8_0x4], x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    // SAFETY: forwarded from the caller (b < x.len() = group.len()).
    unsafe { gemv4::<0>(x, xd, |b| load_q8_0x4(group.get_unchecked(b))) }
}

/// One repacked group of four Q4_0 rows · one Q8_0 token; see [`gemv4`].
///
/// # Safety
/// The CPU must have `dotprod`; `group`, `x` and `xd` the same length.
#[target_feature(enable = "dotprod")]
pub unsafe fn gemv4_q4_0x4(group: &[BlockQ4_0x4], x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    // SAFETY: forwarded from the caller (b < x.len() = group.len()).
    unsafe { gemv4::<4>(x, xd, |b| load_q4_0x4(group.get_unchecked(b))) }
}

/// One repacked group of four Q5_0 rows · one Q8_0 token; see [`gemv4`].
///
/// # Safety
/// The CPU must have `dotprod`; `group`, `x` and `xd` the same length.
#[target_feature(enable = "dotprod")]
pub unsafe fn gemv4_q5_0x4(group: &[BlockQ5_0x4], x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    // SAFETY: forwarded from the caller (b < x.len() = group.len()).
    unsafe { gemv4::<3>(x, xd, |b| load_q5_0x4(group.get_unchecked(b))) }
}
