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

use crate::quant::{BlockQ4_0, BlockQ8_0, QK, f16_to_f32};
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

/// (a) Q8_0 row · f32 x.
pub fn dot_q8_0_f32(row: &[BlockQ8_0], x: &[f32]) -> f32 {
    assert_eq!(x.len(), row.len() * QK);
    // SAFETY: register-only NEON op (AArch64 baseline).
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    for (blk, xs) in row.iter().zip(x.chunks_exact(QK)) {
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
    for (blk, xs) in row.iter().zip(x.chunks_exact(QK)) {
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
