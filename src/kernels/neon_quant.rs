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

use crate::quant::{BlockQ4_0, BlockQ4_0x4, BlockQ4_K, BlockQ5_0, BlockQ5_0x4, BlockQ6_K, BlockQ8_0, BlockQ8_0x4, QK, QK_K, f16_to_f32, q4_k_scale_min};
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

// --- K-quants (`sdot`) ---------------------------------------------------
//
// A 256-weight super-block meets 8 Q8_0 activation blocks. The activation
// side ([`XK`]) is loaded once per super-block and shared by the rows of a
// `dot4`; each sub-block's integer dot is scaled in f32 by `d·sc·d_x`.

/// One activation super-block: 8 Q8_0 blocks as int8 vectors (values 0..16,
/// 16..32 of each), their scales, and each block's Σ q · d_x (Q4_K's mins
/// multiply it).
struct XK {
    v: [[int8x16_t; 2]; 8],
    d: [f32; 8],
    s: [f32; 8],
}

#[inline(always)]
fn load_xk(x: &[BlockQ8_0; 8], xd: &[f32; 8]) -> XK {
    // SAFETY: each block holds 32 int8s.
    unsafe {
        let v: [[int8x16_t; 2]; 8] = std::array::from_fn(|j| {
            let q = x[j].qs.as_ptr();
            [vld1q_s8(q), vld1q_s8(q.add(16))]
        });
        let d = *xd;
        let s = std::array::from_fn(|j| (vaddlvq_s8(v[j][0]) as i32 + vaddlvq_s8(v[j][1]) as i32) as f32 * d[j]);
        XK { v, d, s }
    }
}

/// One Q4_K super-block · `x`: adds Σ d·sc·d_x·(q·x) into `acc` and
/// returns the min term dmin · Σ m · Σx·d_x, to subtract.
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn q4_k_block(w: &BlockQ4_K, x: &XK, acc: &mut float32x4_t) -> f32 {
    let (d, dmin) = (f16_to_f32(w.d), f16_to_f32(w.dmin));
    let mut mins = 0.0;
    for c in 0..4 {
        // SAFETY: 128 bytes of nibbles; c < 4, so offsets stay below 128.
        unsafe {
            let (a, b) = (vld1q_u8(w.qs.as_ptr().add(32 * c)), vld1q_u8(w.qs.as_ptr().add(32 * c + 16)));
            let m4 = vdupq_n_u8(0x0F);
            let halves = [[vandq_u8(a, m4), vandq_u8(b, m4)], [vshrq_n_u8(a, 4), vshrq_n_u8(b, 4)]];
            for (h, q) in halves.iter().enumerate() {
                let j = 2 * c + h;
                let (sc, m) = q4_k_scale_min(&w.scales, j);
                let z = vdupq_n_s32(0);
                let s = vdotq_s32(vdotq_s32(z, vreinterpretq_s8_u8(q[0]), x.v[j][0]), vreinterpretq_s8_u8(q[1]), x.v[j][1]);
                *acc = vfmaq_n_f32(*acc, vcvtq_f32_s32(s), d * sc as f32 * x.d[j]);
                mins += m as f32 * x.s[j];
            }
        }
    }
    dmin * mins
}

/// One Q6_K super-block · `x`, added into `acc`.
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn q6_k_block(w: &BlockQ6_K, x: &XK, acc: &mut float32x4_t) {
    let d = f16_to_f32(w.d);
    for n in 0..2 {
        // SAFETY: ql holds 128 bytes and qh 64; n < 2 keeps every 16-byte
        // load inside them (64n + 48 + 16 <= 128, 32n + 16 + 16 <= 64).
        unsafe {
            let ql = w.ql.as_ptr().add(64 * n);
            let qh = [vld1q_u8(w.qh.as_ptr().add(32 * n)), vld1q_u8(w.qh.as_ptr().add(32 * n + 16))];
            let (m4, m3, off) = (vdupq_n_u8(0x0F), vdupq_n_u8(3), vdupq_n_s8(32));
            for k in 0..4 {
                let j = 4 * n + k;
                let lo = |i: usize| {
                    let v = vld1q_u8(ql.add(32 * (k % 2) + 16 * i));
                    if k < 2 { vandq_u8(v, m4) } else { vshrq_n_u8(v, 4) }
                };
                let hi = |i: usize| vshlq_n_u8(vandq_u8(vshlq_u8(qh[i], vdupq_n_s8(-2 * k as i8)), m3), 4);
                let q = |i: usize| vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(lo(i), hi(i))), off);
                let z = vdupq_n_s32(0);
                let s = vmulq_n_s32(vdotq_s32(z, q(0), x.v[j][0]), w.scales[2 * j] as i32);
                let s = vmlaq_n_s32(s, vdotq_s32(z, q(1), x.v[j][1]), w.scales[2 * j + 1] as i32);
                *acc = vfmaq_n_f32(*acc, vcvtq_f32_s32(s), d * x.d[j]);
            }
        }
    }
}

/// The activation's block scales as f32 (the single-row dots get only the
/// blocks).
fn scales_f32(x: &[BlockQ8_0]) -> Vec<f32> {
    x.iter().map(|b| f16_to_f32(b.d)).collect()
}

/// (b) Q4_K row · Q8_0 x with `sdot`.
pub fn dot_q4_k_q8_0(row: &[BlockQ4_K], x: &[BlockQ8_0]) -> f32 {
    assert_eq!(row.len() * (QK_K / QK), x.len());
    if !has_dotprod() {
        return crate::quant::scalar::dot_q4_k_q8_0(row, x);
    }
    let xd = scales_f32(x);
    // SAFETY: dotprod is present (checked above); 8 activation blocks per
    // super-block (checked above).
    unsafe { dot4_q4_k_q8_0([row; 4], x, &xd)[0] }
}

/// (b) Q6_K row · Q8_0 x with `sdot`.
pub fn dot_q6_k_q8_0(row: &[BlockQ6_K], x: &[BlockQ8_0]) -> f32 {
    assert_eq!(row.len() * (QK_K / QK), x.len());
    if !has_dotprod() {
        return crate::quant::scalar::dot_q6_k_q8_0(row, x);
    }
    let xd = scales_f32(x);
    // SAFETY: as in dot_q4_k_q8_0.
    unsafe { dot4_q6_k_q8_0([row; 4], x, &xd)[0] }
}

/// Four Q4_K rows · one Q8_0 activation.
///
/// # Safety
/// The CPU must have `dotprod`; every row must have `x.len() / 8` blocks,
/// `xd` as many entries as `x`.
#[target_feature(enable = "dotprod")]
pub unsafe fn dot4_q4_k_q8_0(rows: [&[BlockQ4_K]; 4], x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    let (mut acc, mut mins) = ([vdupq_n_f32(0.0); 4], [0.0f32; 4]);
    for (b, (xs, ds)) in x.as_chunks::<{ QK_K / QK }>().0.iter().zip(xd.as_chunks::<{ QK_K / QK }>().0).enumerate() {
        // SAFETY: 8 blocks per chunk; every row has x.len() / 8 blocks.
        unsafe {
            let xk = load_xk(xs, ds);
            for r in 0..4 {
                mins[r] += q4_k_block(rows[r].get_unchecked(b), &xk, &mut acc[r]);
            }
        }
    }
    std::array::from_fn(|r| vaddvq_f32(acc[r]) - mins[r])
}

/// Four Q6_K rows · one Q8_0 activation.
///
/// # Safety
/// As for [`dot4_q4_k_q8_0`].
#[target_feature(enable = "dotprod")]
pub unsafe fn dot4_q6_k_q8_0(rows: [&[BlockQ6_K]; 4], x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    let mut acc = [vdupq_n_f32(0.0); 4];
    for (b, (xs, ds)) in x.as_chunks::<{ QK_K / QK }>().0.iter().zip(xd.as_chunks::<{ QK_K / QK }>().0).enumerate() {
        // SAFETY: 8 blocks per chunk; every row has x.len() / 8 blocks.
        unsafe {
            let xk = load_xk(xs, ds);
            for r in 0..4 {
                q6_k_block(rows[r].get_unchecked(b), &xk, &mut acc[r]);
            }
        }
    }
    acc.map(|a| vaddvq_f32(a))
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

/// Token pairs per `smmla` tile: a tile is 4 rows × 2·NP tokens.
pub const NP: usize = 4;

/// The result of a tile: `[row pair][token pair]` → `[r0·t0, r0·t1, r1·t0,
/// r1·t1]`.
pub type Tiles = [[[f32; 4]; NP]; 2];

/// Accumulators `[NP·rp + tp]` → [`Tiles`].
#[inline(always)]
fn store_tiles(acc: [float32x4_t; 2 * NP]) -> Tiles {
    let mut out = [[[0.0; 4]; NP]; 2];
    for (o, a) in out.iter_mut().flatten().zip(&acc) {
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
unsafe fn mmla4<const F: i32>(nb: usize, x: [&[PairQ8]; NP], load: impl Fn(usize) -> Rows4) -> Tiles {
    // SAFETY: register-only NEON ops (AArch64 baseline).
    let (mut acc, z) = unsafe { ([vdupq_n_f32(0.0); 2 * NP], vdupq_n_s32(0)) };
    for b in 0..nb {
        let (w, d) = load(b);
        // SAFETY: the caller guarantees every pair slice has nb entries;
        // each PairQ8 holds 64 int8s and 4 floats.
        unsafe {
            // [d0, d1, d2, d3] → [d0, d0, d1, d1] and [d2, d2, d3, d3].
            let dw = f16x4_to_f32(d);
            let dw = [vzip1q_f32(dw, dw), vzip2q_f32(dw, dw)];
            for tp in 0..NP {
                let p = x[tp].get_unchecked(b);
                let q = p.qs.as_ptr();
                let xv = [vld1q_s8(q), vld1q_s8(q.add(16)), vld1q_s8(q.add(32)), vld1q_s8(q.add(48))];
                let dx = vld1q_f32(p.d.as_ptr());
                for rp in 0..2 {
                    // One chain of four per tile: the tiles (and the next
                    // block's) are independent, which is enough to keep
                    // the `smmla` units busy without extra adds.
                    let s = smmla(smmla(smmla(smmla(z, w[rp][0], xv[0]), w[rp][1], xv[1]), w[rp][2], xv[2]), w[rp][3], xv[3]);
                    acc[NP * rp + tp] = vfmaq_f32(acc[NP * rp + tp], to_f32::<F>(s), vmulq_f32(dw[rp], dx));
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

/// Repacked Q5_0 (see [`BlockQ5_0x4`]): each vector is its nibbles moved to
/// bits 3..7 and its inverted fifth bit inserted as bit 7, i.e. 8·(q − 16).
#[inline(always)]
fn load_q5_0x4(w: &BlockQ5_0x4) -> Rows4 {
    let q = w.qs.as_ptr();
    // SAFETY: 64 bytes of nibbles and 16 of fifth bits per block;
    // register-only ops after the loads.
    unsafe {
        let h = vld1q_u8(w.qh.as_ptr());
        let (m78, m80) = (vdupq_n_u8(0x78), vdupq_n_u8(0x80));
        let lo = |v: uint8x16_t| vandq_u8(vshlq_n_u8::<3>(v), m78);
        let hi = |v: uint8x16_t| vandq_u8(vshrq_n_u8::<1>(v), m78);
        // Vector i: bit i of h moved to bit 7 and inserted.
        let fifth = |i: i8, x: uint8x16_t| vreinterpretq_s8_u8(vbslq_u8(m80, vshlq_u8(h, vdupq_n_s8(7 - i)), x));
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
pub unsafe fn mmla4_q8_0(rows: [&[BlockQ8_0]; 4], x: [&[PairQ8]; NP]) -> Tiles {
    // SAFETY: forwarded from the caller (b < nb = every row's length).
    unsafe {
        mmla4::<0>(x[0].len(), x, |b| {
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
pub unsafe fn mmla4_q4_0(rows: [&[BlockQ4_0]; 4], x: [&[PairQ8]; NP]) -> Tiles {
    // SAFETY: forwarded from the caller (b < nb = every row's length);
    // 16 bytes of nibbles per block.
    unsafe {
        mmla4::<0>(x[0].len(), x, |b| {
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
pub unsafe fn mmla4_q5_0(rows: [&[BlockQ5_0]; 4], x: [&[PairQ8]; NP]) -> Tiles {
    // SAFETY: forwarded from the caller (b < nb = every row's length).
    unsafe {
        mmla4::<0>(x[0].len(), x, |b| {
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
pub unsafe fn mmla4_q8_0x4(group: &[BlockQ8_0x4], x: [&[PairQ8]; NP]) -> Tiles {
    // SAFETY: forwarded from the caller (b < nb = group.len()).
    unsafe { mmla4::<0>(x[0].len(), x, |b| load_q8_0x4(group.get_unchecked(b))) }
}

/// One repacked group of four Q4_0 rows × four tokens. See [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], `group` with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q4_0x4(group: &[BlockQ4_0x4], x: [&[PairQ8]; NP]) -> Tiles {
    // SAFETY: forwarded from the caller (b < nb = group.len()).
    unsafe { mmla4::<4>(x[0].len(), x, |b| load_q4_0x4(group.get_unchecked(b))) }
}

/// One repacked group of four Q5_0 rows × four tokens. See [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], `group` with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q5_0x4(group: &[BlockQ5_0x4], x: [&[PairQ8]; NP]) -> Tiles {
    // SAFETY: forwarded from the caller (b < nb = group.len()).
    unsafe { mmla4::<3>(x[0].len(), x, |b| load_q5_0x4(group.get_unchecked(b))) }
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
pub unsafe fn mmla4_k(g: &[KBlock4], x: [&[PairQ8]; NP]) -> Tiles {
    // SAFETY: register-only NEON ops (AArch64 baseline).
    let (mut acc, z) = unsafe { ([vdupq_n_f32(0.0); 2 * NP], vdupq_n_s32(0)) };
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
            for tp in 0..NP {
                let p = x[tp].get_unchecked(b);
                let xq = p.qs.as_ptr();
                let xv = [vld1q_s8(xq), vld1q_s8(xq.add(16)), vld1q_s8(xq.add(32)), vld1q_s8(xq.add(48))];
                let (dx, sx) = (vld1q_f32(p.d.as_ptr()), vld1q_f32(p.s.as_ptr()));
                for rp in 0..2 {
                    let lo = smmla(smmla(z, wv[rp][0], xv[0]), wv[rp][1], xv[1]);
                    let hi = smmla(smmla(z, wv[rp][2], xv[2]), wv[rp][3], xv[3]);
                    let a = &mut acc[NP * rp + tp];
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
    for b in 0..x.len() {
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
            // Even and odd blocks into separate accumulators, so
            // consecutive FMAs do not wait on each other.
            acc[b & 1] = vfmaq_f32(acc[b & 1], to_f32::<F>(s), scales);
        }
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
