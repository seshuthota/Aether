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

use crate::quant::{BlockQ4_0, BlockQ4_0x4, BlockQ8_0, BlockQ8_0x4, QK, f16_to_f32};
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
/// `[d0, d1, d0, d1]` as f32 to match the lanes of an `smmla` result.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct PairQ8 {
    pub qs: [i8; 2 * QK],
    pub d: [f32; 4],
}

/// Q8_0 activations of `t` tokens (`xq` is `[t, nb]` blocks, `xd` their
/// scales as f32) → `[ceil(t/2), nb]` [`PairQ8`]s: tokens 2p and 2p + 1 in
/// pair p, an odd last token paired with itself. Done once per matmul, so
/// the kernel never interleaves activations itself.
pub fn interleave_pairs(xq: &[BlockQ8_0], xd: &[f32], t: usize) -> Vec<PairQ8> {
    let nb = xq.len() / t;
    let mut out = Vec::with_capacity(t.div_ceil(2) * nb);
    for p in 0..t.div_ceil(2) {
        let (t0, t1) = (2 * p, (2 * p + 1).min(t - 1));
        for b in 0..nb {
            let (x0, x1) = (&xq[t0 * nb + b].qs, &xq[t1 * nb + b].qs);
            let mut qs = [0; 2 * QK];
            for k in 0..QK / 8 {
                qs[16 * k..16 * k + 8].copy_from_slice(&x0[8 * k..8 * k + 8]);
                qs[16 * k + 8..16 * k + 16].copy_from_slice(&x1[8 * k..8 * k + 8]);
            }
            let (d0, d1) = (xd[t0 * nb + b], xd[t1 * nb + b]);
            out.push(PairQ8 { qs, d: [d0, d1, d0, d1] });
        }
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

/// int32 sums → f32; `X16` when the weights came in ×16 (repacked Q4_0),
/// undone exactly by a fixed-point conversion with 4 fraction bits.
#[inline(always)]
fn to_f32<const X16: bool>(s: int32x4_t) -> float32x4_t {
    // SAFETY: register-only NEON ops.
    unsafe { if X16 { vcvtq_n_f32_s32::<4>(s) } else { vcvtq_f32_s32(s) } }
}

/// Four weight rows · the four tokens of pairs `x[0]`, `x[1]`, over `nb`
/// blocks, where `load(b)` gives block column `b` of the rows as
/// [`Rows4`]. Per block: 16 `smmla`s for 16 outputs.
///
/// Returns four 2×2 tiles, tile `2·rp + tp` for row pair `rp` and token pair
/// `tp`, each `[r0·t0, r0·t1, r1·t0, r1·t1]`. Every output has its own lane,
/// integer sums and float additions, so its value does not depend on the
/// rows or tokens it shares a tile with (passing the same row or pair twice
/// is how callers handle the edges), nor on whether the rows were repacked.
///
/// # Safety
/// The CPU must have i8mm; `load` must be valid for `0..nb` and both pair
/// slices must have `nb` entries.
#[target_feature(enable = "i8mm")]
unsafe fn mmla4<const X16: bool>(nb: usize, x: [&[PairQ8]; 2], load: impl Fn(usize) -> Rows4) -> [[f32; 4]; 4] {
    // SAFETY: register-only NEON ops (AArch64 baseline).
    let (mut acc, z) = unsafe { ([vdupq_n_f32(0.0); 4], vdupq_n_s32(0)) };
    for b in 0..nb {
        let (w, d) = load(b);
        // SAFETY: the caller guarantees both pair slices have nb entries;
        // each PairQ8 holds 64 int8s and 4 floats.
        unsafe {
            // [d0, d1, d2, d3] → [d0, d0, d1, d1] and [d2, d2, d3, d3].
            let dw = f16x4_to_f32(d);
            let dw = [vzip1q_f32(dw, dw), vzip2q_f32(dw, dw)];
            for tp in 0..2 {
                let p = x[tp].get_unchecked(b);
                let q = p.qs.as_ptr();
                let xv = [vld1q_s8(q), vld1q_s8(q.add(16)), vld1q_s8(q.add(32)), vld1q_s8(q.add(48))];
                let dx = vld1q_f32(p.d.as_ptr());
                for rp in 0..2 {
                    // One chain of four per tile: the four tiles (and the
                    // next block's) are independent, which is enough to keep
                    // the `smmla` units busy without extra adds.
                    let s = smmla(smmla(smmla(smmla(z, w[rp][0], xv[0]), w[rp][1], xv[1]), w[rp][2], xv[2]), w[rp][3], xv[3]);
                    acc[2 * rp + tp] = vfmaq_f32(acc[2 * rp + tp], to_f32::<X16>(s), vmulq_f32(dw[rp], dx));
                }
            }
        }
    }
    let mut out = [[0.0; 4]; 4];
    for (o, a) in out.iter_mut().zip(acc) {
        // SAFETY: each `o` holds 4 floats.
        unsafe { vst1q_f32(o.as_mut_ptr(), a) };
    }
    out
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

/// Four row-major Q8_0 rows × four tokens with `smmla`; see [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], every row with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q8_0(rows: [&[BlockQ8_0]; 4], x: [&[PairQ8]; 2]) -> [[f32; 4]; 4] {
    // SAFETY: forwarded from the caller (b < nb = every row's length).
    unsafe {
        mmla4::<false>(x[0].len(), x, |b| {
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
pub unsafe fn mmla4_q4_0(rows: [&[BlockQ4_0]; 4], x: [&[PairQ8]; 2]) -> [[f32; 4]; 4] {
    // SAFETY: forwarded from the caller (b < nb = every row's length);
    // 16 bytes of nibbles per block.
    unsafe {
        mmla4::<false>(x[0].len(), x, |b| {
            let r = rows.map(|r| r.get_unchecked(b));
            interleave4(r.map(|r| <[int8x16_t; 2]>::from(unpack_q4(vld1q_u8(r.qs.as_ptr())))), r.map(|r| r.d))
        })
    }
}

/// One repacked group of four Q8_0 rows × four tokens: no interleaving in
/// the loop. See [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], `group` with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q8_0x4(group: &[BlockQ8_0x4], x: [&[PairQ8]; 2]) -> [[f32; 4]; 4] {
    // SAFETY: forwarded from the caller (b < nb = group.len()).
    unsafe { mmla4::<false>(x[0].len(), x, |b| load_q8_0x4(group.get_unchecked(b))) }
}

/// One repacked group of four Q4_0 rows × four tokens. See [`mmla4`].
///
/// # Safety
/// As for [`mmla4`], `group` with `x[0].len()` blocks.
#[target_feature(enable = "i8mm")]
pub unsafe fn mmla4_q4_0x4(group: &[BlockQ4_0x4], x: [&[PairQ8]; 2]) -> [[f32; 4]; 4] {
    // SAFETY: forwarded from the caller (b < nb = group.len()).
    unsafe { mmla4::<true>(x[0].len(), x, |b| load_q4_0x4(group.get_unchecked(b))) }
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
unsafe fn gemv4<const X16: bool>(x: &[BlockQ8_0], xd: &[f32], load: impl Fn(usize) -> Rows4) -> [f32; 4] {
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
            acc[b & 1] = vfmaq_f32(acc[b & 1], to_f32::<X16>(s), scales);
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
    unsafe { gemv4::<false>(x, xd, |b| load_q8_0x4(group.get_unchecked(b))) }
}

/// One repacked group of four Q4_0 rows · one Q8_0 token; see [`gemv4`].
///
/// # Safety
/// The CPU must have `dotprod`; `group`, `x` and `xd` the same length.
#[target_feature(enable = "dotprod")]
pub unsafe fn gemv4_q4_0x4(group: &[BlockQ4_0x4], x: &[BlockQ8_0], xd: &[f32]) -> [f32; 4] {
    // SAFETY: forwarded from the caller (b < x.len() = group.len()).
    unsafe { gemv4::<true>(x, xd, |b| load_q4_0x4(group.get_unchecked(b))) }
}
