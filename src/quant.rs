//! Block quantization, byte-compatible with llama.cpp / GGUF: Q8_0, Q4_0,
//! Q5_0, and the K-quants Q4_K and Q6_K (read from GGUF files only).
//!
//! Decode reads every weight once per token, so its speed is set by bytes
//! per weight. f32 is 4 bytes; Q8_0 is 34 bytes per 32 weights (8.5 bits
//! each), Q4_0 18 bytes per 32 (4.5 bits). Each block of 32 consecutive
//! weights of a row shares one scale `d`, stored as f16:
//!
//! - **Q8_0**: `d = max|x| / 127`, `q = round(x / d)` in −127..=127,
//!   value `= q · d`.
//! - **Q4_0**: `d = m / −8` where `m` is the element with the largest
//!   magnitude (its sign kept), `q = trunc(x / d + 8.5)` clamped to 0..=15,
//!   value `= (q − 8) · d`. Element `j` of the block sits in the low nibble of
//!   byte `j`, element `j + 16` in the high nibble.
//! - **Q5_0**: Q4_0 with one more bit: `d = m / −16`, `q = trunc(x / d +
//!   16.5)` clamped to 0..=31, value `= (q − 16) · d`. The low four bits are
//!   stored as in Q4_0; the fifth bit of element `j` is bit `j` of the
//!   32-bit `qh`. 22 bytes per 32 weights (5.5 bits each). Small models'
//!   "Q4_K_M" files are mostly Q5_0: K-quants need rows a multiple of 256
//!   long, and llama.cpp falls back to Q5_0 for the others (576, 896, …).
//!
//! These follow `quantize_row_q8_0_ref` / `quantize_row_q4_0_ref` /
//! `quantize_row_q5_0_ref` in
//! llama.cpp's ggml-quants.c bit for bit (checked against the `gguf`
//! package's reference implementation), so llama.cpp can be the accuracy
//! reference and GGUF files can be read without conversion.
//!
//! Two ways to multiply a quantized row by an f32 vector `x`:
//!
//! - **(a) dequantize**: turn each weight back into f32 and FMA with `x`.
//! - **(b) quantize the activation**: turn `x` into Q8_0 blocks once per
//!   matmul, then each block pair is an integer dot product of 32 int8s
//!   (one `sdot` instruction per 16), scaled once by `d_w · d_x`. Fewer
//!   instructions per weight, at the cost of rounding `x` to 8 bits — what
//!   llama.cpp does.

/// Weights per block, for both formats.
pub const QK: usize = 32;

/// 32 weights as int8 plus one f16 scale: 34 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8_0 {
    /// Scale as IEEE f16 bits.
    pub d: u16,
    pub qs: [i8; QK],
}

/// 32 weights as 4-bit unsigned (offset 8) plus one f16 scale: 18 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ4_0 {
    pub d: u16,
    /// Byte j: element j in the low nibble, element j + 16 in the high one.
    pub qs: [u8; QK / 2],
}

/// 32 weights as 5-bit unsigned (offset 16) plus one f16 scale: 22 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ5_0 {
    pub d: u16,
    /// Bit j (of the little-endian u32): the fifth bit of element j.
    pub qh: [u8; 4],
    /// The low four bits, laid out as in [`BlockQ4_0`].
    pub qs: [u8; QK / 2],
}

// The layouts must match GGUF's exactly (no padding).
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

pub use crate::format::safetensors::f16_to_f32;

/// f32 → IEEE f16 bits, rounding to nearest, ties to even (as numpy's
/// `astype(float16)` and the ARM / x86 conversion instructions do).
/// Out-of-range values become ±infinity; NaN stays NaN.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15; // f16 biased exponent
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    // Normal: keep the top 10 mantissa bits; subnormal (e <= 0): shift the
    // full 24-bit significand down into f16's 2^-24 units. Either way round
    // the dropped bits to nearest, ties to even.
    let (kept, dropped_bits) = if e > 0 { (((e as u32) << 10) | (man >> 13), 13) } else { (0, (14 - e) as u32) };
    let (h, rem, half) = if e > 0 {
        (kept, man & 0x1FFF, 0x1000)
    } else if dropped_bits > 24 {
        return sign; // below half the smallest subnormal: rounds to zero
    } else {
        let m = man | 0x80_0000;
        (m >> dropped_bits, m & ((1 << dropped_bits) - 1), 1 << (dropped_bits - 1))
    };
    // A carry out of the mantissa correctly bumps the exponent (or reaches
    // infinity at the top).
    let h = if rem > half || (rem == half && h & 1 == 1) { h + 1 } else { h };
    sign | h as u16
}

/// Quantizes `x` (a multiple of 32 long) into Q8_0 blocks.
pub fn quantize_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    assert_eq!(x.len(), out.len() * QK, "length must be blocks × 32");
    for (blk, v) in out.iter_mut().zip(x.as_chunks::<QK>().0) {
        let amax = v.iter().fold(0.0f32, |m, &a| m.max(a.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk.d = f32_to_f16(d);
        for (q, &a) in blk.qs.iter_mut().zip(v) {
            *q = (a * id).round() as i8; // f32::round = C roundf (halves away from 0)
        }
    }
}

/// Quantizes `x` (a multiple of 32 long) into Q4_0 blocks.
pub fn quantize_q4_0(x: &[f32], out: &mut [BlockQ4_0]) {
    assert_eq!(x.len(), out.len() * QK, "length must be blocks × 32");
    for (blk, v) in out.iter_mut().zip(x.as_chunks::<QK>().0) {
        // The element of largest magnitude, first one on ties, sign kept.
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &a in v {
            if amax < a.abs() {
                amax = a.abs();
                max = a;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk.d = f32_to_f16(d);
        let q = |a: f32| ((a * id + 8.5) as i8).min(15) as u8;
        for j in 0..QK / 2 {
            blk.qs[j] = q(v[j]) | (q(v[j + QK / 2]) << 4);
        }
    }
}

/// Quantizes `x` (a multiple of 32 long) into Q5_0 blocks.
pub fn quantize_q5_0(x: &[f32], out: &mut [BlockQ5_0]) {
    assert_eq!(x.len(), out.len() * QK, "length must be blocks × 32");
    for (blk, v) in out.iter_mut().zip(x.as_chunks::<QK>().0) {
        // As Q4_0: the element of largest magnitude, sign kept.
        let (mut amax, mut max) = (0.0f32, 0.0f32);
        for &a in v {
            if amax < a.abs() {
                amax = a.abs();
                max = a;
            }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk.d = f32_to_f16(d);
        let q = |a: f32| ((a * id + 16.5) as i8).min(31) as u8;
        let mut qh = 0u32;
        for j in 0..QK / 2 {
            let (lo, hi) = (q(v[j]), q(v[j + QK / 2]));
            blk.qs[j] = (lo & 0x0F) | ((hi & 0x0F) << 4);
            qh |= ((lo >> 4) as u32) << j | ((hi >> 4) as u32) << (j + QK / 2);
        }
        blk.qh = qh.to_le_bytes();
    }
}

/// Element `j` of a Q5_0 block as an integer in −16..=15.
#[inline(always)]
pub fn q5_0_value(blk: &BlockQ5_0, j: usize) -> i32 {
    let qh = u32::from_le_bytes(blk.qh);
    let lo = if j < QK / 2 { blk.qs[j] & 0x0F } else { blk.qs[j - QK / 2] >> 4 };
    (lo as i32 | (((qh >> j) & 1) as i32) << 4) - 16
}

pub fn dequantize_q8_0(blocks: &[BlockQ8_0], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK);
    for (blk, o) in blocks.iter().zip(out.as_chunks_mut::<QK>().0.iter_mut()) {
        let d = f16_to_f32(blk.d);
        for (v, &q) in o.iter_mut().zip(&blk.qs) {
            *v = q as f32 * d;
        }
    }
}

pub fn dequantize_q4_0(blocks: &[BlockQ4_0], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK);
    for (blk, o) in blocks.iter().zip(out.as_chunks_mut::<QK>().0.iter_mut()) {
        let d = f16_to_f32(blk.d);
        for j in 0..QK / 2 {
            o[j] = ((blk.qs[j] & 0x0F) as i32 - 8) as f32 * d;
            o[j + QK / 2] = ((blk.qs[j] >> 4) as i32 - 8) as f32 * d;
        }
    }
}

pub fn dequantize_q5_0(blocks: &[BlockQ5_0], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK);
    for (blk, o) in blocks.iter().zip(out.as_chunks_mut::<QK>().0.iter_mut()) {
        let d = f16_to_f32(blk.d);
        for (j, v) in o.iter_mut().enumerate() {
            *v = q5_0_value(blk, j) as f32 * d;
        }
    }
}

// --- K-quants: 256-weight super-blocks ----------------------------------
//
// A K-quant block covers 256 weights split into sub-blocks, each with its
// own small integer scale; the scales themselves are quantized against one
// f16 per super-block. That costs fewer bits per weight than one f16 per
// 32, so more bits are left for the weights.
//
// - **Q4_K** (144 bytes, 4.5 bits per weight): 8 sub-blocks of 32, each with
//   a 6-bit scale `sc` and a 6-bit min `m`; value `= d·sc·q − dmin·m` with
//   q in 0..=15.
// - **Q6_K** (210 bytes, 6.56 bits per weight): 16 sub-blocks of 16, each
//   with an int8 scale; value `= d·sc·(q − 32)` with q in 0..=63.
//
// Rows must be a multiple of 256 long. The dot products quantize the
// activation to Q8_0 (as for the other formats); a 32-weight Q8_0 block
// lines up with one Q4_K sub-block or two Q6_K ones. llama.cpp uses its
// own Q8_K (one f32 scale per 256) instead, so results differ from it by
// that rounding, not by any difference in the weights.

/// Weights per K-quant super-block.
pub const QK_K: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_camel_case_types)] // ggml's name
pub struct BlockQ4_K {
    /// Scale of the sub-block scales, f16 bits.
    pub d: u16,
    /// Scale of the sub-block mins, f16 bits.
    pub dmin: u16,
    /// 8 scales and 8 mins, 6 bits each (see [`q4_k_scale_min`]).
    pub scales: [u8; 12],
    /// Bytes 32c..32c+32: sub-block 2c in the low nibbles, 2c+1 in the high.
    pub qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_camel_case_types)] // ggml's name
pub struct BlockQ6_K {
    /// Low four bits (see [`q6_k_values`] for the order).
    pub ql: [u8; QK_K / 2],
    /// High two bits, four values per byte.
    pub qh: [u8; QK_K / 4],
    /// One scale per 16 weights.
    pub scales: [i8; QK_K / 16],
    pub d: u16,
}

const _: () = assert!(std::mem::size_of::<BlockQ4_K>() == 144);
const _: () = assert!(std::mem::size_of::<BlockQ6_K>() == 210);

/// Scale and min of Q4_K sub-block `j` (ggml's `get_scale_min_k4`): the
/// first four are the low 6 bits of bytes j and j+4; the last four take 4
/// bits from byte j+4 and their top 2 bits from the top of bytes j−4 and j.
#[inline(always)]
pub fn q4_k_scale_min(scales: &[u8; 12], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        ((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4), (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4))
    }
}

/// The 256 4-bit values of a Q4_K block, in element order.
pub fn q4_k_values(blk: &BlockQ4_K) -> [u8; QK_K] {
    let mut q = [0; QK_K];
    for (e, v) in q.iter_mut().enumerate() {
        let (c, half, l) = (e / 64, (e / 32) % 2, e % 32);
        *v = (blk.qs[32 * c + l] >> (4 * half)) & 0x0F;
    }
    q
}

/// The 256 values of a Q6_K block as integers in −32..=31, in element
/// order. Element e of half n = e / 128 (r = e % 128, k = r / 32, l = r %
/// 32): low bits from `ql[64n + l + 32·(k % 2)]` (low nibble for k < 2,
/// high for k ≥ 2), high bits from bits 2k..2k+2 of `qh[32n + l]`. Its
/// scale is `scales[e / 16]`.
pub fn q6_k_values(blk: &BlockQ6_K) -> [i8; QK_K] {
    let mut q = [0; QK_K];
    for (e, v) in q.iter_mut().enumerate() {
        let (n, r) = (e / 128, e % 128);
        let (k, l) = (r / 32, r % 32);
        let lo = (blk.ql[64 * n + l + 32 * (k % 2)] >> (4 * (k / 2))) & 0x0F;
        let hi = (blk.qh[32 * n + l] >> (2 * k)) & 3;
        *v = (lo | (hi << 4)) as i8 - 32;
    }
    q
}

pub fn dequantize_q4_k(blocks: &[BlockQ4_K], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK_K);
    for (blk, o) in blocks.iter().zip(out.as_chunks_mut::<QK_K>().0.iter_mut()) {
        let (d, dmin, q) = (f16_to_f32(blk.d), f16_to_f32(blk.dmin), q4_k_values(blk));
        for j in 0..QK_K / 32 {
            let (sc, m) = q4_k_scale_min(&blk.scales, j);
            let (d1, m1) = (d * sc as f32, dmin * m as f32);
            for l in 32 * j..32 * (j + 1) {
                o[l] = d1 * q[l] as f32 - m1;
            }
        }
    }
}

pub fn dequantize_q6_k(blocks: &[BlockQ6_K], out: &mut [f32]) {
    assert_eq!(out.len(), blocks.len() * QK_K);
    for (blk, o) in blocks.iter().zip(out.as_chunks_mut::<QK_K>().0.iter_mut()) {
        let (d, q) = (f16_to_f32(blk.d), q6_k_values(blk));
        for (e, v) in o.iter_mut().enumerate() {
            *v = d * blk.scales[e / 16] as f32 * q[e] as f32;
        }
    }
}

// --- Repacked layouts: four rows interleaved ---------------------------
//
// For `smmla` (and the matching `sdot` GEMV), four rows' blocks at the same
// column are stored together, 8 bytes at a time, so that one 16-byte load is
// already `[row 0 chunk | row 1 chunk]` (or rows 2 and 3) and the kernel
// needs no `zip`s. Group `g` (rows 4g..4g+4), block `b` sits at `g·nb + b`.
// This is llama.cpp's `block_q8_0x4` / `block_q4_0x4` with an 8-byte
// interleave (its `Q4_0_4x8` repack).

/// Four Q8_0 rows at one block column: `qs[32c + 8r..32c + 8r + 8]` holds
/// values `8c..8c + 8` of row `r`, for chunks c = 0..4.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8_0x4 {
    /// The four rows' scales as f16 bits.
    pub d: [u16; 4],
    pub qs: [i8; 4 * QK],
}

/// Four Q4_0 rows at one block column: `qs[32h + 8r..32h + 8r + 8]` holds
/// bytes `8h..8h + 8` of row `r`'s block, for halves h = 0, 1 (so the low
/// nibbles are values `8h..`, the high nibbles values `16 + 8h..`). Every
/// nibble is stored XOR 8: read as a signed 4-bit number it is then `q − 8`
/// directly, so the kernel's unpack is one shift (low nibble, ×16) or one
/// mask (high nibble, ×16), and the ×16 is undone exactly when the integer
/// sum is converted to float.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ4_0x4 {
    pub d: [u16; 4],
    pub qs: [u8; 2 * QK],
}

/// Four Q5_0 rows at one block column. `qs` is laid out as in
/// [`BlockQ4_0x4`] but with plain nibbles. `qh` holds the fifth bits,
/// inverted, arranged for the 16-byte vectors the kernel builds: vector i
/// (row pair rp = i / 4, chunk c = i % 4 of 8 values) takes bit i of every
/// byte of `qh`, and lane k of it is row 2rp + k / 8, value 8c + k % 8.
/// Inverted, the fifth bit is the sign of the 5-bit two's-complement number
/// `q − 16`, so `(nibble << 3) | (bit << 7)` read as int8 is exactly
/// 8·(q − 16): the unpack is a shift, a mask and a bit insert, and the ×8
/// is undone exactly when the integer sum is converted to float.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ5_0x4 {
    pub d: [u16; 4],
    pub qh: [u8; 16],
    pub qs: [u8; 2 * QK],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0x4>() == 4 * 34);
const _: () = assert!(std::mem::size_of::<BlockQ4_0x4>() == 4 * 18);
const _: () = assert!(std::mem::size_of::<BlockQ5_0x4>() == 4 * 22);

/// Row-major Q8_0 blocks (`rows` rows, a multiple of 4) → groups of four.
pub fn repack_q8_0(b: &[BlockQ8_0], rows: usize) -> Vec<BlockQ8_0x4> {
    assert!(rows.is_multiple_of(4) && b.len().is_multiple_of(rows), "rows must be a multiple of 4");
    let nb = b.len() / rows;
    let mut out = Vec::with_capacity(b.len() / 4);
    for g in 0..rows / 4 {
        for k in 0..nb {
            let src = [0, 1, 2, 3].map(|r| &b[(4 * g + r) * nb + k]);
            let mut qs = [0; 4 * QK];
            for c in 0..QK / 8 {
                for (r, blk) in src.iter().enumerate() {
                    qs[32 * c + 8 * r..32 * c + 8 * r + 8].copy_from_slice(&blk.qs[8 * c..8 * c + 8]);
                }
            }
            out.push(BlockQ8_0x4 { d: src.map(|blk| blk.d), qs });
        }
    }
    out
}

/// Row-major Q4_0 blocks (`rows` rows, a multiple of 4) → groups of four,
/// nibbles XOR 8.
pub fn repack_q4_0(b: &[BlockQ4_0], rows: usize) -> Vec<BlockQ4_0x4> {
    assert!(rows.is_multiple_of(4) && b.len().is_multiple_of(rows), "rows must be a multiple of 4");
    let nb = b.len() / rows;
    let mut out = Vec::with_capacity(b.len() / 4);
    for g in 0..rows / 4 {
        for k in 0..nb {
            let src = [0, 1, 2, 3].map(|r| &b[(4 * g + r) * nb + k]);
            let mut qs = [0; 2 * QK];
            for h in 0..2 {
                for (r, blk) in src.iter().enumerate() {
                    for j in 0..8 {
                        qs[32 * h + 8 * r + j] = blk.qs[8 * h + j] ^ 0x88;
                    }
                }
            }
            out.push(BlockQ4_0x4 { d: src.map(|blk| blk.d), qs });
        }
    }
    out
}

/// Row-major Q5_0 blocks (`rows` rows, a multiple of 4) → groups of four.
pub fn repack_q5_0(b: &[BlockQ5_0], rows: usize) -> Vec<BlockQ5_0x4> {
    assert!(rows.is_multiple_of(4) && b.len().is_multiple_of(rows), "rows must be a multiple of 4");
    let nb = b.len() / rows;
    let mut out = Vec::with_capacity(b.len() / 4);
    for g in 0..rows / 4 {
        for k in 0..nb {
            let src = [0, 1, 2, 3].map(|r| &b[(4 * g + r) * nb + k]);
            let mut qs = [0; 2 * QK];
            for h in 0..2 {
                for (r, blk) in src.iter().enumerate() {
                    qs[32 * h + 8 * r..32 * h + 8 * r + 8].copy_from_slice(&blk.qs[8 * h..8 * h + 8]);
                }
            }
            let mut qh = [0u8; 16];
            for (lane, byte) in qh.iter_mut().enumerate() {
                for i in 0..8 {
                    let (row, value) = (2 * (i / 4) + lane / 8, 8 * (i % 4) + lane % 8);
                    let fifth = (u32::from_le_bytes(src[row].qh) >> value) & 1;
                    *byte |= ((fifth ^ 1) as u8) << i;
                }
            }
            out.push(BlockQ5_0x4 { d: src.map(|blk| blk.d), qh, qs });
        }
    }
    out
}

/// Row `r` (0..4) of a group of repacked Q8_0 blocks, as f32.
pub fn dequantize_q8_0x4(group: &[BlockQ8_0x4], r: usize, out: &mut [f32]) {
    assert_eq!(out.len(), group.len() * QK);
    for (blk, o) in group.iter().zip(out.as_chunks_mut::<QK>().0.iter_mut()) {
        let d = f16_to_f32(blk.d[r]);
        for c in 0..QK / 8 {
            for j in 0..8 {
                o[8 * c + j] = blk.qs[32 * c + 8 * r + j] as f32 * d;
            }
        }
    }
}

/// Row `r` (0..4) of a group of repacked Q4_0 blocks, as f32.
pub fn dequantize_q4_0x4(group: &[BlockQ4_0x4], r: usize, out: &mut [f32]) {
    assert_eq!(out.len(), group.len() * QK);
    for (blk, o) in group.iter().zip(out.as_chunks_mut::<QK>().0.iter_mut()) {
        let d = f16_to_f32(blk.d[r]);
        for h in 0..2 {
            for j in 0..8 {
                let q = blk.qs[32 * h + 8 * r + j] ^ 0x88;
                o[8 * h + j] = ((q & 0x0F) as i32 - 8) as f32 * d;
                o[16 + 8 * h + j] = ((q >> 4) as i32 - 8) as f32 * d;
            }
        }
    }
}

/// Row `r` (0..4) of a group of repacked Q5_0 blocks, as f32.
pub fn dequantize_q5_0x4(group: &[BlockQ5_0x4], r: usize, out: &mut [f32]) {
    assert_eq!(out.len(), group.len() * QK);
    for (blk, o) in group.iter().zip(out.as_chunks_mut::<QK>().0.iter_mut()) {
        let d = f16_to_f32(blk.d[r]);
        for (value, v) in o.iter_mut().enumerate() {
            let (c, j) = (value / 8, value % 8);
            let byte = blk.qs[32 * (c % 2) + 8 * r + j];
            let lo = if c < 2 { byte & 0x0F } else { byte >> 4 };
            let (i, lane) = (4 * (r / 2) + c, 8 * (r % 2) + j);
            let fifth = ((blk.qh[lane] >> i) & 1) ^ 1;
            *v = ((lo | (fifth << 4)) as i32 - 16) as f32 * d;
        }
    }
}

/// Scalar dot products of one quantized row with `x`. These are the
/// portable versions (host builds); `kernels::neon_quant` has the NEON ones.
pub mod scalar {
    use super::*;

    /// (a) Q8_0 row · f32 `x`.
    pub fn dot_q8_0_f32(row: &[BlockQ8_0], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (blk, xs) in row.iter().zip(x.as_chunks::<QK>().0) {
            let mut s = 0.0f32;
            for (&q, &v) in blk.qs.iter().zip(xs) {
                s += q as f32 * v;
            }
            acc += f16_to_f32(blk.d) * s;
        }
        acc
    }

    /// (b) Q8_0 row · Q8_0 `x`: integer dot per block, scaled by d_w · d_x.
    pub fn dot_q8_0_q8_0(row: &[BlockQ8_0], x: &[BlockQ8_0]) -> f32 {
        let mut acc = 0.0f32;
        for (w, v) in row.iter().zip(x) {
            let s: i32 = w.qs.iter().zip(&v.qs).map(|(&a, &b)| a as i32 * b as i32).sum();
            acc += s as f32 * (f16_to_f32(w.d) * f16_to_f32(v.d));
        }
        acc
    }

    /// (a) Q4_0 row · f32 `x`.
    pub fn dot_q4_0_f32(row: &[BlockQ4_0], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (blk, xs) in row.iter().zip(x.as_chunks::<QK>().0) {
            let mut s = 0.0f32;
            for j in 0..QK / 2 {
                s += ((blk.qs[j] & 0x0F) as i32 - 8) as f32 * xs[j];
                s += ((blk.qs[j] >> 4) as i32 - 8) as f32 * xs[j + QK / 2];
            }
            acc += f16_to_f32(blk.d) * s;
        }
        acc
    }

    /// (b) Q4_0 row · Q8_0 `x`.
    pub fn dot_q4_0_q8_0(row: &[BlockQ4_0], x: &[BlockQ8_0]) -> f32 {
        let mut acc = 0.0f32;
        for (w, v) in row.iter().zip(x) {
            let mut s = 0i32;
            for j in 0..QK / 2 {
                s += ((w.qs[j] & 0x0F) as i32 - 8) * v.qs[j] as i32;
                s += ((w.qs[j] >> 4) as i32 - 8) * v.qs[j + QK / 2] as i32;
            }
            acc += s as f32 * (f16_to_f32(w.d) * f16_to_f32(v.d));
        }
        acc
    }

    /// (a) Q5_0 row · f32 `x`.
    pub fn dot_q5_0_f32(row: &[BlockQ5_0], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (blk, xs) in row.iter().zip(x.as_chunks::<QK>().0) {
            let s: f32 = xs.iter().enumerate().map(|(j, &v)| q5_0_value(blk, j) as f32 * v).sum();
            acc += f16_to_f32(blk.d) * s;
        }
        acc
    }

    /// (b) Q4_K row · Q8_0 `x` (8 activation blocks per super-block): per
    /// sub-block `d_x · (d·sc·Σ q·x − dmin·m·Σ x)`, both sums exact
    /// integers.
    pub fn dot_q4_k_q8_0(row: &[BlockQ4_K], x: &[BlockQ8_0]) -> f32 {
        let mut acc = 0.0f32;
        for (w, xs) in row.iter().zip(x.as_chunks::<{ QK_K / QK }>().0) {
            let (d, dmin, q) = (f16_to_f32(w.d), f16_to_f32(w.dmin), q4_k_values(w));
            for (j, v) in xs.iter().enumerate() {
                let (sc, m) = q4_k_scale_min(&w.scales, j);
                let qx: i32 = v.qs.iter().zip(&q[32 * j..]).map(|(&a, &b)| a as i32 * b as i32).sum();
                let sx: i32 = v.qs.iter().map(|&a| a as i32).sum();
                acc += f16_to_f32(v.d) * (d * sc as f32 * qx as f32 - dmin * m as f32 * sx as f32);
            }
        }
        acc
    }

    /// (b) Q6_K row · Q8_0 `x`: per activation block, two 16-weight
    /// sub-blocks' integer sums weighted by their scales.
    pub fn dot_q6_k_q8_0(row: &[BlockQ6_K], x: &[BlockQ8_0]) -> f32 {
        let mut acc = 0.0f32;
        for (w, xs) in row.iter().zip(x.as_chunks::<{ QK_K / QK }>().0) {
            let (d, q) = (f16_to_f32(w.d), q6_k_values(w));
            for (j, v) in xs.iter().enumerate() {
                let sub = |h: usize| -> i32 { (16 * h..16 * h + 16).map(|l| q[32 * j + l] as i32 * v.qs[l] as i32).sum() };
                let s = w.scales[2 * j] as i32 * sub(0) + w.scales[2 * j + 1] as i32 * sub(1);
                acc += f16_to_f32(v.d) * d * s as f32;
            }
        }
        acc
    }

    /// (b) Q5_0 row · Q8_0 `x`.
    pub fn dot_q5_0_q8_0(row: &[BlockQ5_0], x: &[BlockQ8_0]) -> f32 {
        let mut acc = 0.0f32;
        for (w, v) in row.iter().zip(x) {
            let s: i32 = v.qs.iter().enumerate().map(|(j, &b)| q5_0_value(w, j) * b as i32).sum();
            acc += s as f32 * (f16_to_f32(w.d) * f16_to_f32(v.d));
        }
        acc
    }
}

/// Reinterprets raw GGUF / golden-file bytes as blocks (alignment 2).
pub fn blocks_from_bytes<B: Copy>(bytes: &[u8]) -> Vec<B> {
    let size = std::mem::size_of::<B>();
    assert_eq!(bytes.len() % size, 0, "not a whole number of blocks");
    bytes
        .chunks_exact(size)
        // SAFETY: B is a plain #[repr(C)] struct of integers (any bit pattern
        // is valid); read_unaligned copes with any byte alignment.
        .map(|c| unsafe { std::ptr::read_unaligned(c.as_ptr() as *const B) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{golden, max_errors, read_npy};
    use std::path::Path;

    #[test]
    fn f16_rounding_is_round_to_nearest_even() {
        // Every finite f16 converts back to itself.
        for h in 0..=0xFFFFu16 {
            let f = f16_to_f32(h);
            if f.is_finite() {
                assert_eq!(f32_to_f16(f), h, "{h:#06x}");
            }
        }
        // Halfway between 1.0 and the next f16 (1 + 2^-10): ties to even (1.0).
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00);
        // Halfway between 1+2^-10 and 1+2^-9: ties to even (1+2^-9).
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
        assert_eq!(f32_to_f16(1e6), 0x7C00); // overflow → inf
        assert_eq!(f32_to_f16(2f32.powi(-26)), 0); // below half the smallest subnormal
        assert_eq!(f32_to_f16(3.0 * 2f32.powi(-26)), 1); // above half → smallest subnormal
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    fn cases() -> Vec<std::path::PathBuf> {
        let Some(root) = golden("quant") else { return Vec::new() };
        let mut v: Vec<_> = std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()).collect();
        v.sort();
        v
    }

    fn load(dir: &Path) -> (usize, usize, Vec<f32>, Vec<f32>) {
        let (shape, w) = read_npy(&dir.join("w.npy"));
        (shape[0], shape[1], w, read_npy(&dir.join("x.npy")).1)
    }

    /// Our quantizers produce exactly llama.cpp's bytes.
    #[test]
    fn quantizers_are_byte_identical_to_llama_cpp() {
        for dir in cases() {
            let (_, _, w, x) = load(&dir);
            let mut q8 = vec![BlockQ8_0 { d: 0, qs: [0; QK] }; w.len() / QK];
            quantize_q8_0(&w, &mut q8);
            assert_eq!(q8, blocks_from_bytes::<BlockQ8_0>(&std::fs::read(dir.join("w_q8_0.bin")).unwrap()), "{}", dir.display());
            let mut q4 = vec![BlockQ4_0 { d: 0, qs: [0; QK / 2] }; w.len() / QK];
            quantize_q4_0(&w, &mut q4);
            assert_eq!(q4, blocks_from_bytes::<BlockQ4_0>(&std::fs::read(dir.join("w_q4_0.bin")).unwrap()), "{}", dir.display());
            let mut q5 = vec![BlockQ5_0 { d: 0, qh: [0; 4], qs: [0; QK / 2] }; w.len() / QK];
            quantize_q5_0(&w, &mut q5);
            assert_eq!(q5, blocks_from_bytes::<BlockQ5_0>(&std::fs::read(dir.join("w_q5_0.bin")).unwrap()), "{}", dir.display());
            let mut xq = vec![BlockQ8_0 { d: 0, qs: [0; QK] }; x.len() / QK];
            quantize_q8_0(&x, &mut xq);
            assert_eq!(xq, blocks_from_bytes::<BlockQ8_0>(&std::fs::read(dir.join("x_q8_0.bin")).unwrap()));
            // The NEON quantizer (activations, AArch64): the same bytes.
            #[cfg(target_arch = "aarch64")]
            for (v, file) in [(&w, "w_q8_0.bin"), (&x, "x_q8_0.bin")] {
                let mut q = vec![BlockQ8_0 { d: 0, qs: [0; QK] }; v.len() / QK];
                crate::kernels::neon_quant::quantize_q8_0(v, &mut q);
                assert_eq!(q, blocks_from_bytes::<BlockQ8_0>(&std::fs::read(dir.join(file)).unwrap()), "{} neon {file}", dir.display());
            }
        }
    }

    fn k_cases() -> Vec<std::path::PathBuf> {
        let Some(root) = golden("quant_k") else { return Vec::new() };
        let mut v: Vec<_> = std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()).collect();
        v.sort();
        v
    }

    /// Q4_K / Q6_K: random blocks decode to exactly gguf's values, and the
    /// scalar dot products with a Q8_0 activation match the float64 "b"
    /// reference.
    #[test]
    fn k_quants_match_gguf() {
        for dir in k_cases() {
            let (xs, x) = read_npy(&dir.join("x.npy"));
            let (t, n) = (xs[0], xs[1]);
            let xq: Vec<BlockQ8_0> = blocks_from_bytes(&std::fs::read(dir.join("x_q8_0.bin")).unwrap());
            let (nb, nk) = (n / QK, n / QK_K);
            let bytes = |name: &str| std::fs::read(dir.join(format!("w_{name}.bin"))).unwrap();
            let (q4, q6): (Vec<BlockQ4_K>, Vec<BlockQ6_K>) = (blocks_from_bytes(&bytes("q4_k")), blocks_from_bytes(&bytes("q6_k")));
            let d = q4.len() / nk;
            let mut ours = vec![0.0; d * n];
            dequantize_q4_k(&q4, &mut ours);
            assert_eq!(ours, read_npy(&dir.join("deq_q4_k.npy")).1, "{} q4_k", dir.display());
            dequantize_q6_k(&q6, &mut ours);
            assert_eq!(ours, read_npy(&dir.join("deq_q6_k.npy")).1, "{} q6_k", dir.display());
            let check = |name: &str, f: &dyn Fn(usize, &[BlockQ8_0]) -> f32| {
                let want = read_npy(&dir.join(format!("out_{name}_b.npy"))).1;
                let ours: Vec<f32> = (0..t * d).map(|k| f(k % d, &xq[k / d * nb..(k / d + 1) * nb])).collect();
                let (_, rel) = max_errors(&ours, &want);
                assert!(rel < 1e-6 * (n as f32).sqrt(), "{} {name}: rel {rel:.2e}", dir.display());
            };
            check("q4_k", &|i, x| scalar::dot_q4_k_q8_0(&q4[i * nk..(i + 1) * nk], x));
            check("q6_k", &|i, x| scalar::dot_q6_k_q8_0(&q6[i * nk..(i + 1) * nk], x));
            let _ = x;
        }
    }

    /// Repacking only moves bytes: every row of a repacked matrix
    /// dequantizes to exactly the values of the row-major one.
    #[test]
    fn repacked_rows_dequantize_like_row_major() {
        for dir in cases() {
            let (d, n, _, _) = load(&dir);
            if !d.is_multiple_of(4) {
                continue;
            }
            let q8: Vec<BlockQ8_0> = blocks_from_bytes(&std::fs::read(dir.join("w_q8_0.bin")).unwrap());
            let q4: Vec<BlockQ4_0> = blocks_from_bytes(&std::fs::read(dir.join("w_q4_0.bin")).unwrap());
            let q5: Vec<BlockQ5_0> = blocks_from_bytes(&std::fs::read(dir.join("w_q5_0.bin")).unwrap());
            let (p8, p4, p5) = (repack_q8_0(&q8, d), repack_q4_0(&q4, d), repack_q5_0(&q5, d));
            let nb = n / QK;
            let (mut want, mut got) = (vec![0.0; n], vec![0.0; n]);
            for i in 0..d {
                let g = i / 4 * nb..(i / 4 + 1) * nb;
                dequantize_q8_0(&q8[i * nb..(i + 1) * nb], &mut want);
                dequantize_q8_0x4(&p8[g.clone()], i % 4, &mut got);
                assert_eq!(want, got, "{} q8_0 row {i}", dir.display());
                dequantize_q4_0(&q4[i * nb..(i + 1) * nb], &mut want);
                dequantize_q4_0x4(&p4[g.clone()], i % 4, &mut got);
                assert_eq!(want, got, "{} q4_0 row {i}", dir.display());
                dequantize_q5_0(&q5[i * nb..(i + 1) * nb], &mut want);
                dequantize_q5_0x4(&p5[g], i % 4, &mut got);
                assert_eq!(want, got, "{} q5_0 row {i}", dir.display());
            }
        }
    }

    /// The four dot products (scalar everywhere, NEON on AArch64) against
    /// numpy's float64 reference.
    #[test]
    fn scalar_dots_match_reference() {
        for dir in cases() {
            let (d, n, _, x) = load(&dir);
            let q8: Vec<BlockQ8_0> = blocks_from_bytes(&std::fs::read(dir.join("w_q8_0.bin")).unwrap());
            let q4: Vec<BlockQ4_0> = blocks_from_bytes(&std::fs::read(dir.join("w_q4_0.bin")).unwrap());
            let q5: Vec<BlockQ5_0> = blocks_from_bytes(&std::fs::read(dir.join("w_q5_0.bin")).unwrap());
            let xq: Vec<BlockQ8_0> = blocks_from_bytes(&std::fs::read(dir.join("x_q8_0.bin")).unwrap());
            let nb = n / QK;
            let check = |name: &str, f: &dyn Fn(usize) -> f32| {
                let want = read_npy(&dir.join(format!("out_{name}.npy"))).1;
                let ours: Vec<f32> = (0..d).map(f).collect();
                let (_, rel) = max_errors(&ours, &want);
                assert!(rel < 1e-6 * (n as f32).sqrt().max(4.0), "{} {name}: rel {rel:.2e}", dir.display());
            };
            check("q8_0_a", &|i| scalar::dot_q8_0_f32(&q8[i * nb..(i + 1) * nb], &x));
            check("q8_0_b", &|i| scalar::dot_q8_0_q8_0(&q8[i * nb..(i + 1) * nb], &xq));
            check("q4_0_a", &|i| scalar::dot_q4_0_f32(&q4[i * nb..(i + 1) * nb], &x));
            check("q4_0_b", &|i| scalar::dot_q4_0_q8_0(&q4[i * nb..(i + 1) * nb], &xq));
            check("q5_0_a", &|i| scalar::dot_q5_0_f32(&q5[i * nb..(i + 1) * nb], &x));
            check("q5_0_b", &|i| scalar::dot_q5_0_q8_0(&q5[i * nb..(i + 1) * nb], &xq));
            // Dequantized rows (what `Weight::row_f32` returns) times x in
            // f64: the "a" reference, which used gguf's dequantizer.
            check("q5_0_a", &|i| {
                let mut r = vec![0.0; n];
                dequantize_q5_0(&q5[i * nb..(i + 1) * nb], &mut r);
                r.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32
            });
            #[cfg(target_arch = "aarch64")]
            {
                use crate::kernels::neon_quant as nq;
                check("q8_0_a", &|i| nq::dot_q8_0_f32(&q8[i * nb..(i + 1) * nb], &x));
                check("q8_0_b", &|i| nq::dot_q8_0_q8_0(&q8[i * nb..(i + 1) * nb], &xq));
                check("q4_0_a", &|i| nq::dot_q4_0_f32(&q4[i * nb..(i + 1) * nb], &x));
                check("q4_0_b", &|i| nq::dot_q4_0_q8_0(&q4[i * nb..(i + 1) * nb], &xq));
                check("q5_0_b", &|i| nq::dot_q5_0_q8_0(&q5[i * nb..(i + 1) * nb], &xq));
            }
        }
    }
}
