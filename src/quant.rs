//! Block quantization, byte-compatible with llama.cpp / GGUF: Q8_0 and Q4_0.
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
//!
//! These follow `quantize_row_q8_0_ref` / `quantize_row_q4_0_ref` in
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

// The layouts must match GGUF's exactly (no padding).
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

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

const _: () = assert!(std::mem::size_of::<BlockQ8_0x4>() == 4 * 34);
const _: () = assert!(std::mem::size_of::<BlockQ4_0x4>() == 4 * 18);

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
            let (p8, p4) = (repack_q8_0(&q8, d), repack_q4_0(&q4, d));
            let nb = n / QK;
            let (mut want, mut got) = (vec![0.0; n], vec![0.0; n]);
            for i in 0..d {
                let g = i / 4 * nb..(i / 4 + 1) * nb;
                dequantize_q8_0(&q8[i * nb..(i + 1) * nb], &mut want);
                dequantize_q8_0x4(&p8[g.clone()], i % 4, &mut got);
                assert_eq!(want, got, "{} q8_0 row {i}", dir.display());
                dequantize_q4_0(&q4[i * nb..(i + 1) * nb], &mut want);
                dequantize_q4_0x4(&p4[g], i % 4, &mut got);
                assert_eq!(want, got, "{} q4_0 row {i}", dir.display());
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
            #[cfg(target_arch = "aarch64")]
            {
                use crate::kernels::neon_quant as nq;
                check("q8_0_a", &|i| nq::dot_q8_0_f32(&q8[i * nb..(i + 1) * nb], &x));
                check("q8_0_b", &|i| nq::dot_q8_0_q8_0(&q8[i * nb..(i + 1) * nb], &xq));
                check("q4_0_a", &|i| nq::dot_q4_0_f32(&q4[i * nb..(i + 1) * nb], &x));
                check("q4_0_b", &|i| nq::dot_q4_0_q8_0(&q4[i * nb..(i + 1) * nb], &xq));
            }
        }
    }
}
