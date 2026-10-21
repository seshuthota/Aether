//! The safetensors format (Hugging Face): a JSON index followed by raw bytes.
//!
//! ```text
//! u64 (little-endian)   N = length of the header in bytes
//! N bytes               JSON: {"<name>": {"dtype": "BF16", "shape": [576, 576],
//!                              "data_offsets": [start, end]}, ...,
//!                              "__metadata__": {...}}   (optional)
//! rest                  tensor data; offsets are relative to here
//! ```
//!
//! Data is row-major, little-endian, packed with no padding between tensors.
//! The file is memory-mapped; [`SafeTensors::get`] returns a view into the
//! mapping, and conversion to f32 happens only when a caller asks.

use super::json::Json;
use super::mmap::Mmap;
use std::io;
use std::ops::Range;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
}

impl Dtype {
    fn parse(s: &str) -> Option<Dtype> {
        match s {
            "F32" => Some(Dtype::F32),
            "F16" => Some(Dtype::F16),
            "BF16" => Some(Dtype::BF16),
            _ => None,
        }
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }
}

/// One tensor's entry in the header.
#[derive(Debug, Clone)]
pub struct Info {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range within the file (header already accounted for).
    bytes: Range<usize>,
}

/// A tensor's raw bytes plus how to read them.
pub struct View<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

impl View<'_> {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Converts every element to f32 into `out` (length `numel`).
    pub fn to_f32(&self, out: &mut [f32]) {
        assert_eq!(out.len(), self.numel());
        match self.dtype {
            Dtype::F32 => {
                for (o, b) in out.iter_mut().zip(self.bytes.as_chunks::<4>().0) {
                    *o = f32::from_le_bytes(*b);
                }
            }
            Dtype::BF16 => {
                for (o, b) in out.iter_mut().zip(self.bytes.as_chunks::<2>().0) {
                    *o = bf16_to_f32(u16::from_le_bytes(*b));
                }
            }
            Dtype::F16 => {
                for (o, b) in out.iter_mut().zip(self.bytes.as_chunks::<2>().0) {
                    *o = f16_to_f32(u16::from_le_bytes(*b));
                }
            }
        }
    }
}

/// bfloat16 is the top 16 bits of an f32 (same sign and 8-bit exponent, 7
/// mantissa bits instead of 23), so widening is exact: put the 16 bits back
/// on top and fill the low 16 with zeros.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// IEEE half precision: 1 sign, 5 exponent (bias 15), 10 mantissa bits.
/// Every f16 value is exactly representable as f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = match exp {
        0 if man == 0 => sign, // ±0
        0 => {
            // Subnormal: value = man · 2^-24; renormalise for f32.
            let shift = man.leading_zeros() - 21; // bring the top set bit to bit 10
            let m = (man << shift) & 0x3FF;
            sign | ((127 - 15 - shift + 1) << 23) | (m << 13)
        }
        0x1F => sign | (0xFF << 23) | (man << 13), // inf / NaN
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

pub struct SafeTensors {
    map: Mmap,
    tensors: Vec<(String, Info)>,
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl SafeTensors {
    pub fn open(path: impl AsRef<Path>) -> io::Result<SafeTensors> {
        let path = path.as_ref();
        let map = Mmap::open(path)?;
        let b = map.as_bytes();
        let n = b.get(..8).map(|h| u64::from_le_bytes(h.try_into().unwrap()) as usize).ok_or_else(|| bad("no header length".into()))?;
        let header = b.get(8..8 + n).ok_or_else(|| bad(format!("header length {n} beyond file")))?;
        let header = std::str::from_utf8(header).map_err(|e| bad(format!("header not UTF-8: {e}")))?;
        let json = Json::parse(header).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let data_start = 8 + n;
        let mut tensors = Vec::new();
        for (name, entry) in json.as_object().ok_or_else(|| bad("header is not an object".into()))? {
            if name == "__metadata__" {
                continue;
            }
            let field = |k: &str| entry.get(k).ok_or_else(|| bad(format!("{name}: no {k}")));
            let dtype = field("dtype")?.as_str().and_then(Dtype::parse).ok_or_else(|| bad(format!("{name}: unsupported dtype")))?;
            let shape: Vec<usize> = field("shape")?
                .as_array()
                .ok_or_else(|| bad(format!("{name}: shape")))?
                .iter()
                .map(|d| d.as_usize().ok_or_else(|| bad(format!("{name}: bad dim"))))
                .collect::<io::Result<_>>()?;
            let offs = field("data_offsets")?.as_array().ok_or_else(|| bad(format!("{name}: offsets")))?;
            let (Some(s), Some(e)) = (offs.first().and_then(Json::as_usize), offs.get(1).and_then(Json::as_usize)) else {
                return Err(bad(format!("{name}: bad offsets")));
            };
            let bytes = data_start + s..data_start + e;
            if bytes.end > b.len() || e - s != shape.iter().product::<usize>() * dtype.size() {
                return Err(bad(format!("{name}: offsets {s}..{e} do not match shape {shape:?} {dtype:?}")));
            }
            tensors.push((name.clone(), Info { dtype, shape, bytes }));
        }
        Ok(SafeTensors { map, tensors })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(n, _)| n.as_str())
    }

    pub fn info(&self, name: &str) -> Option<&Info> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, i)| i)
    }

    pub fn get(&self, name: &str) -> Option<View<'_>> {
        let info = self.info(name)?;
        Some(View { dtype: info.dtype, shape: &info.shape, bytes: &self.map.as_bytes()[info.bytes.clone()] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_widening_is_exact() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC040), -3.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert!(bf16_to_f32(0x7FC0).is_nan());
    }

    /// The edge cases of the format: normals, the largest finite value,
    /// subnormals at both ends, infinities, NaN and negative zero.
    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // largest finite
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2f32.powi(-24)); // largest subnormal
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14)); // smallest normal
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    /// SmolLM2's real file: every tensor has a valid range, and the
    /// embedding is the expected [49152, 576] bf16.
    #[test]
    fn reads_smollm2_header() {
        let Some(p) = crate::testutil::golden("models/SmolLM2-135M/model.safetensors") else { return };
        let st = SafeTensors::open(p).unwrap();
        let e = st.get("model.embed_tokens.weight").unwrap();
        assert_eq!((e.dtype, e.shape), (Dtype::BF16, &[49152, 576][..]));
        assert_eq!(st.names().count(), 1 + 30 * 9 + 1);
    }
}
