//! A weight matrix in one of the storage formats the kernels understand.
//!
//! The model never looks inside: it hands a `Weight` to `kernels::linear*`,
//! which picks the kernel for the format. Shapes are implied by the call
//! (`rows = out.len()`, `cols = x.len()`), as for plain f32 matrices.

use crate::aligned::AlignedBuf;
use crate::quant::{self, BlockQ4_0, BlockQ8_0, QK};

pub enum Weight {
    F32(AlignedBuf),
    /// Row-major blocks: row `i` is blocks `i·cols/32 .. (i+1)·cols/32`.
    Q8_0(Vec<BlockQ8_0>),
    Q4_0(Vec<BlockQ4_0>),
}

/// Storage formats a model can be converted to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    F32,
    Q8_0,
    Q4_0,
}

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        match s.to_ascii_lowercase().as_str() {
            "f32" => Some(Format::F32),
            "q8_0" => Some(Format::Q8_0),
            "q4_0" => Some(Format::Q4_0),
            _ => None,
        }
    }
}

impl Weight {
    pub fn format(&self) -> Format {
        match self {
            Weight::F32(_) => Format::F32,
            Weight::Q8_0(_) => Format::Q8_0,
            Weight::Q4_0(_) => Format::Q4_0,
        }
    }

    /// Bytes of storage: what decode has to stream per use.
    pub fn bytes(&self) -> usize {
        match self {
            Weight::F32(b) => b.len() * 4,
            Weight::Q8_0(b) => std::mem::size_of_val(b.as_slice()),
            Weight::Q4_0(b) => std::mem::size_of_val(b.as_slice()),
        }
    }

    /// Converts an f32 matrix with rows of `cols` floats to `format`.
    /// Rows must be a whole number of 32-weight blocks to be quantized; any
    /// other matrix stays f32.
    pub fn convert(self, cols: usize, format: Format) -> Weight {
        let Weight::F32(f) = self else { return self };
        if format == Format::F32 || !cols.is_multiple_of(QK) {
            return Weight::F32(f);
        }
        match format {
            Format::Q8_0 => {
                let mut b = vec![BlockQ8_0 { d: 0, qs: [0; QK] }; f.len() / QK];
                quant::quantize_q8_0(&f, &mut b);
                Weight::Q8_0(b)
            }
            Format::Q4_0 => {
                let mut b = vec![BlockQ4_0 { d: 0, qs: [0; QK / 2] }; f.len() / QK];
                quant::quantize_q4_0(&f, &mut b);
                Weight::Q4_0(b)
            }
            Format::F32 => unreachable!(),
        }
    }

    /// Row `i` (of `out.len()` columns) as f32: an embedding lookup.
    pub fn row_f32(&self, i: usize, out: &mut [f32]) {
        let cols = out.len();
        match self {
            Weight::F32(f) => out.copy_from_slice(&f[i * cols..(i + 1) * cols]),
            Weight::Q8_0(b) => quant::dequantize_q8_0(&b[i * cols / QK..(i + 1) * cols / QK], out),
            Weight::Q4_0(b) => quant::dequantize_q4_0(&b[i * cols / QK..(i + 1) * cols / QK], out),
        }
    }
}
