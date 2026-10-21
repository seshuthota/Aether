//! A weight matrix in one of the storage formats the kernels understand.
//!
//! The model never looks inside: it hands a `Weight` to `kernels::linear*`,
//! which picks the kernel for the format. Shapes are implied by the call
//! (`rows = out.len()`, `cols = x.len()`), as for plain f32 matrices.

use crate::aligned::AlignedBuf;
use crate::format::mmap::Mmap;
use crate::quant::{self, BlockQ4_0, BlockQ8_0, QK};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

/// `len` values of type `T` inside a memory-mapped file, used in place.
/// Holds a reference to the mapping so it stays alive as long as any view.
pub struct Mapped<T> {
    map: Arc<Mmap>,
    offset: usize,
    len: usize,
    _t: PhantomData<T>,
}

// SAFETY: the mapping is read-only; a Mapped only ever hands out `&[T]`.
unsafe impl<T: Sync> Send for Mapped<T> {}
unsafe impl<T: Sync> Sync for Mapped<T> {}

impl<T: Copy> Mapped<T> {
    /// A view of `len` values at byte `offset`. Fails if the range runs past
    /// the file or the address is not aligned for `T` (a misaligned `&[T]`
    /// would be undefined behaviour).
    pub fn new(map: Arc<Mmap>, offset: usize, len: usize) -> std::io::Result<Mapped<T>> {
        let bytes = len * std::mem::size_of::<T>();
        let bad = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
        if offset + bytes > map.len() {
            return Err(bad(format!("tensor at {offset} (+{bytes} B) beyond the file")));
        }
        if (map.as_bytes().as_ptr() as usize + offset) % std::mem::align_of::<T>() != 0 {
            return Err(bad(format!("tensor at {offset} is not aligned for its type")));
        }
        Ok(Mapped { map, offset, len, _t: PhantomData })
    }
}

impl<T> Deref for Mapped<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `new` checked bounds and alignment; T is a plain-old-data
        // type (f32 or a block of integers) for which any bytes are valid;
        // the mapping lives as long as `self.map`.
        unsafe { std::slice::from_raw_parts(self.map.as_bytes().as_ptr().add(self.offset) as *const T, self.len) }
    }
}

/// Values owned in memory, or used in place from a mapped file.
pub enum Store<T> {
    Owned(Vec<T>),
    Mapped(Mapped<T>),
}

impl<T> Deref for Store<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Store::Owned(v) => v,
            Store::Mapped(m) => m,
        }
    }
}

/// f32 values: a 64-byte-aligned owned buffer, or used in place.
pub enum Floats {
    Buf(AlignedBuf),
    Mapped(Mapped<f32>),
}

impl Deref for Floats {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            Floats::Buf(b) => b,
            Floats::Mapped(m) => m,
        }
    }
}

pub enum Weight {
    F32(Floats),
    /// Row-major blocks: row `i` is blocks `i·cols/32 .. (i+1)·cols/32`.
    Q8_0(Store<BlockQ8_0>),
    Q4_0(Store<BlockQ4_0>),
}

impl From<AlignedBuf> for Weight {
    fn from(b: AlignedBuf) -> Weight {
        Weight::F32(Floats::Buf(b))
    }
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
            Weight::Q8_0(b) => std::mem::size_of_val::<[BlockQ8_0]>(b),
            Weight::Q4_0(b) => std::mem::size_of_val::<[BlockQ4_0]>(b),
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
                Weight::Q8_0(Store::Owned(b))
            }
            Format::Q4_0 => {
                let mut b = vec![BlockQ4_0 { d: 0, qs: [0; QK / 2] }; f.len() / QK];
                quant::quantize_q4_0(&f, &mut b);
                Weight::Q4_0(Store::Owned(b))
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
