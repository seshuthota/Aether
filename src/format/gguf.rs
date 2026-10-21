//! GGUF, llama.cpp's model file format (version 3).
//!
//! ```text
//! u32 magic "GGUF" | u32 version | u64 tensor_count | u64 metadata_count
//! metadata_count × { string key | u32 type | value }
//! tensor_count   × { string name | u32 n_dims | u64 dims[n_dims] | u32 ggml_type | u64 offset }
//! padding to `general.alignment` (default 32)
//! tensor data: each tensor at data_start + offset (offsets are aligned too)
//! ```
//!
//! Strings are `u64 length` + UTF-8 bytes. Metadata values are typed: the
//! integer and float widths, bool, string, or an array (`u32 element type`,
//! `u64 count`, elements). Dimensions are listed **fastest-varying first**:
//! a weight with 896 columns and 4864 rows has `dims = [896, 4864]` — the
//! reverse of the `[rows, cols]` shape PyTorch would print.
//!
//! The file is memory-mapped; tensor data is handed out as byte ranges of
//! the mapping, so quantized weights are used in place without copying.

use super::mmap::Mmap;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// A metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U64(v) => Some(v),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F64(v) => Some(v),
            Value::U64(v) => Some(v as f64),
            Value::I64(v) => Some(v as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self { Some(s) } else { None }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        if let Value::Arr(a) = self { Some(a) } else { None }
    }
}

/// ggml tensor types this engine reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    /// Any other type, by its ggml id (refused when its data is requested).
    Other(u32),
}

impl GgmlType {
    fn from_id(id: u32) -> GgmlType {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            8 => GgmlType::Q8_0,
            30 => GgmlType::BF16,
            other => GgmlType::Other(other),
        }
    }

    /// Bytes for `n` elements (n a multiple of the block size for blocks).
    pub fn bytes(self, n: usize) -> Option<usize> {
        match self {
            GgmlType::F32 => Some(n * 4),
            GgmlType::F16 | GgmlType::BF16 => Some(n * 2),
            GgmlType::Q8_0 => Some(n / 32 * 34),
            GgmlType::Q4_0 => Some(n / 32 * 18),
            GgmlType::Other(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Fastest-varying first: `dims[0]` is the row length (columns).
    pub dims: Vec<usize>,
    pub ty: GgmlType,
    /// Absolute byte offset of the data in the file.
    pub offset: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    pub map: Arc<Mmap>,
    pub version: u32,
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Little-endian reader over the mapped header.
struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        let s = self.b.get(self.at..self.at + n).ok_or_else(|| bad(format!("file ends at byte {} inside the header", self.at)))?;
        self.at += n;
        Ok(s)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> io::Result<String> {
        let n = self.u64()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|e| bad(format!("string is not UTF-8: {e}")))
    }

    fn value(&mut self, ty: u32) -> io::Result<Value> {
        let b = |r: &mut Self, n| r.take(n).map(|s| s.to_vec());
        Ok(match ty {
            0 => Value::U64(b(self, 1)?[0] as u64),
            1 => Value::I64(b(self, 1)?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(b(self, 1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let (et, n) = (self.u32()?, self.u64()? as usize);
                Value::Arr((0..n).map(|_| self.value(et)).collect::<io::Result<_>>()?)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(bad(format!("unknown metadata type {other}"))),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Gguf> {
        let map = Arc::new(Mmap::open(path.as_ref())?);
        let mut r = Reader { b: map.as_bytes(), at: 0 };
        if r.take(4)? != b"GGUF" {
            return Err(bad(format!("{}: not a GGUF file", path.as_ref().display())));
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(bad(format!("GGUF version {version} (only 3 is supported)")));
        }
        let (n_tensors, n_kv) = (r.u64()? as usize, r.u64()? as usize);
        let mut metadata = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            metadata.push((key, r.value(ty)?));
        }
        let mut infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let dims = (0..n_dims).map(|_| r.u64().map(|d| d as usize)).collect::<io::Result<Vec<_>>>()?;
            let ty = GgmlType::from_id(r.u32()?);
            let offset = r.u64()? as usize;
            infos.push((name, dims, ty, offset));
        }
        let mut g = Gguf { map: map.clone(), version, metadata, tensors: Vec::new() };
        let align = g.get("general.alignment").and_then(Value::as_u64).unwrap_or(32) as usize;
        let data_start = r.at.div_ceil(align) * align;
        for (name, dims, ty, offset) in infos {
            let info = TensorInfo { name, dims, ty, offset: data_start + offset };
            if let Some(n) = info.ty.bytes(info.numel())
                && info.offset + n > map.len()
            {
                return Err(bad(format!("{}: data runs past the end of the file", info.name)));
            }
            g.tensors.push(info);
        }
        Ok(g)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The raw bytes of a tensor's data.
    pub fn data(&self, t: &TensorInfo) -> io::Result<&[u8]> {
        let n = t.ty.bytes(t.numel()).ok_or_else(|| bad(format!("{}: unsupported tensor type {:?}", t.name, t.ty)))?;
        Ok(&self.map.as_bytes()[t.offset..t.offset + n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's own conversion of SmolLM2 (convert_hf_to_gguf.py, then
    /// llama-quantize --pure Q8_0).
    #[test]
    fn reads_llama_cpp_files() {
        let Some(p) = crate::testutil::golden("gguf/smollm2-135m-q8_0.gguf") else { return };
        let g = Gguf::open(p).unwrap();
        assert_eq!(g.get("general.architecture").and_then(Value::as_str), Some("llama"));
        assert_eq!(g.get("llama.embedding_length").and_then(Value::as_u64), Some(576));
        assert_eq!(g.get("tokenizer.ggml.tokens").and_then(Value::as_array).map(|a| a.len()), Some(49152));
        let q = g.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!((q.ty, &q.dims[..]), (GgmlType::Q8_0, &[576, 576][..]));
        let ffn = g.tensor("blk.0.ffn_down.weight").unwrap();
        assert_eq!(ffn.dims, [1536, 576]); // [cols, rows]: down maps 1536 → 576
        assert_eq!(g.data(q).unwrap().len(), 576 * 576 / 32 * 34);
    }
}
