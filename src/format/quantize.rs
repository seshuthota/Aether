//! Quantizing a GGUF file into another, as `llama-quantize` does: with
//! `--pure`, every matrix whose name ends in `weight` (not the norms) into
//! one format, the embedding included; or llama.cpp's Q4_K_M mix, a format
//! per matrix ([`q4_k_m_types`]). Vectors (norms, biases) are copied as
//! they are. The metadata is copied byte for byte, with `general.file_type`
//! set to the new format (and the importance matrix recorded when one is
//! used), and tensors are written in llama.cpp's order (by layer, then by
//! name), so for the same input the output file is the same, byte for
//! byte.

use super::gguf::{GgmlType, Gguf, Writer};
use super::imatrix::Imatrix;
use crate::aligned::AlignedBuf;
use crate::quant::{self, BlockQ4_K, BlockQ6_K, QK, QK_K};
use crate::weight::{Format, Weight};
use std::io::{self, BufWriter};
use std::path::Path;

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// What to quantize into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Recipe {
    /// Every matrix in one format (`llama-quantize --pure`).
    Pure(Format),
    /// llama.cpp's Q4_K_M mix ([`q4_k_m_types`]).
    Q4KM,
}

impl Recipe {
    pub fn parse(s: &str) -> Option<Recipe> {
        match s.to_ascii_lowercase().as_str() {
            "q4_k_m" => Some(Recipe::Q4KM),
            f => Format::parse(f).filter(|&f| f != Format::F32).map(Recipe::Pure),
        }
    }

    /// llama.cpp's `llama_ftype` for the file.
    fn file_type(self) -> u32 {
        match self {
            Recipe::Pure(Format::F32) => 0,
            Recipe::Pure(Format::Q4_0) => 2,
            Recipe::Pure(Format::Q8_0) => 7,
            Recipe::Pure(Format::Q5_0) => 8,
            Recipe::Q4KM => 15,
        }
    }
}

/// The format of each tensor of `g`, in `order`, for llama.cpp's Q4_K_M
/// (`llama_tensor_get_type` for a dense model, not Falcon or 70B, without
/// overrides). Matrices are Q4_K except:
/// - the classifier (or the embedding, when it is also the classifier):
///   Q6_K, or Q8_0 when its rows are not a multiple of 256 values;
/// - attn_v and ffn_down: Q6_K in the first and last eighth of the layers
///   and every third one between (`use_more_bits`), counted in `order`;
/// - then a K-quant whose rows are not a multiple of 256 falls back to a
///   32-value format: Q4_K to Q5_0, Q6_K to Q8_0.
///
/// Q5_K (fused attn_qkv) and its fallback Q5_1 are not written here.
pub fn q4_k_m_types(g: &Gguf, order: &[usize]) -> io::Result<Vec<GgmlType>> {
    let arch = g.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("");
    let n_layer = g.get(&format!("{arch}.block_count")).and_then(|v| v.as_u64()).ok_or_else(|| bad(format!("no {arch}.block_count")))? as usize;
    let is_v = |n: &str| n.contains("attn_v.weight") || n.contains("attn_qkv.weight") || n.contains("attn_kv_b.weight");
    let n_attn_v = g.tensors.iter().filter(|t| is_v(&t.name)).count();
    let tied = !g.tensors.iter().any(|t| t.name == "output.weight");
    let more_bits = |i: usize, n: usize| i < n / 8 || i >= 7 * n / 8 || (i as isize - (n / 8) as isize).rem_euclid(3) == 2;
    let (mut i_attn_v, mut i_down) = (0, 0);
    let mut types = Vec::new();
    for &i in order {
        let t = &g.tensors[i];
        let name = t.name.as_str();
        if !is_matrix(name, &t.dims) {
            types.push(t.ty);
            continue;
        }
        let cols = t.dims[0];
        let mut ty = GgmlType::Q4_K;
        if name == "output.weight" || (tied && name == "token_embd.weight") {
            ty = if cols.is_multiple_of(QK_K) { GgmlType::Q6_K } else { GgmlType::Q8_0 };
        } else if name == "token_embd.weight" {
        } else if is_v(name) {
            if name.contains("attn_qkv.weight") || name.contains("attn_kv_b.weight") {
                return Err(bad(format!("{name}: fused or MLA attention (Q5_K) is not supported")));
            }
            if more_bits(i_attn_v, n_attn_v) {
                ty = GgmlType::Q6_K;
            }
            i_attn_v += 1;
        } else if name.contains("ffn_down") {
            if more_bits(i_down, n_layer) {
                ty = GgmlType::Q6_K;
            }
            i_down += 1;
        }
        if !cols.is_multiple_of(QK_K) {
            ty = match ty {
                GgmlType::Q4_K => GgmlType::Q5_0,
                GgmlType::Q6_K => GgmlType::Q8_0,
                ty => ty,
            };
            if !cols.is_multiple_of(QK) {
                return Err(bad(format!("{name}: {cols} columns, not a multiple of {QK}")));
            }
        }
        types.push(ty);
    }
    Ok(types)
}

/// `x`'s rows of `cols` quantized by `f` (a row quantizer with an
/// optional importance per column), as bytes.
fn rows_into<B: Copy>(x: &[f32], cols: usize, per_row: usize, qw: Option<&[f32]>, f: impl Fn(&[f32], Option<&[f32]>, &mut [B])) -> Vec<u8> {
    // SAFETY: B is a plain #[repr(C)] struct of integers; zero is valid.
    let mut out = vec![unsafe { std::mem::zeroed::<B>() }; x.len() / cols * per_row];
    for (row, b) in x.chunks_exact(cols).zip(out.chunks_exact_mut(per_row)) {
        f(row, qw, b);
    }
    bytes_of(&out)
}

/// Whether `llama-quantize` converts this tensor (`tensor_allows_quantization`
/// for the dense models read here).
fn is_matrix(name: &str, dims: &[usize]) -> bool {
    dims.len() >= 2 && name.ends_with("weight") && !name.contains("_norm.weight")
}

/// llama.cpp's tensor order (`weight_name_comparer`): tensors outside the
/// layers first, then layer by layer, by name within each.
fn order_key(name: &str) -> (i64, &str) {
    let layer = name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|n| n.parse().ok()).unwrap_or(-1);
    (layer, name)
}

/// One tensor's values as f32 (from f32 or f16 data).
fn floats(g: &Gguf, i: usize) -> io::Result<AlignedBuf> {
    let t = &g.tensors[i];
    let data = g.data(t)?;
    let mut buf = AlignedBuf::zeros(t.numel());
    let out = buf.as_mut_slice();
    match t.ty {
        GgmlType::F32 => out.iter_mut().zip(data.as_chunks::<4>().0).for_each(|(o, b)| *o = f32::from_le_bytes(*b)),
        GgmlType::F16 => out.iter_mut().zip(data.as_chunks::<2>().0).for_each(|(o, b)| *o = quant::f16_to_f32(u16::from_le_bytes(*b))),
        ty => return Err(bad(format!("{}: {ty:?} input (only F32 and F16 are quantized)", t.name))),
    }
    Ok(buf)
}

fn bytes_of<B: Copy>(blocks: &[B]) -> Vec<u8> {
    // SAFETY: B is a plain #[repr(C)] block of integers with no padding.
    unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, std::mem::size_of_val(blocks)) }.to_vec()
}

/// Quantizes the GGUF at `input` into `output` with every matrix in
/// `format`. With an importance matrix (`imatrix`, read from
/// `imatrix_path`, which is recorded in the file as given), each matrix it
/// has an entry for is quantized with it, as `--pure --imatrix` does.
pub fn quantize(input: &Path, output: &Path, format: Format, imatrix: Option<(&Imatrix, &str)>) -> io::Result<()> {
    quantize_with(input, output, Recipe::Pure(format), imatrix)
}

/// [`quantize`] by any [`Recipe`].
pub fn quantize_with(input: &Path, output: &Path, recipe: Recipe, imatrix: Option<(&Imatrix, &str)>) -> io::Result<()> {
    let g = Gguf::open(input)?;
    let mut w = Writer::with_metadata_of(&g);
    w.set_u32("general.quantization_version", 2);
    w.set_u32("general.file_type", recipe.file_type());
    for key in ["split.no", "split.count", "split.tensors.count"] {
        w.remove(key);
    }
    if let Some((im, path)) = imatrix {
        // The values llama-quantize records (strings cut at 127 bytes).
        let cut = |s: &str| s.char_indices().nth(127).map_or(s.to_string(), |(i, _)| s[..i].to_string());
        w.set_str("quantize.imatrix.file", &cut(path));
        if let Some(d) = im.datasets.first() {
            w.set_str("quantize.imatrix.dataset", &cut(d));
        }
        w.set_u32("quantize.imatrix.entries_count", im.len() as u32);
        if im.chunks > 0 {
            w.set_u32("quantize.imatrix.chunks_count", im.chunks as u32);
        }
    }
    let mut order: Vec<usize> = (0..g.tensors.len()).collect();
    order.sort_by(|&a, &b| order_key(&g.tensors[a].name).cmp(&order_key(&g.tensors[b].name)));
    let types = match recipe {
        Recipe::Pure(format) => {
            let target = match format {
                Format::F32 => GgmlType::F32,
                Format::Q8_0 => GgmlType::Q8_0,
                Format::Q4_0 => GgmlType::Q4_0,
                Format::Q5_0 => GgmlType::Q5_0,
            };
            let mut v = Vec::new();
            for &i in &order {
                let t = &g.tensors[i];
                let quantized = is_matrix(&t.name, &t.dims);
                if quantized && !t.dims[0].is_multiple_of(QK) {
                    return Err(bad(format!("{}: {} columns, not a multiple of {QK}", t.name, t.dims[0])));
                }
                v.push(if quantized { target } else { t.ty });
            }
            v
        }
        Recipe::Q4KM => q4_k_m_types(&g, &order)?,
    };
    for (&i, &ty) in order.iter().zip(&types) {
        let t = &g.tensors[i];
        w.add_tensor(&t.name, &t.dims, ty);
    }
    let mut out = BufWriter::new(std::fs::File::create(output)?);
    w.write(&mut out, |k| {
        let t = &g.tensors[order[k]];
        if !is_matrix(&t.name, &t.dims) {
            return Ok(g.data(t)?.to_vec());
        }
        let cols = t.dims[0];
        let qw = imatrix.and_then(|(im, _)| im.get(&t.name));
        if let Some(qw) = qw
            && qw.len() != cols
        {
            return Err(bad(format!("imatrix entry for {} has {} values, the matrix {cols} columns", t.name, qw.len())));
        }
        let x = floats(&g, order[k])?;
        let format = match types[k] {
            GgmlType::Q4_K => return Ok(rows_into::<BlockQ4_K>(&x, cols, cols / QK_K, qw, quant::quantize_q4_k_row)),
            GgmlType::Q6_K => return Ok(rows_into::<BlockQ6_K>(&x, cols, cols / QK_K, qw, quant::quantize_q6_k_row)),
            GgmlType::Q8_0 => Format::Q8_0,
            GgmlType::Q4_0 => Format::Q4_0,
            GgmlType::Q5_0 => Format::Q5_0,
            GgmlType::F32 => Format::F32,
            ty => unreachable!("{ty:?} is not a target here"),
        };
        Ok(match Weight::from(x).convert_weighted(cols, format, qw) {
            Weight::F32(f) => bytes_of(&f[..]),
            Weight::Q8_0(b) => bytes_of(&b[..]),
            Weight::Q4_0(b) => bytes_of(&b[..]),
            Weight::Q5_0(b) => bytes_of(&b[..]),
            _ => unreachable!("convert_weighted makes F32, Q8_0, Q4_0 or Q5_0"),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{golden, host_golden};
    use std::path::PathBuf;

    /// A repo-relative reference file, if it is present here.
    fn file(rel: &str) -> Option<PathBuf> {
        match rel.strip_prefix("hostdata/") {
            Some(r) => host_golden(r),
            None => golden(rel.strip_prefix("testdata/").unwrap()),
        }
    }

    /// Quantizes llama.cpp's f32 conversion and compares the whole file
    /// with what `llama-quantize --pure` made of it (tools/fetch_models.sh).
    fn same_as_llama_quantize(f32_file: &str, reference: &str, recipe: Recipe, imatrix: Option<&str>) {
        let (Some(input), Some(reference)) = (file(f32_file), file(reference)) else { return };
        let im = match imatrix {
            Some(p) => match file(p) {
                Some(f) => Some((Imatrix::open(f).unwrap(), p)),
                None => return,
            },
            None => None,
        };
        let out = std::env::temp_dir().join(format!("aether-quantize-{}-{}", std::process::id(), reference.file_name().unwrap().to_str().unwrap()));
        quantize_with(&input, &out, recipe, im.as_ref().map(|(im, p)| (im, *p))).unwrap();
        let (ours, theirs) = (std::fs::read(&out).unwrap(), std::fs::read(&reference).unwrap());
        std::fs::remove_file(&out).unwrap();
        assert_eq!(ours.len(), theirs.len(), "file sizes differ");
        if let Some(at) = ours.iter().zip(&theirs).position(|(a, b)| a != b) {
            panic!("{}: first differing byte at {at}", reference.display());
        }
    }

    #[test]
    fn writes_llama_quantize_files() {
        let f32 = "hostdata/gguf/smollm2-135m-f32.gguf";
        same_as_llama_quantize(f32, "testdata/gguf/smollm2-135m-q8_0.gguf", Recipe::Pure(Format::Q8_0), None);
        same_as_llama_quantize(f32, "testdata/gguf/smollm2-135m-q4_0.gguf", Recipe::Pure(Format::Q4_0), None);
        same_as_llama_quantize(f32, "testdata/gguf/smollm2-135m-q5_0.gguf", Recipe::Pure(Format::Q5_0), None);
    }

    #[test]
    fn writes_llama_quantize_files_with_imatrix() {
        let im = Some("hostdata/imatrix/smollm2-135m.imatrix.gguf");
        let f32 = "hostdata/gguf/smollm2-135m-f32.gguf";
        same_as_llama_quantize(f32, "hostdata/imatrix/smollm2-135m-q4_0-im.gguf", Recipe::Pure(Format::Q4_0), im);
        same_as_llama_quantize(f32, "hostdata/imatrix/smollm2-135m-q5_0-im.gguf", Recipe::Pure(Format::Q5_0), im);
    }

    /// llama-quantize's Q4_K_M mix (tools/fetch_models.sh), without and
    /// with the imatrix: every tensor's format and bytes.
    #[test]
    fn writes_llama_quantize_q4_k_m() {
        let f32 = "hostdata/gguf/smollm2-135m-f32.gguf";
        same_as_llama_quantize(f32, "testdata/gguf/smollm2-135m-q4_k_m.gguf", Recipe::Q4KM, None);
        same_as_llama_quantize(f32, "hostdata/imatrix/smollm2-135m-q4_k_m-im.gguf", Recipe::Q4KM, Some("hostdata/imatrix/smollm2-135m.imatrix.gguf"));
    }
}
