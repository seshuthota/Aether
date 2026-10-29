//! Importance matrices from llama.cpp's `llama-imatrix`: for each matrix,
//! the mean of x² per input column over a calibration text, where x is the
//! activation that matrix multiplies. A column the model leans on hard gets
//! a large entry, and the quantizers spend their accuracy there
//! ([`crate::quant::quantize_q4_0_row`] and friends).
//!
//! The file is a GGUF (`general.type = "imatrix"`) with two tensors per
//! matrix, named after the matrix (`blk.0.attn_q.weight`):
//!
//! ```text
//! <name>.in_sum2   f32 [cols, n_mat]   sum of x² per column
//! <name>.counts    f32 [n_mat]         how many activations went in
//! ```
//!
//! `n_mat` is 1 except for stacked expert matrices. The entry used for
//! quantizing is `in_sum2 / counts`, as `llama-quantize` computes it.

use super::gguf::{GgmlType, Gguf, Value};
use crate::model::transformer::{Observer, Transformer};
use std::collections::HashMap;
use std::io;
use std::path::Path;

pub struct Imatrix {
    entries: HashMap<String, Vec<f32>>,
    /// How many chunks of calibration text went in.
    pub chunks: u64,
    /// The calibration files, as `llama-imatrix` recorded them.
    pub datasets: Vec<String>,
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn f32s(g: &Gguf, name: &str) -> io::Result<Vec<f32>> {
    let t = g.tensor(name).ok_or_else(|| bad(format!("missing {name}")))?;
    if t.ty != GgmlType::F32 {
        return Err(bad(format!("{name} is {:?}, expected F32", t.ty)));
    }
    Ok(g.data(t)?.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect())
}

impl Imatrix {
    /// Reads an imatrix GGUF (the format `llama-imatrix` writes by default;
    /// the older `.dat` format is not read).
    pub fn open(path: impl AsRef<Path>) -> io::Result<Imatrix> {
        let g = Gguf::open(path)?;
        if g.get("general.type").and_then(Value::as_str) != Some("imatrix") {
            return Err(bad("not an imatrix GGUF (general.type is not \"imatrix\")".into()));
        }
        let mut entries = HashMap::new();
        for t in &g.tensors {
            let Some(name) = t.name.strip_suffix(".in_sum2") else { continue };
            let sums = f32s(&g, &t.name)?;
            let counts = f32s(&g, &format!("{name}.counts"))?;
            let cols = t.dims[0];
            if sums.len() != cols * counts.len() {
                return Err(bad(format!("{name}: {} sums for {} counts of {cols} columns", sums.len(), counts.len())));
            }
            let mut e = Vec::with_capacity(sums.len());
            for (s, &n) in sums.chunks_exact(cols).zip(&counts) {
                // A matrix that saw no input during calibration: no
                // preference between columns.
                if n > 0.0 { e.extend(s.iter().map(|&v| v / n)) } else { e.extend(std::iter::repeat_n(1.0, cols)) }
            }
            entries.insert(name.to_string(), e);
        }
        if entries.is_empty() {
            return Err(bad("no .in_sum2 tensors in the imatrix".into()));
        }
        let chunks = g.get("imatrix.chunk_count").and_then(Value::as_u64).unwrap_or(0);
        let datasets = g.get("imatrix.datasets").and_then(Value::as_array).unwrap_or(&[]).iter().filter_map(|v| v.as_str().map(String::from)).collect();
        Ok(Imatrix { entries, chunks, datasets })
    }

    /// The importance of each input column of matrix `name` (a llama.cpp
    /// tensor name such as `blk.3.ffn_down.weight`), if it was recorded.
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.entries.get(name).map(Vec::as_slice)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Computes an importance matrix the way `llama-imatrix -c n_ctx` does:
    /// `tokens` cut into `n_chunks` chunks of `n_ctx` (the first token of
    /// each replaced by `bos` if given), each run through the model from
    /// position 0, and x² summed per input column of every layer matrix
    /// over every token.
    pub fn collect(model: &mut Transformer, tokens: &[usize], bos: Option<usize>, n_ctx: usize, n_chunks: usize) -> Imatrix {
        let mut c = Collector::default();
        let chunks = (tokens.len() / n_ctx).min(n_chunks);
        for ch in 0..chunks {
            let mut chunk = tokens[ch * n_ctx..(ch + 1) * n_ctx].to_vec();
            if let Some(bos) = bos {
                chunk[0] = bos;
            }
            model.forward_batch_traced(&chunk, 0, &mut c);
        }
        let mut entries = HashMap::new();
        for ((tap, l), (sums, n)) in c.sums {
            let e: Vec<f32> = sums.iter().map(|&s| (s / n as f64) as f32).collect();
            // The matrices each tap feeds (q, k and v share their input,
            // as do gate and up).
            let names: &[&str] = match tap {
                "attn_norm" => &["attn_q", "attn_k", "attn_v"],
                "attn" => &["attn_output"],
                "ffn_norm" => &["ffn_gate", "ffn_up"],
                _ => &["ffn_down"],
            };
            for m in names {
                entries.insert(format!("blk.{l}.{m}.weight"), e.clone());
            }
        }
        Imatrix { entries, chunks: chunks as u64, datasets: Vec::new() }
    }
}

/// Sums of x² per column (f64) and the number of vectors, per (tap, layer),
/// for the taps that are a matrix's input.
#[derive(Default)]
struct Collector {
    sums: HashMap<(&'static str, usize), (Vec<f64>, u64)>,
}

impl Observer for Collector {
    fn tap(&mut self, name: &str, layer: Option<usize>, _: usize, data: &[f32]) {
        let Some(l) = layer else { return };
        let tap = match name {
            "attn_norm" => "attn_norm",
            "attn" => "attn",
            "ffn_norm" => "ffn_norm",
            "swiglu" => "swiglu",
            _ => return,
        };
        let (s, n) = self.sums.entry((tap, l)).or_insert_with(|| (vec![0.0; data.len()], 0));
        for (s, &x) in s.iter_mut().zip(data) {
            *s += (x * x) as f64;
        }
        *n += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::transformer::Weights;
    use crate::quant::{self, BlockQ4_0, BlockQ4_K, BlockQ5_0, BlockQ6_K, BlockQ8_0, QK, QK_K};
    use crate::testutil::host_golden;
    use crate::weight::{Format, Weight};

    const MODELS: [&str; 2] = ["smollm2-135m", "qwen2.5-0.5b"];

    /// The files llama.cpp made for model `m` (see
    /// results/phase-6.md "Importance matrices"): its f32 GGUF, the
    /// imatrix, and `llama-quantize --imatrix` output for `q`.
    fn files(m: &str, q: &str) -> Option<(std::path::PathBuf, Imatrix, std::path::PathBuf)> {
        let f32 = host_golden(&format!("gguf/{m}-f32.gguf"))?;
        let im = Imatrix::open(host_golden(&format!("imatrix/{m}.imatrix.gguf"))?).unwrap();
        Some((f32, im, host_golden(&format!("imatrix/{m}-{q}-im.gguf"))?))
    }

    fn bytes_of(w: &Weight) -> &[u8] {
        fn b<T>(s: &[T]) -> &[u8] {
            // SAFETY: plain #[repr(C)] blocks of integers, read as bytes.
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
        }
        match w {
            Weight::Q8_0(s) => b(s),
            Weight::Q4_0(s) => b(s),
            Weight::Q5_0(s) => b(s),
            other => panic!("unexpected {}", other.format_name()),
        }
    }

    fn matrices(w: &Weights) -> Vec<(String, &Weight)> {
        let mut v = vec![("token_embd".to_string(), &w.token_embedding)];
        for (i, l) in w.layers.iter().enumerate() {
            for (n, m) in [("attn_q", &l.wq), ("attn_k", &l.wk), ("attn_v", &l.wv), ("attn_output", &l.wo), ("ffn_gate", &l.w_gate), ("ffn_down", &l.w_down), ("ffn_up", &l.w_up)] {
                v.push((format!("blk.{i}.{n}"), m));
            }
        }
        v.extend(w.classifier.iter().map(|c| ("output".to_string(), c)));
        v
    }

    #[test]
    fn reads_llama_imatrix_files() {
        let Some((_, im, _)) = files("smollm2-135m", "q4_0") else { return };
        // 30 layers × 7 matrices; llama-imatrix records no embedding.
        assert_eq!(im.len(), 210);
        assert_eq!(im.chunks, 100);
        assert_eq!(im.get("blk.0.attn_q.weight").unwrap().len(), 576);
        assert_eq!(im.get("blk.0.ffn_down.weight").unwrap().len(), 1536);
        assert!(im.get("token_embd.weight").is_none());
        // q, k and v multiply the same input, so they share one entry.
        assert_eq!(im.get("blk.3.attn_q.weight"), im.get("blk.3.attn_v.weight"));
    }

    /// `-q q4_0 / q5_0 --imatrix` gives exactly `llama-quantize --pure
    /// --imatrix`'s bytes for every matrix, and the imatrix does change
    /// them (against the same conversion without it).
    #[test]
    fn imatrix_conversion_matches_llama_quantize_pure() {
        for m in MODELS {
            for (q, format) in [("q4_0", Format::Q4_0), ("q5_0", Format::Q5_0)] {
                let Some((f32, im, want)) = files(m, q) else { return };
                let (c, w, _) = crate::model::gguf::load(&f32, 64).unwrap();
                let ours = w.convert(&c, format, Some(&im)).unwrap();
                let plain = crate::model::gguf::load(&f32, 64).unwrap().1.convert(&c, format, None).unwrap();
                let theirs = crate::model::gguf::load(&want, 64).unwrap().1;
                let mut changed = 0;
                for (((name, a), (_, b)), (_, p)) in matrices(&ours).into_iter().zip(matrices(&theirs)).zip(matrices(&plain)) {
                    assert!(bytes_of(a) == bytes_of(b), "{m} {q} {name}");
                    changed += (bytes_of(a) != bytes_of(p)) as usize;
                }
                // Every layer matrix; the embedding has no entry.
                assert_eq!(changed, 7 * c.n_layers, "{m} {q}");
            }
        }
    }

    /// Rows of `x` (each `cols` long) quantized by `f` into blocks of `B`,
    /// as bytes.
    fn quantize<B: Copy>(x: &[f32], cols: usize, per_row: usize, qw: Option<&[f32]>, f: impl Fn(&[f32], Option<&[f32]>, &mut [B])) -> Vec<u8> {
        // SAFETY: B is a plain #[repr(C)] struct of integers; zero is valid.
        let mut out = vec![unsafe { std::mem::zeroed::<B>() }; x.len() / cols * per_row];
        for (row, b) in x.chunks_exact(cols).zip(out.chunks_exact_mut(per_row)) {
            f(row, qw, b);
        }
        // SAFETY: as above, read back as bytes.
        unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, std::mem::size_of_val(&out[..])) }.to_vec()
    }

    /// Every quantized tensor of llama.cpp's Q4_K_M mix made with the
    /// imatrix (Q4_K, Q6_K, Q5_0 and Q8_0, per tensor as llama.cpp chose):
    /// our quantizer for that type, given the same f32 rows and imatrix
    /// entry, writes the same bytes.
    #[test]
    fn k_quant_mix_with_imatrix_matches_llama_quantize() {
        use super::super::gguf::{GgmlType, Gguf};
        for m in MODELS {
            let Some((f32, im, want)) = files(m, "q4_k_m") else { return };
            let (src, dst) = (Gguf::open(&f32).unwrap(), Gguf::open(&want).unwrap());
            let mut seen = std::collections::BTreeMap::<String, usize>::new();
            for t in &dst.tensors {
                let s = src.tensor(&t.name).unwrap();
                assert_eq!(s.ty, GgmlType::F32);
                let x: Vec<f32> = src.data(s).unwrap().as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
                let cols = t.dims[0];
                let qw = im.get(&t.name).filter(|q| q.len() == cols);
                let got = match t.ty {
                    GgmlType::F32 => continue,
                    GgmlType::Q8_0 => quantize::<BlockQ8_0>(&x, cols, cols / QK, qw, |r, _, b| quant::quantize_q8_0(r, b)),
                    GgmlType::Q4_0 => quantize::<BlockQ4_0>(&x, cols, cols / QK, qw, quant::quantize_q4_0_row),
                    GgmlType::Q5_0 => quantize::<BlockQ5_0>(&x, cols, cols / QK, qw, quant::quantize_q5_0_row),
                    GgmlType::Q4_K => quantize::<BlockQ4_K>(&x, cols, cols / QK_K, qw, quant::quantize_q4_k_row),
                    GgmlType::Q6_K => quantize::<BlockQ6_K>(&x, cols, cols / QK_K, qw, quant::quantize_q6_k_row),
                    other => panic!("{}: {other:?}", t.name),
                };
                assert!(got == dst.data(t).unwrap(), "{m} {} ({:?}, imatrix {})", t.name, t.ty, qw.is_some());
                *seen.entry(format!("{:?}{}", t.ty, if qw.is_some() { "+im" } else { "" })).or_default() += 1;
            }
            eprintln!("{m}: {seen:?}");
            assert!(seen.contains_key("Q4_K+im") && seen.contains_key("Q6_K+im"), "{m}: {seen:?}");
        }
    }

    /// Our own imatrix, collected over the same 10 chunks of calibration
    /// text as `llama-imatrix --chunks 10` on the f32 GGUF, agrees with
    /// llama.cpp's entry by entry (llama.cpp sums in f32 and computes in a
    /// different order, hence a tolerance rather than equality).
    #[test]
    fn collected_imatrix_matches_llama_imatrix() {
        let (Some(f32), Some(want), Some(calib)) = (host_golden("gguf/smollm2-135m-f32.gguf"), host_golden("imatrix/smollm2-135m-10.imatrix.gguf"), host_golden("imatrix/calib.txt")) else { return };
        let want = Imatrix::open(want).unwrap();
        let cores: Vec<usize> = (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect();
        let mut s = crate::session::Session::open(&f32, None, 512, Format::F32, Some(&cores)).unwrap();
        let tokens = s.text.encode(&std::fs::read_to_string(calib).unwrap());
        let ours = Imatrix::collect(&mut s.model, &tokens, s.text.chunk_bos(), 512, 10);
        assert_eq!((ours.len(), ours.chunks), (want.len(), want.chunks));
        // Relative difference of every entry, and its sign.
        let mut rel = Vec::new();
        for (name, w) in &want.entries {
            let o = ours.get(name).unwrap_or_else(|| panic!("no {name}"));
            assert_eq!(o.len(), w.len(), "{name}");
            rel.extend(o.iter().zip(w).map(|(&a, &b)| ((a - b) / b.abs().max(1e-6)) as f64));
        }
        let bias = rel.iter().sum::<f64>() / rel.len() as f64;
        let mut abs: Vec<f64> = rel.iter().map(|r| r.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let (median, p99, max) = (abs[abs.len() / 2], abs[abs.len() * 99 / 100], abs[abs.len() - 1]);
        eprintln!("relative difference: median {median:.1e}, 99th percentile {p99:.1e}, max {max:.1e}, mean signed {bias:.1e}");
        // Measured: median 2.4e-5, p99 2.7e-4, max 1.7e-3 (blk.19.ffn_down,
        // a column whose sum one large activation dominates), no bias.
        assert!(median < 1e-4 && p99 < 1e-3 && max < 5e-3 && bias.abs() < 1e-4);
    }
}
