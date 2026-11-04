//! Loader for GGUF model files (llama.cpp's format): `llama`, `qwen2`,
//! `lfm3` and `gemma4` architectures, tensors in f32 / f16 / bf16 / Q8_0 /
//! Q4_0 / Q5_0 / Q4_K / Q6_K (and Q5_K for Gemma 4's per-layer embedding
//! table, which is only looked up).
//!
//! Hyperparameters come from metadata keys prefixed with the architecture
//! name (`qwen2.embedding_length`, `qwen2.attention.head_count_kv`, …).
//! Tensor names follow llama.cpp: `token_embd`, `blk.N.attn_q`,
//! `blk.N.ffn_down`, `output_norm`, `output` (absent when tied).
//!
//! **RoPE layout depends on the architecture.** llama.cpp's converter
//! permutes the rows of Q and K for `llama` models so that the cheaper
//! adjacent-pair rotation gives the same result as Hugging Face's
//! rotate-half on the original rows; `qwen2` files are not permuted and use
//! rotate-half ("NeoX") directly. Using the wrong one for a file still runs —
//! and produces garbage.
//!
//! f32 and quantized matrices are used **in place** from the memory map (no
//! copy: the kernel can evict and re-read those pages under memory
//! pressure); f16/bf16 matrices are widened to owned f32. Norm gains and
//! biases (small, always f32) are copied.

use super::config::{Activation, Config, RopeLayout};
use super::transformer::{AttentionWeights, ConvWeights, LayerWeights, Mixer, PleTable, PleWeights, Weights};
use crate::aligned::AlignedBuf;
use crate::format::gguf::{GgmlType, Gguf, Value};
use crate::format::safetensors::{bf16_to_f32, f16_to_f32};
use crate::weight::{Floats, Mapped, Rows, Store, Weight};
use std::io;
use std::path::Path;

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Opens a GGUF file and builds the config and weights from it.
pub fn load(path: impl AsRef<Path>, max_seq: usize) -> io::Result<(Config, Weights, Gguf)> {
    let g = Gguf::open(path.as_ref())?;
    let arch = g.get("general.architecture").and_then(Value::as_str).ok_or_else(|| bad("no general.architecture".into()))?.to_string();
    let rope_layout = match arch.as_str() {
        "llama" => RopeLayout::AdjacentPairs,
        "qwen2" | "lfm3" | "gemma4" => RopeLayout::Halves,
        other => return Err(bad(format!("architecture {other:?} is not supported (llama, qwen2, lfm3, gemma4)"))),
    };
    let key = |k: &str| g.get(&format!("{arch}.{k}"));
    let num = |k: &str| key(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| bad(format!("missing {arch}.{k}")));
    let arr = |k: &str| key(k).and_then(Value::as_array);
    let lfm3 = arch == "lfm3";
    let gemma4 = arch == "gemma4";
    let n_heads = num("attention.head_count")?;
    let dim = num("embedding_length")?;
    let n_layers = num("block_count")?;
    let embd = g.tensor("token_embd.weight").ok_or_else(|| bad("no token_embd.weight".into()))?;
    let head_dim = key("attention.key_length").and_then(Value::as_u64).map(|v| v as usize).unwrap_or(dim / n_heads);
    let kv_arr: Option<Vec<usize>> = arr("attention.head_count_kv").map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect());
    let n_kv_heads = kv_arr
        .as_ref()
        .and_then(|a| a.iter().copied().find(|&v| v > 0))
        .or_else(|| key("attention.head_count_kv").and_then(Value::as_u64).map(|v| v as usize))
        .unwrap_or(n_heads);
    let window = key("attention.sliding_window").and_then(Value::as_u64).unwrap_or(256) as usize;
    let sliding: Option<Vec<bool>> = arr("attention.sliding_window_pattern").map(|a| a.iter().map(|v| matches!(v, Value::Bool(true))).collect());
    let theta_full = key("rope.freq_base").and_then(Value::as_f64).unwrap_or(10000.0) as f32;
    let theta_swa = key("rope.freq_base_swa").and_then(Value::as_f64).unwrap_or(theta_full as f64) as f32;
    let pairs_full = match g.tensor("rope_freqs.weight") {
        Some(t) if t.dims == [head_dim / 2] => {
            let factors = widen(g.data(t)?, t.ty, head_dim / 2);
            factors.iter().filter(|&&f| f < 1e29).count()
        }
        _ => head_dim / 2,
    };
    // Gemma 4: sliding layers have their own head size; the last
    // `shared_kv_layers` layers read the cache of the last layer of the same
    // kind before them that has one (llama.cpp's `kv_source`).
    let head_dim_swa = key("attention.key_length_swa").and_then(Value::as_u64).map_or(head_dim, |v| v as usize);
    let is_sliding = |l: usize| (lfm3 || gemma4) && sliding.as_ref().and_then(|a| a.get(l)).copied().unwrap_or(false);
    let shared = key("attention.shared_kv_layers").and_then(Value::as_u64).unwrap_or(0) as usize;
    if shared > 0 && n_layers < shared + 2 {
        return Err(bad(format!("{shared} shared KV layers of {n_layers}")));
    }
    let kv_from = n_layers - shared;
    let ffn: Vec<usize> = match arr("feed_forward_length") {
        Some(a) => a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect(),
        None => vec![num("feed_forward_length")?; n_layers],
    };
    if ffn.len() != n_layers || ffn.contains(&0) {
        return Err(bad(format!("feed_forward_length {ffn:?} for {n_layers} layers")));
    }
    let conv_width = key("shortconv.l_cache").and_then(Value::as_u64).map(|v| v as usize).unwrap_or(7);
    let layers = (0..n_layers)
        .map(|l| {
            let is_conv = lfm3 && kv_arr.as_ref().is_some_and(|a| a.get(l).copied().unwrap_or(0) == 0);
            if is_conv {
                crate::model::config::LayerKind::Conv { width: conv_width }
            } else {
                let sl = is_sliding(l);
                let (theta, pairs) = if sl { (theta_swa, head_dim_swa / 2) } else { (theta_full, pairs_full) };
                crate::model::config::LayerKind::Attention {
                    window: if sl { Some(window) } else { None },
                    rope: crate::model::config::Rope { theta, pairs, layout: rope_layout },
                }
            }
        })
        .collect::<Vec<_>>();
    let config = Config {
        dim,
        hidden_dim: ffn.iter().copied().max().unwrap_or(0),
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size: embd.dims[1],
        seq_len: num("context_length")?.min(max_seq),
        rope_theta: theta_full,
        rope_layout,
        norm_eps: key("attention.layer_norm_rms_epsilon").and_then(Value::as_f64).ok_or_else(|| bad("missing rms epsilon".into()))? as f32,
        layers,
        qk_norm: lfm3 || gemma4,
        output_gate: lfm3,
        sandwich_norm: lfm3 || gemma4,
        head_dims: if gemma4 { (0..n_layers).map(|l| if is_sliding(l) { head_dim_swa } else { head_dim }).collect() } else { Vec::new() },
        hidden_dims: if ffn.iter().any(|&f| f != ffn[0]) { ffn.clone() } else { Vec::new() },
        kv_source: if shared > 0 { (0..n_layers).map(|l| if l < kv_from { l } else if is_sliding(l) { kv_from - 2 } else { kv_from - 1 }).collect() } else { Vec::new() },
        attn_scale: if gemma4 { Some(1.0) } else { None },
        embed_scale: if gemma4 { (dim as f32).sqrt() } else { 1.0 },
        logit_softcap: key("final_logit_softcapping").and_then(Value::as_f64).map(|v| v as f32).filter(|&v| v > 0.0),
        v_norm: gemma4,
        activation: if gemma4 { Activation::GeluTanh } else { Activation::SwiGlu },
        ple_dim: key("embedding_length_per_layer_input").and_then(Value::as_u64).unwrap_or(0) as usize,
    };
    let c = &config;
    if gemma4 && (0..c.n_layers).any(|l| c.kv_source_of(l) != l && c.window(l) != c.window(c.kv_source_of(l))) {
        return Err(bad("a shared-KV layer reads a layer of the other kind".into()));
    }
    let dim = c.dim;

    // A matrix `[rows, cols]` (GGUF dims [cols, rows]), in place if possible.
    let matrix = |name: &str, rows: usize, cols: usize| -> io::Result<Weight> {
        let t = g.tensor(name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        if t.dims != [cols, rows] {
            return Err(bad(format!("{name}: dims {:?}, expected [{cols}, {rows}]", t.dims)));
        }
        let n = rows * cols;
        Ok(match t.ty {
            GgmlType::F32 => Weight::F32(Floats::Mapped(Mapped::new(g.map.clone(), t.offset, n)?)),
            GgmlType::Q8_0 => Weight::Q8_0(Store::Mapped(Mapped::new(g.map.clone(), t.offset, n / 32)?)),
            GgmlType::Q4_0 => Weight::Q4_0(Store::Mapped(Mapped::new(g.map.clone(), t.offset, n / 32)?)),
            GgmlType::Q5_0 => Weight::Q5_0(Store::Mapped(Mapped::new(g.map.clone(), t.offset, n / 32)?)),
            GgmlType::Q4_K => Weight::Q4_K(Store::Mapped(Mapped::new(g.map.clone(), t.offset, n / 256)?)),
            GgmlType::Q6_K => Weight::Q6_K(Store::Mapped(Mapped::new(g.map.clone(), t.offset, n / 256)?)),
            GgmlType::F16 | GgmlType::BF16 => Weight::from(widen(g.data(t)?, t.ty, n)),
            GgmlType::Q5_K => return Err(bad(format!("{name}: Q5_K is only supported for lookup tables"))),
            GgmlType::Other(id) => return Err(bad(format!("{name}: ggml type {id} is not supported"))),
        })
    };
    // A table read a row at a time, left in the mapped file.
    let rows = |name: &str, rows: usize, cols: usize| -> io::Result<Rows> {
        let t = g.tensor(name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        if t.dims != [cols, rows] {
            return Err(bad(format!("{name}: dims {:?}, expected [{cols}, {rows}]", t.dims)));
        }
        let (n, m) = (rows * cols, g.map.clone());
        Ok(match t.ty {
            GgmlType::Q4_0 => Rows::Q4_0(Mapped::new(m, t.offset, n / 32)?),
            GgmlType::Q8_0 => Rows::Q8_0(Mapped::new(m, t.offset, n / 32)?),
            GgmlType::Q5_K => Rows::Q5_K(Mapped::new(m, t.offset, n / 256)?),
            GgmlType::BF16 => Rows::BF16(Mapped::new(m, t.offset, n)?),
            GgmlType::F32 => Rows::F32(Mapped::new(m, t.offset, n)?),
            other => return Err(bad(format!("{name}: {other:?} is not supported for a lookup table"))),
        })
    };
    // A small f32 vector, copied (norm gains, biases).
    let vector = |name: &str, n: usize| -> io::Result<Option<AlignedBuf>> {
        let Some(t) = g.tensor(name) else { return Ok(None) };
        if t.dims != [n] {
            return Err(bad(format!("{name}: dims {:?}, expected [{n}]", t.dims)));
        }
        Ok(Some(widen(g.data(t)?, t.ty, n)))
    };
    let required = |name: &str, n: usize| vector(name, n)?.ok_or_else(|| bad(format!("missing tensor {name}")));
    let optional_matrix = |name: &str, rows: usize, cols: usize| -> io::Result<Option<Weight>> {
        if g.tensor(name).is_some() { matrix(name, rows, cols).map(Some) } else { Ok(None) }
    };

    let layers = (0..c.n_layers)
        .map(|l| {
            let p = format!("blk.{l}");
            let (hd, qd, kv_dim, hidden, own) = (c.head_dim_of(l), c.q_dim(l), c.kv_dim_of(l), c.hidden_of(l), c.owns_kv(l));
            // Shared-KV layers compute no K/V; files that still carry those
            // tensors (unsloth's Gemma 4 Q4_K_M) have them ignored.
            let own_matrix = |name: &str| if own { matrix(name, kv_dim, dim).map(Some) } else { Ok(None) };
            let own_vector = |name: &str, n: usize| if own { vector(name, n) } else { Ok(None) };
            let mixer = match c.kind(l) {
                crate::model::config::LayerKind::Attention { .. } => Mixer::Attention(AttentionWeights {
                    wq: matrix(&format!("{p}.attn_q.weight"), qd, dim)?,
                    wk: own_matrix(&format!("{p}.attn_k.weight"))?,
                    wv: own_matrix(&format!("{p}.attn_v.weight"))?,
                    wo: matrix(&format!("{p}.attn_output.weight"), dim, qd)?,
                    bq: vector(&format!("{p}.attn_q.bias"), qd)?,
                    bk: own_vector(&format!("{p}.attn_k.bias"), kv_dim)?,
                    bv: own_vector(&format!("{p}.attn_v.bias"), kv_dim)?,
                    q_norm: vector(&format!("{p}.attn_q_norm.weight"), hd)?,
                    k_norm: own_vector(&format!("{p}.attn_k_norm.weight"), hd)?,
                    gate: optional_matrix(&format!("{p}.attn_gate.weight"), n_heads, dim)?,
                }),
                crate::model::config::LayerKind::Conv { width } => Mixer::Conv(ConvWeights {
                    in_proj: matrix(&format!("{p}.shortconv.in_proj.weight"), 3 * dim, dim)?,
                    conv: matrix(&format!("{p}.shortconv.conv.weight"), dim, width)?,
                    out_proj: matrix(&format!("{p}.shortconv.out_proj.weight"), dim, dim)?,
                }),
            };
            Ok(LayerWeights {
                attn_norm: required(&format!("{p}.attn_norm.weight"), dim)?,
                mixer,
                post_attn_norm: vector(&format!("{p}.post_attention_norm.weight"), dim)?,
                ffn_norm: required(&format!("{p}.ffn_norm.weight"), dim)?,
                post_ffn_norm: vector(&format!("{p}.post_ffw_norm.weight"), dim)?,
                w_gate: matrix(&format!("{p}.ffn_gate.weight"), hidden, dim)?,
                w_up: matrix(&format!("{p}.ffn_up.weight"), hidden, dim)?,
                w_down: matrix(&format!("{p}.ffn_down.weight"), dim, hidden)?,
                ple: if c.ple_dim > 0 {
                    Some(PleWeights {
                        inp_gate: matrix(&format!("{p}.inp_gate.weight"), c.ple_dim, dim)?,
                        proj: matrix(&format!("{p}.proj.weight"), dim, c.ple_dim)?,
                        post_norm: required(&format!("{p}.post_norm.weight"), dim)?,
                    })
                } else {
                    None
                },
                out_scale: vector(&format!("{p}.layer_output_scale.weight"), 1)?.map(|v| v[0]),
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    let weights = Weights {
        token_embedding: matrix("token_embd.weight", c.vocab_size, dim)?,
        layers,
        final_norm: required("output_norm.weight", dim)?,
        classifier: if g.tensor("output.weight").is_some() { Some(matrix("output.weight", c.vocab_size, dim)?) } else { None },
        ple: if c.ple_dim > 0 {
            let n = c.n_layers * c.ple_dim;
            Some(PleTable {
                table: rows("per_layer_token_embd.weight", c.vocab_size, n)?,
                proj: matrix("per_layer_model_proj.weight", n, dim)?,
                proj_norm: required("per_layer_proj_norm.weight", c.ple_dim)?,
            })
        } else {
            None
        },
    };
    Ok((config, weights, g))
}

/// f32 / f16 / bf16 bytes → an owned f32 buffer.
fn widen(bytes: &[u8], ty: GgmlType, n: usize) -> AlignedBuf {
    let mut out = AlignedBuf::zeros(n);
    match ty {
        GgmlType::F32 => out.iter_mut().zip(bytes.as_chunks::<4>().0).for_each(|(o, b)| *o = f32::from_le_bytes(*b)),
        GgmlType::F16 => out.iter_mut().zip(bytes.as_chunks::<2>().0).for_each(|(o, b)| *o = f16_to_f32(u16::from_le_bytes(*b))),
        GgmlType::BF16 => out.iter_mut().zip(bytes.as_chunks::<2>().0).for_each(|(o, b)| *o = bf16_to_f32(u16::from_le_bytes(*b))),
        _ => unreachable!("only called for float types"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Transformer;
    use crate::model::reference::check_against_transformers;
    use crate::testutil::{host_golden, max_errors};
    use crate::format::safetensors::SafeTensors;
    use crate::model::Observer;
    use std::collections::HashMap;

    fn model(rel: &str) -> Option<Transformer> {
        let (config, weights, _) = load(host_golden(rel)?, 256).unwrap();
        Some(Transformer::new(config, weights))
    }

    /// LFM3's mixed stack: 13 short-conv layers, 6 full and 5 sliding
    /// attention layers, QK-norm, per-head gate, sandwich norms and the
    /// zero-tail RoPE factors.
    #[test]
    fn lfm3_q4km_loads() {
        let Some(p) = crate::testutil::golden("gguf/LFM3-1.7B-Q4_K_M.gguf") else { return };
        let (c, w, _) = load(p, 2048).unwrap();
        assert_eq!((c.n_layers, c.dim, c.head_dim, c.n_heads, c.n_kv_heads), (24, 2048, 128, 16, 4));
        assert_eq!(c.kind(0), crate::model::config::LayerKind::Conv { width: 7 });
        assert!(c.is_attention(2) && c.is_attention(4));
        assert_eq!(c.window(2), None);
        assert_eq!(c.window(4), Some(256));
        assert_eq!(c.rope(2).pairs, 32);
        assert_eq!(c.rope(4).pairs, 64);
        assert_eq!(c.rope(2).theta, 1e6);
        assert_eq!(c.rope(4).theta, 1e4);
        assert!(c.qk_norm && c.output_gate && c.sandwich_norm);
        assert!(w.layers[0].conv().is_some());
        let a = w.layers[2].attention().unwrap();
        assert_eq!(a.q_norm.as_ref().unwrap().len(), 128);
        assert_eq!(a.k_norm.as_ref().unwrap().len(), 128);
        assert!(a.gate.is_some());
        assert!(w.layers[0].post_attn_norm.is_some() && w.layers[0].post_ffn_norm.is_some());
    }


    #[derive(Default)]
    struct Collect(HashMap<(String, i32, usize), Vec<f32>>);
    impl Observer for Collect {
        fn tap(&mut self, name: &str, layer: Option<usize>, pos: usize, data: &[f32]) {
            self.0.insert((name.to_string(), layer.map_or(-1, |l| l as i32), pos), data.to_vec());
        }
    }

    fn ref_tensor(st: &SafeTensors, name: &str) -> (Vec<usize>, Vec<f32>) {
        let v = st.get(name).unwrap_or_else(|| panic!("reference has no {name}"));
        let mut all = vec![0.0; v.numel()];
        v.to_f32(&mut all);
        (v.shape.to_vec(), all)
    }

    fn ref_row(st: &SafeTensors, name: &str, row: usize, width: usize) -> Vec<f32> {
        let (shape, all) = ref_tensor(st, name);
        assert!(row < shape[0] && shape[1] == width);
        all[row * width..(row + 1) * width].to_vec()
    }

    fn check_logits(label: &str, ours: &[f32], want: &[f32]) {
        let (_, rel) = max_errors(ours, want);
        assert!(rel < 1e-3, "{label}: logits relative error {rel:.2e}");
    }

    /// LFM3 against the pinned HF repository (f32, eager): every layer's
    /// residual after a 29-token prompt, full short logits, selected logits
    /// from a 320-token prompt (past the 256-position sliding window), and
    /// 16 greedy tokens.
    #[test]
    fn lfm3_matches_transformers() {
        let (Some(p), Some(refp)) = (host_golden("gguf/LFM3-1.7B-BF16.gguf"), host_golden("lfm3/ref.safetensors")) else { return };
        let (c, w, _) = load(p, 320).unwrap();
        let mut t = Transformer::new(c.clone(), w);
        let st = SafeTensors::open(refp).unwrap();
        let vocab = c.vocab_size;

        let (_, ids) = ref_tensor(&st, "short_ids");
        let ids: Vec<usize> = ids.into_iter().map(|x| x as usize).collect();
        let (_, want) = ref_tensor(&st, "short_logits");
        let ours = t.forward_batch_logits(&ids, 0, 0).to_vec();
        for pos in 0..ids.len() {
            check_logits(&format!("short logits pos {pos}"), &ours[pos * vocab..(pos + 1) * vocab], &want[pos * vocab..(pos + 1) * vocab]);
        }

        let mut obs = Collect::default();
        t.forward_batch_traced(&ids, 0, &mut obs);
        for l in 0..c.n_layers {
            for pos in 0..ids.len() {
                let key = ("resid_ffn".to_string(), l as i32, pos);
                let got = obs.0.get(&key).unwrap_or_else(|| panic!("no tap {key:?}"));
                let want = ref_row(&st, &format!("L{l}.resid_ffn"), pos, c.dim);
                let (_, rel) = max_errors(got, &want);
                if rel >= 1e-3 {
                    let mut best = (usize::MAX, f32::INFINITY);
                    for ll in 0..c.n_layers {
                        let (_, r) = max_errors(got, &ref_row(&st, &format!("L{ll}.resid_ffn"), pos, c.dim));
                        if r < best.1 { best = (ll, r); }
                    }
                    println!("L{l} pos{pos} rel {rel:.3e}; best HF layer {:?}; got {:?} want {:?}", best, &got[..4], &want[..4]);
                }
                assert!(rel < 1e-3, "L{l}.resid_ffn pos {pos}: relative error {rel:.2e}");
            }
        }

        let (_, long) = ref_tensor(&st, "long_ids");
        let long: Vec<usize> = long.into_iter().map(|x| x as usize).collect();
        let (_, posv) = ref_tensor(&st, "long_positions");
        let (_, want_long) = ref_tensor(&st, "long_logits");
        let long_logits = t.forward_batch_logits(&long, 0, 0).to_vec();
        for (r, pv) in posv.iter().enumerate() {
            let pos = *pv as usize;
            check_logits(&format!("long logits pos {pos}"), &long_logits[pos * vocab..(pos + 1) * vocab], &want_long[r * vocab..(r + 1) * vocab]);
        }

        let (_, greedy) = ref_tensor(&st, "greedy_ids");
        let mut logits = t.forward_batch(&ids, 0).to_vec();
        for k in 0..16 {
            let next = crate::kernels::argmax(&logits);
            assert_eq!(next, greedy[ids.len() + k] as usize, "greedy token {k}");
            logits.copy_from_slice(t.forward(next, ids.len() + k));
        }
        println!("LFM3: 29 short + 9 long logits, every layer tap and 16 greedy tokens match HF");
    }


    /// Gemma 4 E2B's shapes, from both files: per-layer head sizes and FFN
    /// widths, shared KV (the Q4_K_M file's unused K/V in layers 15–34
    /// ignored), partial RoPE on full layers, and the per-layer embedding
    /// table left mapped in its own format.
    #[test]
    fn gemma4_loads() {
        for (file, table) in [("gguf/gemma-4-E2B-it-Q4_0.gguf", "q4_0"), ("gguf/gemma-4-E2B-it-Q4_K_M.gguf", "q5_k")] {
            let Some(p) = crate::testutil::golden(file) else { continue };
            let (c, w, _) = load(p, 1024).unwrap();
            assert_eq!((c.n_layers, c.dim, c.n_heads, c.n_kv_heads, c.vocab_size, c.ple_dim), (35, 1536, 8, 1, 262144, 256));
            let full = [4, 9, 14, 19, 24, 29, 34];
            for l in 0..35 {
                let is_full = full.contains(&l);
                assert_eq!(c.window(l), if is_full { None } else { Some(512) }, "layer {l}");
                assert_eq!(c.head_dim_of(l), if is_full { 512 } else { 256 }, "layer {l}");
                assert_eq!(c.hidden_of(l), if l < 15 { 6144 } else { 12288 }, "layer {l}");
                assert_eq!(c.kv_source_of(l), if l < 15 { l } else if is_full { 14 } else { 13 }, "layer {l}");
                assert_eq!(c.rope(l).pairs, if is_full { 64 } else { 128 }, "layer {l}");
                assert_eq!(c.rope(l).theta, if is_full { 1e6 } else { 1e4 }, "layer {l}");
                let a = w.layers[l].attention().unwrap();
                assert_eq!((a.wk.is_some(), a.wv.is_some(), a.k_norm.is_some()), (l < 15, l < 15, l < 15), "layer {l}");
                assert!(w.layers[l].ple.is_some() && w.layers[l].out_scale.is_some());
            }
            assert_eq!((c.attn_scale, c.logit_softcap, c.v_norm), (Some(1.0), Some(30.0), true));
            assert_eq!(c.embed_scale, (1536f32).sqrt());
            assert_eq!(w.ple.as_ref().unwrap().table.format_name(), table);
        }
    }

    /// Gemma 4 E2B (BF16 file) against transformers' f32 run
    /// (`tools/hf_dump_gemma4.py`): every layer's output and the per-layer
    /// inputs of a 30-token prompt, its logits by the batched and the
    /// one-token path, logits of a 600-token prompt around the 512 window,
    /// and a 16-token greedy continuation.
    #[test]
    fn gemma4_matches_transformers() {
        let (Some(p), Some(refp)) = (host_golden("gguf/gemma-4-E2B-it-BF16.gguf"), host_golden("gemma4/ref.safetensors")) else { return };
        let (c, w, _) = load(p, 640).unwrap();
        let mut t = Transformer::new(c.clone(), w);
        let st = SafeTensors::open(refp).unwrap();
        let vocab = c.vocab_size;
        let as_ids = |v: Vec<f32>| -> Vec<usize> { v.into_iter().map(|x| x as usize).collect() };

        let ids = as_ids(ref_tensor(&st, "short_ids").1);
        let (_, want) = ref_tensor(&st, "short_logits");
        let mut obs = Collect::default();
        t.forward_batch_traced(&ids, 0, &mut obs);
        let mut worst = 0f32;
        for pos in 0..ids.len() {
            let n = c.n_layers * c.ple_dim;
            let (_, rel) = max_errors(&obs.0[&("ple".to_string(), -1, pos)], &ref_row(&st, "ple", pos, n));
            assert!(rel < 1e-3, "ple pos {pos}: relative error {rel:.2e}");
            for l in 0..c.n_layers {
                let got = &obs.0[&("layer_out".to_string(), l as i32, pos)];
                let (_, rel) = max_errors(got, &ref_row(&st, &format!("L{l}.layer_out"), pos, c.dim));
                worst = worst.max(rel);
                assert!(rel < 1e-3, "L{l}.layer_out pos {pos}: relative error {rel:.2e}");
            }
        }
        let ours = t.forward_batch_logits(&ids, 0, 0).to_vec();
        for pos in 0..ids.len() {
            check_logits(&format!("short logits pos {pos}"), &ours[pos * vocab..(pos + 1) * vocab], &want[pos * vocab..(pos + 1) * vocab]);
        }
        for (pos, &tok) in ids.iter().enumerate() {
            let one = t.forward(tok, pos).to_vec();
            check_logits(&format!("one-token logits pos {pos}"), &one, &want[pos * vocab..(pos + 1) * vocab]);
        }

        let long = as_ids(ref_tensor(&st, "long_ids").1);
        let (_, posv) = ref_tensor(&st, "long_positions");
        let (_, want_long) = ref_tensor(&st, "long_logits");
        let long_logits = t.forward_batch_logits(&long, 0, 0).to_vec();
        for (r, pv) in posv.iter().enumerate() {
            let pos = *pv as usize;
            check_logits(&format!("long logits pos {pos}"), &long_logits[pos * vocab..(pos + 1) * vocab], &want_long[r * vocab..(r + 1) * vocab]);
        }

        let greedy = as_ids(ref_tensor(&st, "greedy_ids").1);
        let mut logits = t.forward_batch(&ids, 0).to_vec();
        for k in 0..16 {
            let next = crate::kernels::argmax(&logits);
            assert_eq!(next, greedy[ids.len() + k], "greedy token {k}");
            logits.copy_from_slice(t.forward(next, ids.len() + k));
        }
        println!("Gemma 4: {} short + {} long logits, every layer (worst {worst:.2e}) and 16 greedy tokens match HF", ids.len(), posv.len());
    }

    /// Speculative decoding must roll the conv state back with the KV cache:
    /// after a batched check whose two drafts are rejected, the next batch
    /// gives the logits of a run that never saw them. Both runs put every
    /// position through a 3-token batch (on the phone a single token takes
    /// the GEMV path, whose rounding differs), so they agree bit for bit.
    /// The same check without the restore must fail, so the bound is known
    /// to catch a missing rollback.
    #[test]
    fn lfm3_conv_state_rolls_back() {
        let Some(p) = crate::testutil::golden("gguf/LFM3-1.7B-Q4_K_M.gguf") else { return };
        let (c, w, _) = load(&p, 64).unwrap();
        let (c2, w2, _) = load(&p, 64).unwrap();
        let mut plain = Transformer::new(c, w);
        let mut spec = Transformer::new(c2, w2);
        let tokens: Vec<usize> = (0..16).map(|i| 100 + (i * 37) % 5000).collect();
        let next = crate::kernels::argmax(plain.forward_batch(&tokens, 0));
        let vocab = plain.config.vocab_size;
        // Positions 16, 17, 18: the certain token, then two fixed tokens.
        let want = plain.forward_batch_logits(&[next, 777, 4321], 16, 0)[vocab..3 * vocab].to_vec();
        // Logits error at 17 and 18 after a check with rejected drafts at
        // 17 and 18, with or without the restore.
        let mut run = |restore: bool| {
            assert_eq!(crate::kernels::argmax(spec.forward_batch(&tokens, 0)), next);
            spec.forward_batch_logits(&[next, 1234, 5678], 16, 0);
            if restore {
                spec.restore_conv(16);
            }
            let got = &spec.forward_batch_logits(&[777, 4321, 99], 17, 0)[..2 * vocab];
            max_errors(got, &want).1
        };
        let rel = run(true);
        assert!(rel < 1e-5, "conv rollback: logits rel {rel:.2e}");
        let bad = run(false);
        assert!(bad > 1e-2, "without the restore: logits rel {bad:.2e}");
        println!("conv rollback: logits rel {rel:.2e}; without the restore {bad:.2e}");
    }

    /// llama.cpp's own f32 conversion of SmolLM2 (`llama`: Q/K rows
    /// permuted, adjacent-pair RoPE), weights used in place from the mmap.
    #[test]
    fn smollm2_gguf_matches_transformers() {
        if let Some(mut t) = model("gguf/smollm2-135m-f32.gguf") {
            assert_eq!(t.config.rope_layout, RopeLayout::AdjacentPairs);
            check_against_transformers(&mut t, "smollm2-135m/acts.safetensors", "SmolLM2 (GGUF)");
        }
    }

    /// llama.cpp's f32 conversion of Qwen2.5 (`qwen2`: rotate-half, biases).
    #[test]
    fn qwen2_gguf_matches_transformers() {
        if let Some(mut t) = model("gguf/qwen2.5-0.5b-f32.gguf") {
            assert_eq!(t.config.rope_layout, RopeLayout::Halves);
            check_against_transformers(&mut t, "qwen2.5-0.5b/acts.safetensors", "Qwen2.5 (GGUF)");
        }
    }

    /// `--outtype bf16`: the checkpoint's own bf16 matrices, widened at
    /// load, so the same reference as the f32 file.
    #[test]
    fn qwen2_bf16_gguf_matches_transformers() {
        if let Some(mut t) = model("gguf/qwen2.5-0.5b-bf16.gguf") {
            check_against_transformers(&mut t, "qwen2.5-0.5b/acts.safetensors", "Qwen2.5 (bf16 GGUF)");
        }
    }

    /// `--outtype f16`: matrices rounded to f16 by the converter, against
    /// transformers with the same matrices rounded by torch (`hf_dump.py … f16`).
    #[test]
    fn qwen2_f16_gguf_matches_transformers() {
        if let Some(mut t) = model("gguf/qwen2.5-0.5b-f16.gguf") {
            check_against_transformers(&mut t, "qwen2.5-0.5b-f16/acts.safetensors", "Qwen2.5 (f16 GGUF)");
        }
    }
}
