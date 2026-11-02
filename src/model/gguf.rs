//! Loader for GGUF model files (llama.cpp's format): `llama` and `qwen2`
//! architectures, tensors in f32 / f16 / bf16 / Q8_0 / Q4_0.
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

use super::config::{Config, RopeLayout};
use super::transformer::{AttentionWeights, ConvWeights, LayerWeights, Mixer, Weights};
use crate::aligned::AlignedBuf;
use crate::format::gguf::{GgmlType, Gguf, Value};
use crate::format::safetensors::{bf16_to_f32, f16_to_f32};
use crate::weight::{Floats, Mapped, Store, Weight};
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
        "qwen2" | "lfm3" => RopeLayout::Halves,
        other => return Err(bad(format!("architecture {other:?} is not supported (llama, qwen2, lfm3)"))),
    };
    let key = |k: &str| g.get(&format!("{arch}.{k}"));
    let num = |k: &str| key(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| bad(format!("missing {arch}.{k}")));
    let arr = |k: &str| key(k).and_then(Value::as_array);
    let lfm3 = arch == "lfm3";
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
    let conv_width = key("shortconv.l_cache").and_then(Value::as_u64).map(|v| v as usize).unwrap_or(7);
    let layers = (0..n_layers)
        .map(|l| {
            let is_conv = lfm3 && kv_arr.as_ref().is_some_and(|a| a.get(l).copied().unwrap_or(0) == 0);
            if is_conv {
                crate::model::config::LayerKind::Conv { width: conv_width }
            } else {
                let is_sliding = lfm3 && sliding.as_ref().and_then(|a| a.get(l)).copied().unwrap_or(false);
                let (theta, pairs) = if is_sliding { (theta_swa, head_dim / 2) } else { (theta_full, pairs_full) };
                crate::model::config::LayerKind::Attention {
                    window: if is_sliding { Some(window) } else { None },
                    rope: crate::model::config::Rope { theta, pairs, layout: rope_layout },
                }
            }
        })
        .collect::<Vec<_>>();
    let config = Config {
        dim,
        hidden_dim: num("feed_forward_length")?,
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
        qk_norm: lfm3,
        output_gate: lfm3,
        sandwich_norm: lfm3,
    };
    let c = &config;
    let (dim, kv_dim, hidden) = (c.dim, c.kv_dim(), c.hidden_dim);

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
            GgmlType::Other(id) => return Err(bad(format!("{name}: ggml type {id} is not supported"))),
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
            let mixer = match c.kind(l) {
                crate::model::config::LayerKind::Attention { .. } => Mixer::Attention(AttentionWeights {
                    wq: matrix(&format!("{p}.attn_q.weight"), dim, dim)?,
                    wk: matrix(&format!("{p}.attn_k.weight"), kv_dim, dim)?,
                    wv: matrix(&format!("{p}.attn_v.weight"), kv_dim, dim)?,
                    wo: matrix(&format!("{p}.attn_output.weight"), dim, dim)?,
                    bq: vector(&format!("{p}.attn_q.bias"), dim)?,
                    bk: vector(&format!("{p}.attn_k.bias"), kv_dim)?,
                    bv: vector(&format!("{p}.attn_v.bias"), kv_dim)?,
                    q_norm: vector(&format!("{p}.attn_q_norm.weight"), head_dim)?,
                    k_norm: vector(&format!("{p}.attn_k_norm.weight"), head_dim)?,
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
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    let weights = Weights {
        token_embedding: matrix("token_embd.weight", c.vocab_size, dim)?,
        layers,
        final_norm: required("output_norm.weight", dim)?,
        classifier: if g.tensor("output.weight").is_some() { Some(matrix("output.weight", c.vocab_size, dim)?) } else { None },
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
    use crate::testutil::host_golden;

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
