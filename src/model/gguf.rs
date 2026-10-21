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
use super::transformer::{LayerWeights, Weights};
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
        "qwen2" => RopeLayout::Halves,
        other => return Err(bad(format!("architecture {other:?} is not supported (llama, qwen2)"))),
    };
    let key = |k: &str| g.get(&format!("{arch}.{k}"));
    let num = |k: &str| key(k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| bad(format!("missing {arch}.{k}")));
    let n_heads = num("attention.head_count")?;
    let embd = g.tensor("token_embd.weight").ok_or_else(|| bad("no token_embd.weight".into()))?;
    let config = Config {
        dim: num("embedding_length")?,
        hidden_dim: num("feed_forward_length")?,
        n_layers: num("block_count")?,
        n_heads,
        n_kv_heads: num("attention.head_count_kv").unwrap_or(n_heads),
        vocab_size: embd.dims[1],
        seq_len: num("context_length")?.min(max_seq),
        rope_theta: key("rope.freq_base").and_then(Value::as_f64).unwrap_or(10000.0) as f32,
        rope_layout,
        norm_eps: key("attention.layer_norm_rms_epsilon").and_then(Value::as_f64).ok_or_else(|| bad("missing rms epsilon".into()))? as f32,
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

    let layers = (0..c.n_layers)
        .map(|l| {
            let p = format!("blk.{l}");
            Ok(LayerWeights {
                attn_norm: required(&format!("{p}.attn_norm.weight"), dim)?,
                wq: matrix(&format!("{p}.attn_q.weight"), dim, dim)?,
                wk: matrix(&format!("{p}.attn_k.weight"), kv_dim, dim)?,
                wv: matrix(&format!("{p}.attn_v.weight"), kv_dim, dim)?,
                wo: matrix(&format!("{p}.attn_output.weight"), dim, dim)?,
                bq: vector(&format!("{p}.attn_q.bias"), dim)?,
                bk: vector(&format!("{p}.attn_k.bias"), kv_dim)?,
                bv: vector(&format!("{p}.attn_v.bias"), kv_dim)?,
                ffn_norm: required(&format!("{p}.ffn_norm.weight"), dim)?,
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
}
