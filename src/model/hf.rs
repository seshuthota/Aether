//! Loader for Hugging Face Llama-architecture checkpoints (`"model_type"`
//! `"llama"`: SmolLM2; `"qwen2"`: the same plus q/k/v biases):
//! `config.json` for the hyperparameters, `model.safetensors` for the weights.
//!
//! Tensor names follow transformers' `LlamaForCausalLM`:
//!
//! ```text
//! model.embed_tokens.weight                       [vocab, dim]
//! model.layers.{l}.input_layernorm.weight         [dim]
//! model.layers.{l}.self_attn.{q,k,v,o}_proj.weight
//! model.layers.{l}.post_attention_layernorm.weight
//! model.layers.{l}.mlp.{gate,up,down}_proj.weight
//! model.norm.weight                               [dim]
//! lm_head.weight                                  [vocab, dim] (absent when tied)
//! ```
//!
//! `nn.Linear` stores its weight as `[out, in]` row-major, the same layout
//! as the engine's matrices, so tensors are copied in as they are (widened
//! from bf16 to f32). transformers applies RoPE in the rotate-half layout.

use super::config::{Config, RopeLayout};
use super::transformer::{LayerWeights, Weights};
use crate::aligned::AlignedBuf;
use crate::format::json::Json;
use crate::format::safetensors::SafeTensors;
use crate::weight::Weight;
use std::io;
use std::path::Path;

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Reads `config.json` into a [`Config`]. The KV cache is sized for
/// `max_seq` positions (or the model's own limit, if smaller): SmolLM2
/// allows 8192, which for 30 layers would cost ~380 MB of cache in f32.
pub fn load_config(dir: &Path, max_seq: usize) -> io::Result<Config> {
    let text = std::fs::read_to_string(dir.join("config.json"))?;
    let j = Json::parse(&text).map_err(|e| bad(format!("config.json: {e}")))?;
    let num = |k: &str| j.get(k).and_then(Json::as_usize).ok_or_else(|| bad(format!("config.json: missing {k}")));
    let float = |k: &str, default: f64| j.get(k).and_then(Json::as_f64).unwrap_or(default) as f32;
    let model_type = j.get("model_type").and_then(Json::as_str).unwrap_or("?");
    if model_type != "llama" && model_type != "qwen2" {
        return Err(bad(format!("model_type {model_type:?} is not supported (llama, qwen2)")));
    }
    let n_heads = num("num_attention_heads")?;
    Ok(Config {
        dim: num("hidden_size")?,
        hidden_dim: num("intermediate_size")?,
        n_layers: num("num_hidden_layers")?,
        n_heads,
        n_kv_heads: j.get("num_key_value_heads").and_then(Json::as_usize).unwrap_or(n_heads),
        vocab_size: num("vocab_size")?,
        seq_len: num("max_position_embeddings")?.min(max_seq),
        rope_theta: float("rope_theta", 10000.0),
        rope_layout: RopeLayout::Halves,
        norm_eps: float("rms_norm_eps", 1e-5),
    })
}

/// Loads config and weights from a checkpoint directory.
pub fn load(dir: impl AsRef<Path>, max_seq: usize) -> io::Result<(Config, Weights)> {
    let dir = dir.as_ref();
    let config = load_config(dir, max_seq)?;
    let st = SafeTensors::open(dir.join("model.safetensors"))?;
    let c = &config;
    let (dim, kv_dim, hidden) = (c.dim, c.kv_dim(), c.hidden_dim);

    // One tensor, checked against the shape the config implies, widened to f32.
    let tensor = |name: &str, shape: &[usize]| -> io::Result<AlignedBuf> {
        let v = st.get(name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        if v.shape != shape {
            return Err(bad(format!("{name}: shape {:?}, expected {shape:?}", v.shape)));
        }
        let mut buf = AlignedBuf::zeros(v.numel());
        v.to_f32(&mut buf);
        Ok(buf)
    };

    let mat = |name: &str, shape: &[usize]| tensor(name, shape).map(Weight::from);
    // A bias vector if the checkpoint has one (Qwen2's q/k/v), else None.
    let bias = |name: &str, n: usize| -> io::Result<Option<AlignedBuf>> { if st.info(name).is_some() { tensor(name, &[n]).map(Some) } else { Ok(None) } };
    let layers = (0..c.n_layers)
        .map(|l| {
            let p = format!("model.layers.{l}");
            Ok(LayerWeights {
                attn_norm: tensor(&format!("{p}.input_layernorm.weight"), &[dim])?,
                wq: mat(&format!("{p}.self_attn.q_proj.weight"), &[dim, dim])?,
                wk: mat(&format!("{p}.self_attn.k_proj.weight"), &[kv_dim, dim])?,
                wv: mat(&format!("{p}.self_attn.v_proj.weight"), &[kv_dim, dim])?,
                wo: mat(&format!("{p}.self_attn.o_proj.weight"), &[dim, dim])?,
                bq: bias(&format!("{p}.self_attn.q_proj.bias"), dim)?,
                bk: bias(&format!("{p}.self_attn.k_proj.bias"), kv_dim)?,
                bv: bias(&format!("{p}.self_attn.v_proj.bias"), kv_dim)?,
                ffn_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[dim])?,
                w_gate: mat(&format!("{p}.mlp.gate_proj.weight"), &[hidden, dim])?,
                w_up: mat(&format!("{p}.mlp.up_proj.weight"), &[hidden, dim])?,
                w_down: mat(&format!("{p}.mlp.down_proj.weight"), &[dim, hidden])?,
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    let classifier = match st.info("lm_head.weight") {
        Some(_) => Some(mat("lm_head.weight", &[c.vocab_size, dim])?),
        None => None, // tied: the classifier reuses the embedding table
    };
    let weights = Weights {
        token_embedding: mat("model.embed_tokens.weight", &[c.vocab_size, dim])?,
        layers,
        final_norm: tensor("model.norm.weight", &[dim])?,
        classifier,
    };
    Ok((config, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Transformer;
    use crate::model::reference::check_against_transformers;
    use crate::testutil::golden;

    fn model(dir: &str) -> Option<Transformer> {
        let (config, weights) = load(golden(dir)?, 256).unwrap();
        Some(Transformer::new(config, weights))
    }

    /// SmolLM2-135M from its HF safetensors.
    #[test]
    fn smollm2_matches_transformers() {
        if let Some(mut t) = model("models/SmolLM2-135M") {
            check_against_transformers(&mut t, "smollm2-135m/acts.safetensors", "SmolLM2 (safetensors)");
        }
    }

    /// SmolLM2-360M: 15 query heads sharing 5 K/V heads (135M: 9 and 3),
    /// 32 layers, dim 960, theta 1e5. Same tokenizer.json as 135M, byte
    /// for byte, so the 135M tokenizer test covers it.
    #[test]
    fn smollm2_360m_matches_transformers() {
        if let Some(mut t) = model("models/SmolLM2-360M") {
            check_against_transformers(&mut t, "smollm2-360m/acts.safetensors", "SmolLM2-360M (safetensors)");
        }
    }

    /// Qwen2.5-0.5B: q/k/v biases, 14 query heads sharing 2 K/V heads,
    /// theta 1e6, eps 1e-6, 151k vocabulary.
    #[test]
    fn qwen2_matches_transformers() {
        if let Some(mut t) = model("models/Qwen2.5-0.5B-Instruct") {
            check_against_transformers(&mut t, "qwen2.5-0.5b/acts.safetensors", "Qwen2.5 (safetensors)");
        }
    }
}
