//! Loader for Hugging Face Llama-architecture checkpoints (SmolLM2, and
//! anything else with `"model_type": "llama"`): `config.json` for the
//! hyperparameters, `model.safetensors` for the weights.
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
    if model_type != "llama" {
        return Err(bad(format!("model_type {model_type:?} is not supported by this loader (llama only)")));
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

    let layers = (0..c.n_layers)
        .map(|l| {
            let p = format!("model.layers.{l}");
            Ok(LayerWeights {
                attn_norm: tensor(&format!("{p}.input_layernorm.weight"), &[dim])?,
                wq: tensor(&format!("{p}.self_attn.q_proj.weight"), &[dim, dim])?,
                wk: tensor(&format!("{p}.self_attn.k_proj.weight"), &[kv_dim, dim])?,
                wv: tensor(&format!("{p}.self_attn.v_proj.weight"), &[kv_dim, dim])?,
                wo: tensor(&format!("{p}.self_attn.o_proj.weight"), &[dim, dim])?,
                ffn_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[dim])?,
                w_gate: tensor(&format!("{p}.mlp.gate_proj.weight"), &[hidden, dim])?,
                w_up: tensor(&format!("{p}.mlp.up_proj.weight"), &[hidden, dim])?,
                w_down: tensor(&format!("{p}.mlp.down_proj.weight"), &[dim, hidden])?,
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    let classifier = match st.info("lm_head.weight") {
        Some(_) => Some(tensor("lm_head.weight", &[c.vocab_size, dim])?),
        None => None, // tied: the classifier reuses the embedding table
    };
    let weights = Weights {
        token_embedding: tensor("model.embed_tokens.weight", &[c.vocab_size, dim])?,
        layers,
        final_norm: tensor("model.norm.weight", &[dim])?,
        classifier,
    };
    Ok((config, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Observer, Transformer};
    use crate::testutil::{golden, max_errors};
    use std::collections::{BTreeMap, HashMap};

    /// Activations keyed by (name, layer or -1, position).
    #[derive(Default)]
    struct Collect(HashMap<(String, i32, usize), Vec<f32>>);

    impl Observer for Collect {
        fn tap(&mut self, name: &str, layer: Option<usize>, pos: usize, data: &[f32]) {
            self.0.insert((name.to_string(), layer.map_or(-1, |l| l as i32), pos), data.to_vec());
        }
    }

    /// Row `pos` of a [T, width] reference tensor.
    fn row(st: &SafeTensors, name: &str, pos: usize) -> Vec<f32> {
        let v = st.get(name).unwrap_or_else(|| panic!("reference has no {name}"));
        let width = v.shape[1];
        let mut all = vec![0.0; v.numel()];
        v.to_f32(&mut all);
        all[pos * width..(pos + 1) * width].to_vec()
    }

    fn ids(st: &SafeTensors, name: &str) -> Vec<usize> {
        let v = st.get(name).unwrap();
        let mut f = vec![0.0; v.numel()];
        v.to_f32(&mut f);
        f.into_iter().map(|x| x as usize).collect()
    }

    /// SmolLM2-135M against Hugging Face transformers (tools/hf_dump.py):
    /// every layer boundary at every prompt position after a batched
    /// prefill, then 48 greedy tokens identical to `generate`.
    #[test]
    fn smollm2_matches_transformers() {
        let (Some(dir), Some(acts)) = (golden("models/SmolLM2-135M"), golden("smollm2-135m/acts.safetensors")) else { return };
        let (config, weights) = load(&dir, 256).unwrap();
        let reference = SafeTensors::open(acts).unwrap();
        let (prompt, greedy) = (ids(&reference, "prompt_ids"), ids(&reference, "greedy_ids"));
        let mut t = Transformer::new(config, weights);

        let mut obs = Collect::default();
        let mut logits = t.forward_batch_traced(&prompt, 0, &mut obs).to_vec();
        let mut worst: BTreeMap<String, f32> = BTreeMap::new();
        for ((name, layer, pos), ours) in &obs.0 {
            let key = if *layer < 0 { name.clone() } else { format!("L{layer}.{name}") };
            if matches!(name.as_str(), "q" | "k" | "attn" | "swiglu") {
                continue; // not hooked on the HF side (q/k are pre-RoPE there)
            }
            let (_, rel) = max_errors(ours, &row(&reference, &key, *pos));
            let w = worst.entry(name.clone()).or_default();
            *w = w.max(rel);
        }
        for (name, rel) in &worst {
            println!("{name:<12} worst relative error vs transformers {rel:.2e}");
            assert!(*rel < 1e-3, "{name}: relative error {rel:.2e}");
        }

        // Greedy continuation, one token at a time.
        for (pos, &want) in greedy.iter().enumerate().skip(prompt.len()) {
            let next = crate::kernels::argmax(&logits);
            assert_eq!(next, want, "greedy token at position {pos}");
            logits.copy_from_slice(t.forward(next, pos));
        }
        println!("{} greedy tokens identical to transformers.generate", greedy.len() - prompt.len());
    }
}
