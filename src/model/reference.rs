//! Test-only: compare a model against Hugging Face transformers' recorded
//! activations (tools/hf_dump.py), whatever file it was loaded from.

use crate::format::safetensors::SafeTensors;
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

/// Every layer boundary at every prompt position after a batched prefill
/// (relative error < 1e-3), then 48 greedy tokens identical to
/// `transformers.generate`. q and k are not compared: transformers records
/// them before RoPE, and llama.cpp's GGUF permutes them for Llama models.
pub fn check_against_transformers(t: &mut Transformer, acts: &str, label: &str) {
    let Some(acts) = golden(acts) else { return };
    let reference = SafeTensors::open(acts).unwrap();
    let (prompt, greedy) = (ids(&reference, "prompt_ids"), ids(&reference, "greedy_ids"));

    let mut obs = Collect::default();
    let mut logits = t.forward_batch_traced(&prompt, 0, &mut obs).to_vec();
    let mut worst: BTreeMap<String, f32> = BTreeMap::new();
    for ((name, layer, pos), ours) in &obs.0 {
        if matches!(name.as_str(), "q" | "k" | "attn" | "swiglu") {
            continue;
        }
        let key = if *layer < 0 { name.clone() } else { format!("L{layer}.{name}") };
        let (_, rel) = max_errors(ours, &row(&reference, &key, *pos));
        let w = worst.entry(name.clone()).or_default();
        *w = w.max(rel);
    }
    for (name, rel) in &worst {
        println!("{label} {name:<12} worst relative error vs transformers {rel:.2e}");
        assert!(*rel < 1e-3, "{label} {name}: relative error {rel:.2e}");
    }
    for (pos, &want) in greedy.iter().enumerate().skip(prompt.len()) {
        let next = crate::kernels::argmax(&logits);
        assert_eq!(next, want, "{label}: greedy token at position {pos}");
        logits.copy_from_slice(t.forward(next, pos));
    }
    println!("{label}: {} greedy tokens identical to transformers.generate", greedy.len() - prompt.len());
}
