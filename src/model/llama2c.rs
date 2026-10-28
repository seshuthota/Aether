//! Loader for llama2.c's model format (`stories15M.bin` and friends).
//!
//! The file is a 28-byte header of seven little-endian `i32`s, followed by
//! every weight as raw `f32`, tensor after tensor, in a fixed order:
//!
//! ```text
//! dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
//! token_embedding   [vocab, dim]
//! rms_att           [layers, dim]
//! wq                [layers, dim, dim]           (rows = outputs)
//! wk, wv            [layers, kv_dim, dim]
//! wo                [layers, dim, dim]
//! rms_ffn           [layers, dim]
//! w1                [layers, hidden, dim]        (gate)
//! w2                [layers, dim, hidden]        (down)
//! w3                [layers, hidden, dim]        (up)
//! rms_final         [dim]
//! freq_cis_real/imag [seq_len, head_size/2] each (legacy RoPE table, skipped)
//! wcls              [vocab, dim]                 (only if vocab_size < 0)
//! ```
//!
//! A negative `vocab_size` in the header means the classifier has its own
//! weights at the end; positive means it reuses the embedding table ("tied").
//! Each tensor of all layers is stored contiguously, so layer `l`'s `wq` is
//! the `l`-th `dim*dim` slice of the `wq` block.

use super::config::{Config, RopeLayout};
use super::transformer::{LayerWeights, Weights};
use crate::aligned::AlignedBuf;
use crate::weight::Weight;
use std::io;
use std::path::Path;

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Reads a whole llama2.c model file into memory.
pub fn load(path: impl AsRef<Path>) -> io::Result<(Config, Weights)> {
    let bytes = std::fs::read(path.as_ref())?;
    if bytes.len() < 28 {
        return Err(bad(format!("{}: too short for a header", path.as_ref().display())));
    }
    let h: Vec<i32> = bytes[..28].as_chunks::<4>().0.iter().map(|&b| i32::from_le_bytes(b)).collect();
    if h.iter().enumerate().any(|(i, &v)| v == 0 || (v < 0 && i != 5)) {
        return Err(bad(format!("bad header {h:?}")));
    }
    let shared = h[5] > 0;
    let config = Config {
        dim: h[0] as usize,
        hidden_dim: h[1] as usize,
        n_layers: h[2] as usize,
        n_heads: h[3] as usize,
        n_kv_heads: h[4] as usize,
        vocab_size: h[5].unsigned_abs() as usize,
        seq_len: h[6] as usize,
        rope_theta: 10000.0,
        rope_layout: RopeLayout::AdjacentPairs,
        norm_eps: 1e-5,
    };
    let c = &config;
    let (dim, hidden, layers, kv_dim) = (c.dim, c.hidden_dim, c.n_layers, c.kv_dim());

    // Walks the f32 payload in file order, handing out one tensor at a time.
    let mut cursor = Cursor { bytes: &bytes[28..], at: 0 };
    let token_embedding = cursor.take(c.vocab_size * dim)?;
    let rms_att = cursor.take_layers(layers, dim)?;
    let wq = cursor.take_layers(layers, dim * dim)?;
    let wk = cursor.take_layers(layers, kv_dim * dim)?;
    let wv = cursor.take_layers(layers, kv_dim * dim)?;
    let wo = cursor.take_layers(layers, dim * dim)?;
    let rms_ffn = cursor.take_layers(layers, dim)?;
    let w1 = cursor.take_layers(layers, hidden * dim)?;
    let w2 = cursor.take_layers(layers, dim * hidden)?;
    let w3 = cursor.take_layers(layers, hidden * dim)?;
    let final_norm = cursor.take(dim)?;
    cursor.skip(c.seq_len * c.head_size())?; // freq_cis_real + freq_cis_imag (each seq_len * head_size/2)
    let classifier = if shared { None } else { Some(Weight::from(cursor.take(c.vocab_size * dim)?)) };
    if cursor.remaining() != 0 {
        return Err(bad(format!("{} unexpected trailing bytes", cursor.remaining())));
    }

    let mut per_layer = [rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3].map(|v| v.into_iter());
    let layers = (0..layers)
        .map(|_| {
            let mut next = |i: usize| per_layer[i].next().unwrap();
            LayerWeights {
                attn_norm: next(0),
                wq: Weight::from(next(1)),
                wk: Weight::from(next(2)),
                wv: Weight::from(next(3)),
                wo: Weight::from(next(4)),
                bq: None,
                bk: None,
                bv: None,
                ffn_norm: next(5),
                w_gate: Weight::from(next(6)),
                w_down: Weight::from(next(7)),
                w_up: Weight::from(next(8)),
            }
        })
        .collect();
    Ok((config, Weights { token_embedding: Weight::from(token_embedding), layers, final_norm, classifier }))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> io::Result<AlignedBuf> {
        let end = self.at + n * 4;
        let src = self.bytes.get(self.at..end).ok_or_else(|| bad("file ends inside a tensor".into()))?;
        let mut buf = AlignedBuf::zeros(n);
        for (d, &b) in buf.iter_mut().zip(src.as_chunks::<4>().0) {
            *d = f32::from_le_bytes(b);
        }
        self.at = end;
        Ok(buf)
    }

    /// `layers` consecutive tensors of `n` floats each.
    fn take_layers(&mut self, layers: usize, n: usize) -> io::Result<Vec<AlignedBuf>> {
        (0..layers).map(|_| self.take(n)).collect()
    }

    fn skip(&mut self, n: usize) -> io::Result<()> {
        self.at += n * 4;
        if self.at > self.bytes.len() {
            return Err(bad("file ends inside a skipped tensor".into()));
        }
        Ok(())
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::sampler::Sampler;
    use crate::model::{Observer, Transformer};
    use crate::testutil::{Key, golden, max_errors, read_dump};
    use std::collections::{BTreeMap, HashMap};

    /// Collects every activation the forward pass reports.
    #[derive(Default)]
    struct Collect(HashMap<Key, Vec<f32>>);

    impl Observer for Collect {
        fn tap(&mut self, name: &str, layer: Option<usize>, pos: usize, data: &[f32]) {
            self.0.insert((name.to_string(), layer.map_or(-1, |l| l as i32), pos), data.to_vec());
        }
    }

    fn model() -> Option<Transformer> {
        let path = golden("models/stories15M/stories15M.bin")?;
        let (config, weights) = load(path).unwrap();
        Some(Transformer::new(config, weights))
    }

    /// Replays a llama2.c run: same prompt tokens, then our own choice of
    /// each next token, which must equal the token llama2.c chose. Returns
    /// our activations and the reference dump.
    type Activations = HashMap<Key, Vec<f32>>;

    fn replay(dump: &str, sampler: Sampler) -> Option<(Activations, Activations)> {
        replay_on(dump, sampler, None)
    }

    /// [`replay`] with the matrix products split across threads on `cores`.
    fn replay_on(dump: &str, mut sampler: Sampler, cores: Option<&[usize]>) -> Option<(Activations, Activations)> {
        let mut t = model()?;
        if let Some(cores) = cores {
            t.set_pool(crate::pool::ThreadPool::new(cores).unwrap());
        }
        let (reference, _) = read_dump(&golden(dump)?);
        let prompt: Vec<usize> = (0..).map_while(|i| reference.get(&("prompt".into(), -1, i))).map(|v| v[0] as usize).collect();
        assert!(!prompt.is_empty(), "dump has no prompt tokens");
        let mut obs = Collect::default();
        let (mut token, mut pos) = (prompt[0], 0);
        while let Some(expected_next) = reference.get(&("next".into(), -1, pos)).map(|v| v[0] as usize).or_else(|| prompt.get(pos + 1).copied()) {
            let mut logits = t.forward_traced(token, pos, &mut obs).to_vec();
            let next = if pos + 1 < prompt.len() { prompt[pos + 1] } else { sampler.sample(&mut logits) };
            assert_eq!(next, expected_next, "token at position {} differs from llama2.c", pos + 1);
            token = next;
            pos += 1;
        }
        println!("{dump}: {pos} positions, every token identical to llama2.c");
        Some((obs.0, reference))
    }

    /// Every activation at every layer boundary and position, compared with
    /// llama2.c. Prints the worst error per activation name, so a bug shows
    /// up as the first name (in forward order) where the error jumps.
    #[test]
    fn greedy_matches_llama2c_at_every_layer() {
        let Some((ours, reference)) = replay("stories15M/greedy.dump", Sampler::new(0.0, 0.9, 0, 0)) else { return };
        let mut worst: BTreeMap<String, (f32, f32)> = BTreeMap::new();
        for (key, r) in &reference {
            if key.0 == "prompt" || key.0 == "next" {
                continue;
            }
            let o = ours.get(key).unwrap_or_else(|| panic!("we never produced {key:?}"));
            let (abs, rel) = max_errors(o, r);
            let w = worst.entry(key.0.clone()).or_default();
            *w = (w.0.max(abs), w.1.max(rel));
        }
        for (name, (abs, rel)) in &worst {
            println!("{name:<12} max abs {abs:.3e}  max rel {rel:.3e}");
        }
        for (name, (_, rel)) in &worst {
            assert!(*rel < 1e-4, "{name}: relative error {rel:.3e} vs llama2.c");
        }
    }

    /// Prefill in batches through `forward_batch` (GEMM path), then decode
    /// one token at a time. Batch sizes 5, 12, 13 are deliberately not
    /// multiples of the 4×4 GEMM tile. Every activation at every position is
    /// compared with llama2.c's, and every greedy token must match. Runs on
    /// one thread, then on four (batched attention hands out (token, head)
    /// pairs across threads), which must give the same activations bit for
    /// bit.
    #[test]
    fn batched_prefill_matches_llama2c() {
        let Some(one) = batched_prefill_run(None) else { return };
        let many = std::thread::spawn(|| {
            let cores: Vec<usize> = crate::sys::affinity::online_cpus().unwrap().into_iter().take(4).collect();
            batched_prefill_run(Some(&cores)).unwrap()
        })
        .join()
        .unwrap();
        assert!(one == many, "four threads changed a batched-prefill activation");
    }

    fn batched_prefill_run(cores: Option<&[usize]>) -> Option<Activations> {
        let (Some(mut t), Some(path)) = (model(), golden("stories15M/greedy.dump")) else { return None };
        if let Some(cores) = cores {
            t.set_pool(crate::pool::ThreadPool::new(cores).unwrap());
        }
        let (reference, _) = read_dump(&path);
        let get = |name: &str, pos: usize| reference.get(&(name.to_string(), -1, pos)).map(|v| v[0] as usize);
        let prompt: Vec<usize> = (0..).map_while(|i| get("prompt", i)).collect();
        // The full token sequence llama2.c fed itself: prompt, then its picks.
        let seq: Vec<usize> = (0..64).map(|p| if p < prompt.len() { prompt[p] } else { get("next", p - 1).unwrap() }).collect();

        let mut obs = Collect::default();
        let mut start = 0;
        for len in [5, 12, 13] {
            let logits = t.forward_batch_traced(&seq[start..start + len], start, &mut obs).to_vec();
            start += len;
            assert_eq!(crate::kernels::argmax(&logits), seq[start], "token after batch ending at {start}");
        }
        for pos in start..63 {
            let logits = t.forward_traced(seq[pos], pos, &mut obs).to_vec();
            assert_eq!(crate::kernels::argmax(&logits), seq[pos + 1], "token at position {}", pos + 1);
        }
        let mut worst: f32 = 0.0;
        for (key, r) in &reference {
            if key.0 == "prompt" || key.0 == "next" || key.2 >= 63 {
                continue;
            }
            // A batch reports final_norm/logits only for its last position.
            let Some(o) = obs.0.get(key) else {
                assert!(key.0 == "final_norm" || key.0 == "logits", "missing {key:?}");
                continue;
            };
            let (_, rel) = max_errors(o, r);
            assert!(rel < 1e-4, "{key:?}: relative error {rel:.3e} vs llama2.c");
            worst = worst.max(rel);
        }
        println!("batched prefill (5+12+13) then decode, {} threads: tokens identical, worst relative error {worst:.2e}", cores.map_or(1, <[usize]>::len));
        Some(obs.0)
    }

    /// Four threads: every activation still within tolerance of llama2.c and
    /// every greedy token identical (threading splits rows, not sums).
    #[test]
    fn threaded_greedy_matches_llama2c() {
        std::thread::spawn(|| {
            let cores: Vec<usize> = crate::sys::affinity::online_cpus().unwrap().into_iter().take(4).collect();
            let Some((ours, reference)) = replay_on("stories15M/greedy.dump", Sampler::new(0.0, 0.9, 0, 0), Some(&cores)) else { return };
            for (key, r) in &reference {
                if key.0 != "prompt" && key.0 != "next" {
                    let (_, rel) = max_errors(&ours[key], r);
                    assert!(rel < 1e-4, "{key:?}: relative error {rel:.3e} vs llama2.c");
                }
            }
        })
        .join()
        .unwrap();
    }

    /// A greedy run over the model's whole context (`long.dump`, ~220
    /// positions), so attention past 128 positions is split into chunks and
    /// merged. Decode one token at a time on one thread and on four (chunks
    /// handed out across threads), then a single 200-token prefill on one and
    /// on four: every token identical to llama2.c, every activation within
    /// tolerance, and threads never change a bit.
    #[test]
    fn long_context_matches_llama2c() {
        let Some(path) = golden("stories15M/long.dump") else { return };
        let (reference, _) = read_dump(&path);
        let check = |ours: &Activations, label: &str| {
            let mut worst: f32 = 0.0;
            for (key, r) in &reference {
                if key.0 == "prompt" || key.0 == "next" {
                    continue;
                }
                let Some(o) = ours.get(key) else { continue };
                let (_, rel) = max_errors(o, r);
                assert!(rel < 1e-4, "{label} {key:?}: relative error {rel:.3e} vs llama2.c");
                worst = worst.max(rel);
            }
            println!("long context, {label}: worst relative error {worst:.2e}");
        };
        let cores: Vec<usize> = crate::sys::affinity::online_cpus().unwrap().into_iter().take(4).collect();
        let (one, _) = replay("stories15M/long.dump", Sampler::new(0.0, 0.9, 0, 0)).unwrap();
        check(&one, "decode, 1 thread");
        let c = cores.clone();
        let many = std::thread::spawn(move || replay_on("stories15M/long.dump", Sampler::new(0.0, 0.9, 0, 0), Some(&c)).unwrap().0).join().unwrap();
        assert!(one == many, "four threads changed a long-context decode activation");

        let get = |name: &str, pos: usize| reference.get(&(name.to_string(), -1, pos)).map(|v| v[0] as usize);
        let prompt: Vec<usize> = (0..).map_while(|i| get("prompt", i)).collect();
        let seq: Vec<usize> = (0..201).map(|p| if p < prompt.len() { prompt[p] } else { get("next", p - 1).unwrap() }).collect();
        let prefill = |cores: Option<Vec<usize>>| {
            let seq = seq.clone();
            std::thread::spawn(move || {
                let mut t = model().unwrap();
                if let Some(cores) = cores {
                    t.set_pool(crate::pool::ThreadPool::new(&cores).unwrap());
                }
                let mut obs = Collect::default();
                let logits = t.forward_batch_traced(&seq[..200], 0, &mut obs).to_vec();
                assert_eq!(crate::kernels::argmax(&logits), seq[200], "token after a 200-token prefill");
                obs.0
            })
            .join()
            .unwrap()
        };
        let (p1, p4) = (prefill(None), prefill(Some(cores)));
        check(&p1, "200-token prefill");
        assert!(p1 == p4, "four threads changed a long prefill activation");
    }

    /// Temperature 0.8, top-p 0.9, seed 42: the same xorshift stream and the
    /// same nucleus procedure must pick the same tokens as `run -s 42`.
    #[test]
    fn seeded_sampling_matches_llama2c() {
        replay("stories15M/sampled.dump", Sampler::new(0.8, 0.9, 0, 42));
    }
}
