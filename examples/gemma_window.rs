//! Gemma 4 past its 512-position sliding window, on the GPU: the logits at
//! positions around 512 from the GPU's decode step, from its prompt pass
//! (batches of `GemmaGpu::batch`), and from the CPU path (verified against
//! HF transformers and llama.cpp, results/phase-gemma4.md). A window that
//! starts one position off, or a shared layer reading the wrong cache,
//! shows as a jump in the GPU-vs-CPU error from position 512 on.
//!
//!   cargo run --release --features vulkan --example gemma_window -- \
//!       testdata/gguf/gemma-4-E2B-it-Q4_0.gguf testdata/wikitext/wikitext-2-raw/wiki.test.raw
//!
//! Per checked position: the largest logit difference over the largest
//! logit, and whether the top token agrees, GPU decode vs GPU prompt pass
//! and GPU decode vs CPU.

use aether::kernels::vulkan::Gpu;
use aether::model::gpu::{GemmaGpu, Options};
use aether::session::Session;
use aether::weight::Format;
use std::path::PathBuf;

const N: usize = 640;
const AT: [usize; 12] = [0, 100, 255, 500, 510, 511, 512, 513, 520, 600, 620, 639];

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0
}

fn compare(a: &[f32], b: &[f32]) -> (f32, bool) {
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    (d / scale, argmax(a) == argmax(b))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: gemma_window MODEL.gguf TEXT"));
    let raw = std::fs::read_to_string(args.next().expect("TEXT")).unwrap();
    let Session { mut model, text, .. } = Session::open(&path, None, N, Format::F32, None).unwrap();
    let tokens = &text.encode(&raw)[..N];

    let mut cpu = Vec::new();
    for (p, &t) in tokens.iter().enumerate() {
        let l = model.forward(t, p);
        if AT.contains(&p) {
            cpu.push(l.to_vec());
        }
    }
    drop(model);

    let gpu = Gpu::new().unwrap();
    let (config, weights, _) = aether::model::gguf::load(&path, N).unwrap();
    let mut m = GemmaGpu::with(&gpu, config, weights, Options { all_logits: true, ..Options::default() }).unwrap();
    let vocab = m.config.vocab_size;
    let mut dec = Vec::new();
    for (p, &t) in tokens.iter().enumerate() {
        let l = m.forward(&gpu, t, p).unwrap();
        if AT.contains(&p) {
            dec.push(l.to_vec());
        }
    }
    let mut bat = Vec::new();
    let mut p = 0;
    while p < N {
        let n = m.batch.min(N - p);
        let all = m.forward_batch(&gpu, &tokens[p..p + n], p).unwrap();
        for i in 0..n {
            if AT.contains(&(p + i)) {
                bat.push(all[i * vocab..(i + 1) * vocab].to_vec());
            }
        }
        p += n;
    }

    println!("# {} on {}, {N} tokens of {}, prompt batch {}", path.display(), gpu.info.name, "wikitext", m.batch);
    println!("position | decode vs prompt pass: rel, same top | GPU decode vs CPU: rel, same top");
    for (i, &p) in AT.iter().enumerate() {
        let (r1, s1) = compare(&dec[i], &bat[i]);
        let (r2, s2) = compare(&dec[i], &cpu[i]);
        println!("{p:8} | {r1:.2e} {s1} | {r2:.2e} {s2}");
    }
}
