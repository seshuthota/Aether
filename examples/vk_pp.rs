//! The GPU's prompt pass timed as `llama-bench` times `pp N`: an N-token
//! prompt from an empty cache, fed `GpuModel::batch` tokens at a time
//! (`forward_batch`), one warm-up run, then the mean of REPS runs, logits
//! for the last token only. So the numbers can be put next to
//! llama-bench's in the same session.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_pp -- \
//!       testdata/gguf/qwen2.5-0.5b-q4_0.gguf [f16] [profile] [attn1] [REPS]
//!
//! `attn1`: the prompt's attention one workgroup per token
//! (`Options::attn_per_token`), the shader before attention_tile.comp.
//!
//! With `profile`, also the GPU time of each kind of stage over one run of
//! each length (summed over layers and batches; timestamps slow the run a
//! little, so the tok/s of a profiled run are not the plain ones).

use aether::kernels::vulkan::Gpu;
use aether::model::gpu::{GpuModel, Options};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: vk_pp MODEL.gguf [f16] [REPS]");
    let f16_gemm = args.iter().any(|a| a == "f16");
    let profile = args.iter().any(|a| a == "profile");
    let attn_per_token = args.iter().any(|a| a == "attn1");
    let reps: usize = args.iter().skip(1).find_map(|a| a.parse().ok()).unwrap_or(5);
    let (config, weights, _) = aether::model::gguf::load(path, 1024).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut m = GpuModel::with(&gpu, config, weights, Options { f16_gemm, profile, attn_per_token, ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
    println!("# {path} on {} | batches of {}, {} | attention {} | mean of {reps} runs", gpu.info.name, m.batch, m.gemm_shader(), if attn_per_token { "per token" } else { "tiled" });
    println!("test,tok_per_s,ms");
    for n in [64usize, 128, 256, 512] {
        let tokens: Vec<usize> = (0..n).map(|i| 100 + i).collect();
        let mut stages: Vec<(&'static str, f64)> = Vec::new();
        let mut run = || {
            stages.clear();
            let mut p = 0;
            while p < n {
                let k = m.batch.min(n - p);
                std::hint::black_box(m.forward_batch(&gpu, &tokens[p..p + k], p).unwrap());
                for &(name, t) in &m.batch_profile {
                    match stages.iter_mut().find(|(s, _)| *s == name) {
                        Some(e) => e.1 += t,
                        None => stages.push((name, t)),
                    }
                }
                p += k;
            }
        };
        run();
        let t0 = Instant::now();
        (0..reps).for_each(|_| run());
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("pp{n},{:.1},{ms:.1}", n as f64 * 1e3 / ms);
        if profile {
            let parts: Vec<String> = stages.iter().map(|(k, t)| format!("{k} {:.2}", t * 1e3)).collect();
            println!("# pp{n} GPU ms: {}", parts.join(", "));
        }
    }
}
