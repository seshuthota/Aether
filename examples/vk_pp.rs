//! The GPU's prompt pass timed as `llama-bench` times `pp N`: an N-token
//! prompt from an empty cache, fed `GpuModel::batch` tokens at a time
//! (`forward_batch`), one warm-up run, then the mean of REPS runs, logits
//! for the last token only. So the numbers can be put next to
//! llama-bench's in the same session.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_pp -- \
//!       testdata/gguf/qwen2.5-0.5b-q4_0.gguf [f16] [profile] [attn1] [chunked] [REPS] [depth D]...
//!
//! `attn1`: the prompt's attention one workgroup per token
//! (`Options::attn_per_token`), the shader before attention_tile.comp.
//! `chunked`: the prompt's attention in two passes
//! (`Options::attn_chunked`), as before flash attention.
//!
//! `depth D`: fill the cache to position D through `forward_batch`
//! (untimed), then time the prompt from there, as llama-bench's `-d` does.
//! A timed run writes the same positions again, so it needs no clearing.
//! Repeatable (`depth 0 depth 1024 depth 4096`); the loader is sized for
//! the deepest plus the longest prompt.
//!
//! With `profile`, also the GPU time of each kind of stage over one run of
//! each length and depth (summed over layers and batches; timestamps slow
//! the run a little, so the tok/s of a profiled run are not the plain ones).

use aether::kernels::vulkan::Gpu;
use aether::model::gpu::{GpuModel, Options};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: vk_pp MODEL.gguf [f16] [profile] [attn1] [chunked] [REPS] [depth D]...");
    let (mut f16_gemm, mut profile, mut attn_per_token, mut attn_chunked, mut reps) = (false, false, false, false, 5);
    let mut depths: Vec<usize> = Vec::new();
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "f16" => f16_gemm = true,
            "profile" => profile = true,
            "attn1" => attn_per_token = true,
            "chunked" => attn_chunked = true,
            "depth" => depths.push(it.next().expect("depth needs a value").parse().expect("depth")),
            _ => reps = a.parse().expect("unknown argument (f16, profile, attn1, chunked, REPS, depth D)"),
        }
    }
    // With an explicit depth, time only that; the label says which.
    let explicit_depth = !depths.is_empty();
    if !explicit_depth {
        depths.push(0);
    }
    let lengths = [64usize, 128, 256, 512];
    let seq = depths.iter().max().unwrap() + lengths.iter().max().unwrap();
    let (config, weights, _) = aether::model::gguf::load(path, seq).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut m = GpuModel::with(&gpu, config, weights, Options { f16_gemm, profile, attn_per_token, attn_chunked, ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
    let attention = if attn_per_token { "per token" } else if attn_chunked { "chunked" } else { "flash" };
    println!("# {path} on {} | batches of {}, {} | attention {attention} | mean of {reps} runs", gpu.info.name, m.batch, m.gemm_shader());
    println!("test,tok_per_s,ms");
    for &depth in &depths {
        if depth > 0 {
            let fill: Vec<usize> = (0..depth).map(|i| 100 + i).collect();
            let mut p = 0;
            while p < depth {
                let k = m.batch.min(depth - p);
                m.forward_batch(&gpu, &fill[p..p + k], p).unwrap();
                p += k;
            }
        }
        for n in lengths {
            let tokens: Vec<usize> = (0..n).map(|i| 100 + depth + i).collect();
            let label = if explicit_depth { format!("pp{n}@d{depth}") } else { format!("pp{n}") };
            let mut stages: Vec<(&'static str, f64)> = Vec::new();
            let mut run = || {
                stages.clear();
                let mut p = 0;
                while p < n {
                    let k = m.batch.min(n - p);
                    std::hint::black_box(m.forward_batch(&gpu, &tokens[p..p + k], depth + p).unwrap());
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
            println!("{label},{:.1},{ms:.1}", n as f64 * 1e3 / ms);
            if profile {
                let parts: Vec<String> = stages.iter().map(|(k, t)| format!("{k} {:.2}", t * 1e3)).collect();
                println!("# {label} GPU ms: {}", parts.join(", "));
            }
        }
    }
}
