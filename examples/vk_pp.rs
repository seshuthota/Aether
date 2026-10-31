//! The GPU's prompt pass timed as `llama-bench` times `pp N`: an N-token
//! prompt from an empty cache, fed `GpuModel::batch` tokens at a time
//! (`forward_batch`), one warm-up run, then the mean of REPS runs, logits
//! for the last token only. So the numbers can be put next to
//! llama-bench's in the same session.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_pp -- \
//!       testdata/gguf/qwen2.5-0.5b-q4_0.gguf [f16] [REPS]

use aether::kernels::vulkan::Gpu;
use aether::model::gpu::{GpuModel, Options};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: vk_pp MODEL.gguf [f16] [REPS]");
    let f16_gemm = args.iter().any(|a| a == "f16");
    let reps: usize = args.iter().skip(1).find_map(|a| a.parse().ok()).unwrap_or(5);
    let (config, weights, _) = aether::model::gguf::load(path, 1024).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut m = GpuModel::with(&gpu, config, weights, Options { f16_gemm, ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
    println!("# {path} on {} | batches of {}, {} | mean of {reps} runs", gpu.info.name, m.batch, m.gemm_shader());
    println!("test,tok_per_s,ms");
    for n in [64usize, 128, 256, 512] {
        let tokens: Vec<usize> = (0..n).map(|i| 100 + i).collect();
        let mut run = || {
            let mut p = 0;
            while p < n {
                let k = m.batch.min(n - p);
                std::hint::black_box(m.forward_batch(&gpu, &tokens[p..p + k], p).unwrap());
                p += k;
            }
        };
        run();
        let t0 = Instant::now();
        (0..reps).for_each(|_| run());
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("pp{n},{:.1},{ms:.1}", n as f64 * 1e3 / ms);
    }
}
