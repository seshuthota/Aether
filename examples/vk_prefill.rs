//! The GPU's prompt pass (`GpuModel::forward_batch`): time per batch and
//! where it goes, per kind of stage, from GPU timestamps; at several batch
//! sizes, from position 0.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_prefill -- \
//!       testdata/gguf/qwen2.5-0.5b-q4_0.gguf   (or any Q4_K_M file: qx_gemm)

use aether::kernels::vulkan::Gpu;
use aether::model::gpu::{GpuModel, Options};
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: vk_prefill MODEL.gguf");
    let (config, weights, _) = aether::model::gguf::load(&path, 1024).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut m = GpuModel::with(&gpu, config, weights, Options { profile: true, ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
    println!("# {path} on {} | batch up to {}", gpu.info.name, m.batch);
    for n in [16usize, 64, 128] {
        let n = n.min(m.batch);
        let tokens: Vec<usize> = (0..n).map(|i| 100 + i).collect();
        // Warm up, then time.
        m.forward_batch(&gpu, &tokens, 0).unwrap();
        let t0 = Instant::now();
        m.forward_batch(&gpu, &tokens, 0).unwrap();
        let wall = t0.elapsed().as_secs_f64();
        let total: f64 = m.batch_profile.iter().map(|s| s.1).sum();
        println!("{n} tokens: {:.1} ms wall ({:.0} tok/s), {:.1} ms on the GPU", wall * 1e3, n as f64 / wall, total * 1e3);
        for (name, t) in &m.batch_profile {
            println!("  {name:<20} {:8.3} ms  {:5.1} %", t * 1e3, t / total * 100.0);
        }
    }
}
