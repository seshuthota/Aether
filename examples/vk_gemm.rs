//! The Q4_0 matrix-matrix product (prefill) on the GPU
//! (`shaders/q4_0_gemm.comp`) against the CPU's (`linear_batch_on`: int8
//! activations and `smmla`, six threads, repacked rows), at Qwen2.5-0.5B
//! shapes. GFLOP/s = 2 · rows · cols · tokens / time.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_gemm -- [TOKENS]
//!
//! GPU: GPU timestamps around back-to-back dispatches, median of 5 after a
//! warm-up. CPU: wall time per call, median of `bench`.

use aether::aligned::AlignedBuf;
use aether::bench::bench;
use aether::kernels::vulkan::{Gpu, MatQ4_0};
use aether::pool::ThreadPool;
use aether::weight::{Format, Weight};
use std::hint::black_box;

fn main() {
    let t: usize = std::env::args().nth(1).map_or(256, |v| v.parse().unwrap());
    let gpu = Gpu::new().unwrap_or_else(|e| panic!("{e}"));
    let ks = [gpu.q4_0_gemm_kernel().unwrap()];
    let pool = ThreadPool::new(&[2, 3, 4, 5, 6, 7]).unwrap();
    println!("# {} | {t} tokens | CPU: cores 2-7", gpu.info.name);
    println!("shape,rows,cols,gpu_ms,gpu_gflops,cpu_ms,cpu_gflops");
    for (d, n, what) in [(896, 896, "wq/wo"), (4864, 896, "gate/up"), (896, 4864, "down")] {
        let mut w = AlignedBuf::zeros(d * n);
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 7 % 13) as f32 - 6.0) * 0.01;
        }
        let x: Vec<f32> = (0..t * n).map(|j| ((j % 7) as f32 - 3.0) * 0.1).collect();
        let q = Weight::from(w.clone_buf()).convert(n, Format::Q4_0);
        let Weight::Q4_0(blocks) = &q else { unreachable!() };
        let flops = 2.0 * (d * n * t) as f64;

        let m = MatQ4_0::new(&gpu, blocks, d).unwrap();
        let mut xb = gpu.buffer(t * n * 4).unwrap();
        xb.as_mut_slice::<f32>().copy_from_slice(&x);
        let out = gpu.buffer(t * d * 4).unwrap();
        let mut cols = String::new();
        for k in &ks {
            let mut once = gpu.q4_0_gemm(k, &m, &xb, t, &out, 1).unwrap();
            let reps = ((0.02 / once) as u32).clamp(1, 200);
            let mut warm = 0.0;
            while warm < 0.2 {
                warm += gpu.q4_0_gemm(k, &m, &xb, t, &out, reps).unwrap();
            }
            let mut v: Vec<f64> = (0..5).map(|_| gpu.q4_0_gemm(k, &m, &xb, t, &out, reps).unwrap() / reps as f64).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            once = v[2];
            cols += &format!("{:.2},{:.0},", once * 1e3, flops / once / 1e9);
        }

        let r = Weight::from(w.clone_buf()).convert(n, Format::Q4_0).repack(n);
        let mut o = vec![0.0f32; t * d];
        let s = bench(3, 20, || aether::kernels::linear_batch_on(Some(&pool), black_box(&mut o), black_box(&x), black_box(&r), t));
        println!("{what},{d},{n},{cols}{:.2},{:.0}", s.median / 1e6, flops / s.median);
    }
}
