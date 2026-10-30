//! The Q4_0 matrix-matrix product (prefill) on the GPU, f32
//! (`shaders/q4_0_gemm.comp`) and f16 (`shaders/q4_0_gemm_h.comp`, its
//! time without the activations' transpose), the multi-format
//! `qx_gemm.comp` and `qx_gemm_h.comp` (likewise) on the same matrix in
//! Q4_0 and Q5_0, against the CPU's (`linear_batch_on`: int8
//! activations and `smmla`, six threads, repacked rows), at Qwen2.5-0.5B
//! shapes. GFLOP/s = 2 · rows · cols · tokens / time.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_gemm -- [TOKENS]
//!
//! GPU: GPU timestamps around back-to-back dispatches, median of 5 after a
//! warm-up. CPU: wall time per call, median of `bench`.

use aether::aligned::AlignedBuf;
use aether::bench::bench;
use aether::kernels::vulkan::{Gpu, MatQ, MatQ4_0, MatQ4_0T, MatQT};
use aether::pool::ThreadPool;
use aether::weight::{Format, Weight};
use std::hint::black_box;

fn main() {
    let t: usize = std::env::args().nth(1).map_or(256, |v| v.parse().unwrap());
    let gpu = Gpu::new().unwrap_or_else(|e| panic!("{e}"));
    let ks = [gpu.q4_0_gemm_kernel().unwrap()];
    let pool = if cfg!(target_arch = "aarch64") { Some(ThreadPool::new(&[2, 3, 4, 5, 6, 7]).unwrap()) } else { None };
    println!("# {} | {t} tokens | CPU: cores 2-7", gpu.info.name);
    let f16 = gpu.to_f16t_kernel(false).and_then(|t| Ok((t, gpu.q4_0_gemm_h_kernel(false)?))).ok();
    println!("shape,rows,cols,gpu_ms,gpu_gflops,f16_ms,f16_gflops,qx_q4_0_ms,qx_q4_0_gflops,qx_q5_0_ms,qx_q5_0_gflops,qxh_q4_0_ms,qxh_q4_0_gflops,qxh_q5_0_ms,qxh_q5_0_gflops,cpu_ms,cpu_gflops");
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

        match &f16 {
            Some((tk, mk)) => {
                let mt = MatQ4_0T::new(&gpu, blocks, d).unwrap();
                let xt = gpu.buffer(n * Gpu::f16t_pad(t) * 2).unwrap();
                // Warm up, then the median of 21 back-to-back products.
                gpu.q4_0_gemm_h(tk, mk, &mt, &xb, &xt, t, &out, 5).unwrap();
                let s = gpu.q4_0_gemm_h(tk, mk, &mt, &xb, &xt, t, &out, 21).unwrap();
                cols += &format!("{:.2},{:.0},", s * 1e3, flops / s / 1e9);
            }
            None => cols += ",,",
        }

        // The multi-format shader (`qx_gemm.comp`, for Q4_K_M files) on the
        // same matrix in Q4_0 and in Q5_0 (most of a Q4_K_M file).
        for f in [Format::Q4_0, Format::Q5_0] {
            let mq = MatQ::new(&gpu, &Weight::from(w.clone_buf()).convert(n, f), d).unwrap();
            let k = gpu.qx_gemm_kernel(mq.ty, false, false).unwrap();
            gpu.qx_gemm(&k, &mq, &xb, t, &out, 0, 5).unwrap();
            let s = gpu.qx_gemm(&k, &mq, &xb, t, &out, 0, 21).unwrap() / 21.0;
            cols += &format!("{:.2},{:.0},", s * 1e3, flops / s / 1e9);
        }
        for f in [Format::Q4_0, Format::Q5_0] {
            match &f16 {
                Some((tk, _)) => {
                    let mq = MatQT::new(&gpu, &Weight::from(w.clone_buf()).convert(n, f), d).unwrap();
                    let k = gpu.qx_gemm_h_kernel(mq.ty, false).unwrap();
                    let xt = gpu.buffer(n * Gpu::f16t_pad(t) * 2).unwrap();
                    gpu.qx_gemm_h(tk, &k, &mq, &xb, &xt, t, &out, 5).unwrap();
                    let s = gpu.qx_gemm_h(tk, &k, &mq, &xb, &xt, t, &out, 21).unwrap();
                    cols += &format!("{:.2},{:.0},", s * 1e3, flops / s / 1e9);
                }
                None => cols += ",,",
            }
        }

        // The CPU's repacked kernels are the phone's (NEON); elsewhere the
        // GPU columns only.
        if !cfg!(target_arch = "aarch64") {
            println!("{what},{d},{n},{cols},");
            continue;
        }
        let r = Weight::from(w.clone_buf()).convert(n, Format::Q4_0).repack(n);
        let mut o = vec![0.0f32; t * d];
        let s = bench(3, 20, || aether::kernels::linear_batch_on(pool.as_ref(), black_box(&mut o), black_box(&x), black_box(&r), t));
        println!("{what},{d},{n},{cols}{:.2},{:.0}", s.median / 1e6, flops / s.median);
    }
}
