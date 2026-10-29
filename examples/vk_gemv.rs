//! The Q4_0 matrix-vector product on the GPU (`shaders/q4_0_gemv.comp`)
//! against the CPU's (`linear_on`, six threads, repacked rows), at the
//! shapes of a Qwen2.5-0.5B decode step. GB/s of Q4_0 weight bytes.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_gemv
//!
//! GPU: GPU timestamps around 200 back-to-back dispatches (no submit or
//! wait in the time), median of 5 after a warm-up. CPU: wall time per
//! call, median of `bench`.

use aether::aligned::AlignedBuf;
use aether::bench::bench;
use aether::kernels::vulkan::{Gpu, MatQ4_0};
use aether::pool::ThreadPool;
use aether::weight::{Format, Weight};
use std::hint::black_box;

fn main() {
    let gpu = Gpu::new().unwrap_or_else(|e| panic!("{e}"));
    let rs = [4u32, 8, 12, 16];
    let ks: Vec<_> = rs.iter().map(|&r| gpu.q4_0_gemv_kernel(r).unwrap()).collect();
    let pool = ThreadPool::new(&[2, 3, 4, 5, 6, 7]).unwrap();
    println!("# {} | CPU: cores 2-7", gpu.info.name);
    println!("shape,rows,cols,q4_0_bytes,{}cpu_us,cpu_gbps", rs.map(|r| format!("gpu_R{r}_us,gpu_R{r}_gbps,")).concat());
    for (d, n, what) in [(896, 896, "wq/wo"), (128, 896, "wk/wv"), (1152, 896, "q+k+v"), (4864, 896, "gate/up"), (9728, 896, "gate+up"), (896, 4864, "down"), (151936, 896, "classifier")] {
        let mut w = AlignedBuf::zeros(d * n);
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 7 % 13) as f32 - 6.0) * 0.01;
        }
        let x: Vec<f32> = (0..n).map(|j| ((j % 7) as f32 - 3.0) * 0.1).collect();
        let q = Weight::from(w.clone_buf()).convert(n, Format::Q4_0);
        let Weight::Q4_0(blocks) = &q else { unreachable!() };
        let bytes = q.bytes() as f64;

        let m = MatQ4_0::new(&gpu, blocks, d).unwrap();
        let mut xb = gpu.buffer(n * 4).unwrap();
        xb.as_mut_slice::<f32>().copy_from_slice(&x);
        let out = gpu.buffer(d * 4).unwrap();
        let reps = 200;
        let mut gpu_cols = String::new();
        for k in &ks {
            let mut warm = 0.0;
            while warm < 0.2 {
                warm += gpu.q4_0_gemv(k, &m, &xb, &out, reps).unwrap();
            }
            let mut t: Vec<f64> = (0..5).map(|_| gpu.q4_0_gemv(k, &m, &xb, &out, reps).unwrap() / reps as f64).collect();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            gpu_cols += &format!("{:.1},{:.1},", t[2] * 1e6, bytes / t[2] / 1e9);
        }

        // The same, one dispatch per submit (as a model's step submits one
        // dispatch of each matrix), with the CPU idle in between.
        let k1 = &ks[rs.iter().position(|&r| r == if d < 4096 { 4 } else if d < 65536 { 8 } else { 16 }).unwrap()];
        let mut single: Vec<f64> = (0..50).map(|_| gpu.q4_0_gemv(k1, &m, &xb, &out, 1).unwrap()).collect();
        single.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("# {what}: one dispatch per submit {:.1} us (median of 50)", single[25] * 1e6);
        for gap_ms in [1u64, 5, 20] {
            let mut v: Vec<f64> = (0..20)
                .map(|_| {
                    std::thread::sleep(std::time::Duration::from_millis(gap_ms));
                    gpu.q4_0_gemv(k1, &m, &xb, &out, 1).unwrap()
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!("#   after {gap_ms} ms idle: {:.1} us (median of 20)", v[10] * 1e6);
        }

        let r = Weight::from(w.clone_buf()).convert(n, Format::Q4_0).repack(n);
        let mut o = vec![0.0f32; d];
        let iters = ((2e9 / bytes) as usize).clamp(20, 5000);
        let s = bench(5, iters, || aether::kernels::linear_on(Some(&pool), black_box(&mut o), black_box(&x), black_box(&r)));
        println!("{what},{d},{n},{},{gpu_cols}{:.1},{:.1}", q.bytes(), s.median / 1e3, bytes / s.median);
    }
}
