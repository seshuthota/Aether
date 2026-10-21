//! Phase 3 GEMM benchmark: prefill-shaped products in GFLOPS.
//!
//!   cargo run --release --target aarch64-linux-android --example gemm_bench -- --cores 7
//!
//! Each case computes out[t] = W · x[t] for T tokens: 2·T·d·n floating-point
//! operations (one multiply and one add per weight per token). Three ways:
//! the scalar loop, the NEON GEMV once per token (weights re-read for every
//! token), and the 4×4-tile NEON GEMM (weights reused across 4 tokens per
//! load, rows reused across all tokens while they sit in cache).

use aether::aligned::AlignedBuf;
use aether::bench::bench;
use aether::kernels::scalar;
use aether::sys::affinity;
use std::hint::black_box;

type Batched = fn(&mut [f32], &[f32], &[f32], usize);

fn main() {
    let mut cores = vec![7];
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--cores" => cores = affinity::parse_cpu_list(&it.next().expect("--cores LIST")).unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    affinity::pin_current_thread(&cores).unwrap();

    #[allow(unused_mut)] // only the aarch64 build adds the NEON variants
    let mut kernels: Vec<(&str, Batched)> = vec![("scalar", scalar::matmul_batch)];
    #[cfg(target_arch = "aarch64")]
    {
        use aether::kernels::neon;
        fn gemv_per_token(out: &mut [f32], x: &[f32], w: &[f32], t: usize) {
            let (n, d) = (x.len() / t, out.len() / t);
            for (o, xi) in out.chunks_exact_mut(d).zip(x.chunks_exact(n)) {
                neon::gemv(o, xi, w);
            }
        }
        kernels.push(("neon_gemv_per_token", gemv_per_token));
        kernels.push(("neon_gemm", neon::gemm));
    }

    // (T tokens, d rows, n cols)
    let shapes = [(32, 288, 288), (64, 768, 288), (64, 288, 768), (128, 896, 896), (128, 4864, 896), (128, 896, 4864), (256, 1024, 1024)];
    println!("# gemm_bench, cores {cores:?}");
    println!("kernel,tokens,rows,cols,median_ms,median_gflops,best_gflops");
    for (t, d, n) in shapes {
        let mut w = AlignedBuf::zeros(d * n);
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 0.01;
        }
        let mut x = AlignedBuf::zeros(t * n);
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.1;
        }
        let mut out = AlignedBuf::zeros(t * d);
        let flops = 2.0 * (t * d * n) as f64;
        let iters = ((5e9 / flops) as usize).clamp(3, 500);
        for (name, k) in &kernels {
            let s = bench(2, iters, || k(black_box(&mut out), black_box(&x), black_box(&w), t));
            println!("{name},{t},{d},{n},{:.3},{:.2},{:.2}", s.median / 1e6, flops / s.median, flops / s.min);
        }
    }
}
