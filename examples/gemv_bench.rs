//! Phase 3 GEMV benchmark: every step of the NEON matrix-vector kernel, in
//! GB/s of weights streamed, at sizes from cache-resident to DRAM-sized.
//!
//!   cargo run --release --target aarch64-linux-android --example gemv_bench -- --cores 7
//!
//! GB/s = bytes of W (rows × cols × 4) / time; x and out are tiny next to W.
//! Compare each line with the Phase 1 read sweep at the same working-set
//! size: a GEMV can at best match a pure read of the same bytes.

use aether::aligned::AlignedBuf;
use aether::bench::bench;
use aether::kernels::scalar;
use aether::sys::affinity;
use aether::weight::{Format, Weight};
use std::hint::black_box;

type Kernel = fn(&mut [f32], &[f32], &[f32]);

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

    #[allow(unused_mut)] // only the aarch64 build adds the NEON steps
    let mut kernels: Vec<(&str, Kernel)> = vec![("scalar", scalar::matmul)];
    #[cfg(target_arch = "aarch64")]
    {
        use aether::kernels::neon;
        kernels.push(("neon_one_acc", neon::gemv_one_acc));
        kernels.push(("neon_row_4acc", neon::gemv_row_4acc));
        kernels.push(("neon_gemv", neon::gemv));
    }

    // (rows, cols, what): cache-resident, real layers, DRAM-sized.
    let shapes = [
        (288, 288, "stories15M wq (324 KB, L2)"),
        (768, 288, "stories15M w1 (864 KB, L2)"),
        (32000, 288, "stories15M classifier (36.9 MB)"),
        (4864, 896, "Qwen2.5-0.5B up (17.4 MB)"),
        (896, 4864, "Qwen2.5-0.5B down (17.4 MB)"),
        (8192, 8192, "synthetic (256 MB, DRAM)"),
    ];
    println!("# gemv_bench, cores {cores:?}");
    println!("kernel,rows,cols,weight_bytes,median_us,median_gbps,best_gbps,what");
    for (d, n, what) in shapes {
        let mut w = AlignedBuf::zeros(d * n);
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i % 13) as f32 - 6.0) * 0.01;
        }
        let x: Vec<f32> = (0..n).map(|j| ((j % 7) as f32 - 3.0) * 0.1).collect();
        let mut out = vec![0.0f32; d];
        let bytes = (d * n * 4) as f64;
        // Enough runs for ~0.5 s of samples at a few GB/s, at least 5.
        let iters = ((2e9 / bytes) as usize).clamp(5, 2000);
        for (name, k) in &kernels {
            let s = bench(3, iters, || k(black_box(&mut out), black_box(&x), black_box(&w)));
            println!("{name},{d},{n},{},{:.1},{:.2},{:.2},{what}", d * n * 4, s.median / 1e3, bytes / s.median, bytes / s.min);
        }
        // Quantized versions of the same matrix, through the engine's
        // linear() (activation quantized to Q8_0 + integer dots, style b),
        // plus the style-a kernels directly. GB/s at the quantized size.
        for format in [Format::Q8_0, Format::Q4_0] {
            let q = Weight::from(w.clone_buf()).convert(n, format);
            let qbytes = q.bytes() as f64;
            let s = bench(3, iters, || aether::kernels::linear_on(None, black_box(&mut out), black_box(&x), black_box(&q)));
            println!("{format:?}_b_sdot,{d},{n},{},{:.1},{:.2},{:.2},{what}", q.bytes(), s.median / 1e3, qbytes / s.median, qbytes / s.min);
            // The same through the four-row repacked layout, where the CPU
            // uses it (as models do).
            if aether::kernels::repack_weights() {
                let r = Weight::from(w.clone_buf()).convert(n, format).repack(n);
                let s = bench(3, iters, || aether::kernels::linear_on(None, black_box(&mut out), black_box(&x), black_box(&r)));
                println!("{format:?}_b_sdot_x4,{d},{n},{},{:.1},{:.2},{:.2},{what}", r.bytes(), s.median / 1e3, qbytes / s.median, qbytes / s.min);
            }
            #[cfg(target_arch = "aarch64")]
            {
                use aether::kernels::neon_quant as nq;
                let nb = n / 32;
                let s = bench(3, iters, || match &q {
                    Weight::Q8_0(b) => out.iter_mut().enumerate().for_each(|(i, o)| *o = nq::dot_q8_0_f32(black_box(&b[i * nb..(i + 1) * nb]), &x)),
                    Weight::Q4_0(b) => out.iter_mut().enumerate().for_each(|(i, o)| *o = nq::dot_q4_0_f32(black_box(&b[i * nb..(i + 1) * nb]), &x)),
                    _ => unreachable!(),
                });
                println!("{format:?}_a_dequant,{d},{n},{},{:.1},{:.2},{:.2},{what}", q.bytes(), s.median / 1e3, qbytes / s.median, qbytes / s.min);
            }
        }
    }
}
