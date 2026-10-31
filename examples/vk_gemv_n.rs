//! The decode shader for several tokens at once (`qx_gemv_n_kernel`, the
//! products of a few-token pass: speculative decoding's check) against the
//! one-token decode shader (`qx_gemv_kernel`), at Qwen2.5-0.5B's shapes,
//! in Q4_0 and Q5_0. GPU time per dispatch; a pass of n tokens pays off
//! when it costs much less than n one-token dispatches.
//!
//! Path `f32` is `qx_gemv_n` (x in f32). Path `q8` is `qx_gemv_q8` (x
//! quantized to 8 bits by `quant_x`, int8 dot products; where the GPU has
//! them), with max_tokens 1 too; `quant_us` is the time of `quant_x` for
//! those tokens, not included in `us`.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_gemv_n
//!
//! GPU timestamps around back-to-back dispatches, the median of 5 after a
//! warm-up.

use aether::aligned::AlignedBuf;
use aether::kernels::vulkan::{Gpu, MatQ};
use aether::weight::{Format, Weight};

fn main() {
    let gpu = Gpu::new().unwrap_or_else(|e| panic!("{e}"));
    println!("# {} | int8 dot products: {}, accelerated: {}", gpu.info.name, gpu.info.int_dot, gpu.info.int_dot_accelerated);
    println!("path,shape,format,rows,cols,one_token_us,max_tokens,tokens,us,us_per_token,vs_one,quant_us");
    let median = |f: &mut dyn FnMut(u32) -> f64| {
        let reps = ((0.02 / f(1)) as u32).clamp(1, 200);
        f(reps);
        let mut v: Vec<f64> = (0..5).map(|_| f(reps) / reps as f64).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[2]
    };
    for (d, n, what) in [(896, 896, "wq/wo"), (9728, 896, "gate/up"), (896, 4864, "down")] {
        let mut w = AlignedBuf::zeros(d * n);
        for (i, v) in w.iter_mut().enumerate() {
            *v = ((i * 7 % 13) as f32 - 6.0) * 0.01;
        }
        let mut x = gpu.buffer(8 * n * 4).unwrap();
        for (j, v) in x.as_mut_slice::<f32>().iter_mut().enumerate() {
            *v = ((j % 7) as f32 - 3.0) * 0.1;
        }
        let out = gpu.buffer(8 * d * 4).unwrap();
        let xq = gpu.buffer(8 * (n / 32) * 48).unwrap();
        for f in [Format::Q4_0, Format::Q5_0] {
            let m = MatQ::new(&gpu, &Weight::from(w.clone_buf()).convert(n, f), d).unwrap();
            let k1 = gpu.qx_gemv_kernel(m.ty, if d >= 4096 { 8 } else { 4 }, false, false, false).unwrap();
            let one = median(&mut |r| gpu.qx_gemv(&k1, &m, &x, &out, r).unwrap());
            for max in [2, 4, 8] {
                let kn = gpu.qx_gemv_n_kernel(m.ty, false, false, max).unwrap();
                for t in [1, 2, 4, 8].into_iter().filter(|&t| t <= max as usize) {
                    let s = median(&mut |r| gpu.qx_gemv_n(&kn, &m, &x, t, &out, r).unwrap());
                    println!("f32,{what},{f:?},{d},{n},{:.1},{max},{t},{:.1},{:.1},{:.2},0", one * 1e6, s * 1e6, s * 1e6 / t as f64, s / one);
                }
            }
            if !gpu.info.int_dot {
                continue;
            }
            let qk = gpu.quant_x_kernel(false).unwrap();
            for max in [1, 2, 4, 8] {
                let kq = gpu.qx_gemv_q8_kernel(m.ty, false, max).unwrap();
                for t in [1, 2, 4, 8].into_iter().filter(|&t| t <= max as usize) {
                    let q = median(&mut |r| gpu.quant_x(&qk, &x, n / 32, t, &xq, r).unwrap());
                    let s = median(&mut |r| gpu.qx_gemv_q8(&kq, &m, &xq, t, &out, r).unwrap());
                    println!("q8,{what},{f:?},{d},{n},{:.1},{max},{t},{:.1},{:.1},{:.2},{:.1}", one * 1e6, s * 1e6, s * 1e6 / t as f64, s / one, q * 1e6);
                }
            }
        }
    }
}
