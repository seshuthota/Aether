//! What a few-token pass (`GpuModel::forward_few`, prompt lookup's check
//! of its drafts) costs against one decode step, at position 256: the
//! wall time of the call (record + submit + wait + reading the logits),
//! and with `profile` the GPU time per kind of stage.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_few -- \
//!       testdata/gguf/qwen2.5-0.5b-q4_0.gguf [profile] [f32]
//!
//! `f32`: the few-token products with x in f32 (`qx_gemv_n`), not the
//! int8 ones. Median of 15 calls after 3 warm-ups.

use aether::kernels::vulkan::Gpu;
use aether::model::gpu::{GpuModel, Options};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: vk_few MODEL.gguf [profile] [f32]");
    let rest: Vec<String> = args.collect();
    let (profile, few_f32) = (rest.iter().any(|a| a == "profile"), rest.iter().any(|a| a == "f32"));
    let (config, weights, _) = aether::model::gguf::load(&path, 512).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut m = GpuModel::with(&gpu, config, weights, Options { profile, few_f32, ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
    let pos = 256;
    for p in 0..pos {
        m.forward(&gpu, (p * 7919 + 13) % m.config.vocab_size, p).unwrap();
    }
    println!("# {path} on {} | few-token shader {} | position {pos}", gpu.info.name, m.few_shader());
    println!("tokens,wall_ms,vs_one");
    let median = |f: &mut dyn FnMut()| {
        (0..3).for_each(|_| f());
        let mut v: Vec<f64> = (0..15)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[7]
    };
    let one = median(&mut || {
        std::hint::black_box(m.forward(&gpu, 13, pos).unwrap());
    });
    println!("1,{one:.2},1.00");
    for n in 2..=m.few {
        let tokens: Vec<usize> = (0..n).map(|i| 13 + i).collect();
        let ms = median(&mut || {
            std::hint::black_box(m.forward_few(&gpu, &tokens, pos).unwrap());
        });
        println!("{n},{ms:.2},{:.2}", ms / one);
        if profile {
            let total: f64 = m.batch_profile.iter().map(|(_, s)| s).sum();
            let parts: Vec<String> = m.batch_profile.iter().map(|(k, s)| format!("{k} {:.2}", s * 1e3)).collect();
            println!("#   GPU {:.2} ms: {}", total * 1e3, parts.join(", "));
        }
    }
    if profile {
        let p = m.profile(&gpu).unwrap();
        let total: f64 = p.iter().map(|(_, s)| s).sum();
        println!("# one decode step, GPU {:.2} ms", total * 1e3);
    }
}
