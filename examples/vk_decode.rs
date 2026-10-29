//! Where a GPU decode step's time goes: per kind of stage, from GPU
//! timestamps written after every stage, at several context depths; and
//! the wall time per token around it (submit + wait + the CPU's part).
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_decode -- \
//!       testdata/gguf/qwen2.5-0.5b-q4_0.gguf [plain|split|queued|qprofile|spin|sustain SECONDS [spin]]
//!
//! `queued`: 8 forward passes per submit (timing only), as a GPU that
//! never waits for the CPU would run.
//!
//! `split`: the step's wall time apart from the CPU's part after it
//! (copying the logits out, argmax).
//!
//! `sustain`: decode continuously for SECONDS, as `bench_suite --sustain`
//! does on the CPU (64 tokens, then again from position 0), reporting tok/s
//! every 10 s, with the monitor sampling clocks, temperatures and battery
//! power, so the two can be compared token for token and joule for joule.

use aether::kernels::vulkan::Gpu;
use aether::model::gpu::GpuModel;
use aether::monitor::Monitor;
use std::time::{Duration, Instant};

fn main() {
    let path = std::env::args().nth(1).expect("usage: vk_decode MODEL.gguf [plain|sleep]");
    // plain: no timestamps, only wall time (the timestamps cost ~2 ms per
    // token); spin: also poll the fence instead of sleeping on it.
    let mode = std::env::args().nth(2).unwrap_or_default();
    let depths = [0usize, 256, 1024];
    let (config, weights, _) = aether::model::gguf::load(&path, depths[2] + 64).unwrap();
    let gpu = Gpu::new().unwrap();
    // queued: the step recorded 8 times in one submit, so the GPU runs 8
    // forward passes back to back with no CPU gap between them.
    // qprofile: the same with the per-stage profile.
    let repeat = if mode == "queued" || mode == "qprofile" { 8 } else { 1 };
    let profiled = mode.is_empty() || mode == "qprofile";
    let mut m = GpuModel::with_repeat(&gpu, config, weights, profiled, repeat).unwrap_or_else(|e| panic!("{e}"));
    m.spin = mode == "spin" || std::env::args().nth(4).is_some_and(|a| a == "spin");
    println!("# {path} on {} | {} | fence: {}", gpu.info.name, if mode.is_empty() { "profiled" } else { &mode }, if m.spin { "polled" } else { "slept on" });
    if mode == "sustain" {
        let secs: u64 = std::env::args().nth(3).map_or(120, |v| v.parse().unwrap());
        let monitor = Monitor::start(Duration::from_millis(500), None).unwrap();
        let start = Instant::now();
        let (mut total, mut window_tokens, mut window_start) = (0u64, 0u64, Instant::now());
        println!("sustain_t_s,tok_per_s");
        while start.elapsed() < Duration::from_secs(secs) {
            for p in 0..64 {
                m.forward(&gpu, 100 + p, p).unwrap();
            }
            total += 64;
            window_tokens += 64;
            if window_start.elapsed() >= Duration::from_secs(10) {
                println!("{:.0},{:.1}", start.elapsed().as_secs_f64(), window_tokens as f64 / window_start.elapsed().as_secs_f64());
                (window_tokens, window_start) = (0, Instant::now());
            }
        }
        let secs = start.elapsed().as_secs_f64();
        let report = monitor.stop();
        println!("# sustained: {total} tokens in {secs:.0} s = {:.1} tok/s average", total as f64 / secs);
        println!("# {}", report.summary());
        if let Some((p, src)) = report.best_avg_power_w() {
            println!("# energy per token: {:.3} J ({p:.2} W from {src})", p * secs / total as f64);
        }
        return;
    }
    for d in depths {
        // Warm up (the GPU's clocks rise under load), then start again
        // from position 0: feeding positions 0, 1, … again is allowed.
        for p in 0..64 {
            m.forward(&gpu, 100 + p, p).unwrap();
        }
        let mut pos = 0;
        while pos < d {
            m.forward(&gpu, 100 + pos % 1000, pos).unwrap();
            pos += 1;
        }
        let n = 64;
        if mode == "split" {
            // Wall time of the step (submit + wait) apart from the CPU's
            // part after it: copying the logits out and taking the argmax.
            let (mut step, mut read) = (0.0, 0.0);
            let mut logits = vec![0f32; m.config.vocab_size];
            for _ in 0..n {
                let t0 = Instant::now();
                let out = m.forward(&gpu, 100 + pos % 1000, pos).unwrap();
                let t1 = Instant::now();
                logits.copy_from_slice(out);
                std::hint::black_box(logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)));
                step += (t1 - t0).as_secs_f64();
                read += t1.elapsed().as_secs_f64();
                pos += 1;
            }
            println!("depth {d}: step (submit + wait) {:.2} ms, logits copy + argmax {:.2} ms per token", step / n as f64 * 1e3, read / n as f64 * 1e3);
            continue;
        }
        let mut sums: Vec<(&str, f64)> = Vec::new();
        let t0 = Instant::now();
        for _ in 0..n {
            m.forward(&gpu, 100 + pos % 1000, pos).unwrap();
            pos += 1;
            for (name, t) in m.profile(&gpu).unwrap() {
                match sums.iter_mut().find(|(s, _)| *s == name) {
                    Some(e) => e.1 += t,
                    None => sums.push((name, t)),
                }
            }
        }
        let n = n * repeat;
        let wall = t0.elapsed().as_secs_f64() / n as f64;
        let gpu_total: f64 = sums.iter().map(|s| s.1).sum::<f64>() / n as f64;
        if sums.is_empty() {
            println!("depth {d}: {:.2} ms per token wall ({:.1} tok/s)", wall * 1e3, 1.0 / wall);
            continue;
        }
        println!("depth {d}: {:.2} ms per token wall ({:.1} tok/s), {:.2} ms on the GPU", wall * 1e3, 1.0 / wall, gpu_total * 1e3);
        for (name, t) in sums {
            println!("  {name:<20} {:7.3} ms  {:5.1} %", t / n as f64 * 1e3, t / n as f64 / gpu_total * 100.0);
        }
    }
}
