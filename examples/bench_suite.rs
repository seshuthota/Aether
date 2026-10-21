//! Phase 7 benchmark suite: prefill and decode speed, time to first token,
//! peak memory, and optionally minutes of sustained decoding with power and
//! temperature — the numbers `results/phase-7.md` is built from.
//!
//!   cargo run --release --target aarch64-linux-android --example bench_suite -- \
//!       testdata/gguf/qwen2.5-0.5b-q4_0.gguf -c 2-7 --sustain 300
//!
//! Flags: -q f32|q8_0|q4_0 (convert at load; GGUF keeps its formats),
//! -c CORES, --pp LIST (prompt lengths, default 64,256,512), --tg N (tokens
//! per decode measurement, default 64), --depth LIST (context already in
//! the cache when decoding starts, default 0,256,512), --reps N (default 3),
//! --sustain SECONDS (0 = skip).
//!
//! Like llama-bench, token ids are synthetic (a fixed pseudo-random
//! sequence): speed does not depend on which tokens they are.

use aether::model::{Transformer, load_any};
use aether::monitor::Monitor;
use aether::pool::ThreadPool;
use aether::sys::affinity;
use aether::weight::Format;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn list(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.parse().expect("number list")).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: bench_suite MODEL [flags]"));
    let (mut format, mut cores, mut pp, mut tg, mut depths, mut reps, mut sustain) =
        (Format::F32, None, vec![64, 256, 512], 64usize, vec![0, 256, 512], 3usize, 0u64);
    while let Some(flag) = args.next() {
        let val = args.next().expect("flag value");
        match flag.as_str() {
            "-q" => format = Format::parse(&val).expect("format"),
            "-c" => cores = Some(affinity::parse_cpu_list(&val).unwrap()),
            "--pp" => pp = list(&val),
            "--tg" => tg = val.parse().unwrap(),
            "--depth" => depths = list(&val),
            "--reps" => reps = val.parse().unwrap(),
            "--sustain" => sustain = val.parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    let max_seq = pp.iter().max().unwrap().max(&(depths.iter().max().unwrap() + tg)) + 1;
    let t_load = Instant::now();
    let (config, weights) = load_any(&path, max_seq).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let weights = weights.convert(&config, format);
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let mut model = Transformer::new(config, weights);
    let threads = cores.as_ref().map_or(1, |c: &Vec<usize>| c.len());
    if let Some(c) = &cores {
        model.set_pool(ThreadPool::new(c).unwrap());
    }
    let vocab = model.config.vocab_size;
    // Fixed pseudo-random token ids (an LCG), away from special tokens.
    let tokens: Vec<usize> = (0..max_seq as u64).map(|i| 100 + ((i * 2654435761 + 12345) % (vocab as u64 - 200)) as usize).collect();

    println!("# bench_suite {} | {} | weights {:.1} MB | cores {:?} ({threads} threads) | load {load_ms:.0} ms", path.display(), model.weights.layers[0].wq.format_name(), model.weights.bytes() as f64 / 1e6, cores.as_deref().unwrap_or(&[]));
    println!("test,tok_per_s,ms");
    for &p in &pp {
        let times: Vec<f64> = (0..reps.max(1))
            .map(|_| {
                let t = Instant::now();
                model.forward_batch(&tokens[..p], 0);
                t.elapsed().as_secs_f64()
            })
            .collect();
        let t = median(times);
        println!("pp{p},{:.1},{:.1}", p as f64 / t, t * 1e3);
    }
    for &d in &depths {
        let times: Vec<f64> = (0..reps.max(1))
            .map(|_| {
                if d > 0 {
                    model.forward_batch(&tokens[..d], 0); // fill the cache (untimed)
                }
                let t = Instant::now();
                for pos in d..d + tg {
                    model.forward(tokens[pos], pos);
                }
                t.elapsed().as_secs_f64()
            })
            .collect();
        let t = median(times);
        println!("tg{tg}@d{d},{:.1},{:.1}", tg as f64 / t, t * 1e3);
    }
    // Time to first token for a 64-token prompt: prefill, then the first
    // token is the argmax of the returned logits.
    let t = Instant::now();
    let first = aether::kernels::argmax(model.forward_batch(&tokens[..64.min(max_seq)], 0));
    println!("ttft_pp64,{:.1},{:.1}", 1.0 / t.elapsed().as_secs_f64(), t.elapsed().as_secs_f64() * 1e3);
    let _ = first;
    println!("# peak RSS {}", peak_rss());

    if sustain > 0 {
        // Decode continuously (restarting at position 0 every `tg` tokens so
        // the context stays short), reporting tok/s every 10 s, with the
        // monitor sampling clocks, temperature and battery charge.
        let monitor = Monitor::start(Duration::from_millis(500), None).unwrap();
        let start = Instant::now();
        let (mut total, mut window_tokens, mut window_start) = (0u64, 0u64, Instant::now());
        println!("sustain_t_s,tok_per_s");
        while start.elapsed() < Duration::from_secs(sustain) {
            for pos in 0..tg {
                model.forward(tokens[pos], pos);
            }
            total += tg as u64;
            window_tokens += tg as u64;
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
    }
}

fn peak_rss() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).map(|l| l["VmHWM:".len()..].trim().to_string()))
        .unwrap_or_else(|| "?".into())
}
