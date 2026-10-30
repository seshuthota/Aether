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
//! --sustain SECONDS (0 = skip), --warmup N (untimed repeats of each test
//! before the timed ones, default 0), --json (also print one JSON line with
//! the device, model, configuration and results, `{"device":..}`), --monitor
//! (sample clocks, temperatures and battery power during the whole run and
//! add a `system_state` block: what state the SoC was in while it measured),
//! --backend cpu|gpu|dsp (default cpu; gpu needs `--features vulkan`, dsp
//! `--features hexagon`, both a GGUF), --dsp-threads N (default 6).
//!
//! On the GPU and the DSP a prompt runs in passes of the model's batch
//! size; a GPU model that is not all Q4_0 (a Q4_K_M file) has no batched
//! pass and runs its prompt token by token (said in the header). With
//! --sustain, energy per token is battery power × time / tokens for any
//! backend (the phone unplugged, adb over Wi-Fi).
//!
//! Like llama-bench, token ids are synthetic (a fixed pseudo-random
//! sequence): speed does not depend on which tokens they are.

use aether::model::{Config, Transformer, load_any};
use aether::model::transformer::Weights;
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

/// Nearest-rank percentile of `v` (`q` in 0..=1). With few samples the high
/// percentiles collapse onto the maximum; that is what the data supports.
fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((q * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1]
}

/// One Android system property (`getprop`), empty if unavailable (host runs).
fn prop(name: &str) -> String {
    std::process::Command::new("getprop")
        .arg(name)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn jstr(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Peak resident memory in MB from `VmHWM` (kB in /proc/self/status).
fn peak_rss_mb() -> f64 {
    peak_rss().split_whitespace().next().and_then(|k| k.parse::<f64>().ok()).unwrap_or(0.0) / 1024.0
}

/// The engine being measured.
enum Engine {
    Cpu(Box<Transformer>),
    // The model before the device: its buffers are freed while the device lives.
    #[cfg(feature = "vulkan")]
    Gpu(Box<aether::model::gpu::GpuModel>, aether::kernels::vulkan::Gpu),
    #[cfg(feature = "hexagon")]
    Dsp(Box<aether::model::dsp::DspModel>, aether::kernels::hexagon::Dsp),
}

impl Engine {
    fn new(backend: &str, config: Config, weights: Weights, dsp_threads: u32) -> Engine {
        let _ = dsp_threads; // used with --features hexagon
        match backend {
            "cpu" => Engine::Cpu(Box::new(Transformer::new(config, weights))),
            #[cfg(feature = "vulkan")]
            "gpu" => {
                let gpu = aether::kernels::vulkan::Gpu::new().unwrap();
                let m = aether::model::gpu::GpuModel::new(&gpu, config, weights, false).unwrap_or_else(|e| panic!("{e}"));
                Engine::Gpu(Box::new(m), gpu)
            }
            #[cfg(feature = "hexagon")]
            "dsp" => {
                let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
                let dsp = aether::kernels::hexagon::Dsp::open(&dir).unwrap_or_else(|e| panic!("{e}"));
                let m = aether::model::dsp::DspModel::new(&dsp, config, weights, dsp_threads, 128, false).unwrap_or_else(|e| panic!("{e}"));
                Engine::Dsp(Box::new(m), dsp)
            }
            other => panic!("backend {other}: cpu, or gpu / dsp built with --features vulkan / hexagon"),
        }
    }

    fn config(&self) -> &Config {
        match self {
            Engine::Cpu(m) => &m.config,
            #[cfg(feature = "vulkan")]
            Engine::Gpu(m, _) => &m.config,
            #[cfg(feature = "hexagon")]
            Engine::Dsp(m, _) => &m.config,
        }
    }

    /// Whether a prompt goes through a batched pass (else token by token).
    fn batched(&self) -> bool {
        match self {
            #[cfg(feature = "vulkan")]
            Engine::Gpu(m, _) => m.batched(),
            _ => true,
        }
    }

    /// Runs `tokens` as a prompt from position `pos` and returns the most
    /// likely next token.
    fn prefill(&mut self, tokens: &[usize], pos: usize) -> usize {
        match self {
            Engine::Cpu(m) => aether::kernels::argmax(m.forward_batch(tokens, pos)),
            #[cfg(feature = "vulkan")]
            Engine::Gpu(m, gpu) => {
                if !m.batched() {
                    for (i, &t) in tokens.iter().enumerate() {
                        m.forward(gpu, t, pos + i).unwrap();
                    }
                    return m.greedy();
                }
                for (i, c) in tokens.chunks(m.batch).enumerate() {
                    m.forward_batch(gpu, c, pos + i * m.batch).unwrap();
                }
                m.greedy()
            }
            #[cfg(feature = "hexagon")]
            Engine::Dsp(m, dsp) => {
                let b = m.batch;
                for (i, c) in tokens.chunks(b).enumerate() {
                    m.prefill(dsp, c, pos + i * b).unwrap();
                }
                m.greedy()
            }
        }
    }

    fn decode(&mut self, token: usize, pos: usize) {
        match self {
            Engine::Cpu(m) => {
                m.forward(token, pos);
            }
            #[cfg(feature = "vulkan")]
            Engine::Gpu(m, gpu) => {
                m.forward(gpu, token, pos).unwrap();
            }
            #[cfg(feature = "hexagon")]
            Engine::Dsp(m, dsp) => {
                m.forward(dsp, token, pos).unwrap();
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: bench_suite MODEL [flags]"));
    let (mut format, mut cores, mut pp, mut tg, mut depths, mut reps, mut sustain) =
        (Format::F32, None, vec![64, 256, 512], 64usize, vec![0, 256, 512], 3usize, 0u64);
    let (mut warmup, mut json, mut monitor_on) = (0usize, false, false);
    let (mut backend, mut dsp_threads) = ("cpu".to_string(), 6u32);
    while let Some(flag) = args.next() {
        if flag == "--json" || flag == "--monitor" {
            if flag == "--json" {
                json = true;
            } else {
                monitor_on = true;
            }
            continue;
        }
        let val = args.next().expect("flag value");
        match flag.as_str() {
            "-q" => format = Format::parse(&val).expect("format"),
            "-c" => cores = Some(affinity::parse_cpu_list(&val).unwrap()),
            "--pp" => pp = list(&val),
            "--tg" => tg = val.parse().unwrap(),
            "--depth" => depths = list(&val),
            "--reps" => reps = val.parse().unwrap(),
            "--sustain" => sustain = val.parse().unwrap(),
            "--warmup" => warmup = val.parse().unwrap(),
            "--backend" => backend = val,
            "--dsp-threads" => dsp_threads = val.parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    let max_seq = pp.iter().max().unwrap().max(&(depths.iter().max().unwrap() + tg)) + 1;
    let t_load = Instant::now();
    // The GPU and DSP take the matrices as the GGUF stores them.
    let (config, weights) = if backend == "cpu" {
        let (config, weights) = load_any(&path, max_seq).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let w = weights.convert(&config, format, None).unwrap();
        (config, w)
    } else {
        let (config, weights, _) = aether::model::gguf::load(&path, max_seq).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        (config, weights)
    };
    // Every matrix format in the file, in order of first use ("q5_0+q8_0+q4_k+q6_k" for a Q4_K_M).
    let mut formats: Vec<&str> = Vec::new();
    for l in &weights.layers {
        for w in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
            if !formats.contains(&w.format_name()) {
                formats.push(w.format_name());
            }
        }
    }
    let (quant, weight_mb) = (formats.join("+"), weights.bytes() as f64 / 1e6);
    // Until the model is ready: includes the repacking in Transformer::new,
    // the upload to the GPU or the DSP.
    let mut model = Engine::new(&backend, config, weights, dsp_threads);
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    let threads = cores.as_ref().map_or(1, |c: &Vec<usize>| c.len());
    if let (Some(c), Engine::Cpu(m)) = (&cores, &mut model) {
        m.set_pool(ThreadPool::new(c).unwrap());
    }
    let vocab = model.config().vocab_size;
    // Fixed pseudo-random token ids (an LCG), away from special tokens.
    let tokens: Vec<usize> = (0..max_seq as u64).map(|i| 100 + ((i * 2654435761 + 12345) % (vocab as u64 - 200)) as usize).collect();

    // Sampler for the whole run, kept off the benchmark's own cores.
    let monitor = monitor_on.then(|| {
        let busy = cores.as_deref().unwrap_or(&[]);
        let spare: Vec<usize> = affinity::online_cpus().unwrap_or_default().into_iter().filter(|c| !busy.contains(c)).take(1).collect();
        Monitor::start(Duration::from_millis(500), (!spare.is_empty()).then_some(&spare[..])).unwrap()
    });

    println!("# bench_suite {} | {quant} | weights {weight_mb:.1} MB | backend {backend} | cores {:?} ({threads} threads) | load {load_ms:.0} ms", path.display(), cores.as_deref().unwrap_or(&[]));
    if !model.batched() {
        println!("# prompts token by token: this backend has no batched pass for {quant}");
    }
    println!("test,tok_per_s,ms");
    let mut prefill_json = Vec::new();
    for &p in &pp {
        for _ in 0..warmup {
            model.prefill(&tokens[..p], 0);
        }
        let times: Vec<f64> = (0..reps.max(1))
            .map(|_| {
                let t = Instant::now();
                model.prefill(&tokens[..p], 0);
                t.elapsed().as_secs_f64()
            })
            .collect();
        let p95 = percentile(&times, 0.95);
        let t = median(times);
        println!("pp{p},{:.1},{:.1}", p as f64 / t, t * 1e3);
        prefill_json.push(format!("{{\"prompt_tokens\":{p},\"median_ms\":{:.1},\"p95_ms\":{:.1},\"tokens_per_second\":{:.1}}}", t * 1e3, p95 * 1e3, p as f64 / t));
    }
    let mut decode_json = Vec::new();
    for &d in &depths {
        // Every token's own time, across all reps (reps × tg samples), so the
        // per-token median and p95 come from real samples.
        let mut per_token: Vec<f64> = Vec::new();
        let run = |model: &mut Engine, per_token: &mut Vec<f64>| {
            if d > 0 {
                model.prefill(&tokens[..d], 0); // fill the cache (untimed)
            }
            let t = Instant::now();
            for (pos, &tok) in tokens.iter().enumerate().skip(d).take(tg) {
                let tt = Instant::now();
                model.decode(tok, pos);
                per_token.push(tt.elapsed().as_secs_f64());
            }
            t.elapsed().as_secs_f64()
        };
        for _ in 0..warmup {
            run(&mut model, &mut Vec::new());
        }
        let times: Vec<f64> = (0..reps.max(1)).map(|_| run(&mut model, &mut per_token)).collect();
        let t = median(times);
        println!("tg{tg}@d{d},{:.1},{:.1}", tg as f64 / t, t * 1e3);
        let (m, p95) = (median(per_token.clone()), percentile(&per_token, 0.95));
        println!("# tg{tg}@d{d} per token: median {:.2} ms, p95 {:.2} ms ({} samples)", m * 1e3, p95 * 1e3, per_token.len());
        decode_json.push(format!("{{\"context_depth\":{d},\"median_ms_per_token\":{:.2},\"p95_ms_per_token\":{:.2},\"tokens_per_second\":{:.1}}}", m * 1e3, p95 * 1e3, tg as f64 / t));
    }
    // Time to first token for a 64-token prompt: prefill, then the first
    // token is the argmax of the returned logits.
    let t = Instant::now();
    let first = model.prefill(&tokens[..64.min(max_seq)], 0);
    let ttft_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("ttft_pp64,{:.1},{ttft_ms:.1}", 1.0 / t.elapsed().as_secs_f64());
    let _ = first;
    println!("# peak RSS {}", peak_rss());
    let mut system_json = String::new();
    if let Some(m) = monitor {
        let report = m.stop();
        println!("# {}", report.summary());
        let freqs: Vec<String> = report.min_freq_khz().iter().map(|(c, f)| format!("\"cpu{c}\":{}", f.map_or("null".into(), |k| k.to_string()))).collect();
        system_json = format!(
            ",\"system_state\":{{\"battery_status\":{},\"max_temp_c\":{},\"min_freq_khz\":{{{}}},\"avg_power_w\":{}}}",
            jstr(report.battery_status.as_deref().unwrap_or("unknown")),
            report.max_temp_c().map_or("null".into(), |t| format!("{t:.1}")),
            freqs.join(","),
            report.best_avg_power_w().map_or("null".into(), |(p, _)| format!("{p:.2}")),
        );
    }
    if json {
        println!(
            "{{\"device\":{{\"model\":{},\"android_version\":{},\"soc\":{}}},\"model\":{{\"name\":{},\"quantization\":{},\"context_length\":{max_seq}}},\"configuration\":{{\"backend\":{},\"threads\":{threads},\"cores\":{:?},\"warmup_runs\":{warmup},\"measured_runs\":{}}},\"results\":{{\"model_load_ms\":{load_ms:.0},\"prefill\":[{}],\"decode\":[{}],\"ttft_pp64_ms\":{:.1},\"peak_memory_mb\":{:.0}}}{}}}",
            jstr(&prop("ro.product.model")),
            jstr(&prop("ro.build.version.release")),
            jstr(&prop("ro.soc.model")),
            jstr(&path.display().to_string()),
            jstr(&quant),
            jstr(&backend),
            cores.as_deref().unwrap_or(&[]),
            reps.max(1),
            prefill_json.join(","),
            decode_json.join(","),
            ttft_ms,
            peak_rss_mb(),
            system_json,
        );
    }

    if sustain > 0 {
        // Decode continuously (restarting at position 0 every `tg` tokens so
        // the context stays short), reporting tok/s every 10 s, with the
        // monitor sampling clocks, temperature and battery charge.
        let monitor = Monitor::start(Duration::from_millis(500), None).unwrap();
        let start = Instant::now();
        let (mut total, mut window_tokens, mut window_start) = (0u64, 0u64, Instant::now());
        println!("sustain_t_s,tok_per_s");
        while start.elapsed() < Duration::from_secs(sustain) {
            for (pos, &tok) in tokens.iter().enumerate().take(tg) {
                model.decode(tok, pos);
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
