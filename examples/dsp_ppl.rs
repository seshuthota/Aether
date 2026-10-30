//! Perplexity of a `--pure` Q4_0 GGUF (or Q4_K_M, without BATCH) with every token through the DSP
//! decode step (`model::dsp::DspModel`), or with BATCH > 1 through its
//! prefill (BATCH tokens a call, products on HMX), by `llama-perplexity`'s
//! procedure (wikitext-2 test, chunks of CTX tokens, the second half of
//! each scored), so the number can be compared with llama.cpp's for the
//! same file. Also prints a greedy continuation, the decode speed and,
//! with BATCH > 1, the prefill speed. With `few` after BATCH, scores
//! through the few-token pass instead (`DspModel::few`, 8 tokens a call;
//! its logits are the decode step's) and prints its time for 1 to 8
//! tokens against one decode step.
//!
//!   cargo run --release --target aarch64-linux-android --features hexagon --example dsp_ppl -- \
//!       testdata/gguf/smollm2-135m-q4_0.gguf [CHUNKS] [CTX] [THREADS] [DMA_MASK] [BATCH] [few]
//!
//!   dsp_ppl MODEL.gguf depth 0,512,2048 [THREADS]
//!
//! `depth` does not score perplexity. It decodes up to each depth (the
//! depths must increase), clears the stage profile, and times 64 more
//! decode steps there, so the print is the same stages as the run above
//! but for a long cache instead of positions 0–63.

use aether::eval::{Score, log_probs};
use aether::kernels::hexagon::{Dsp, MAX_FEW};
use aether::model::dsp::DspModel;
use aether::session::Session;
use aether::weight::Format;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// Temperature, CPU clocks and battery status, read on the phone at `when`.
fn conditions(when: &str) {
    use aether::sys::sysfs::{self, cpu_cur_freq_khz, cpu_scaling_max_freq_khz};
    let zones = sysfs::thermal_zones();
    let one = |z: &sysfs::ThermalZone| format!("{} {:.1} C", z.kind, z.temp_millic().unwrap_or(0) as f64 / 1000.0);
    let zone0 = zones.iter().find(|z| z.index == 0).map(one).unwrap_or_else(|| "zone0 unread".into());
    let hot = zones
        .iter()
        .filter(|z| z.kind.starts_with("cpu"))
        .max_by_key(|z| z.temp_millic().unwrap_or(0))
        .map(|z| format!(", hottest cpu {}", one(z)))
        .unwrap_or_default();
    let mhz = |cpu: usize| {
        let cur = cpu_cur_freq_khz(cpu).unwrap_or(0) / 1000;
        let cap = cpu_scaling_max_freq_khz(cpu).unwrap_or(0) / 1000;
        format!("cpu{cpu} {cur}/{cap} MHz")
    };
    let battery = sysfs::Battery::find().and_then(|b| b.status().ok()).unwrap_or_else(|| "unknown".into());
    println!("# {when}: zone0 {zone0}{hot}, {}, {}, battery {battery}", mhz(0), mhz(7));
    let _ = std::io::stdout().flush();
}

/// 64 decode steps at each depth, one cache filled in order. The stage
/// times are the DSP's qtimer (`DspModel::take_profile`), not wall time.
fn profile_depths(path: PathBuf, spec: &str, threads: u32) {
    const STEPS: usize = 64;
    let depths: Vec<usize> = spec.split(',').map(|s| s.parse().expect("depth")).collect();
    assert!(!depths.is_empty(), "need at least one depth");
    for w in depths.windows(2) {
        assert!(w[1] > w[0], "depths must increase: {spec}");
    }
    let seq = depths.last().copied().unwrap() + STEPS;
    let (config, weights, g) = aether::model::gguf::load(&path, seq).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(config.seq_len >= seq, "model context {} is shorter than {seq}", config.seq_len);
    let text = aether::session::Text::from_gguf(&g).unwrap();
    let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let dsp = Dsp::open(&dir).unwrap_or_else(|e| panic!("{e}"));
    let t0 = Instant::now();
    let mut m = DspModel::new(&dsp, config, weights, threads, 1, false).unwrap_or_else(|e| panic!("{e}"));
    println!("# {} on the DSP, {threads} threads, DMA mask {:#x} | upload {:.0} ms", path.display(), m.dma, t0.elapsed().as_secs_f64() * 1e3);
    conditions("at start");

    let data = std::env::var("AETHER_TESTDATA").unwrap_or_else(|_| "testdata".into());
    let raw = std::fs::read_to_string(PathBuf::from(data).join("wikitext/wikitext-2-raw/wiki.test.raw")).unwrap();
    let tokens = text.encode(&raw);
    assert!(tokens.len() >= seq, "wikitext has {} tokens, need {seq}", tokens.len());

    let mut pos = 0;
    for &depth in &depths {
        if pos < depth {
            println!("# filling positions {pos}..{depth}");
            let _ = std::io::stdout().flush();
            while pos < depth {
                let logits = m.forward(&dsp, tokens[pos], pos).unwrap();
                assert_eq!(aether::kernels::argmax(logits), m.greedy(), "the DSP's argmax at position {pos}");
                pos += 1;
            }
        }
        // Drop the fill. The next decode sees the zeros (`cpu_wrote` on the profile).
        m.take_profile();
        conditions(&format!("before depth {depth}"));
        let t1 = Instant::now();
        let mut dsp_us = 0u64;
        for _ in 0..STEPS {
            let logits = m.forward(&dsp, tokens[pos], pos).unwrap();
            assert_eq!(aether::kernels::argmax(logits), m.greedy(), "the DSP's argmax at position {pos}");
            dsp_us += m.last_us;
            pos += 1;
        }
        let wall = t1.elapsed().as_secs_f64();
        let prof = m.take_profile();
        let total: f64 = prof.iter().sum();
        println!("# depth {depth}: 64 decode steps at positions {depth}..{}", depth + STEPS - 1);
        for (name, us) in aether::model::dsp::STAGES.iter().zip(&prof) {
            println!("#   {name:24} {:8.1} us/token {:5.1}%", us / STEPS as f64, 100.0 * us / total);
        }
        println!(
            "#   {STEPS} steps in {:.1} ms wall: {:.1} tok/s ({:.2} ms per token on the DSP)",
            wall * 1e3,
            STEPS as f64 / wall,
            dsp_us as f64 / STEPS as f64 / 1e3
        );
        conditions(&format!("after depth {depth}"));
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: dsp_ppl MODEL.gguf [CHUNKS CTX THREADS DMA_MASK BATCH few | depth 0,512,2048 [THREADS]]"));
    let first = args.next();
    if first.as_deref() == Some("depth") {
        let spec = args.next().expect("depths, comma-separated and ascending, e.g. 0,512,2048");
        let threads: u32 = args.next().map_or(6, |v| v.parse().unwrap());
        profile_depths(path, &spec, threads);
        return;
    }
    let chunks: usize = first.map_or(20, |v| v.parse().unwrap());
    let ctx: usize = args.next().map_or(512, |v| v.parse().unwrap());
    let threads: u32 = args.next().map_or(6, |v| v.parse().unwrap());
    // Bit k: product k (q/k/v, wo, gate/up, down, classifier) by DMA.
    let dma: u32 = args.next().map_or(aether::model::dsp::DMA_ALL, |v| u32::from_str_radix(v.trim_start_matches("0x"), 16).unwrap());
    let batch: usize = args.next().map_or(1, |v| v.parse().unwrap());
    let few = args.next().is_some_and(|v| v == "few");
    // The tokenizer from a session; the weights again, as stored (the
    // session's copy is repacked for the CPU kernels).
    let text = Session::open(&path, None, ctx, Format::F32, None).unwrap().text;
    let (config, weights, _) = aether::model::gguf::load(&path, ctx).unwrap();
    let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let dsp = Dsp::open(&dir).unwrap_or_else(|e| panic!("{e}"));
    let t0 = Instant::now();
    let vocab = config.vocab_size;
    let mut m = DspModel::new(&dsp, config, weights, threads, batch, batch > 1).unwrap_or_else(|e| panic!("{e}"));
    m.dma = dma;
    println!("# {} on the DSP, {threads} threads, DMA mask {dma:#x}, batch {batch} | upload {:.0} ms", path.display(), t0.elapsed().as_secs_f64() * 1e3);

    // Greedy continuation (eyeball check) and decode speed; with a batch,
    // the prompt by one prefill.
    let prompt = text.encode("The capital of France is");
    let (mut out, mut tok, mut dsp_us) = (String::new(), 0, 0);
    let mut start = 0;
    if batch > 1 {
        let logits = m.prefill(&dsp, &prompt, 0).unwrap();
        tok = aether::kernels::argmax(&logits[logits.len() - vocab..]);
        assert_eq!(tok, m.greedy(), "the DSP's argmax after the prompt");
        // The decode loop below feeds tok first, and prints it.
        start = prompt.len();
        m.take_profile();
    }
    let t1 = Instant::now();
    let steps = 64;
    for p in start..steps {
        let next = if p < prompt.len() { prompt[p] } else { tok };
        if p >= prompt.len() {
            out += &String::from_utf8_lossy(&text.piece(prompt.get(p - 1).copied().unwrap_or(next), next));
        }
        let logits = m.forward(&dsp, next, p).unwrap();
        tok = aether::kernels::argmax(logits);
        assert_eq!(tok, m.greedy(), "the DSP's argmax at position {p}");
        dsp_us += m.last_us;
    }
    let dt = t1.elapsed().as_secs_f64();
    let steps = steps - start;
    let prof = m.take_profile();
    let total: f64 = prof.iter().sum();
    for (name, us) in aether::model::dsp::STAGES.iter().zip(&prof) {
        println!("#   {name:24} {:8.1} us/token {:5.1}%", us / steps as f64, 100.0 * us / total);
    }
    println!("# greedy: The capital of France is{out}");
    println!("# {steps} positions in {:.1} ms: {:.1} tok/s ({:.2} ms per token on the DSP)", dt * 1e3, steps as f64 / dt, dsp_us as f64 / steps as f64 / 1e3);

    let data = std::env::var("AETHER_TESTDATA").unwrap_or_else(|_| "testdata".into());
    let raw = std::fs::read_to_string(PathBuf::from(data).join("wikitext/wikitext-2-raw/wiki.test.raw")).unwrap();
    let tokens = text.encode(&raw);
    let (first, bos) = (ctx / 2, text.chunk_bos());

    // Prefill speed: the first n tokens of the text at position 0.
    if batch > 1 {
        for n in [32, 64, 128, batch].into_iter().filter(|&n| n <= batch && n < ctx) {
            let mut us: Vec<u64> = (0..5)
                .map(|_| {
                    m.prefill(&dsp, &tokens[..n], 0).unwrap();
                    m.last_us
                })
                .collect();
            us.sort();
            let u = us[2] as f64;
            println!("# prefill {n:4} tokens: {:8.1} ms on the DSP, {:7.1} tok/s (median of 5)", u / 1e3, n as f64 / (u / 1e6));
        }
    }
    // The few-token pass's time against one decode step, at position 0.
    if few {
        let median = |m: &mut DspModel, n: usize| {
            let mut us: Vec<u64> = (0..5)
                .map(|_| {
                    if n == 0 {
                        m.forward(&dsp, tokens[0], 0).unwrap();
                    } else {
                        m.few(&dsp, &tokens[..n], 0).unwrap();
                    }
                    m.last_us
                })
                .collect();
            us.sort();
            us[2] as f64
        };
        let one = median(&mut m, 0);
        println!("# decode step: {:.0} us on the DSP (median of 5)", one);
        for n in 1..=MAX_FEW {
            let u = median(&mut m, n);
            println!("# few-token pass, {n} tokens: {u:6.0} us on the DSP, {:.2}x one decode step, {:.0} us per token", u / one, u / n as f64);
        }
    }
    let mut s = Score::default();
    let t2 = Instant::now();
    for ch in 0..chunks.min(tokens.len() / ctx) {
        let mut chunk = tokens[ch * ctx..(ch + 1) * ctx].to_vec();
        if let Some(b) = bos {
            chunk[0] = b;
        }
        let mut score = |p: usize, logits: &[f32]| {
            if p >= first {
                let v = -log_probs(logits)[chunk[p + 1]];
                s.nll += v;
                s.nll2 += v * v;
                s.count += 1;
            }
        };
        if few {
            for p0 in (0..ctx - 1).step_by(MAX_FEW) {
                let n = MAX_FEW.min(ctx - 1 - p0);
                let logits = m.few(&dsp, &chunk[p0..p0 + n], p0).unwrap();
                for i in 0..n {
                    score(p0 + i, &logits[i * vocab..][..vocab]);
                }
            }
        } else if batch > 1 {
            for p0 in (0..ctx - 1).step_by(batch) {
                let n = batch.min(ctx - 1 - p0);
                let logits = m.prefill(&dsp, &chunk[p0..p0 + n], p0).unwrap();
                for i in 0..n {
                    score(p0 + i, &logits[i * vocab..][..vocab]);
                }
            }
        } else {
            for (p, &t) in chunk[..ctx - 1].iter().enumerate() {
                let logits = m.forward(&dsp, t, p).unwrap();
                score(p, logits);
            }
        }
        eprint!("[{}]{:.4},", ch + 1, s.ppl());
    }
    let dt = t2.elapsed().as_secs_f64();
    eprintln!();
    println!("Final estimate: PPL = {:.4} +/- {:.5}  ({} tokens scored, {:.1} s, {:.1} tok/s through the DSP)", s.ppl(), s.ppl_err(), s.count, dt, (chunks * (ctx - 1)) as f64 / dt);
}
