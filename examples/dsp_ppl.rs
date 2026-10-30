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

use aether::eval::{Score, log_probs};
use aether::kernels::hexagon::{Dsp, MAX_FEW};
use aether::model::dsp::DspModel;
use aether::session::Session;
use aether::weight::Format;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: dsp_ppl MODEL.gguf [CHUNKS] [CTX] [THREADS] [DMA_MASK] [BATCH] [few]"));
    let chunks: usize = args.next().map_or(20, |v| v.parse().unwrap());
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
