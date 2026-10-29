//! Perplexity of a `--pure` Q4_0 GGUF with every token through the DSP
//! decode step (`model::dsp::DspModel`), by `llama-perplexity`'s procedure
//! (wikitext-2 test, chunks of CTX tokens, the second half of each scored),
//! so the number can be compared with llama.cpp's for the same file. Also
//! prints a greedy continuation and the decode speed.
//!
//!   cargo run --release --target aarch64-linux-android --features hexagon --example dsp_ppl -- \
//!       testdata/gguf/smollm2-135m-q4_0.gguf [CHUNKS] [CTX] [THREADS]

use aether::eval::{Score, log_probs};
use aether::kernels::hexagon::Dsp;
use aether::model::dsp::DspModel;
use aether::session::Session;
use aether::weight::Format;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: dsp_ppl MODEL.gguf [CHUNKS] [CTX] [THREADS]"));
    let chunks: usize = args.next().map_or(20, |v| v.parse().unwrap());
    let ctx: usize = args.next().map_or(512, |v| v.parse().unwrap());
    let threads: u32 = args.next().map_or(6, |v| v.parse().unwrap());
    // The tokenizer from a session; the weights again, as stored (the
    // session's copy is repacked for the CPU kernels).
    let text = Session::open(&path, None, ctx, Format::F32, None).unwrap().text;
    let (config, weights, _) = aether::model::gguf::load(&path, ctx).unwrap();
    let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
    let dsp = Dsp::open(&dir).unwrap_or_else(|e| panic!("{e}"));
    let t0 = Instant::now();
    let mut m = DspModel::new(&dsp, config, weights, threads).unwrap_or_else(|e| panic!("{e}"));
    println!("# {} on the DSP, {threads} threads | upload {:.0} ms", path.display(), t0.elapsed().as_secs_f64() * 1e3);

    // Greedy continuation (eyeball check) and decode speed.
    let prompt = text.encode("The capital of France is");
    let (mut out, mut tok, mut dsp_us) = (String::new(), 0, 0);
    let t1 = Instant::now();
    let steps = 64;
    for p in 0..steps {
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
    let mut s = Score::default();
    let t2 = Instant::now();
    for ch in 0..chunks.min(tokens.len() / ctx) {
        let mut chunk = tokens[ch * ctx..(ch + 1) * ctx].to_vec();
        if let Some(b) = bos {
            chunk[0] = b;
        }
        for (p, &t) in chunk[..ctx - 1].iter().enumerate() {
            let logits = m.forward(&dsp, t, p).unwrap();
            if p >= first {
                let v = -log_probs(logits)[chunk[p + 1]];
                s.nll += v;
                s.nll2 += v * v;
                s.count += 1;
            }
        }
        eprint!("[{}]{:.4},", ch + 1, s.ppl());
    }
    let dt = t2.elapsed().as_secs_f64();
    eprintln!();
    println!("Final estimate: PPL = {:.4} +/- {:.5}  ({} tokens scored, {:.1} s, {:.1} tok/s through the DSP)", s.ppl(), s.ppl_err(), s.count, dt, (chunks * (ctx - 1)) as f64 / dt);
}
