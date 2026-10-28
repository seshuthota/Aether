//! `aether`: generate text with a llama2.c model file, a Hugging Face
//! checkpoint directory (llama / qwen2), or a GGUF file.
//!
//!   aether stories15M.bin -i "Once upon a time" -n 256
//!   aether SmolLM2-135M/ -i "The capital of France is" -t 0
//!   aether qwen2.5-0.5b-q4_0.gguf --chat -i "Why is the sky blue?" -c 2-7
//!
//! Flags follow llama2.c's `run` so the two can be compared directly:
//!   -z PATH   tokenizer.bin for a .bin model (default: next to the model)
//!   -t F      temperature, 0 = greedy (default 1.0)
//!   -p F      top-p (default 0.9)
//!   -k N      top-k, 0 = off (default 0; not in llama2.c)
//!   -s N      random seed (default: from the clock)
//!   -n N      positions to run, prompt included (default 256, capped at seq_len)
//!   -i TEXT   prompt
//!   -c LIST   cores to run on, one thread each, e.g. 7 or 0-7 (default:
//!             one unpinned thread)
//!   -q FMT    store the matrices as f32 (default), q8_0 or q4_0 (GGUF
//!             files keep the formats they were saved in)
//!   --chat    wrap the prompt in the ChatML chat template (instruct models)
//!   --lookup K  greedy decoding with prompt lookup: up to K draft tokens
//!             copied from the last earlier match of the last --ngram N
//!             tokens (default 2), checked in one batched pass (ignores -t)
//!   --system TEXT   system message for --chat (default: Qwen's)
//!
//! Scoring mode (like llama.cpp's llama-perplexity), instead of generating:
//!   --ppl FILE    perplexity of the model on the text in FILE
//!   --ctx N       chunk length (default 512)
//!   --chunks N    score only the first N chunks (default: all)
//!   --kl          also load the f32 model and report KL divergence and
//!                 top-token agreement of the -q model against it
//!   --base PATH   the reference model for --kl (default: MODEL as f32;
//!                 give the HF directory or f32 GGUF when MODEL is a
//!                 quantized GGUF)
//!
//! After the text, a timing line goes to stderr: prefill (the whole prompt
//! in one batched pass) and decode (one forward pass per token), measured
//! separately.

use aether::model::sampler::Sampler;
use aether::session::Session;
use aether::weight::Format;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let mut tok_path = model_path.with_file_name("tokenizer.bin");
    let mut cores = None;
    let mut format = Format::F32;
    let (mut ppl_file, mut ctx, mut chunks, mut kl) = (None::<PathBuf>, 512usize, usize::MAX, false);
    let (mut chat, mut system) = (false, aether::chat::QWEN_DEFAULT_SYSTEM.to_string());
    let mut base_path: Option<PathBuf> = None;
    let (mut lookup, mut ngram) = (0usize, 2usize);
    let (mut temp, mut top_p, mut top_k, mut seed, mut steps, mut prompt) = (1.0f32, 0.9f32, 0usize, 0u64, 256usize, String::new());
    while let Some(flag) = args.next() {
        if flag == "--kl" || flag == "--chat" {
            if flag == "--kl" { kl = true } else { chat = true }
            continue;
        }
        let val = args.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "-z" => tok_path = val.into(),
            "-t" => temp = val.parse().unwrap_or_else(|_| usage()),
            "-p" => top_p = val.parse().unwrap_or_else(|_| usage()),
            "-k" => top_k = val.parse().unwrap_or_else(|_| usage()),
            "-s" => seed = val.parse().unwrap_or_else(|_| usage()),
            "-n" => steps = val.parse().unwrap_or_else(|_| usage()),
            "-i" => prompt = val,
            "--ppl" => ppl_file = Some(val.into()),
            "--lookup" => lookup = val.parse().unwrap_or_else(|_| usage()),
            "--ngram" => ngram = val.parse().unwrap_or_else(|_| usage()),
            "--system" => system = val,
            "--base" => base_path = Some(val.into()),
            "--ctx" => ctx = val.parse().unwrap_or_else(|_| usage()),
            "--chunks" => chunks = val.parse().unwrap_or_else(|_| usage()),
            "-q" => format = Format::parse(&val).unwrap_or_else(|| usage()),
            "-c" => cores = Some(aether::sys::affinity::parse_cpu_list(&val).unwrap_or_else(|e| panic!("{e}"))),
            _ => usage(),
        }
    }
    if seed == 0 {
        seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    }

    if let Some(file) = ppl_file {
        let base = kl.then(|| base_path.unwrap_or_else(|| model_path.clone()));
        return score(&model_path, &tok_path, &file, format, ctx, chunks, base, cores);
    }
    let mut session = Session::open(&model_path, Some(&tok_path), steps, format, cores.as_deref()).unwrap_or_else(|e| fail(&model_path, e));
    eprintln!("weights: {}, {:.1} MB", session.model.weights.layers[0].wq.format_name(), session.model.weights.bytes() as f64 / 1e6);
    let mut sampler = Sampler::new(temp, top_p, top_k, seed);
    if chat {
        prompt = aether::chat::chatml(&system, &prompt);
    }
    let mut out = std::io::stdout().lock();
    let mut write = |piece: &[u8]| out.write_all(piece).and_then(|_| out.flush()).is_ok();
    let stats = if lookup > 0 {
        let t = session.text.encode(&prompt);
        write(&session.text.first_piece(t[0]));
        for p in 0..t.len() - 1 {
            write(&session.text.piece(t[p], t[p + 1]));
        }
        let (stats, look) = session.generate_lookup(&prompt, steps, lookup, ngram, write);
        eprintln!("\nlookup: {} passes, {} of {} drafts accepted", look.passes, look.accepted, look.drafted);
        stats
    } else {
        session.generate(&prompt, &mut sampler, steps, true, write)
    };
    writeln!(out).unwrap();

    let rate = |n: usize, t: Duration| n as f64 / t.as_secs_f64();
    let (np, n_dec) = (stats.prefill_tokens, stats.decode_tokens);
    eprint!("prefill: {np} tokens in {:.1} ms ({:.1} tok/s)", stats.prefill.as_secs_f64() * 1e3, rate(np, stats.prefill));
    if n_dec > 0 {
        eprint!(" | decode: {n_dec} tokens in {:.1} ms ({:.1} tok/s)", stats.decode.as_secs_f64() * 1e3, rate(n_dec, stats.decode));
    }
    eprintln!(" | peak RSS {}", peak_rss());
}

/// The process's peak resident memory (`VmHWM` in /proc/self/status): how
/// much RAM the run actually held at its worst moment. Memory-mapped weight
/// pages count while resident, but the kernel can drop them again.
fn peak_rss() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).map(|l| l["VmHWM:".len()..].trim().to_string()))
        .unwrap_or_else(|| "?".into())
}

/// `--ppl` mode: perplexity (and with `--kl`, KL divergence against the f32
/// model) over a text file, printed like llama-perplexity.
#[allow(clippy::too_many_arguments)]
fn score(model_path: &Path, tok_path: &Path, file: &Path, format: Format, ctx: usize, chunks: usize, base: Option<PathBuf>, cores: Option<Vec<usize>>) {
    let open = |p: &Path, f: Format, c: Option<&[usize]>| Session::open(p, Some(tok_path), ctx, f, c).unwrap_or_else(|e| fail(p, e));
    let Session { mut model, text } = open(model_path, format, cores.as_deref());
    let kl = base.is_some();
    let mut base = base.map(|b| open(&b, Format::F32, None).model);
    let raw = std::fs::read_to_string(file).unwrap_or_else(|e| fail(file, e));
    let t0 = Instant::now();
    let tokens = text.encode(&raw);
    eprintln!("{}: {} tokens (tokenized in {:.0} ms), weights {} {:.1} MB", file.display(), tokens.len(), t0.elapsed().as_secs_f64() * 1e3, model.weights.layers[0].wq.format_name(), model.weights.bytes() as f64 / 1e6);
    let t1 = Instant::now();
    let s = aether::eval::perplexity(&mut model, base.as_mut(), &tokens, text.chunk_bos(), ctx, chunks, |i, s| {
        eprint!("[{i}]{:.4},", s.ppl());
    });
    eprintln!();
    println!("Final estimate: PPL = {:.4} +/- {:.5}  ({} tokens scored in {:.1} s)", s.ppl(), s.ppl_err(), s.count, t1.elapsed().as_secs_f64());
    if kl {
        println!("PPL(base f32) = {:.4}, PPL(Q)/PPL(base) = {:.6}", s.ppl_base(), s.ppl() / s.ppl_base());
        println!("Mean KLD: {:.6}   Same top p: {:.3} %", s.mean_kld(), s.same_top_pct());
    }
}

fn usage() -> ! {
    eprintln!("usage: aether (MODEL.bin | HF_MODEL_DIR) [-z tokenizer.bin] [-t temp] [-p top_p] [-k top_k] [-s seed] [-n steps] [-i prompt] [-c cores] [-q f32|q8_0|q4_0]");
    std::process::exit(2)
}

fn fail(path: &Path, e: std::io::Error) -> ! {
    eprintln!("{}: {e}", path.display());
    std::process::exit(1)
}
