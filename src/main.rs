//! `aether`: generate text with a llama2.c model file or a Hugging Face
//! Llama-style checkpoint directory.
//!
//!   aether stories15M.bin -i "Once upon a time" -n 256
//!   aether SmolLM2-135M/ -i "The capital of France is" -t 0
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
//!
//! After the text, a timing line goes to stderr: prefill (the whole prompt
//! in one batched pass) and decode (one forward pass per token), measured
//! separately.

use aether::model::sampler::Sampler;
use aether::model::{Transformer, hf, llama2c};
use aether::pool::ThreadPool;
use aether::tokenizer::{bytelevel, llama2c as l2c_tok};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The two model families differ only in how text becomes tokens and back,
/// and in which token ends a generation.
enum Text {
    /// llama2.c: BOS first; BOS again means "story over".
    Llama2c(l2c_tok::Tokenizer),
    /// Hugging Face byte-level BPE: no BOS added; stops at end-of-text or
    /// the chat end-of-turn token.
    Hf { tok: Box<bytelevel::Tokenizer>, start: usize, stops: Vec<usize> },
}

impl Text {
    fn encode(&self, prompt: &str) -> Vec<usize> {
        match self {
            Text::Llama2c(t) => t.encode(prompt, true, false),
            Text::Hf { tok, start, .. } => {
                let ids = tok.encode(prompt);
                if ids.is_empty() { vec![*start] } else { ids }
            }
        }
    }

    /// Bytes to print for `next` following `prev`.
    fn piece(&self, prev: usize, next: usize) -> Vec<u8> {
        match self {
            Text::Llama2c(t) => {
                let p = t.decode(prev, next);
                // Like llama2.c's safe_printf: skip lone non-printable bytes.
                if p.len() == 1 && !(p[0].is_ascii_graphic() || p[0].is_ascii_whitespace()) { Vec::new() } else { p }
            }
            Text::Hf { tok, .. } => tok.token_bytes(next),
        }
    }

    /// Bytes to print for the first prompt token, which has no predecessor:
    /// nothing for llama2.c (it is BOS), the token itself otherwise.
    fn first_piece(&self, id: usize) -> Vec<u8> {
        match self {
            Text::Llama2c(_) => Vec::new(),
            Text::Hf { tok, .. } => tok.token_bytes(id),
        }
    }

    fn is_stop(&self, id: usize) -> bool {
        match self {
            Text::Llama2c(_) => id == l2c_tok::BOS,
            Text::Hf { stops, .. } => stops.contains(&id),
        }
    }
}

/// Loads a model directory (Hugging Face) or file (llama2.c).
fn load(model_path: &Path, tok_path: &Path, max_seq: usize) -> (Transformer, Text) {
    if model_path.is_dir() {
        let (config, weights) = hf::load(model_path, max_seq).unwrap_or_else(|e| fail(model_path, e));
        let tok_file = model_path.join("tokenizer.json");
        let tok = bytelevel::Tokenizer::load(&tok_file).unwrap_or_else(|e| fail(&tok_file, e));
        let cfg = std::fs::read_to_string(model_path.join("config.json")).unwrap_or_default();
        let cfg = aether::format::json::Json::parse(&cfg).unwrap_or(aether::format::json::Json::Null);
        let id = |k: &str| cfg.get(k).and_then(|v| v.as_usize());
        let mut stops: Vec<usize> = id("eos_token_id").into_iter().collect();
        stops.extend(tok.token_id("<|im_end|>"));
        let start = id("bos_token_id").or(stops.first().copied()).unwrap_or(0);
        (Transformer::new(config, weights), Text::Hf { tok: Box::new(tok), start, stops })
    } else {
        let (config, weights) = llama2c::load(model_path).unwrap_or_else(|e| fail(model_path, e));
        let tok = l2c_tok::Tokenizer::load(tok_path, config.vocab_size).unwrap_or_else(|e| fail(tok_path, e));
        (Transformer::new(config, weights), Text::Llama2c(tok))
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let mut tok_path = model_path.with_file_name("tokenizer.bin");
    let mut cores = None;
    let (mut temp, mut top_p, mut top_k, mut seed, mut steps, mut prompt) = (1.0f32, 0.9f32, 0usize, 0u64, 256usize, String::new());
    while let Some(flag) = args.next() {
        let val = args.next().unwrap_or_else(|| usage());
        match flag.as_str() {
            "-z" => tok_path = val.into(),
            "-t" => temp = val.parse().unwrap_or_else(|_| usage()),
            "-p" => top_p = val.parse().unwrap_or_else(|_| usage()),
            "-k" => top_k = val.parse().unwrap_or_else(|_| usage()),
            "-s" => seed = val.parse().unwrap_or_else(|_| usage()),
            "-n" => steps = val.parse().unwrap_or_else(|_| usage()),
            "-i" => prompt = val,
            "-c" => cores = Some(aether::sys::affinity::parse_cpu_list(&val).unwrap_or_else(|e| panic!("{e}"))),
            _ => usage(),
        }
    }
    if seed == 0 {
        seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    }

    let (mut model, text) = load(&model_path, &tok_path, steps);
    let steps = steps.min(model.config.seq_len);
    if let Some(cores) = cores {
        // One thread per core; this thread becomes the first of them.
        model.set_pool(ThreadPool::new(&cores).unwrap_or_else(|e| panic!("thread pool on {cores:?}: {e}")));
    }
    let mut sampler = Sampler::new(temp, top_p, top_k, seed);

    let prompt_tokens = text.encode(&prompt);
    let mut out = std::io::stdout().lock();
    out.write_all(&text.first_piece(prompt_tokens[0])).unwrap();
    let mut print = |prev: usize, next: usize| {
        out.write_all(&text.piece(prev, next)).unwrap();
        out.flush().unwrap();
    };

    // Prefill: the whole prompt in one batched pass (GEMM). Its logits are
    // those after the last prompt token, so they already give the first
    // generated token: time-to-first-token is the prefill time.
    let np = prompt_tokens.len().min(steps);
    let t0 = Instant::now();
    let mut logits = model.forward_batch(&prompt_tokens[..np], 0).to_vec();
    let prefill = t0.elapsed();
    for p in 0..np - 1 {
        print(prompt_tokens[p], prompt_tokens[p + 1]);
    }

    // Decode: one forward pass per new token, until `steps` positions or a
    // stop token.
    let t1 = Instant::now();
    let mut pos = np;
    if np == prompt_tokens.len() {
        let mut token = prompt_tokens[np - 1];
        loop {
            let next = sampler.sample(&mut logits);
            if text.is_stop(next) {
                break;
            }
            print(token, next);
            token = next;
            if pos >= steps {
                break;
            }
            logits.copy_from_slice(model.forward(token, pos));
            pos += 1;
        }
    }
    let decode = t1.elapsed();
    drop(print); // ends the closure's borrow of `out`
    writeln!(out).unwrap();

    let rate = |n: usize, t: Duration| n as f64 / t.as_secs_f64();
    eprint!("prefill: {np} tokens in {:.1} ms ({:.1} tok/s)", prefill.as_secs_f64() * 1e3, rate(np, prefill));
    let n_dec = pos - np;
    if n_dec > 0 {
        eprint!(" | decode: {n_dec} tokens in {:.1} ms ({:.1} tok/s)", decode.as_secs_f64() * 1e3, rate(n_dec, decode));
    }
    eprintln!();
}

fn usage() -> ! {
    eprintln!("usage: aether (MODEL.bin | HF_MODEL_DIR) [-z tokenizer.bin] [-t temp] [-p top_p] [-k top_k] [-s seed] [-n steps] [-i prompt] [-c cores]");
    std::process::exit(2)
}

fn fail(path: &Path, e: std::io::Error) -> ! {
    eprintln!("{}: {e}", path.display());
    std::process::exit(1)
}
