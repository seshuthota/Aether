//! `aether`: generate text with a llama2.c-format model.
//!
//!   aether stories15M.bin -i "Once upon a time" -n 256
//!
//! Flags follow llama2.c's `run` so the two can be compared directly:
//!   -z PATH   tokenizer.bin (default: next to the model)
//!   -t F      temperature, 0 = greedy (default 1.0)
//!   -p F      top-p (default 0.9)
//!   -k N      top-k, 0 = off (default 0; not in llama2.c)
//!   -s N      random seed (default: from the clock)
//!   -n N      positions to run, prompt included (default 256, capped at seq_len)
//!   -i TEXT   prompt
//!   -c LIST   pin to these cores, e.g. 7 or 0-5 (default: no pinning)
//!
//! After the text, a timing line goes to stderr: prefill (the whole prompt
//! in one batched pass) and decode (one forward pass per token), measured
//! separately.

use aether::model::Transformer;
use aether::model::llama2c;
use aether::model::sampler::Sampler;
use aether::tokenizer::llama2c::{BOS, Tokenizer};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let mut tok_path = model_path.with_file_name("tokenizer.bin");
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
            "-c" => {
                let cores = aether::sys::affinity::parse_cpu_list(&val).unwrap_or_else(|e| panic!("{e}"));
                aether::sys::affinity::pin_current_thread(&cores).unwrap_or_else(|e| panic!("pinning to {cores:?}: {e}"));
            }
            _ => usage(),
        }
    }
    if seed == 0 {
        seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    }

    let (config, weights) = llama2c::load(&model_path).unwrap_or_else(|e| fail(&model_path, e));
    let tokenizer = Tokenizer::load(&tok_path, config.vocab_size).unwrap_or_else(|e| fail(&tok_path, e));
    let steps = steps.min(config.seq_len);
    let mut model = Transformer::new(config, weights);
    let mut sampler = Sampler::new(temp, top_p, top_k, seed);

    let prompt_tokens = tokenizer.encode(&prompt, true, false);
    let mut out = std::io::stdout().lock();
    // Like llama2.c's safe_printf: skip lone non-printable bytes.
    let mut print = |prev: usize, next: usize| {
        let piece = tokenizer.decode(prev, next);
        if !(piece.len() == 1 && !(piece[0].is_ascii_graphic() || piece[0].is_ascii_whitespace())) {
            out.write_all(&piece).unwrap();
            out.flush().unwrap();
        }
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

    // Decode: one forward pass per new token, until `steps` positions or BOS.
    let t1 = Instant::now();
    let mut pos = np;
    if np == prompt_tokens.len() {
        let mut token = prompt_tokens[np - 1];
        loop {
            let next = sampler.sample(&mut logits);
            if next == BOS {
                break; // BOS starts a new story: this one is over
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
    eprintln!("usage: aether MODEL.bin [-z tokenizer.bin] [-t temp] [-p top_p] [-k top_k] [-s seed] [-n steps] [-i prompt] [-c cores]");
    std::process::exit(2)
}

fn fail(path: &std::path::Path, e: std::io::Error) -> ! {
    eprintln!("{}: {e}", path.display());
    std::process::exit(1)
}
