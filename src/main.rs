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
//!   -q FMT    store the matrices as f32 (default), q8_0, q4_0 or q5_0
//!             (GGUF files keep the formats they were saved in)
//!   --imatrix FILE  quantize -q q4_0 / q5_0 with llama.cpp's importance
//!             matrix (a GGUF from llama-imatrix), as llama-quantize
//!             --pure --imatrix does
//!   --chat    wrap the prompt in the ChatML chat template (instruct models)
//!   --lookup K  greedy decoding with prompt lookup: up to K draft tokens
//!             copied from the last earlier match of the last --ngram N
//!             tokens (default 2), checked in one batched pass; with -t > 0
//!             by speculative sampling, which keeps the sampler's distribution
//!   --draft PATH  speculative decoding with a draft model of the same
//!             vocabulary (e.g. SmolLM2-135M for SmolLM2-360M; -q applies to
//!             it too): up to --draft-k K (default 4) greedy draft tokens,
//!             checked like --lookup's
//!   --gpu     decode on the GPU through Vulkan (builds with --features
//!             vulkan; a --pure Q4_0 GGUF). With -c, the prompt runs on the
//!             CPU (batched) and its KV cache goes to the GPU; without, on
//!             the GPU (batched as well)
//!   --times   with --gpu: print the time of every decode token
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

use aether::format::imatrix::Imatrix;
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
    let mut imatrix_path: Option<PathBuf> = None;
    let (mut ppl_file, mut ctx, mut chunks, mut kl) = (None::<PathBuf>, 512usize, usize::MAX, false);
    let (mut chat, mut system) = (false, aether::chat::QWEN_DEFAULT_SYSTEM.to_string());
    let mut base_path: Option<PathBuf> = None;
    let (mut lookup, mut ngram) = (0usize, 2usize);
    let (mut draft_path, mut draft_k) = (None::<PathBuf>, 4usize);
    let (mut gpu, mut times) = (false, false);
    let (mut temp, mut top_p, mut top_k, mut seed, mut steps, mut prompt) = (1.0f32, 0.9f32, 0usize, 0u64, 256usize, String::new());
    while let Some(flag) = args.next() {
        if flag == "--kl" || flag == "--chat" || flag == "--gpu" || flag == "--times" {
            match flag.as_str() {
                "--kl" => kl = true,
                "--chat" => chat = true,
                "--times" => times = true,
                _ => gpu = true,
            }
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
            "--draft" => draft_path = Some(val.into()),
            "--draft-k" => draft_k = val.parse().unwrap_or_else(|_| usage()),
            "--ngram" => ngram = val.parse().unwrap_or_else(|_| usage()),
            "--system" => system = val,
            "--base" => base_path = Some(val.into()),
            "--ctx" => ctx = val.parse().unwrap_or_else(|_| usage()),
            "--chunks" => chunks = val.parse().unwrap_or_else(|_| usage()),
            "-q" => format = Format::parse(&val).unwrap_or_else(|| usage()),
            "--imatrix" => imatrix_path = Some(val.into()),
            "-c" => cores = Some(aether::sys::affinity::parse_cpu_list(&val).unwrap_or_else(|e| panic!("{e}"))),
            _ => usage(),
        }
    }
    if seed == 0 {
        seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    }

    let imatrix = imatrix_path.map(|p| {
        let im = aether::format::imatrix::Imatrix::open(&p).unwrap_or_else(|e| fail(&p, e));
        eprintln!("imatrix: {} matrices, {} chunks of {:?}", im.len(), im.chunks, im.datasets);
        im
    });
    let imatrix = imatrix.as_ref();

    if let Some(file) = ppl_file {
        let base = kl.then(|| base_path.unwrap_or_else(|| model_path.clone()));
        return score(&model_path, &tok_path, &file, format, imatrix, ctx, chunks, base, cores);
    }
    if gpu {
        if chat {
            prompt = aether::chat::chatml(&system, &prompt);
        }
        return generate_gpu(&model_path, &prompt, Sampler::new(temp, top_p, top_k, seed), steps, cores.as_deref(), times);
    }
    let mut session = Session::open_imatrix(&model_path, Some(&tok_path), steps, format, imatrix, cores.as_deref()).unwrap_or_else(|e| fail(&model_path, e));
    eprintln!("weights: {}, {:.1} MB", session.model.weights.layers[0].wq.format_name(), session.model.weights.bytes() as f64 / 1e6);
    let mut sampler = Sampler::new(temp, top_p, top_k, seed);
    if chat {
        prompt = aether::chat::chatml(&system, &prompt);
    }
    let mut out = std::io::stdout().lock();
    let mut write = |piece: &[u8]| out.write_all(piece).and_then(|_| out.flush()).is_ok();
    let stats = if lookup > 0 || draft_path.is_some() {
        let t = session.text.encode(&prompt);
        write(&session.text.first_piece(t[0]));
        for p in 0..t.len() - 1 {
            write(&session.text.piece(t[p], t[p + 1]));
        }
        let (stats, look) = if let Some(path) = &draft_path {
            // No cores of its own: it borrows the target's threads.
            let d = Session::open(path, None, steps, format, None).unwrap_or_else(|e| fail(path, e));
            eprintln!("draft: {}, {:.1} MB", d.model.weights.layers[0].wq.format_name(), d.model.weights.bytes() as f64 / 1e6);
            let mut draft = aether::session::Drafter::new(d);
            session.generate_draft(&mut draft, &prompt, &mut sampler, steps, draft_k, write)
        } else {
            session.generate_lookup(&prompt, &mut sampler, steps, lookup, ngram, write)
        };
        eprintln!("\nspeculative: {} passes, {} of {} drafts accepted", look.passes, look.accepted, look.drafted);
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
fn score(model_path: &Path, tok_path: &Path, file: &Path, format: Format, imatrix: Option<&Imatrix>, ctx: usize, chunks: usize, base: Option<PathBuf>, cores: Option<Vec<usize>>) {
    let open = |p: &Path, f: Format, im: Option<&Imatrix>, c: Option<&[usize]>| Session::open_imatrix(p, Some(tok_path), ctx, f, im, c).unwrap_or_else(|e| fail(p, e));
    let Session { mut model, text } = open(model_path, format, imatrix, cores.as_deref());
    let kl = base.is_some();
    let mut base = base.map(|b| open(&b, Format::F32, None, None).model);
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

/// `--gpu`: every position, prompt included, through the GPU decode step
/// (`session::GpuSession`); sampling on the CPU.
#[cfg(feature = "vulkan")]
fn generate_gpu(path: &Path, prompt: &str, mut sampler: Sampler, steps: usize, cores: Option<&[usize]>, times: bool) {
    let s = match cores {
        Some(c) => aether::session::GpuSession::open_hybrid(path, steps, c),
        None => aether::session::GpuSession::open(path, steps),
    };
    let mut s = s.unwrap_or_else(|e| fail(path, e));
    eprintln!("GPU: {}", s.gpu.info.name);
    let mut out = std::io::stdout().lock();
    let t = s.text.encode(prompt);
    let _ = out.write_all(&s.text.first_piece(t[0]));
    for p in 0..t.len() - 1 {
        let _ = out.write_all(&s.text.piece(t[p], t[p + 1]));
    }
    // --times: the time between emitted tokens (one GPU step plus
    // sampling), to see whether decode starts slow and speeds up.
    let mut stamps = times.then(Vec::new);
    let st = s.generate(prompt, &mut sampler, steps, |piece| {
        if let Some(v) = &mut stamps {
            v.push(std::time::Instant::now());
        }
        out.write_all(piece).and_then(|_| out.flush()).is_ok()
    });
    writeln!(out).unwrap();
    if let Some(v) = stamps {
        let ms: Vec<f64> = v.windows(2).map(|w| (w[1] - w[0]).as_secs_f64() * 1e3).collect();
        eprintln!("token ms: {}", ms.iter().map(|m| format!("{m:.1}")).collect::<Vec<_>>().join(" "));
    }
    let rate = |n: usize, t: Duration| n as f64 / t.as_secs_f64();
    let how = if s.cpu.is_some() { "on the CPU" } else { "on the GPU" };
    eprintln!("prefill ({how}): {} tokens in {:.1} ms ({:.1} tok/s) | decode: {} tokens in {:.1} ms ({:.1} tok/s)", st.prefill_tokens, st.prefill.as_secs_f64() * 1e3, rate(st.prefill_tokens, st.prefill), st.decode_tokens, st.decode.as_secs_f64() * 1e3, rate(st.decode_tokens, st.decode));
}

#[cfg(not(feature = "vulkan"))]
fn generate_gpu(_: &Path, _: &str, _: Sampler, _: usize, _: Option<&[usize]>, _: bool) {
    eprintln!("--gpu needs a build with --features vulkan");
    std::process::exit(2)
}

fn usage() -> ! {
    eprintln!("usage: aether (MODEL.bin | HF_MODEL_DIR) [-z tokenizer.bin] [-t temp] [-p top_p] [-k top_k] [-s seed] [-n steps] [-i prompt] [-c cores] [-q f32|q8_0|q4_0|q5_0] [--imatrix FILE]");
    std::process::exit(2)
}

fn fail(path: &Path, e: std::io::Error) -> ! {
    eprintln!("{}: {e}", path.display());
    std::process::exit(1)
}
