//! Perplexity of a `--pure` Q4_0 GGUF with every token through the GPU
//! decode step (`model::gpu::GpuModel`), by `llama-perplexity`'s procedure
//! (wikitext-2 test, chunks of CTX tokens, the second half of each scored),
//! so the number can be compared with llama.cpp's for the same file. Also
//! prints a greedy continuation and the decode speed.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_ppl -- \
//!       testdata/gguf/smollm2-135m-q4_0.gguf [CHUNKS] [CTX] [batch|batch-f16|batch-f32|few|few-f32|decode-f32] [BASE.gguf]
//!
//! `batch`: score each chunk through `GpuModel::forward_batch` (the prompt
//! path: matrix-matrix products, up to `batch` tokens per pass) instead of
//! token by token through the decode step; f16 products where the GPU has
//! fp16 arithmetic, with `batch-f32` the f32 shaders (the shader used is
//! printed). A Q4_0 file's products are int8 (`q4_0_gemm_q8`) where the
//! GPU has int8 dot products; `batch-f16` keeps the f16 ones.
//!
//! `decode-f32`: token by token through the decode step with
//! `Options::decode_f32` (x in f32; by default it is in 8 bits, with int8
//! dot products, where the GPU has them).
//!
//! `few`: score each chunk `GpuModel::few` tokens at a time through
//! `GpuModel::forward_few` (the decode shader for several tokens, as
//! speculative decoding checks its drafts): with x in 8 bits and int8
//! dot products where the GPU has them, `few-f32` with x in f32.
//!
//! `BASE.gguf` (the f32 file llama.cpp's `--kl-divergence-base` was made
//! from): also run each chunk through that model on the CPU and print KL
//! divergence and top-token agreement against it, as `aether --ppl --kl`
//! and llama-perplexity's `--kl-divergence` do.

use aether::eval::{Score, log_probs};
use aether::model::Transformer;
use aether::kernels::vulkan::Gpu;
use aether::model::gpu::{GpuModel, Options};
use aether::session::Session;
use aether::weight::Format;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: vk_ppl MODEL.gguf [CHUNKS] [CTX]"));
    let chunks: usize = args.next().map_or(20, |v| v.parse().unwrap());
    let ctx: usize = args.next().map_or(512, |v| v.parse().unwrap());
    let mode = args.next().unwrap_or_default();
    let mut base: Option<Transformer> = args.next().map(|b| Session::open(&PathBuf::from(b), None, ctx, Format::F32, None).unwrap().model);
    let (batched, f32_gemm, f16_gemm, few, few_f32) = (mode.starts_with("batch"), mode == "batch-f32", mode == "batch-f16", mode.starts_with("few"), mode == "few-f32");
    // The tokenizer from a session; the weights again, as stored (the
    // session's copy is repacked for the CPU kernels).
    let text = Session::open(&path, None, ctx, Format::F32, None).unwrap().text;
    let (config, weights, _) = aether::model::gguf::load(&path, ctx).unwrap();
    let gpu = Gpu::new().unwrap();
    let t0 = Instant::now();
    let mut m = GpuModel::with(&gpu, config, weights, Options { all_logits: batched, f32_gemm, f16_gemm, few_f32, decode_f32: mode == "decode-f32", ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
    let vocab = m.config.vocab_size;
    println!("# {} on {} | upload + record {:.0} ms", path.display(), gpu.info.name, t0.elapsed().as_secs_f64() * 1e3);

    // Greedy continuation (eyeball check) and decode speed.
    let prompt = text.encode("The capital of France is");
    let (mut out, mut tok) = (String::new(), 0);
    let t1 = Instant::now();
    let steps = 64;
    for p in 0..steps {
        let next = if p < prompt.len() { prompt[p] } else { tok };
        if p >= prompt.len() {
            out += &String::from_utf8_lossy(&text.piece(prompt.get(p - 1).copied().unwrap_or(next), next));
        }
        let logits = m.forward(&gpu, next, p).unwrap();
        tok = aether::kernels::argmax(logits);
    }
    let dt = t1.elapsed().as_secs_f64();
    println!("# greedy: The capital of France is{out}");
    println!("# {steps} positions in {:.1} ms: {:.1} tok/s", dt * 1e3, steps as f64 / dt);

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
        let reference = base.as_mut().map(|b| b.forward_batch_logits(&chunk, 0, first).to_vec());
        let mut score = |p: usize, logits: &[f32]| {
            if p >= first {
                let lb = reference.as_ref().map(|r| log_probs(&r[(p - first) * vocab..(p - first + 1) * vocab]));
                s.add(&log_probs(logits), chunk[p + 1], lb.as_deref());
            }
        };
        if batched || few {
            let mut p = 0;
            while p < ctx - 1 {
                let n = if few { m.few } else { m.batch }.min(ctx - 1 - p);
                let all = if few { m.forward_few(&gpu, &chunk[p..p + n], p) } else { m.forward_batch(&gpu, &chunk[p..p + n], p) }.unwrap().to_vec();
                for i in 0..n {
                    score(p + i, &all[i * vocab..(i + 1) * vocab]);
                }
                p += n;
            }
        } else {
            for (p, &t) in chunk[..ctx - 1].iter().enumerate() {
                let logits = m.forward(&gpu, t, p).unwrap();
                score(p, logits);
            }
        }
        eprint!("[{}]{:.4},", ch + 1, s.ppl());
        if base.is_some() {
            eprint!("kld {:.5},", s.mean_kld());
        }
    }
    let dt = t2.elapsed().as_secs_f64();
    eprintln!();
    println!("Final estimate: PPL = {:.4} +/- {:.5}  ({} tokens scored, {:.1} s, {:.1} tok/s through the GPU{})", s.ppl(), s.ppl_err(), s.count, dt, (chunks * (ctx - 1)) as f64 / dt, if batched { format!(", batches of {}, {}", m.batch, m.gemm_shader()) } else if few { format!(", {} tokens a pass, {}", m.few, m.few_shader()) } else { String::new() });
    if base.is_some() {
        println!("PPL(base f32) = {:.4}, PPL(Q)/PPL(base) = {:.6}", s.ppl_base(), s.ppl() / s.ppl_base());
        println!("Mean KLD: {:.6}   Same top p: {:.3} %   (time and tok/s include the base model on the CPU)", s.mean_kld(), s.same_top_pct());
    }
}
