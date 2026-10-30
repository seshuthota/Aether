//! Perplexity of a `--pure` Q4_0 GGUF with every token through the GPU
//! decode step (`model::gpu::GpuModel`), by `llama-perplexity`'s procedure
//! (wikitext-2 test, chunks of CTX tokens, the second half of each scored),
//! so the number can be compared with llama.cpp's for the same file. Also
//! prints a greedy continuation and the decode speed.
//!
//!   cargo run --release --target aarch64-linux-android --features vulkan --example vk_ppl -- \
//!       testdata/gguf/smollm2-135m-q4_0.gguf [CHUNKS] [CTX] [batch|batch-f32]
//!
//! `batch`: score each chunk through `GpuModel::forward_batch` (the prompt
//! path: matrix-matrix products, up to `batch` tokens per pass) instead of
//! token by token through the decode step; f16 products where the GPU has
//! fp16 arithmetic and the model is all Q4_0, with `batch-f32` the f32
//! shader (the shader used is printed).

use aether::eval::{Score, log_probs};
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
    let (batched, f32_gemm) = (mode.starts_with("batch"), mode == "batch-f32");
    // The tokenizer from a session; the weights again, as stored (the
    // session's copy is repacked for the CPU kernels).
    let text = Session::open(&path, None, ctx, Format::F32, None).unwrap().text;
    let (config, weights, _) = aether::model::gguf::load(&path, ctx).unwrap();
    let gpu = Gpu::new().unwrap();
    let t0 = Instant::now();
    let mut m = GpuModel::with(&gpu, config, weights, Options { all_logits: batched, f32_gemm, ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
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
        let mut score = |p: usize, logits: &[f32]| {
            if p >= first {
                let v = -log_probs(logits)[chunk[p + 1]];
                s.nll += v;
                s.nll2 += v * v;
                s.count += 1;
            }
        };
        if batched {
            let mut p = 0;
            while p < ctx - 1 {
                let n = m.batch.min(ctx - 1 - p);
                let all = m.forward_batch(&gpu, &chunk[p..p + n], p).unwrap().to_vec();
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
    }
    let dt = t2.elapsed().as_secs_f64();
    eprintln!();
    println!("Final estimate: PPL = {:.4} +/- {:.5}  ({} tokens scored, {:.1} s, {:.1} tok/s through the GPU{})", s.ppl(), s.ppl_err(), s.count, dt, (chunks * (ctx - 1)) as f64 / dt, if batched { format!(", batches of {}, {}", m.batch, m.gemm_shader()) } else { String::new() });
}
