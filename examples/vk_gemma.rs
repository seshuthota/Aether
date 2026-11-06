//! Gemma 4 through the GPU (`model::gpu::GemmaGpu`): a greedy continuation
//! by the decode step, then perplexity and KL divergence against the
//! log-probabilities llama.cpp saved with `--kl-divergence-base` (its BF16
//! run; `eval::BaseFile`), on that file's tokens, every position through
//! the decode step (`decode`) or the prompt pass (`batch`).
//!
//!   cargo run --release --features vulkan --example vk_gemma -- \
//!       testdata/gguf/gemma-4-E2B-it-Q4_0.gguf KLBASE [CHUNKS] [decode|batch]
//!
//! The numbers compare with `aether --ppl FILE --kl-base KLBASE` (the CPU
//! path) and with `llama-perplexity --kl-divergence` on the same file.

use aether::eval::{BaseFile, perplexity_vs_base};
use aether::kernels::vulkan::Gpu;
use aether::model::gpu::{GemmaGpu, Options};
use aether::session::Session;
use aether::weight::Format;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect("usage: vk_gemma MODEL.gguf KLBASE [CHUNKS] [decode|batch]"));
    let base_path = PathBuf::from(args.next().expect("KLBASE"));
    let chunks: usize = args.next().map_or(4, |v| v.parse().unwrap());
    let batched = args.next().as_deref() == Some("batch");
    let mut base = BaseFile::open(&base_path).unwrap();
    let ctx = base.n_ctx;
    let text = Session::open(&path, None, ctx, Format::F32, None).unwrap().text;
    let (config, weights, _) = aether::model::gguf::load(&path, ctx).unwrap();
    let gpu = Gpu::new().unwrap();
    let t0 = Instant::now();
    let mut m = GemmaGpu::with(&gpu, config, weights, Options { all_logits: batched, ..Options::default() }).unwrap_or_else(|e| panic!("{e}"));
    let vocab = m.config.vocab_size;
    println!("# {} on {} | upload + record {:.0} ms, prompt batch {}", path.display(), gpu.info.name, t0.elapsed().as_secs_f64() * 1e3, m.batch);

    let prompt = text.encode("Once upon a time, in a small village by the sea, there lived");
    let (mut out, mut prev, steps) = (String::new(), 0, prompt.len() + 24);
    let t1 = Instant::now();
    for p in 0..steps {
        let next = if p < prompt.len() { prompt[p] } else { m.greedy() };
        if p >= prompt.len() {
            out += &String::from_utf8_lossy(&text.piece(prev, next));
        }
        m.forward(&gpu, next, p).unwrap();
        prev = next;
    }
    let dt = t1.elapsed().as_secs_f64();
    println!("# greedy: {out:?}");
    println!("# {steps} positions in {:.1} ms: {:.1} tok/s", dt * 1e3, steps as f64 / dt);

    let t2 = Instant::now();
    let s = perplexity_vs_base(
        &mut base,
        text.chunk_bos(),
        chunks,
        |chunk, first| {
            let mut rows = Vec::with_capacity((chunk.len() - first) * vocab);
            if batched {
                let mut p = 0;
                while p < chunk.len() {
                    let n = m.batch.min(chunk.len() - p);
                    let all = m.forward_batch(&gpu, &chunk[p..p + n], p).unwrap();
                    for i in 0..n {
                        if p + i >= first {
                            rows.extend_from_slice(&all[i * vocab..(i + 1) * vocab]);
                        }
                    }
                    p += n;
                }
            } else {
                for (p, &t) in chunk.iter().enumerate() {
                    let l = m.forward(&gpu, t, p).unwrap();
                    if p >= first {
                        rows.extend_from_slice(l);
                    }
                }
            }
            rows
        },
        |i, s| eprint!("[{i}]{:.4} kld {:.5},", s.ppl(), s.mean_kld()),
    )
    .unwrap();
    eprintln!();
    println!(
        "Final estimate: PPL = {:.4} +/- {:.5}  ({} tokens scored, {:.1} s, {})",
        s.ppl(),
        s.ppl_err(),
        s.count,
        t2.elapsed().as_secs_f64(),
        if batched { format!("prompt pass, batches of {}", m.batch) } else { "decode".into() }
    );
    println!("PPL(base file) = {:.4}   Mean KLD: {:.6}   Same top p: {:.3} %", s.ppl_base(), s.mean_kld(), s.same_top_pct());
}
