//! How far apart the two ways of running a token are: decode (one token at
//! a time, `sdot` GEMV, chunked attention past 64 positions) and the batched
//! pass (prefill / lookup verification, `smmla`). Greedy-decodes 100 tokens
//! after a chat prompt, then re-runs the same tokens in batches of 1, 2 and
//! 5 and prints the largest logit difference.
//!
//!   cargo run --release --target aarch64-linux-android --example logit_gap -- MODEL PROMPT_FILE
//!
//! On f32 weights the paths agree within ~5e-5; with Q8_0 / Q4_0 they differ
//! by up to ~0.6, because activations are re-quantized to int8 before every
//! matmul and a rounding difference of 1e-7 occasionally flips one int8
//! value (results/phase-6.md, "Prompt lookup decoding").
use aether::session::Session;
use aether::weight::Format;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cores: Vec<usize> = (2..8).collect();
    let mut s = Session::open(std::path::Path::new(&a[1]), None, 512, Format::F32, Some(&cores)).unwrap();
    let prompt = aether::chat::chatml(aether::chat::QWEN_DEFAULT_SYSTEM, std::fs::read_to_string(&a[2]).unwrap().trim_end());
    let mut toks = s.text.encode(&prompt);
    let np = toks.len();
    let mut logits = s.model.forward_batch(&toks, 0).to_vec();
    let mut dec = Vec::new();
    for _ in 0..100 {
        let t = aether::kernels::argmax(&logits);
        toks.push(t);
        logits.copy_from_slice(s.model.forward(t, toks.len() - 1));
        dec.push(logits.clone());
    }
    let v = logits.len();
    for bs in [1usize, 2, 5] {
        let mut worst = 0.0f32;
        let mut first = 0.0f32;
        for b in (0..100 - bs).step_by(bs) {
            let rows = s.model.forward_batch_logits(&toks[np + b..np + b + bs], np + b, 0).to_vec();
            for r in 0..bs {
                let d = rows[r * v..(r + 1) * v].iter().zip(&dec[b + r]).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
                if b == 0 && r == 0 { first = d; }
                worst = worst.max(d);
            }
        }
        println!("batch {bs}: first {first:.5} worst {worst:.5}");
    }
}
