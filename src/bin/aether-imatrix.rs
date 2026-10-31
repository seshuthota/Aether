//! An importance matrix from our own forward pass, written in the GGUF
//! format `llama-imatrix` writes (`format::imatrix`), for `aether-quantize
//! --imatrix` or llama.cpp's `llama-quantize --imatrix`. Like
//! `llama-imatrix -m MODEL -f TEXT -o OUT -c CTX --chunks N`: the text cut
//! into chunks of CTX tokens, each run from position 0.
//!
//!   aether-imatrix MODEL.gguf TEXT OUT.gguf [CTX] [CHUNKS]
//!
//! MODEL should be the f32 (or f16) file the quantized one will be made from.

use aether::format::imatrix::Imatrix;
use aether::session::Session;
use aether::weight::Format;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(model), Some(text), Some(out)) = (args.first(), args.get(1), args.get(2)) else {
        eprintln!("usage: aether-imatrix MODEL.gguf TEXT OUT.gguf [CTX] [CHUNKS]");
        std::process::exit(2)
    };
    let ctx: usize = args.get(3).map_or(512, |s| s.parse().expect("CTX"));
    let chunks: usize = args.get(4).map_or(usize::MAX, |s| s.parse().expect("CHUNKS"));
    let cores: Vec<usize> = (0..std::thread::available_parallelism().map_or(1, |n| n.get())).collect();
    let fail = |what: &str, e: std::io::Error| -> ! {
        eprintln!("{what}: {e}");
        std::process::exit(1)
    };
    let mut s = Session::open(std::path::Path::new(model), None, ctx, Format::F32, Some(&cores)).unwrap_or_else(|e| fail(model, e));
    let tokens = s.text.encode(&std::fs::read_to_string(text).unwrap_or_else(|e| fail(text, e)));
    let t = Instant::now();
    let mut im = Imatrix::collect(&mut s.model, &tokens, s.text.chunk_bos(), ctx, chunks);
    im.datasets = vec![text.clone()];
    im.write(out).unwrap_or_else(|e| fail(out, e));
    eprintln!("{out}: {} matrices, {} chunks of {ctx} in {:.1} s", im.len(), im.chunks, t.elapsed().as_secs_f64());
}
