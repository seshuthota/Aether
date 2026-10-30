//! Quantizes a GGUF (f32 or f16, e.g. from llama.cpp's
//! `convert_hf_to_gguf.py`) into Q8_0, Q4_0 or Q5_0, every matrix in that
//! format, as `llama-quantize --pure [--imatrix FILE]` does, or into
//! llama.cpp's Q4_K_M mix as `llama-quantize [--imatrix FILE] … Q4_K_M`
//! does; for the same input the file is the same, byte for byte
//! (`format::quantize`).
//!
//!   aether-quantize [--imatrix FILE] IN.gguf OUT.gguf q8_0|q4_0|q5_0|q4_k_m

use aether::format::imatrix::Imatrix;
use aether::format::quantize::{Recipe, quantize_with};
use std::path::Path;
use std::time::Instant;

fn usage() -> ! {
    eprintln!("usage: aether-quantize [--imatrix FILE] IN.gguf OUT.gguf q8_0|q4_0|q5_0|q4_k_m");
    std::process::exit(2)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let imatrix_path = match args.iter().position(|a| a == "--imatrix") {
        Some(i) if i + 1 < args.len() => Some(args.drain(i..i + 2).nth(1).unwrap()),
        Some(_) => usage(),
        None => None,
    };
    let [input, output, format] = &args[..] else { usage() };
    let Some(recipe) = Recipe::parse(format) else { usage() };
    let imatrix = imatrix_path.as_ref().map(|p| match Imatrix::open(p) {
        Ok(im) => (im, p.as_str()),
        Err(e) => {
            eprintln!("{p}: {e}");
            std::process::exit(1)
        }
    });
    let t = Instant::now();
    if let Err(e) = quantize_with(Path::new(input), Path::new(output), recipe, imatrix.as_ref().map(|(im, p)| (im, *p))) {
        eprintln!("{input} -> {output}: {e}");
        std::process::exit(1)
    }
    eprintln!("{output}: {recipe:?} in {:.1} s", t.elapsed().as_secs_f64());
}
