//! Models: hyperparameters, weights, the forward pass, and sampling.

pub mod config;
pub mod gguf;
pub mod hf;
pub mod llama2c;
#[cfg(test)]
pub(crate) mod reference;
pub mod sampler;
pub mod transformer;

pub use config::{Config, RopeLayout};
pub use transformer::{NoTrace, Observer, Transformer};

/// Loads any supported model: a `.gguf` file, a Hugging Face checkpoint
/// directory, or a llama2.c `.bin` file. The KV cache is sized for at most
/// `max_seq` positions.
pub fn load_any(path: &std::path::Path, max_seq: usize) -> std::io::Result<(Config, transformer::Weights)> {
    if path.extension().is_some_and(|e| e == "gguf") {
        gguf::load(path, max_seq).map(|(c, w, _)| (c, w))
    } else if path.is_dir() {
        hf::load(path, max_seq)
    } else {
        llama2c::load(path).map(|(mut c, w)| {
            c.seq_len = c.seq_len.min(max_seq);
            (c, w)
        })
    }
}
