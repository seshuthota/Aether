//! Model hyperparameters: the numbers that fix every tensor's shape.

/// How RoPE pairs up the elements of a head (see `kernels::scalar`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// `(x[2i], x[2i+1])`: llama2.c, and Llama-architecture GGUF files.
    AdjacentPairs,
    /// `(x[i], x[i + d/2])` ("rotate_half"): HF transformers, Qwen2 GGUF.
    Halves,
}

/// Shape and constants of a decoder-only transformer.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Width of the residual stream (the embedding size).
    pub dim: usize,
    /// Width of the MLP's inner layer.
    pub hidden_dim: usize,
    pub n_layers: usize,
    /// Query heads.
    pub n_heads: usize,
    /// Key/value heads. Fewer than `n_heads` means grouped-query attention.
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    /// Longest sequence the KV cache is sized for.
    pub seq_len: usize,
    pub rope_theta: f32,
    pub rope_layout: RopeLayout,
    pub norm_eps: f32,
}

impl Config {
    /// Width of one attention head.
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Width of the K (and V) vector of one token: all K/V heads together.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_size()
    }

    /// How many query heads share each K/V head.
    pub fn kv_group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
}
