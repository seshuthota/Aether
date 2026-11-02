//! Model hyperparameters: the numbers that fix every tensor's shape.

/// How RoPE pairs up the elements of a head (see `kernels::scalar`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// `(x[2i], x[2i+1])`: llama2.c, and Llama-architecture GGUF files.
    AdjacentPairs,
    /// `(x[i], x[i + d/2])` ("rotate_half"): HF transformers, Qwen2 GGUF.
    Halves,
}

/// Rotary embedding of one attention layer: the angle of pair `i` is
/// `pos * theta^(-2i / head_dim)`, for `i < pairs`; the remaining pairs are
/// left unrotated (LFM3's zero-tail partial RoPE).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rope {
    pub theta: f32,
    pub pairs: usize,
    pub layout: RopeLayout,
}

/// What a layer does between the residual stream and the MLP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerKind {
    /// Attention over the KV cache. `window = Some(w)` masks positions more
    /// than `w` back; `None` sees the whole prefix.
    Attention { window: Option<usize>, rope: Rope },
    /// A depthwise causal short convolution with `width` taps.
    Conv { width: usize },
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
    /// Key/value heads of every attention layer. Fewer than `n_heads` means
    /// grouped-query attention.
    pub n_kv_heads: usize,
    /// Width of one attention head (stored: some models are not
    /// `dim / n_heads`, and LFM3's conv layers have none).
    pub head_dim: usize,
    pub vocab_size: usize,
    /// Longest sequence the KV cache is sized for.
    pub seq_len: usize,
    /// RoPE constants of the all-attention models (kept for the GPU/DSP
    /// paths, which do not handle mixed stacks yet).
    pub rope_theta: f32,
    pub rope_layout: RopeLayout,
    pub norm_eps: f32,
    /// One entry per layer; `len() == n_layers`.
    pub layers: Vec<LayerKind>,
    /// RMSNorm on q/k per head before RoPE.
    pub qk_norm: bool,
    /// A sigmoid gate per attention head on the attention output.
    pub output_gate: bool,
    /// RMSNorm on the mixer and FFN outputs before they enter the residual.
    pub sandwich_norm: bool,
}

impl Config {
    /// Width of one attention head.
    pub fn head_size(&self) -> usize {
        self.head_dim
    }

    /// Width of the K (and V) vector of one token: all K/V heads together.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_size()
    }

    /// How many query heads share each K/V head.
    pub fn kv_group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    pub fn kind(&self, layer: usize) -> LayerKind {
        self.layers[layer]
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        matches!(self.layers[layer], LayerKind::Attention { .. })
    }

    /// The positions a token at `i` can see in this layer, or `None` for all
    /// of `0..=i`.
    pub fn window(&self, layer: usize) -> Option<usize> {
        match self.layers[layer] {
            LayerKind::Attention { window, .. } => window,
            LayerKind::Conv { .. } => None,
        }
    }

    pub fn rope(&self, layer: usize) -> Rope {
        match self.layers[layer] {
            LayerKind::Attention { rope, .. } => rope,
            LayerKind::Conv { .. } => Rope { theta: self.rope_theta, pairs: self.head_dim / 2, layout: self.rope_layout },
        }
    }

    /// All-attention config for the existing model families.
    pub fn all_attention(&mut self, theta: f32, layout: RopeLayout) {
        self.rope_theta = theta;
        self.rope_layout = layout;
        self.layers = (0..self.n_layers)
            .map(|_| LayerKind::Attention { window: None, rope: Rope { theta, pairs: self.head_dim / 2, layout } })
            .collect();
    }
}
