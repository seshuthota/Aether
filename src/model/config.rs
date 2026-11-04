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

/// The MLP's nonlinearity on the gate branch, multiplied by the up branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `silu(gate) · up` (Llama, Qwen, LFM3).
    SwiGlu,
    /// `gelu(gate) · up`, GELU's tanh approximation (Gemma).
    GeluTanh,
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
    /// Per-layer head size when layers differ (Gemma 4: 256 on sliding
    /// layers, 512 on full ones); empty means `head_dim` everywhere.
    pub head_dims: Vec<usize>,
    /// Per-layer MLP width when layers differ; empty means `hidden_dim`.
    pub hidden_dims: Vec<usize>,
    /// The layer whose K/V cache each layer reads (Gemma 4's shared KV:
    /// layers 15–34 read 13 or 14 and compute no K/V); empty means its own.
    pub kv_source: Vec<usize>,
    /// Scale of the attention scores; `None` is 1/√(head size).
    pub attn_scale: Option<f32>,
    /// The embedding row is multiplied by this (Gemma: √dim).
    pub embed_scale: f32,
    /// Logits become `cap · tanh(logits / cap)` (Gemma).
    pub logit_softcap: Option<f32>,
    /// RMSNorm without a gain on v per head (Gemma 4).
    pub v_norm: bool,
    pub activation: Activation,
    /// Width of the per-layer embedding of each layer (Gemma 4: 256); 0
    /// for none.
    pub ple_dim: usize,
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

    /// Head size of `layer`'s attention.
    pub fn head_dim_of(&self, layer: usize) -> usize {
        self.head_dims.get(layer).copied().unwrap_or(self.head_dim)
    }

    /// Width of `layer`'s queries (all query heads), and of its attention
    /// output before `wo`.
    pub fn q_dim(&self, layer: usize) -> usize {
        self.n_heads * self.head_dim_of(layer)
    }

    /// Width of `layer`'s K (and V) row.
    pub fn kv_dim_of(&self, layer: usize) -> usize {
        self.n_kv_heads * self.head_dim_of(layer)
    }

    /// `layer`'s MLP width.
    pub fn hidden_of(&self, layer: usize) -> usize {
        self.hidden_dims.get(layer).copied().unwrap_or(self.hidden_dim)
    }

    /// The layer whose cache `layer` reads (itself unless KV is shared).
    pub fn kv_source_of(&self, layer: usize) -> usize {
        self.kv_source.get(layer).copied().unwrap_or(layer)
    }

    /// Whether `layer` computes and stores its own K/V.
    pub fn owns_kv(&self, layer: usize) -> bool {
        self.kv_source_of(layer) == layer
    }

    /// The widest q / attention output, K/V row and MLP over all layers
    /// (buffer sizes).
    pub fn max_q_dim(&self) -> usize {
        (0..self.n_layers).map(|l| self.q_dim(l)).max().unwrap_or(0).max(self.n_heads * self.head_dim)
    }

    pub fn max_kv_dim(&self) -> usize {
        (0..self.n_layers).map(|l| self.kv_dim_of(l)).max().unwrap_or(0).max(self.kv_dim())
    }

    pub fn max_hidden(&self) -> usize {
        (0..self.n_layers).map(|l| self.hidden_of(l)).max().unwrap_or(0).max(self.hidden_dim)
    }

    /// Whether every layer has the same shape and none of Gemma's extras,
    /// as the GPU and DSP paths assume.
    pub fn uniform(&self) -> bool {
        self.head_dims.is_empty() && self.hidden_dims.is_empty() && self.kv_source.is_empty() && self.ple_dim == 0
    }

    pub fn has_window(&self) -> bool {
        (0..self.n_layers).any(|l| self.window(l).is_some())
    }

    pub fn kind(&self, layer: usize) -> LayerKind {
        self.layers[layer]
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        matches!(self.layers[layer], LayerKind::Attention { .. })
    }

    pub fn has_conv(&self) -> bool {
        self.layers.iter().any(|l| matches!(l, LayerKind::Conv { .. }))
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
