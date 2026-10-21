//! The decoder-only transformer forward pass, one token at a time.
//!
//! For each layer the residual stream `x` (one vector of `dim` floats) goes
//! through:
//!
//! ```text
//! xb = rmsnorm(x)                          attention block
//! q, k, v = Wq·xb, Wk·xb, Wv·xb
//! rotate q, k by position (RoPE); store k, v in the KV cache
//! each head: softmax(q·K / sqrt(hs)) · V   over positions 0..=pos
//! x += Wo · (heads concatenated)
//! xb = rmsnorm(x)                          MLP block
//! x += Wdown · (silu(Wgate·xb) * (Wup·xb))
//! ```
//!
//! then `logits = Wcls · rmsnorm(x)`. The KV cache keeps every earlier
//! token's k and v, so a new token costs one pass over the weights plus a
//! read of the cache, instead of recomputing the whole prefix.

use super::config::{Config, RopeLayout};
use crate::aligned::AlignedBuf;
use crate::kernels::scalar as k;

/// One layer's weights. Matrices are row-major `[out, in]`.
pub struct LayerWeights {
    pub attn_norm: AlignedBuf,
    pub wq: AlignedBuf,
    pub wk: AlignedBuf,
    pub wv: AlignedBuf,
    pub wo: AlignedBuf,
    pub ffn_norm: AlignedBuf,
    /// W1 in llama2.c: the branch that goes through SiLU.
    pub w_gate: AlignedBuf,
    /// W2: projects the MLP back to `dim`.
    pub w_down: AlignedBuf,
    /// W3: the branch multiplied by the gate.
    pub w_up: AlignedBuf,
}

pub struct Weights {
    /// `[vocab, dim]`: row `t` is token `t`'s embedding.
    pub token_embedding: AlignedBuf,
    pub layers: Vec<LayerWeights>,
    pub final_norm: AlignedBuf,
    /// `[vocab, dim]`, or `None` when tied to `token_embedding`.
    pub classifier: Option<AlignedBuf>,
}

/// Receives intermediate activations as the forward pass produces them.
/// Tests use it to compare every layer boundary against a reference dump;
/// [`NoTrace`] compiles to nothing.
pub trait Observer {
    /// `layer` is `None` outside the layer stack (embedding, final norm,
    /// logits).
    fn tap(&mut self, name: &str, layer: Option<usize>, pos: usize, data: &[f32]);
}

/// The observer that ignores everything.
pub struct NoTrace;

impl Observer for NoTrace {
    #[inline(always)]
    fn tap(&mut self, _: &str, _: Option<usize>, _: usize, _: &[f32]) {}
}

/// Buffers reused by every forward call, allocated once.
struct State {
    x: AlignedBuf,      // residual stream [dim]
    xb: AlignedBuf,     // normed input / attention output [dim]
    xb2: AlignedBuf,    // projection output [dim]
    hb: AlignedBuf,     // MLP gate [hidden]
    hb2: AlignedBuf,    // MLP up [hidden]
    q: AlignedBuf,      // [dim]
    att: AlignedBuf,    // attention scores, one row per head [n_heads, seq_len]
    logits: AlignedBuf, // [vocab]
    /// Keys of every position so far: `[layer, pos, kv_dim]`.
    key_cache: AlignedBuf,
    value_cache: AlignedBuf,
}

pub struct Transformer {
    pub config: Config,
    pub weights: Weights,
    state: State,
}

impl Transformer {
    pub fn new(config: Config, weights: Weights) -> Transformer {
        let c = &config;
        let cache = c.n_layers * c.seq_len * c.kv_dim();
        let state = State {
            x: AlignedBuf::zeros(c.dim),
            xb: AlignedBuf::zeros(c.dim),
            xb2: AlignedBuf::zeros(c.dim),
            hb: AlignedBuf::zeros(c.hidden_dim),
            hb2: AlignedBuf::zeros(c.hidden_dim),
            q: AlignedBuf::zeros(c.dim),
            att: AlignedBuf::zeros(c.n_heads * c.seq_len),
            logits: AlignedBuf::zeros(c.vocab_size),
            key_cache: AlignedBuf::zeros(cache),
            value_cache: AlignedBuf::zeros(cache),
        };
        Transformer { config, weights, state }
    }

    /// Runs `token` at position `pos` and returns the logits for the next
    /// token. Positions must be fed in order 0, 1, 2, … (each call reads the
    /// cache entries of all earlier positions).
    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        self.forward_traced(token, pos, &mut NoTrace)
    }

    /// [`Transformer::forward`] that reports every intermediate to `obs`.
    pub fn forward_traced<O: Observer>(&mut self, token: usize, pos: usize, obs: &mut O) -> &[f32] {
        let c = &self.config;
        let w = &self.weights;
        let s = &mut self.state;
        let (dim, hs, kv_dim, group) = (c.dim, c.head_size(), c.kv_dim(), c.kv_group());
        assert!(pos < c.seq_len, "position {pos} beyond seq_len {}", c.seq_len);
        assert!(token < c.vocab_size, "token {token} beyond vocab {}", c.vocab_size);

        s.x.copy_from_slice(&w.token_embedding[token * dim..(token + 1) * dim]);
        obs.tap("embed", None, pos, &s.x);

        for (l, lw) in w.layers.iter().enumerate() {
            // --- attention ---
            k::rmsnorm(&mut s.xb, &s.x, &lw.attn_norm, c.norm_eps);
            obs.tap("attn_norm", Some(l), pos, &s.xb);

            // This position's k and v go straight into the cache.
            let layer_off = l * c.seq_len * kv_dim;
            let at = layer_off + pos * kv_dim;
            k::matmul(&mut s.q, &s.xb, &lw.wq);
            k::matmul(&mut s.key_cache[at..at + kv_dim], &s.xb, &lw.wk);
            k::matmul(&mut s.value_cache[at..at + kv_dim], &s.xb, &lw.wv);

            match c.rope_layout {
                RopeLayout::AdjacentPairs => {
                    k::rope_adjacent(&mut s.q, pos, hs, c.rope_theta);
                    k::rope_adjacent(&mut s.key_cache[at..at + kv_dim], pos, hs, c.rope_theta);
                }
                RopeLayout::Halves => unimplemented!("rotate_half RoPE arrives with SmolLM2 (Phase 5)"),
            }
            obs.tap("q", Some(l), pos, &s.q);
            obs.tap("k", Some(l), pos, &s.key_cache[at..at + kv_dim]);
            obs.tap("v", Some(l), pos, &s.value_cache[at..at + kv_dim]);

            let (kc, vc) = (&s.key_cache[layer_off..], &s.value_cache[layer_off..]);
            for (h, (out, scores)) in s.xb.chunks_exact_mut(hs).zip(s.att.chunks_exact_mut(c.seq_len)).enumerate() {
                let q = &s.q[h * hs..(h + 1) * hs];
                let kv_off = (h / group) * hs; // which K/V head this query head reads
                let keys = |t: usize| &kc[t * kv_dim + kv_off..t * kv_dim + kv_off + hs];
                let values = |t: usize| &vc[t * kv_dim + kv_off..t * kv_dim + kv_off + hs];
                k::attention_head(out, q, keys, values, pos, scores);
            }
            obs.tap("attn", Some(l), pos, &s.xb);

            k::matmul(&mut s.xb2, &s.xb, &lw.wo);
            obs.tap("attn_out", Some(l), pos, &s.xb2);
            k::add_inplace(&mut s.x, &s.xb2);
            obs.tap("resid_attn", Some(l), pos, &s.x);

            // --- MLP ---
            k::rmsnorm(&mut s.xb, &s.x, &lw.ffn_norm, c.norm_eps);
            obs.tap("ffn_norm", Some(l), pos, &s.xb);
            k::matmul(&mut s.hb, &s.xb, &lw.w_gate);
            k::matmul(&mut s.hb2, &s.xb, &lw.w_up);
            k::swiglu(&mut s.hb, &s.hb2);
            obs.tap("swiglu", Some(l), pos, &s.hb);
            k::matmul(&mut s.xb, &s.hb, &lw.w_down);
            obs.tap("ffn_out", Some(l), pos, &s.xb);
            k::add_inplace(&mut s.x, &s.xb);
            obs.tap("resid_ffn", Some(l), pos, &s.x);
        }

        k::rmsnorm_inplace(&mut s.x, &w.final_norm, c.norm_eps);
        obs.tap("final_norm", None, pos, &s.x);
        let cls = w.classifier.as_ref().unwrap_or(&w.token_embedding);
        k::matmul(&mut s.logits, &s.x, cls);
        obs.tap("logits", None, pos, &s.logits);
        &s.logits
    }
}
