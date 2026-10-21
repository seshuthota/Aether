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
use crate::kernels as k;
use crate::pool::ThreadPool;
use crate::weight::{Format, Weight};

/// One layer's weights. Matrices are row-major `[out, in]`, in any storage
/// format; norm gains are always f32.
pub struct LayerWeights {
    pub attn_norm: AlignedBuf,
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub wo: Weight,
    pub ffn_norm: AlignedBuf,
    /// W1 in llama2.c: the branch that goes through SiLU.
    pub w_gate: Weight,
    /// W2: projects the MLP back to `dim`.
    pub w_down: Weight,
    /// W3: the branch multiplied by the gate.
    pub w_up: Weight,
}

pub struct Weights {
    /// `[vocab, dim]`: row `t` is token `t`'s embedding.
    pub token_embedding: Weight,
    pub layers: Vec<LayerWeights>,
    pub final_norm: AlignedBuf,
    /// `[vocab, dim]`, or `None` when tied to `token_embedding`.
    pub classifier: Option<Weight>,
}

impl Weights {
    /// Converts every matrix (not the norms) to `format`, as llama.cpp's
    /// `llama-quantize --pure` does: the embedding/classifier included.
    pub fn convert(self, c: &Config, format: Format) -> Weights {
        let (dim, hidden) = (c.dim, c.hidden_dim);
        Weights {
            token_embedding: self.token_embedding.convert(dim, format),
            layers: self
                .layers
                .into_iter()
                .map(|l| LayerWeights {
                    attn_norm: l.attn_norm,
                    wq: l.wq.convert(dim, format),
                    wk: l.wk.convert(dim, format),
                    wv: l.wv.convert(dim, format),
                    wo: l.wo.convert(dim, format),
                    ffn_norm: l.ffn_norm,
                    w_gate: l.w_gate.convert(dim, format),
                    w_down: l.w_down.convert(hidden, format),
                    w_up: l.w_up.convert(dim, format),
                })
                .collect(),
            final_norm: self.final_norm,
            classifier: self.classifier.map(|w| w.convert(dim, format)),
        }
    }

    /// Bytes of all matrices and norms: what one decode step streams.
    pub fn bytes(&self) -> usize {
        let norms = |l: &LayerWeights| (l.attn_norm.len() + l.ffn_norm.len()) * 4;
        let mats = |l: &LayerWeights| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_down, &l.w_up].iter().map(|w| w.bytes()).sum::<usize>();
        self.token_embedding.bytes()
            + self.layers.iter().map(|l| norms(l) + mats(l)).sum::<usize>()
            + self.final_norm.len() * 4
            + self.classifier.as_ref().map_or(0, Weight::bytes)
    }
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

/// Buffers for [`Transformer::forward_batch`]: the single-token buffers,
/// one row per token. Grown on demand to the largest batch seen.
struct Batch {
    cap: usize,
    x: AlignedBuf,   // [cap, dim]
    xb: AlignedBuf,  // [cap, dim]
    xb2: AlignedBuf, // [cap, dim]
    q: AlignedBuf,   // [cap, dim]
    hb: AlignedBuf,  // [cap, hidden]
    hb2: AlignedBuf, // [cap, hidden]
}

impl Batch {
    fn new(c: &Config, cap: usize) -> Batch {
        let z = |n: usize| AlignedBuf::zeros(cap * n);
        Batch { cap, x: z(c.dim), xb: z(c.dim), xb2: z(c.dim), q: z(c.dim), hb: z(c.hidden_dim), hb2: z(c.hidden_dim) }
    }
}

pub struct Transformer {
    pub config: Config,
    pub weights: Weights,
    state: State,
    batch: Batch,
    /// Threads for the matrix products; `None` runs them on the caller.
    pool: Option<ThreadPool>,
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
        let batch = Batch::new(&config, 0);
        Transformer { config, weights, state, batch, pool: None }
    }

    /// Splits every matrix product across `pool`'s threads from now on.
    /// Results are bit-identical to single-threaded runs (each output row is
    /// still one thread's sum in the same order).
    pub fn set_pool(&mut self, pool: ThreadPool) {
        self.pool = Some(pool);
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
        let pool = self.pool.as_ref();
        let (hs, kv_dim, group) = (c.head_size(), c.kv_dim(), c.kv_group());
        assert!(pos < c.seq_len, "position {pos} beyond seq_len {}", c.seq_len);
        assert!(token < c.vocab_size, "token {token} beyond vocab {}", c.vocab_size);

        w.token_embedding.row_f32(token, &mut s.x);
        obs.tap("embed", None, pos, &s.x);

        for (l, lw) in w.layers.iter().enumerate() {
            // --- attention ---
            k::rmsnorm(&mut s.xb, &s.x, &lw.attn_norm, c.norm_eps);
            obs.tap("attn_norm", Some(l), pos, &s.xb);

            // This position's k and v go straight into the cache.
            let layer_off = l * c.seq_len * kv_dim;
            let at = layer_off + pos * kv_dim;
            k::linear_on(pool, &mut s.q, &s.xb, &lw.wq);
            k::linear_on(pool, &mut s.key_cache[at..at + kv_dim], &s.xb, &lw.wk);
            k::linear_on(pool, &mut s.value_cache[at..at + kv_dim], &s.xb, &lw.wv);

            match c.rope_layout {
                RopeLayout::AdjacentPairs => {
                    k::rope_adjacent(&mut s.q, pos, hs, c.rope_theta);
                    k::rope_adjacent(&mut s.key_cache[at..at + kv_dim], pos, hs, c.rope_theta);
                }
                RopeLayout::Halves => {
                    k::rope_halves(&mut s.q, pos, hs, c.rope_theta);
                    k::rope_halves(&mut s.key_cache[at..at + kv_dim], pos, hs, c.rope_theta);
                }
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

            k::linear_on(pool, &mut s.xb2, &s.xb, &lw.wo);
            obs.tap("attn_out", Some(l), pos, &s.xb2);
            k::add_inplace(&mut s.x, &s.xb2);
            obs.tap("resid_attn", Some(l), pos, &s.x);

            // --- MLP ---
            k::rmsnorm(&mut s.xb, &s.x, &lw.ffn_norm, c.norm_eps);
            obs.tap("ffn_norm", Some(l), pos, &s.xb);
            k::linear_on(pool, &mut s.hb, &s.xb, &lw.w_gate);
            k::linear_on(pool, &mut s.hb2, &s.xb, &lw.w_up);
            k::swiglu(&mut s.hb, &s.hb2);
            obs.tap("swiglu", Some(l), pos, &s.hb);
            k::linear_on(pool, &mut s.xb, &s.hb, &lw.w_down);
            obs.tap("ffn_out", Some(l), pos, &s.xb);
            k::add_inplace(&mut s.x, &s.xb);
            obs.tap("resid_ffn", Some(l), pos, &s.x);
        }

        k::rmsnorm_inplace(&mut s.x, &w.final_norm, c.norm_eps);
        obs.tap("final_norm", None, pos, &s.x);
        let cls = w.classifier.as_ref().unwrap_or(&w.token_embedding);
        k::linear_on(pool, &mut s.logits, &s.x, cls);
        obs.tap("logits", None, pos, &s.logits);
        &s.logits
    }

    /// Runs `tokens` at positions `start .. start + tokens.len()` in one pass
    /// (prefill) and returns the logits after the last one. Equivalent to
    /// calling [`Transformer::forward`] once per token, but every weight
    /// matrix is multiplied by all the tokens at once (GEMM), so each weight
    /// is fetched once per batch instead of once per token.
    pub fn forward_batch(&mut self, tokens: &[usize], start: usize) -> &[f32] {
        self.forward_batch_traced(tokens, start, &mut NoTrace)
    }

    /// [`Transformer::forward_batch`] reporting every intermediate to `obs`,
    /// one call per token, with the same names and positions as
    /// [`Transformer::forward_traced`].
    pub fn forward_batch_traced<O: Observer>(&mut self, tokens: &[usize], start: usize, obs: &mut O) -> &[f32] {
        let t = tokens.len();
        assert!(t > 0, "empty batch");
        assert!(start + t <= self.config.seq_len, "positions {start}..{} beyond seq_len {}", start + t, self.config.seq_len);
        if self.batch.cap < t {
            self.batch = Batch::new(&self.config, t);
        }
        let c = &self.config;
        let w = &self.weights;
        let (s, b, pool) = (&mut self.state, &mut self.batch, self.pool.as_ref());
        let (dim, hidden, hs, kv_dim, group) = (c.dim, c.hidden_dim, c.head_size(), c.kv_dim(), c.kv_group());

        for (i, &tok) in tokens.iter().enumerate() {
            assert!(tok < c.vocab_size, "token {tok} beyond vocab {}", c.vocab_size);
            w.token_embedding.row_f32(tok, &mut b.x[i * dim..(i + 1) * dim]);
            obs.tap("embed", None, start + i, &b.x[i * dim..(i + 1) * dim]);
        }

        for (l, lw) in w.layers.iter().enumerate() {
            // --- attention ---
            for i in 0..t {
                k::rmsnorm(&mut b.xb[i * dim..(i + 1) * dim], &b.x[i * dim..(i + 1) * dim], &lw.attn_norm, c.norm_eps);
                obs.tap("attn_norm", Some(l), start + i, &b.xb[i * dim..(i + 1) * dim]);
            }
            // K and V for positions start..start+t are consecutive rows of
            // this layer's cache, so the GEMM writes them in place.
            let layer_off = l * c.seq_len * kv_dim;
            let (c0, c1) = (layer_off + start * kv_dim, layer_off + (start + t) * kv_dim);
            k::linear_batch_on(pool, &mut b.q[..t * dim], &b.xb[..t * dim], &lw.wq, t);
            k::linear_batch_on(pool, &mut s.key_cache[c0..c1], &b.xb[..t * dim], &lw.wk, t);
            k::linear_batch_on(pool, &mut s.value_cache[c0..c1], &b.xb[..t * dim], &lw.wv, t);
            for i in 0..t {
                let pos = start + i;
                let (q, kr) = (&mut b.q[i * dim..(i + 1) * dim], &mut s.key_cache[c0 + i * kv_dim..c0 + (i + 1) * kv_dim]);
                match c.rope_layout {
                    RopeLayout::AdjacentPairs => {
                        k::rope_adjacent(q, pos, hs, c.rope_theta);
                        k::rope_adjacent(kr, pos, hs, c.rope_theta);
                    }
                    RopeLayout::Halves => {
                        k::rope_halves(q, pos, hs, c.rope_theta);
                        k::rope_halves(kr, pos, hs, c.rope_theta);
                    }
                }
                obs.tap("q", Some(l), pos, &b.q[i * dim..(i + 1) * dim]);
                obs.tap("k", Some(l), pos, &s.key_cache[c0 + i * kv_dim..c0 + (i + 1) * kv_dim]);
                obs.tap("v", Some(l), pos, &s.value_cache[c0 + i * kv_dim..c0 + (i + 1) * kv_dim]);
            }

            // Causal attention: token i (position start+i) sees 0..=start+i.
            let (kc, vc) = (&s.key_cache[layer_off..], &s.value_cache[layer_off..]);
            for i in 0..t {
                let pos = start + i;
                let xb = &mut b.xb[i * dim..(i + 1) * dim];
                for (h, (out, scores)) in xb.chunks_exact_mut(hs).zip(s.att.chunks_exact_mut(c.seq_len)).enumerate() {
                    let q = &b.q[i * dim + h * hs..i * dim + (h + 1) * hs];
                    let kv_off = (h / group) * hs;
                    let keys = |p: usize| &kc[p * kv_dim + kv_off..p * kv_dim + kv_off + hs];
                    let values = |p: usize| &vc[p * kv_dim + kv_off..p * kv_dim + kv_off + hs];
                    k::attention_head(out, q, keys, values, pos, scores);
                }
                obs.tap("attn", Some(l), pos, &b.xb[i * dim..(i + 1) * dim]);
            }

            k::linear_batch_on(pool, &mut b.xb2[..t * dim], &b.xb[..t * dim], &lw.wo, t);
            for i in 0..t {
                obs.tap("attn_out", Some(l), start + i, &b.xb2[i * dim..(i + 1) * dim]);
            }
            k::add_inplace(&mut b.x[..t * dim], &b.xb2[..t * dim]);
            for i in 0..t {
                obs.tap("resid_attn", Some(l), start + i, &b.x[i * dim..(i + 1) * dim]);
            }

            // --- MLP ---
            for i in 0..t {
                k::rmsnorm(&mut b.xb[i * dim..(i + 1) * dim], &b.x[i * dim..(i + 1) * dim], &lw.ffn_norm, c.norm_eps);
                obs.tap("ffn_norm", Some(l), start + i, &b.xb[i * dim..(i + 1) * dim]);
            }
            k::linear_batch_on(pool, &mut b.hb[..t * hidden], &b.xb[..t * dim], &lw.w_gate, t);
            k::linear_batch_on(pool, &mut b.hb2[..t * hidden], &b.xb[..t * dim], &lw.w_up, t);
            k::swiglu(&mut b.hb[..t * hidden], &b.hb2[..t * hidden]);
            for i in 0..t {
                obs.tap("swiglu", Some(l), start + i, &b.hb[i * hidden..(i + 1) * hidden]);
            }
            k::linear_batch_on(pool, &mut b.xb[..t * dim], &b.hb[..t * hidden], &lw.w_down, t);
            for i in 0..t {
                obs.tap("ffn_out", Some(l), start + i, &b.xb[i * dim..(i + 1) * dim]);
            }
            k::add_inplace(&mut b.x[..t * dim], &b.xb[..t * dim]);
            for i in 0..t {
                obs.tap("resid_ffn", Some(l), start + i, &b.x[i * dim..(i + 1) * dim]);
            }
        }

        // Only the last token's logits are needed to continue generating.
        let last = &b.x[(t - 1) * dim..t * dim];
        s.x.copy_from_slice(last);
        k::rmsnorm_inplace(&mut s.x, &w.final_norm, c.norm_eps);
        obs.tap("final_norm", None, start + t - 1, &s.x);
        let cls = w.classifier.as_ref().unwrap_or(&w.token_embedding);
        k::linear_on(pool, &mut s.logits, &s.x, cls);
        obs.tap("logits", None, start + t - 1, &s.logits);
        &s.logits
    }
}
