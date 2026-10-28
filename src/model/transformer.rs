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
use crate::pool::{SharedOut, ThreadPool};
use crate::weight::{Format, Weight};

/// One layer's weights. Matrices are row-major `[out, in]`, in any storage
/// format; norm gains are always f32.
pub struct LayerWeights {
    pub attn_norm: AlignedBuf,
    pub wq: Weight,
    pub wk: Weight,
    pub wv: Weight,
    pub wo: Weight,
    /// Biases added after the q / k / v projections (Qwen2 has them;
    /// Llama-style models do not).
    pub bq: Option<AlignedBuf>,
    pub bk: Option<AlignedBuf>,
    pub bv: Option<AlignedBuf>,
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
        self.map_matrices(c, |w, cols| w.convert(cols, format))
    }

    /// Repacks every quantized matrix four rows at a time for the `smmla`
    /// kernels (see [`Weight::repack`]).
    pub fn repack(self, c: &Config) -> Weights {
        self.map_matrices(c, |w, cols| w.repack(cols))
    }

    /// Applies `f(matrix, columns)` to every matrix, not the norms.
    fn map_matrices(self, c: &Config, f: impl Fn(Weight, usize) -> Weight) -> Weights {
        let (dim, hidden) = (c.dim, c.hidden_dim);
        Weights {
            token_embedding: f(self.token_embedding, dim),
            layers: self
                .layers
                .into_iter()
                .map(|l| LayerWeights {
                    attn_norm: l.attn_norm,
                    wq: f(l.wq, dim),
                    wk: f(l.wk, dim),
                    wv: f(l.wv, dim),
                    wo: f(l.wo, dim),
                    bq: l.bq,
                    bk: l.bk,
                    bv: l.bv,
                    ffn_norm: l.ffn_norm,
                    w_gate: f(l.w_gate, dim),
                    w_down: f(l.w_down, hidden),
                    w_up: f(l.w_up, dim),
                })
                .collect(),
            final_norm: self.final_norm,
            classifier: self.classifier.map(|w| f(w, dim)),
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

/// Positions per chunk when one head's attention is split across positions
/// (flash-decoding): at depth 512 a decode step has 14 heads × 4 chunks to
/// share among the threads instead of 14 long heads. Heads that see at most
/// this many positions are computed in one piece, as before.
const ATTN_CHUNK: usize = 128;

/// Scratch space for attention, one region per query head: its scores
/// (`seq_len` floats), and for a head split into chunks, each chunk's
/// unnormalised output (`head_size` floats) and `[max, sum]`.
struct AttScratch {
    scores: AlignedBuf,
    parts: AlignedBuf,
    stats: AlignedBuf,
}

impl AttScratch {
    fn new(c: &Config) -> AttScratch {
        let chunks = c.seq_len.div_ceil(ATTN_CHUNK);
        AttScratch {
            scores: AlignedBuf::zeros(c.n_heads * c.seq_len),
            parts: AlignedBuf::zeros(c.n_heads * chunks * c.head_size()),
            stats: AlignedBuf::zeros(c.n_heads * chunks * 2),
        }
    }
}

/// The K/V slices of query head `h` (GQA: heads share K/V heads in groups).
fn kv_head<'a>(c: &Config, kc: &'a [f32], vc: &'a [f32], h: usize) -> (impl Fn(usize) -> &'a [f32] + Copy, impl Fn(usize) -> &'a [f32] + Copy) {
    let (hs, kv_dim) = (c.head_size(), c.kv_dim());
    let kv_off = (h / c.kv_group()) * hs; // which K/V head this query head reads
    (move |t: usize| &kc[t * kv_dim + kv_off..t * kv_dim + kv_off + hs], move |t: usize| &vc[t * kv_dim + kv_off..t * kv_dim + kv_off + hs])
}

/// Chunk `ch` of a head at position `pos`.
fn chunk_range(ch: usize, pos: usize) -> std::ops::Range<usize> {
    ch * ATTN_CHUNK..((ch + 1) * ATTN_CHUNK).min(pos + 1)
}

/// One query head at position `pos`, on this thread: in one piece up to
/// [`ATTN_CHUNK`] positions, otherwise chunk by chunk ([`k::attention_part`])
/// and merged ([`k::attention_merge`]). `scores` needs `pos + 1` floats;
/// `parts` and `stats` room for the chunks. Every path that splits heads
/// across threads computes the same chunks and the same merge, so the
/// result does not depend on the threads.
#[allow(clippy::too_many_arguments)]
fn one_head<'a>(out: &mut [f32], q: &[f32], keys: impl Fn(usize) -> &'a [f32] + Copy, values: impl Fn(usize) -> &'a [f32] + Copy, pos: usize, scores: &mut [f32], parts: &mut [f32], stats: &mut [f32]) {
    let chunks = (pos + 1).div_ceil(ATTN_CHUNK);
    if chunks == 1 {
        return k::attention_head(out, q, keys, values, pos, scores);
    }
    let hs = q.len();
    for ch in 0..chunks {
        let (m, s) = k::attention_part(&mut parts[ch * hs..(ch + 1) * hs], q, keys, values, chunk_range(ch, pos), scores);
        stats[2 * ch..2 * ch + 2].copy_from_slice(&[m, s]);
    }
    k::attention_merge(out, &parts[..chunks * hs], stats[..2 * chunks].as_chunks::<2>().0);
}

/// All query heads of one token: `out[h] = attention(q[h], cache, pos)`.
/// Heads are independent (each writes its own slice of `out` and its own
/// scratch), so with a pool they run in parallel; past [`ATTN_CHUNK`]
/// positions every (head, chunk) pair is its own item, and the chunks of
/// each head are merged once they are all done.
#[allow(clippy::too_many_arguments)]
fn attention(pool: Option<&ThreadPool>, c: &Config, out: &mut [f32], q: &[f32], kc: &[f32], vc: &[f32], att: &mut AttScratch, pos: usize) {
    let (hs, seq, nh) = (c.head_size(), c.seq_len, c.n_heads);
    let max_chunks = seq.div_ceil(ATTN_CHUNK);
    let (ps, ss) = (max_chunks * hs, max_chunks * 2); // per-head parts / stats
    let Some(p) = pool.filter(|p| p.threads() > 1) else {
        for h in 0..nh {
            let (keys, values) = kv_head(c, kc, vc, h);
            let (sc, pa, st) = (&mut att.scores[h * seq..(h + 1) * seq], &mut att.parts[h * ps..(h + 1) * ps], &mut att.stats[h * ss..(h + 1) * ss]);
            one_head(&mut out[h * hs..(h + 1) * hs], &q[h * hs..(h + 1) * hs], keys, values, pos, sc, pa, st);
        }
        return;
    };
    assert!(out.len() >= nh * hs && att.scores.len() >= nh * seq && att.parts.len() >= nh * ps && att.stats.len() >= nh * ss);
    let chunks = (pos + 1).div_ceil(ATTN_CHUNK);
    let (o, sc, pa, st) = (SharedOut(out.as_mut_ptr()), SharedOut(att.scores.as_mut_ptr()), SharedOut(att.parts.as_mut_ptr()), SharedOut(att.stats.as_mut_ptr()));
    p.for_blocks(nh * chunks, 1, &|r| {
        for j in r {
            let (h, ch) = (j / chunks, j % chunks);
            let (keys, values) = kv_head(c, kc, vc, h);
            let qh = &q[h * hs..(h + 1) * hs];
            if chunks == 1 {
                // SAFETY: head h is handed to exactly one thread, and its
                // output slice and score row are disjoint from every other
                // head's (lengths checked above).
                let (oh, sh) = unsafe { (std::slice::from_raw_parts_mut(o.ptr().add(h * hs), hs), std::slice::from_raw_parts_mut(sc.ptr().add(h * seq), seq)) };
                k::attention_head(oh, qh, keys, values, pos, sh);
            } else {
                // SAFETY: (h, ch) is handed to exactly one thread; its scores
                // (chunk ch of head h's row), part and stats are disjoint
                // from every other item's (lengths checked above).
                let (sh, part, stat) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(sc.ptr().add(h * seq + ch * ATTN_CHUNK), ATTN_CHUNK.min(seq - ch * ATTN_CHUNK)),
                        std::slice::from_raw_parts_mut(pa.ptr().add(h * ps + ch * hs), hs),
                        std::slice::from_raw_parts_mut(st.ptr().add(h * ss + 2 * ch), 2),
                    )
                };
                let (m, s) = k::attention_part(part, qh, keys, values, chunk_range(ch, pos), sh);
                stat.copy_from_slice(&[m, s]);
            }
        }
    });
    if chunks > 1 {
        for h in 0..nh {
            let stats = &att.stats[h * ss..h * ss + 2 * chunks];
            k::attention_merge(&mut out[h * hs..(h + 1) * hs], &att.parts[h * ps..h * ps + chunks * hs], stats.as_chunks::<2>().0);
        }
    }
}

/// [`attention`] for `t` tokens at positions `start..start + t` (prefill):
/// every (token, head) pair is independent, so with a pool all `t ·
/// n_heads` of them are handed out in one call, rather than one call per
/// token with only `n_heads` items to share (which leaves threads waiting
/// at the end of each of those calls). Items go out longest first (the
/// last token sees the most positions), so the cheap ones fill in at the
/// end. Each block of items has its own scratch; every head goes through
/// [`one_head`], as in [`attention`].
#[allow(clippy::too_many_arguments)]
fn attention_batch(pool: Option<&ThreadPool>, c: &Config, out: &mut [f32], q: &[f32], kc: &[f32], vc: &[f32], att: &mut AttScratch, start: usize, t: usize) {
    let Some(p) = pool.filter(|p| p.threads() > 1 && t > 1) else {
        for i in 0..t {
            let dim = c.n_heads * c.head_size();
            attention(pool, c, &mut out[i * dim..(i + 1) * dim], &q[i * dim..(i + 1) * dim], kc, vc, att, start + i);
        }
        return;
    };
    let (hs, nh) = (c.head_size(), c.n_heads);
    assert!(out.len() >= t * nh * hs && q.len() >= t * nh * hs);
    let o = SharedOut(out.as_mut_ptr());
    let items = t * nh;
    let chunks = (start + t).div_ceil(ATTN_CHUNK);
    p.for_blocks(items, crate::pool::block_size(items, p.threads(), 1), &|r| {
        let (mut sc, mut parts, mut stats) = (vec![0.0; start + t], vec![0.0; chunks * hs], vec![0.0; chunks * 2]);
        for j in r {
            // Item j: token t-1-j/nh (longest first), head j%nh.
            let (i, h) = (t - 1 - j / nh, j % nh);
            let (keys, values) = kv_head(c, kc, vc, h);
            let qh = &q[(i * nh + h) * hs..(i * nh + h + 1) * hs];
            // SAFETY: item (i, h) is handed to exactly one thread, and its
            // slice of `out` is disjoint from every other item's (length
            // checked above).
            let oh = unsafe { std::slice::from_raw_parts_mut(o.ptr().add((i * nh + h) * hs), hs) };
            one_head(oh, qh, keys, values, start + i, &mut sc, &mut parts, &mut stats);
        }
    });
}

/// `x += bias` when the layer has one.
fn add_bias(x: &mut [f32], bias: Option<&AlignedBuf>) {
    if let Some(b) = bias {
        k::add_inplace(x, b);
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
    att: AttScratch,    // attention scores (one row per head) and chunk results
    logits: AlignedBuf, // [vocab]
    /// Keys of every position so far: `[layer, pos, kv_dim]`.
    key_cache: AlignedBuf,
    value_cache: AlignedBuf,
}

/// Buffers for [`Transformer::forward_batch`]: the single-token buffers,
/// one row per token. Grown on demand to the largest batch seen.
struct Batch {
    cap: usize,
    /// Logits of several positions, `[k, vocab]` (perplexity evaluation).
    logits: Vec<f32>,
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
        Batch { cap, logits: Vec::new(), x: z(c.dim), xb: z(c.dim), xb2: z(c.dim), q: z(c.dim), hb: z(c.hidden_dim), hb2: z(c.hidden_dim) }
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
    /// On CPUs where the repacked `smmla` kernels run (see
    /// [`k::repack_weights`]), quantized matrices are repacked here, once.
    pub fn new(config: Config, weights: Weights) -> Transformer {
        let weights = if k::repack_weights() { weights.repack(&config) } else { weights };
        let c = &config;
        let cache = c.n_layers * c.seq_len * c.kv_dim();
        let state = State {
            x: AlignedBuf::zeros(c.dim),
            xb: AlignedBuf::zeros(c.dim),
            xb2: AlignedBuf::zeros(c.dim),
            hb: AlignedBuf::zeros(c.hidden_dim),
            hb2: AlignedBuf::zeros(c.hidden_dim),
            q: AlignedBuf::zeros(c.dim),
            att: AttScratch::new(c),
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
        let (hs, kv_dim) = (c.head_size(), c.kv_dim());
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
            add_bias(&mut s.q, lw.bq.as_ref());
            add_bias(&mut s.key_cache[at..at + kv_dim], lw.bk.as_ref());
            add_bias(&mut s.value_cache[at..at + kv_dim], lw.bv.as_ref());

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
            attention(pool, c, &mut s.xb, &s.q, kc, vc, &mut s.att, pos);
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
        self.batch_impl(tokens, start, tokens.len() - 1, obs)
    }

    /// Runs `tokens` at positions `start ..` and returns the logits of every
    /// token from index `from` on, `[tokens.len() - from, vocab]` row-major
    /// (row `r` predicts the token after `tokens[from + r]`). Used to score
    /// text: perplexity needs a prediction at every position.
    pub fn forward_batch_logits(&mut self, tokens: &[usize], start: usize, from: usize) -> &[f32] {
        assert!(from < tokens.len());
        self.batch_impl(tokens, start, from, &mut NoTrace)
    }

    fn batch_impl<O: Observer>(&mut self, tokens: &[usize], start: usize, logits_from: usize, obs: &mut O) -> &[f32] {
        let t = tokens.len();
        assert!(t > 0, "empty batch");
        assert!(start + t <= self.config.seq_len, "positions {start}..{} beyond seq_len {}", start + t, self.config.seq_len);
        if self.batch.cap < t {
            self.batch = Batch::new(&self.config, t);
        }
        let c = &self.config;
        let w = &self.weights;
        let (s, b, pool) = (&mut self.state, &mut self.batch, self.pool.as_ref());
        let (dim, hidden, hs, kv_dim) = (c.dim, c.hidden_dim, c.head_size(), c.kv_dim());

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
                add_bias(&mut b.q[i * dim..(i + 1) * dim], lw.bq.as_ref());
                add_bias(&mut s.key_cache[c0 + i * kv_dim..c0 + (i + 1) * kv_dim], lw.bk.as_ref());
                add_bias(&mut s.value_cache[c0 + i * kv_dim..c0 + (i + 1) * kv_dim], lw.bv.as_ref());
            }
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
            attention_batch(pool, c, &mut b.xb[..t * dim], &b.q[..t * dim], kc, vc, &mut s.att, start, t);
            for i in 0..t {
                obs.tap("attn", Some(l), start + i, &b.xb[i * dim..(i + 1) * dim]);
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

        let cls = w.classifier.as_ref().unwrap_or(&w.token_embedding);
        if logits_from + 1 == t {
            // Only the last token's logits are needed to continue generating.
            let last = &b.x[(t - 1) * dim..t * dim];
            s.x.copy_from_slice(last);
            k::rmsnorm_inplace(&mut s.x, &w.final_norm, c.norm_eps);
            obs.tap("final_norm", None, start + t - 1, &s.x);
            k::linear_on(pool, &mut s.logits, &s.x, cls);
            obs.tap("logits", None, start + t - 1, &s.logits);
            return &s.logits;
        }
        // Several positions: normalise those rows in place and run the
        // classifier over all of them as one batched product.
        let rows = t - logits_from;
        for i in logits_from..t {
            k::rmsnorm_inplace(&mut b.x[i * dim..(i + 1) * dim], &w.final_norm, c.norm_eps);
        }
        b.logits.resize(rows * c.vocab_size, 0.0);
        k::linear_batch_on(pool, &mut b.logits, &b.x[logits_from * dim..t * dim], cls, rows);
        &b.logits
    }
}
