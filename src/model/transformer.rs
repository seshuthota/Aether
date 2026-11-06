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

use super::config::{Activation, Config, RopeLayout};
use crate::aligned::AlignedBuf;
use crate::kernels as k;
use crate::pool::{SharedOut, ThreadPool, par_chunks_mut};
use crate::format::imatrix::Imatrix;
use crate::weight::{Format, Weight};
use std::io;

/// Attention weights of one layer. Matrices are row-major `[out, in]`, in
/// any storage format; norm gains and biases are always f32.
pub struct AttentionWeights {
    pub wq: Weight,
    /// `None` in layers that read another layer's cache (Gemma 4's shared
    /// KV), which compute no K/V.
    pub wk: Option<Weight>,
    pub wv: Option<Weight>,
    pub wo: Weight,
    /// Biases added after the q / k / v projections (Qwen2 has them;
    /// Llama-style models do not).
    pub bq: Option<AlignedBuf>,
    pub bk: Option<AlignedBuf>,
    pub bv: Option<AlignedBuf>,
    /// Per-head RMSNorm gains before RoPE (LFM3).
    pub q_norm: Option<AlignedBuf>,
    pub k_norm: Option<AlignedBuf>,
    /// One sigmoid gate per query head, from the normed layer input.
    pub gate: Option<Weight>,
}

impl AttentionWeights {
    /// K and V, for layers that compute their own.
    pub fn kv(&self) -> Option<(&Weight, &Weight)> {
        self.wk.as_ref().zip(self.wv.as_ref())
    }
}

/// The gated short convolution of one layer.
pub struct ConvWeights {
    /// `[3·dim, dim]`: B, C and x, in that order.
    pub in_proj: Weight,
    /// `[dim, width]`: row `ch` holds the width taps of channel `ch`.
    pub conv: Weight,
    /// `[dim, dim]`.
    pub out_proj: Weight,
}

pub enum Mixer {
    Attention(AttentionWeights),
    Conv(ConvWeights),
}

/// One layer's weights.
pub struct LayerWeights {
    pub attn_norm: AlignedBuf,
    pub mixer: Mixer,
    /// RMSNorm on the mixer output before the residual add (LFM3).
    pub post_attn_norm: Option<AlignedBuf>,
    pub ffn_norm: AlignedBuf,
    /// RMSNorm on the FFN output before the residual add (LFM3).
    pub post_ffn_norm: Option<AlignedBuf>,
    /// W1 in llama2.c: the branch that goes through SiLU.
    pub w_gate: Weight,
    /// W2: projects the MLP back to `dim`.
    pub w_down: Weight,
    /// W3: the branch multiplied by the gate.
    pub w_up: Weight,
    /// Gemma 4's per-layer embedding block, after the FFN.
    pub ple: Option<PleWeights>,
    /// The layer's output is multiplied by this (Gemma 4's
    /// `layer_output_scale`).
    pub out_scale: Option<f32>,
}

/// Gemma 4's per-layer embedding block of one layer: `x += post_norm(proj ·
/// (gelu(inp_gate · x) ⊙ ple[layer]))`.
pub struct PleWeights {
    /// `[ple_dim, dim]`.
    pub inp_gate: Weight,
    /// `[dim, ple_dim]`.
    pub proj: Weight,
    pub post_norm: AlignedBuf,
}

/// Gemma 4's per-layer embedding inputs, computed once per token: the
/// token's row of `table` (`[vocab, n_layers · ple_dim]`) plus a projection
/// of the scaled embedding, normed per layer.
pub struct PleTable {
    pub table: crate::weight::Rows,
    /// `[n_layers · ple_dim, dim]`.
    pub proj: Weight,
    /// `[ple_dim]`, shared by every layer's slice.
    pub proj_norm: AlignedBuf,
}

impl LayerWeights {
    pub fn attention(&self) -> Option<&AttentionWeights> {
        match &self.mixer {
            Mixer::Attention(a) => Some(a),
            Mixer::Conv(_) => None,
        }
    }

    pub fn conv(&self) -> Option<&ConvWeights> {
        match &self.mixer {
            Mixer::Conv(c) => Some(c),
            Mixer::Attention(_) => None,
        }
    }

    /// Every matrix of this layer, with its llama.cpp tensor name (without
    /// the `blk.N.` prefix), in a stable order.
    pub fn matrices(&self) -> Vec<(&'static str, &Weight)> {
        let mut v = vec![("ffn_gate", &self.w_gate), ("ffn_down", &self.w_down), ("ffn_up", &self.w_up)];
        match &self.mixer {
            Mixer::Attention(a) => {
                v.push(("attn_q", &a.wq));
                v.extend(a.wk.iter().map(|w| ("attn_k", w)));
                v.extend(a.wv.iter().map(|w| ("attn_v", w)));
                v.push(("attn_output", &a.wo));
                if let Some(g) = &a.gate {
                    v.push(("attn_gate", g));
                }
            }
            Mixer::Conv(c) => v.extend([("shortconv.in_proj", &c.in_proj), ("shortconv.conv", &c.conv), ("shortconv.out_proj", &c.out_proj)]),
        }
        if let Some(p) = &self.ple {
            v.extend([("inp_gate", &p.inp_gate), ("proj", &p.proj)]);
        }
        v
    }
}

pub struct Weights {
    /// `[vocab, dim]`: row `t` is token `t`'s embedding.
    pub token_embedding: Weight,
    pub layers: Vec<LayerWeights>,
    pub final_norm: AlignedBuf,
    /// `[vocab, dim]`, or `None` when tied to `token_embedding`.
    pub classifier: Option<Weight>,
    /// Gemma 4's per-layer embeddings.
    pub ple: Option<PleTable>,
}

impl Weights {
    /// The per-layer blocks' F32 matrices (Q4_K_M's `inp_gate` / `proj`) in
    /// Q8_0, as the GPU loads them: its shaders read no F32 matrix, and Q8_0
    /// takes 29 MB where F32 takes 110. Quantized with the Q8_0 quantizer
    /// that matches llama.cpp's byte for byte; any other format is kept.
    /// `aether --ple-q8` runs the CPU path the same way, for the gate on
    /// accuracy (plans/gemma4-close.md, S4).
    pub fn ple_to_q8_0(&mut self, c: &Config) {
        for l in &mut self.layers {
            if let Some(p) = &mut l.ple {
                for (w, cols) in [(&mut p.inp_gate, c.dim), (&mut p.proj, c.ple_dim)] {
                    if let Weight::F32(_) = w {
                        let f = std::mem::replace(w, Weight::Q8_0(crate::weight::Store::Owned(Vec::new())));
                        *w = f.convert(cols, Format::Q8_0);
                    }
                }
            }
        }
    }

    /// Converts every matrix (not the norms) to `format`, as llama.cpp's
    /// `llama-quantize --pure` does: the embedding/classifier included.
    /// With an importance matrix, each matrix it has an entry for is
    /// quantized with that entry (`--pure --imatrix`); llama-imatrix records
    /// none for the embedding, which is then quantized without one. An
    /// entry of the wrong length is an error, as in `llama-quantize`.
    pub fn convert(self, c: &Config, format: Format, imatrix: Option<&Imatrix>) -> io::Result<Weights> {
        let wrong = std::cell::RefCell::new(None);
        let w = self.map_matrices(c, |w, cols, name| {
            let qw = imatrix.and_then(|im| im.get(name)).filter(|qw| {
                let ok = qw.len() == cols;
                if !ok && name != "token_embd.weight" {
                    wrong.borrow_mut().get_or_insert(format!("imatrix entry for {name} has {} values, the matrix {cols} columns", qw.len()));
                }
                ok
            });
            w.convert_weighted(cols, format, qw)
        });
        match wrong.into_inner() {
            Some(msg) => Err(io::Error::new(io::ErrorKind::InvalidData, msg)),
            None => Ok(w),
        }
    }

    /// Repacks every quantized matrix four rows at a time for the `smmla`
    /// kernels (see [`Weight::repack`]).
    pub fn repack(self, c: &Config) -> Weights {
        self.map_matrices(c, |w, cols, _| w.repack(cols))
    }

    /// Applies `f(matrix, columns, llama.cpp tensor name)` to every matrix,
    /// not the norms.
    fn map_matrices(self, c: &Config, f: impl Fn(Weight, usize, &str) -> Weight) -> Weights {
        let dim = c.dim;
        Weights {
            token_embedding: f(self.token_embedding, dim, "token_embd.weight"),
            layers: self
                .layers
                .into_iter()
                .enumerate()
                .map(|(i, l)| {
                    let g = |w, cols, name: &str| f(w, cols, &format!("blk.{i}.{name}.weight"));
                    let mixer = match l.mixer {
                        Mixer::Attention(a) => Mixer::Attention(AttentionWeights {
                            wq: g(a.wq, dim, "attn_q"),
                            wk: a.wk.map(|w| g(w, dim, "attn_k")),
                            wv: a.wv.map(|w| g(w, dim, "attn_v")),
                            wo: g(a.wo, c.q_dim(i), "attn_output"),
                            bq: a.bq,
                            bk: a.bk,
                            bv: a.bv,
                            q_norm: a.q_norm,
                            k_norm: a.k_norm,
                            gate: a.gate.map(|w| g(w, dim, "attn_gate")),
                        }),
                        // The depthwise kernel is small and stays f32.
                        Mixer::Conv(cw) => Mixer::Conv(ConvWeights {
                            in_proj: g(cw.in_proj, dim, "shortconv.in_proj"),
                            conv: cw.conv,
                            out_proj: g(cw.out_proj, dim, "shortconv.out_proj"),
                        }),
                    };
                    LayerWeights {
                        attn_norm: l.attn_norm,
                        mixer,
                        post_attn_norm: l.post_attn_norm,
                        ffn_norm: l.ffn_norm,
                        post_ffn_norm: l.post_ffn_norm,
                        w_gate: g(l.w_gate, dim, "ffn_gate"),
                        w_down: g(l.w_down, c.hidden_of(i), "ffn_down"),
                        w_up: g(l.w_up, dim, "ffn_up"),
                        ple: l.ple.map(|p| PleWeights { inp_gate: g(p.inp_gate, dim, "inp_gate"), proj: g(p.proj, c.ple_dim, "proj"), post_norm: p.post_norm }),
                        out_scale: l.out_scale,
                    }
                })
                .collect(),
            final_norm: self.final_norm,
            classifier: self.classifier.map(|w| f(w, dim, "output.weight")),
            // The table is only looked up, never converted or repacked.
            ple: self.ple.map(|p| PleTable { table: p.table, proj: f(p.proj, dim, "per_layer_model_proj.weight"), proj_norm: p.proj_norm }),
        }
    }

    /// Bytes of all matrices and norms: what one decode step streams.
    pub fn bytes(&self) -> usize {
        let norms = |l: &LayerWeights| {
            let a = l.attn_norm.len() + l.ffn_norm.len() + l.post_attn_norm.as_ref().map_or(0, |n| n.len()) + l.post_ffn_norm.as_ref().map_or(0, |n| n.len());
            let qk = l.attention().map_or(0, |a| a.q_norm.as_ref().map_or(0, |n| n.len()) + a.k_norm.as_ref().map_or(0, |n| n.len()));
            (a + qk) * 4
        };
        let mats = |l: &LayerWeights| l.matrices().iter().map(|(_, w)| w.bytes()).sum::<usize>();
        self.token_embedding.bytes()
            + self.layers.iter().map(|l| norms(l) + mats(l)).sum::<usize>()
            + self.final_norm.len() * 4
            + self.classifier.as_ref().map_or(0, Weight::bytes)
            + self.ple.as_ref().map_or(0, |p| p.proj.bytes())
    }

    /// The storage format of the first matrix, for the CLI's load line.
    pub fn format_name(&self) -> &'static str {
        self.layers.iter().flat_map(|l| l.matrices()).next().map_or_else(|| self.token_embedding.format_name(), |(_, w)| w.format_name())
    }
}

/// All query heads of one token (decode): `out[h] = attention(q[h], cache,
/// pos)`. Past [`ATTN_CHUNK`] positions with head size 64 or 48 it is
/// computed in chunks ([`attention_chunked`]), otherwise whole
/// ([`attention_whole`]).
#[allow(clippy::too_many_arguments)]
fn attention(pool: Option<&ThreadPool>, c: &Config, layer: usize, out: &mut [f32], q: &[f32], kc: &[f32], vc: &[f32], scores: &mut [f32], pos: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        let hs = c.head_dim_of(layer);
        if c.window(layer).is_none() && (hs == 64 || hs == 48 || hs == 128) && pos + 1 > ATTN_CHUNK {
            return attention_chunked(pool, c, layer, out, q, kc, vc, pos);
        }
    }
    attention_whole(pool, c, layer, out, q, kc, vc, scores, pos)
}

/// [`attention`] with every head over the whole context in one piece (no
/// chunks): the same result, bit for bit, as prefill's grouped attention,
/// so prefill on one thread (which goes token by token) uses this. Heads
/// are independent (each writes its own slice of `out` and its own row of
/// `scores`), so with a pool they run in parallel, one thread per head at a
/// time.
#[allow(clippy::too_many_arguments)]
fn attention_whole(pool: Option<&ThreadPool>, c: &Config, layer: usize, out: &mut [f32], q: &[f32], kc: &[f32], vc: &[f32], scores: &mut [f32], pos: usize) {
    let (hs, kv_dim, group, seq) = (c.head_dim_of(layer), c.kv_dim_of(layer), c.kv_group(), c.seq_len);
    let window = c.window(layer).unwrap_or(usize::MAX);
    let head = |h: usize, out_h: &mut [f32], scores_h: &mut [f32]| {
        let kv_off = (h / group) * hs; // which K/V head this query head reads
        let keys = |t: usize| &kc[t * kv_dim + kv_off..t * kv_dim + kv_off + hs];
        let values = |t: usize| &vc[t * kv_dim + kv_off..t * kv_dim + kv_off + hs];
        k::attention_head_window(out_h, &q[h * hs..(h + 1) * hs], keys, values, pos, window, scores_h);
    };
    match pool.filter(|p| p.threads() > 1) {
        None => {
            for (h, (o, sc)) in out.chunks_exact_mut(hs).zip(scores.chunks_exact_mut(seq)).enumerate() {
                head(h, o, sc);
            }
        }
        Some(p) => {
            assert!(out.len() >= c.n_heads * hs && scores.len() >= c.n_heads * seq);
            let (o, sc) = (SharedOut(out.as_mut_ptr()), SharedOut(scores.as_mut_ptr()));
            p.for_blocks(c.n_heads, 1, &|r| {
                for h in r {
                    // SAFETY: head h is handed to exactly one thread, and its
                    // output slice and score row are disjoint from every other
                    // head's (both lengths checked above).
                    let (oh, sh) = unsafe {
                        (std::slice::from_raw_parts_mut(o.ptr().add(h * hs), hs), std::slice::from_raw_parts_mut(sc.ptr().add(h * seq), seq))
                    };
                    head(h, oh, sh);
                }
            });
        }
    }
}

/// Positions per part of [`attention_chunked`].
#[cfg(target_arch = "aarch64")]
const ATTN_CHUNK: usize = 64;

/// [`attention`] past [`ATTN_CHUNK`] positions (flash-decoding): the work
/// items are (K/V head, chunk of 64 positions), so the threads share
/// `n_kv_heads · chunks` equal items instead of `n_heads` whole heads (14 on
/// 6 threads left some idle), and each item runs all the query heads of its
/// K/V head over its chunk ([`k::neon::attention_group_part`]: each key and
/// value row is loaded once for all of them). The parts of each head are
/// then merged in context order ([`k::neon::merge_parts`]); the chunks
/// depend only on `pos`, so the result does not depend on the threads.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn attention_chunked(pool: Option<&ThreadPool>, c: &Config, layer: usize, out: &mut [f32], q: &[f32], kc: &[f32], vc: &[f32], pos: usize) {
    let (hs, kv_dim, g) = (c.head_dim_of(layer), c.kv_dim_of(layer), c.kv_group());
    let (n, gh) = (pos + 1, g * hs);
    let nc = n.div_ceil(ATTN_CHUNK);
    let items = c.n_kv_heads * nc;
    // Per item: g·hs of unnormalised output, then 2g of (max, sum).
    let per = gh + 2 * g;
    let mut parts = vec![0.0f32; items * per];
    let item = |i: usize, part: &mut [f32]| {
        let (kvh, ch) = (i / nc, i % nc);
        let (t0, len) = (ch * ATTN_CHUNK, ATTN_CHUNK.min(n - ch * ATTN_CHUNK));
        let off = kvh * hs;
        let keys = |t: usize| &kc[(t0 + t) * kv_dim + off..(t0 + t) * kv_dim + off + hs];
        let values = |t: usize| &vc[(t0 + t) * kv_dim + off..(t0 + t) * kv_dim + off + hs];
        let mut scores = [0.0f32; 8 * ATTN_CHUNK];
        let mut big;
        let scores: &mut [f32] = if g * ATTN_CHUNK <= scores.len() {
            &mut scores[..]
        } else {
            big = vec![0.0f32; g * ATTN_CHUNK];
            &mut big[..]
        };
        k::neon::attention_group_part(part, &q[kvh * gh..(kvh + 1) * gh], hs, keys, values, len, scores);
    };
    par_chunks_mut(pool, &mut parts, per, &item);
    for kvh in 0..c.n_kv_heads {
        for h in 0..g {
            let ps: Vec<(&[f32], f32, f32)> = (0..nc)
                .map(|ch| {
                    let p = &parts[(kvh * nc + ch) * per..(kvh * nc + ch + 1) * per];
                    (&p[h * hs..(h + 1) * hs], p[gh + 2 * h], p[gh + 2 * h + 1])
                })
                .collect();
            let o = (kvh * g + h) * hs;
            k::neon::merge_parts(&mut out[o..o + hs], &ps);
        }
    }
}

/// [`attention`] for `t` tokens at positions `start..start + t` (prefill):
/// every (token, K/V head) pair is independent, so with a pool all `t ·
/// n_kv_heads` of them are handed out in one call, rather than one call
/// per token with only a few items to share (which leaves threads waiting
/// at the end of each of those calls). An item is all the query heads that
/// read one K/V head ([`k::attention_group`]), so each cached key and value
/// is loaded once per token rather than once per query head. Items go out
/// longest first (the last token sees the most positions), so the cheap
/// ones fill in at the end. Each block of items has its own scores buffer;
/// every head's result is computed exactly as in [`attention`].
#[allow(clippy::too_many_arguments)]
fn attention_batch(pool: Option<&ThreadPool>, c: &Config, layer: usize, out: &mut [f32], q: &[f32], kc: &[f32], vc: &[f32], scores: &mut [f32], start: usize, t: usize) {
    let window = c.window(layer).unwrap_or(usize::MAX);
    let Some(p) = pool.filter(|p| p.threads() > 1 && t > 1) else {
        for i in 0..t {
            let dim = c.q_dim(layer);
            attention_whole(pool, c, layer, &mut out[i * dim..(i + 1) * dim], &q[i * dim..(i + 1) * dim], kc, vc, scores, start + i);
        }
        return;
    };
    let (hs, kv_dim, group, nh) = (c.head_dim_of(layer), c.kv_dim_of(layer), c.kv_group(), c.n_heads);
    let nkv = nh / group;
    assert!(out.len() >= t * nh * hs && q.len() >= t * nh * hs);
    let o = SharedOut(out.as_mut_ptr());
    let items = t * nkv;
    p.for_blocks(items, crate::pool::block_size(items, p.threads(), 1), &|r| {
        let mut sc = vec![0.0; group * (start + t)];
        for j in r {
            // Item j: token t-1-j/nkv (longest first), K/V head j%nkv and
            // the `group` query heads that read it (consecutive in q).
            let (i, kvh) = (t - 1 - j / nkv, j % nkv);
            let kv_off = kvh * hs;
            let keys = |p: usize| &kc[p * kv_dim + kv_off..p * kv_dim + kv_off + hs];
            let values = |p: usize| &vc[p * kv_dim + kv_off..p * kv_dim + kv_off + hs];
            let h0 = (i * nh + kvh * group) * hs;
            let qg = &q[h0..h0 + group * hs];
            // SAFETY: item (i, kvh) is handed to exactly one thread, and its
            // slice of `out` is disjoint from every other item's (length
            // checked above).
            let og = unsafe { std::slice::from_raw_parts_mut(o.ptr().add(h0), group * hs) };
            k::attention_group_window(og, qg, hs, keys, values, start + i, window, &mut sc);
        }
    });
}

/// One token through a gated short convolution: `in_proj` → split B, C, x →
/// `u = B ⊙ x` → causal depthwise conv with the 6 saved rows → `C ⊙ y` →
/// `out_proj`. Writes the mixer output to `s.xb2`.
fn conv_token(pool: Option<&ThreadPool>, c: &Config, s: &mut State, l: usize, cw: &ConvWeights, width: usize, pos: usize) {
    let dim = c.dim;
    assert!(width <= 7 && width >= 1);
    k::linear_on(pool, &mut s.conv[..3 * dim], &s.xb, &cw.in_proj);
    let (b, rest) = s.conv[..3 * dim].split_at_mut(dim);
    let (cc, x) = rest.split_at_mut(dim);
    for i in 0..dim {
        s.xb[i] = b[i] * x[i];
    }
    let w = match &cw.conv {
        Weight::F32(f) => &f[..],
        _ => panic!("the short-conv kernel must be f32"),
    };
    let state = &mut s.conv_state[s.conv_off[l]..s.conv_off[l] + 6 * dim];
    k::short_conv_step(&s.xb[..dim], w, state, dim, width, &mut s.q[..dim]);
    let base = s.conv_off[l] * (MAX_DRAFT + 1);
    let slot = base + (pos % (MAX_DRAFT + 1)) * 6 * dim;
    s.conv_snap[slot..slot + 6 * dim].copy_from_slice(state);
    for ch in 0..dim {
        s.xb2[ch] = cc[ch] * s.q[ch];
    }
    k::linear_on(pool, &mut s.xb, &s.xb2, &cw.out_proj);
    s.xb2.copy_from_slice(&s.xb);
}

/// [`conv_token`] for `t` tokens; the conv state is advanced token by token.
#[allow(clippy::too_many_arguments)]
fn conv_batch(pool: Option<&ThreadPool>, c: &Config, s: &mut State, b: &mut Batch, l: usize, cw: &ConvWeights, width: usize, start: usize, t: usize) {
    let dim = c.dim;
    assert!(width <= 7 && width >= 1);
    k::linear_batch_on(pool, &mut b.conv[..t * 3 * dim], &b.xb[..t * dim], &cw.in_proj, t);
    let w = match &cw.conv {
        Weight::F32(f) => &f[..],
        _ => panic!("the short-conv kernel must be f32"),
    };
    let state = &mut s.conv_state[s.conv_off[l]..s.conv_off[l] + 6 * dim];
    for i in 0..t {
        let bcx = &b.conv[i * 3 * dim..(i + 1) * 3 * dim];
        let (bb, rest) = bcx.split_at(dim);
        let (_cc, xx) = rest.split_at(dim);
        for ch in 0..dim {
            b.hb2[i * dim + ch] = bb[ch] * xx[ch];
        }
    }
    if t <= MAX_DRAFT {
        let off = s.conv_off[l];
        let base = off * (MAX_DRAFT + 1);
        for i in 0..t {
            let state = &mut s.conv_state[off..off + 6 * dim];
            k::short_conv_step(&b.hb2[i * dim..(i + 1) * dim], w, state, dim, width, &mut b.q[i * dim..(i + 1) * dim]);
            let slot = base + ((start + i) % (MAX_DRAFT + 1)) * 6 * dim;
            s.conv_snap[slot..slot + 6 * dim].copy_from_slice(state);
        }
    } else {
        k::short_conv_rows(&b.hb2[..t * dim], w, state, dim, width, &mut b.q[..t * dim], t);
    }
    for i in 0..t {
        let bcx = &b.conv[i * 3 * dim..(i + 1) * 3 * dim];
        let (_, rest) = bcx.split_at(dim);
        let (cc, _) = rest.split_at(dim);
        for ch in 0..dim {
            b.xb2[i * dim + ch] = cc[ch] * b.q[i * dim + ch];
        }
    }
    k::linear_batch_on(pool, &mut b.xb[..t * dim], &b.xb2[..t * dim], &cw.out_proj, t);
    b.xb2[..t * dim].copy_from_slice(&b.xb[..t * dim]);
}

/// One token's Q or K row: `+= bias` (if any), then this layer's RoPE at
/// `pos`.
fn bias_rope(c: &Config, layer: usize, v: &mut [f32], bias: Option<&AlignedBuf>, pos: usize) {
    add_bias(v, bias);
    let (rope, hs) = (c.rope(layer), c.head_dim_of(layer));
    match rope.layout {
        RopeLayout::AdjacentPairs => k::rope_adjacent(v, pos, hs, rope.theta),
        RopeLayout::Halves => k::rope_halves_pairs(v, pos, hs, rope.theta, rope.pairs),
    }
}

/// `x += bias` when the layer has one.
fn add_bias(x: &mut [f32], bias: Option<&AlignedBuf>) {
    if let Some(b) = bias {
        k::add_inplace(x, b);
    }
}

/// The MLP's `gate = act(gate) · up`.
fn activate(c: &Config, gate: &mut [f32], up: &[f32]) {
    match c.activation {
        Activation::SwiGlu => k::swiglu(gate, up),
        Activation::GeluTanh => k::gelu_tanh(gate, up),
    }
}

/// Gemma 4's per-layer inputs of one token, `[n_layers, ple_dim]`, from
/// the scaled embedding `e` (llama.cpp's `project_per_layer_inputs`):
/// `(norm(proj · e / √dim) + table[token] · √ple_dim) / √2`, the norm per
/// layer's slice with the shared gain.
pub(crate) fn ple_inputs(pool: Option<&ThreadPool>, c: &Config, p: &PleTable, token: usize, e: &[f32], out: &mut [f32]) {
    k::linear_on(pool, out, e, &p.proj);
    ple_finish(c, p, token, out);
}

/// [`ple_inputs`] after the projection `out = proj · e`.
fn ple_finish(c: &Config, p: &PleTable, token: usize, out: &mut [f32]) {
    k::scale_inplace(out, 1.0 / (c.dim as f32).sqrt());
    for slice in out.chunks_exact_mut(c.ple_dim) {
        k::rmsnorm_inplace(slice, &p.proj_norm, c.norm_eps);
    }
    let mut row = vec![0.0f32; out.len()];
    p.table.row_f32(token, &mut row);
    let s = (c.ple_dim as f32).sqrt();
    for (o, &r) in out.iter_mut().zip(&row) {
        *o = (*o + r * s) * std::f32::consts::FRAC_1_SQRT_2;
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

/// The deepest speculative draft the conv-state snapshots cover.
pub(crate) const MAX_DRAFT: usize = 32;

/// Prefill is cut into chunks this large for hybrid models, so the sliding
/// ring only has to keep `window + chunk` positions.
const PREFILL_CHUNK: usize = 256;

/// Buffers reused by every forward call, allocated once.
struct State {
    x: AlignedBuf,      // residual stream [dim]
    xb: AlignedBuf,     // normed input [dim] / attention output [q_dim]
    xb2: AlignedBuf,    // projection output [dim]
    hb: AlignedBuf,     // MLP gate [hidden]
    hb2: AlignedBuf,    // MLP up [hidden]
    conv: AlignedBuf,   // conv B,C,x [3·dim]
    q: AlignedBuf,      // [q_dim]
    att: AlignedBuf,    // attention scores, one row per head [n_heads, seq_len]
    gate: AlignedBuf,   // one attention gate per head [n_heads]
    logits: AlignedBuf, // [vocab]
    /// Keys of every attention layer, concatenated: layer `l` starts at
    /// float `cache_off[l]` and has `cache_rows[l]` positions of
    /// `kv_dim_of(l)` floats (none for a layer reading another's cache).
    key_cache: AlignedBuf,
    value_cache: AlignedBuf,
    cache_off: Vec<usize>,
    cache_rows: Vec<usize>,
    /// The last 6 `u` rows of every conv layer, concatenated.
    conv_state: AlignedBuf,
    conv_off: Vec<usize>,
    /// The last `MAX_DRAFT + 1` states of every conv layer, for rollback.
    conv_snap: AlignedBuf,
    /// A linear copy of the sliding window for the current layer/token
    /// (or chunk): `window + PREFILL_CHUNK` K/V rows.
    slide_k: AlignedBuf,
    slide_v: AlignedBuf,
    /// Gemma 4: this token's per-layer embedding inputs `[n_layers,
    /// ple_dim]`, and one layer's gate `[ple_dim]`.
    ple_in: AlignedBuf,
    ple_g: AlignedBuf,
}

/// Buffers for [`Transformer::forward_batch`]: the single-token buffers,
/// one row per token. Grown on demand to the largest batch seen.
struct Batch {
    cap: usize,
    /// Logits of several positions, `[k, vocab]` (perplexity evaluation).
    logits: Vec<f32>,
    x: AlignedBuf,   // [cap, dim]
    xb: AlignedBuf,  // [cap, dim] / [cap, q_dim]
    xb2: AlignedBuf, // [cap, dim]
    q: AlignedBuf,    // [cap, q_dim]
    gate: AlignedBuf, // [cap, n_heads]
    hb: AlignedBuf,   // [cap, hidden]
    hb2: AlignedBuf,  // [cap, hidden]
    conv: AlignedBuf, // [cap, 3·dim]
    ple_in: AlignedBuf, // [cap, n_layers · ple_dim]
    ple_g: AlignedBuf,  // [cap, ple_dim]
}

impl Batch {
    fn new(c: &Config, cap: usize) -> Batch {
        let z = |n: usize| AlignedBuf::zeros(cap * n);
        Batch {
            cap,
            logits: Vec::new(),
            x: z(c.dim),
            xb: z(c.dim.max(c.max_q_dim())),
            xb2: z(c.dim),
            q: z(c.max_q_dim()),
            gate: z(c.n_heads),
            hb: z(c.max_hidden()),
            hb2: z(c.max_hidden()),
            conv: z(3 * c.dim),
            ple_in: z(c.n_layers * c.ple_dim),
            ple_g: z(c.ple_dim),
        }
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
        let mut cache_off = Vec::with_capacity(c.n_layers);
        let mut cache_rows = Vec::with_capacity(c.n_layers);
        let mut total = 0;
        for l in 0..c.n_layers {
            cache_off.push(total);
            let rows = if c.is_attention(l) && c.owns_kv(l) {
                match c.window(l) {
                    Some(w) => (w + PREFILL_CHUNK).min(c.seq_len),
                    None => c.seq_len,
                }
            } else {
                0
            };
            cache_rows.push(rows);
            total += rows * c.kv_dim_of(l);
        }
        let max_window = (0..c.n_layers).filter_map(|l| c.window(l)).max().unwrap_or(0);
        let mut conv_off = Vec::with_capacity(c.n_layers);
        let mut conv_total = 0;
        for l in 0..c.n_layers {
            conv_off.push(conv_total);
            if let crate::model::config::LayerKind::Conv { .. } = c.kind(l) {
                conv_total += 6 * c.dim;
            }
        }
        let state = State {
            x: AlignedBuf::zeros(c.dim),
            xb: AlignedBuf::zeros(c.dim.max(c.max_q_dim())),
            xb2: AlignedBuf::zeros(c.dim),
            hb: AlignedBuf::zeros(c.max_hidden()),
            hb2: AlignedBuf::zeros(c.max_hidden()),
            conv: AlignedBuf::zeros(3 * c.dim),
            q: AlignedBuf::zeros(c.max_q_dim()),
            att: AlignedBuf::zeros(c.n_heads * c.seq_len),
            gate: AlignedBuf::zeros(c.n_heads),
            logits: AlignedBuf::zeros(c.vocab_size),
            key_cache: AlignedBuf::zeros(total),
            value_cache: AlignedBuf::zeros(total),
            cache_off,
            cache_rows,
            conv_state: AlignedBuf::zeros(conv_total),
            conv_off,
            conv_snap: AlignedBuf::zeros(conv_total * (MAX_DRAFT + 1)),
            slide_k: AlignedBuf::zeros(if max_window > 0 { (max_window + PREFILL_CHUNK) * c.max_kv_dim() } else { 0 }),
            slide_v: AlignedBuf::zeros(if max_window > 0 { (max_window + PREFILL_CHUNK) * c.max_kv_dim() } else { 0 }),
            ple_in: AlignedBuf::zeros(c.n_layers * c.ple_dim),
            ple_g: AlignedBuf::zeros(c.ple_dim),
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

    /// Exchanges thread pools with `other` (a draft model borrows the
    /// target's threads for a few steps instead of spinning its own).
    pub fn swap_pool(&mut self, other: &mut Transformer) {
        std::mem::swap(&mut self.pool, &mut other.pool);
    }

    /// Restores every conv layer to the state after position `pos` (the
    /// snapshots of the last [`MAX_DRAFT`] + 1 tokens). Speculative decoding
    /// calls this after rejecting drafts; the KV entries of rejected drafts
    /// are simply overwritten later.
    pub fn restore_conv(&mut self, pos: usize) {
        let c = &self.config;
        let s = &mut self.state;
        for l in 0..c.n_layers {
            if c.is_attention(l) {
                continue;
            }
            let off = s.conv_off[l];
            let slot = off * (MAX_DRAFT + 1) + (pos % (MAX_DRAFT + 1)) * 6 * c.dim;
            s.conv_state[off..off + 6 * c.dim].copy_from_slice(&s.conv_snap[slot..slot + 6 * c.dim]);
        }
    }

    /// A debug view of the conv state (tests and speculative rollback).
    pub fn conv_state(&self) -> &[f32] {
        &self.state.conv_state
    }

    /// The KV cache, keys and values, each `[layer, pos, kv_dim]` (with
    /// `seq_len` positions per layer), e.g. to hand a prefilled prompt to
    /// the GPU decode path.
    pub fn kv_cache(&self) -> (&[f32], &[f32]) {
        (&self.state.key_cache, &self.state.value_cache)
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
        let dim = c.dim;
        assert!(pos < c.seq_len, "position {pos} beyond seq_len {}", c.seq_len);
        assert!(token < c.vocab_size, "token {token} beyond vocab {}", c.vocab_size);

        if pos == 0 {
            s.conv_state.fill(0.0);
        }
        w.token_embedding.row_f32(token, &mut s.x);
        if c.embed_scale != 1.0 {
            k::scale_inplace(&mut s.x, c.embed_scale);
        }
        obs.tap("embed", None, pos, &s.x);
        if let Some(p) = &w.ple {
            ple_inputs(pool, c, p, token, &s.x, &mut s.ple_in);
            obs.tap("ple", None, pos, &s.ple_in);
        }

        for (l, lw) in w.layers.iter().enumerate() {
            let (hs, qd, kv_dim) = (c.head_dim_of(l), c.q_dim(l), c.kv_dim_of(l));
            k::rmsnorm(&mut s.xb[..dim], &s.x, &lw.attn_norm, c.norm_eps);
            obs.tap("attn_norm", Some(l), pos, &s.xb[..dim]);

            match &lw.mixer {
                Mixer::Attention(a) => {
                    let window = c.window(l);
                    // A layer that shares KV reads its source's cache, which
                    // already holds this position (the source runs first).
                    let (src, own) = (c.kv_source_of(l), c.owns_kv(l));
                    let layer_off = s.cache_off[src];
                    let (att_pos, at);
                    if let Some(w) = window {
                        let ring = s.cache_rows[src];
                        let pre = (w - 1).min(pos);
                        let rows = if own { pre } else { pre + 1 };
                        for j in 0..rows {
                            let from = layer_off + ((pos - pre + j) % ring) * kv_dim;
                            let dst = j * kv_dim;
                            s.slide_k[dst..dst + kv_dim].copy_from_slice(&s.key_cache[from..from + kv_dim]);
                            s.slide_v[dst..dst + kv_dim].copy_from_slice(&s.value_cache[from..from + kv_dim]);
                        }
                        att_pos = pre;
                        at = pre * kv_dim;
                    } else {
                        att_pos = pos;
                        at = pos * kv_dim;
                    }
                    if let Some(g) = &a.gate {
                        k::linear_on(pool, &mut s.gate, &s.xb[..dim], g);
                    }
                    let (k, v): (&mut [f32], &mut [f32]) = match window {
                        Some(_) => (&mut s.slide_k[..], &mut s.slide_v[..]),
                        None => (&mut s.key_cache[layer_off..], &mut s.value_cache[layer_off..]),
                    };
                    let q = &mut s.q[..qd];
                    match (&a.wk, &a.wv) {
                        (Some(wk), Some(wv)) if own => {
                            let qkv: &mut [&mut [f32]] = &mut [&mut *q, &mut k[at..at + kv_dim], &mut v[at..at + kv_dim]];
                            k::linear_many_on(pool, qkv, &s.xb[..dim], &[&a.wq, wk, wv]);
                        }
                        _ => k::linear_on(pool, q, &s.xb[..dim], &a.wq),
                    }
                    add_bias(q, a.bq.as_ref());
                    if let Some(n) = &a.q_norm {
                        k::rmsnorm_heads(q, n, hs, c.norm_eps);
                    }
                    bias_rope(c, l, q, None, pos);
                    if own {
                        add_bias(&mut k[at..at + kv_dim], a.bk.as_ref());
                        add_bias(&mut v[at..at + kv_dim], a.bv.as_ref());
                        if let Some(n) = &a.k_norm {
                            k::rmsnorm_heads(&mut k[at..at + kv_dim], n, hs, c.norm_eps);
                        }
                        if c.v_norm {
                            k::rmsnorm_heads_plain(&mut v[at..at + kv_dim], hs, c.norm_eps);
                        }
                        bias_rope(c, l, &mut k[at..at + kv_dim], None, pos);
                    }
                    obs.tap("q", Some(l), pos, q);
                    obs.tap("k", Some(l), pos, &k[at..at + kv_dim]);
                    obs.tap("v", Some(l), pos, &v[at..at + kv_dim]);
                    // The kernels scale scores by 1/√hs; another scale is
                    // folded into q.
                    if let Some(sc) = c.attn_scale {
                        k::scale_inplace(q, sc * (hs as f32).sqrt());
                    }

                    attention(pool, c, l, &mut s.xb[..qd], q, k, v, &mut s.att, att_pos);
                    obs.tap("attn", Some(l), pos, &s.xb[..qd]);
                    if a.gate.is_some() {
                        k::gate_heads(&mut s.xb[..qd], &s.gate, hs);
                    }

                    if window.is_some() && own {
                        let ring = s.cache_rows[l];
                        let slot = layer_off + (pos % ring) * kv_dim;
                        s.key_cache[slot..slot + kv_dim].copy_from_slice(&s.slide_k[at..at + kv_dim]);
                        s.value_cache[slot..slot + kv_dim].copy_from_slice(&s.slide_v[at..at + kv_dim]);
                    }

                    k::linear_on(pool, &mut s.xb2, &s.xb[..qd], &a.wo);
                }
                Mixer::Conv(cw) => {
                    let width = match c.kind(l) {
                        crate::model::config::LayerKind::Conv { width } => width,
                        _ => unreachable!(),
                    };
                    conv_token(pool, c, s, l, cw, width, pos);
                }
            }
            if let Some(n) = &lw.post_attn_norm {
                k::rmsnorm_inplace(&mut s.xb2, n, c.norm_eps);
            }
            obs.tap("attn_out", Some(l), pos, &s.xb2);
            k::add_inplace(&mut s.x, &s.xb2);
            obs.tap("resid_attn", Some(l), pos, &s.x);

            // --- MLP ---
            let hidden = c.hidden_of(l);
            k::rmsnorm(&mut s.xb[..dim], &s.x, &lw.ffn_norm, c.norm_eps);
            obs.tap("ffn_norm", Some(l), pos, &s.xb[..dim]);
            k::linear_many_on(pool, &mut [&mut s.hb[..hidden], &mut s.hb2[..hidden]], &s.xb[..dim], &[&lw.w_gate, &lw.w_up]);
            activate(c, &mut s.hb[..hidden], &s.hb2[..hidden]);
            obs.tap("swiglu", Some(l), pos, &s.hb[..hidden]);
            k::linear_on(pool, &mut s.xb[..dim], &s.hb[..hidden], &lw.w_down);
            if let Some(n) = &lw.post_ffn_norm {
                k::rmsnorm_inplace(&mut s.xb[..dim], n, c.norm_eps);
            }
            obs.tap("ffn_out", Some(l), pos, &s.xb[..dim]);
            k::add_inplace(&mut s.x, &s.xb[..dim]);
            obs.tap("resid_ffn", Some(l), pos, &s.x);

            // --- per-layer embedding (Gemma 4) ---
            if let Some(p) = &lw.ple {
                let pd = c.ple_dim;
                k::linear_on(pool, &mut s.ple_g, &s.x, &p.inp_gate);
                k::gelu_tanh(&mut s.ple_g, &s.ple_in[l * pd..(l + 1) * pd]);
                k::linear_on(pool, &mut s.xb[..dim], &s.ple_g, &p.proj);
                k::rmsnorm_inplace(&mut s.xb[..dim], &p.post_norm, c.norm_eps);
                obs.tap("ple_out", Some(l), pos, &s.xb[..dim]);
                k::add_inplace(&mut s.x, &s.xb[..dim]);
            }
            if let Some(sc) = lw.out_scale {
                k::scale_inplace(&mut s.x, sc);
            }
            obs.tap("layer_out", Some(l), pos, &s.x);
        }

        k::rmsnorm_inplace(&mut s.x, &w.final_norm, c.norm_eps);
        obs.tap("final_norm", None, pos, &s.x);
        let cls = w.classifier.as_ref().unwrap_or(&w.token_embedding);
        k::linear_on(pool, &mut s.logits, &s.x, cls);
        if let Some(cap) = c.logit_softcap {
            k::softcap(&mut s.logits, cap);
        }
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
        if t <= PREFILL_CHUNK || !(self.config.has_conv() || self.config.has_window()) {
            return self.batch_impl(tokens, start, t - 1, obs);
        }
        let mut last = Vec::new();
        for a in (0..t).step_by(PREFILL_CHUNK) {
            let b = (a + PREFILL_CHUNK).min(t);
            last = self.batch_impl(&tokens[a..b], start + a, b - a - 1, obs).to_vec();
        }
        self.state.logits[..last.len()].copy_from_slice(&last);
        &self.state.logits[..last.len()]
    }

    /// Runs `tokens` at positions `start ..` and returns the logits of every
    /// token from index `from` on, `[tokens.len() - from, vocab]` row-major
    /// (row `r` predicts the token after `tokens[from + r]`). Used to score
    /// text: perplexity needs a prediction at every position.
    pub fn forward_batch_logits(&mut self, tokens: &[usize], start: usize, from: usize) -> &[f32] {
        assert!(from < tokens.len());
        let t = tokens.len();
        if t <= PREFILL_CHUNK || !(self.config.has_conv() || self.config.has_window()) {
            return self.batch_impl(tokens, start, from, &mut NoTrace);
        }
        let vocab = self.config.vocab_size;
        let mut out = Vec::with_capacity((t - from) * vocab);
        for a in (0..t).step_by(PREFILL_CHUNK) {
            let b = (a + PREFILL_CHUNK).min(t);
            let local_from = if b <= from { b - a - 1 } else { from.saturating_sub(a) };
            let rows = self.batch_impl(&tokens[a..b], start + a, local_from, &mut NoTrace).to_vec();
            if b > from {
                let first = from.max(a);
                let skip = first - a - local_from;
                let count = b - first;
                out.extend_from_slice(&rows[skip * vocab..(skip + count) * vocab]);
            }
        }
        self.batch.logits = out;
        &self.batch.logits
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
        if start == 0 {
            s.conv_state.fill(0.0);
        }
        let dim = c.dim;

        for (i, &tok) in tokens.iter().enumerate() {
            assert!(tok < c.vocab_size, "token {tok} beyond vocab {}", c.vocab_size);
            let x = &mut b.x[i * dim..(i + 1) * dim];
            w.token_embedding.row_f32(tok, x);
            if c.embed_scale != 1.0 {
                k::scale_inplace(x, c.embed_scale);
            }
            obs.tap("embed", None, start + i, x);
        }
        if let Some(p) = &w.ple {
            let n = c.n_layers * c.ple_dim;
            let x = &b.x;
            k::linear_batch_on(pool, &mut b.ple_in[..t * n], &x[..t * dim], &p.proj, t);
            par_chunks_mut(pool, &mut b.ple_in[..t * n], n, &|i, o| ple_finish(c, p, tokens[i], o));
            for i in 0..t {
                obs.tap("ple", None, start + i, &b.ple_in[i * n..(i + 1) * n]);
            }
        }

        for (l, lw) in w.layers.iter().enumerate() {
            let (hs, qd, kv_dim, hidden) = (c.head_dim_of(l), c.q_dim(l), c.kv_dim_of(l), c.hidden_of(l));
            // --- attention ---
            // Per-token steps between the matmuls go across the pool too
            // (`par_chunks_mut`); alone on the calling thread they left the
            // workers waiting for a third of prefill.
            let x = &b.x;
            par_chunks_mut(pool, &mut b.xb[..t * dim], dim, &|i, o| k::rmsnorm(o, &x[i * dim..(i + 1) * dim], &lw.attn_norm, c.norm_eps));
            for i in 0..t {
                obs.tap("attn_norm", Some(l), start + i, &b.xb[i * dim..(i + 1) * dim]);
            }
            match &lw.mixer {
                Mixer::Attention(a) => {
                    let window = c.window(l);
                    // A layer that shares KV reads its source's cache, which
                    // already holds these positions (the source runs first).
                    let (src, own) = (c.kv_source_of(l), c.owns_kv(l));
                    let layer_off = s.cache_off[src];
                    let (att_start, c0, c1) = if let Some(w) = window {
                        let ring = s.cache_rows[src];
                        let pre = (w - 1).min(start);
                        let rows = if own { pre } else { pre + t };
                        for j in 0..rows {
                            let from = layer_off + ((start - pre + j) % ring) * kv_dim;
                            let dst = j * kv_dim;
                            s.slide_k[dst..dst + kv_dim].copy_from_slice(&s.key_cache[from..from + kv_dim]);
                            s.slide_v[dst..dst + kv_dim].copy_from_slice(&s.value_cache[from..from + kv_dim]);
                        }
                        (pre, pre * kv_dim, (pre + t) * kv_dim)
                    } else {
                        (start, start * kv_dim, (start + t) * kv_dim)
                    };
                    let (k, v): (&mut [f32], &mut [f32]) = match window {
                        Some(_) => (&mut s.slide_k[..], &mut s.slide_v[..]),
                        None => (&mut s.key_cache[layer_off..], &mut s.value_cache[layer_off..]),
                    };
                    let q = &mut b.q[..t * qd];
                    match (&a.wk, &a.wv) {
                        (Some(wk), Some(wv)) if own => {
                            let qkv: &mut [&mut [f32]] = &mut [&mut *q, &mut k[c0..c1], &mut v[c0..c1]];
                            k::linear_many_batch_on(pool, qkv, &b.xb[..t * dim], &[&a.wq, wk, wv], t);
                        }
                        _ => k::linear_batch_on(pool, q, &b.xb[..t * dim], &a.wq, t),
                    }
                    par_chunks_mut(pool, q, qd, &|_, q| add_bias(q, a.bq.as_ref()));
                    if let Some(n) = &a.q_norm {
                        par_chunks_mut(pool, q, qd, &|_, q| k::rmsnorm_heads(q, n, hs, c.norm_eps));
                    }
                    par_chunks_mut(pool, q, qd, &|i, q| bias_rope(c, l, q, None, start + i));
                    if own {
                        par_chunks_mut(pool, &mut k[c0..c1], kv_dim, &|_, kr| add_bias(kr, a.bk.as_ref()));
                        par_chunks_mut(pool, &mut v[c0..c1], kv_dim, &|_, vv| add_bias(vv, a.bv.as_ref()));
                        if let Some(n) = &a.k_norm {
                            par_chunks_mut(pool, &mut k[c0..c1], kv_dim, &|_, kr| k::rmsnorm_heads(kr, n, hs, c.norm_eps));
                        }
                        if c.v_norm {
                            par_chunks_mut(pool, &mut v[c0..c1], kv_dim, &|_, vv| k::rmsnorm_heads_plain(vv, hs, c.norm_eps));
                        }
                        par_chunks_mut(pool, &mut k[c0..c1], kv_dim, &|i, kr| bias_rope(c, l, kr, None, start + i));
                    }
                    if let Some(g) = &a.gate {
                        k::linear_batch_on(pool, &mut b.gate[..t * c.n_heads], &b.xb[..t * dim], g, t);
                    }
                    for i in 0..t {
                        let pos = start + i;
                        obs.tap("q", Some(l), pos, &q[i * qd..(i + 1) * qd]);
                        obs.tap("k", Some(l), pos, &k[c0 + i * kv_dim..c0 + (i + 1) * kv_dim]);
                        obs.tap("v", Some(l), pos, &v[c0 + i * kv_dim..c0 + (i + 1) * kv_dim]);
                    }
                    // The kernels scale scores by 1/√hs; another scale is
                    // folded into q.
                    if let Some(sc) = c.attn_scale {
                        k::scale_inplace(q, sc * (hs as f32).sqrt());
                    }

                    attention_batch(pool, c, l, &mut b.xb[..t * qd], q, k, v, &mut s.att, att_start, t);
                    for i in 0..t {
                        obs.tap("attn", Some(l), start + i, &b.xb[i * qd..(i + 1) * qd]);
                        if a.gate.is_some() {
                            k::gate_heads(&mut b.xb[i * qd..(i + 1) * qd], &b.gate[i * c.n_heads..(i + 1) * c.n_heads], hs);
                        }
                    }
                    if window.is_some() && own {
                        let ring = s.cache_rows[l];
                        for i in 0..t {
                            let slot = layer_off + ((start + i) % ring) * kv_dim;
                            s.key_cache[slot..slot + kv_dim].copy_from_slice(&s.slide_k[c0 + i * kv_dim..c0 + (i + 1) * kv_dim]);
                            s.value_cache[slot..slot + kv_dim].copy_from_slice(&s.slide_v[c0 + i * kv_dim..c0 + (i + 1) * kv_dim]);
                        }
                    }
                    k::linear_batch_on(pool, &mut b.xb2[..t * dim], &b.xb[..t * qd], &a.wo, t);
                }
                Mixer::Conv(cw) => {
                    let width = match c.kind(l) {
                        crate::model::config::LayerKind::Conv { width } => width,
                        _ => unreachable!(),
                    };
                    conv_batch(pool, c, s, b, l, cw, width, start, t);
                }
            }
            if let Some(n) = &lw.post_attn_norm {
                par_chunks_mut(pool, &mut b.xb2[..t * dim], dim, &|_, o| k::rmsnorm_inplace(o, n, c.norm_eps));
            }
            for i in 0..t {
                obs.tap("attn_out", Some(l), start + i, &b.xb2[i * dim..(i + 1) * dim]);
            }
            let xb2 = &b.xb2;
            par_chunks_mut(pool, &mut b.x[..t * dim], dim, &|i, x| k::add_inplace(x, &xb2[i * dim..(i + 1) * dim]));
            for i in 0..t {
                obs.tap("resid_attn", Some(l), start + i, &b.x[i * dim..(i + 1) * dim]);
            }

            // --- MLP ---
            let x = &b.x;
            par_chunks_mut(pool, &mut b.xb[..t * dim], dim, &|i, o| k::rmsnorm(o, &x[i * dim..(i + 1) * dim], &lw.ffn_norm, c.norm_eps));
            for i in 0..t {
                obs.tap("ffn_norm", Some(l), start + i, &b.xb[i * dim..(i + 1) * dim]);
            }
            k::linear_many_batch_on(pool, &mut [&mut b.hb[..t * hidden], &mut b.hb2[..t * hidden]], &b.xb[..t * dim], &[&lw.w_gate, &lw.w_up], t);
            let hb2 = &b.hb2;
            par_chunks_mut(pool, &mut b.hb[..t * hidden], hidden, &|i, g| activate(c, g, &hb2[i * hidden..(i + 1) * hidden]));
            for i in 0..t {
                obs.tap("swiglu", Some(l), start + i, &b.hb[i * hidden..(i + 1) * hidden]);
            }
            k::linear_batch_on(pool, &mut b.xb[..t * dim], &b.hb[..t * hidden], &lw.w_down, t);
            if let Some(n) = &lw.post_ffn_norm {
                par_chunks_mut(pool, &mut b.xb[..t * dim], dim, &|_, o| k::rmsnorm_inplace(o, n, c.norm_eps));
            }
            for i in 0..t {
                obs.tap("ffn_out", Some(l), start + i, &b.xb[i * dim..(i + 1) * dim]);
            }
            let xb = &b.xb;
            par_chunks_mut(pool, &mut b.x[..t * dim], dim, &|i, x| k::add_inplace(x, &xb[i * dim..(i + 1) * dim]));
            for i in 0..t {
                obs.tap("resid_ffn", Some(l), start + i, &b.x[i * dim..(i + 1) * dim]);
            }

            // --- per-layer embedding (Gemma 4) ---
            if let Some(p) = &lw.ple {
                let (pd, n) = (c.ple_dim, c.n_layers * c.ple_dim);
                k::linear_batch_on(pool, &mut b.ple_g[..t * pd], &b.x[..t * dim], &p.inp_gate, t);
                let ple_in = &b.ple_in;
                par_chunks_mut(pool, &mut b.ple_g[..t * pd], pd, &|i, g| k::gelu_tanh(g, &ple_in[i * n + l * pd..i * n + (l + 1) * pd]));
                k::linear_batch_on(pool, &mut b.xb[..t * dim], &b.ple_g[..t * pd], &p.proj, t);
                par_chunks_mut(pool, &mut b.xb[..t * dim], dim, &|_, o| k::rmsnorm_inplace(o, &p.post_norm, c.norm_eps));
                for i in 0..t {
                    obs.tap("ple_out", Some(l), start + i, &b.xb[i * dim..(i + 1) * dim]);
                }
                let xb = &b.xb;
                par_chunks_mut(pool, &mut b.x[..t * dim], dim, &|i, x| k::add_inplace(x, &xb[i * dim..(i + 1) * dim]));
            }
            if let Some(sc) = lw.out_scale {
                par_chunks_mut(pool, &mut b.x[..t * dim], dim, &|_, x| k::scale_inplace(x, sc));
            }
            for i in 0..t {
                obs.tap("layer_out", Some(l), start + i, &b.x[i * dim..(i + 1) * dim]);
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
            if let Some(cap) = c.logit_softcap {
                k::softcap(&mut s.logits, cap);
            }
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
        if let Some(cap) = c.logit_softcap {
            let v = c.vocab_size;
            par_chunks_mut(pool, &mut b.logits, v, &|_, r| k::softcap(r, cap));
        }
        &b.logits
    }
}
