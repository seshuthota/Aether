//! The decode step on the Hexagon DSP (`--features hexagon`): the whole
//! forward pass of one token in one call (dsp/model.c). The CPU uploads the
//! model once into a shared buffer; per token it writes the token's
//! embedding and the RoPE table, and the DSP runs every layer and the
//! classifier and finds the greedy next token.
//!
//! Like the GPU path, it needs every matrix in Q4_0 (a `--pure` Q4_0 GGUF).

use crate::kernels::hexagon::{Dsp, Result, Shared, q4_0_bytes, repack_q4_0};
use crate::model::transformer::Weights;
use crate::weight::Weight;
use crate::model::{Config, RopeLayout};
use crate::quant::BlockQ4_0;

fn q4_0<'a>(w: &'a Weight, what: &str) -> Result<&'a [BlockQ4_0]> {
    match w {
        Weight::Q4_0(b) => Ok(b),
        _ => Err(format!("{what} is {}, the DSP path needs Q4_0 (a --pure Q4_0 GGUF)", w.format_name())),
    }
}

/// The stages of the DSP's step that it times (dsp/model.c's `P_` enum):
/// each from the end of the one before to when every thread has finished
/// it. `start` is from the call's start to the threads' start.
pub const STAGES: [&str; 13] = [
    "attn norm + quantize",
    "q/k/v",
    "bias + RoPE + KV store",
    "attention + quantize",
    "wo",
    "ffn norm + quantize",
    "gate/up",
    "SwiGLU + quantize",
    "down",
    "final norm + quantize",
    "classifier",
    "argmax",
    "thread start",
];

/// Every product's weights by DMA into VTCM ([`DspModel::dma`]): 6.8 ms a
/// token on Qwen2.5-0.5B against 10.4 with `l2fetch` (results/phase-l5.md).
pub const DMA_ALL: u32 = 0x1f;

/// Places byte ranges, 128-byte aligned (an HVX vector), one after another.
#[derive(Default)]
struct Layout(usize);

impl Layout {
    fn take(&mut self, bytes: usize) -> usize {
        let at = self.0;
        self.0 = (at + bytes).div_ceil(128) * 128;
        at
    }
}

/// What dsp/model.c's `desc` holds, in its order (see the enum there).
mod d {
    pub const DIM: usize = 0;
    pub const HIDDEN: usize = 1;
    pub const LAYERS: usize = 2;
    pub const HEADS: usize = 3;
    pub const KV_HEADS: usize = 4;
    pub const VOCAB: usize = 5;
    pub const SEQ_LEN: usize = 6;
    pub const HALVES: usize = 7;
    pub const EPS: usize = 8;
    pub const W_FD: usize = 9;
    pub const S_FD: usize = 10;
    pub const W_BYTES: usize = 11;
    pub const S_BYTES: usize = 12;
    pub const FINAL_NORM: usize = 13;
    pub const CLS: usize = 14;
    pub const X: usize = 15;
    pub const XB: usize = 16;
    pub const XQ: usize = 17;
    pub const QKV: usize = 18;
    pub const ATT: usize = 19;
    pub const HB: usize = 20;
    pub const LOGITS: usize = 21;
    pub const KC: usize = 22;
    pub const VC: usize = 23;
    pub const ROT: usize = 24;
    pub const PROF: usize = 25;
    pub const LAYER0: usize = 26;
    pub const L_N: usize = 9;
}

pub struct DspModel {
    pub config: Config,
    weights: Weights,
    /// The matrices, norms and biases, kept alive while the DSP uses them.
    _w: Shared,
    state: Shared,
    x_at: usize,
    rot_at: usize,
    logits_at: usize,
    prof_at: usize,
    pub threads: u32,
    /// Which products fetch by DMA ([`Dsp::model_decode`]); all by default.
    pub dma: u32,
    next: usize,
    /// The DSP's own time for the last step, in microseconds.
    pub last_us: u64,
}

impl DspModel {
    pub fn new(dsp: &Dsp, config: Config, weights: Weights, threads: u32) -> Result<DspModel> {
        let c = &config;
        let (dim, kv_dim, hid, vocab) = (c.dim, c.kv_dim(), c.hidden_dim, c.vocab_size);
        let nl = weights.layers.len();

        // The weights buffer: first where everything goes, then the bytes.
        let mut wl = Layout::default();
        let mut desc = vec![0u32; d::LAYER0 + nl * d::L_N];
        let mat = |wl: &mut Layout, rows: usize, cols: usize| wl.take(q4_0_bytes(rows, cols / 32));
        let floats = |wl: &mut Layout, n: usize| wl.take(4 * n);
        for l in 0..nl {
            let o = &mut desc[d::LAYER0 + l * d::L_N..][..d::L_N];
            o[0] = mat(&mut wl, dim + 2 * kv_dim, dim) as u32;
            o[1] = mat(&mut wl, dim, dim) as u32;
            o[2] = mat(&mut wl, 2 * hid, dim) as u32;
            o[3] = mat(&mut wl, dim, hid) as u32;
            o[4] = floats(&mut wl, dim) as u32;
            o[5] = floats(&mut wl, dim) as u32;
            o[6] = floats(&mut wl, dim) as u32;
            o[7] = floats(&mut wl, kv_dim) as u32;
            o[8] = floats(&mut wl, kv_dim) as u32;
        }
        desc[d::FINAL_NORM] = floats(&mut wl, dim) as u32;
        desc[d::CLS] = mat(&mut wl, vocab, dim) as u32;
        let mut w = dsp.shared(wl.0)?;
        {
            let out = w.as_mut_slice::<u8>();
            let put_f32 = |out: &mut [u8], at: u32, v: &[f32]| {
                for (i, x) in v.iter().enumerate() {
                    out[at as usize + 4 * i..][..4].copy_from_slice(&x.to_le_bytes());
                }
            };
            let put_mat = |out: &mut [u8], at: u32, ws: &[(&Weight, usize)], what: &str| -> Result<()> {
                let mut blocks = Vec::new();
                for (w, _) in ws {
                    blocks.extend_from_slice(q4_0(w, what)?);
                }
                repack_q4_0(&blocks, ws.iter().map(|w| w.1).sum(), &mut out[at as usize..]);
                Ok(())
            };
            for (l, lw) in weights.layers.iter().enumerate() {
                let o = desc[d::LAYER0 + l * d::L_N..][..d::L_N].to_vec();
                put_mat(out, o[0], &[(&lw.wq, dim), (&lw.wk, kv_dim), (&lw.wv, kv_dim)], &format!("layer {l} wq/wk/wv"))?;
                put_mat(out, o[1], &[(&lw.wo, dim)], "wo")?;
                put_mat(out, o[2], &[(&lw.w_gate, hid), (&lw.w_up, hid)], "w_gate/w_up")?;
                put_mat(out, o[3], &[(&lw.w_down, dim)], "w_down")?;
                put_f32(out, o[4], &lw.attn_norm);
                put_f32(out, o[5], &lw.ffn_norm);
                // No bias: zeros.
                for (at, b, n) in [(o[6], &lw.bq, dim), (o[7], &lw.bk, kv_dim), (o[8], &lw.bv, kv_dim)] {
                    match b {
                        Some(b) => put_f32(out, at, b),
                        None => out[at as usize..][..4 * n].fill(0),
                    }
                }
            }
            put_f32(out, desc[d::FINAL_NORM], &weights.final_norm);
            put_mat(out, desc[d::CLS], &[(weights.classifier.as_ref().unwrap_or(&weights.token_embedding), vocab)], "classifier")?;
        }

        // The state buffer.
        let mut sl = Layout::default();
        let kv = nl * c.seq_len * kv_dim;
        for (i, n) in [
            (d::X, 4 * dim),
            (d::XB, 4 * dim),
            (d::XQ, 40 * hid.max(dim) / 32),
            (d::QKV, 4 * (dim + 2 * kv_dim)),
            (d::ATT, 4 * c.n_heads * c.seq_len.div_ceil(32) * 32),
            (d::HB, 4 * 2 * hid),
            (d::LOGITS, 4 * vocab),
            (d::KC, 4 * kv),
            (d::VC, 4 * kv),
            (d::ROT, 4 * c.head_size()),
            (d::PROF, 8 * STAGES.len()),
        ] {
            desc[i] = sl.take(n) as u32;
        }
        let state = dsp.shared(sl.0)?;

        desc[d::DIM] = dim as u32;
        desc[d::HIDDEN] = hid as u32;
        desc[d::LAYERS] = nl as u32;
        desc[d::HEADS] = c.n_heads as u32;
        desc[d::KV_HEADS] = c.n_kv_heads as u32;
        desc[d::VOCAB] = vocab as u32;
        desc[d::SEQ_LEN] = c.seq_len as u32;
        desc[d::HALVES] = (c.rope_layout == RopeLayout::Halves) as u32;
        desc[d::EPS] = c.norm_eps.to_bits();
        desc[d::W_FD] = w.fd();
        desc[d::S_FD] = state.fd();
        desc[d::W_BYTES] = wl.0 as u32;
        desc[d::S_BYTES] = sl.0 as u32;
        dsp.model_load(&desc)?;
        let (x_at, rot_at, logits_at, prof_at) = (desc[d::X] as usize, desc[d::ROT] as usize, desc[d::LOGITS] as usize, desc[d::PROF] as usize);
        Ok(DspModel { config, weights, _w: w, state, x_at, rot_at, logits_at, prof_at, threads, dma: DMA_ALL, next: 0, last_us: 0 })
    }

    /// Runs `token` at position `pos` and returns the logits for the next
    /// token. Positions must be fed in order, as for the CPU path.
    pub fn forward(&mut self, dsp: &Dsp, token: usize, pos: usize) -> Result<&[f32]> {
        let c = &self.config;
        assert!(pos < c.seq_len && token < c.vocab_size);
        let (dim, hs) = (c.dim, c.head_size());
        let s = self.state.as_mut_slice::<f32>();
        self.weights.token_embedding.row_f32(token, &mut s[self.x_at / 4..][..dim]);
        for (i, (sin, cos)) in crate::kernels::scalar::rotations(pos, hs, c.rope_theta).into_iter().enumerate() {
            s[self.rot_at / 4 + 2 * i] = sin;
            s[self.rot_at / 4 + 2 * i + 1] = cos;
        }
        let (next, us) = dsp.model_decode(pos as u32, self.threads, self.dma)?;
        self.next = next as usize;
        self.last_us = us;
        Ok(&self.state.as_slice::<f32>()[self.logits_at / 4..][..self.config.vocab_size])
    }

    /// Microseconds the DSP spent in each of [`STAGES`] since the last
    /// call, summed over the steps; resets them.
    pub fn take_profile(&mut self) -> Vec<f64> {
        let p = &mut self.state.as_mut_slice::<u64>()[self.prof_at / 8..][..STAGES.len()];
        // The qtimer runs at 19.2 MHz.
        let us = p.iter().map(|&t| t as f64 / 19.2).collect();
        p.fill(0);
        us
    }

    /// The most likely next token after the last [`DspModel::forward`],
    /// found on the DSP.
    pub fn greedy(&self) -> usize {
        self.next
    }
}
