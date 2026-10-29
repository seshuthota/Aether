//! A whole decode step on the GPU (`--features vulkan`; IMPLEMENTATION.md
//! L.3): the transformer of [`super::Transformer::forward`] as Vulkan
//! compute shaders, for models whose matrices are all Q4_0 (a `--pure`
//! Q4_0 GGUF).
//!
//! Every buffer lives in memory the CPU and GPU share. The whole token,
//! about a dozen dispatches per layer with a barrier after each step, is
//! recorded once into one command buffer. What changes from token to token
//! is in the `params` buffer the shaders read: the position, and the RoPE
//! rotation table, computed on the CPU exactly as the CPU path does. A
//! step is then: dequantize the token's embedding row on the CPU straight
//! into the input buffer, write `params`, submit, wait, read the logits.

use super::transformer::Weights;
use super::{Config, RopeLayout};
use crate::kernels::vulkan::{Bound, Buffer, Commands, Gpu, Kernel, MatQ4_0, Recorder, Result, groups_for};
use crate::quant::BlockQ4_0;
use crate::weight::Weight;

/// Rows per 32 lanes for the GEMV shader by the matrix's rows (measured,
/// results/phase-l3.md step 2).
fn gemv_r(rows: usize) -> u32 {
    match rows {
        0..4096 => 4,
        4096..65536 => 8,
        _ => 16,
    }
}

fn q4_0<'a>(w: &'a Weight, what: &str) -> Result<&'a [BlockQ4_0]> {
    match w {
        Weight::Q4_0(b) => Ok(b),
        _ => Err(format!("{what} is {}, the GPU path needs Q4_0 (a --pure Q4_0 GGUF)", w.format_name())),
    }
}

fn push(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Layer {
    /// wq, wk and wv stacked (dim + 2·kv_dim rows), and w_gate over w_up
    /// (2·hidden_dim rows): one dispatch each instead of three and two.
    /// Each dispatch costs ~10 µs beyond its bytes, and the stacked
    /// matrices reach 58 against 51 GB/s (gate, up) and more for the small
    /// q/k/v (results/phase-l3.md).
    wqkv: MatQ4_0,
    wo: MatQ4_0,
    w_gate_up: MatQ4_0,
    w_down: MatQ4_0,
    attn_norm: Buffer,
    ffn_norm: Buffer,
    bq: Buffer,
    bk: Buffer,
    bv: Buffer,
}

pub struct GpuModel {
    pub config: Config,
    /// The embedding stays on the CPU side too: one row is dequantized per
    /// token.
    weights: Weights,
    x: Buffer,
    logits: Buffer,
    kc: Buffer,
    vc: Buffer,
    params: Buffer,
    step: Commands,
    /// With profiling: the stage each timestamp ends ([`GpuModel::profile`]).
    stages: Vec<&'static str>,
    /// Poll the fence instead of sleeping on it ([`Gpu::submit`]). Off by
    /// default: in a sustained run polling was slower (86 vs 103 tok/s) and
    /// cost more power (results/phase-l3.md step 5).
    pub spin: bool,
    // Everything the recorded commands use must outlive them; held here.
    _keep: Vec<Buffer>,
    _mats: Vec<MatQ4_0>,
    _layers: Vec<Layer>,
    _kernels: Vec<Kernel>,
}

impl GpuModel {
    /// Uploads the matrices and records the decode step. With `profile`,
    /// the step writes a GPU timestamp after every stage
    /// ([`GpuModel::profile`]).
    pub fn new(gpu: &Gpu, config: Config, weights: Weights, profile: bool) -> Result<GpuModel> {
        GpuModel::with_repeat(gpu, config, weights, profile, 1)
    }

    /// [`GpuModel::new`] whose step runs the whole forward pass `repeat`
    /// times in one submit (at the same position, so only the timing is
    /// meaningful): the GPU then runs without the CPU's gap between tokens.
    pub fn with_repeat(gpu: &Gpu, config: Config, weights: Weights, profile: bool, repeat: usize) -> Result<GpuModel> {
        let c = &config;
        let (dim, kv_dim, hid, hs) = (c.dim, c.kv_dim(), c.hidden_dim, c.head_size());
        assert!(hs <= 64, "the attention shaders need head_size <= 64");
        // Attention works in chunks of 64 positions (shaders/attention.comp).
        let chunks = c.seq_len.div_ceil(64);
        let f32buf = |v: &[f32]| -> Result<Buffer> {
            let mut b = gpu.buffer(v.len().max(1) * 4)?;
            b.as_mut_slice::<f32>()[..v.len()].copy_from_slice(v);
            Ok(b)
        };
        let zeros = |n: usize| f32buf(&vec![0.0; n]);
        let mat = |w: &Weight, rows: usize, what: &str| MatQ4_0::new(gpu, q4_0(w, what)?, rows);
        // Matrices of the same width, one under the other.
        let stacked = |ws: &[(&Weight, usize)], what: &str| -> Result<MatQ4_0> {
            let mut blocks = Vec::new();
            for (w, _) in ws {
                blocks.extend_from_slice(q4_0(w, what)?);
            }
            MatQ4_0::new(gpu, &blocks, ws.iter().map(|w| w.1).sum())
        };

        let mut layers = Vec::new();
        for (l, lw) in weights.layers.iter().enumerate() {
            let bias = |b: &Option<crate::aligned::AlignedBuf>, n: usize| match b {
                Some(b) => f32buf(b),
                None => zeros(n),
            };
            layers.push(Layer {
                wqkv: stacked(&[(&lw.wq, dim), (&lw.wk, kv_dim), (&lw.wv, kv_dim)], &format!("layer {l} wq/wk/wv"))?,
                wo: mat(&lw.wo, dim, "wo")?,
                w_gate_up: stacked(&[(&lw.w_gate, hid), (&lw.w_up, hid)], "w_gate/w_up")?,
                w_down: mat(&lw.w_down, dim, "w_down")?,
                attn_norm: f32buf(&lw.attn_norm)?,
                ffn_norm: f32buf(&lw.ffn_norm)?,
                bq: bias(&lw.bq, dim)?,
                bk: bias(&lw.bk, kv_dim)?,
                bv: bias(&lw.bv, kv_dim)?,
            });
        }
        let cls = mat(weights.classifier.as_ref().unwrap_or(&weights.token_embedding), c.vocab_size, "classifier")?;
        let final_norm = f32buf(&weights.final_norm)?;

        let x = zeros(dim)?;
        // q, k, v: one buffer, the output of wqkv; hb (gate) then hb2 (up)
        // likewise. The kernels after them bind views into it.
        let (xb, xb2, qkv, hb12) = (zeros(dim)?, zeros(dim)?, zeros(dim + 2 * kv_dim)?, zeros(2 * hid)?);
        let (k_at, v_at, up_at) = (4 * dim, 4 * (dim + kv_dim), 4 * hid);
        let logits = zeros(c.vocab_size)?;
        let cache = c.n_layers * c.seq_len * kv_dim;
        let (kc, vc) = (zeros(cache)?, zeros(cache)?);
        let parts = zeros(c.n_heads * chunks * (hs + 2))?;
        // pos, 3 × padding, then (sin, cos) per rotation pair.
        let params = zeros(4 + hs)?;

        let rms = Kernel::new(gpu, crate::spirv!("rmsnorm"), 3, 8)?;
        let post = Kernel::new(gpu, crate::spirv!("qkv_post"), 9, 20)?;
        let attn = Kernel::new(gpu, crate::spirv!("attention"), 5, 20)?;
        let merge = Kernel::new(gpu, crate::spirv!("attention_merge"), 3, 8)?;
        let add = Kernel::new(gpu, crate::spirv!("add"), 2, 4)?;
        let swiglu = Kernel::new(gpu, crate::spirv!("swiglu"), 2, 4)?;
        let gemvs = [4, 8, 16].map(|r| gpu.q4_0_gemv_kernel(r));
        let gemvs: Vec<Kernel> = gemvs.into_iter().collect::<Result<_>>()?;
        let gemv_k = |rows: usize| &gemvs[match gemv_r(rows) {
            4 => 0,
            8 => 1,
            _ => 2,
        }];

        let eps = c.norm_eps.to_bits();
        let halves = u32::from(c.rope_layout == RopeLayout::Halves);
        let one = [1, 1, 1];
        let elems = |n: usize| [n.div_ceil(256) as u32, 1, 1];
        // Bind everything first (binding can fail; recording cannot).
        struct Bl {
            norm1: Bound,
            qkv: Bound,
            post: Bound,
            attn: Bound,
            merge: Bound,
            o: Bound,
            add1: Bound,
            norm2: Bound,
            gate_up: Bound,
            swiglu: Bound,
            down: Bound,
            add2: Bound,
        }
        let gemv_bind = |m: &MatQ4_0, input: &Buffer, out: &Buffer| gemv_k(m.rows).bind(gpu, &[&m.qs, &m.d, input, out]);
        let mut bound = Vec::new();
        for ly in &layers {
            bound.push(Bl {
                norm1: rms.bind(gpu, &[&x, &ly.attn_norm, &xb])?,
                qkv: gemv_bind(&ly.wqkv, &xb, &qkv)?,
                post: post.bind_at(gpu, &[(&qkv, 0), (&qkv, k_at), (&qkv, v_at), (&ly.bq, 0), (&ly.bk, 0), (&ly.bv, 0), (&kc, 0), (&vc, 0), (&params, 0)])?,
                attn: attn.bind(gpu, &[&qkv, &kc, &vc, &parts, &params])?,
                merge: merge.bind(gpu, &[&parts, &xb2, &params])?,
                o: gemv_bind(&ly.wo, &xb2, &xb)?,
                add1: add.bind(gpu, &[&x, &xb])?,
                norm2: rms.bind(gpu, &[&x, &ly.ffn_norm, &xb])?,
                gate_up: gemv_bind(&ly.w_gate_up, &xb, &hb12)?,
                swiglu: swiglu.bind_at(gpu, &[(&hb12, 0), (&hb12, up_at)])?,
                down: gemv_bind(&ly.w_down, &hb12, &xb)?,
                add2: add.bind(gpu, &[&x, &xb])?,
            });
        }
        let fnorm = rms.bind(gpu, &[&x, &final_norm, &xb])?;
        let fcls = gemv_bind(&cls, &xb, &logits)?;
        let gemv_push = |m: &MatQ4_0| push(&[m.rows as u32, m.nb as u32]);
        let gemv_groups = |m: &MatQ4_0| groups_for(m.rows, 4 * gemv_r(m.rows) as usize);

        // Each stage ends with a barrier, and with `profile` a timestamp.
        let mut stages = Vec::new();
        let rec = |r: &mut Recorder, stages: &mut Vec<&'static str>| {
            let mut end = |r: &mut Recorder, name: &'static str| {
                r.barrier();
                r.stamp();
                stages.push(name);
            };
            for (l, (b, ly)) in bound.iter().zip(&layers).enumerate() {
                let layer_off = (l * c.seq_len * kv_dim) as u32;
                r.dispatch(&b.norm1, &push(&[dim as u32, eps]), one);
                end(r, "rmsnorm");
                r.dispatch(&b.qkv, &gemv_push(&ly.wqkv), gemv_groups(&ly.wqkv));
                end(r, "q/k/v");
                r.dispatch(&b.post, &push(&[dim as u32, kv_dim as u32, hs as u32, halves, layer_off]), one);
                end(r, "bias + RoPE + cache");
                r.dispatch(&b.attn, &push(&[c.n_heads as u32, c.n_kv_heads as u32, hs as u32, chunks as u32, layer_off]), [c.n_heads as u32, chunks as u32, 1]);
                end(r, "attention");
                r.dispatch(&b.merge, &push(&[hs as u32, chunks as u32]), [c.n_heads as u32, 1, 1]);
                end(r, "attention merge");
                r.dispatch(&b.o, &gemv_push(&ly.wo), gemv_groups(&ly.wo));
                end(r, "wo");
                r.dispatch(&b.add1, &push(&[dim as u32]), elems(dim));
                end(r, "add");
                r.dispatch(&b.norm2, &push(&[dim as u32, eps]), one);
                end(r, "rmsnorm");
                r.dispatch(&b.gate_up, &gemv_push(&ly.w_gate_up), gemv_groups(&ly.w_gate_up));
                end(r, "gate/up");
                r.dispatch(&b.swiglu, &push(&[hid as u32]), elems(hid));
                end(r, "swiglu");
                r.dispatch(&b.down, &gemv_push(&ly.w_down), gemv_groups(&ly.w_down));
                end(r, "down");
                r.dispatch(&b.add2, &push(&[dim as u32]), elems(dim));
                end(r, "add");
            }
            r.dispatch(&fnorm, &push(&[dim as u32, eps]), one);
            end(r, "rmsnorm");
            r.dispatch(&fcls, &gemv_push(&cls), gemv_groups(&cls));
            end(r, "classifier");
        };
        let rec_all = |r: &mut Recorder, stages: &mut Vec<&'static str>| (0..repeat).for_each(|_| rec(r, stages));
        let step = if profile { gpu.record_timed(|r| rec_all(r, &mut stages))? } else { gpu.record(|r| rec_all(r, &mut stages))? };
        if !profile {
            stages.clear();
        }

        let mut keep = vec![xb, xb2, qkv, hb12, parts, final_norm];
        keep.shrink_to_fit();
        let mut kernels = vec![rms, post, attn, merge, add, swiglu];
        kernels.extend(gemvs);
        Ok(GpuModel { config, weights, x, logits, kc, vc, params, step, stages, spin: false, _keep: keep, _mats: vec![cls], _layers: layers, _kernels: kernels })
    }

    /// Runs `token` at position `pos` and returns the logits for the next
    /// token. As for the CPU path, positions must be fed in order.
    pub fn forward(&mut self, gpu: &Gpu, token: usize, pos: usize) -> Result<&[f32]> {
        let c = &self.config;
        assert!(pos < c.seq_len && token < c.vocab_size);
        self.weights.token_embedding.row_f32(token, &mut self.x.as_mut_slice::<f32>()[..c.dim]);
        let p = self.params.as_mut_slice::<u32>();
        p[0] = pos as u32;
        for (i, (sin, cos)) in crate::kernels::scalar::rotations(pos, c.head_size(), c.rope_theta).into_iter().enumerate() {
            p[4 + 2 * i] = sin.to_bits();
            p[5 + 2 * i] = cos.to_bits();
        }
        gpu.submit(&self.step, self.spin)?;
        Ok(&self.logits.as_slice::<f32>()[..c.vocab_size])
    }

    /// Copies positions `0..n` of a CPU [`super::Transformer`]'s KV cache
    /// (same layout: `[layer, pos, kv_dim]`, `seq_len` positions per layer)
    /// into the GPU's, so decoding can continue on the GPU at position `n`
    /// after a prefill on the CPU.
    pub fn load_kv(&mut self, keys: &[f32], values: &[f32], seq_len: usize, n: usize) {
        let c = &self.config;
        let kv = c.kv_dim();
        assert!(n <= seq_len.min(c.seq_len) && keys.len() == c.n_layers * seq_len * kv && values.len() == keys.len());
        for (src, dst) in [(keys, &mut self.kc), (values, &mut self.vc)] {
            let dst = dst.as_mut_slice::<f32>();
            for l in 0..c.n_layers {
                let (s0, d0) = (l * seq_len * kv, l * c.seq_len * kv);
                dst[d0..d0 + n * kv].copy_from_slice(&src[s0..s0 + n * kv]);
            }
        }
    }

    /// GPU seconds of the last [`GpuModel::forward`] per kind of stage
    /// (summed over layers), in order of first appearance; empty unless
    /// the model was made with `profile`.
    pub fn profile(&self, gpu: &Gpu) -> Result<Vec<(&'static str, f64)>> {
        let t = gpu.stamps(&self.step)?;
        let mut out: Vec<(&'static str, f64)> = Vec::new();
        for (i, name) in self.stages.iter().enumerate() {
            let dt = t[i + 1] - t[i];
            match out.iter_mut().find(|(n, _)| n == name) {
                Some(e) => e.1 += dt,
                None => out.push((name, dt)),
            }
        }
        Ok(out)
    }
}
