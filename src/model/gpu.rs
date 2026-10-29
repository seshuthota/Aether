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
use crate::kernels::vulkan::{Buffer, Commands, Gpu, Kernel, MatQ4_0, Result, groups_for};
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
    wq: MatQ4_0,
    wk: MatQ4_0,
    wv: MatQ4_0,
    wo: MatQ4_0,
    w_gate: MatQ4_0,
    w_up: MatQ4_0,
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
    params: Buffer,
    step: Commands,
    // Everything the recorded commands use must outlive them; held here.
    _keep: Vec<Buffer>,
    _mats: Vec<MatQ4_0>,
    _layers: Vec<Layer>,
    _kernels: Vec<Kernel>,
}

impl GpuModel {
    /// Uploads the matrices and records the decode step.
    pub fn new(gpu: &Gpu, config: Config, weights: Weights) -> Result<GpuModel> {
        let c = &config;
        let (dim, kv_dim, hid, hs) = (c.dim, c.kv_dim(), c.hidden_dim, c.head_size());
        assert!(256 % hs == 0, "the attention shader needs head_size dividing 256");
        let f32buf = |v: &[f32]| -> Result<Buffer> {
            let mut b = gpu.buffer(v.len().max(1) * 4)?;
            b.as_mut_slice::<f32>()[..v.len()].copy_from_slice(v);
            Ok(b)
        };
        let zeros = |n: usize| f32buf(&vec![0.0; n]);
        let mat = |w: &Weight, rows: usize, what: &str| MatQ4_0::new(gpu, q4_0(w, what)?, rows);

        let mut layers = Vec::new();
        for (l, lw) in weights.layers.iter().enumerate() {
            let bias = |b: &Option<crate::aligned::AlignedBuf>, n: usize| match b {
                Some(b) => f32buf(b),
                None => zeros(n),
            };
            layers.push(Layer {
                wq: mat(&lw.wq, dim, &format!("layer {l} wq"))?,
                wk: mat(&lw.wk, kv_dim, "wk")?,
                wv: mat(&lw.wv, kv_dim, "wv")?,
                wo: mat(&lw.wo, dim, "wo")?,
                w_gate: mat(&lw.w_gate, hid, "w_gate")?,
                w_up: mat(&lw.w_up, hid, "w_up")?,
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
        let (xb, xb2, q, k, v) = (zeros(dim)?, zeros(dim)?, zeros(dim)?, zeros(kv_dim)?, zeros(kv_dim)?);
        let (hb, hb2) = (zeros(hid)?, zeros(hid)?);
        let logits = zeros(c.vocab_size)?;
        let cache = c.n_layers * c.seq_len * kv_dim;
        let (kc, vc) = (zeros(cache)?, zeros(cache)?);
        let scores = zeros(c.n_heads * c.seq_len)?;
        // pos, 3 × padding, then (sin, cos) per rotation pair.
        let params = zeros(4 + hs)?;

        let rms = Kernel::new(gpu, crate::spirv!("rmsnorm"), 3, 8)?;
        let post = Kernel::new(gpu, crate::spirv!("qkv_post"), 9, 20)?;
        let attn = Kernel::new(gpu, crate::spirv!("attention"), 6, 20)?;
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
            norm1: crate::kernels::vulkan::Bound,
            q: crate::kernels::vulkan::Bound,
            k: crate::kernels::vulkan::Bound,
            v: crate::kernels::vulkan::Bound,
            post: crate::kernels::vulkan::Bound,
            attn: crate::kernels::vulkan::Bound,
            o: crate::kernels::vulkan::Bound,
            add1: crate::kernels::vulkan::Bound,
            norm2: crate::kernels::vulkan::Bound,
            gate: crate::kernels::vulkan::Bound,
            up: crate::kernels::vulkan::Bound,
            swiglu: crate::kernels::vulkan::Bound,
            down: crate::kernels::vulkan::Bound,
            add2: crate::kernels::vulkan::Bound,
        }
        let gemv_bind = |m: &MatQ4_0, input: &Buffer, out: &Buffer| gemv_k(m.rows).bind(gpu, &[&m.qs, &m.d, input, out]);
        let mut bound = Vec::new();
        for ly in &layers {
            bound.push(Bl {
                norm1: rms.bind(gpu, &[&x, &ly.attn_norm, &xb])?,
                q: gemv_bind(&ly.wq, &xb, &q)?,
                k: gemv_bind(&ly.wk, &xb, &k)?,
                v: gemv_bind(&ly.wv, &xb, &v)?,
                post: post.bind(gpu, &[&q, &k, &v, &ly.bq, &ly.bk, &ly.bv, &kc, &vc, &params])?,
                attn: attn.bind(gpu, &[&q, &kc, &vc, &scores, &xb2, &params])?,
                o: gemv_bind(&ly.wo, &xb2, &xb)?,
                add1: add.bind(gpu, &[&x, &xb])?,
                norm2: rms.bind(gpu, &[&x, &ly.ffn_norm, &xb])?,
                gate: gemv_bind(&ly.w_gate, &xb, &hb)?,
                up: gemv_bind(&ly.w_up, &xb, &hb2)?,
                swiglu: swiglu.bind(gpu, &[&hb, &hb2])?,
                down: gemv_bind(&ly.w_down, &hb, &xb)?,
                add2: add.bind(gpu, &[&x, &xb])?,
            });
        }
        let fnorm = rms.bind(gpu, &[&x, &final_norm, &xb])?;
        let fcls = gemv_bind(&cls, &xb, &logits)?;
        let gemv_push = |m: &MatQ4_0| push(&[m.rows as u32, m.nb as u32]);
        let gemv_groups = |m: &MatQ4_0| groups_for(m.rows, 4 * gemv_r(m.rows) as usize);

        let step = gpu.record(|r| {
            for (l, (b, ly)) in bound.iter().zip(&layers).enumerate() {
                let layer_off = (l * c.seq_len * kv_dim) as u32;
                r.dispatch(&b.norm1, &push(&[dim as u32, eps]), one);
                r.barrier();
                for (bd, m) in [(&b.q, &ly.wq), (&b.k, &ly.wk), (&b.v, &ly.wv)] {
                    r.dispatch(bd, &gemv_push(m), gemv_groups(m));
                }
                r.barrier();
                r.dispatch(&b.post, &push(&[dim as u32, kv_dim as u32, hs as u32, halves, layer_off]), one);
                r.barrier();
                r.dispatch(&b.attn, &push(&[c.n_heads as u32, c.n_kv_heads as u32, hs as u32, c.seq_len as u32, layer_off]), [c.n_heads as u32, 1, 1]);
                r.barrier();
                r.dispatch(&b.o, &gemv_push(&ly.wo), gemv_groups(&ly.wo));
                r.barrier();
                r.dispatch(&b.add1, &push(&[dim as u32]), elems(dim));
                r.barrier();
                r.dispatch(&b.norm2, &push(&[dim as u32, eps]), one);
                r.barrier();
                r.dispatch(&b.gate, &gemv_push(&ly.w_gate), gemv_groups(&ly.w_gate));
                r.dispatch(&b.up, &gemv_push(&ly.w_up), gemv_groups(&ly.w_up));
                r.barrier();
                r.dispatch(&b.swiglu, &push(&[hid as u32]), elems(hid));
                r.barrier();
                r.dispatch(&b.down, &gemv_push(&ly.w_down), gemv_groups(&ly.w_down));
                r.barrier();
                r.dispatch(&b.add2, &push(&[dim as u32]), elems(dim));
                r.barrier();
            }
            r.dispatch(&fnorm, &push(&[dim as u32, eps]), one);
            r.barrier();
            r.dispatch(&fcls, &gemv_push(&cls), gemv_groups(&cls));
        })?;

        let mut keep = vec![xb, xb2, q, k, v, hb, hb2, kc, vc, scores, final_norm];
        keep.shrink_to_fit();
        let mut kernels = vec![rms, post, attn, add, swiglu];
        kernels.extend(gemvs);
        Ok(GpuModel { config, weights, x, logits, params, step, _keep: keep, _mats: vec![cls], _layers: layers, _kernels: kernels })
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
        gpu.submit(&self.step)?;
        Ok(&self.logits.as_slice::<f32>()[..c.vocab_size])
    }
}
