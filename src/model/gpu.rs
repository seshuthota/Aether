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
//!
//! A prompt goes through [`GpuModel::forward_batch`]: the same stages for
//! up to [`GpuModel::batch`] tokens at once, the matrix products as tiled
//! matrix-matrix products (`q4_0_gemm.comp`) that read each weight once
//! per 64 tokens. The activation buffers hold `batch` rows; decoding uses
//! the first.

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

/// Bytes the batched attention's partial results may take: the batch is
/// as many tokens as fit (at most 128).
const PARTS_BUDGET: usize = 32 << 20;

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

/// One layer's kernels with their buffers bound: the matrix products both
/// as GEMV (decode) and GEMM (prompt).
struct Bl {
    norm1: Bound,
    qkv: Bound,
    post: Bound,
    attn: Bound,
    merge: Bound,
    o: Bound,
    norm2: Bound,
    gate_up: Bound,
    down: Bound,
    qkv_mm: Bound,
    o_mm: Bound,
    gate_up_mm: Bound,
    down_mm: Bound,
}

/// What [`GpuModel::with`] builds.
#[derive(Clone, Copy, Default)]
pub struct Options {
    /// A GPU timestamp after every stage of the decode step
    /// ([`GpuModel::profile`]).
    pub profile: bool,
    /// Record the decode step this many times in one submit (timing only;
    /// 0 counts as 1).
    pub repeat: usize,
    /// [`GpuModel::forward_batch`] returns the logits of every token, not
    /// only the last (perplexity), at `batch × vocab` floats of memory.
    pub all_logits: bool,
}

pub struct GpuModel {
    pub config: Config,
    /// Tokens per [`GpuModel::forward_batch`].
    pub batch: usize,
    /// The embedding stays on the CPU side too: one row is dequantized per
    /// token.
    weights: Weights,
    x: Buffer,
    logits: Buffer,
    /// `batch × vocab` logits with [`Options::all_logits`].
    logits_all: Option<Buffer>,
    kc: Buffer,
    vc: Buffer,
    params: Buffer,
    step: Commands,
    /// With profiling: the stage each timestamp ends ([`GpuModel::profile`]).
    stages: Vec<&'static str>,
    profile: bool,
    /// With profiling: GPU seconds per stage of the last
    /// [`GpuModel::forward_batch`].
    pub batch_profile: Vec<(&'static str, f64)>,
    /// Poll the fence instead of sleeping on it ([`Gpu::submit`]). Off by
    /// default: in a sustained run polling was slower (86 vs 103 tok/s) and
    /// cost more power (results/phase-l3.md step 5).
    pub spin: bool,
    bound: Vec<Bl>,
    fnorm: Bound,
    /// The classifier in decode (the final norm fused in), and after a
    /// batch (reading the last token's row, normalized by `fnorm`).
    fcls: Bound,
    fcls_b: Bound,
    fcls_mm: Option<Bound>,
    // Everything the recorded commands use must outlive them; held here.
    _keep: Vec<Buffer>,
    cls: MatQ4_0,
    layers: Vec<Layer>,
    _kernels: Vec<Kernel>,
}

impl GpuModel {
    /// Uploads the matrices and records the decode step. With `profile`,
    /// the step writes a GPU timestamp after every stage
    /// ([`GpuModel::profile`]).
    pub fn new(gpu: &Gpu, config: Config, weights: Weights, profile: bool) -> Result<GpuModel> {
        GpuModel::with(gpu, config, weights, Options { profile, ..Options::default() })
    }

    /// [`GpuModel::new`] whose step runs the whole forward pass `repeat`
    /// times in one submit (at the same position, so only the timing is
    /// meaningful): the GPU then runs without the CPU's gap between tokens.
    pub fn with_repeat(gpu: &Gpu, config: Config, weights: Weights, profile: bool, repeat: usize) -> Result<GpuModel> {
        GpuModel::with(gpu, config, weights, Options { profile, repeat, ..Options::default() })
    }

    pub fn with(gpu: &Gpu, config: Config, weights: Weights, opt: Options) -> Result<GpuModel> {
        let c = &config;
        let (dim, kv_dim, hid, hs) = (c.dim, c.kv_dim(), c.hidden_dim, c.head_size());
        assert!(hs <= 64, "the attention shaders need head_size <= 64");
        // Attention works in chunks of 64 positions (shaders/attention.comp).
        let chunks = c.seq_len.div_ceil(64);
        let parts_per_token = c.n_heads * chunks * (hs + 2);
        let batch = (PARTS_BUDGET / (4 * parts_per_token)).clamp(1, 128);
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

        // One row per token of a batch; decoding uses row 0.
        let x = zeros(batch * dim)?;
        // q, k, v: one buffer, the output of wqkv, which the kernels after
        // it bind views into; gate then up likewise, read whole by down.
        let qkv_dim = dim + 2 * kv_dim;
        let (xb, xb2, qkv, hb12) = (zeros(batch * dim)?, zeros(batch * dim)?, zeros(batch * qkv_dim)?, zeros(batch * 2 * hid)?);
        let (k_at, v_at) = (4 * dim, 4 * (dim + kv_dim));
        // The CPU reads all the logits after every step: from memory it
        // caches (coherent, so no invalidation), not uncached memory, when
        // the device offers that for storage buffers (Adreno: type 6).
        let cached = |n: usize| {
            use ash::vk::MemoryPropertyFlags as M;
            match gpu.buffer_with(n * 4, M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_COHERENT | M::HOST_CACHED) {
                Ok(b) => Ok(b),
                Err(_) => zeros(n),
            }
        };
        let logits = cached(c.vocab_size)?;
        let logits_all = if opt.all_logits { Some(cached(batch * c.vocab_size)?) } else { None };
        let cache = c.n_layers * c.seq_len * kv_dim;
        let (kc, vc) = (zeros(cache)?, zeros(cache)?);
        let parts = zeros(batch * parts_per_token)?;
        // pos, 3 × padding, then (sin, cos) per rotation pair, per token.
        let params = zeros(4 + batch * hs)?;

        let rms = Kernel::new(gpu, crate::spirv!("rmsnorm"), 3, 12)?;
        let post = Kernel::new(gpu, crate::spirv!("qkv_post"), 9, 24)?;
        let attn = Kernel::new(gpu, crate::spirv!("attention"), 5, 24)?;
        let merge = Kernel::new(gpu, crate::spirv!("attention_merge"), 3, 8)?;
        // In decode, wo adds its product to the residual stream x; down does
        // too, and reads gate and up, applying the SwiGLU as it loads them;
        // q/k/v, gate/up and the classifier apply the RMS norm before them
        // as they load x.
        let wo_k = gpu.q4_0_gemv_kernel_fused(gemv_r(dim), true, false, false)?;
        let down_k = gpu.q4_0_gemv_kernel_fused(gemv_r(dim), true, true, false)?;
        let rs = [4, 8, 16];
        let by_r = |r: u32| rs.iter().position(|&x| x == r).unwrap();
        let gemvs: Vec<Kernel> = rs.map(|r| gpu.q4_0_gemv_kernel(r)).into_iter().collect::<Result<_>>()?;
        let normed: Vec<Kernel> = rs.map(|r| gpu.q4_0_gemv_kernel_fused(r, false, false, true)).into_iter().collect::<Result<_>>()?;
        let (mm, mm_acc, mm_glu) = (gpu.q4_0_gemm_kernel()?, gpu.q4_0_gemm_kernel_fused(true, false)?, gpu.q4_0_gemm_kernel_fused(true, true)?);

        // Bind everything first (binding can fail; recording cannot).
        // The GEMV's binding 4 is the norm's weights, unused (any buffer)
        // without the norm.
        let normed_bind = |m: &MatQ4_0, out: &Buffer, w: &Buffer| normed[by_r(gemv_r(m.rows))].bind(gpu, &[&m.qs, &m.d, &x, out, w]);
        let mut bound = Vec::new();
        for ly in &layers {
            bound.push(Bl {
                norm1: rms.bind(gpu, &[&x, &ly.attn_norm, &xb])?,
                qkv: normed_bind(&ly.wqkv, &qkv, &ly.attn_norm)?,
                post: post.bind_at(gpu, &[(&qkv, 0), (&qkv, k_at), (&qkv, v_at), (&ly.bq, 0), (&ly.bk, 0), (&ly.bv, 0), (&kc, 0), (&vc, 0), (&params, 0)])?,
                attn: attn.bind(gpu, &[&qkv, &kc, &vc, &parts, &params])?,
                merge: merge.bind(gpu, &[&parts, &xb2, &params])?,
                o: wo_k.bind(gpu, &[&ly.wo.qs, &ly.wo.d, &xb2, &x, &x])?,
                norm2: rms.bind(gpu, &[&x, &ly.ffn_norm, &xb])?,
                gate_up: normed_bind(&ly.w_gate_up, &hb12, &ly.ffn_norm)?,
                down: down_k.bind(gpu, &[&ly.w_down.qs, &ly.w_down.d, &hb12, &x, &x])?,
                qkv_mm: mm.bind(gpu, &[&ly.wqkv.qs, &ly.wqkv.d, &xb, &qkv])?,
                o_mm: mm_acc.bind(gpu, &[&ly.wo.qs, &ly.wo.d, &xb2, &x])?,
                gate_up_mm: mm.bind(gpu, &[&ly.w_gate_up.qs, &ly.w_gate_up.d, &xb, &hb12])?,
                down_mm: mm_glu.bind(gpu, &[&ly.w_down.qs, &ly.w_down.d, &hb12, &x])?,
            });
        }
        let fnorm = rms.bind(gpu, &[&x, &final_norm, &xb])?;
        let fcls = normed_bind(&cls, &logits, &final_norm)?;
        let fcls_b = gemvs[by_r(gemv_r(cls.rows))].bind(gpu, &[&cls.qs, &cls.d, &xb, &logits, &xb])?;
        let fcls_mm = match &logits_all {
            Some(l) => Some(mm.bind(gpu, &[&cls.qs, &cls.d, &xb, l])?),
            None => None,
        };

        let mut keep = vec![xb, xb2, qkv, hb12, parts, final_norm];
        keep.shrink_to_fit();
        let mut kernels = vec![rms, post, attn, merge, wo_k, down_k, mm, mm_acc, mm_glu];
        kernels.extend(gemvs);
        kernels.extend(normed);
        let mut m = GpuModel {
            config,
            batch,
            weights,
            x,
            logits,
            logits_all,
            kc,
            vc,
            params,
            // Replaced just below: recording needs the bound kernels in place.
            step: gpu.record(|_| {})?,
            stages: Vec::new(),
            profile: opt.profile,
            batch_profile: Vec::new(),
            spin: false,
            bound,
            fnorm,
            fcls,
            fcls_b,
            fcls_mm,
            _keep: keep,
            cls,
            layers,
            _kernels: kernels,
        };
        let mut stages = Vec::new();
        let rec_all = |r: &mut Recorder, stages: &mut Vec<&'static str>| (0..opt.repeat.max(1)).for_each(|_| m.record_pass(r, 1, 0, None, stages));
        let step = if opt.profile { gpu.record_timed(|r| rec_all(r, &mut stages))? } else { gpu.record(|r| rec_all(r, &mut stages))? };
        if !opt.profile {
            stages.clear();
        }
        m.step = step;
        m.stages = stages;
        Ok(m)
    }

    /// Records one forward pass over `n` tokens, the first at position
    /// `pos0` (with `n` = 1: the decode step; the position comes from
    /// `params` either way, `pos0` only sizes the attention). The last
    /// token's logits go to `logits`, or with `all` every token's to that
    /// buffer's binding. Each stage ends with a barrier and a timestamp
    /// (which only a timed recording writes).
    fn record_pass(&self, r: &mut Recorder, n: usize, pos0: usize, all: Option<&Bound>, stages: &mut Vec<&'static str>) {
        let c = &self.config;
        let (dim, kv_dim, hs) = (c.dim, c.kv_dim(), c.head_size());
        let eps = c.norm_eps.to_bits();
        let halves = u32::from(c.rope_layout == RopeLayout::Halves);
        let qkv_dim = (dim + 2 * kv_dim) as u32;
        let nt = n as u32;
        // Decode covers every chunk the cache can hold (it is recorded
        // once); a batch only those its last token reaches.
        let chunks = if n == 1 { c.seq_len.div_ceil(64) } else { (pos0 + n).div_ceil(64) } as u32;
        let mut end = |r: &mut Recorder, name: &'static str| {
            r.barrier();
            r.stamp();
            stages.push(name);
        };
        // A matrix product: GEMV for one token, GEMM for several.
        let gemv = |r: &mut Recorder, k: &Bound, m: &MatQ4_0| r.dispatch(k, &push(&[m.rows as u32, m.nb as u32, eps]), groups_for(m.rows, 4 * gemv_r(m.rows) as usize));
        let product = |r: &mut Recorder, gemv_b: &Bound, gemm: &Bound, m: &MatQ4_0| {
            if n == 1 {
                gemv(r, gemv_b, m);
            } else {
                r.dispatch(gemm, &push(&[m.rows as u32, m.nb as u32, nt]), [m.rows.div_ceil(64) as u32, n.div_ceil(64) as u32, 1]);
            }
        };
        for (l, (b, ly)) in self.bound.iter().zip(&self.layers).enumerate() {
            let layer_off = (l * c.seq_len * kv_dim) as u32;
            // In decode the norms are part of the products after them.
            if n > 1 {
                r.dispatch(&b.norm1, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
                end(r, "rmsnorm");
            }
            product(r, &b.qkv, &b.qkv_mm, &ly.wqkv);
            end(r, "q/k/v");
            r.dispatch(&b.post, &push(&[dim as u32, kv_dim as u32, hs as u32, halves, layer_off, qkv_dim]), [nt, 1, 1]);
            end(r, "bias + RoPE + cache");
            r.dispatch(&b.attn, &push(&[c.n_heads as u32, c.n_kv_heads as u32, hs as u32, chunks, layer_off, qkv_dim]), [c.n_heads as u32, chunks, nt]);
            end(r, "attention");
            r.dispatch(&b.merge, &push(&[hs as u32, chunks]), [c.n_heads as u32, nt, 1]);
            end(r, "attention merge");
            product(r, &b.o, &b.o_mm, &ly.wo);
            end(r, "wo + add");
            if n > 1 {
                r.dispatch(&b.norm2, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
                end(r, "rmsnorm");
            }
            product(r, &b.gate_up, &b.gate_up_mm, &ly.w_gate_up);
            end(r, "gate/up");
            product(r, &b.down, &b.down_mm, &ly.w_down);
            end(r, "swiglu + down + add");
        }
        match all {
            Some(cls_mm) => {
                r.dispatch(&self.fnorm, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
                end(r, "rmsnorm");
                let m = &self.cls;
                r.dispatch(cls_mm, &push(&[m.rows as u32, m.nb as u32, nt]), [m.rows.div_ceil(64) as u32, n.div_ceil(64) as u32, 1]);
            }
            None if n == 1 => gemv(r, &self.fcls, &self.cls),
            None => {
                // Only the last token: its row of x, normalized into row 0.
                r.dispatch(&self.fnorm, &push(&[dim as u32, eps, nt - 1]), [1, 1, 1]);
                end(r, "rmsnorm");
                gemv(r, &self.fcls_b, &self.cls);
            }
        }
        end(r, "classifier");
    }

    /// Writes token `t`'s embedding into row `row` of x and its position's
    /// rotation table into `params`.
    fn set_token(&mut self, row: usize, token: usize, pos: usize) {
        let c = &self.config;
        assert!(pos < c.seq_len && token < c.vocab_size);
        let dim = c.dim;
        self.weights.token_embedding.row_f32(token, &mut self.x.as_mut_slice::<f32>()[row * dim..(row + 1) * dim]);
        let hs = c.head_size();
        let p = self.params.as_mut_slice::<u32>();
        for (i, (sin, cos)) in crate::kernels::scalar::rotations(pos, hs, c.rope_theta).into_iter().enumerate() {
            p[4 + row * hs + 2 * i] = sin.to_bits();
            p[5 + row * hs + 2 * i] = cos.to_bits();
        }
    }

    /// Runs `token` at position `pos` and returns the logits for the next
    /// token. As for the CPU path, positions must be fed in order.
    pub fn forward(&mut self, gpu: &Gpu, token: usize, pos: usize) -> Result<&[f32]> {
        self.set_token(0, token, pos);
        self.params.as_mut_slice::<u32>()[0] = pos as u32;
        gpu.submit(&self.step, self.spin)?;
        Ok(&self.logits.as_slice::<f32>()[..self.config.vocab_size])
    }

    /// Runs `tokens` (at most [`GpuModel::batch`]) at positions `pos ..`
    /// in one pass, as the CPU's `forward_batch` does, and returns the
    /// last token's logits, or with [`Options::all_logits`] every token's
    /// (`tokens.len() × vocab`). The pass is recorded for this call: its
    /// size depends on the number of tokens and the position.
    pub fn forward_batch(&mut self, gpu: &Gpu, tokens: &[usize], pos: usize) -> Result<&[f32]> {
        let n = tokens.len();
        assert!(n > 0 && n <= self.batch && pos + n <= self.config.seq_len, "{n} tokens at {pos}: batch {}, seq_len {}", self.batch, self.config.seq_len);
        for (i, &t) in tokens.iter().enumerate() {
            self.set_token(i, t, pos + i);
        }
        self.params.as_mut_slice::<u32>()[0] = pos as u32;
        let mut stages = Vec::new();
        let rec = |r: &mut Recorder| self.record_pass(r, n, pos, self.fcls_mm.as_ref(), &mut stages);
        let cmds = if self.profile { gpu.record_timed(rec)? } else { gpu.record(rec)? };
        gpu.submit(&cmds, self.spin)?;
        if self.profile {
            self.batch_profile = per_stage(&stages, &gpu.stamps(&cmds)?);
        }
        let v = self.config.vocab_size;
        Ok(match &self.logits_all {
            Some(l) => &l.as_slice::<f32>()[..n * v],
            None => &self.logits.as_slice::<f32>()[..v],
        })
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
        Ok(per_stage(&self.stages, &gpu.stamps(&self.step)?))
    }
}

/// Seconds per kind of stage from the timestamps `t` taken after each of
/// `stages`, in order of first appearance.
fn per_stage(stages: &[&'static str], t: &[f64]) -> Vec<(&'static str, f64)> {
    let mut out: Vec<(&'static str, f64)> = Vec::new();
    for (i, name) in stages.iter().enumerate() {
        let dt = t[i + 1] - t[i];
        match out.iter_mut().find(|(n, _)| n == name) {
            Some(e) => e.1 += dt,
            None => out.push((name, dt)),
        }
    }
    out
}
