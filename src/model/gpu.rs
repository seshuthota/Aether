//! A whole decode step on the GPU (`--features vulkan`; IMPLEMENTATION.md
//! L.3): the transformer of [`super::Transformer::forward`] as Vulkan
//! compute shaders. Decode runs any model whose matrices are in the
//! formats of a llama.cpp Q4_K_M file (Q4_0, Q5_0, Q8_0, Q4_K, Q6_K): Q4_0
//! through the tuned `q4_0_gemv.comp`, the others through `qx_gemv.comp`.
//! The prompt pass needs every matrix in Q4_0 (a `--pure` Q4_0 GGUF).
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
use crate::kernels::vulkan::{Bound, Buffer, Commands, Gpu, Kernel, MatQ, MatQ4_0, MatQ4_0T, Recorder, Result, groups_for};
use crate::quant::BlockQ4_0;
use crate::weight::{Store, Weight};

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

/// A matrix on the GPU: Q4_0 in the layout of the Q4_0 shaders (decode and
/// prompt), any other format in `qx_gemv`'s (decode only).
enum Mat {
    Q4_0(MatQ4_0),
    Qx(MatQ),
}

/// The matrices `ws` (all the same format), one under the other, as one
/// owned weight.
fn concat(ws: &[(&Weight, usize)]) -> Option<Weight> {
    macro_rules! join {
        ($v:ident) => {
            Weight::$v(Store::Owned(ws.iter().map(|(w, _)| if let Weight::$v(b) = w { Some(b.to_vec()) } else { None }).collect::<Option<Vec<_>>>()?.concat()))
        };
    }
    Some(match ws[0].0 {
        Weight::Q4_0(_) => join!(Q4_0),
        Weight::Q5_0(_) => join!(Q5_0),
        Weight::Q8_0(_) => join!(Q8_0),
        Weight::Q4_K(_) => join!(Q4_K),
        Weight::Q6_K(_) => join!(Q6_K),
        _ => return None,
    })
}

impl Mat {
    /// Uploads `ws` (matrices of the same width and format, with their
    /// rows) stacked into one.
    fn new(gpu: &Gpu, ws: &[(&Weight, usize)], what: &str) -> Result<Mat> {
        let rows = ws.iter().map(|w| w.1).sum();
        if let Weight::Q4_0(_) = ws[0].0 {
            let mut blocks = Vec::new();
            for (w, _) in ws {
                blocks.extend_from_slice(q4_0(w, what)?);
            }
            return Ok(Mat::Q4_0(MatQ4_0::new(gpu, &blocks, rows)?));
        }
        let joined;
        let w = match ws {
            [(w, _)] => *w,
            _ => {
                joined = concat(ws).ok_or_else(|| format!("{what}: matrices of different formats"))?;
                &joined
            }
        };
        Ok(Mat::Qx(MatQ::new(gpu, w, rows).map_err(|e| format!("{what}: {e}"))?))
    }

    fn rows(&self) -> usize {
        match self {
            Mat::Q4_0(m) => m.rows,
            Mat::Qx(m) => m.rows,
        }
    }

    fn nb(&self) -> usize {
        match self {
            Mat::Q4_0(m) => m.nb,
            Mat::Qx(m) => m.nb,
        }
    }

    /// ggml's type id.
    fn ty(&self) -> u32 {
        match self {
            Mat::Q4_0(_) => 2,
            Mat::Qx(m) => m.ty,
        }
    }

    /// The two weight buffers, bindings 0 and 1 of every product shader.
    fn bufs(&self) -> [&Buffer; 2] {
        match self {
            Mat::Q4_0(m) => [&m.qs, &m.d],
            Mat::Qx(m) => [&m.qs, &m.sc],
        }
    }

    fn q4_0(&self) -> Option<&MatQ4_0> {
        if let Mat::Q4_0(m) = self { Some(m) } else { None }
    }
}

/// Matrices whose products go into one buffer (q, k, v; gate, up): one
/// stacked matrix when they share a format, else one each, writing at a
/// byte offset of the output.
fn parts(gpu: &Gpu, ws: &[(&Weight, usize)], what: &str) -> Result<Vec<(Mat, usize)>> {
    if ws.iter().all(|(w, _)| w.format_name() == ws[0].0.format_name()) {
        return Ok(vec![(Mat::new(gpu, ws, what)?, 0)]);
    }
    let mut at = 0usize;
    let mut v = Vec::new();
    for &w in ws {
        let align = gpu.info.storage_offset_align as usize;
        if !at.is_multiple_of(align) {
            return Err(format!("{what}: an output offset of {at} bytes is not a multiple of {align}"));
        }
        v.push((Mat::new(gpu, &[w], what)?, at));
        at += 4 * w.1;
    }
    Ok(v)
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
    /// With matrices of different formats (a Q4_K_M file), one part per
    /// format, each with the byte offset of its rows in the output.
    wqkv: Vec<(Mat, usize)>,
    wo: Mat,
    w_gate_up: Vec<(Mat, usize)>,
    w_down: Mat,
    attn_norm: Buffer,
    ffn_norm: Buffer,
    bq: Buffer,
    bk: Buffer,
    bv: Buffer,
}

/// One layer's kernels with their buffers bound: the matrix products both
/// as GEMV (decode, one per part) and GEMM (prompt, Q4_0 models only).
struct Bl {
    norm1: Bound,
    qkv: Vec<Bound>,
    post: Bound,
    attn: Bound,
    merge: Bound,
    o: Bound,
    norm2: Bound,
    gate_up: Vec<Bound>,
    swiglu: Bound,
    down: Bound,
    /// q/k/v, wo, gate/up, down.
    mm: Option<[Bound; 4]>,
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
    /// The prompt's products with the f32 shader (`q4_0_gemm.comp`) even
    /// where the f16 one (`q4_0_gemm_h.comp`, [`F16`]) could run.
    pub f32_gemm: bool,
}

/// The prompt's products in f16 (`shaders/q4_0_gemm_h.comp`), where the
/// GPU has fp16 arithmetic: a second copy of every matrix in that
/// shader's layout, and the activations transposed to f16 in `xt` before
/// each product.
struct F16 {
    /// Per layer: q/k/v, wo, gate/up, down; then the classifier.
    mats: Vec<[MatQ4_0T; 4]>,
    cls: MatQ4_0T,
    /// The transpose from xb, from xb2 (attention's output) and from
    /// gate/up (with the SwiGLU).
    t_xb: Bound,
    t_xb2: Bound,
    t_glu: Bound,
    /// Per layer, the four products (reading xt); the classifier's with
    /// every token's logits.
    mm: Vec<[Bound; 4]>,
    cls_mm: Option<Bound>,
    _xt: Buffer,
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
    /// The two passes of the argmax of `logits` into `best`.
    amax: Bound,
    best: Buffer,
    // Everything the recorded commands use must outlive them; held here.
    _keep: Vec<Buffer>,
    cls: Mat,
    layers: Vec<Layer>,
    f16: Option<F16>,
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
        let mat = |w: &Weight, rows: usize, what: &str| Mat::new(gpu, &[(w, rows)], what);

        let mut layers = Vec::new();
        for (l, lw) in weights.layers.iter().enumerate() {
            let bias = |b: &Option<crate::aligned::AlignedBuf>, n: usize| match b {
                Some(b) => f32buf(b),
                None => zeros(n),
            };
            layers.push(Layer {
                wqkv: parts(gpu, &[(&lw.wq, dim), (&lw.wk, kv_dim), (&lw.wv, kv_dim)], &format!("layer {l} wq/wk/wv"))?,
                wo: mat(&lw.wo, dim, "wo")?,
                w_gate_up: parts(gpu, &[(&lw.w_gate, hid), (&lw.w_up, hid)], "w_gate/w_up")?,
                w_down: mat(&lw.w_down, dim, "w_down")?,
                attn_norm: f32buf(&lw.attn_norm)?,
                ffn_norm: f32buf(&lw.ffn_norm)?,
                bq: bias(&lw.bq, dim)?,
                bk: bias(&lw.bk, kv_dim)?,
                bv: bias(&lw.bv, kv_dim)?,
            });
        }
        let cls = mat(weights.classifier.as_ref().unwrap_or(&weights.token_embedding), c.vocab_size, "classifier")?;
        // The prompt pass's shaders read Q4_0 only.
        let single_q4_0 = |p: &[(Mat, usize)]| p.len() == 1 && p[0].0.q4_0().is_some();
        let pure = cls.q4_0().is_some() && layers.iter().all(|ly| single_q4_0(&ly.wqkv) && ly.wo.q4_0().is_some() && single_q4_0(&ly.w_gate_up) && ly.w_down.q4_0().is_some());
        let final_norm = f32buf(&weights.final_norm)?;

        // One row per token of a batch; decoding uses row 0.
        let x = zeros(batch * dim)?;
        // q, k, v: one buffer, the output of wqkv, which the kernels after
        // it bind views into; gate then up likewise.
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
        let swiglu = Kernel::new(gpu, crate::spirv!("swiglu"), 2, 4)?;
        let argmax = Kernel::new(gpu, crate::spirv!("argmax"), 3, 8)?;
        // In decode, wo and down add their products to the residual stream
        // x; q/k/v and the classifier apply the RMS norm before them as they
        // load x. gate/up and down do not fuse theirs: every workgroup
        // would redo the norm's sum (9,728 gate/up rows) or the SwiGLU's
        // exponentials (4,864 per subgroup), and on the phone both fused
        // stages were slower than the separate kernels.
        // The GEMV kernels, made as the matrices need them: by format, rows
        // per 32 lanes, and whether they add to the output (wo, down) or
        // normalize the input (q/k/v, the classifier).
        let mut gemv_ks: Vec<((u32, u32, bool, bool), Kernel)> = Vec::new();
        // Binds the GEMV of `m` (input `x_in`, output `out` at a byte
        // offset); binding 4 is the norm's weights, unused (any buffer)
        // without the norm.
        let mut gemv_bind = |m: &Mat, accum: bool, norm: Option<&Buffer>, x_in: &Buffer, out: (&Buffer, usize)| -> Result<Bound> {
            let key = (m.ty(), gemv_r(m.rows()), accum, norm.is_some());
            let i = match gemv_ks.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    let (ty, r, accum, norm) = key;
                    let k = if ty == 2 { gpu.q4_0_gemv_kernel_fused(r, accum, false, norm)? } else { gpu.qx_gemv_kernel(ty, r, accum, false, norm)? };
                    gemv_ks.push((key, k));
                    gemv_ks.len() - 1
                }
            };
            let [q, d] = m.bufs();
            gemv_ks[i].1.bind_at(gpu, &[(q, 0), (d, 0), (x_in, 0), out, (norm.unwrap_or(x_in), 0)])
        };
        let (mm, mm_acc, mm_glu) = (gpu.q4_0_gemm_kernel()?, gpu.q4_0_gemm_kernel_fused(true, false)?, gpu.q4_0_gemm_kernel_fused(true, true)?);
        let mut f16_kernels = Vec::new();

        // Bind everything first (binding can fail; recording cannot).
        let mut bound = Vec::new();
        for ly in &layers {
            let mm = if pure {
                let (wqkv, wo, wgu, wd) = (ly.wqkv[0].0.q4_0().unwrap(), ly.wo.q4_0().unwrap(), ly.w_gate_up[0].0.q4_0().unwrap(), ly.w_down.q4_0().unwrap());
                Some([
                    mm.bind(gpu, &[&wqkv.qs, &wqkv.d, &xb, &qkv])?,
                    mm_acc.bind(gpu, &[&wo.qs, &wo.d, &xb2, &x])?,
                    mm.bind(gpu, &[&wgu.qs, &wgu.d, &xb, &hb12])?,
                    mm_glu.bind(gpu, &[&wd.qs, &wd.d, &hb12, &x])?,
                ])
            } else {
                None
            };
            bound.push(Bl {
                norm1: rms.bind(gpu, &[&x, &ly.attn_norm, &xb])?,
                qkv: ly.wqkv.iter().map(|(m, at)| gemv_bind(m, false, Some(&ly.attn_norm), &x, (&qkv, *at))).collect::<Result<_>>()?,
                post: post.bind_at(gpu, &[(&qkv, 0), (&qkv, k_at), (&qkv, v_at), (&ly.bq, 0), (&ly.bk, 0), (&ly.bv, 0), (&kc, 0), (&vc, 0), (&params, 0)])?,
                attn: attn.bind(gpu, &[&qkv, &kc, &vc, &parts, &params])?,
                merge: merge.bind(gpu, &[&parts, &xb2, &params])?,
                o: gemv_bind(&ly.wo, true, None, &xb2, (&x, 0))?,
                norm2: rms.bind(gpu, &[&x, &ly.ffn_norm, &xb])?,
                gate_up: ly.w_gate_up.iter().map(|(m, at)| gemv_bind(m, false, None, &xb, (&hb12, *at))).collect::<Result<_>>()?,
                swiglu: swiglu.bind_at(gpu, &[(&hb12, 0), (&hb12, 4 * hid)])?,
                down: gemv_bind(&ly.w_down, true, None, &hb12, (&x, 0))?,
                mm,
            });
        }
        let fnorm = rms.bind(gpu, &[&x, &final_norm, &xb])?;
        let fcls = gemv_bind(&cls, false, Some(&final_norm), &x, (&logits, 0))?;
        let fcls_b = gemv_bind(&cls, false, None, &xb, (&logits, 0))?;
        let fcls_mm = match (&logits_all, cls.q4_0()) {
            (Some(l), Some(c)) if pure => Some(mm.bind(gpu, &[&c.qs, &c.d, &xb, l])?),
            _ => None,
        };

        let f16 = if pure && gpu.info.fp16 && !opt.f32_gemm {
            let (tk, tk_glu, mh, mh_acc) = (gpu.to_f16t_kernel(false)?, gpu.to_f16t_kernel(true)?, gpu.q4_0_gemm_h_kernel(false)?, gpu.q4_0_gemm_h_kernel(true)?);
            let xt = gpu.buffer(dim.max(hid) * Gpu::f16t_pad(batch) * 2)?;
            let mat_t = |w: &Weight, rows: usize, what: &str| MatQ4_0T::new(gpu, q4_0(w, what)?, rows);
            let stacked_t = |ws: &[(&Weight, usize)], what: &str| -> Result<MatQ4_0T> {
                let mut blocks = Vec::new();
                for (w, _) in ws {
                    blocks.extend_from_slice(q4_0(w, what)?);
                }
                MatQ4_0T::new(gpu, &blocks, ws.iter().map(|w| w.1).sum())
            };
            let mut mats = Vec::new();
            let mut mm = Vec::new();
            for (l, lw) in weights.layers.iter().enumerate() {
                let m = [
                    stacked_t(&[(&lw.wq, dim), (&lw.wk, kv_dim), (&lw.wv, kv_dim)], &format!("layer {l} wq/wk/wv"))?,
                    mat_t(&lw.wo, dim, "wo")?,
                    stacked_t(&[(&lw.w_gate, hid), (&lw.w_up, hid)], "w_gate/w_up")?,
                    mat_t(&lw.w_down, dim, "w_down")?,
                ];
                mm.push([
                    mh.bind(gpu, &[&m[0].q, &m[0].d, &xt, &qkv])?,
                    mh_acc.bind(gpu, &[&m[1].q, &m[1].d, &xt, &x])?,
                    mh.bind(gpu, &[&m[2].q, &m[2].d, &xt, &hb12])?,
                    mh_acc.bind(gpu, &[&m[3].q, &m[3].d, &xt, &x])?,
                ]);
                mats.push(m);
            }
            let cls_t = mat_t(weights.classifier.as_ref().unwrap_or(&weights.token_embedding), c.vocab_size, "classifier")?;
            let cls_mm = match &logits_all {
                Some(l) => Some(mh.bind(gpu, &[&cls_t.q, &cls_t.d, &xt, l])?),
                None => None,
            };
            let f = F16 {
                t_xb: tk.bind(gpu, &[&xb, &xt])?,
                t_xb2: tk.bind(gpu, &[&xb2, &xt])?,
                t_glu: tk_glu.bind(gpu, &[&hb12, &xt])?,
                mats,
                cls: cls_t,
                mm,
                cls_mm,
                _xt: xt,
            };
            f16_kernels = vec![tk, tk_glu, mh, mh_acc];
            Some(f)
        } else {
            None
        };

        // (value, index) per argmax workgroup; the index read by the CPU.
        let (amax_parts, best) = (zeros(2 * AMAX_GROUPS)?, cached(1)?);
        let amax = argmax.bind(gpu, &[&logits, &amax_parts, &best])?;

        let mut keep = vec![xb, xb2, qkv, hb12, parts, final_norm, amax_parts];
        keep.shrink_to_fit();
        let mut kernels = vec![rms, post, attn, merge, swiglu, argmax, mm, mm_acc, mm_glu];
        kernels.append(&mut f16_kernels);
        kernels.extend(gemv_ks.into_iter().map(|(_, k)| k));
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
            amax,
            best,
            _keep: keep,
            cls,
            layers,
            f16,
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
        let gemv = |r: &mut Recorder, k: &Bound, m: &Mat| r.dispatch(k, &push(&[m.rows() as u32, m.nb() as u32, eps]), groups_for(m.rows(), 4 * gemv_r(m.rows()) as usize));
        // With f16 products, product i of layer l (q/k/v, wo, gate/up,
        // down) transposes its input (`t`) and runs the f16 shader. A batch
        // only runs on a Q4_0 model (`forward_batch` checks), whose
        // products are one matrix each.
        let f16 = self.f16.as_ref().filter(|_| n > 1);
        let product = |r: &mut Recorder, gemv_b: &[Bound], gemm: Option<&Bound>, ms: &[&Mat], h: Option<(&Bound, &Bound, &MatQ4_0T)>| {
            if n == 1 {
                gemv_b.iter().zip(ms).for_each(|(k, m)| gemv(r, k, m));
            } else if let Some((t, mm, mt)) = h {
                Gpu::record_q4_0_gemm_h(r, t, mm, mt, n);
            } else {
                let m = ms[0];
                r.dispatch(gemm.unwrap(), &push(&[m.rows() as u32, m.nb() as u32, nt]), [m.rows().div_ceil(64) as u32, n.div_ceil(64) as u32, 1]);
            }
        };
        fn mats(p: &[(Mat, usize)]) -> Vec<&Mat> {
            p.iter().map(|(m, _)| m).collect()
        }
        fn mm(b: &Bl, i: usize) -> Option<&Bound> {
            b.mm.as_ref().map(|m| &m[i])
        }
        let h = |l: usize, i: usize, t: fn(&F16) -> &Bound| f16.map(|f| (t(f), &f.mm[l][i], &f.mats[l][i]));
        for (l, (b, ly)) in self.bound.iter().zip(&self.layers).enumerate() {
            let layer_off = (l * c.seq_len * kv_dim) as u32;
            // In decode the first norm is part of q/k/v.
            if n > 1 {
                r.dispatch(&b.norm1, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
                end(r, "rmsnorm");
            }
            product(r, &b.qkv, mm(b, 0), &mats(&ly.wqkv), h(l, 0, |f| &f.t_xb));
            end(r, "q/k/v");
            r.dispatch(&b.post, &push(&[dim as u32, kv_dim as u32, hs as u32, halves, layer_off, qkv_dim]), [nt, 1, 1]);
            end(r, "bias + RoPE + cache");
            r.dispatch(&b.attn, &push(&[c.n_heads as u32, c.n_kv_heads as u32, hs as u32, chunks, layer_off, qkv_dim]), [c.n_heads as u32, chunks, nt]);
            end(r, "attention");
            r.dispatch(&b.merge, &push(&[hs as u32, chunks]), [c.n_heads as u32, nt, 1]);
            end(r, "attention merge");
            product(r, std::slice::from_ref(&b.o), mm(b, 1), &[&ly.wo], h(l, 1, |f| &f.t_xb2));
            end(r, "wo + add");
            r.dispatch(&b.norm2, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
            end(r, "rmsnorm");
            product(r, &b.gate_up, mm(b, 2), &mats(&ly.w_gate_up), h(l, 2, |f| &f.t_xb));
            end(r, "gate/up");
            // The GEMM applies the SwiGLU as it loads gate and up.
            if n == 1 {
                r.dispatch(&b.swiglu, &push(&[c.hidden_dim as u32]), [c.hidden_dim.div_ceil(256) as u32, 1, 1]);
                end(r, "swiglu");
                product(r, std::slice::from_ref(&b.down), mm(b, 3), &[&ly.w_down], None);
                end(r, "down + add");
            } else {
                product(r, std::slice::from_ref(&b.down), mm(b, 3), &[&ly.w_down], h(l, 3, |f| &f.t_glu));
                end(r, "swiglu + down + add");
            }
        }
        match all {
            Some(cls_mm) => {
                r.dispatch(&self.fnorm, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
                end(r, "rmsnorm");
                match f16.and_then(|f| f.cls_mm.as_ref().map(|mm| (f, mm))) {
                    Some((f, mm)) => Gpu::record_q4_0_gemm_h(r, &f.t_xb, mm, &f.cls, n),
                    None => {
                        let m = &self.cls;
                        r.dispatch(cls_mm, &push(&[m.rows() as u32, m.nb() as u32, nt]), [m.rows().div_ceil(64) as u32, n.div_ceil(64) as u32, 1]);
                    }
                }
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
        if all.is_none() {
            let v = c.vocab_size as u32;
            r.dispatch(&self.amax, &push(&[v, 0]), [AMAX_GROUPS as u32, 1, 1]);
            r.barrier();
            r.dispatch(&self.amax, &push(&[AMAX_GROUPS as u32, 1]), [1, 1, 1]);
            end(r, "argmax");
        }
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

    /// The most likely next token after the last [`GpuModel::forward`] or
    /// [`GpuModel::forward_batch`] (without [`Options::all_logits`]),
    /// found on the GPU: greedy decoding need not read the logits.
    pub fn greedy(&self) -> usize {
        self.best.as_slice::<u32>()[0] as usize
    }

    /// Whether [`GpuModel::forward_batch`] can run: every matrix is Q4_0.
    pub fn batched(&self) -> bool {
        self.bound.iter().all(|b| b.mm.is_some())
    }

    /// Runs `tokens` (at most [`GpuModel::batch`]) at positions `pos ..`
    /// in one pass, as the CPU's `forward_batch` does, and returns the
    /// last token's logits, or with [`Options::all_logits`] every token's
    /// (`tokens.len() × vocab`). The pass is recorded for this call: its
    /// size depends on the number of tokens and the position.
    pub fn forward_batch(&mut self, gpu: &Gpu, tokens: &[usize], pos: usize) -> Result<&[f32]> {
        if !self.batched() {
            return Err("the GPU prompt pass needs every matrix in Q4_0 (a --pure Q4_0 GGUF)".into());
        }
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

/// Workgroups of the argmax's first pass.
const AMAX_GROUPS: usize = 64;

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
