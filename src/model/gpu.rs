//! A whole decode step on the GPU (`--features vulkan`; IMPLEMENTATION.md
//! L.3): the transformer of [`super::Transformer::forward`] as Vulkan
//! compute shaders. Decode runs any model whose matrices are in the
//! formats of a llama.cpp Q4_K_M file (Q4_0, Q5_0, Q8_0, Q4_K, Q6_K): Q4_0
//! through the tuned `q4_0_gemv.comp`, the others through `qx_gemv.comp`.
//! The prompt pass of a Q4_0 model uses `q4_0_gemm.comp` (or its f16
//! sibling `q4_0_gemm_h.comp`); any other model's, `qx_gemm.comp` (or
//! `qx_gemm_h.comp`).
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
use crate::kernels::vulkan::{Bound, Buffer, Commands, Gpu, Kernel, MatQ, MatQ4_0, MatQ4_0T, MatQT, Recorder, Result, groups_for};
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

/// `f` of `ws` (matrices of the same width and format, with their rows)
/// stacked into one, and its rows.
fn stacked<R>(ws: &[(&Weight, usize)], what: &str, f: impl FnOnce(&Weight, usize) -> Result<R>) -> Result<R> {
    let rows = ws.iter().map(|w| w.1).sum();
    let joined;
    let w = match ws {
        [(w, _)] => *w,
        _ => {
            joined = concat(ws).ok_or_else(|| format!("{what}: matrices of different formats"))?;
            &joined
        }
    };
    f(w, rows).map_err(|e| format!("{what}: {e}"))
}

/// The Q4_0 blocks of `ws`, one matrix under the other.
fn q4_0_stacked(ws: &[(&Weight, usize)], what: &str) -> Result<Vec<BlockQ4_0>> {
    let mut blocks = Vec::new();
    for (w, _) in ws {
        blocks.extend_from_slice(q4_0(w, what)?);
    }
    Ok(blocks)
}

impl Mat {
    /// Uploads `ws` (matrices of the same width and format, with their
    /// rows) stacked into one.
    fn new(gpu: &Gpu, ws: &[(&Weight, usize)], what: &str) -> Result<Mat> {
        if let Weight::Q4_0(_) = ws[0].0 {
            return Ok(Mat::Q4_0(MatQ4_0::new(gpu, &q4_0_stacked(ws, what)?, ws.iter().map(|w| w.1).sum())?));
        }
        stacked(ws, what, |w, rows| Ok(Mat::Qx(MatQ::new(gpu, w, rows)?)))
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
fn parts<M>(gpu: &Gpu, ws: &[(&Weight, usize)], what: &str, new: impl Fn(&[(&Weight, usize)]) -> Result<M>) -> Result<Vec<(M, usize)>> {
    if ws.iter().all(|(w, _)| w.format_name() == ws[0].0.format_name()) {
        return Ok(vec![(new(ws)?, 0)]);
    }
    let mut at = 0usize;
    let mut v = Vec::new();
    for &w in ws {
        let align = gpu.info.storage_offset_align as usize;
        if !at.is_multiple_of(align) {
            return Err(format!("{what}: an output offset of {at} bytes is not a multiple of {align}"));
        }
        v.push((new(&[w])?, at));
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

/// [`GpuModel::forward_few`] takes at most this many tokens. Its products
/// run the decode shader for several tokens (`qx_gemv_n_kernel`): each
/// weight is read once, against a whole 64-token tile of the prefill
/// shaders. Where decode is limited by reading the weights, a pass of a
/// few tokens should cost little more than one decode step. On the PC's
/// Intel GPU, where decode is limited by arithmetic (~11 GB/s), it costs
/// 4-20 steps (`vk_gemv_n`), so `forward_batch` does not use it; the
/// phone decides.
pub const FEW: usize = 8;

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
/// as GEMV (decode) and GEMM (prompt), one per part.
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
    /// q/k/v, wo, gate/up, down: `q4_0_gemm` for a Q4_0 model, else
    /// `qx_gemm` per part.
    mm: [Vec<Bound>; 4],
    /// The same for 2 to [`FEW`] tokens, per part.
    few: [Vec<Bound>; 4],
}

/// Which logits [`GpuModel::pass`] writes; [`Logits`] with the binding.
#[derive(Clone, Copy)]
enum Want {
    Last,
    All,
    Few,
}

/// Which logits a pass writes.
#[derive(Clone, Copy)]
enum Logits<'a> {
    /// The last token's, into `logits`, and their argmax.
    Last,
    /// Every token's, by this prefill product (`Options::all_logits`).
    All(&'a Bound),
    /// Every token's, by the few-token product into `logits_few`.
    Few,
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
    /// The prompt's products with the f32 shaders (`q4_0_gemm.comp`,
    /// `qx_gemm.comp`) even where the f16 ones ([`F16`]) could run.
    pub f32_gemm: bool,
}

/// A matrix in the layout of an f16 prompt shader: `q4_0_gemm_h.comp` for
/// a Q4_0 model, else `qx_gemm_h.comp`, with the length of a token's
/// output row (the stacked q/k/v or gate/up when it is one part of it).
enum HMat {
    Q4_0(MatQ4_0T),
    Qx(MatQT, usize),
}

/// One f16 product: the matrix and its kernel, bound to `xt` and the
/// output.
struct HPart {
    mm: Bound,
    m: HMat,
}

/// The prompt's products in f16 (`shaders/q4_0_gemm_h.comp`,
/// `shaders/qx_gemm_h.comp`), where the GPU has fp16 arithmetic: a second
/// copy of every matrix in that shader's layout, and the activations
/// transposed to f16 in `xt` before each product.
struct F16 {
    /// The transpose from xb, from xb2 (attention's output) and from
    /// gate/up (with the SwiGLU).
    t_xb: Bound,
    t_xb2: Bound,
    t_glu: Bound,
    /// Per layer, the four products (reading xt), one part per format as
    /// in [`Layer`]; the classifier's with every token's logits.
    mm: Vec<[Vec<HPart>; 4]>,
    cls_mm: Option<HPart>,
    _xt: Buffer,
}

impl F16 {
    /// Records the transpose `t` of `n` tokens, then the parts' products.
    fn record(r: &mut Recorder, t: &Bound, ps: &[HPart], n: usize) {
        let nb = match &ps[0].m {
            HMat::Q4_0(m) => m.nb,
            HMat::Qx(m, _) => m.nb,
        };
        Gpu::record_f16t(r, t, 32 * nb, n);
        r.barrier();
        for p in ps {
            match &p.m {
                HMat::Q4_0(m) => Gpu::record_q4_0_gemm_h_mm(r, &p.mm, m, n),
                HMat::Qx(m, stride) => Gpu::record_qx_gemm_h(r, &p.mm, m, n, *stride),
            }
        }
    }
}

/// The f16 products of `ws` stacked (writing `out`, with `accum` adding to
/// it), from `xt`: one Q4_0 matrix in a `pure` model, else one per format
/// as [`parts`] splits them. Kernels are made as needed into `ks`, by
/// format (0: `q4_0_gemm_h`) and `accum`.
#[allow(clippy::too_many_arguments)]
fn h_parts(gpu: &Gpu, ks: &mut Vec<((u32, bool), Kernel)>, pure: bool, ws: &[(&Weight, usize)], what: &str, accum: bool, xt: &Buffer, out: &Buffer) -> Result<Vec<HPart>> {
    let total = ws.iter().map(|w| w.1).sum();
    let mats = if pure {
        vec![(HMat::Q4_0(MatQ4_0T::new(gpu, &q4_0_stacked(ws, what)?, total)?), 0)]
    } else {
        parts(gpu, ws, what, |ws| stacked(ws, what, |w, rows| MatQT::new(gpu, w, rows)))?.into_iter().map(|(m, at)| (HMat::Qx(m, total), at)).collect()
    };
    mats.into_iter()
        .map(|(m, at)| {
            let (key, [q, s]) = match &m {
                HMat::Q4_0(t) => ((0, accum), [&t.q, &t.d]),
                HMat::Qx(t, _) => ((t.ty, accum), [&t.q, &t.s]),
            };
            let i = match ks.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    ks.push((key, if key.0 == 0 { gpu.q4_0_gemm_h_kernel(accum)? } else { gpu.qx_gemm_h_kernel(key.0, accum)? }));
                    ks.len() - 1
                }
            };
            Ok(HPart { mm: ks[i].1.bind_at(gpu, &[(q, 0), (s, 0), (xt, 0), (out, at)])?, m })
        })
        .collect()
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
    /// `few × vocab` logits for [`GpuModel::forward_few`].
    logits_few: Buffer,
    /// Tokens a few-token pass takes: [`FEW`], or the batch if smaller.
    pub few: usize,
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
    fcls_few: Bound,
    /// The two passes of the argmax of `logits` into `best`.
    amax: Bound,
    best: Buffer,
    // Everything the recorded commands use must outlive them; held here.
    _keep: Vec<Buffer>,
    cls: Mat,
    layers: Vec<Layer>,
    f16: Option<F16>,
    /// Every matrix is Q4_0: the prompt pass uses the Q4_0 shaders.
    pure: bool,
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
                wqkv: parts(gpu, &[(&lw.wq, dim), (&lw.wk, kv_dim), (&lw.wv, kv_dim)], &format!("layer {l} wq/wk/wv"), |ws| Mat::new(gpu, ws, "wq/wk/wv"))?,
                wo: mat(&lw.wo, dim, "wo")?,
                w_gate_up: parts(gpu, &[(&lw.w_gate, hid), (&lw.w_up, hid)], "w_gate/w_up", |ws| Mat::new(gpu, ws, "w_gate/w_up"))?,
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
        let few = FEW.min(batch);
        let logits_few = cached(few * c.vocab_size)?;
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
        // A model that is not all Q4_0: the multi-format GEMM per part, by
        // format and fusion.
        let mut gemm_ks: Vec<((u32, bool, bool), Kernel)> = Vec::new();
        let mut gemm_bind = |m: &Mat, accum: bool, glu: bool, x_in: &Buffer, out: (&Buffer, usize)| -> Result<Bound> {
            let key = (m.ty(), accum, glu);
            let i = match gemm_ks.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    gemm_ks.push((key, gpu.qx_gemm_kernel(m.ty(), accum, glu)?));
                    gemm_ks.len() - 1
                }
            };
            let [q, d] = m.bufs();
            gemm_ks[i].1.bind_at(gpu, &[(q, 0), (d, 0), (x_in, 0), out])
        };

        // The few-token products, by format and fusion (the kernel takes up
        // to `few` tokens; at least 2 for the shader, unused if `few` < 2).
        let mut few_ks: Vec<((u32, bool, bool), Kernel)> = Vec::new();
        let mut few_bind = |m: &Mat, accum: bool, glu: bool, x_in: &Buffer, out: (&Buffer, usize)| -> Result<Bound> {
            let key = (m.ty(), accum, glu);
            let i = match few_ks.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    few_ks.push((key, gpu.qx_gemv_n_kernel(m.ty(), accum, glu, few.max(2) as u32)?));
                    few_ks.len() - 1
                }
            };
            let [q, d] = m.bufs();
            few_ks[i].1.bind_at(gpu, &[(q, 0), (d, 0), (x_in, 0), out, (x_in, 0)])
        };

        // Bind everything first (binding can fail; recording cannot).
        let mut bound = Vec::new();
        for ly in &layers {
            let mm = if pure {
                let (wqkv, wo, wgu, wd) = (ly.wqkv[0].0.q4_0().unwrap(), ly.wo.q4_0().unwrap(), ly.w_gate_up[0].0.q4_0().unwrap(), ly.w_down.q4_0().unwrap());
                [
                    vec![mm.bind(gpu, &[&wqkv.qs, &wqkv.d, &xb, &qkv])?],
                    vec![mm_acc.bind(gpu, &[&wo.qs, &wo.d, &xb2, &x])?],
                    vec![mm.bind(gpu, &[&wgu.qs, &wgu.d, &xb, &hb12])?],
                    vec![mm_glu.bind(gpu, &[&wd.qs, &wd.d, &hb12, &x])?],
                ]
            } else {
                [
                    ly.wqkv.iter().map(|(m, at)| gemm_bind(m, false, false, &xb, (&qkv, *at))).collect::<Result<_>>()?,
                    vec![gemm_bind(&ly.wo, true, false, &xb2, (&x, 0))?],
                    ly.w_gate_up.iter().map(|(m, at)| gemm_bind(m, false, false, &xb, (&hb12, *at))).collect::<Result<_>>()?,
                    vec![gemm_bind(&ly.w_down, true, true, &hb12, (&x, 0))?],
                ]
            };
            let few = [
                ly.wqkv.iter().map(|(m, at)| few_bind(m, false, false, &xb, (&qkv, *at))).collect::<Result<_>>()?,
                vec![few_bind(&ly.wo, true, false, &xb2, (&x, 0))?],
                ly.w_gate_up.iter().map(|(m, at)| few_bind(m, false, false, &xb, (&hb12, *at))).collect::<Result<_>>()?,
                vec![few_bind(&ly.w_down, true, true, &hb12, (&x, 0))?],
            ];
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
                few,
            });
        }
        let fnorm = rms.bind(gpu, &[&x, &final_norm, &xb])?;
        let fcls = gemv_bind(&cls, false, Some(&final_norm), &x, (&logits, 0))?;
        let fcls_b = gemv_bind(&cls, false, None, &xb, (&logits, 0))?;
        let fcls_mm = match (&logits_all, cls.q4_0()) {
            (Some(l), Some(c)) if pure => Some(mm.bind(gpu, &[&c.qs, &c.d, &xb, l])?),
            (Some(l), _) => Some(gemm_bind(&cls, false, false, &xb, (l, 0))?),
            (None, _) => None,
        };
        let fcls_few = few_bind(&cls, false, false, &xb, (&logits_few, 0))?;

        let f16 = if gpu.info.fp16 && !opt.f32_gemm {
            let (tk, tk_glu) = (gpu.to_f16t_kernel(false)?, gpu.to_f16t_kernel(true)?);
            let xt = gpu.buffer(dim.max(hid) * Gpu::f16t_pad(batch) * 2)?;
            let mut ks = Vec::new();
            let mut mm = Vec::new();
            for (l, lw) in weights.layers.iter().enumerate() {
                let mut h = |ws: &[(&Weight, usize)], what: &str, accum: bool, out: &Buffer| h_parts(gpu, &mut ks, pure, ws, &format!("layer {l} {what}"), accum, &xt, out);
                mm.push([
                    h(&[(&lw.wq, dim), (&lw.wk, kv_dim), (&lw.wv, kv_dim)], "wq/wk/wv", false, &qkv)?,
                    h(&[(&lw.wo, dim)], "wo", true, &x)?,
                    h(&[(&lw.w_gate, hid), (&lw.w_up, hid)], "w_gate/w_up", false, &hb12)?,
                    h(&[(&lw.w_down, dim)], "w_down", true, &x)?,
                ]);
            }
            let cls_mm = match &logits_all {
                Some(l) => h_parts(gpu, &mut ks, pure, &[(weights.classifier.as_ref().unwrap_or(&weights.token_embedding), c.vocab_size)], "classifier", false, &xt, l)?.pop(),
                None => None,
            };
            let f = F16 { t_xb: tk.bind(gpu, &[&xb, &xt])?, t_xb2: tk.bind(gpu, &[&xb2, &xt])?, t_glu: tk_glu.bind(gpu, &[&hb12, &xt])?, mm, cls_mm, _xt: xt };
            f16_kernels = vec![tk, tk_glu];
            f16_kernels.extend(ks.into_iter().map(|(_, k)| k));
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
        kernels.extend(gemm_ks.into_iter().map(|(_, k)| k));
        kernels.extend(few_ks.into_iter().map(|(_, k)| k));
        let mut m = GpuModel {
            config,
            batch,
            weights,
            x,
            logits,
            logits_all,
            logits_few,
            few,
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
            fcls_few,
            amax,
            best,
            _keep: keep,
            cls,
            layers,
            f16,
            pure,
            _kernels: kernels,
        };
        let mut stages = Vec::new();
        let rec_all = |r: &mut Recorder, stages: &mut Vec<&'static str>| (0..opt.repeat.max(1)).for_each(|_| m.record_pass(r, 1, 0, Logits::Last, stages));
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
    /// `params` either way, `pos0` only sizes the attention), writing the
    /// logits `out` says ([`Logits::Few`] runs the few-token products).
    /// Each stage ends with a barrier and a timestamp (which only a timed
    /// recording writes).
    fn record_pass(&self, r: &mut Recorder, n: usize, pos0: usize, out: Logits, stages: &mut Vec<&'static str>) {
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
        // A GEMM over `n` tokens: `q4_0_gemm` takes rows, nb, tokens;
        // `qx_gemm` also the length of a token's output row (`stride`: the
        // stacked q/k/v or gate/up when this matrix is one part of it).
        let pure = self.pure;
        let gemm = |r: &mut Recorder, k: &Bound, m: &Mat, stride: usize| {
            let p = if pure { push(&[m.rows() as u32, m.nb() as u32, nt]) } else { push(&[m.rows() as u32, m.nb() as u32, nt, stride as u32]) };
            r.dispatch(k, &p, [m.rows().div_ceil(64) as u32, n.div_ceil(64) as u32, 1]);
        };
        // With Logits::Few: the decode shader for several tokens, 4 rows per
        // 32 lanes (`stride` as for `qx_gemm`; 0 is the matrix's rows).
        let few = n > 1 && matches!(out, Logits::Few);
        let gemv_n = |r: &mut Recorder, k: &Bound, m: &Mat, stride: usize| r.dispatch(k, &push(&[m.rows() as u32, m.nb() as u32, 0, nt, stride as u32]), groups_for(m.rows(), 16));
        // With f16 products, product i of layer l (q/k/v, wo, gate/up,
        // down) transposes its input (`t`) and runs the f16 shaders.
        let f16 = self.f16.as_ref().filter(|_| n > 1 && !few);
        let product = |r: &mut Recorder, gemv_b: &[Bound], gemm_b: &[Bound], few_b: &[Bound], ms: &[&Mat], stride: usize, h: Option<(&Bound, &[HPart])>| {
            if n == 1 {
                gemv_b.iter().zip(ms).for_each(|(k, m)| gemv(r, k, m));
            } else if few {
                few_b.iter().zip(ms).for_each(|(k, m)| gemv_n(r, k, m, stride));
            } else if let Some((t, ps)) = h {
                F16::record(r, t, ps, n);
            } else {
                gemm_b.iter().zip(ms).for_each(|(k, m)| gemm(r, k, m, stride));
            }
        };
        fn mats(p: &[(Mat, usize)]) -> Vec<&Mat> {
            p.iter().map(|(m, _)| m).collect()
        }
        let h = |l: usize, i: usize, t: fn(&F16) -> &Bound| f16.map(|f| (t(f), &f.mm[l][i][..]));
        for (l, (b, ly)) in self.bound.iter().zip(&self.layers).enumerate() {
            let layer_off = (l * c.seq_len * kv_dim) as u32;
            // In decode the first norm is part of q/k/v.
            if n > 1 {
                r.dispatch(&b.norm1, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
                end(r, "rmsnorm");
            }
            product(r, &b.qkv, &b.mm[0], &b.few[0], &mats(&ly.wqkv), qkv_dim as usize, h(l, 0, |f| &f.t_xb));
            end(r, "q/k/v");
            r.dispatch(&b.post, &push(&[dim as u32, kv_dim as u32, hs as u32, halves, layer_off, qkv_dim]), [nt, 1, 1]);
            end(r, "bias + RoPE + cache");
            r.dispatch(&b.attn, &push(&[c.n_heads as u32, c.n_kv_heads as u32, hs as u32, chunks, layer_off, qkv_dim]), [c.n_heads as u32, chunks, nt]);
            end(r, "attention");
            r.dispatch(&b.merge, &push(&[hs as u32, chunks]), [c.n_heads as u32, nt, 1]);
            end(r, "attention merge");
            product(r, std::slice::from_ref(&b.o), &b.mm[1], &b.few[1], &[&ly.wo], 0, h(l, 1, |f| &f.t_xb2));
            end(r, "wo + add");
            r.dispatch(&b.norm2, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
            end(r, "rmsnorm");
            product(r, &b.gate_up, &b.mm[2], &b.few[2], &mats(&ly.w_gate_up), 2 * c.hidden_dim, h(l, 2, |f| &f.t_xb));
            end(r, "gate/up");
            // The GEMM applies the SwiGLU as it loads gate and up.
            if n == 1 {
                r.dispatch(&b.swiglu, &push(&[c.hidden_dim as u32]), [c.hidden_dim.div_ceil(256) as u32, 1, 1]);
                end(r, "swiglu");
                product(r, std::slice::from_ref(&b.down), &b.mm[3], &b.few[3], &[&ly.w_down], 0, None);
                end(r, "down + add");
            } else {
                product(r, std::slice::from_ref(&b.down), &b.mm[3], &b.few[3], &[&ly.w_down], 0, h(l, 3, |f| &f.t_glu));
                end(r, "swiglu + down + add");
            }
        }
        match out {
            Logits::All(cls_mm) => {
                r.dispatch(&self.fnorm, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
                end(r, "rmsnorm");
                match f16.and_then(|f| f.cls_mm.as_ref().map(|mm| (f, mm))) {
                    Some((f, mm)) => F16::record(r, &f.t_xb, std::slice::from_ref(mm), n),
                    None => gemm(r, cls_mm, &self.cls, 0),
                }
            }
            Logits::Few => {
                r.dispatch(&self.fnorm, &push(&[dim as u32, eps, 0]), [nt, 1, 1]);
                end(r, "rmsnorm");
                gemv_n(r, &self.fcls_few, &self.cls, 0);
            }
            Logits::Last if n == 1 => gemv(r, &self.fcls, &self.cls),
            Logits::Last => {
                // Only the last token: its row of x, normalized into row 0.
                r.dispatch(&self.fnorm, &push(&[dim as u32, eps, nt - 1]), [1, 1, 1]);
                end(r, "rmsnorm");
                gemv(r, &self.fcls_b, &self.cls);
            }
        }
        end(r, "classifier");
        if let Logits::Last = out {
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

    /// Which shader the prompt pass's products use: `q4_0_gemm_h` (f16),
    /// `q4_0_gemm` (f32), or for any format `qx_gemm_h` (f16) or `qx_gemm`
    /// (f32).
    pub fn gemm_shader(&self) -> &'static str {
        match (self.pure, self.f16.is_some()) {
            (true, true) => "q4_0_gemm_h",
            (true, false) => "q4_0_gemm",
            (false, true) => "qx_gemm_h",
            (false, false) => "qx_gemm",
        }
    }

    /// Runs `tokens` (at most [`GpuModel::batch`]) at positions `pos ..`
    /// in one pass, as the CPU's `forward_batch` does, and returns the
    /// last token's logits, or with [`Options::all_logits`] every token's
    /// (`tokens.len() × vocab`). The pass is recorded for this call: its
    /// size depends on the number of tokens and the position.
    pub fn forward_batch(&mut self, gpu: &Gpu, tokens: &[usize], pos: usize) -> Result<&[f32]> {
        self.pass(gpu, tokens, pos, if self.logits_all.is_some() { Want::All } else { Want::Last })?;
        let v = self.config.vocab_size;
        Ok(match &self.logits_all {
            Some(l) => &l.as_slice::<f32>()[..tokens.len() * v],
            None => &self.logits.as_slice::<f32>()[..v],
        })
    }

    /// Runs 1 to [`GpuModel::few`] tokens at positions `pos ..` in one
    /// pass and returns every token's logits (`tokens.len() × vocab`):
    /// speculative decoding's check of its draft tokens, at about the cost
    /// of one decode step.
    pub fn forward_few(&mut self, gpu: &Gpu, tokens: &[usize], pos: usize) -> Result<&[f32]> {
        let n = tokens.len();
        assert!(n <= self.few, "{n} tokens, at most {}", self.few);
        if n == 1 {
            return self.forward(gpu, tokens[0], pos);
        }
        self.pass(gpu, tokens, pos, Want::Few)?;
        Ok(&self.logits_few.as_slice::<f32>()[..n * self.config.vocab_size])
    }

    /// Records and runs one pass over `tokens`, writing the logits `want`
    /// says.
    fn pass(&mut self, gpu: &Gpu, tokens: &[usize], pos: usize, want: Want) -> Result<()> {
        let n = tokens.len();
        assert!(n > 0 && n <= self.batch && pos + n <= self.config.seq_len, "{n} tokens at {pos}: batch {}, seq_len {}", self.batch, self.config.seq_len);
        for (i, &t) in tokens.iter().enumerate() {
            self.set_token(i, t, pos + i);
        }
        self.params.as_mut_slice::<u32>()[0] = pos as u32;
        let mut stages = Vec::new();
        let logits = match (want, &self.fcls_mm) {
            (Want::All, Some(b)) => Logits::All(b),
            (Want::Few, _) => Logits::Few,
            _ => Logits::Last,
        };
        let rec = |r: &mut Recorder| self.record_pass(r, n, pos, logits, &mut stages);
        let cmds = if self.profile { gpu.record_timed(rec)? } else { gpu.record(rec)? };
        gpu.submit(&cmds, self.spin)?;
        if self.profile {
            self.batch_profile = per_stage(&stages, &gpu.stamps(&cmds)?);
        }
        Ok(())
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
