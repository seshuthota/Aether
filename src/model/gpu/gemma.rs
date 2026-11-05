//! Gemma 4 on the GPU (`plans/gemma4-close.md` part B): the transformer of
//! [`crate::model::Transformer::forward_traced`] for models with per-layer
//! shapes, shared KV and per-layer embeddings, as Vulkan compute shaders.
//!
//! It is a separate path from [`super::GpuModel`] so that the tuned,
//! uniform-model path stays as it is (its fusions assume one head size, one
//! FFN width, SwiGLU and no extra norms). The same matrix layouts and
//! product shaders run here, unfused and in f32:
//!
//! - every matrix in `qx_gemv`'s layout ([`MatQ`]), any of Q4_0, Q5_0,
//!   Q8_0, Q4_K, Q6_K; decode runs `qx_gemv` (x in f32), a prompt `qx_gemm`;
//! - attention through `attention.comp` + `attention_merge.comp` for decode
//!   and prompt alike (one workgroup per head, 64-position chunk and token),
//!   with the sliding window as a start position on a full-length cache
//!   (no ring), and layers 15–34 reading the cache of their source layer;
//! - Gemma's own steps through `qkv_norm.comp`, `geglu.comp`,
//!   `norm_add.comp` and `softcap.comp`.
//!
//! The CPU does per token what it can do cheaply and the GPU cannot hold:
//! the embedding row (× √dim) and the per-layer inputs
//! ([`crate::model::transformer::ple_inputs`]: one row of the 1.3 GB table
//! and the BF16 projection), written into the shared buffers with the RoPE
//! tables, as the uniform path writes its embedding.
//!
//! Not here yet: int8 / f16 products, the few-token pass (lookup), the
//! hybrid CPU prefill.

use super::{AMAX_GROUPS, Mat, gemv_r, parts, per_stage, push};
use crate::kernels::vulkan::{Bound, Buffer, Commands, Gpu, Kernel, MatQ, Recorder, Result, groups_for};
use crate::model::Config;
use crate::model::transformer::Weights;
use crate::weight::Weight;
use std::ops::Range;

/// Bytes the attention's partial results may take; the prompt batch is as
/// many tokens as fit (at most 128), as for the uniform path's chunked
/// attention.
const PARTS_BUDGET: usize = 32 << 20;

/// One layer's buffers and bound kernels.
struct Layer {
    /// q (and k, v when the layer owns its cache) stacked, one part per
    /// format with the byte offset of its rows in `qkv`.
    qkv: Vec<(Mat, usize)>,
    wo: Mat,
    gate_up: Vec<(Mat, usize)>,
    down: Mat,
    inp_gate: Mat,
    proj: Mat,
    /// Row length of q/k/v in `qkv` (q only for a layer sharing K/V).
    qkv_dim: usize,
    hs: usize,
    hidden: usize,
    /// Float offset of the cache this layer reads (its own or its source's).
    cache_off: usize,
    own: bool,
    window: usize,
    /// This layer's rotation table: first pair in a token's tables, pairs.
    rot_off: usize,
    pairs: usize,
    out_scale: f32,
    // Bound kernels: decode (GEMV) and prompt (GEMM) products per part.
    norm1: Bound,
    qkv_v: Vec<Bound>,
    qkv_m: Vec<Bound>,
    post: Bound,
    attn: Bound,
    merge: Bound,
    wo_v: Bound,
    wo_m: Bound,
    add_attn: Bound,
    norm2: Bound,
    gate_up_v: Vec<Bound>,
    gate_up_m: Vec<Bound>,
    glu: Bound,
    down_v: Bound,
    down_m: Bound,
    add_ffn: Bound,
    pg_v: Bound,
    pg_m: Bound,
    ple_glu: Bound,
    proj_v: Bound,
    proj_m: Bound,
    add_ple: Bound,
}

pub struct GemmaGpu {
    pub config: Config,
    /// Tokens per [`GemmaGpu::forward_batch`].
    pub batch: usize,
    weights: Weights,
    x: Buffer,
    ple_in: Buffer,
    logits: Buffer,
    logits_all: Option<Buffer>,
    params: Buffer,
    best: Buffer,
    step: Commands,
    stages: Vec<&'static str>,
    profile: bool,
    pub batch_profile: Vec<(&'static str, f64)>,
    pub spin: bool,
    layers: Vec<Layer>,
    cls: Mat,
    fnorm: Bound,
    cls_v: Bound,
    cls_m: Option<Bound>,
    cap: Bound,
    cap_all: Option<Bound>,
    amax: Bound,
    /// Pairs of RoPE tables per token: every distinct (head size, θ) once.
    rot_stride: usize,
    /// The distinct tables: (head size, θ, first pair).
    rot_kinds: Vec<(usize, f32, usize)>,
    _keep: Vec<Buffer>,
    _kernels: Vec<Kernel>,
}

/// Every matrix in `qx_gemv`'s layout (Q4_0 too: the uniform path's Q4_0
/// layout is for its own shaders).
fn mat(gpu: &Gpu, ws: &[(&Weight, usize)], what: &str) -> Result<Mat> {
    super::stacked(ws, what, |w, rows| Ok(Mat::Qx(MatQ::new(gpu, w, rows)?)))
}

impl GemmaGpu {
    pub fn with(gpu: &Gpu, config: Config, mut weights: Weights, opt: super::Options) -> Result<GemmaGpu> {
        // No shader reads an F32 matrix: Q4_K_M's per-layer ones go to Q8_0.
        weights.ple_to_q8_0(&config);
        let c = &config;
        let dim = c.dim;
        if c.layers.iter().any(|l| matches!(l, crate::model::config::LayerKind::Conv { .. })) {
            return Err("LFM3's hybrid conv/attention stack is CPU-only; the GPU path is a later plan".into());
        }
        let ple = weights.ple.as_ref().ok_or("the GPU's per-layer path is for Gemma 4 (per-layer embeddings)")?;
        let (nl, pd) = (c.n_layers, c.ple_dim);
        let max_hs = (0..nl).map(|l| c.head_dim_of(l)).max().unwrap();
        let max_q = c.max_q_dim();
        let max_hid = c.max_hidden();
        let max_qkv = (0..nl).map(|l| c.q_dim(l) + if c.owns_kv(l) { 2 * c.kv_dim_of(l) } else { 0 }).max().unwrap();
        // Prompt batch: as many tokens as PARTS_BUDGET holds partial
        // results for (every head and 64-position chunk), at most 128.
        let chunks_all = c.seq_len.div_ceil(64);
        let parts_per_token = c.n_heads * chunks_all * (max_hs + 2);
        let batch = (PARTS_BUDGET / (4 * parts_per_token)).clamp(1, 128).min(c.seq_len);
        let _ = (ple, opt.repeat);

        let f32buf = |v: &[f32]| -> Result<Buffer> {
            let mut b = gpu.buffer(v.len().max(1) * 4)?;
            b.as_mut_slice::<f32>()[..v.len()].copy_from_slice(v);
            Ok(b)
        };
        let zeros = |n: usize| f32buf(&vec![0.0; n]);

        // Activations: one row per token of a batch; decode uses row 0.
        let x = zeros(batch * dim)?;
        let xb = zeros(batch * dim)?;
        let att = zeros(batch * max_q)?;
        let tmp = zeros(batch * dim)?;
        let qkv = zeros(batch * max_qkv)?;
        let hb12 = zeros(batch * 2 * max_hid)?;
        let hg = zeros(batch * max_hid)?;
        let (pg, pg2) = (zeros(batch * pd)?, zeros(batch * pd)?);
        let ple_in = zeros(batch * nl * pd)?;
        let cached = |n: usize| {
            use ash::vk::MemoryPropertyFlags as M;
            match gpu.buffer_with(n * 4, M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_COHERENT | M::HOST_CACHED) {
                Ok(b) => Ok(b),
                Err(_) => zeros(n),
            }
        };
        let logits = cached(c.vocab_size)?;
        let logits_all = if opt.all_logits { Some(cached(batch * c.vocab_size)?) } else { None };

        // The cache: full length for every layer that owns one (sliding
        // layers too: the window is a start position, not a ring).
        let mut cache_off = vec![0usize; nl];
        let mut total = 0;
        for l in 0..nl {
            if c.owns_kv(l) {
                cache_off[l] = total;
                total += c.seq_len * c.kv_dim_of(l);
            }
        }
        for l in 0..nl {
            cache_off[l] = cache_off[c.kv_source_of(l)];
        }
        let (kc, vc) = (zeros(total)?, zeros(total)?);
        let parts_buf = zeros(batch * parts_per_token)?;

        // RoPE tables per token: one per distinct (head size, θ).
        let mut rot_kinds: Vec<(usize, f32, usize)> = Vec::new();
        let mut rot_stride = 0;
        for l in 0..nl {
            let (hs, theta) = (c.head_dim_of(l), c.rope(l).theta);
            if !rot_kinds.iter().any(|&(h, t, _)| h == hs && t == theta) {
                rot_kinds.push((hs, theta, rot_stride));
                rot_stride += hs / 2;
            }
        }
        let params = zeros(4 + batch * 2 * rot_stride)?;

        let rms = Kernel::new(gpu, crate::spirv!("rmsnorm"), 3, 12)?;
        let post = Kernel::new(gpu, crate::spirv!("qkv_norm"), 6, 48)?;
        let attn = Kernel::new(gpu, crate::spirv!("attention"), 5, 28)?;
        let merge = Kernel::new(gpu, crate::spirv!("attention_merge"), 3, 12)?;
        let geglu = Kernel::new(gpu, crate::spirv!("geglu"), 3, 16)?;
        let norm_add = Kernel::new(gpu, crate::spirv!("norm_add"), 3, 12)?;
        let softcap = Kernel::new(gpu, crate::spirv!("softcap"), 1, 8)?;
        let argmax = Kernel::new(gpu, crate::spirv!("argmax"), 3, 8)?;

        let mut gemv_ks: Vec<((u32, u32), Kernel)> = Vec::new();
        let mut gemv = |m: &Mat, x_in: &Buffer, out: (&Buffer, usize)| -> Result<Bound> {
            let key = (m.ty(), gemv_r(m.rows()));
            let i = match gemv_ks.iter().position(|(k, _)| *k == key) {
                Some(i) => i,
                None => {
                    gemv_ks.push((key, gpu.qx_gemv_kernel(key.0, key.1, false, false, false)?));
                    gemv_ks.len() - 1
                }
            };
            let [q, d] = m.bufs();
            gemv_ks[i].1.bind_at(gpu, &[(q, 0), (d, 0), (x_in, 0), out, (x_in, 0)])
        };
        let mut gemm_ks: Vec<(u32, Kernel)> = Vec::new();
        let mut gemm = |m: &Mat, x_in: &Buffer, out: (&Buffer, usize)| -> Result<Bound> {
            let i = match gemm_ks.iter().position(|(k, _)| *k == m.ty()) {
                Some(i) => i,
                None => {
                    gemm_ks.push((m.ty(), gpu.qx_gemm_kernel(m.ty(), false, false)?));
                    gemm_ks.len() - 1
                }
            };
            let [q, d] = m.bufs();
            gemm_ks[i].1.bind_at(gpu, &[(q, 0), (d, 0), (x_in, 0), out])
        };

        let mut layers = Vec::new();
        let mut keep_norms = Vec::new();
        for (l, lw) in weights.layers.iter().enumerate() {
            let what = |s: &str| format!("layer {l} {s}");
            let a = lw.attention().ok_or_else(|| what("is not attention"))?;
            let (hs, qd, kvd, hid, own) = (c.head_dim_of(l), c.q_dim(l), c.kv_dim_of(l), c.hidden_of(l), c.owns_kv(l));
            let mut ws = vec![(&a.wq, qd)];
            if own {
                let (wk, wv) = a.kv().ok_or_else(|| what("owns its cache but has no K/V"))?;
                ws.extend([(wk, kvd), (wv, kvd)]);
            }
            let qkv_parts = parts(gpu, &ws, &what("wq/wk/wv"), |ws| mat(gpu, ws, "wq/wk/wv"))?;
            let gate_up = parts(gpu, &[(&lw.w_gate, hid), (&lw.w_up, hid)], &what("w_gate/w_up"), |ws| mat(gpu, ws, "w_gate/w_up"))?;
            let p = lw.ple.as_ref().ok_or_else(|| what("has no per-layer block"))?;
            let need = |n: &Option<crate::aligned::AlignedBuf>, s: &str| n.as_ref().ok_or_else(|| what(s)).and_then(|b| f32buf(b));
            let (attn_norm, ffn_norm) = (f32buf(&lw.attn_norm)?, f32buf(&lw.ffn_norm)?);
            let post_attn = need(&lw.post_attn_norm, "has no post-attention norm")?;
            let post_ffn = need(&lw.post_ffn_norm, "has no post-FFN norm")?;
            let post_ple = f32buf(&p.post_norm)?;
            let q_norm = need(&a.q_norm, "has no q norm")?;
            // A layer sharing K/V has no k norm (and qkv_norm does not read it).
            let k_norm = match &a.k_norm {
                Some(n) => f32buf(n)?,
                None => zeros(1)?,
            };
            let (wo, down) = (mat(gpu, &[(&a.wo, dim)], &what("wo"))?, mat(gpu, &[(&lw.w_down, dim)], &what("w_down"))?);
            let (inp_gate, proj) = (mat(gpu, &[(&p.inp_gate, pd)], &what("inp_gate"))?, mat(gpu, &[(&p.proj, dim)], &what("proj"))?);
            let rope = c.rope(l);
            let rot_off = rot_kinds.iter().find(|&&(h, t, _)| h == hs && t == rope.theta).unwrap().2;
            let qkv_dim = qd + if own { 2 * kvd } else { 0 };
            let ly = Layer {
                norm1: rms.bind(gpu, &[&x, &attn_norm, &xb])?,
                qkv_v: qkv_parts.iter().map(|(m, at)| gemv(m, &xb, (&qkv, *at))).collect::<Result<_>>()?,
                qkv_m: qkv_parts.iter().map(|(m, at)| gemm(m, &xb, (&qkv, *at))).collect::<Result<_>>()?,
                post: post.bind(gpu, &[&qkv, &q_norm, &k_norm, &kc, &vc, &params])?,
                attn: attn.bind(gpu, &[&qkv, &kc, &vc, &parts_buf, &params])?,
                merge: merge.bind(gpu, &[&parts_buf, &att, &params])?,
                wo_v: gemv(&wo, &att, (&tmp, 0))?,
                wo_m: gemm(&wo, &att, (&tmp, 0))?,
                add_attn: norm_add.bind(gpu, &[&x, &tmp, &post_attn])?,
                norm2: rms.bind(gpu, &[&x, &ffn_norm, &xb])?,
                gate_up_v: gate_up.iter().map(|(m, at)| gemv(m, &xb, (&hb12, *at))).collect::<Result<_>>()?,
                gate_up_m: gate_up.iter().map(|(m, at)| gemm(m, &xb, (&hb12, *at))).collect::<Result<_>>()?,
                glu: geglu.bind_at(gpu, &[(&hb12, 0), (&hb12, 4 * hid), (&hg, 0)])?,
                down_v: gemv(&down, &hg, (&tmp, 0))?,
                down_m: gemm(&down, &hg, (&tmp, 0))?,
                add_ffn: norm_add.bind(gpu, &[&x, &tmp, &post_ffn])?,
                pg_v: gemv(&inp_gate, &x, (&pg, 0))?,
                pg_m: gemm(&inp_gate, &x, (&pg, 0))?,
                ple_glu: geglu.bind_at(gpu, &[(&pg, 0), (&ple_in, 4 * l * pd), (&pg2, 0)]).map_err(|e| format!("{}: {e}", what("per-layer input offset")))?,
                proj_v: gemv(&proj, &pg2, (&tmp, 0))?,
                proj_m: gemm(&proj, &pg2, (&tmp, 0))?,
                add_ple: norm_add.bind(gpu, &[&x, &tmp, &post_ple])?,
                qkv: qkv_parts,
                wo,
                gate_up,
                down,
                inp_gate,
                proj,
                qkv_dim,
                hs,
                hidden: hid,
                cache_off: cache_off[l],
                own,
                window: c.window(l).unwrap_or(0),
                rot_off,
                pairs: rope.pairs,
                out_scale: lw.out_scale.unwrap_or(1.0),
            };
            keep_norms.extend([attn_norm, ffn_norm, post_attn, post_ffn, post_ple, q_norm, k_norm]);
            layers.push(ly);
        }
        let cls = mat(gpu, &[(weights.classifier.as_ref().unwrap_or(&weights.token_embedding), c.vocab_size)], "classifier")?;
        let final_norm = f32buf(&weights.final_norm)?;
        let fnorm = rms.bind(gpu, &[&x, &final_norm, &xb])?;
        let cls_v = gemv(&cls, &xb, (&logits, 0))?;
        let cls_m = match &logits_all {
            Some(la) => Some(gemm(&cls, &xb, (la, 0))?),
            None => None,
        };
        let cap = softcap.bind(gpu, &[&logits])?;
        let cap_all = match &logits_all {
            Some(la) => Some(softcap.bind(gpu, &[la])?),
            None => None,
        };
        let (amax_parts, best) = (zeros(2 * AMAX_GROUPS)?, cached(1)?);
        let amax = argmax.bind(gpu, &[&logits, &amax_parts, &best])?;

        let mut keep = vec![xb, att, tmp, qkv, hb12, hg, pg, pg2, kc, vc, parts_buf, final_norm, amax_parts];
        keep.extend(keep_norms);
        let mut kernels = vec![rms, post, attn, merge, geglu, norm_add, softcap, argmax];
        kernels.extend(gemv_ks.into_iter().map(|(_, k)| k));
        kernels.extend(gemm_ks.into_iter().map(|(_, k)| k));
        let mut m = GemmaGpu {
            config,
            batch,
            weights,
            x,
            ple_in,
            logits,
            logits_all,
            params,
            best,
            step: gpu.record(|_| {})?,
            stages: Vec::new(),
            profile: opt.profile,
            batch_profile: Vec::new(),
            spin: false,
            layers,
            cls,
            fnorm,
            cls_v,
            cls_m,
            cap,
            cap_all,
            amax,
            rot_stride,
            rot_kinds,
            _keep: keep,
            _kernels: kernels,
        };
        let mut stages = Vec::new();
        let step = if opt.profile { gpu.record_timed(|r| m.record_pass(r, 1, 0, false, 0..nl, true, &mut stages))? } else { gpu.record(|r| m.record_pass(r, 1, 0, false, 0..nl, true, &mut stages))? };
        if !opt.profile {
            stages.clear();
        }
        m.step = step;
        m.stages = stages;
        Ok(m)
    }

    /// Records one forward pass over `n` tokens, the first at position
    /// `pos0` (n = 1: the decode step, recorded once for every position: its
    /// attention covers every chunk the cache can hold). `all`: every
    /// token's logits into `logits_all`, else the last token's into
    /// `logits` and their argmax. Only the layers in `layers`; the final
    /// norm and the classifier when `head`.
    fn record_pass(&self, r: &mut Recorder, n: usize, pos0: usize, all: bool, layers: Range<usize>, head: bool, stages: &mut Vec<&'static str>) {
        let c = &self.config;
        let (dim, pd) = (c.dim, c.ple_dim);
        let eps = c.norm_eps;
        let nt = n as u32;
        let chunks = if n == 1 { c.seq_len.div_ceil(64) } else { (pos0 + n).div_ceil(64) } as u32;
        let mut end = |r: &mut Recorder, name: &'static str| {
            r.barrier();
            r.stamp();
            stages.push(name);
        };
        // A product: GEMV for one token, GEMM for several (`stride`: the
        // length of a token's output row when the matrix is one part of it).
        let product = |r: &mut Recorder, v: &[Bound], g: &[Bound], ms: &[&Mat], stride: usize| {
            if n == 1 {
                for (k, m) in v.iter().zip(ms) {
                    r.dispatch(k, &push(&[m.rows() as u32, m.nb() as u32, eps.to_bits()]), groups_for(m.rows(), 4 * gemv_r(m.rows()) as usize));
                }
            } else {
                for (k, m) in g.iter().zip(ms) {
                    r.dispatch(k, &push(&[m.rows() as u32, m.nb() as u32, nt, stride as u32]), [m.rows().div_ceil(64) as u32, n.div_ceil(64) as u32, 1]);
                }
            }
        };
        let fpush = |u: &[u32], f: &[f32]| -> Vec<u8> {
            let mut p = push(u);
            for v in f {
                p.extend(v.to_le_bytes());
            }
            p
        };
        let norm = |r: &mut Recorder, b: &Bound| r.dispatch(b, &fpush(&[dim as u32], &[eps]).into_iter().chain(0u32.to_le_bytes()).collect::<Vec<u8>>(), [nt, 1, 1]);
        let add = |r: &mut Recorder, b: &Bound, scale: f32| r.dispatch(b, &fpush(&[dim as u32], &[eps, scale]), [nt, 1, 1]);
        fn mats(p: &[(Mat, usize)]) -> Vec<&Mat> {
            p.iter().map(|(m, _)| m).collect()
        }
        for ly in &self.layers[layers.clone()] {
            norm(r, &ly.norm1);
            end(r, "rmsnorm");
            product(r, &ly.qkv_v, &ly.qkv_m, &mats(&ly.qkv), ly.qkv_dim);
            end(r, "q/k/v");
            let qscale = c.attn_scale.map_or(1.0, |s| s * (ly.hs as f32).sqrt());
            let heads = c.n_heads + if ly.own { 2 * c.n_kv_heads } else { 0 };
            r.dispatch(
                &ly.post,
                &fpush(&[ly.hs as u32, c.n_heads as u32, c.n_kv_heads as u32, ly.qkv_dim as u32, ly.cache_off as u32, ly.rot_off as u32, self.rot_stride as u32, ly.pairs as u32, ly.own.into(), c.v_norm.into()], &[qscale, eps]),
                [nt, heads as u32, 1],
            );
            end(r, "q/k/v norm + RoPE + cache");
            r.dispatch(&ly.attn, &push(&[c.n_heads as u32, c.n_kv_heads as u32, ly.hs as u32, chunks, ly.cache_off as u32, ly.qkv_dim as u32, ly.window as u32]), [c.n_heads as u32, chunks, nt]);
            end(r, "attention");
            r.dispatch(&ly.merge, &push(&[ly.hs as u32, chunks, ly.window as u32]), [c.n_heads as u32, nt, 1]);
            end(r, "attention merge");
            product(r, std::slice::from_ref(&ly.wo_v), std::slice::from_ref(&ly.wo_m), &[&ly.wo], dim);
            end(r, "wo");
            add(r, &ly.add_attn, 1.0);
            end(r, "post-attention norm + add");
            norm(r, &ly.norm2);
            end(r, "rmsnorm");
            product(r, &ly.gate_up_v, &ly.gate_up_m, &mats(&ly.gate_up), 2 * ly.hidden);
            end(r, "gate/up");
            let h = ly.hidden as u32;
            r.dispatch(&ly.glu, &push(&[h, 2 * h, 2 * h, h]), [h.div_ceil(256), nt, 1]);
            end(r, "gelu");
            product(r, std::slice::from_ref(&ly.down_v), std::slice::from_ref(&ly.down_m), &[&ly.down], dim);
            end(r, "down");
            add(r, &ly.add_ffn, 1.0);
            end(r, "post-FFN norm + add");
            product(r, std::slice::from_ref(&ly.pg_v), std::slice::from_ref(&ly.pg_m), &[&ly.inp_gate], pd);
            end(r, "per-layer gate");
            let p = pd as u32;
            r.dispatch(&ly.ple_glu, &push(&[p, p, (c.n_layers * pd) as u32, p]), [p.div_ceil(256), nt, 1]);
            end(r, "per-layer gelu");
            product(r, std::slice::from_ref(&ly.proj_v), std::slice::from_ref(&ly.proj_m), &[&ly.proj], dim);
            end(r, "per-layer proj");
            add(r, &ly.add_ple, ly.out_scale);
            end(r, "per-layer norm + add + scale");
        }
        if !head {
            return;
        }
        let v = c.vocab_size as u32;
        let cap = c.logit_softcap;
        match (&self.cls_m, all) {
            (Some(cm), true) => {
                norm(r, &self.fnorm);
                end(r, "rmsnorm");
                r.dispatch(cm, &push(&[self.cls.rows() as u32, self.cls.nb() as u32, nt, self.cls.rows() as u32]), [self.cls.rows().div_ceil(64) as u32, n.div_ceil(64) as u32, 1]);
                end(r, "classifier");
                if let (Some(cap), Some(b)) = (cap, &self.cap_all) {
                    r.dispatch(b, &fpush(&[nt * v], &[cap]), groups_for(n * c.vocab_size, 256));
                    end(r, "soft-cap");
                }
            }
            _ => {
                // The last token's row of x, normalized into row 0.
                r.dispatch(&self.fnorm, &fpush(&[dim as u32], &[eps]).into_iter().chain((nt - 1).to_le_bytes()).collect::<Vec<u8>>(), [1, 1, 1]);
                end(r, "rmsnorm");
                let m = &self.cls;
                r.dispatch(&self.cls_v, &push(&[m.rows() as u32, m.nb() as u32, eps.to_bits()]), groups_for(m.rows(), 4 * gemv_r(m.rows()) as usize));
                end(r, "classifier");
                if let Some(cap) = cap {
                    r.dispatch(&self.cap, &fpush(&[v], &[cap]), groups_for(c.vocab_size, 256));
                    end(r, "soft-cap");
                }
                r.dispatch(&self.amax, &push(&[v, 0]), [AMAX_GROUPS as u32, 1, 1]);
                r.barrier();
                r.dispatch(&self.amax, &push(&[AMAX_GROUPS as u32, 1]), [1, 1, 1]);
                end(r, "argmax");
            }
        }
    }

    /// Writes token `token`'s scaled embedding into row `row` of x, its
    /// per-layer inputs into row `row` of `ple_in`, and its position's RoPE
    /// tables into `params`.
    fn set_token(&mut self, row: usize, token: usize, pos: usize) {
        let c = &self.config;
        assert!(pos < c.seq_len && token < c.vocab_size);
        let dim = c.dim;
        let mut e = vec![0f32; dim];
        self.weights.token_embedding.row_f32(token, &mut e);
        if c.embed_scale != 1.0 {
            crate::kernels::scale_inplace(&mut e, c.embed_scale);
        }
        self.x.as_mut_slice::<f32>()[row * dim..(row + 1) * dim].copy_from_slice(&e);
        let n = c.n_layers * c.ple_dim;
        let p = self.weights.ple.as_ref().unwrap();
        crate::model::transformer::ple_inputs(None, c, p, token, &e, &mut self.ple_in.as_mut_slice::<f32>()[row * n..(row + 1) * n]);
        let params = self.params.as_mut_slice::<u32>();
        for &(hs, theta, off) in &self.rot_kinds {
            let base = 4 + 2 * (row * self.rot_stride + off);
            for (i, (sin, cos)) in crate::kernels::scalar::rotations(pos, hs, theta).into_iter().enumerate() {
                params[base + 2 * i] = sin.to_bits();
                params[base + 2 * i + 1] = cos.to_bits();
            }
        }
    }

    /// Runs `token` at position `pos` and returns the logits for the next
    /// token. Positions must be fed in order.
    pub fn forward(&mut self, gpu: &Gpu, token: usize, pos: usize) -> Result<&[f32]> {
        self.set_token(0, token, pos);
        self.params.as_mut_slice::<u32>()[0] = pos as u32;
        gpu.submit(&self.step, self.spin)?;
        Ok(&self.logits.as_slice::<f32>()[..self.config.vocab_size])
    }

    /// The most likely next token after the last pass that wrote only the
    /// last token's logits.
    pub fn greedy(&self) -> usize {
        self.best.as_slice::<u32>()[0] as usize
    }

    /// Runs `tokens` (at most [`GemmaGpu::batch`]) at positions `pos ..` in
    /// one pass and returns the last token's logits, or with
    /// `Options::all_logits` every token's.
    pub fn forward_batch(&mut self, gpu: &Gpu, tokens: &[usize], pos: usize) -> Result<&[f32]> {
        let n = tokens.len();
        assert!(n > 0 && n <= self.batch && pos + n <= self.config.seq_len, "{n} tokens at {pos}: batch {}, seq_len {}", self.batch, self.config.seq_len);
        for (i, &t) in tokens.iter().enumerate() {
            self.set_token(i, t, pos + i);
        }
        self.params.as_mut_slice::<u32>()[0] = pos as u32;
        let all = self.logits_all.is_some();
        // One submission per layer, then one for the classifier: a whole
        // prompt pass in one command buffer outlasts the driver's watchdog
        // (on the PC's Intel GPU, i915 stopped it partway through layer 25
        // without an error: "Fence expiration time out" in the kernel log,
        // and the later rows of that product kept the previous layer's
        // values).
        let nl = self.layers.len();
        let mut profile: Vec<(&'static str, f64)> = Vec::new();
        for l in 0..=nl {
            let mut stages = Vec::new();
            let rec = |r: &mut Recorder| self.record_pass(r, n, pos, all, l.min(nl)..(l + 1).min(nl), l == nl, &mut stages);
            let cmds = if self.profile { gpu.record_timed(rec)? } else { gpu.record(rec)? };
            gpu.submit(&cmds, self.spin)?;
            if self.profile {
                for (name, t) in per_stage(&stages, &gpu.stamps(&cmds)?) {
                    match profile.iter_mut().find(|(n, _)| *n == name) {
                        Some(e) => e.1 += t,
                        None => profile.push((name, t)),
                    }
                }
            }
        }
        if self.profile {
            self.batch_profile = profile;
        }
        let v = self.config.vocab_size;
        Ok(match &self.logits_all {
            Some(l) => &l.as_slice::<f32>()[..n * v],
            None => &self.logits.as_slice::<f32>()[..v],
        })
    }

    /// GPU seconds of the last [`GemmaGpu::forward`] per kind of stage;
    /// empty unless made with `profile`.
    pub fn profile(&self, gpu: &Gpu) -> Result<Vec<(&'static str, f64)>> {
        Ok(per_stage(&self.stages, &gpu.stamps(&self.step)?))
    }
}
