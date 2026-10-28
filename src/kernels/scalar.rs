//! Straightforward scalar versions of every op in a decoder layer.
//!
//! These are the engine's first implementation, not a test reference: they
//! are checked against llama2.c's activations (see `model::transformer`
//! tests) and later replaced by NEON versions checked the same way.
//!
//! Order of operations follows llama2.c exactly (left-to-right sums,
//! `w * (scale * x)` in rmsnorm, division by sqrt(head_size) after the dot
//! product). Floating-point addition is not associative, so the same math in
//! a different order gives slightly different bits; matching the order makes
//! most activations bit-identical on the host, which makes real bugs stand
//! out against a zero background.

/// Matrix-vector product: `out = W · x`, with `W` stored row-major as
/// `out.len()` rows of `x.len()` columns.
///
/// This is decode's dominant cost: every weight is read once and used for
/// exactly one multiply-add, so it moves 4 bytes per 2 FLOPs.
#[inline(never)] // keep a separate symbol so profiles attribute time per kernel
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32]) {
    let n = x.len();
    assert_eq!(w.len(), out.len() * n, "W must be out.len() x x.len()");
    for (o, row) in out.iter_mut().zip(w.chunks_exact(n)) {
        let mut acc = 0.0f32;
        for (&wi, &xi) in row.iter().zip(x) {
            acc += wi * xi;
        }
        *o = acc;
    }
}

/// Batched [`matmul`]: `out[t] = W · x[t]` for `t` tokens, `x` `[t, n]`,
/// `out` `[t, d]`. The scalar version simply runs one matmul per token.
pub fn matmul_batch(out: &mut [f32], x: &[f32], w: &[f32], t: usize) {
    let (n, d) = (x.len() / t.max(1), out.len() / t.max(1));
    assert_eq!((x.len(), out.len()), (t * n, t * d));
    for (o, xi) in out.chunks_exact_mut(d).zip(x.chunks_exact(n)) {
        matmul(o, xi, w);
    }
}

/// RMSNorm: `out = weight * x / sqrt(mean(x²) + eps)`.
///
/// Unlike LayerNorm there is no mean subtraction: it only rescales the vector
/// to unit root-mean-square, then applies a learned per-channel gain. `eps`
/// keeps the division finite for an all-zero input.
#[inline(never)] // keep a separate symbol so profiles attribute time per kernel
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    assert_eq!(out.len(), x.len());
    assert_eq!(weight.len(), x.len());
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += eps;
    let scale = 1.0 / ss.sqrt();
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = w * (scale * v);
    }
}

/// RMSNorm in place (`x` is both input and output), as the final norm uses it.
pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(weight.len(), x.len());
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    ss /= x.len() as f32;
    ss += eps;
    let scale = 1.0 / ss.sqrt();
    for (v, &w) in x.iter_mut().zip(weight) {
        *v = w * (scale * *v);
    }
}

/// Softmax in place: `x_i = exp(x_i - max) / Σ exp(x_j - max)`.
///
/// Subtracting the max changes nothing mathematically (it cancels in the
/// ratio) but keeps every exponent ≤ 0, so `exp` cannot overflow to infinity.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Rotary position embedding, **adjacent-pair** layout (llama2.c, and GGUF
/// files of Llama-architecture models): within each head, elements
/// `(2i, 2i+1)` form a 2-D point rotated by angle `pos · θ^(−2i/head_size)`.
///
/// Each pair rotates at its own frequency: the first pairs spin fast (they
/// encode fine position), the last ones slowly (coarse position). The dot
/// product of a rotated q and k then depends only on their distance.
pub fn rope_adjacent(v: &mut [f32], pos: usize, head_size: usize, theta: f32) {
    assert_eq!(v.len() % head_size, 0);
    for head in v.chunks_exact_mut(head_size) {
        for i in (0..head_size).step_by(2) {
            let freq = 1.0 / theta.powf(i as f32 / head_size as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = (angle.sin(), angle.cos());
            let (v0, v1) = (head[i], head[i + 1]);
            head[i] = v0 * cos - v1 * sin;
            head[i + 1] = v0 * sin + v1 * cos;
        }
    }
}

/// Rotary position embedding, **halves** layout ("rotate_half", Hugging Face
/// transformers and Qwen2-style GGUF): within each head, element `i` is
/// paired with element `i + head_size/2`, and pair `i` rotates by angle
/// `pos · θ^(−2i/head_size)`. Same rotation as [`rope_adjacent`], applied to
/// a different pairing of the elements — mixing the two up still runs, and
/// produces garbage.
pub fn rope_halves(v: &mut [f32], pos: usize, head_size: usize, theta: f32) {
    assert_eq!(v.len() % head_size, 0);
    let half = head_size / 2;
    for head in v.chunks_exact_mut(head_size) {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_size as f32);
            let angle = pos as f32 * freq;
            let (sin, cos) = (angle.sin(), angle.cos());
            let (v0, v1) = (head[i], head[i + half]);
            head[i] = v0 * cos - v1 * sin;
            head[i + half] = v0 * sin + v1 * cos;
        }
    }
}

/// SwiGLU in place: `gate_i = silu(gate_i) * up_i`, where
/// `silu(x) = x · sigmoid(x) = x / (1 + e^(−x))`.
///
/// The MLP computes two projections of the same input; one, passed through
/// SiLU, decides per channel how much of the other gets through.
#[inline(never)] // keep a separate symbol so profiles attribute time per kernel
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    for (g, &u) in gate.iter_mut().zip(up) {
        let mut v = *g;
        v *= 1.0 / (1.0 + (-v).exp());
        v *= u;
        *g = v;
    }
}

/// Element-wise `x += y` (residual connections).
pub fn add_inplace(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for (a, &b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Attention of one query head over positions `0..=pos`.
///
/// - `q`: this head's query, `head_size` floats;
/// - `keys(t)` / `values(t)`: this head's key/value at position `t` (with
///   grouped-query attention several query heads read the same K/V head);
/// - `scores`: scratch space of at least `pos + 1` floats;
/// - `out`: the weighted sum of values, `head_size` floats.
///
/// score_t = q·k_t / sqrt(head_size); weights = softmax(scores);
/// out = Σ weight_t · v_t.
#[inline(never)] // keep a separate symbol so profiles attribute time per kernel
pub fn attention_head<'a>(
    out: &mut [f32],
    q: &[f32],
    keys: impl Fn(usize) -> &'a [f32],
    values: impl Fn(usize) -> &'a [f32],
    pos: usize,
    scores: &mut [f32],
) {
    let head_size = q.len();
    let scores = &mut scores[..=pos];
    let norm = (head_size as f32).sqrt();
    for (t, s) in scores.iter_mut().enumerate() {
        let k = keys(t);
        let mut dot = 0.0f32;
        for (&a, &b) in q.iter().zip(k) {
            dot += a * b;
        }
        *s = dot / norm;
    }
    softmax(scores);
    out.fill(0.0);
    for (t, &a) in scores.iter().enumerate() {
        for (o, &v) in out.iter_mut().zip(values(t)) {
            *o += a * v;
        }
    }
}

/// Index of the largest element (the first one, on ties).
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > x[best] {
            best = i;
        }
    }
    best
}
