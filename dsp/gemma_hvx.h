// Gemma 4's element-wise steps on HVX (plans/gemma4-close.md, C3), as
// inline functions: included by model.c and by dsp/sim/gemma_ew.c, which
// runs them on the PC with the SDK's hexagon-sim against the numpy
// goldens of tools/gen_kernels.py (sim_gemma_elementwise_matches_numpy in
// src/kernels/hexagon.rs). Lengths are whole vectors (32 floats) unless
// a function says otherwise; pointers to vectors are 128-byte aligned.
#ifndef AETHER_GEMMA_HVX_H
#define AETHER_GEMMA_HVX_H

#include <math.h>

#include "vmath_hvx.h"

#define NV(n) (((n) + 31) / 32)

// GELU (tanh approximation) of g, times u: 0.5 g (1 + tanh z) u with
// z = √(2/π) (g + 0.044715 g³). 0.5 (1 + tanh z) = σ(2z) = 1 / (1 + e^-2z),
// the identity the CPU's NEON kernel uses, here on vexp and vrecip. The
// denominator is at least 1 and, with vexp's clamp at 80, below 1e35,
// which vrecip covers.
static inline HVX_Vector gelu_mul(HVX_Vector g, HVX_Vector u) {
    HVX_Vector g3 = VMUL(VMUL(g, g), g);
    HVX_Vector z2 = VMUL(splat_f(2.0f * 0.7978845608f), VADD(g, VMUL(splat_f(0.044715f), g3)));
    HVX_Vector s = vrecip(VADD(splat_f(1.0f), vexp(Q6_Vsf_vfneg_Vsf(z2))));
    return VMUL(VMUL(g, s), u);
}

// g[i] = gelu(g[i]) u[i] for n floats.
static inline void gelu_rows(float *g, const float *u, uint32 n) {
    HVX_Vector *gv = (HVX_Vector *) g;
    const HVX_Vector *uv = (const HVX_Vector *) u;
    for (uint32 i = 0; i < NV(n); i++) gv[i] = gelu_mul(gv[i], uv[i]);
}

// cap · tanh(x / cap). On |y| (y = x / cap): tanh |y| = (1 − e) / (1 + e)
// with e = e^-2|y| in (0, 1], so the reciprocal's argument stays in
// [1, 2] and nothing overflows; then x's sign goes back on.
static inline HVX_Vector softcap_v(HVX_Vector x, float cap) {
    HVX_Vector sign = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x80000000));
    HVX_Vector a = Q6_V_vand_VV(VMUL(x, splat_f(1.0f / cap)), Q6_V_vsplat_R(0x7fffffff));
    HVX_Vector e = vexp(VMUL(a, splat_f(-2.0f)));
    HVX_Vector t = VMUL(VSUB(splat_f(1.0f), e), vrecip(VADD(splat_f(1.0f), e)));
    return Q6_V_vor_VV(VMUL(t, splat_f(cap)), sign);
}

static inline void softcap_rows(float *x, uint32 n, float cap) {
    HVX_Vector *v = (HVX_Vector *) x;
    for (uint32 i = 0; i < NV(n); i++) v[i] = softcap_v(v[i], cap);
}

// 1 / √(mean(x²) + eps) over n floats (the padding adds zeros).
static inline float rms_inv(const float *x, uint32 n, float eps) {
    const HVX_Vector *v = (const HVX_Vector *) x;
    HVX_Vector acc = Q6_V_vzero();
    for (uint32 i = 0; i < NV(n); i++) acc = VADD(acc, VMUL(v[i], v[i]));
    return 1.0f / sqrtf(vsum(acc) / n + eps);
}

// o = x / rms(x), times w where w is not NULL (o may be x).
static inline void rmsnorm_hvx(float *o, const float *x, const float *w, uint32 n, float eps) {
    HVX_Vector inv = splat_f(rms_inv(x, n, eps));
    const HVX_Vector *xv = (const HVX_Vector *) x, *wv = (const HVX_Vector *) w;
    HVX_Vector *ov = (HVX_Vector *) o;
    for (uint32 i = 0; i < NV(n); i++) ov[i] = w ? VMUL(wv[i], VMUL(xv[i], inv)) : VMUL(xv[i], inv);
}

// One head of q, k or v after its product (hs floats), in
// place: RMSNorm with gain w (none when NULL: v), then RoPE on its first
// `pairs` pairs (i, i + hs / 2) by rot (sin, cos per pair), then × scale
// (q's √hs, for Gemma's attention scale of 1; else 1). pairs 0: no RoPE.
static inline void head_norm_rope(float *x, const float *w, uint32 hs, uint32 pairs, const float *rot, float scale, float eps) {
    rmsnorm_hvx(x, x, w, hs, eps);
    uint32 half = hs / 2;
    for (uint32 i = 0; i < pairs; i++) {
        float s = rot[2 * i], c = rot[2 * i + 1], a = x[i], b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
    if (scale != 1.0f) {
        HVX_Vector sv = splat_f(scale), *v = (HVX_Vector *) x;
        for (uint32 i = 0; i < NV(hs); i++) v[i] = VMUL(v[i], sv);
    }
}

// x = (x + w ⊙ rmsnorm(t)) · scale over n floats: Gemma's post-norm and
// residual add (scale is the per-layer output scale after the PLE block,
// else 1).
static inline void norm_add(float *x, const float *t, const float *w, uint32 n, float scale, float eps) {
    HVX_Vector inv = splat_f(rms_inv(t, n, eps)), sv = splat_f(scale);
    HVX_Vector *xv = (HVX_Vector *) x;
    const HVX_Vector *tv = (const HVX_Vector *) t, *wv = (const HVX_Vector *) w;
    for (uint32 i = 0; i < NV(n); i++) xv[i] = VMUL(VADD(xv[i], VMUL(wv[i], VMUL(tv[i], inv))), sv);
}

#endif
