// The products of the formats of a llama.cpp Q4_K_M file on HVX: Q5_0,
// Q8_0, Q4_K and Q6_K weights times Q8_0 activations (q4_0_rows's
// layout), for 1 to MAX_FEW tokens. Inline, like q4_0_hvx.h, so the
// simulator tests in dsp/sim/ can run them on the PC.
//
// Every format is laid out as Q4_0 is: groups of 32 rows, and per group
// each 32-value piece of the rows as whole vectors, word j of every
// vector for row j (written by the CPU, `repack_qx` in
// src/kernels/hexagon.rs). Slot k of a piece is four values: 4k .. 4k+3
// for k < 4, 16 + 4(k-4) .. for k >= 4, the values one vrmpy takes with
// the activation's word k. Per piece:
//
//   Q4_0  4 nibble vectors (byte i of slot k's word: low nibble slot k,
//         high nibble slot k+4, as Q4_0's own qs), scales           5
//   Q5_0  Q4_0's nibbles, one vector of fifth bits (bit 8i + k of row
//         j's word: byte i of slot k), scales; value q + 16h - 16   6
//   Q8_0  8 vectors of values + 128 as uint8 (vector k: slot k),
//         scales (there is no vrmpy of signed bytes by a scalar)     9
//   Q4_K  nibbles, d·scale, dmin·min (f32, exact); value
//         d·scale·q - dmin·min                                      6
//   Q6_K  8 vectors of int8 values q - 32, d·scale of values 0..15
//         and of 16..31 (each 16 values has its own scale, so no
//         offset by Σ x of the block: the activation's word is
//         splatted for the signed vector by vector vrmpy)          10
#ifndef AETHER_QX_HVX_H
#define AETHER_QX_HVX_H

#include "q4_0_hvx.h"

enum { F_Q4_0, F_Q5_0, F_Q8_0, F_Q4_K, F_Q6_K, N_FORMATS };

// Vectors per piece of a group, by format.
static inline uint32 piece_vectors(uint32 fmt) {
    static const uint8 v[N_FORMATS] = {5, 6, 9, 6, 10};
    return v[fmt];
}

// One token's activation block b: its int8 words, scale and sum.
#define X_BLOCK(xt, nb, b) \
    const int32 *xq = (const int32 *) (xt) + 8 * (b); \
    float xd = ((const float *) ((xt) + 32 * (nb)))[b]; \
    int32 xs = ((const int32 *) ((xt) + 36 * (nb)))[b]

// The fifth bits of slot k as 16 (or 0) in each byte, from the bits
// vector hb: bit 8i + k of a word moved to bit 8i + 4.
static inline HVX_Vector bit4(HVX_Vector hb, int k) {
    HVX_Vector s = k <= 4 ? Q6_Vw_vasl_VwR(hb, 4 - k) : Q6_Vuw_vlsr_VuwR(hb, k - 4);
    return Q6_V_vand_VV(s, Q6_V_vsplat_R(0x10101010));
}

static inline void group_rows_n_q5_0(const HVX_Vector *w, const char *const *x, uint32 n, uint32 nb, HVX_Vector *out, uint32 ostride, int accumulate) {
    const HVX_Vector low = Q6_Vb_vsplat_R(0x0f);
    HVX_Vector acc[MAX_FEW];
    for (uint32 t = 0; t < n; t++) acc[t] = Q6_V_vzero();
    for (uint32 b = 0; b < nb; b++, w += 6) {
        HVX_Vector v[8];
        for (int k = 0; k < 4; k++) {
            v[k] = Q6_V_vor_VV(Q6_V_vand_VV(w[k], low), bit4(w[4], k));
            v[4 + k] = Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(w[k], 4), low), bit4(w[4], 4 + k));
        }
        for (uint32 t = 0; t < n; t++) {
            X_BLOCK(x[t], nb, b);
            HVX_Vector s0 = Q6_V_vsplat_R(-16 * xs), s1 = Q6_V_vzero();
            for (int k = 0; k < 4; k++) {
                s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, v[k], xq[k]);
                s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, v[4 + k], xq[4 + k]);
            }
            acc[t] = block_acc(acc[t], Q6_Vw_vadd_VwVw(s0, s1), w[5], splat_f(xd));
        }
    }
    for (uint32 t = 0; t < n; t++) {
        HVX_Vector *o = out + (size_t) t * ostride;
        *o = accumulate ? Q6_Vsf_vadd_VsfVsf(*o, acc[t]) : acc[t];
    }
}

static inline void group_rows_n_q8_0(const HVX_Vector *w, const char *const *x, uint32 n, uint32 nb, HVX_Vector *out, uint32 ostride, int accumulate) {
    HVX_Vector acc[MAX_FEW];
    for (uint32 t = 0; t < n; t++) acc[t] = Q6_V_vzero();
    for (uint32 b = 0; b < nb; b++, w += 9) {
        for (uint32 t = 0; t < n; t++) {
            X_BLOCK(x[t], nb, b);
            HVX_Vector s0 = Q6_V_vsplat_R(-128 * xs), s1 = Q6_V_vzero();
            for (int k = 0; k < 4; k++) {
                s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, w[k], xq[k]);
                s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, w[4 + k], xq[4 + k]);
            }
            acc[t] = block_acc(acc[t], Q6_Vw_vadd_VwVw(s0, s1), w[8], splat_f(xd));
        }
    }
    for (uint32 t = 0; t < n; t++) {
        HVX_Vector *o = out + (size_t) t * ostride;
        *o = accumulate ? Q6_Vsf_vadd_VsfVsf(*o, acc[t]) : acc[t];
    }
}

static inline void group_rows_n_q4_k(const HVX_Vector *w, const char *const *x, uint32 n, uint32 nb, HVX_Vector *out, uint32 ostride, int accumulate) {
    const HVX_Vector low = Q6_Vb_vsplat_R(0x0f);
    HVX_Vector acc[MAX_FEW];
    for (uint32 t = 0; t < n; t++) acc[t] = Q6_V_vzero();
    for (uint32 b = 0; b < nb; b++, w += 6) {
        HVX_Vector lo[4], hi[4];
        for (int k = 0; k < 4; k++) {
            lo[k] = Q6_V_vand_VV(w[k], low);
            hi[k] = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(w[k], 4), low);
        }
        for (uint32 t = 0; t < n; t++) {
            X_BLOCK(x[t], nb, b);
            HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
            for (int k = 0; k < 4; k++) {
                s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, lo[k], xq[k]);
                s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, hi[k], xq[4 + k]);
            }
            // xd (d·scale·Σ q x - dmin·min·Σ x)
            HVX_Vector s = Q6_Vsf_equals_Vw(Q6_Vw_vadd_VwVw(s0, s1));
            HVX_Vector m = Q6_Vsf_vmpy_VsfVsf(w[5], Q6_Vsf_equals_Vw(Q6_V_vsplat_R(xs)));
            HVX_Vector r = Q6_Vsf_vsub_VsfVsf(Q6_Vsf_vmpy_VsfVsf(s, w[4]), m);
            acc[t] = Q6_Vsf_vadd_VsfVsf(acc[t], Q6_Vsf_vmpy_VsfVsf(r, splat_f(xd)));
        }
    }
    for (uint32 t = 0; t < n; t++) {
        HVX_Vector *o = out + (size_t) t * ostride;
        *o = accumulate ? Q6_Vsf_vadd_VsfVsf(*o, acc[t]) : acc[t];
    }
}

static inline void group_rows_n_q6_k(const HVX_Vector *w, const char *const *x, uint32 n, uint32 nb, HVX_Vector *out, uint32 ostride, int accumulate) {
    HVX_Vector acc[MAX_FEW];
    for (uint32 t = 0; t < n; t++) acc[t] = Q6_V_vzero();
    for (uint32 b = 0; b < nb; b++, w += 10) {
        for (uint32 t = 0; t < n; t++) {
            X_BLOCK(x[t], nb, b);
            (void) xs;
            // Values 0..15 (slots 0..3) and 16..31 (slots 4..7) have their
            // own scales.
            HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
            for (int k = 0; k < 4; k++) {
                s0 = Q6_Vw_vrmpyacc_VwVbVb(s0, w[k], Q6_V_vsplat_R(xq[k]));
                s1 = Q6_Vw_vrmpyacc_VwVbVb(s1, w[4 + k], Q6_V_vsplat_R(xq[4 + k]));
            }
            HVX_Vector r = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(s0), w[8]), Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(s1), w[9]));
            acc[t] = Q6_Vsf_vadd_VsfVsf(acc[t], Q6_Vsf_vmpy_VsfVsf(r, splat_f(xd)));
        }
    }
    for (uint32 t = 0; t < n; t++) {
        HVX_Vector *o = out + (size_t) t * ostride;
        *o = accumulate ? Q6_Vsf_vadd_VsfVsf(*o, acc[t]) : acc[t];
    }
}

#undef X_BLOCK

// Groups g0 .. g1 of W (format fmt) times n (at most MAX_FEW) tokens' x:
// token t's group g into out[t · ostride + g], the next group l2fetched
// while one is computed.
static inline void qx_rows_n_on(uint32 fmt, const HVX_Vector *wall, const char *const *x, uint32 n, uint32 nb, HVX_Vector *out, uint32 ostride, uint32 g0, uint32 g1, int accumulate) {
    const uint32 gv = piece_vectors(fmt) * nb;
    for (uint32 g = g0; g < g1; g++) {
        const HVX_Vector *w = wall + (size_t) g * gv;
        if (g + 1 < g1) prefetch(w + gv, gv * 128);
        switch (fmt) {
            case F_Q4_0: group_rows_n(w, x, n, nb, out + g, ostride, accumulate); break;
            case F_Q5_0: group_rows_n_q5_0(w, x, n, nb, out + g, ostride, accumulate); break;
            case F_Q8_0: group_rows_n_q8_0(w, x, n, nb, out + g, ostride, accumulate); break;
            case F_Q4_K: group_rows_n_q4_k(w, x, n, nb, out + g, ostride, accumulate); break;
            default: group_rows_n_q6_k(w, x, n, nb, out + g, ostride, accumulate); break;
        }
    }
}

#endif
