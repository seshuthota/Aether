// The Q4_0 x Q8_0 products on HVX, as inline functions: included by
// aether_dsp.c and by the simulator tests in dsp/sim/ (which run them on
// the PC with the SDK's hexagon-sim), so they need only the HVX
// intrinsics. The layouts are q4_0_rows's (dsp.h, aether_dsp.c).
#ifndef AETHER_Q4_0_HVX_H
#define AETHER_Q4_0_HVX_H

#include "AEEStdDef.h"
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

// Asks the L2 for `bytes` from p on (rows of 4 KB), without waiting.
static inline void prefetch(const void *p, uint32 bytes) {
    uint32 rows = (bytes + 4095) / 4096;
    Q6_l2fetch_AP((void *) p, (uint64) 4096 << 32 | 4096 << 16 | (rows > 0xffff ? 0xffff : rows));
}

static inline HVX_Vector splat_f(float f) {
    union { float f; int32 i; } u = {f};
    return Q6_V_vsplat_R(u.i);
}

// One block of one group: its rows' integer dot products, low and high
// nibbles in separate sums (two dependency chains, not one).
static inline HVX_Vector block_dot(const HVX_Vector *w, const int32 *x, int32 xsum, HVX_Vector low) {
    HVX_Vector s0 = Q6_V_vsplat_R(-8 * xsum), s1 = Q6_V_vzero();
    for (int k = 0; k < 4; k++) {
        s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, Q6_V_vand_VV(w[k], low), x[k]);
        s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(w[k], 4), low), x[4 + k]);
    }
    return Q6_Vw_vadd_VwVw(s0, s1);
}

// acc + (the block's sums) * (rows' scales) * (x's scale).
static inline HVX_Vector block_acc(HVX_Vector acc, HVX_Vector s, HVX_Vector scales, HVX_Vector xd) {
    return Q6_Vsf_vadd_VsfVsf(acc, Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(s), Q6_Vsf_vmpy_VsfVsf(scales, xd)));
}

// Two groups (w0, then w1 right after it) into out[0] and out[1]: the
// vector unit has independent work while each vrmpy's result is still on
// its way, and x is read once for both.
static inline void pair_rows(const HVX_Vector *w0, const char *xqb, uint32 nb, HVX_Vector *out, int accumulate) {
    const int32 *xq = (const int32 *) xqb;
    const float *xd = (const float *) (xqb + 32 * nb);
    const int32 *xs = (const int32 *) (xqb + 36 * nb);
    const HVX_Vector low = Q6_Vb_vsplat_R(0x0f);
    const HVX_Vector *w1 = w0 + (size_t) nb * 5;
    HVX_Vector a0 = Q6_V_vzero(), a1 = a0;
    for (uint32 b = 0; b < nb; b++, w0 += 5, w1 += 5) {
        const int32 *x = xq + 8 * b;
        HVX_Vector d = splat_f(xd[b]);
        HVX_Vector s0 = block_dot(w0, x, xs[b], low), s1 = block_dot(w1, x, xs[b], low);
        a0 = block_acc(a0, s0, w0[4], d);
        a1 = block_acc(a1, s1, w1[4], d);
    }
    out[0] = accumulate ? Q6_Vsf_vadd_VsfVsf(out[0], a0) : a0;
    out[1] = accumulate ? Q6_Vsf_vadd_VsfVsf(out[1], a1) : a1;
}

// Up to this many tokens in one q4_0_rows_n call.
#define MAX_FEW 8

// One group of 32 rows (w: its nb blocks of 5 vectors) times n tokens'
// Q8_0 x (x[t] in q4_0_rows's layout) into out[t · ostride] (vectors),
// added to what is there with `accumulate`. Each block's weights are split
// into nibbles once and used for every token: where reading the weights is
// the cost, n tokens cost about one.
static inline void group_rows_n(const HVX_Vector *w, const char *const *x, uint32 n, uint32 nb, HVX_Vector *out, uint32 ostride, int accumulate) {
    const HVX_Vector low = Q6_Vb_vsplat_R(0x0f);
    HVX_Vector acc[MAX_FEW];
    for (uint32 t = 0; t < n; t++) acc[t] = Q6_V_vzero();
    for (uint32 b = 0; b < nb; b++, w += 5) {
        HVX_Vector lo[4], hi[4];
        for (int k = 0; k < 4; k++) {
            lo[k] = Q6_V_vand_VV(w[k], low);
            hi[k] = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(w[k], 4), low);
        }
        for (uint32 t = 0; t < n; t++) {
            const int32 *xq = (const int32 *) x[t] + 8 * b;
            const float *xd = (const float *) (x[t] + 32 * nb);
            const int32 *xs = (const int32 *) (x[t] + 36 * nb);
            HVX_Vector s0 = Q6_V_vsplat_R(-8 * xs[b]), s1 = Q6_V_vzero();
            for (int k = 0; k < 4; k++) {
                s0 = Q6_Vw_vrmpyacc_VwVubRb(s0, lo[k], xq[k]);
                s1 = Q6_Vw_vrmpyacc_VwVubRb(s1, hi[k], xq[4 + k]);
            }
            acc[t] = block_acc(acc[t], Q6_Vw_vadd_VwVw(s0, s1), w[4], splat_f(xd[b]));
        }
    }
    for (uint32 t = 0; t < n; t++) {
        HVX_Vector *o = out + (size_t) t * ostride;
        *o = accumulate ? Q6_Vsf_vadd_VsfVsf(*o, acc[t]) : acc[t];
    }
}

// Groups g0 .. g1 of W times n (at most MAX_FEW) tokens' x: token t's
// group g into out[t · ostride + g], the next group l2fetched while one
// is computed.
static inline void q4_0_rows_n_on(const HVX_Vector *wall, const char *const *x, uint32 n, uint32 nb, HVX_Vector *out, uint32 ostride, uint32 g0, uint32 g1, int accumulate) {
    for (uint32 g = g0; g < g1; g++) {
        const HVX_Vector *w = wall + (size_t) g * nb * 5;
        if (g + 1 < g1) prefetch(w + 5 * nb, nb * 640);
        group_rows_n(w, x, n, nb, out + g, ostride, accumulate);
    }
}

#endif
