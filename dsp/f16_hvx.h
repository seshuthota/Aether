// IEEE f16 on HVX, for the K/V cache. Measured on hexagon-sim (v79),
// against numpy astype(float16):
//
// Q6_Wsf_vcvt_Vhf splits one vector of 64 halves into even lanes (the
// low f32 vector) and odd lanes (the high one), not into 0..31 and
// 32..63. Q6_Vhf_vcvt_VsfVsf interleaves the other way: the first f32
// vector lands in the even halves. A word vshuff / vdeal undoes it.
// Rounding is nearest, ties to even. The one miss against numpy is
// −0, which this convert stores as +0.
#ifndef AETHER_F16_HVX_H
#define AETHER_F16_HVX_H

#include "AEEStdDef.h"
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

// Even and odd lanes of one natural-order f16 vector, as f32.
static inline void hf_even_odd(HVX_Vector h, HVX_Vector *even, HVX_Vector *odd) {
    HVX_VectorPair sp = Q6_Wsf_vcvt_Vhf(h);
    *even = Q6_V_lo_W(sp);
    *odd = Q6_V_hi_W(sp);
}

// Even/odd f32 lanes (as hf_even_odd returns them) back into 0..31 and 32..63.
static inline void even_odd_to_halves(HVX_Vector even, HVX_Vector odd, HVX_Vector *lo, HVX_Vector *hi) {
    HVX_VectorPair sh = Q6_W_vshuff_VVR(odd, even, -4);
    *lo = Q6_V_lo_W(sh);
    *hi = Q6_V_hi_W(sh);
}

// One natural-order f16 vector as f32 lanes 0..31 and 32..63.
static inline void hf_to_f32(HVX_Vector h, HVX_Vector *lo, HVX_Vector *hi) {
    HVX_Vector even, odd;
    hf_even_odd(h, &even, &odd);
    even_odd_to_halves(even, odd, lo, hi);
}

// f32 lanes 0..31 and 32..63 as one natural-order f16 vector.
static inline HVX_Vector f32_to_hf(HVX_Vector lo, HVX_Vector hi) {
    HVX_VectorPair d = Q6_W_vdeal_VVR(hi, lo, -4);
    return Q6_Vhf_vcvt_VsfVsf(Q6_V_lo_W(d), Q6_V_hi_W(d));
}

// One f32, the same convert qkv_post uses for a single cache element.
// Lane 0 of the even half is lane 0 of the first vector.
static inline uint16 f32_to_f16_bits(float x) {
    union { float f; int32 i; } u = {x};
    union { HVX_Vector v; uint16 h[64]; } hf;
    hf.v = Q6_Vhf_vcvt_VsfVsf(Q6_V_vsplat_R(u.i), Q6_V_vzero());
    return hf.h[0];
}

#endif
