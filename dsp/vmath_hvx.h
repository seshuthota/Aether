// HVX float helpers (IEEE single precision, Hexagon v73 and later):
// e^x, 1 / d, and a vector's sum and largest lane. Inline functions,
// included by model.c and by the simulator tests in dsp/sim/ (which run
// them on the PC with the SDK's hexagon-sim), so they need only the HVX
// intrinsics (splat_f is in q4_0_hvx.h).
#ifndef AETHER_VMATH_HVX_H
#define AETHER_VMATH_HVX_H

#include "q4_0_hvx.h"

#define VADD Q6_Vsf_vadd_VsfVsf
#define VSUB Q6_Vsf_vsub_VsfVsf
#define VMUL Q6_Vsf_vmpy_VsfVsf

// e^x in each lane, x clamped to [-80, 80] (enough for the softmax, whose
// exponents are at most 0, and for SwiGLU's e^-g). x = n ln2 + r with n
// an integer and |r| < ln2 (ln2 split in two, so n ln2 is exact enough);
// e^r by its Taylor series to r^8 (error below 1e-7 relative), times 2^n
// put straight into the exponent bits.
static inline HVX_Vector vexp(HVX_Vector x) {
    x = Q6_Vsf_vfmin_VsfVsf(Q6_Vsf_vfmax_VsfVsf(x, splat_f(-80.0f)), splat_f(80.0f));
    HVX_Vector n = Q6_Vw_equals_Vsf(VMUL(x, splat_f(1.44269504f)));
    HVX_Vector nf = Q6_Vsf_equals_Vw(n);
    HVX_Vector rr = VSUB(VSUB(x, VMUL(nf, splat_f(0.693145751953125f))), VMUL(nf, splat_f(1.428606765330187e-6f)));
    static const float c[9] = {1.0f, 1.0f, 1 / 2.0f, 1 / 6.0f, 1 / 24.0f, 1 / 120.0f, 1 / 720.0f, 1 / 5040.0f, 1 / 40320.0f};
    HVX_Vector p = splat_f(c[8]);
    for (int i = 7; i >= 0; i--) p = VADD(VMUL(p, rr), splat_f(c[i]));
    HVX_Vector two_n = Q6_Vw_vasl_VwR(Q6_Vw_vadd_VwVw(n, Q6_V_vsplat_R(127)), 23);
    return VMUL(p, two_n);
}

// 1 / d in each lane for d >= 1 (and below 1e35): a first guess from the
// bits, then three Newton steps y = y (2 - d y), each doubling the
// correct bits (about 4 to 30).
static inline HVX_Vector vrecip(HVX_Vector d) {
    HVX_Vector y = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x7EF311C7), d);
    for (int i = 0; i < 3; i++) y = VMUL(y, VSUB(splat_f(2.0f), VMUL(d, y)));
    return y;
}

// The sum of a vector's 32 lanes.
static inline float vsum(HVX_Vector v) {
    for (int b = 64; b >= 4; b >>= 1) v = VADD(v, Q6_V_vror_VR(v, b));
    union {
        HVX_Vector v;
        float f[32];
    } u = {v};
    return u.f[0];
}

// The largest of a vector's 32 lanes.
static inline float vmax_lanes(HVX_Vector v) {
    for (int b = 64; b >= 4; b >>= 1) v = Q6_Vsf_vfmax_VsfVsf(v, Q6_V_vror_VR(v, b));
    union {
        HVX_Vector v;
        float f[32];
    } u = {v};
    return u.f[0];
}

#endif
