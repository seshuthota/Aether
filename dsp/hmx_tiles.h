// Dequantize one 32-row group into HMX weight tiles. The tile layout is
// hmx.c's: 16 HVX vectors, vector i holds inputs 2i and 2i+1, and lane n
// is output row n (one fp16 pair). Every format uses the same layout, so
// the matrix unit does not care which quantizer the weights came from.
//
// The bytes are our repacked groups (dsp/qx_hvx.h). Q4_0's path is the
// one hmx.c used to inline: nibbles, subtract 8, times the row's scale.
// Q5_0 adds the fifth bit and subtracts 16. Q8_0 subtracts 128 from the
// stored byte. Q4_K's nibbles are the code q itself, and the f32 vectors
// are already d·scale and dmin·min, so the tile holds q·(d·scale) −
// (dmin·min). Q6_K's bytes are the signed code q−32, times d·scale of
// inputs 0..15 or 16..31.
#ifndef AETHER_HMX_TILES_H
#define AETHER_HMX_TILES_H

#include "qx_hvx.h"

// f32 vectors a (the even input of a pair) and b (the odd one) as one
// fp16 vector. A scale used for both inputs of a lane passes the same
// vector twice.
static inline HVX_Vector to_f16_pairs(HVX_Vector a, HVX_Vector b) {
    HVX_Vector z = Q6_V_vzero();
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vadd_VsfVsf(b, z), Q6_Vqf32_vadd_VsfVsf(a, z)));
}

// The two halfwords of each lane (low input, high input) times the row's
// f32 scale, minus `offset` when `has_offset` is set, then rounded to
// fp16. Doing the multiply in f32 matters for Q4_K: the scale and the
// offset are large and nearly cancel, and an f16 multiply flushes the
// difference to zero.
static inline HVX_Vector pair_f32(HVX_Vector p, HVX_Vector scale, HVX_Vector offset, int has_offset) {
    HVX_Vector lo = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(p, 16), 16));
    HVX_Vector hi = Q6_Vsf_equals_Vw(Q6_Vw_vasr_VwR(p, 16));
    lo = Q6_Vsf_vmpy_VsfVsf(lo, scale);
    hi = Q6_Vsf_vmpy_VsfVsf(hi, scale);
    if (has_offset) {
        lo = Q6_Vsf_vsub_VsfVsf(lo, offset);
        hi = Q6_Vsf_vsub_VsfVsf(hi, offset);
    }
    return to_f16_pairs(lo, hi);
}

// One quant vector's four bytes per lane into the two tile vectors of
// that slot: bytes 0, 1 are inputs 4k and 4k+1, bytes 2, 3 are 4k+2 and
// 4k+3. `bias` is subtracted before the f16 scale (8, 16, 128 or 0).
// `extend` sign-extends each byte, for Q6_K's int8 codes.
static inline void tile_pair(HVX_Vector q, HVX_Vector scale, int bias, int extend, HVX_Vector *t0, HVX_Vector *t1) {
    const HVX_Vector b0 = Q6_V_vsplat_R(0xff), b1 = Q6_V_vsplat_R(0xff0000);
    HVX_Vector p01 = Q6_V_vor_VV(Q6_V_vand_VV(q, b0), Q6_V_vand_VV(Q6_Vw_vasl_VwR(q, 8), b1));
    HVX_Vector p23 = Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(q, 16), b0), Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(q, 8), b1));
    if (extend) {
        p01 = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(p01, 8), 8);
        p23 = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(p23, 8), 8);
    }
    HVX_Vector off = Q6_Vh_vsplat_R(bias);
    *t0 = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_Vh_vsub_VhVh(p01, off)), scale);
    *t1 = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_Vh_vsub_VhVh(p23, off)), scale);
}

// Group g's nb blocks, `piece_vectors(fmt)` vectors each, into nb tiles.
static inline void hmx_weight_tiles(HVX_Vector *t, const HVX_Vector *w, uint32 nb, uint32 fmt) {
    const HVX_Vector nib = Q6_Vb_vsplat_R(0x0f);
    for (uint32 b = 0; b < nb; b++, t += 16) {
        if (fmt == F_Q8_0) {
            HVX_Vector s = to_f16_pairs(w[8], w[8]);
            for (int k = 0; k < 8; k++) tile_pair(w[k], s, 128, 0, &t[2 * k], &t[2 * k + 1]);
            w += 9;
        } else if (fmt == F_Q6_K) {
            const HVX_Vector b0 = Q6_V_vsplat_R(0xff), b1 = Q6_V_vsplat_R(0xff0000);
            for (int k = 0; k < 8; k++) {
                HVX_Vector q = w[k];
                HVX_Vector p01 = Q6_V_vor_VV(Q6_V_vand_VV(q, b0), Q6_V_vand_VV(Q6_Vw_vasl_VwR(q, 8), b1));
                HVX_Vector p23 = Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(q, 16), b0), Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(q, 8), b1));
                // The byte is the signed code q − 32.
                p01 = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(p01, 8), 8);
                p23 = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(p23, 8), 8);
                HVX_Vector s = k < 4 ? w[8] : w[9];
                t[2 * k] = pair_f32(p01, s, s, 0);
                t[2 * k + 1] = pair_f32(p23, s, s, 0);
            }
            w += 10;
        } else if (fmt == F_Q4_K) {
            const HVX_Vector b0 = Q6_V_vsplat_R(0xff), b1 = Q6_V_vsplat_R(0xff0000);
            for (int k = 0; k < 4; k++)
                for (int h = 0; h < 2; h++) {
                    HVX_Vector q = Q6_V_vand_VV(h ? Q6_Vuh_vlsr_VuhR(w[k], 4) : w[k], nib);
                    HVX_Vector p01 = Q6_V_vor_VV(Q6_V_vand_VV(q, b0), Q6_V_vand_VV(Q6_Vw_vasl_VwR(q, 8), b1));
                    HVX_Vector p23 = Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(q, 16), b0), Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(q, 8), b1));
                    t[8 * h + 2 * k] = pair_f32(p01, w[4], w[5], 1);
                    t[8 * h + 2 * k + 1] = pair_f32(p23, w[4], w[5], 1);
                }
            w += 6;
        } else if (fmt == F_Q5_0) {
            HVX_Vector s = to_f16_pairs(w[5], w[5]);
            for (int k = 0; k < 4; k++)
                for (int h = 0; h < 2; h++) {
                    int slot = h ? 4 + k : k;
                    HVX_Vector q = Q6_V_vor_VV(Q6_V_vand_VV(h ? Q6_Vuh_vlsr_VuhR(w[k], 4) : w[k], nib), bit4(w[4], slot));
                    tile_pair(q, s, 16, 0, &t[8 * h + 2 * k], &t[8 * h + 2 * k + 1]);
                }
            w += 6;
        } else {
            // Q4_0. Quant vector k's lane holds bytes 4k..4k+3; the low
            // nibbles are inputs 4k..4k+3 and the high nibbles the same
            // 16 on. The value is the nibble minus 8.
            HVX_Vector s = to_f16_pairs(w[4], w[4]);
            for (int k = 0; k < 4; k++)
                for (int h = 0; h < 2; h++) {
                    HVX_Vector q = Q6_V_vand_VV(h ? Q6_Vuh_vlsr_VuhR(w[k], 4) : w[k], nib);
                    tile_pair(q, s, 8, 0, &t[8 * h + 2 * k], &t[8 * h + 2 * k + 1]);
                }
            w += 5;
        }
    }
}

#endif
