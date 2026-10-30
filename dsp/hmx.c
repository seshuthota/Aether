// Matrix-matrix products on HMX, the DSP's matrix unit, for the prompt
// (many tokens at once). HMX multiplies 32 x 32 tiles of fp16 held in
// VTCM and adds into its own accumulator: one instruction pair streams a
// row of activation tiles and a column of weight tiles through it, another
// writes the 32 x 32 result out as fp16.
//
// The tiles (2 KB, 16 HVX vectors each):
// - activation, 32 tokens x 32 inputs: vector j holds tokens 2j and 2j + 1,
//   their values for input c at fp16 elements 2c and 2c + 1;
// - weight, 32 inputs x 32 outputs: vector i holds inputs 2i and 2i + 1,
//   for output n at fp16 elements 2n and 2n + 1 (one 32-bit lane);
// - the result, 32 tokens x 32 outputs, laid out as an activation tile.
// (As llama.cpp's backend lays them out; HMX is not in the SDK's docs.)
//
// The weights come in our Q4_0 layout of 32-row groups (aether_dsp.c),
// whose 32-bit lane j is row j: that is output j of a weight tile, so
// turning a block into a tile needs no shuffling across lanes.

#include <string.h>

#include "AEEStdErr.h"
#include "HAP_compute_res.h"
#include "HAP_perf.h"
#include "qurt_barrier.h"
#include "aether_dsp.h"
#include "dsp.h"
#include "hmx_hexagon_protos.h"
#include "hvx_hexagon_protos.h"

#define TILE 2048  // bytes
// The most tiles one streamed load takes: longer streams gave wrong sums.
#define DEEP 32

// Where things go in VTCM. HMX's streamed loads must not cross the 4 MB
// boundary between VTCM's two banks, so the activations have the first
// bank and the weights the second.
#define A_AT 0
#define A_MAX (3u << 20)
// The weights' two buffers: one filled while HMX reads the other.
#define W_AT (4u << 20)
#define W_BUF (1536u << 10)
#define OUT_AT (7u << 20)
#define SCALES_AT (OUT_AT + TILE)

// The f32 vectors a (token 2j) and b (token 2j + 1) as one activation
// vector: fp16, interleaved.
static inline HVX_Vector to_f16_pairs(HVX_Vector a, HVX_Vector b) {
    HVX_Vector z = Q6_V_vzero();
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vadd_VsfVsf(b, z), Q6_Vqf32_vadd_VsfVsf(a, z)));
}

// Rows of X (`tokens` of nb * 32 floats) into activation tiles, tile
// (r, b) for tokens 32r.. and inputs 32b..; past `tokens`, zeros.
static void activations(HVX_Vector *a, const float *x, uint32 nb, uint32 tokens, uint32 rows) {
    for (uint32 r = 0; r < rows; r += 2)
        for (uint32 b = 0; b < nb; b++) {
            HVX_Vector v0 = r < tokens ? *(const HVX_Vector *) (x + ((size_t) r * nb + b) * 32) : Q6_V_vzero();
            HVX_Vector v1 = r + 1 < tokens ? *(const HVX_Vector *) (x + ((size_t) (r + 1) * nb + b) * 32) : Q6_V_vzero();
            a[((r / 32) * nb + b) * 16 + (r % 32) / 2] = to_f16_pairs(v0, v1);
        }
}

// Group g's nb blocks (five vectors each: four of quant bytes, the rows'
// f32 scales) into nb weight tiles. Quant vector k's lane j has bytes
// 4k..4k+3 of row j's block; byte i holds input i in its low nibble and
// input i + 16 in its high one. So the low nibbles of bytes 0, 1 are the
// pair of inputs 4k, 4k + 1 (tile vector 2k), of bytes 2, 3 the pair
// 4k + 2, 4k + 3 (vector 2k + 1); the high nibbles the same 16 on.
static void weights(HVX_Vector *t, const HVX_Vector *w, uint32 nb) {
    const HVX_Vector nib = Q6_Vb_vsplat_R(0x0f), b0 = Q6_V_vsplat_R(0xff), b1 = Q6_V_vsplat_R(0xff0000), eight = Q6_Vh_vsplat_R(8);
    for (uint32 b = 0; b < nb; b++, w += 5, t += 16) {
        // The rows' scales, each twice (both inputs of a lane).
        HVX_Vector s = to_f16_pairs(w[4], w[4]);
        for (int k = 0; k < 4; k++)
            for (int h = 0; h < 2; h++) {
                HVX_Vector q = Q6_V_vand_VV(h ? Q6_Vuh_vlsr_VuhR(w[k], 4) : w[k], nib);
                // Bytes 0, 1 and bytes 2, 3 of each lane as its two halfwords.
                HVX_Vector p01 = Q6_V_vor_VV(Q6_V_vand_VV(q, b0), Q6_V_vand_VV(Q6_Vw_vasl_VwR(q, 8), b1));
                HVX_Vector p23 = Q6_V_vor_VV(Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(q, 16), b0), Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(q, 8), b1));
                // Q4_0 stores q + 8.
                t[8 * h + 2 * k] = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_Vh_vsub_VhVh(p01, eight)), s);
                t[8 * h + 2 * k + 1] = Q6_Vhf_vmpy_VhfVhf(Q6_Vhf_equals_Vh(Q6_Vh_vsub_VhVh(p23, eight)), s);
            }
    }
}

struct gemm_job {
    const HVX_Vector *w;
    const float *x;
    float *out;
    char *vtcm;
    uint32 rows, nb, tokens, threads;
    uint32 chunk;  // row groups dequantized per step
    qurt_barrier_t bar;
    int err;
};

// HMX on the groups g0 .. g1 whose weight tiles are in wt: for each and
// each row of 32 tokens, the sum over all nb tiles, into out as f32.
static void gemm_groups(struct gemm_job *j, const HVX_Vector *wt, uint32 g0, uint32 g1) {
    const uint32 nb = j->nb, rt = (j->tokens + 31) / 32, stride = (j->rows + 31) / 32 * 32;
    const HVX_Vector *a = (const HVX_Vector *) (j->vtcm + A_AT);
    HVX_Vector *o = (HVX_Vector *) (j->vtcm + OUT_AT);
    const HVX_Vector one = Q6_Vh_vsplat_R(0x3c00);
    for (uint32 g = g0; g < g1; g++, wt += nb * 16)
        for (uint32 r = 0; r < rt; r++) {
            Q6_mxclracc_hf();
            // A row of activation tiles and the group's column of weight
            // tiles, nb of each, streamed through (":deep"), DEEP at a time.
            for (uint32 k = 0; k < nb; k += DEEP) {
                uint32 lim = (nb - k < DEEP ? nb - k : DEEP) * TILE - 1;
                asm volatile("{ activation.hf = mxmem(%0, %1):deep\n weight.hf = mxmem(%2, %3) }" : : "r"(a + ((size_t) r * nb + k) * 16), "r"(lim), "r"(wt + (size_t) k * 16), "r"(lim) : "memory");
            }
            asm volatile("mxmem(%0, %1):after.hf = acc" : : "r"(o), "r"(0) : "memory");
            // The result tile to f32 rows: vector i has tokens 2i, 2i + 1
            // interleaved; multiplying by 1.0 into qf32 pairs splits them.
            for (uint32 i = 0; i < 16; i++) {
                uint32 tok = 32 * r + 2 * i;
                HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(o[i], one);
                if (tok < j->tokens) *(HVX_Vector *) (j->out + (size_t) tok * stride + 32 * g) = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
                if (tok + 1 < j->tokens) *(HVX_Vector *) (j->out + (size_t) (tok + 1) * stride + 32 * g) = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
            }
        }
}

// A pipeline over chunks of row groups: in step s the workers (threads
// 1..) dequantize chunk s into one weight buffer while thread 0 runs HMX
// on chunk s - 1 in the other. Dequantizing is most of the work (about
// 7 in 8 parts on one thread), so it is the part spread out. With one
// thread, thread 0 does both in turn.
static void gemm_thread(void *arg, uint32 t) {
    struct gemm_job *j = arg;
    char *v = j->vtcm;
    const uint32 nb = j->nb, groups = (j->rows + 31) / 32, chunks = (groups + j->chunk - 1) / j->chunk;
    const uint32 workers = j->threads > 1 ? j->threads - 1 : 1, wk = j->threads > 1 ? t - 1 : 0;
    const int drive = t == 0, deq = j->threads == 1 || t > 0;
    int locked = 0;
    if (drive) {
        activations((HVX_Vector *) (v + A_AT), j->x, nb, j->tokens, 32 * ((j->tokens + 31) / 32));
        // Output scales: each column's 32-bit word is its scale (low half,
        // fp16 1.0) and a bias (high half, 0), then zeros.
        HVX_Vector *sc = (HVX_Vector *) (v + SCALES_AT);
        sc[0] = Q6_V_vsplat_R(0x3c00), sc[1] = Q6_V_vzero();
        if (HAP_compute_res_hmx_lock(vtcm_ctx))
            j->err = AEE_EFAILED;
        else {
            locked = 1;
            asm volatile("bias = mxmem2(%0)" : : "r"(sc));
        }
    }
    for (uint32 s = 0; s <= chunks; s++) {
        if (deq && s < chunks) {
            uint32 g0 = s * j->chunk, n = (groups - g0 < j->chunk ? groups - g0 : j->chunk);
            HVX_Vector *wt = (HVX_Vector *) (v + W_AT + (s % 2) * W_BUF);
            for (uint32 g = g0 + n * wk / workers; g < g0 + n * (wk + 1) / workers; g++)
                weights(wt + (size_t) (g - g0) * nb * 16, j->w + (size_t) g * nb * 5, nb);
        }
        if (drive && locked && s > 0) {
            uint32 g0 = (s - 1) * j->chunk, g1 = g0 + j->chunk < groups ? g0 + j->chunk : groups;
            gemm_groups(j, (const HVX_Vector *) (v + W_AT + ((s - 1) % 2) * W_BUF), g0, g1);
        }
        if (j->threads > 1) qurt_barrier_wait(&j->bar);
    }
    if (locked) HAP_compute_res_hmx_unlock(vtcm_ctx);
}

AEEResult aether_dsp_hmx_gemm(remote_handle64 h, int w_fd, int x_fd, int out_fd, uint32 rows, uint32 nb, uint32 tokens, uint32 threads, uint64 *us) {
    (void) h;
    uint32 bytes = 0;
    char *v = vtcm_get(&bytes);
    if (!v || bytes < (8u << 20)) return AEE_ENOMEMORY;
    uint32 rt = (tokens + 31) / 32;
    if (!tokens || !nb || threads < 1 || threads > MAX_THREADS || (size_t) rt * nb * TILE > A_MAX || (size_t) nb * TILE > W_BUF) return AEE_EBADPARM;
    const HVX_Vector *w = mapped(w_fd);
    float *x = mapped(x_fd), *out = mapped(out_fd);
    int err = AEE_EBADPARM;
    if (w && x && out) {
        uint32 stride = (rows + 31) / 32 * 32, groups = (rows + 31) / 32;
        cpu_wrote(x, tokens * nb * 128);
        // As many groups as fit a buffer, but no more than a sixth of them
        // (so the pipeline fills soon), and a multiple of the workers.
        uint32 workers = threads > 1 ? threads - 1 : 1, chunk = W_BUF / (nb * TILE);
        if (chunk > (groups + 5) / 6) chunk = (groups + 5) / 6;
        if (chunk > workers) chunk -= chunk % workers;
        struct gemm_job j = {w, x, out, v, rows, nb, tokens, threads, chunk};
        qurt_barrier_init(&j.bar, threads);
        err = run_threads(threads, gemm_thread, &j, us);
        qurt_barrier_destroy(&j.bar);
        if (!err) err = j.err;
        cpu_reads(out, tokens * stride * 4);
    }
    if (w) release(w_fd);
    if (x) release(x_fd);
    if (out) release(out_fd);
    return err;
}
