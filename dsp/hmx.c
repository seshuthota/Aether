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

// Asks the L2 for `bytes` from p on (rows of 4 KB), without waiting.
static inline void prefetch_l2(const void *p, uint32 bytes) {
    uint32 rows = (bytes + 4095) / 4096;
    if (rows) Q6_l2fetch_AP((void *) p, (uint64) 4096 << 32 | 4096 << 16 | (rows > 0xffff ? 0xffff : rows));
}

// The f32 vectors a (token 2j) and b (token 2j + 1) as one activation
// vector: fp16, interleaved.
static inline HVX_Vector to_f16_pairs(HVX_Vector a, HVX_Vector b) {
    HVX_Vector z = Q6_V_vzero();
    return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(Q6_Vqf32_vadd_VsfVsf(b, z), Q6_Vqf32_vadd_VsfVsf(a, z)));
}

// Rows of X (`tokens` of nb * 32 floats, `stride` floats apart) into
// activation tiles, tile (r, b) for tokens 32r.. and inputs 32b..; past
// `tokens`, zeros.
static void activations(HVX_Vector *a, const float *x, uint32 stride, uint32 nb, uint32 tokens, uint32 rows) {
    for (uint32 r = 0; r < rows; r += 2)
        for (uint32 b = 0; b < nb; b++) {
            HVX_Vector v0 = r < tokens ? *(const HVX_Vector *) (x + (size_t) r * stride + b * 32) : Q6_V_vzero();
            HVX_Vector v1 = r + 1 < tokens ? *(const HVX_Vector *) (x + (size_t) (r + 1) * stride + b * 32) : Q6_V_vzero();
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

// HMX on the groups g0 .. g1 whose weight tiles are in wt: for each and
// each row of 32 tokens, the sum over all nb tiles, into out as f32.
static void gemm_groups(const struct hmx_product *p, const HVX_Vector *wt, uint32 g0, uint32 g1) {
    const uint32 nb = p->nb, rt = (p->tokens + 31) / 32;
    const HVX_Vector *a = (const HVX_Vector *) (p->vtcm + A_AT);
    HVX_Vector *o = (HVX_Vector *) (p->vtcm + OUT_AT);
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
                HVX_VectorPair pr = Q6_Wqf32_vmpy_VhfVhf(o[i], one);
                for (uint32 e = 0; e < 2 && tok + e < p->tokens; e++) {
                    HVX_Vector *dst = (HVX_Vector *) (p->out + (size_t) (tok + e) * p->out_stride + 32 * g);
                    HVX_Vector v = Q6_Vsf_equals_Vqf32(e ? Q6_V_hi_W(pr) : Q6_V_lo_W(pr));
                    *dst = p->accumulate ? Q6_Vsf_vadd_VsfVsf(*dst, v) : v;
                }
            }
        }
}

// A pipeline over chunks of row groups: in step s the workers (threads
// 1..) dequantize chunk s into one weight buffer while thread 0 runs HMX
// on chunk s - 1 in the other. Dequantizing is most of the work (about
// 7 in 8 parts on one thread), so it is the part spread out. With one
// thread, thread 0 does both in turn.
int hmx_rows(const struct hmx_product *p, uint32 t, uint32 T, qurt_barrier_t *bar) {
    char *v = p->vtcm;
    const uint32 nb = p->nb, groups = (p->rows + 31) / 32;
    // As many groups as fit a buffer, but no more than a sixth of them
    // (so the pipeline fills soon), and a multiple of the workers.
    const uint32 workers = T > 1 ? T - 1 : 1, wk = T > 1 ? t - 1 : 0;
    uint32 chunk = W_BUF / (nb * TILE);
    if (chunk > (groups + 5) / 6) chunk = (groups + 5) / 6;
    if (chunk > workers) chunk -= chunk % workers;
    const uint32 chunks = (groups + chunk - 1) / chunk;
    const int drive = t == 0, deq = T == 1 || t > 0;
    int locked = 0, err = 0;
    if (drive) {
        activations((HVX_Vector *) (v + A_AT), p->x, p->x_stride, nb, p->tokens, 32 * ((p->tokens + 31) / 32));
        // Output scales: each column's 32-bit word is its scale (low half,
        // fp16 1.0) and a bias (high half, 0), then zeros.
        HVX_Vector *sc = (HVX_Vector *) (v + SCALES_AT);
        sc[0] = Q6_V_vsplat_R(0x3c00), sc[1] = Q6_V_vzero();
        if (HAP_compute_res_hmx_lock(vtcm_ctx))
            err = AEE_EFAILED;
        else {
            locked = 1;
            asm volatile("bias = mxmem2(%0)" : : "r"(sc));
        }
    }
    // Worker wk's groups of chunk s: from .. to.
    uint32 from = 0, to = 0;
#define SHARE(s)                                                                    \
    do {                                                                            \
        uint32 c0 = (s) * chunk, n = groups - c0 < chunk ? groups - c0 : chunk;     \
        from = c0 + n * wk / workers, to = c0 + n * (wk + 1) / workers;       \
    } while (0)
    if (deq) {
        SHARE(0);
        prefetch_l2(p->w + (size_t) from * nb * 5, (to - from) * nb * 640);
    }
    for (uint32 s = 0; s <= chunks; s++) {
        if (deq && s < chunks) {
            // Ask the L2 for this worker's share of the next chunk while
            // dequantizing this one: without it a thread waits on DRAM
            // for most of the time (~0.8 GB/s).
            if (s + 1 < chunks) {
                SHARE(s + 1);
                prefetch_l2(p->w + (size_t) from * nb * 5, (to - from) * nb * 640);
            }
            SHARE(s);
            HVX_Vector *wt = (HVX_Vector *) (v + W_AT + (s % 2) * W_BUF);
            for (uint32 g = from; g < to; g++)
                weights(wt + (size_t) (g - s * chunk) * nb * 16, p->w + (size_t) g * nb * 5, nb);
        }
        if (drive && locked && s > 0) {
            uint32 g0 = (s - 1) * chunk, g1 = g0 + chunk < groups ? g0 + chunk : groups;
            gemm_groups(p, (const HVX_Vector *) (v + W_AT + ((s - 1) % 2) * W_BUF), g0, g1);
        }
        if (T > 1) qurt_barrier_wait(bar);
    }
#undef SHARE
    if (locked) HAP_compute_res_hmx_unlock(vtcm_ctx);
    return err;
}

int hmx_fits(uint32 nb, uint32 tokens) {
    return tokens && nb && (size_t) (tokens + 31) / 32 * nb * TILE <= A_MAX && (size_t) nb * TILE <= W_BUF;
}

struct gemm_job {
    struct hmx_product p;
    uint32 threads;
    qurt_barrier_t bar;
    int err;
};

static void gemm_thread(void *arg, uint32 t) {
    struct gemm_job *j = arg;
    int err = hmx_rows(&j->p, t, j->threads, &j->bar);
    if (err) j->err = err;
}

AEEResult aether_dsp_hmx_gemm(remote_handle64 h, int w_fd, int x_fd, int out_fd, uint32 rows, uint32 nb, uint32 tokens, uint32 threads, uint64 *us) {
    (void) h;
    uint32 bytes = 0;
    char *v = vtcm_get(&bytes);
    if (!v || bytes < (8u << 20)) return AEE_ENOMEMORY;
    if (!hmx_fits(nb, tokens) || threads < 1 || threads > MAX_THREADS) return AEE_EBADPARM;
    const HVX_Vector *w = mapped(w_fd);
    float *x = mapped(x_fd), *out = mapped(out_fd);
    int err = AEE_EBADPARM;
    if (w && x && out) {
        uint32 stride = (rows + 31) / 32 * 32;
        cpu_wrote(x, tokens * nb * 128);
        struct gemm_job j = {{w, rows, nb, x, nb * 32, tokens, out, stride, 0, v}, threads};
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
