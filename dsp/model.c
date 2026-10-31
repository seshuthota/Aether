// A whole decode step on the DSP: one call per token (`model_decode`) runs
// every layer and the classifier, on DSP threads that move through the
// layers together (each does its share of a step, then all wait at a
// barrier before the next step that needs the result).
//
// The CPU sets the model up once (`model_load`): it uploads the weights
// into one shared buffer (matrices in our layouts of qx_hvx.h, q/k/v and
// gate/up one after another; norms and biases in f32) and allocates a second
// shared buffer for the state (activations, KV cache, logits). `desc`
// gives the shapes and where everything is. Per token the CPU writes the
// token's embedding into x and the RoPE table (sin, cos per pair, computed
// as the CPU path computes them), and the DSP does the rest.
//
// The activations are f32; before every matrix product they are quantized
// to Q8_0, as on our CPU path and llama.cpp's.
//
// A product is up to three runs of rows, each in one format: q/k/v are
// one run when all three have the same format (their groups are back to
// back), else a run per format change, and so for gate/up. The prefill
// (on HMX) takes Q4_0 only.
//
// One model per DSP process (it lives in a static).

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_perf.h"
#include "aether_dsp.h"
#include "dsp.h"
#include "hvx_hexagon_protos.h"
#include "qx_hvx.h"
#include "qurt_barrier.h"

// desc[], in order. Offsets are bytes into the weights (W) or state (S)
// buffer, all 128-byte aligned.
enum {
    D_DIM,
    D_HIDDEN,
    D_LAYERS,
    D_HEADS,
    D_KV_HEADS,
    D_VOCAB,
    D_SEQ_LEN,
    D_HALVES,  // RoPE pairs (i, i + hs/2), else (2i, 2i + 1)
    D_EPS,     // f32 bits
    D_W_FD,
    D_S_FD,
    D_W_BYTES,
    D_S_BYTES,
    D_FINAL_NORM,  // W
    D_CLS,         // W
    D_X,           // S: dim
    D_XB,          // S: dim (norm output, attention output)
    D_XQ,          // S: MAX_FEW rows of Q8_0 of up to hidden, each
                   // 40 max(dim, hidden) / 32 bytes rounded up to 128
    D_QKV,         // S: dim + 2 kv_dim
    D_ATT,         // S: n_heads rows of seq_len scores, rounded up to 32
    D_HB,          // S: 2 hidden (gate, up)
    D_LOGITS,      // S: vocab
    D_KC,          // S: layers * kv_dim * att_stride: keys transposed (see attend)
    D_VC,
    D_ROT,   // S: head_size / 2 (sin, cos) pairs
    D_PROF,   // S: N_STAGES u64 of qtimer ticks, then at ATTN_CY_OFF the
              // per-thread streamed-attention counters
    D_BATCH,  // the most tokens of a prefill: X, XB, QKV, HB, ROT have this
              // many rows and at least MAX_FEW, LOGITS this many with all
              // logits and at least MAX_FEW, ATT at least MAX_THREADS
              // (decode uses row 0 of each)
    D_CLS_FMT,
    D_LAYER0,
};
// Per layer, from D_LAYER0 on: W offsets, formats (F_ of qx_hvx.h) and
// rows. q/k/v: three runs of (offset, format, rows), gate/up two (unused
// ones 0 rows).
enum { L_QKV = 0, L_WO = 9, L_WO_FMT, L_GU, L_WDOWN = 17, L_WDOWN_FMT, L_ATTN_NORM, L_FFN_NORM, L_BQ, L_BK, L_BV, L_N };

// Rows of a matrix in one format.
struct run {
    const HVX_Vector *w;
    uint32 fmt, rows;
};

#define MAX_LAYERS 64
// The classifier's fetch in a prefill without every token's logits: DMA.
#define DMA_DEFAULT 0x10

// The stages of a step that thread 0 times (from the end of the previous
// one to when every thread has finished this one).
enum { P_NORM1, P_QKV, P_POST, P_ATTEND, P_WO, P_NORM2, P_GATE_UP, P_SWIGLU, P_DOWN, P_FINAL_NORM, P_CLS, P_ARGMAX, P_START, N_STAGES };

static struct model {
    int loaded;
    uint32 dim, hid, layers, heads, kv_heads, vocab, seq_len, att_stride, halves, hs, kv_dim, batch, xq_stride;
    float eps;
    int w_fd, s_fd;
    char *w, *s;
    uint32 s_bytes;
    float *x, *xb, *qkv, *att, *hb, *logits, *kc, *vc, *rot;
    uint64 *prof;
    // Streamed-attention pcycles, see ATTN_CY_OFF.
    uint64 *cy;
    char *xq;
    const float *final_norm;
    struct run cls;
    struct {
        struct run qkv[3], wo, gu[2], wdown;
        const float *attn_norm, *ffn_norm, *bq, *bk, *bv;
    } l[MAX_LAYERS];
    // Every product Q4_0 in one run (what the prefill takes).
    int q4_0_only;
} M;

// Positions per streamed K/V block, and the shortest context that streams.
// Below that, attend() prefetches from DDR. At 128 and 256 a layer is one
// or two blocks, so two of six threads do the work and Qwen's attention
// is about 100 µs slower than the prefetch; SmolLM2 is already faster.
// From 512 up the stream is faster on both (results/phase-l5.md).
#define ATTN_BLOCK 256
#define ATTN_STREAM_AT 256
// Per-thread streamed-attention counters, placed after the stage timers
// on their own cache lines. Nine u64s a thread: pcycles of DMA wait,
// score + softmax, value accumulation, barriers + merge; pcycles inside
// exp (a subset of the score part); then qtimer ticks of the same four
// parts. The CPU sizes this the same way (dsp.rs).
#define ATTN_CY_OFF 128
#define ATTN_CY_STRIDE 9
enum { CY_DMA, CY_SCORE, CY_VALUE, CY_MERGE, CY_EXP };
// Query heads per K/V head this path stores partials for. Qwen has 7,
// SmolLM2 has 3.
#define ATTN_GMAX 16

// One partial per (K/V head, position block): g·hs unnormalised outputs,
// then each query head's (max, sum of e^(s−max)), padded so the next
// partial stays on a vector. Big enough for the loaded context.
static struct attn_buf {
    float *p;
    uint32 per, blocks, g;
} Attn;

// Reads n runs from d (offset, format, rows each) into r; whether they
// are valid and add up to `rows`.
static int read_runs(struct run *r, const uint32 *d, uint32 n, uint32 rows) {
    uint32 sum = 0;
    for (uint32 i = 0; i < n; i++, d += 3) {
        r[i] = (struct run) {(const HVX_Vector *) (M.w + d[0]), d[1], d[2]};
        if (r[i].fmt >= N_FORMATS || r[i].rows % 32) return 0;
        if (r[i].rows && (i > 0 || r[i].fmt != F_Q4_0)) M.q4_0_only = 0;
        sum += r[i].rows;
    }
    return sum == rows;
}

AEEResult aether_dsp_model_load(remote_handle64 h, const uint32 *d, int n) {
    (void) h;
    if (n < D_LAYER0 || d[D_LAYERS] > MAX_LAYERS || n != D_LAYER0 + (int) d[D_LAYERS] * L_N) return AEE_EBADPARM;
    if (M.loaded) {
        release(M.w_fd);
        release(M.s_fd);
        free(Attn.p);
        Attn.p = NULL;
        M.loaded = 0;
    }
    M.dim = d[D_DIM], M.hid = d[D_HIDDEN], M.layers = d[D_LAYERS], M.heads = d[D_HEADS], M.kv_heads = d[D_KV_HEADS];
    M.vocab = d[D_VOCAB], M.seq_len = d[D_SEQ_LEN], M.halves = d[D_HALVES], M.batch = d[D_BATCH];
    memcpy(&M.eps, &d[D_EPS], 4);
    M.hs = M.dim / M.heads, M.kv_dim = M.hs * M.kv_heads;
    // Rounded up to a vector. A power-of-two stride lands every key element
    // on the same cache sets, so add one vector (32 floats). The state
    // buffer uses the same rule (DspModel in dsp.rs).
    M.att_stride = (M.seq_len + 31) / 32 * 32;
    if (M.att_stride && (M.att_stride & (M.att_stride - 1)) == 0) M.att_stride += 32;
    M.xq_stride = (40 * (M.hid > M.dim ? M.hid : M.dim) / 32 + 127) / 128 * 128;
    // Every matrix's rows must be whole groups of 32 (the products write
    // whole groups, and add them in place into x), and each head whole
    // Q8_0 blocks (the attention quantizes its heads' outputs).
    if (M.dim % 32 || M.hid % 32 || M.hs % 32 || M.hs > 256 || M.vocab % 32 || M.heads % M.kv_heads || !M.batch) return AEE_EBADPARM;
    M.w_fd = d[D_W_FD], M.s_fd = d[D_S_FD];
    M.w = mapped(M.w_fd), M.s = mapped(M.s_fd);
    if (!M.w || !M.s) return AEE_EBADPARM;
    M.s_bytes = d[D_S_BYTES];
    // The CPU wrote all of the weights; drop anything stale.
    cpu_wrote(M.w, d[D_W_BYTES]);
    cpu_will_write(M.s, M.s_bytes);
    char *s = M.s, *w = M.w;
    M.x = (float *) (s + d[D_X]), M.xb = (float *) (s + d[D_XB]), M.xq = s + d[D_XQ], M.qkv = (float *) (s + d[D_QKV]);
    M.att = (float *) (s + d[D_ATT]), M.hb = (float *) (s + d[D_HB]), M.logits = (float *) (s + d[D_LOGITS]);
    M.kc = (float *) (s + d[D_KC]), M.vc = (float *) (s + d[D_VC]), M.rot = (float *) (s + d[D_ROT]);
    M.prof = (uint64 *) (s + d[D_PROF]);
    M.cy = (uint64 *) (s + d[D_PROF] + ATTN_CY_OFF);
    M.final_norm = (const float *) (w + d[D_FINAL_NORM]);
    M.q4_0_only = 1;
    uint32 cls[3] = {d[D_CLS], d[D_CLS_FMT], M.vocab};
    int ok = read_runs(&M.cls, cls, 1, M.vocab);
    for (uint32 l = 0; l < M.layers; l++) {
        const uint32 *o = d + D_LAYER0 + l * L_N;
        uint32 wo[3] = {o[L_WO], o[L_WO_FMT], M.dim}, down[3] = {o[L_WDOWN], o[L_WDOWN_FMT], M.dim};
        ok = ok && read_runs(M.l[l].qkv, o + L_QKV, 3, M.dim + 2 * M.kv_dim) && read_runs(&M.l[l].wo, wo, 1, M.dim);
        ok = ok && read_runs(M.l[l].gu, o + L_GU, 2, 2 * M.hid) && read_runs(&M.l[l].wdown, down, 1, M.dim);
        M.l[l].attn_norm = (const float *) (w + o[L_ATTN_NORM]);
        M.l[l].ffn_norm = (const float *) (w + o[L_FFN_NORM]);
        M.l[l].bq = (const float *) (w + o[L_BQ]);
        M.l[l].bk = (const float *) (w + o[L_BK]);
        M.l[l].bv = (const float *) (w + o[L_BV]);
    }
    if (!ok) {
        release(M.w_fd);
        release(M.s_fd);
        return AEE_EBADPARM;
    }
    // VTCM for the products' DMA (see vtcm_get).
    vtcm_get(&(uint32) {0});
    // Partials for streamed attention, one context's worth of blocks.
    uint32 g = M.heads / M.kv_heads;
    Attn.g = g;
    Attn.blocks = (M.seq_len + ATTN_BLOCK - 1) / ATTN_BLOCK;
    Attn.per = (g * M.hs + 2 * g + 31u) & ~31u;
    if (g <= ATTN_GMAX && Attn.blocks)
        Attn.p = memalign(128, (size_t) Attn.blocks * M.kv_heads * Attn.per * sizeof(float));
    M.loaded = 1;
    return AEE_SUCCESS;
}

// roundf (halves away from zero) without the library call: truncate, then
// step away from zero if the fraction, exact here, is at least a half.
static inline int32 round_away(float v) {
    int32 r = (int32) v;
    float f = v - (float) r;
    return r + (f >= 0.5f) - (f <= -0.5f);
}

// Blocks b0 .. b1 of x (nb blocks in all) to Q8_0 as q4_0_rows reads it
// (dsp.h), as llama.cpp's quantize_row_q8_0_ref does: d = amax / 127
// stored as f16, values round(x / d).
static void quantize_q8_0(const float *x, uint32 nb, uint32 b0, uint32 b1, char *xq) {
    signed char *q = (signed char *) xq;
    float *dv = (float *) (xq + 32 * nb);
    int32 *sv = (int32 *) (xq + 36 * nb);
    for (uint32 b = b0; b < b1; b++) {
        const float *v = x + 32 * b;
        float amax = 0;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[i]));
        float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        dv[b] = (float) (__fp16) d;
        int32 sum = 0;
        for (int i = 0; i < 32; i++) {
            int32 r = round_away(v[i] * id);
            q[32 * b + i] = (signed char) r;
            sum += r;
        }
        sv[b] = sum;
    }
}

// The RMS norm of x (n values) times w into o, and quantized into xq,
// for blocks b0 .. b1 only (every thread sums all of x itself, so the
// threads need not wait for each other).
static void norm_quantize_to(float *o, const float *x, const float *w, uint32 n, uint32 b0, uint32 b1, char *xq) {
    float ss = 0;
    for (uint32 i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + M.eps);
    for (uint32 i = 32 * b0; i < 32 * b1; i++) o[i] = w[i] * (x[i] * inv);
    quantize_q8_0(o, n / 32, b0, b1, xq);
}

static void norm_quantize(float *o, const float *x, const float *w, uint32 n, uint32 b0, uint32 b1) {
    norm_quantize_to(o, x, w, n, b0, b1, M.xq);
}

// RoPE pairs p0 .. p1 of q and k (q's dim / 2 first, then k's kv_dim / 2)
// in `qkv` (one token's q, k, v) with their biases, by the table `rot`, k
// into the layer's cache at `pos`; and v's elements v0 .. v1 with their
// bias into the cache.
static void qkv_post(uint32 l, uint32 pos, float *qkv, const float *rot, uint32 p0, uint32 p1, uint32 v0, uint32 v1) {
    const uint32 dim = M.dim, kv = M.kv_dim, hs = M.hs, half = hs / 2;
    const float *bq = M.l[l].bq, *bk = M.l[l].bk, *bv = M.l[l].bv;
    float *q = qkv, *k = qkv + dim, *v = qkv + dim + kv;
    // Key element e of the layer at kc[e * att_stride] (see attend).
    float *kc = M.kc + (size_t) l * kv * M.att_stride + pos, *vc = M.vc + ((size_t) l * M.seq_len + pos) * kv;
    for (uint32 i = v0; i < v1; i++) vc[i] = v[i] + bv[i];
    for (uint32 p = p0; p < p1; p++) {
        int isq = p < dim / 2;
        uint32 pp = isq ? p : p - dim / 2, hd = pp / half, i = pp % half;
        uint32 i0 = M.halves ? hd * hs + i : hd * hs + 2 * i, i1 = M.halves ? i0 + half : i0 + 1;
        float s = rot[2 * i], c = rot[2 * i + 1];
        const float *b = isq ? bq : bk;
        float *src = isq ? q : k;
        float a = src[i0] + b[i0], e = src[i1] + b[i1];
        if (isq) {
            q[i0] = a * c - e * s;
            q[i1] = a * s + e * c;
        } else {
            kc[(size_t) i0 * M.att_stride] = a * c - e * s;
            kc[(size_t) i1 * M.att_stride] = a * s + e * c;
        }
    }
}

// HVX float helpers (IEEE single precision, Hexagon v73 and later;
// splat_f is in q4_0_hvx.h, which qx_hvx.h includes).
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

// Ask L2 for `height` rows of `width` bytes, `stride` bytes apart.
// l2fetch's stride field is 16 bits and the width must not exceed it, so a
// stride that does not fit is left to the loads.
static inline void l2fetch_2d(const void *p, uint32 stride, uint32 width, uint32 height) {
    if (!height || !width || width > stride || stride > 0xffffu) return;
    if (height > 0xffffu) height = 0xffffu;
    Q6_l2fetch_AP((void *) p, (uint64) stride << 32 | (uint64) width << 16 | height);
}

// Head h's attention for the query qh (its hs values) over positions
// 0..pos into out (hs outputs, hs / 32 vectors), with the scores in a.
// Every row of q, the caches and the scores starts on a vector. This is
// the short-context path: each query head reads its K/V from DDR, with
// l2fetch. Past ATTN_STREAM_AT positions, attend_token streams instead.
//
// The keys are stored transposed: per layer, row e of the cache holds key
// element e (of kv_dim) for every position, att_stride apart. So the
// scores of 32 positions are one vector, the sum over the head's hs
// elements of q[e] (in every lane) times row e's 32 positions: no sum
// across the lanes per position.
static void attend(uint32 l, uint32 h, uint32 pos, const float *qh, float *out_h, float *a) {
    const uint32 hs = M.hs, nv = hs / 32, kv = M.kv_dim, kvh = h / (M.heads / M.kv_heads), S = M.att_stride;
    const float *kt = M.kc + ((size_t) l * kv + kvh * hs) * S, *vc = M.vc + (size_t) l * M.seq_len * kv + kvh * hs;
    float scale = 1.0f / sqrtf((float) hs);
    uint32 n = (pos + 1 + 31) / 32;
    HVX_Vector *av = (HVX_Vector *) a;
    for (uint32 j = 0; j < n; j++) {
        // One key chunk (32 positions, one vector on each of the hs rows),
        // four chunks ahead of this one. The first four chunks are not
        // fetched; every later chunk is fetched once.
        if (j + 4 < n) l2fetch_2d(kt + 32 * (j + 4), S * 4, 128, hs);
        // Two sums, so the adds of one wait less on the other.
        HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
        const float *col = kt + 32 * j;
        for (uint32 e = 0; e < hs; e += 2) {
            s0 = VADD(s0, VMUL(splat_f(qh[e]), *(const HVX_Vector *) (col + (size_t) e * S)));
            s1 = VADD(s1, VMUL(splat_f(qh[e + 1]), *(const HVX_Vector *) (col + (size_t) (e + 1) * S)));
        }
        av[j] = VMUL(VADD(s0, s1), splat_f(scale));
    }
    // The lanes past pos (keys not written yet) are -inf: e^-80 after the
    // clamp below, and not used.
    for (uint32 t = pos + 1; t < 32 * n; t++) a[t] = -INFINITY;
    HVX_Vector vm = av[0];
    for (uint32 j = 1; j < n; j++) vm = Q6_Vsf_vfmax_VsfVsf(vm, av[j]);
    float max = vmax_lanes(vm);
    // e^(s - max) over whole vectors.
    HVX_Vector vmax = splat_f(max), vs = Q6_V_vzero();
    for (uint32 j = 0; j < n; j++) {
        av[j] = vexp(VSUB(av[j], vmax));
        if (j + 1 < n) vs = VADD(vs, av[j]);
    }
    float sum = vsum(vs);
    for (uint32 t = 32 * (n - 1); t <= pos; t++) sum += a[t];
    // Scale every score once. Four accumulators then add their partial
    // sums: one chain of `pos` dependent adds stalls the vector unit, and
    // the four chains do not. The adds are in a different order from a
    // single chain, so the output differs in the last bits.
    HVX_Vector vinv = splat_f(1.0f / sum);
    for (uint32 j = 0; j < n; j++) av[j] = VMUL(av[j], vinv);
    HVX_Vector acc[4][8];
    for (uint32 k = 0; k < 4; k++)
        for (uint32 i = 0; i < nv; i++) acc[k][i] = Q6_V_vzero();
    uint32 t = 0, npos = pos + 1;
    for (; t + 4 <= npos; t += 4) {
        // This head's value rows (hs floats, stride kv), one block of 32
        // positions, four blocks ahead. Issued once per block.
        if ((t & 31) == 0 && t + 128 < npos && t + 128 < M.seq_len) {
            uint32 room = npos - (t + 128);
            uint32 cap = M.seq_len - (t + 128);
            if (room > cap) room = cap;
            l2fetch_2d(vc + (size_t) (t + 128) * kv, kv * 4, hs * 4, room < 32 ? room : 32);
        }
        for (uint32 k = 0; k < 4; k++) {
            const HVX_Vector *v = (const HVX_Vector *) (vc + (size_t) (t + k) * kv);
            HVX_Vector p = splat_f(a[t + k]);
            for (uint32 i = 0; i < nv; i++) acc[k][i] = VADD(acc[k][i], VMUL(p, v[i]));
        }
    }
    for (; t < npos; t++) {
        const HVX_Vector *v = (const HVX_Vector *) (vc + (size_t) t * kv);
        HVX_Vector p = splat_f(a[t]);
        for (uint32 i = 0; i < nv; i++) acc[0][i] = VADD(acc[0][i], VMUL(p, v[i]));
    }
    HVX_Vector *out = (HVX_Vector *) out_h;
    for (uint32 i = 0; i < nv; i++) out[i] = VADD(VADD(acc[0][i], acc[1][i]), VADD(acc[2][i], acc[3][i]));
}

// Add one timed span of streamed attention on thread t. `part` is a
// CY_ bucket; the qtimer ticks land nine slots later, after the exp
// subset. Exp is recorded on its own and is also inside the score span.
static void cy_add(uint32 t, uint32 part, uint64 c0, uint64 q0) {
    uint64 *p = M.cy + (size_t) t * ATTN_CY_STRIDE;
    p[part] += HAP_perf_get_pcycles() - c0;
    if (part < CY_EXP) p[5 + part] += HAP_perf_get_qtimer_count() - q0;
}

static void cy_cpu_in(void) { cpu_wrote(M.cy, 8u * MAX_THREADS * ATTN_CY_STRIDE); }
static void cy_cpu_out(void) { cpu_reads(M.cy, 8u * MAX_THREADS * ATTN_CY_STRIDE); }

static void mix_one_64(const float *s, const float *v, uint32 len, float *out);
static void mix_one(const float *s, const float *v, uint32 len, float *out);

// One query head over one packed block. Keys are kpos floats per element
// (element e at k[e * kpos]); values are hs floats per position. The first
// `len` positions are real. Scores stay in `a` (the caller's stack), not
// in VTCM: a scalar load from VTCM stalled the value loop. Writes
// Σ e^(s−max)·v, unnormalised, and (max, sum).
static void block_head(const float *qh, const float *k, uint32 kpos, const float *v, uint32 len, float *acc_out, float *max_out, float *sum_out, float *a, uint32 t) {
    const uint32 hs = M.hs;
    float scale = 1.0f / sqrtf((float) hs);
    uint32 n = kpos / 32;
    HVX_Vector *av = (HVX_Vector *) a;
    uint64 c0 = HAP_perf_get_pcycles(), q0 = HAP_perf_get_qtimer_count();
    for (uint32 j = 0; j < n; j++) {
        HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero();
        const float *col = k + 32 * j;
        for (uint32 e = 0; e < hs; e += 2) {
            s0 = VADD(s0, VMUL(splat_f(qh[e]), *(const HVX_Vector *) (col + (size_t) e * kpos)));
            s1 = VADD(s1, VMUL(splat_f(qh[e + 1]), *(const HVX_Vector *) (col + (size_t) (e + 1) * kpos)));
        }
        av[j] = VMUL(VADD(s0, s1), splat_f(scale));
    }
    for (uint32 u = len; u < kpos; u++) a[u] = -INFINITY;
    HVX_Vector vm = av[0];
    for (uint32 j = 1; j < n; j++) vm = Q6_Vsf_vfmax_VsfVsf(vm, av[j]);
    float max = vmax_lanes(vm);
    HVX_Vector vmax = splat_f(max), vs = Q6_V_vzero();
    uint64 cexp = HAP_perf_get_pcycles();
    for (uint32 j = 0; j < n; j++) {
        av[j] = vexp(VSUB(av[j], vmax));
        if (j + 1 < n) vs = VADD(vs, av[j]);
    }
    M.cy[(size_t) t * ATTN_CY_STRIDE + CY_EXP] += HAP_perf_get_pcycles() - cexp;
    float sum = vsum(vs);
    for (uint32 u = 32 * (n - 1); u < len; u++) sum += a[u];
    *max_out = max;
    *sum_out = sum;
    cy_add(t, CY_SCORE, c0, q0);
    c0 = HAP_perf_get_pcycles();
    q0 = HAP_perf_get_qtimer_count();
    if (hs == 64) mix_one_64(a, v, len, acc_out);
    else mix_one(a, v, len, acc_out);
    cy_add(t, CY_VALUE, c0, q0);
}

// Head size 64: four named accumulator pairs. An array of them lands in
// memory and the inner loop reloads it; naming them keeps the four chains
// in the vector file. Positions 4k+c go into pair c, the tail into pair 0.
static void __attribute__((noinline)) mix_one_64(const float *s, const float *v, uint32 len, float *out) {
    const uint32 hs = 64;
    HVX_Vector a00 = Q6_V_vzero(), a01 = Q6_V_vzero(), a10 = Q6_V_vzero(), a11 = Q6_V_vzero();
    HVX_Vector a20 = Q6_V_vzero(), a21 = Q6_V_vzero(), a30 = Q6_V_vzero(), a31 = Q6_V_vzero();
    uint32 u = 0;
    for (; u + 4 <= len; u += 4) {
        const HVX_Vector *r;
        HVX_Vector p;
        r = (const HVX_Vector *) (v + (size_t) (u + 0) * hs);
        p = splat_f(s[u + 0]);
        a00 = VADD(a00, VMUL(p, r[0]));
        a01 = VADD(a01, VMUL(p, r[1]));
        r = (const HVX_Vector *) (v + (size_t) (u + 1) * hs);
        p = splat_f(s[u + 1]);
        a10 = VADD(a10, VMUL(p, r[0]));
        a11 = VADD(a11, VMUL(p, r[1]));
        r = (const HVX_Vector *) (v + (size_t) (u + 2) * hs);
        p = splat_f(s[u + 2]);
        a20 = VADD(a20, VMUL(p, r[0]));
        a21 = VADD(a21, VMUL(p, r[1]));
        r = (const HVX_Vector *) (v + (size_t) (u + 3) * hs);
        p = splat_f(s[u + 3]);
        a30 = VADD(a30, VMUL(p, r[0]));
        a31 = VADD(a31, VMUL(p, r[1]));
    }
    for (; u < len; u++) {
        const HVX_Vector *r = (const HVX_Vector *) (v + (size_t) u * hs);
        HVX_Vector p = splat_f(s[u]);
        a00 = VADD(a00, VMUL(p, r[0]));
        a01 = VADD(a01, VMUL(p, r[1]));
    }
    HVX_Vector *o = (HVX_Vector *) out;
    o[0] = VADD(VADD(a00, a10), VADD(a20, a30));
    o[1] = VADD(VADD(a01, a11), VADD(a21, a31));
}

// Other head sizes: the same four chains, held in an array. Head size is
// at most 256, so eight vectors a chain.
static void mix_one(const float *s, const float *v, uint32 len, float *out) {
    const uint32 hs = M.hs, nv = hs / 32;
    HVX_Vector acc[4][8];
    for (uint32 c = 0; c < 4; c++)
        for (uint32 i = 0; i < nv; i++) acc[c][i] = Q6_V_vzero();
    uint32 u = 0;
    for (; u + 4 <= len; u += 4) {
        for (uint32 c = 0; c < 4; c++) {
            const HVX_Vector *r = (const HVX_Vector *) (v + (size_t) (u + c) * hs);
            HVX_Vector p = splat_f(s[u + c]);
            for (uint32 i = 0; i < nv; i++) acc[c][i] = VADD(acc[c][i], VMUL(p, r[i]));
        }
    }
    for (; u < len; u++) {
        const HVX_Vector *r = (const HVX_Vector *) (v + (size_t) u * hs);
        HVX_Vector p = splat_f(s[u]);
        for (uint32 i = 0; i < nv; i++) acc[0][i] = VADD(acc[0][i], VMUL(p, r[i]));
    }
    HVX_Vector *o = (HVX_Vector *) out;
    for (uint32 i = 0; i < nv; i++) o[i] = VADD(VADD(acc[0][i], acc[1][i]), VADD(acc[2][i], acc[3][i]));
}

// acc = sa·acc + sb·b, hs floats, both vector-aligned.
static void scale_add(float *acc, float sa, const float *b, float sb) {
    HVX_Vector *o = (HVX_Vector *) acc;
    const HVX_Vector *p = (const HVX_Vector *) b;
    HVX_Vector va = splat_f(sa), vb = splat_f(sb);
    for (uint32 i = 0; i < M.hs / 32; i++) o[i] = VADD(VMUL(va, o[i]), VMUL(vb, p[i]));
}

// Query head h from the block partials of its K/V head, in block order:
// with M the largest block max, out = Σ e^(m_i−M)·acc_i / Σ e^(m_i−M)·sum_i.
static void merge_head(uint32 h, uint32 nblocks, float *out_h) {
    const uint32 g = Attn.g, hs = M.hs, lh = h % g, kvh = h / g;
    float *p0 = Attn.p + (size_t) kvh * nblocks * Attn.per;
    float m = p0[g * hs + 2 * lh], sum = p0[g * hs + 2 * lh + 1];
    HVX_Vector *o = (HVX_Vector *) out_h;
    const HVX_Vector *a0 = (const HVX_Vector *) (p0 + lh * hs);
    for (uint32 i = 0; i < hs / 32; i++) o[i] = a0[i];
    for (uint32 b = 1; b < nblocks; b++) {
        float *pb = Attn.p + ((size_t) kvh * nblocks + b) * Attn.per;
        float mb = pb[g * hs + 2 * lh], lb = pb[g * hs + 2 * lh + 1];
        float m2 = fmaxf(m, mb), sa = expf(m - m2), sb = expf(mb - m2);
        scale_add(out_h, sa, pb + lh * hs, sb);
        m = m2;
        sum = sa * sum + sb * lb;
    }
    HVX_Vector inv = splat_f(1.0f / sum);
    for (uint32 i = 0; i < hs / 32; i++) o[i] = VMUL(o[i], inv);
}

// Where thread t of T starts in n items.
static uint32 span0(uint32 n, uint32 t, uint32 T) { return n * t / T; }
static uint32 span1(uint32 n, uint32 t, uint32 T) { return n * (t + 1) / T; }

// Whether this position streams. Every thread gets the same answer: it
// depends only on the model, the position and the VTCM size.
static int use_stream(uint32 npos, uint32 T) {
    if (!Attn.p || Attn.g > ATTN_GMAX || Attn.g != M.heads / M.kv_heads || npos < ATTN_STREAM_AT) return 0;
    uint32 nblocks = (npos + ATTN_BLOCK - 1) / ATTN_BLOCK;
    if (nblocks > Attn.blocks) return 0;
    // The last block's keys are rounded up to a vector. That has to stay
    // inside the key row.
    uint32 t0 = (nblocks - 1) * ATTN_BLOCK, len = npos - t0, kpos = (len + 31u) & ~31u;
    if (t0 + kpos > M.att_stride) return 0;
    if ((uint64) M.att_stride * 4 > 0xffffffu || (uint64) M.kv_dim * 4 > 0xffffffu) return 0;
    uint32 bytes = 0;
    char *v = vtcm_get(&bytes);
    uint32 slot = 2u * M.hs * ATTN_BLOCK * 4u;
    return v && T && slot && bytes / slot >= T;
}

// DMA one block of K/V head `kvh` (positions t0 .. t0+len) into `dst`.
// Keys are strided by att_stride and land packed; values are strided by
// kv_dim. Returns the value copy, which finishes after the key copy.
static void *dma_block(uint32 t, uint32 l, uint32 kvh, uint32 t0, uint32 len, uint32 kpos, char *dst) {
    const uint32 hs = M.hs;
    const float *kt = M.kc + ((size_t) l * M.kv_dim + kvh * hs) * M.att_stride + t0;
    const float *vc = M.vc + ((size_t) l * M.seq_len + t0) * M.kv_dim + kvh * hs;
    dma_copy_2d(t, dst, kt, kpos * 4, M.att_stride * 4, kpos * 4, hs);
    return dma_copy_2d(t, dst + hs * ATTN_BLOCK * 4, vc, hs * 4, M.kv_dim * 4, hs * 4, len);
}

// Long context: each (K/V head, block of ATTN_BLOCK positions) is one
// piece of work, spread over the threads, so a layer's 2 or 3 K/V heads
// do not leave threads idle the way 14 query heads on 6 threads did.
// The block's keys and values are DMA'd into that thread's VTCM slots
// once and used by every query head of that K/V head (7 on Qwen, 3 on
// SmolLM2). A whole head is 1 MB at depth 2048 and is not copied in one
// go; a block is 2·hs·256 floats. Products own VTCM outside this window
// (VTCM_SHARE per thread). Attention runs only between the q/k/v product
// and wo, and both barriers below are the step's, so every thread enters.
static void attend_stream(qurt_barrier_t *bar, uint32 T, uint32 l, uint32 pos, const float *qkv, float *out, uint32 t) {
    const uint32 hs = M.hs, g = Attn.g, npos = pos + 1;
    const uint32 nblocks = (npos + ATTN_BLOCK - 1) / ATTN_BLOCK;
    const uint32 items = M.kv_heads * nblocks;
    const uint32 k_bytes = hs * ATTN_BLOCK * 4, slot_bytes = 2 * k_bytes;
    uint32 bytes = 0;
    char *vtcm = vtcm_get(&bytes);
    uint32 bufs = bytes / slot_bytes / T >= 2 ? 2 : 1;
    float scores[ATTN_BLOCK] __attribute__((aligned(128)));
    void *pend[2] = {NULL, NULL};
    uint32 mine0 = span0(items, t, T), mine1 = span1(items, t, T);

    // Issue piece `p` into slot `s`: keys then values. The value copy is
    // the one to wait on.
    #define ISSUE(p, s)                                                                                                      \
        do {                                                                                                                 \
            uint32 _kvh = (p) / nblocks, _b = (p) % nblocks, _t0 = _b * ATTN_BLOCK;                                          \
            uint32 _len = npos - _t0 < ATTN_BLOCK ? npos - _t0 : ATTN_BLOCK;                                                 \
            uint32 _kpos = (_len + 31u) & ~31u;                                                                              \
            pend[s] = dma_block(t, l, _kvh, _t0, _len, _kpos, vtcm + ((size_t) t * bufs + (s)) * slot_bytes);                \
        } while (0)
    #define COMPUTE(p, s)                                                                                                    \
        do {                                                                                                                 \
            uint32 _kvh = (p) / nblocks, _b = (p) % nblocks, _t0 = _b * ATTN_BLOCK;                                          \
            uint32 _len = npos - _t0 < ATTN_BLOCK ? npos - _t0 : ATTN_BLOCK;                                                 \
            uint32 _kpos = (_len + 31u) & ~31u;                                                                              \
            char *_base = vtcm + ((size_t) t * bufs + (s)) * slot_bytes;                                                     \
            float *_part = Attn.p + ((size_t) _kvh * nblocks + _b) * Attn.per;                                               \
            const float *_q = qkv + _kvh * g * hs;                                                                          \
            for (uint32 _lh = 0; _lh < g; _lh++)                                                                             \
                block_head(_q + _lh * hs, (const float *) _base, _kpos, (const float *) (_base + k_bytes), _len,             \
                    _part + _lh * hs, &_part[g * hs + 2 * _lh], &_part[g * hs + 2 * _lh + 1], scores, t);                    \
        } while (0)

    if (mine0 < mine1) ISSUE(mine0, 0);
    for (uint32 p = mine0; p < mine1; p++) {
        uint32 s = (p - mine0) % bufs;
        uint64 c0 = HAP_perf_get_pcycles(), q0 = HAP_perf_get_qtimer_count();
        if (bufs > 1 && p + 1 < mine1) ISSUE(p + 1, (s + 1) % bufs);
        dma_wait_copy(pend[s]);
        cy_add(t, CY_DMA, c0, q0);
        COMPUTE(p, s);
        if (bufs == 1 && p + 1 < mine1) {
            c0 = HAP_perf_get_pcycles(), q0 = HAP_perf_get_qtimer_count();
            ISSUE(p + 1, 0);
            cy_add(t, CY_DMA, c0, q0);
        }
    }
    #undef ISSUE
    #undef COMPUTE
    // Every partial is written. Merge this thread's query heads, then
    // wait again so the next token may reuse the partials. Both waits are
    // in this span: an idle thread's time is the barrier, not the merge.
    uint64 c0 = HAP_perf_get_pcycles(), q0 = HAP_perf_get_qtimer_count();
    qurt_barrier_wait(bar);
    for (uint32 h = span0(M.heads, t, T); h < span1(M.heads, t, T); h++) merge_head(h, nblocks, out + h * hs);
    qurt_barrier_wait(bar);
    cy_add(t, CY_MERGE, c0, q0);
}

// One token at `pos`: the prefetch path for a short context (this thread's
// query heads, each reading DDR), or the streamed path once the context
// is long enough that a second read of each K/V head is the cost. Decode
// and the few-token pass both come through here, so they stay bit for bit.
static void attend_token(qurt_barrier_t *bar, uint32 T, uint32 l, uint32 pos, const float *qkv, float *out, uint32 t) {
    if (!use_stream(pos + 1, T)) {
        for (uint32 h = span0(M.heads, t, T); h < span1(M.heads, t, T); h++)
            attend(l, h, pos, qkv + h * M.hs, out + h * M.hs, M.att + (size_t) t * M.att_stride);
        return;
    }
    attend_stream(bar, T, l, pos, qkv, out, t);
}

// SwiGLU into the gate half of hb for blocks (vectors) b0 .. b1: silu(g) u
// with silu(g) = g / (1 + e^-g).
static inline HVX_Vector silu_mul(HVX_Vector g, HVX_Vector u) {
    return VMUL(VMUL(g, vrecip(VADD(splat_f(1.0f), vexp(Q6_Vsf_vfneg_Vsf(g))))), u);
}

static void swiglu(uint32 b0, uint32 b1) {
    HVX_Vector *g = (HVX_Vector *) M.hb;
    const HVX_Vector *u = (const HVX_Vector *) (M.hb + M.hid);
    for (uint32 b = b0; b < b1; b++) g[b] = silu_mul(g[b], u[b]);
}

AEEResult aether_dsp_math_test(remote_handle64 h, uint32 op, const float *a, int an, const float *b, int bn, float *y, int yn) {
    (void) h;
    if (an != bn || an != yn || an % 32 || op > 1) return AEE_EBADPARM;
    for (int i = 0; i < an; i += 32) {
        HVX_Vector va = *(const HVX_UVector *) (a + i), vb = *(const HVX_UVector *) (b + i);
        *(HVX_UVector *) (y + i) = op == 0 ? vexp(va) : silu_mul(va, vb);
    }
    return AEE_SUCCESS;
}

struct step {
    uint32 pos, threads, dma, next;
    uint64 t0;
    qurt_barrier_t bar;
    // Each thread's best logit and its index, for the argmax.
    float best[MAX_THREADS];
    uint32 at[MAX_THREADS];
    // A prefill's tokens, whether it makes every token's logits, and the
    // first error of its HMX products.
    uint32 n, all;
    int err;
    char *vtcm;
};

// The argmax of the vocab logits lg: each thread over its run, then thread
// 0 over theirs (the first of equal logits, as the CPU's argmax), into
// st->next; all threads have finished when it returns.
static void argmax(struct step *st, const float *lg, uint32 t) {
    const uint32 T = st->threads;
    uint32 b = M.vocab * t / T;
    for (uint32 i = b + 1; i < M.vocab * (t + 1) / T; i++)
        if (lg[i] > lg[b]) b = i;
    st->best[t] = lg[b], st->at[t] = b;
    qurt_barrier_wait(&st->bar);
    if (t == 0) {
        uint32 w = 0;
        for (uint32 u = 1; u < T; u++)
            if (st->best[u] > st->best[w]) w = u;
        st->next = st->at[w];
    }
}

// The product of the runs (up to nr; each its rows of the output, one after
// another) and M.xq (nb blocks) into out, each thread a run of whole pairs
// of row groups of each run.
static void product(const struct run *r, uint32 nr, uint32 nb, float *out, int acc, int fetch, uint32 t, uint32 T) {
    for (uint32 i = 0; i < nr && r[i].rows; out += r[i].rows, i++) {
        uint32 groups = r[i].rows / 32;
        qx_rows(r[i].fmt, r[i].w, M.xq, nb, (HVX_Vector *) out, pair_split(groups, t, T), pair_split(groups, t + 1, T), fetch, acc, t);
    }
}

// thread t's share of `n` items.
#define FROM(n) ((n) * t / T)
#define TO(n) ((n) * (t + 1) / T)

static void decode_thread(void *arg, uint32 t) {
    struct step *st = arg;
    const uint32 dim = M.dim, hid = M.hid, kv = M.kv_dim, hs = M.hs, pos = st->pos, T = st->threads;
    qurt_barrier_t *bar = &st->bar;
    uint64 last = HAP_perf_get_qtimer_count();
    if (t == 0) M.prof[P_START] += last - st->t0;
    // After a barrier: the stage that just ended took until now.
#define MARK(p)                                       \
    if (t == 0) {                                     \
        uint64 now = HAP_perf_get_qtimer_count();     \
        M.prof[p] += now - last;                      \
        last = now;                                   \
    }
#define WAIT(p)                \
    qurt_barrier_wait(bar);    \
    MARK(p)
    // Product k (0 q/k/v, 1 wo, 2 gate/up, 3 down, 4 classifier; up to
    // nr runs) by DMA or l2fetch.
#define ROWS(k, runs, nr, x_nb, out, acc) product(runs, nr, x_nb, out, acc, st->dma >> (k) & 1 ? FETCH_DMA : FETCH_L2, t, T)
    for (uint32 l = 0; l < M.layers; l++) {
        norm_quantize(M.xb, M.x, M.l[l].attn_norm, dim, FROM(dim / 32), TO(dim / 32));
        WAIT(P_NORM1);
        ROWS(0, M.l[l].qkv, 3, dim / 32, M.qkv, 0);
        WAIT(P_QKV);
        qkv_post(l, pos, M.qkv, M.rot, FROM(dim / 2 + kv / 2), TO(dim / 2 + kv / 2), FROM(kv), TO(kv));
        WAIT(P_POST);
        // This thread's query heads on a short context, or every thread on
        // the same streamed blocks once the context is long. Then this
        // thread's outputs, whole blocks (hs is a multiple of 32), quantized
        // for wo.
        attend_token(bar, T, l, pos, M.qkv, M.xb, t);
        quantize_q8_0(M.xb, dim / 32, FROM(M.heads) * hs / 32, TO(M.heads) * hs / 32, M.xq);
        WAIT(P_ATTEND);
        ROWS(1, &M.l[l].wo, 1, dim / 32, M.x, 1);
        WAIT(P_WO);
        norm_quantize(M.xb, M.x, M.l[l].ffn_norm, dim, FROM(dim / 32), TO(dim / 32));
        WAIT(P_NORM2);
        ROWS(2, M.l[l].gu, 2, dim / 32, M.hb, 0);
        WAIT(P_GATE_UP);
        // SwiGLU, quantized for down; a run of blocks each.
        swiglu(FROM(hid / 32), TO(hid / 32));
        quantize_q8_0(M.hb, hid / 32, FROM(hid / 32), TO(hid / 32), M.xq);
        WAIT(P_SWIGLU);
        ROWS(3, &M.l[l].wdown, 1, hid / 32, M.x, 1);
        WAIT(P_DOWN);
    }
    norm_quantize(M.xb, M.x, M.final_norm, dim, FROM(dim / 32), TO(dim / 32));
    WAIT(P_FINAL_NORM);
    ROWS(4, &M.cls, 1, dim / 32, M.logits, 0);
    WAIT(P_CLS);
    argmax(st, M.logits, t);
    MARK(P_ARGMAX);
#undef ROWS
#undef WAIT
#undef MARK
}

AEEResult aether_dsp_model_decode(remote_handle64 h, uint32 pos, uint32 threads, uint32 dma, uint32 *next, uint64 *us) {
    (void) h;
    if (!M.loaded || pos >= M.seq_len || threads < 1 || threads > MAX_THREADS) return AEE_EBADPARM;
    // The CPU wrote x and the RoPE table.
    cpu_wrote(M.x, 4 * M.dim);
    cpu_wrote(M.rot, 4 * M.hs);
    cpu_wrote(M.prof, 8 * N_STAGES);
    cy_cpu_in();
    dsp_busy();
    dma_gen++;
    struct step st = {pos, threads, dma, 0, HAP_perf_get_qtimer_count()};
    qurt_barrier_init(&st.bar, threads);
    int err = run_threads(threads, decode_thread, &st, us);
    dsp_free();
    qurt_barrier_destroy(&st.bar);
    // The CPU reads the logits, and writes x next time.
    cpu_reads(M.logits, 4 * M.vocab);
    cpu_reads(M.prof, 8 * N_STAGES);
    cy_cpu_out();
    cpu_will_write(M.x, 4 * M.dim);
    *next = st.next;
    return err;
}

// The RMS norm of the row x (n values) times w into o.
static void norm_row(float *o, const float *x, const float *w, uint32 n) {
    float ss = 0;
    for (uint32 i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + M.eps);
    for (uint32 i = 0; i < n; i++) o[i] = w[i] * (x[i] * inv);
}

// A prefill: tokens at positions pos .. pos + n - 1 through every layer at
// once, row i of each buffer for token i. The products are matrix x matrix
// on HMX (hmx.c: the activations go in as fp16, not Q8_0), the rest as in
// decode_thread but over the rows: norms and RoPE a run of tokens each,
// attention a run of (token, head) pairs each (token i over positions up
// to pos + i: causal), SwiGLU a run of blocks each.
static void prefill_thread(void *arg, uint32 t) {
    struct step *st = arg;
    const uint32 dim = M.dim, hid = M.hid, kv = M.kv_dim, hs = M.hs, qkv_n = dim + 2 * kv, pos = st->pos, n = st->n, T = st->threads;
    qurt_barrier_t *bar = &st->bar;
    char *vtcm = st->vtcm;
    // A product on all threads; thread 0 keeps the first error.
#define HMX(w, rows, nb, x, xs, out, os, acc)                                                                    \
    do {                                                                                                        \
        struct hmx_product p = {w, rows, nb, x, xs, n, out, os, acc, vtcm};                                     \
        int e = hmx_rows(&p, t, T, bar);                                                                        \
        if (e && !st->err) st->err = e;                                                                         \
    } while (0)
    for (uint32 l = 0; l < M.layers; l++) {
        for (uint32 i = FROM(n); i < TO(n); i++) norm_row(M.xb + (size_t) i * dim, M.x + (size_t) i * dim, M.l[l].attn_norm, dim);
        qurt_barrier_wait(bar);
        HMX(M.l[l].qkv[0].w, qkv_n, dim / 32, M.xb, dim, M.qkv, qkv_n, 0);
        for (uint32 i = FROM(n); i < TO(n); i++) qkv_post(l, pos + i, M.qkv + (size_t) i * qkv_n, M.rot + (size_t) i * hs, 0, dim / 2 + kv / 2, 0, kv);
        qurt_barrier_wait(bar);
        // Every token's keys and values are in the cache now. One token at
        // a time, the same attend_token as decode, so a long token streams
        // and a short one stays on the prefetch path.
        for (uint32 i = 0; i < n; i++) attend_token(bar, T, l, pos + i, M.qkv + (size_t) i * qkv_n, M.xb + (size_t) i * dim, t);
        qurt_barrier_wait(bar);
        HMX(M.l[l].wo.w, dim, dim / 32, M.xb, dim, M.x, dim, 1);
        for (uint32 i = FROM(n); i < TO(n); i++) norm_row(M.xb + (size_t) i * dim, M.x + (size_t) i * dim, M.l[l].ffn_norm, dim);
        qurt_barrier_wait(bar);
        HMX(M.l[l].gu[0].w, 2 * hid, dim / 32, M.xb, dim, M.hb, 2 * hid, 0);
        for (uint32 k = FROM(n * (hid / 32)); k < TO(n * (hid / 32)); k++) {
            HVX_Vector *g = (HVX_Vector *) (M.hb + (size_t) (k / (hid / 32)) * 2 * hid) + k % (hid / 32);
            *g = silu_mul(*g, g[hid / 32]);
        }
        qurt_barrier_wait(bar);
        HMX(M.l[l].wdown.w, dim, hid / 32, M.hb, 2 * hid, M.x, dim, 1);
    }
    const float *last;
    if (st->all) {
        // Every token's logits, on HMX.
        for (uint32 i = FROM(n); i < TO(n); i++) norm_row(M.xb + (size_t) i * dim, M.x + (size_t) i * dim, M.final_norm, dim);
        qurt_barrier_wait(bar);
        HMX(M.cls.w, M.vocab, dim / 32, M.xb, dim, M.logits, M.vocab, 0);
        last = M.logits + (size_t) (n - 1) * M.vocab;
    } else {
        // The last token's only, as in decode_thread.
        norm_quantize(M.xb, M.x + (size_t) (n - 1) * dim, M.final_norm, dim, FROM(dim / 32), TO(dim / 32));
        qurt_barrier_wait(bar);
        q4_0_rows(M.cls.w, M.xq, dim / 32, (HVX_Vector *) M.logits, pair_split(M.vocab / 32, t, T), pair_split(M.vocab / 32, t + 1, T), st->dma >> 4 & 1 ? FETCH_DMA : FETCH_L2, 0, t);
        qurt_barrier_wait(bar);
        last = M.logits;
    }
    argmax(st, last, t);
#undef HMX
}

AEEResult aether_dsp_model_prefill(remote_handle64 h, uint32 pos, uint32 n, uint32 threads, uint32 all, uint32 *next, uint64 *us) {
    (void) h;
    if (!M.loaded || !M.q4_0_only || !n || n > M.batch || pos + n > M.seq_len || threads < 1 || threads > MAX_THREADS) return AEE_EBADPARM;
    if (!hmx_fits(M.dim / 32, n) || !hmx_fits(M.hid / 32, n)) return AEE_EBADPARM;
    uint32 bytes = 0;
    char *vtcm = vtcm_get(&bytes);
    if (!vtcm || bytes < (8u << 20)) return AEE_ENOMEMORY;
    // The CPU wrote the tokens' embeddings and RoPE tables.
    cpu_wrote(M.x, 4 * n * M.dim);
    cpu_wrote(M.rot, 4 * n * M.hs);
    cy_cpu_in();
    dsp_busy();
    dma_gen++;
    struct step st = {pos, threads, DMA_DEFAULT, 0, HAP_perf_get_qtimer_count()};
    st.n = n, st.all = all, st.vtcm = vtcm;
    qurt_barrier_init(&st.bar, threads);
    int err = run_threads(threads, prefill_thread, &st, us);
    dsp_free();
    qurt_barrier_destroy(&st.bar);
    if (!err) err = st.err;
    uint32 rows = all ? n : 1;
    cpu_reads(M.logits, 4 * rows * M.vocab);
    cy_cpu_out();
    cpu_will_write(M.x, 4 * n * M.dim);
    *next = st.next;
    return err;
}

// A pass of a few tokens (at most MAX_FEW) at positions pos .. pos + n - 1,
// every token's logits: speculative decoding's check of a certain token
// and its drafts. As decode_thread, the products Q4_0 x Q8_0 on HVX, but
// each group of weights read once for all n tokens (qx_rows_n, by DMA into VTCM), so n
// tokens cost little more than one where reading the weights is the cost.
// Row i of each buffer is token i, its Q8_0 in xq row i. Norms, RoPE and
// quantizing a run of blocks of every token each; attention a run of
// (token, head) pairs each (token i over positions up to pos + i), SwiGLU
// a run of (token, block) pairs each.
static void product_n(const struct run *r, uint32 nr, uint32 nb, const char *const *xs, uint32 n, float *out, uint32 os, int acc, uint32 t, uint32 T) {
    for (uint32 i = 0; i < nr && r[i].rows; out += r[i].rows, i++) {
        uint32 groups = r[i].rows / 32;
        qx_rows_n(r[i].fmt, r[i].w, xs, n, nb, (HVX_Vector *) out, os / 32, pair_split(groups, t, T), pair_split(groups, t + 1, T), FETCH_DMA, acc, t);
    }
}

static void few_thread(void *arg, uint32 t) {
    struct step *st = arg;
    const uint32 dim = M.dim, hid = M.hid, kv = M.kv_dim, hs = M.hs, qkv_n = dim + 2 * kv, pos = st->pos, n = st->n, T = st->threads;
    qurt_barrier_t *bar = &st->bar;
    const char *xs[MAX_FEW];
    for (uint32 i = 0; i < n; i++) xs[i] = M.xq + (size_t) i * M.xq_stride;
#define XQ(i) (M.xq + (size_t) (i) * M.xq_stride)
    // The runs (up to nr) times every token's xq, a run of pairs of groups
    // of each per thread, by DMA; token i's into out + i * os floats.
#define ROWS_N(runs, nr, nb, out, os, acc) product_n(runs, nr, nb, xs, n, out, os, acc, t, T)
#define NORM(w)                                                                                                         \
    for (uint32 i = 0; i < n; i++) norm_quantize_to(M.xb + (size_t) i * dim, M.x + (size_t) i * dim, w, dim, FROM(dim / 32), TO(dim / 32), XQ(i))
    for (uint32 l = 0; l < M.layers; l++) {
        NORM(M.l[l].attn_norm);
        qurt_barrier_wait(bar);
        ROWS_N(M.l[l].qkv, 3, dim / 32, M.qkv, qkv_n, 0);
        qurt_barrier_wait(bar);
        for (uint32 i = 0; i < n; i++) qkv_post(l, pos + i, M.qkv + (size_t) i * qkv_n, M.rot + (size_t) i * hs, FROM(dim / 2 + kv / 2), TO(dim / 2 + kv / 2), FROM(kv), TO(kv));
        qurt_barrier_wait(bar);
        // Every token's keys and values are in the cache now. Same
        // attend_token as decode (bit for bit with a decode step at this
        // position), then this thread's heads quantized for wo.
        for (uint32 i = 0; i < n; i++) {
            float *o = M.xb + (size_t) i * dim;
            attend_token(bar, T, l, pos + i, M.qkv + (size_t) i * qkv_n, o, t);
            quantize_q8_0(o, dim / 32, span0(M.heads, t, T) * hs / 32, span1(M.heads, t, T) * hs / 32, XQ(i));
        }
        qurt_barrier_wait(bar);
        ROWS_N(&M.l[l].wo, 1, dim / 32, M.x, dim, 1);
        qurt_barrier_wait(bar);
        NORM(M.l[l].ffn_norm);
        qurt_barrier_wait(bar);
        ROWS_N(M.l[l].gu, 2, dim / 32, M.hb, 2 * hid, 0);
        qurt_barrier_wait(bar);
        for (uint32 k = FROM(n * (hid / 32)); k < TO(n * (hid / 32)); k++) {
            uint32 i = k / (hid / 32), b = k % (hid / 32);
            float *row = M.hb + (size_t) i * 2 * hid;
            HVX_Vector *g = (HVX_Vector *) row + b;
            *g = silu_mul(*g, g[hid / 32]);
            quantize_q8_0(row, hid / 32, b, b + 1, XQ(i));
        }
        qurt_barrier_wait(bar);
        ROWS_N(&M.l[l].wdown, 1, hid / 32, M.x, dim, 1);
        qurt_barrier_wait(bar);
    }
    NORM(M.final_norm);
    qurt_barrier_wait(bar);
    ROWS_N(&M.cls, 1, dim / 32, M.logits, M.vocab, 0);
#undef NORM
#undef ROWS_N
#undef XQ
}

AEEResult aether_dsp_model_few(remote_handle64 h, uint32 pos, uint32 n, uint32 threads, uint64 *us) {
    (void) h;
    if (!M.loaded || !n || n > MAX_FEW || pos + n > M.seq_len || threads < 1 || threads > MAX_THREADS) return AEE_EBADPARM;
    // The CPU wrote the tokens' embeddings and RoPE tables.
    cpu_wrote(M.x, 4 * n * M.dim);
    cpu_wrote(M.rot, 4 * n * M.hs);
    cy_cpu_in();
    dsp_busy();
    dma_gen++;
    struct step st = {pos, threads, 0, 0, HAP_perf_get_qtimer_count()};
    st.n = n;
    qurt_barrier_init(&st.bar, threads);
    int err = run_threads(threads, few_thread, &st, us);
    dsp_free();
    qurt_barrier_destroy(&st.bar);
    cpu_reads(M.logits, 4 * n * M.vocab);
    cy_cpu_out();
    cpu_will_write(M.x, 4 * n * M.dim);
    return err;
}
