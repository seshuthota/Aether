// A whole decode step on the DSP: one call per token (`model_decode`) runs
// every layer and the classifier, on DSP threads that move through the
// layers together (each does its share of a step, then all wait at a
// barrier before the next step that needs the result).
//
// The CPU sets the model up once (`model_load`): it uploads the weights
// into one shared buffer (matrices in our Q4_0 layout, q/k/v and gate/up
// stacked as on the GPU; norms and biases in f32) and allocates a second
// shared buffer for the state (activations, KV cache, logits). `desc`
// gives the shapes and where everything is. Per token the CPU writes the
// token's embedding into x and the RoPE table (sin, cos per pair, computed
// as the CPU path computes them), and the DSP does the rest.
//
// The activations are f32; before every matrix product they are quantized
// to Q8_0, as on our CPU path and llama.cpp's.
//
// One model per DSP process (it lives in a static).

#include <math.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_perf.h"
#include "aether_dsp.h"
#include "dsp.h"
#include "hvx_hexagon_protos.h"
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
    D_XQ,          // S: Q8_0 of up to hidden
    D_QKV,         // S: dim + 2 kv_dim
    D_ATT,         // S: n_heads rows of seq_len scores, rounded up to 32
    D_HB,          // S: 2 hidden (gate, up)
    D_LOGITS,      // S: vocab
    D_KC,          // S: layers * seq_len * kv_dim
    D_VC,
    D_ROT,   // S: head_size / 2 (sin, cos) pairs
    D_PROF,  // S: N_STAGES u64, qtimer ticks spent per stage (added to)
    D_LAYER0,
};
// Per layer, from D_LAYER0 on: W offsets.
enum { L_WQKV, L_WO, L_WGU, L_WDOWN, L_ATTN_NORM, L_FFN_NORM, L_BQ, L_BK, L_BV, L_N };

#define MAX_LAYERS 64

// The stages of a step that thread 0 times (from the end of the previous
// one to when every thread has finished this one).
enum { P_NORM1, P_QKV, P_POST, P_ATTEND, P_WO, P_NORM2, P_GATE_UP, P_SWIGLU, P_DOWN, P_FINAL_NORM, P_CLS, P_ARGMAX, P_START, N_STAGES };

static struct model {
    int loaded;
    uint32 dim, hid, layers, heads, kv_heads, vocab, seq_len, att_stride, halves, hs, kv_dim;
    float eps;
    int w_fd, s_fd;
    char *w, *s;
    uint32 s_bytes;
    float *x, *xb, *qkv, *att, *hb, *logits, *kc, *vc, *rot;
    uint64 *prof;
    char *xq;
    const float *final_norm;
    const HVX_Vector *cls;
    struct {
        const HVX_Vector *wqkv, *wo, *wgu, *wdown;
        const float *attn_norm, *ffn_norm, *bq, *bk, *bv;
    } l[MAX_LAYERS];
} M;

AEEResult aether_dsp_model_load(remote_handle64 h, const uint32 *d, int n) {
    (void) h;
    if (n < D_LAYER0 || d[D_LAYERS] > MAX_LAYERS || n != D_LAYER0 + (int) d[D_LAYERS] * L_N) return AEE_EBADPARM;
    if (M.loaded) {
        release(M.w_fd);
        release(M.s_fd);
        M.loaded = 0;
    }
    M.dim = d[D_DIM], M.hid = d[D_HIDDEN], M.layers = d[D_LAYERS], M.heads = d[D_HEADS], M.kv_heads = d[D_KV_HEADS];
    M.vocab = d[D_VOCAB], M.seq_len = d[D_SEQ_LEN], M.halves = d[D_HALVES];
    memcpy(&M.eps, &d[D_EPS], 4);
    M.hs = M.dim / M.heads, M.kv_dim = M.hs * M.kv_heads;
    M.att_stride = (M.seq_len + 31) / 32 * 32;
    // Every matrix's rows must be whole groups of 32 (the products write
    // whole groups, and add them in place into x), and each head whole
    // Q8_0 blocks (the attention quantizes its heads' outputs).
    if (M.dim % 32 || M.hid % 32 || M.hs % 32 || M.hs > 256 || M.vocab % 32 || M.heads % M.kv_heads) return AEE_EBADPARM;
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
    M.final_norm = (const float *) (w + d[D_FINAL_NORM]), M.cls = (const HVX_Vector *) (w + d[D_CLS]);
    for (uint32 l = 0; l < M.layers; l++) {
        const uint32 *o = d + D_LAYER0 + l * L_N;
        M.l[l].wqkv = (const HVX_Vector *) (w + o[L_WQKV]);
        M.l[l].wo = (const HVX_Vector *) (w + o[L_WO]);
        M.l[l].wgu = (const HVX_Vector *) (w + o[L_WGU]);
        M.l[l].wdown = (const HVX_Vector *) (w + o[L_WDOWN]);
        M.l[l].attn_norm = (const float *) (w + o[L_ATTN_NORM]);
        M.l[l].ffn_norm = (const float *) (w + o[L_FFN_NORM]);
        M.l[l].bq = (const float *) (w + o[L_BQ]);
        M.l[l].bk = (const float *) (w + o[L_BK]);
        M.l[l].bv = (const float *) (w + o[L_BV]);
    }
    // VTCM for the products' DMA (see vtcm_get).
    vtcm_get(&(uint32) {0});
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

// The RMS norm of x (n values) times w into o, and quantized into M.xq,
// for blocks b0 .. b1 only (every thread sums all of x itself, so the
// threads need not wait for each other).
static void norm_quantize(float *o, const float *x, const float *w, uint32 n, uint32 b0, uint32 b1) {
    float ss = 0;
    for (uint32 i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + M.eps);
    for (uint32 i = 32 * b0; i < 32 * b1; i++) o[i] = w[i] * (x[i] * inv);
    quantize_q8_0(o, n / 32, b0, b1, M.xq);
}

// RoPE pairs p0 .. p1 of q and k (q's dim / 2 first, then k's kv_dim / 2)
// with their biases, k into the layer's cache at `pos`; and v's elements
// v0 .. v1 with their bias into the cache.
static void qkv_post(uint32 l, uint32 pos, uint32 p0, uint32 p1, uint32 v0, uint32 v1) {
    const uint32 dim = M.dim, kv = M.kv_dim, hs = M.hs, half = hs / 2;
    const float *bq = M.l[l].bq, *bk = M.l[l].bk, *bv = M.l[l].bv;
    float *q = M.qkv, *k = M.qkv + dim, *v = M.qkv + dim + kv;
    float *kc = M.kc + ((size_t) l * M.seq_len + pos) * kv, *vc = M.vc + ((size_t) l * M.seq_len + pos) * kv;
    for (uint32 i = v0; i < v1; i++) vc[i] = v[i] + bv[i];
    for (uint32 p = p0; p < p1; p++) {
        int isq = p < dim / 2;
        uint32 pp = isq ? p : p - dim / 2, hd = pp / half, i = pp % half;
        uint32 i0 = M.halves ? hd * hs + i : hd * hs + 2 * i, i1 = M.halves ? i0 + half : i0 + 1;
        float s = M.rot[2 * i], c = M.rot[2 * i + 1];
        const float *b = isq ? bq : bk;
        float *src = isq ? q : k, *dst = isq ? q : kc;
        float a = src[i0] + b[i0], e = src[i1] + b[i1];
        dst[i0] = a * c - e * s;
        dst[i1] = a * s + e * c;
    }
}

// HVX float helpers (IEEE single precision, Hexagon v73 and later).
static inline HVX_Vector splat_f(float f) {
    union {
        float f;
        int32 i;
    } u = {f};
    return Q6_V_vsplat_R(u.i);
}
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

// Head h's attention over positions 0..pos into xb (its hs outputs, hs / 32
// vectors). Every row of q, the caches and the scores starts on a vector.
static void attend(uint32 l, uint32 h, uint32 pos) {
    const uint32 hs = M.hs, nv = hs / 32, kv = M.kv_dim, kvh = h / (M.heads / M.kv_heads);
    const HVX_Vector *q = (const HVX_Vector *) (M.qkv + h * hs);
    const float *kc = M.kc + (size_t) l * M.seq_len * kv + kvh * hs, *vc = M.vc + (size_t) l * M.seq_len * kv + kvh * hs;
    float *a = M.att + (size_t) h * M.att_stride, scale = 1.0f / sqrtf((float) hs), max = -INFINITY;
    for (uint32 t = 0; t <= pos; t++) {
        const HVX_Vector *k = (const HVX_Vector *) (kc + (size_t) t * kv);
        HVX_Vector acc = VMUL(q[0], k[0]);
        for (uint32 i = 1; i < nv; i++) acc = VADD(acc, VMUL(q[i], k[i]));
        a[t] = vsum(acc) * scale;
        max = fmaxf(max, a[t]);
    }
    // e^(s - max) over whole vectors; the lanes past pos are -inf (e^-80
    // after the clamp, and not used below).
    uint32 n = (pos + 1 + 31) / 32;
    for (uint32 t = pos + 1; t < 32 * n; t++) a[t] = -INFINITY;
    HVX_Vector *av = (HVX_Vector *) a, vmax = splat_f(max), vs = Q6_V_vzero();
    for (uint32 j = 0; j < n; j++) {
        av[j] = vexp(VSUB(av[j], vmax));
        if (j + 1 < n) vs = VADD(vs, av[j]);
    }
    float sum = vsum(vs);
    for (uint32 t = 32 * (n - 1); t <= pos; t++) sum += a[t];
    float inv = 1.0f / sum;
    HVX_Vector o[8];
    for (uint32 i = 0; i < nv; i++) o[i] = Q6_V_vzero();
    for (uint32 t = 0; t <= pos; t++) {
        const HVX_Vector *v = (const HVX_Vector *) (vc + (size_t) t * kv);
        HVX_Vector p = splat_f(a[t] * inv);
        for (uint32 i = 0; i < nv; i++) o[i] = VADD(o[i], VMUL(p, v[i]));
    }
    HVX_Vector *out = (HVX_Vector *) (M.xb + h * hs);
    for (uint32 i = 0; i < nv; i++) out[i] = o[i];
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
};

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
    // Row groups of 32 per product, each thread a run of whole pairs.
    // Product k (0 q/k/v, 1 wo, 2 gate/up, 3 down, 4 classifier) by DMA or l2fetch.
#define ROWS(k, w, x_nb, out, rows, acc) q4_0_rows(w, M.xq, x_nb, (HVX_Vector *) (out), pair_split((rows) / 32, t, T), pair_split((rows) / 32, t + 1, T), st->dma >> (k) & 1 ? FETCH_DMA : FETCH_L2, acc, t)
    for (uint32 l = 0; l < M.layers; l++) {
        norm_quantize(M.xb, M.x, M.l[l].attn_norm, dim, FROM(dim / 32), TO(dim / 32));
        WAIT(P_NORM1);
        ROWS(0, M.l[l].wqkv, dim / 32, M.qkv, dim + 2 * kv, 0);
        WAIT(P_QKV);
        qkv_post(l, pos, FROM(dim / 2 + kv / 2), TO(dim / 2 + kv / 2), FROM(kv), TO(kv));
        WAIT(P_POST);
        // A run of heads each, then their outputs' blocks (whole blocks:
        // hs is a multiple of 32) quantized for wo.
        for (uint32 h = FROM(M.heads); h < TO(M.heads); h++) attend(l, h, pos);
        quantize_q8_0(M.xb, dim / 32, FROM(M.heads) * hs / 32, TO(M.heads) * hs / 32, M.xq);
        WAIT(P_ATTEND);
        ROWS(1, M.l[l].wo, dim / 32, M.x, dim, 1);
        WAIT(P_WO);
        norm_quantize(M.xb, M.x, M.l[l].ffn_norm, dim, FROM(dim / 32), TO(dim / 32));
        WAIT(P_NORM2);
        ROWS(2, M.l[l].wgu, dim / 32, M.hb, 2 * hid, 0);
        WAIT(P_GATE_UP);
        // SwiGLU, quantized for down; a run of blocks each.
        swiglu(FROM(hid / 32), TO(hid / 32));
        quantize_q8_0(M.hb, hid / 32, FROM(hid / 32), TO(hid / 32), M.xq);
        WAIT(P_SWIGLU);
        ROWS(3, M.l[l].wdown, hid / 32, M.x, dim, 1);
        WAIT(P_DOWN);
    }
    norm_quantize(M.xb, M.x, M.final_norm, dim, FROM(dim / 32), TO(dim / 32));
    WAIT(P_FINAL_NORM);
    ROWS(4, M.cls, dim / 32, M.logits, M.vocab, 0);
    WAIT(P_CLS);
    // The argmax: each thread over its run, then thread 0 over theirs (the
    // first of equal logits, as the CPU's argmax).
    uint32 b = FROM(M.vocab);
    for (uint32 i = b + 1; i < TO(M.vocab); i++)
        if (M.logits[i] > M.logits[b]) b = i;
    st->best[t] = M.logits[b], st->at[t] = b;
    qurt_barrier_wait(bar);
    if (t == 0) {
        uint32 w = 0;
        for (uint32 u = 1; u < T; u++)
            if (st->best[u] > st->best[w]) w = u;
        st->next = st->at[w];
    }
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
    dma_gen++;
    struct step st = {pos, threads, dma, 0, HAP_perf_get_qtimer_count()};
    qurt_barrier_init(&st.bar, threads);
    int err = run_threads(threads, decode_thread, &st, us);
    qurt_barrier_destroy(&st.bar);
    // The CPU reads the logits, and writes x next time.
    cpu_reads(M.logits, 4 * M.vocab);
    cpu_reads(M.prof, 8 * N_STAGES);
    cpu_will_write(M.x, 4 * M.dim);
    *next = st.next;
    return err;
}
