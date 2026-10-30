// Our library on the Hexagon DSP (cDSP): the calls in aether_dsp.idl. The
// CPU opens it with remote_handle64_open (src/kernels/hexagon.rs); FastRPC
// then calls aether_dsp_open here, and every later call arrives through
// the qaic-generated aether_dsp_skel.c.

#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_compute_res.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "aether_dsp.h"
#include "dsp.h"
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
#include "qurt_barrier.h"
#include "qurt_hvx.h"
#include "qurt_memory.h"
#include "qurt_mutex.h"
#include "qurt_thread.h"

// One HVX vector: 128 bytes, 32 floats.
#define VF 32

struct ctx {
    int unused;
};

AEEResult aether_dsp_open(const char *uri, remote_handle64 *h) {
    (void) uri;
    struct ctx *c = calloc(1, sizeof(*c));
    if (!c) return AEE_ENOMEMORY;
    *h = (remote_handle64) c;
    // A compute client, with the vector units powered up. (HVX contexts
    // themselves are handed to a thread by QuRT on first use.)
    HAP_power_request_t r;
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_set_apptype;
    r.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    int err = HAP_power_set(c, &r);
    if (err) return err;
    // Clocks: core and bus at their highest voltage corner, and no sleep
    // between calls (else every call wakes a sleeping DSP).
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_set_DCVS_v3;
    r.dcvs_v3.set_dcvs_enable = TRUE;
    r.dcvs_v3.dcvs_enable = TRUE;
    r.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    r.dcvs_v3.set_bus_params = TRUE;
    r.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.set_core_params = TRUE;
    r.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    r.dcvs_v3.set_sleep_disable = TRUE;
    r.dcvs_v3.sleep_disable = TRUE;
    err = HAP_power_set(c, &r);
    if (err) return err;
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_set_HVX;
    r.hvx.power_up = TRUE;
    err = HAP_power_set(c, &r);
    if (err) return err;
    // And the matrix unit (HMX), which is off until asked for: its
    // instructions on an unpowered unit hung the DSP.
    memset(&r, 0, sizeof(r));
    r.type = HAP_power_set_HMX;
    r.hmx.power_up = TRUE;
    return HAP_power_set(c, &r);
}

AEEResult aether_dsp_close(remote_handle64 h) {
    free((void *) h);
    return AEE_SUCCESS;
}

AEEResult aether_dsp_ping(remote_handle64 h, uint32 x, uint32 *y) {
    (void) h;
    *y = x + 1;
    return AEE_SUCCESS;
}

// c = a + b, 32 lanes at a time; the tail (n % 32) one by one. FastRPC's
// copies of the buffers need not be 128-byte aligned, so the vectors are
// loaded and stored unaligned.
AEEResult aether_dsp_add(remote_handle64 h, const float *a, int an, const float *b, int bn, float *c, int cn) {
    (void) h;
    if (an != bn || an != cn) return AEE_EBADPARM;
    int i = 0;
    for (; i + VF <= an; i += VF) {
        HVX_Vector va = *(const HVX_UVector *) (a + i);
        HVX_Vector vb = *(const HVX_UVector *) (b + i);
        *(HVX_UVector *) (c + i) = Q6_Vsf_vadd_VsfVsf(va, vb);
    }
    for (; i < an; i++) c[i] = a[i] + b[i];
    return AEE_SUCCESS;
}

// The DSP's address of a buffer the CPU mapped with fastrpc_mmap. Each
// lookup takes a reference that release() drops.
void *mapped(int fd) {
    void *va = NULL;
    uint64 pa;
    return HAP_mmap_get(fd, &va, &pa) == 0 ? va : NULL;
}

void release(int fd) { HAP_mmap_put(fd); }

// The DSP's data cache is write-back and does not see the CPU's writes to
// shared memory (measured: without these, the CPU read zeros where the
// DSP's last third of writes were still in its cache). Before reading what
// the CPU wrote, drop any stale lines; after writing, write them back.
void cpu_wrote(const void *p, uint32 bytes) { qurt_mem_cache_clean((qurt_addr_t) p, bytes, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE); }
void cpu_reads(const void *p, uint32 bytes) { qurt_mem_cache_clean((qurt_addr_t) p, bytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE); }
// For a buffer both sides write in turn: no dirty line of ours may be
// written back later over what the CPU writes next.
void cpu_will_write(const void *p, uint32 bytes) { qurt_mem_cache_clean((qurt_addr_t) p, bytes, QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE); }

AEEResult aether_dsp_add_shared(remote_handle64 h, int a_fd, int b_fd, int c_fd, uint32 n) {
    float *a = mapped(a_fd), *b = mapped(b_fd), *c = mapped(c_fd);
    int err = AEE_EBADPARM;
    if (a && b && c) {
        cpu_wrote(a, 4 * n);
        cpu_wrote(b, 4 * n);
        err = aether_dsp_add(h, a, n, b, n, c, n);
        cpu_reads(c, 4 * n);
    }
    release(a_fd);
    release(b_fd);
    release(c_fd);
    return err;
}

AEEResult aether_dsp_hvx_units(remote_handle64 h, uint32 *n) {
    (void) h;
    // Bits 8-15: the number of 128-byte HVX contexts.
    *n = (qurt_hvx_get_units() >> 8) & 0xff;
    return AEE_SUCCESS;
}

// Each DSP thread's stack; the hardware has 6 HVX contexts on v79.
#define STACK (16 * 1024)

struct spmd {
    void (*fn)(void *, uint32);
    void *arg;
    uint32 t;
    qurt_barrier_t *start;
};

static void spmd_entry(void *p) {
    struct spmd *s = p;
    qurt_barrier_wait(s->start);
    s->fn(s->arg, s->t);
}

int run_threads(uint32 threads, void (*fn)(void *, uint32), void *arg, uint64 *us) {
    if (threads < 1 || threads > MAX_THREADS) return AEE_EBADPARM;
    // Calls from different CPU threads can run at once: stacks per call.
    char *stacks = memalign(128, threads * STACK);
    if (!stacks) return AEE_ENOMEMORY;
    struct spmd jobs[MAX_THREADS];
    qurt_thread_t tid[MAX_THREADS];
    qurt_barrier_t start;
    // The workers and this thread start together.
    qurt_barrier_init(&start, threads + 1);
    for (uint32 t = 0; t < threads; t++) {
        jobs[t] = (struct spmd) {fn, arg, t, &start};
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, stacks + t * STACK);
        qurt_thread_attr_set_stack_size(&attr, STACK);
        qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()) - 1);
        qurt_thread_create(&tid[t], &attr, spmd_entry, &jobs[t]);
    }
    qurt_barrier_wait(&start);
    uint64 t0 = HAP_perf_get_time_us();
    for (uint32 t = 0; t < threads; t++) {
        int status;
        qurt_thread_join(tid[t], &status);
    }
    *us = HAP_perf_get_time_us() - t0;
    qurt_barrier_destroy(&start);
    free(stacks);
    return AEE_SUCCESS;
}

struct read_job {
    const HVX_Vector *v;
    uint32 per;  // vectors per thread
    uint32 reps;
    uint32 prefetch;
    HVX_Vector sum[MAX_THREADS];
};

// Sums thread t's slice `reps` times, four loads in flight (separate sums),
// as whole-vector integer adds: the adds cost little next to the loads.
static void read_slice(void *arg, uint32 t) {
    struct read_job *j = arg;
    HVX_Vector s0 = Q6_V_vzero(), s1 = s0, s2 = s0, s3 = s0;
    for (uint32 r = 0; r < j->reps; r++) {
        const HVX_Vector *v = j->v + t * j->per;
        for (uint32 i = 0; i + 4 <= j->per; i += 4) {
            // Every 16 KB (128 vectors): fetch the next 16 KB into L2, as
            // 8 rows of 2 KB (l2fetch's control: stride, width, height).
            if (j->prefetch && i % 128 == 0 && i + 128 < j->per) Q6_l2fetch_AP((void *) (v + i + 128), (uint64) 2048 << 32 | 2048 << 16 | 8);
            s0 = Q6_Vw_vadd_VwVw(s0, v[i]);
            s1 = Q6_Vw_vadd_VwVw(s1, v[i + 1]);
            s2 = Q6_Vw_vadd_VwVw(s2, v[i + 2]);
            s3 = Q6_Vw_vadd_VwVw(s3, v[i + 3]);
        }
    }
    j->sum[t] = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(s0, s1), Q6_Vw_vadd_VwVw(s2, s3));
}

AEEResult aether_dsp_read_bw(remote_handle64 h, int fd, uint32 bytes, uint32 threads, uint32 reps, uint32 prefetch, uint64 *us, uint32 *check) {
    (void) h;
    if (threads < 1 || threads > MAX_THREADS || bytes % (threads * 4 * 128)) return AEE_EBADPARM;
    const HVX_Vector *v = mapped(fd);
    if (!v) return AEE_EBADPARM;
    cpu_wrote(v, bytes);
    struct read_job j = {v, bytes / 128 / threads, reps, prefetch, {0}};
    int err = run_threads(threads, read_slice, &j, us);
    uint32 c = 0;
    for (uint32 t = 0; t < threads; t++) {
        const uint32 *w = (const uint32 *) &j.sum[t];
        for (int k = 0; k < 32; k++) c += w[k];
    }
    *check = c;
    release(fd);
    return err;
}

AEEResult aether_dsp_refresh(remote_handle64 h, int fd, uint32 bytes) {
    (void) h;
    void *p = mapped(fd);
    if (!p) return AEE_EBADPARM;
    cpu_wrote(p, bytes);
    release(fd);
    return AEE_SUCCESS;
}

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

// The DMA engine's one-dimensional descriptor (v73+ layout, as llama.cpp's
// backend declares it; the SDK's header is older): copy `size` bytes from
// `src` to `dst`, then go on to `next`. The engine sets `done`. With the
// bypass bits the copy does not go through the L2.
struct dma1d {
    void *next;
    uint32 size : 24;
    uint32 desc_size : 2;  // 0: one-dimensional
    uint32 dst_comp : 1;
    uint32 src_comp : 1;
    uint32 dst_bypass : 1;
    uint32 src_bypass : 1;
    uint32 order : 1;
    uint32 done : 1;
    const void *src;
    void *dst;
} __attribute__((aligned(64)));

// Every hardware thread has its own DMA queue: dmstart begins a chain of
// descriptors, dmlink appends one to the chain's tail (and the engine goes
// on to it, even if it had finished). Each thread keeps a ring of
// descriptors, so none is rewritten while the engine may still look at it,
// and appends every copy to its last one; only its first copy in each call
// (dma_gen, counted up per call) starts a chain.
#define DMA_RING 64
static struct dma1d ring[MAX_THREADS][DMA_RING];
static struct dma1d *tail[MAX_THREADS];
static uint32 ring_next[MAX_THREADS], tail_gen[MAX_THREADS];
uint32 dma_gen;

static struct dma1d *dma_copy(uint32 t, void *dst, const void *src, uint32 bytes) {
    struct dma1d *d = &ring[t][ring_next[t]++ % DMA_RING];
    *d = (struct dma1d) {NULL, bytes, 0, 0, 0, 1, 1, 0, 0, src, dst};
    asm volatile(" release(%0):at" : : "r"(d));
    if (tail[t] && tail_gen[t] == dma_gen)
        asm volatile(" dmlink(%0, %1)" : : "r"(tail[t]), "r"(d));
    else
        asm volatile(" dmstart(%0)" : : "r"(d));
    tail[t] = d, tail_gen[t] = dma_gen;
    return d;
}

static inline void dma_wait(volatile struct dma1d *d) {
    while (!d->done) {
        uint32 r;
        asm volatile(" %0 = dmpoll" : "=r"(r) : : "memory");
    }
}

// VTCM, the DSP's 8 MB of on-chip memory, held from the first use on.
// Asked for only by the thread FastRPC calls in on, before it starts
// worker threads: workers that asked at once deadlocked (all but one
// waited inside HAP_compute_res_acquire for the VTCM the one held).
static char *vtcm;
static uint32 vtcm_bytes;
unsigned int vtcm_ctx;

// Calls (each on its own FastRPC thread) may ask at the same time too.
static qurt_mutex_t vtcm_lock = QURT_MUTEX_INIT;

char *vtcm_get(uint32 *bytes) {
    qurt_mutex_lock(&vtcm_lock);
    if (!vtcm) {
        unsigned int size = 0;
        if (HAP_compute_res_query_VTCM(0, &size, NULL, NULL, NULL) || !size) goto out;
        compute_res_attr_t attr;
        HAP_compute_res_attr_init(&attr);
        HAP_compute_res_attr_set_serialize(&attr, 0);
        HAP_compute_res_attr_set_vtcm_param_v2(&attr, size, size, size);
        // With the matrix unit (HMX), which works only on VTCM.
        HAP_compute_res_attr_set_hmx_param(&attr, 1);
        unsigned int ctx = HAP_compute_res_acquire(&attr, 1000000);
        void *p = NULL;
        unsigned int got = 0;
        if (!ctx || HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &p, &got) || !p) goto out;
        vtcm = p, vtcm_bytes = got, vtcm_ctx = ctx;
    }
out:
    qurt_mutex_unlock(&vtcm_lock);
    *bytes = vtcm_bytes;
    return vtcm;
}

static qurt_mutex_t busy_lock = QURT_MUTEX_INIT;
void dsp_busy(void) { qurt_mutex_lock(&busy_lock); }
void dsp_free(void) { qurt_mutex_unlock(&busy_lock); }

void q4_0_rows(const HVX_Vector *wall, const char *xqb, uint32 nb, HVX_Vector *out, uint32 g0, uint32 g1, int fetch, int accumulate, uint32 t) {
    const int32 *xq = (const int32 *) xqb;
    const float *xd = (const float *) (xqb + 32 * nb);
    const int32 *xs = (const int32 *) (xqb + 36 * nb);
    const HVX_Vector low = Q6_Vb_vsplat_R(0x0f);
    uint32 g = g0, pair = 2 * nb * 640;
    char *v = fetch == FETCH_DMA ? vtcm : NULL;
    if (v && 2 * pair * (t + 1) <= vtcm_bytes && g + 2 <= g1) {
        // Two VTCM slots per thread: the DMA fills one with the next pair
        // of groups while the vector unit works on the other.
        char *slot[2] = {v + 2 * pair * t, v + 2 * pair * t + pair};
        uint32 n = (g1 - g) / 2;
        struct dma1d *d[2] = {dma_copy(t, slot[0], wall + (size_t) g * nb * 5, pair), NULL};
        for (uint32 p = 0; p < n; p++, g += 2) {
            if (p + 1 < n) d[(p + 1) % 2] = dma_copy(t, slot[(p + 1) % 2], wall + (size_t) (g + 2) * nb * 5, pair);
            dma_wait(d[p % 2]);
            pair_rows((const HVX_Vector *) slot[p % 2], xqb, nb, out + g, accumulate);
        }
    }
    for (; g + 2 <= g1; g += 2) {
        const HVX_Vector *w0 = wall + (size_t) g * nb * 5;
        if (fetch && g + 4 <= g1) prefetch(w0 + 10 * nb, pair);
        pair_rows(w0, xqb, nb, out + g, accumulate);
    }
    for (; g < g1; g++) {
        const HVX_Vector *w = wall + (size_t) g * nb * 5;
        HVX_Vector a = Q6_V_vzero();
        for (uint32 b = 0; b < nb; b++, w += 5) a = block_acc(a, block_dot(w, xq + 8 * b, xs[b], low), w[4], splat_f(xd[b]));
        out[g] = accumulate ? Q6_Vsf_vadd_VsfVsf(out[g], a) : a;
    }
}

// Q4_0 x Q8_0 matrix-vector product. The weights come in groups of 32
// rows; per group and block of 32 columns there are five vectors (640
// bytes): four of quants, then the 32 rows' f32 scales. Quant vector k
// holds, for row j of the group, bytes 4k..4k+3 of its block (byte i: the
// element i in the low nibble, i + 16 in the high one, as in Q4_0), so a
// 32-bit lane is one row.
//
// vrmpy multiplies each lane's 4 unsigned bytes by 4 signed bytes held in
// a scalar register (4 elements of x) and adds the sum to the lane: eight
// of them give each row's integer dot product over the block. Q4_0 stores
// q + 8 with q in -8..7, so the lane starts at -8 * (sum of the block's x
// values), and the unsigned nibbles go in as they are.
struct gemv_job {
    const HVX_Vector *w;
    const char *x;
    HVX_Vector *out;
    uint32 groups, nb, threads, prefetch;
};

static void gemv_thread(void *arg, uint32 t) {
    struct gemv_job *j = arg;
    q4_0_rows(j->w, j->x, j->nb, j->out, pair_split(j->groups, t, j->threads), pair_split(j->groups, t + 1, j->threads), j->prefetch, 0, t);
}

AEEResult aether_dsp_q4_0_gemv(remote_handle64 h, int w_fd, int x_fd, int out_fd, uint32 rows, uint32 nb, uint32 threads, uint32 prefetch, uint64 *us) {
    (void) h;
    uint32 groups = (rows + 31) / 32;
    if (threads > groups) threads = groups;
    const HVX_Vector *w = mapped(w_fd);
    char *x = mapped(x_fd);
    HVX_Vector *out = mapped(out_fd);
    int err = AEE_EBADPARM;
    if (w && x && out) {
        cpu_wrote(x, nb * 40);
        if (prefetch == FETCH_DMA) vtcm_get(&(uint32) {0});
        dsp_busy();
        dma_gen++;
        struct gemv_job j = {w, x, out, groups, nb, threads, prefetch};
        err = run_threads(threads, gemv_thread, &j, us);
        dsp_free();
        cpu_reads(out, groups * 128);
    }
    if (w) release(w_fd);
    if (x) release(x_fd);
    if (out) release(out_fd);
    return err;
}
