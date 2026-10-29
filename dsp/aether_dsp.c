// Our library on the Hexagon DSP (cDSP): the calls in aether_dsp.idl. The
// CPU opens it with remote_handle64_open (src/kernels/hexagon.rs); FastRPC
// then calls aether_dsp_open here, and every later call arrives through
// the qaic-generated aether_dsp_skel.c.

#include <stdlib.h>
#include <string.h>

#include "AEEStdErr.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include "HAP_power.h"
#include "aether_dsp.h"
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
#include "qurt_barrier.h"
#include "qurt_hvx.h"
#include "qurt_memory.h"
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
static void *mapped(int fd) {
    void *va = NULL;
    uint64 pa;
    return HAP_mmap_get(fd, &va, &pa) == 0 ? va : NULL;
}

static void release(int fd) { HAP_mmap_put(fd); }

// The DSP's data cache is write-back and does not see the CPU's writes to
// shared memory (measured: without these, the CPU read zeros where the
// DSP's last third of writes were still in its cache). Before reading what
// the CPU wrote, drop any stale lines; after writing, write them back.
static void cpu_wrote(const void *p, uint32 bytes) { qurt_mem_cache_clean((qurt_addr_t) p, bytes, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE); }
static void cpu_reads(const void *p, uint32 bytes) { qurt_mem_cache_clean((qurt_addr_t) p, bytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE); }

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

// Up to this many DSP threads; the hardware has 6 HVX contexts on v79.
#define MAX_THREADS 8
#define STACK (16 * 1024)

struct read_job {
    const HVX_Vector *v;  // this thread's slice
    uint32 nv;            // its length in vectors
    uint32 reps;
    uint32 prefetch;
    qurt_barrier_t *start;
    HVX_Vector sum;
};

// Sums the slice `reps` times, four loads in flight (separate sums), as
// whole-vector integer adds: the adds cost little next to the loads.
static void read_slice(void *arg) {
    struct read_job *j = arg;
    qurt_barrier_wait(j->start);
    HVX_Vector s0 = Q6_V_vzero(), s1 = s0, s2 = s0, s3 = s0;
    for (uint32 r = 0; r < j->reps; r++) {
        const HVX_Vector *v = j->v;
        for (uint32 i = 0; i + 4 <= j->nv; i += 4) {
            // Every 16 KB (128 vectors): fetch the next 16 KB into L2, as
            // 8 rows of 2 KB (l2fetch's control: stride, width, height).
            if (j->prefetch && i % 128 == 0 && i + 128 < j->nv) Q6_l2fetch_AP((void *) (v + i + 128), (uint64) 2048 << 32 | 2048 << 16 | 8);
            s0 = Q6_Vw_vadd_VwVw(s0, v[i]);
            s1 = Q6_Vw_vadd_VwVw(s1, v[i + 1]);
            s2 = Q6_Vw_vadd_VwVw(s2, v[i + 2]);
            s3 = Q6_Vw_vadd_VwVw(s3, v[i + 3]);
        }
    }
    j->sum = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(s0, s1), Q6_Vw_vadd_VwVw(s2, s3));
}

AEEResult aether_dsp_read_bw(remote_handle64 h, int fd, uint32 bytes, uint32 threads, uint32 reps, uint32 prefetch, uint64 *us, uint32 *check) {
    (void) h;
    if (threads < 1 || threads > MAX_THREADS || bytes % (threads * 4 * 128)) return AEE_EBADPARM;
    const HVX_Vector *v = mapped(fd);
    if (!v) return AEE_EBADPARM;
    cpu_wrote(v, bytes);
    // Calls from different CPU threads can run at once: stacks per call.
    char *stacks = memalign(128, threads * STACK);
    if (!stacks) {
        release(fd);
        return AEE_ENOMEMORY;
    }
    struct read_job jobs[MAX_THREADS];
    qurt_thread_t tid[MAX_THREADS];
    qurt_barrier_t start;
    // The workers and this thread start together.
    qurt_barrier_init(&start, threads + 1);
    uint32 per = bytes / 128 / threads;
    for (uint32 t = 0; t < threads; t++) {
        jobs[t] = (struct read_job) {v + t * per, per, reps, prefetch, &start, Q6_V_vzero()};
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, stacks + t * STACK);
        qurt_thread_attr_set_stack_size(&attr, STACK);
        qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()) - 1);
        qurt_thread_create(&tid[t], &attr, read_slice, &jobs[t]);
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
    uint32 c = 0;
    for (uint32 t = 0; t < threads; t++) {
        HVX_Vector s = jobs[t].sum;
        const uint32 *w = (const uint32 *) &s;
        for (int k = 0; k < 32; k++) c += w[k];
    }
    *check = c;
    release(fd);
    return AEE_SUCCESS;
}


AEEResult aether_dsp_refresh(remote_handle64 h, int fd, uint32 bytes) {
    (void) h;
    void *p = mapped(fd);
    if (!p) return AEE_EBADPARM;
    cpu_wrote(p, bytes);
    release(fd);
    return AEE_SUCCESS;
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
#define GROUP_BYTES (5 * 128)

struct gemv_job {
    const HVX_Vector *w;
    const int32 *xq;  // x's int8 values, read 4 at a time
    const float *xd;
    const int32 *xs;
    HVX_Vector *out;
    uint32 g0, g1, nb, prefetch;
    qurt_barrier_t *start;
};

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

static void gemv_groups(void *arg) {
    struct gemv_job *j = arg;
    qurt_barrier_wait(j->start);
    const HVX_Vector low = Q6_Vb_vsplat_R(0x0f);
    const uint32 nb = j->nb;
    uint32 g = j->g0;
    // Two groups at a time: independent work for the vector unit while
    // each vrmpy's result is still on its way, and x read once for both.
    for (; g + 2 <= j->g1; g += 2) {
        const HVX_Vector *w0 = j->w + (size_t) g * nb * 5, *w1 = w0 + (size_t) nb * 5;
        if (j->prefetch && g + 4 <= j->g1) prefetch(w0 + 10 * nb, 2 * nb * 640);
        HVX_Vector a0 = Q6_V_vzero(), a1 = a0;
        for (uint32 b = 0; b < nb; b++, w0 += 5, w1 += 5) {
            const int32 *x = j->xq + 8 * b;
            HVX_Vector xd = splat_f(j->xd[b]);
            HVX_Vector s0 = block_dot(w0, x, j->xs[b], low), s1 = block_dot(w1, x, j->xs[b], low);
            a0 = block_acc(a0, s0, w0[4], xd);
            a1 = block_acc(a1, s1, w1[4], xd);
        }
        j->out[g] = a0;
        j->out[g + 1] = a1;
    }
    for (; g < j->g1; g++) {
        const HVX_Vector *w = j->w + (size_t) g * nb * 5;
        HVX_Vector a = Q6_V_vzero();
        for (uint32 b = 0; b < nb; b++, w += 5) a = block_acc(a, block_dot(w, j->xq + 8 * b, j->xs[b], low), w[4], splat_f(j->xd[b]));
        j->out[g] = a;
    }
}

// Where thread t's groups start: an even group, so groups pair up.
static inline uint32 pair_split(uint32 groups, uint32 t, uint32 threads) {
    return t == threads ? groups : (groups / 2 * t / threads) * 2;
}

AEEResult aether_dsp_q4_0_gemv(remote_handle64 h, int w_fd, int x_fd, int out_fd, uint32 rows, uint32 nb, uint32 threads, uint32 prefetch, uint64 *us) {
    (void) h;
    uint32 groups = (rows + 31) / 32;
    if (threads < 1 || threads > MAX_THREADS) return AEE_EBADPARM;
    if (threads > groups) threads = groups;
    const HVX_Vector *w = mapped(w_fd);
    char *x = mapped(x_fd);
    HVX_Vector *out = mapped(out_fd);
    char *stacks = memalign(128, threads * STACK);
    int err = AEE_EBADPARM;
    if (!w || !x || !out || !stacks) goto done;
    cpu_wrote(x, nb * 40);
    struct gemv_job jobs[MAX_THREADS];
    qurt_thread_t tid[MAX_THREADS];
    qurt_barrier_t start;
    qurt_barrier_init(&start, threads + 1);
    for (uint32 t = 0; t < threads; t++) {
        jobs[t] = (struct gemv_job) {w, (const int32 *) x, (const float *) (x + 32 * nb), (const int32 *) (x + 36 * nb), out, pair_split(groups, t, threads), pair_split(groups, t + 1, threads), nb, prefetch, &start};
        qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, stacks + t * STACK);
        qurt_thread_attr_set_stack_size(&attr, STACK);
        qurt_thread_attr_set_priority(&attr, qurt_thread_get_priority(qurt_thread_get_id()) - 1);
        qurt_thread_create(&tid[t], &attr, gemv_groups, &jobs[t]);
    }
    qurt_barrier_wait(&start);
    uint64 t0 = HAP_perf_get_time_us();
    for (uint32 t = 0; t < threads; t++) {
        int status;
        qurt_thread_join(tid[t], &status);
    }
    *us = HAP_perf_get_time_us() - t0;
    qurt_barrier_destroy(&start);
    cpu_reads(out, groups * 128);
    err = AEE_SUCCESS;
done:
    free(stacks);
    if (w) release(w_fd);
    if (x) release(x_fd);
    if (out) release(out_fd);
    return err;
}
