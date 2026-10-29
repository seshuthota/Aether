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
    static char stacks[MAX_THREADS][STACK] __attribute__((aligned(128)));
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
        qurt_thread_attr_set_stack_addr(&attr, stacks[t]);
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

