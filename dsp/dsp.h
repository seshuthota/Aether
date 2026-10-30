// What dsp/aether_dsp.c and dsp/model.c share: shared buffers, cache
// maintenance, DSP threads, and the Q4_0 x Q8_0 matrix-vector product.
#ifndef AETHER_DSP_H
#define AETHER_DSP_H

#include "AEEStdDef.h"
#include "hexagon_types.h"
#include "qurt_barrier.h"

// The DSP's address of a shared buffer by its fd (a reference that
// release() drops), or NULL.
void *mapped(int fd);
void release(int fd);

// The DSP's data cache is write-back and does not see the CPU's writes:
// drop stale lines before reading what the CPU wrote, write back what the
// CPU will read, and both for what each side writes in turn.
void cpu_wrote(const void *p, uint32 bytes);
void cpu_reads(const void *p, uint32 bytes);
void cpu_will_write(const void *p, uint32 bytes);

#define MAX_THREADS 8

// Runs fn(arg, t) on `threads` new DSP threads (t = 0 .. threads - 1) and
// waits for all; *us is the time from their common start to the last one's
// end. Returns 0 or an AEE error.
int run_threads(uint32 threads, void (*fn)(void *arg, uint32 t), void *arg, uint64 *us);

// Rows 32 * g0 .. 32 * g1 of W x for W Q4_0 in our layout of 32-row groups
// (see aether_dsp.c) and x quantized to Q8_0, `nb` blocks: the int8 values
// (32 per block), then the blocks' f32 scales, then the i32 sums of each
// block's values, 40 bytes per block. out[g] is group g's 32 results
// (added to what is there with `accumulate`), on thread t.
void q4_0_rows(const HVX_Vector *w, const char *xq, uint32 nb, HVX_Vector *out, uint32 g0, uint32 g1, int fetch, int accumulate, uint32 t);

// How q4_0_rows gets the weights: plain loads, l2fetch of the next two
// groups into the L2 while two are computed, or DMA of the next two into
// thread t's two slots in VTCM (falling back to l2fetch where they do not
// fit).
enum { FETCH_NONE, FETCH_L2, FETCH_DMA };

// Counted up at the start of every call that uses DMA (threads are new
// in each call, so their DMA chains start afresh).
extern uint32 dma_gen;

// VTCM (all of it, held once asked for) and its size, or NULL. Only the
// calling thread may ask, before it starts workers; q4_0_rows uses what
// was got, and falls back to l2fetch without it.
char *vtcm_get(uint32 *bytes);
// The resource context it was got with (for HAP_compute_res_hmx_lock).
extern unsigned int vtcm_ctx;
// Held for the whole of a call that uses VTCM, the DMA rings (dma_gen) or
// HMX: calls run at once on their own FastRPC threads (the tests do), and
// would overwrite each other's data there.
void dsp_busy(void);
void dsp_free(void);

// out = X Wᵀ on HMX (hmx.c): W `rows` rows of nb Q4_0 blocks in our
// layout, X `tokens` rows of nb * 32 f32 (`x_stride` floats apart), out
// `tokens` rows `out_stride` floats apart (whole groups of 32 written;
// added to what is there with `accumulate`). X and W are rounded to fp16.
struct hmx_product {
    const HVX_Vector *w;
    uint32 rows, nb;
    const float *x;
    uint32 x_stride, tokens;
    float *out;
    uint32 out_stride;
    int accumulate;
    char *vtcm;  // from vtcm_get, 8 MB
};
// Whether X fits VTCM (up to ~320 tokens at nb 152) and a group's tiles a buffer.
int hmx_fits(uint32 nb, uint32 tokens);
// Called by all T threads at once (t = 0 .. T - 1, sharing `bar`), which
// have all finished when it returns. Returns 0 or an AEE error (thread 0).
int hmx_rows(const struct hmx_product *p, uint32 t, uint32 T, qurt_barrier_t *bar);

// Where thread t of `threads` starts in `groups` row groups: an even
// group, so groups pair up.
static inline uint32 pair_split(uint32 groups, uint32 t, uint32 threads) {
    return t == threads ? groups : (groups / 2 * t / threads) * 2;
}

#endif
