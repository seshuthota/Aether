// What dsp/aether_dsp.c and dsp/model.c share: shared buffers, cache
// maintenance, DSP threads, and the Q4_0 x Q8_0 matrix-vector product.
#ifndef AETHER_DSP_H
#define AETHER_DSP_H

#include "AEEStdDef.h"
#include "hexagon_types.h"

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
// (added to what is there with `accumulate`). With `prefetch`, the
// next two groups are fetched into the L2 while two are computed.
void q4_0_rows(const HVX_Vector *w, const char *xq, uint32 nb, HVX_Vector *out, uint32 g0, uint32 g1, int prefetch, int accumulate);

// Where thread t of `threads` starts in `groups` row groups: an even
// group, so groups pair up.
static inline uint32 pair_split(uint32 groups, uint32 t, uint32 threads) {
    return t == threads ? groups : (groups / 2 * t / threads) * 2;
}

#endif
