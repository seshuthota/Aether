// Runs q4_0_rows_n_on (dsp/q4_0_hvx.h) on the Hexagon simulator for the
// test sim_q4_0_rows_n_matches_reference (src/kernels/hexagon.rs), which
// writes the inputs, builds this with the SDK's clang and compares the
// output with numpy's.
//
//   rows_n W X OUT rows nb tokens per_call
//
// W: the matrix in q4_0_rows's layout; X: `tokens` activations of 40·nb
// bytes in its Q8_0 layout; OUT: tokens × rows f32 written here. The
// tokens go per_call (≤ MAX_FEW) at a time.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../q4_0_hvx.h"

static void *load(const char *path, size_t bytes) {
    void *p = memalign(128, bytes);
    FILE *f = fopen(path, "rb");
    if (!p || !f || fread(p, 1, bytes, f) != bytes) {
        printf("cannot read %zu bytes of %s\n", bytes, path);
        exit(1);
    }
    fclose(f);
    return p;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        printf("usage: rows_n W X OUT rows nb tokens per_call\n");
        return 1;
    }
    uint32 rows = atoi(argv[4]), nb = atoi(argv[5]), tokens = atoi(argv[6]), per = atoi(argv[7]);
    uint32 groups = (rows + 31) / 32;
    if (per < 1 || per > MAX_FEW) return 1;
    const HVX_Vector *w = load(argv[1], (size_t) groups * nb * 640);
    const char *x = load(argv[2], (size_t) tokens * 40 * nb);
    HVX_Vector *out = memalign(128, (size_t) tokens * groups * 128);
    const char *xs[MAX_FEW];
    for (uint32 t0 = 0; t0 < tokens; t0 += per) {
        uint32 n = tokens - t0 < per ? tokens - t0 : per;
        for (uint32 t = 0; t < n; t++) xs[t] = x + (size_t) (t0 + t) * 40 * nb;
        q4_0_rows_n_on(w, xs, n, nb, out + (size_t) t0 * groups, groups, 0, groups, 0);
    }
    FILE *f = fopen(argv[3], "wb");
    for (uint32 t = 0; t < tokens; t++) fwrite((const float *) (out + (size_t) t * groups), 4, rows, f);
    fclose(f);
    printf("ok\n");
    return 0;
}
