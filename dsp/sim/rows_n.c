// Runs qx_rows_n_on (dsp/qx_hvx.h) on the Hexagon simulator for the
// tests sim_*_rows_n_matches_reference (src/kernels/hexagon.rs), which
// write the inputs, build this with the SDK's clang and compare the
// output with numpy's.
//
//   rows_n W X OUT rows nb tokens per_call fmt
//
// W: the matrix in format fmt's layout (qx_hvx.h's F_ enum); X: `tokens` activations of 40·nb
// bytes in its Q8_0 layout; OUT: tokens × rows f32 written here. The
// tokens go per_call (≤ MAX_FEW) at a time.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../qx_hvx.h"

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
    if (argc != 9) {
        printf("usage: rows_n W X OUT rows nb tokens per_call fmt\n");
        return 1;
    }
    uint32 rows = atoi(argv[4]), nb = atoi(argv[5]), tokens = atoi(argv[6]), per = atoi(argv[7]), fmt = atoi(argv[8]);
    uint32 groups = (rows + 31) / 32;
    if (per < 1 || per > MAX_FEW || fmt >= N_FORMATS) return 1;
    const HVX_Vector *w = load(argv[1], (size_t) groups * nb * piece_vectors(fmt) * 128);
    const char *x = load(argv[2], (size_t) tokens * 40 * nb);
    HVX_Vector *out = memalign(128, (size_t) tokens * groups * 128);
    const char *xs[MAX_FEW];
    for (uint32 t0 = 0; t0 < tokens; t0 += per) {
        uint32 n = tokens - t0 < per ? tokens - t0 : per;
        for (uint32 t = 0; t < n; t++) xs[t] = x + (size_t) (t0 + t) * 40 * nb;
        qx_rows_n_on(fmt, w, xs, n, nb, out + (size_t) t0 * groups, groups, 0, groups, 0);
    }
    FILE *f = fopen(argv[3], "wb");
    for (uint32 t = 0; t < tokens; t++) fwrite((const float *) (out + (size_t) t * groups), 4, rows, f);
    fclose(f);
    printf("ok\n");
    return 0;
}
