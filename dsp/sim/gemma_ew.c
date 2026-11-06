// Runs Gemma 4's element-wise HVX steps (dsp/gemma_hvx.h) on the Hexagon
// simulator for sim_gemma_elementwise_matches_numpy (src/kernels/
// hexagon.rs), which writes the inputs (f32, each padded with zeros to
// whole vectors), runs this and compares the output with the numpy
// goldens of tools/gen_kernels.py.
//
//   gemma_ew gelu G U OUT n
//   gemma_ew softcap X OUT n cap
//   gemma_ew norm X OUT heads hs eps                 (no gain, per head)
//   gemma_ew qknorm Q K V QN KN ROT OUT nq hs pairs eps
//                    (OUT: q's nq heads ×√hs, then k, then v, each head
//                     padded to whole vectors)
//   gemma_ew normadd X T W OUT rows n scale eps       (rows of n, padded)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../gemma_hvx.h"

static uint32 pad(uint32 n) { return NV(n) * 32; }

static float *load(const char *path, uint32 floats) {
    float *p = memalign(128, (size_t) floats * 4);
    FILE *f = fopen(path, "rb");
    if (!p || !f || fread(p, 4, floats, f) != floats) {
        printf("cannot read %u floats of %s\n", (unsigned) floats, path);
        exit(1);
    }
    fclose(f);
    return p;
}

static void save(const char *path, const float *p, uint32 floats) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(p, 4, floats, f) != floats) {
        printf("cannot write %s\n", path);
        exit(1);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    const char *op = argc > 1 ? argv[1] : "";
    if (!strcmp(op, "gelu") && argc == 6) {
        uint32 n = atoi(argv[5]);
        float *g = load(argv[2], pad(n)), *u = load(argv[3], pad(n));
        gelu_rows(g, u, n);
        save(argv[4], g, pad(n));
    } else if (!strcmp(op, "softcap") && argc == 6) {
        uint32 n = atoi(argv[4]);
        float *x = load(argv[2], pad(n));
        softcap_rows(x, n, atof(argv[5]));
        save(argv[3], x, pad(n));
    } else if (!strcmp(op, "norm") && argc == 7) {
        uint32 heads = atoi(argv[4]), hs = atoi(argv[5]);
        float *x = load(argv[2], heads * pad(hs));
        for (uint32 h = 0; h < heads; h++) rmsnorm_hvx(x + h * pad(hs), x + h * pad(hs), NULL, hs, atof(argv[6]));
        save(argv[3], x, heads * pad(hs));
    } else if (!strcmp(op, "qknorm") && argc == 13) {
        uint32 nq = atoi(argv[9]), hs = atoi(argv[10]), pairs = atoi(argv[11]);
        float eps = atof(argv[12]);
        float *q = load(argv[2], nq * hs), *k = load(argv[3], hs), *v = load(argv[4], hs);
        float *qn = load(argv[5], hs), *kn = load(argv[6], hs), *rot = load(argv[7], hs);
        for (uint32 h = 0; h < nq; h++) head_norm_rope(q + h * hs, qn, hs, pairs, rot, sqrtf((float) hs), eps);
        head_norm_rope(k, kn, hs, pairs, rot, 1.0f, eps);
        head_norm_rope(v, NULL, hs, 0, rot, 1.0f, eps);
        float *out = memalign(128, (size_t) (nq + 2) * hs * 4);
        memcpy(out, q, (size_t) nq * hs * 4);
        memcpy(out + nq * hs, k, (size_t) hs * 4);
        memcpy(out + (nq + 1) * hs, v, (size_t) hs * 4);
        save(argv[8], out, (nq + 2) * hs);
    } else if (!strcmp(op, "normadd") && argc == 10) {
        uint32 rows = atoi(argv[6]), n = atoi(argv[7]);
        float *x = load(argv[2], rows * pad(n)), *t = load(argv[3], rows * pad(n)), *w = load(argv[4], pad(n));
        for (uint32 r = 0; r < rows; r++) norm_add(x + r * pad(n), t + r * pad(n), w, n, atof(argv[8]), atof(argv[9]));
        save(argv[5], x, rows * pad(n));
    } else {
        printf("usage: see the head of dsp/sim/gemma_ew.c\n");
        return 1;
    }
    printf("ok\n");
    return 0;
}
