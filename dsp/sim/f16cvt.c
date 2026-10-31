// Store and reload f32 through the f16 helpers in dsp/f16_hvx.h.
// The host test sim_f16_kv_convert_matches_numpy compares both
// conversions with numpy's astype(float16).
//
//   f16cvt IN OUT n
//
// IN: n f32, n a multiple of 64. OUT: n f16 from the vector store,
// n f16 from the scalar store, then n f32 read back in lane order.
#include <stdio.h>
#include <stdlib.h>

#include "../f16_hvx.h"

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
    if (argc != 4) {
        printf("usage: f16cvt IN OUT n\n");
        return 1;
    }
    uint32 n = atoi(argv[3]);
    if (!n || n % 64) return 1;
    float *x = load(argv[1], (size_t) n * 4);
    uint16 *vec = memalign(128, (size_t) n * 2);
    uint16 *sca = memalign(128, (size_t) n * 2);
    float *back = memalign(128, (size_t) n * 4);
    for (uint32 i = 0; i < n; i += 64) {
        HVX_Vector h = f32_to_hf(*(HVX_Vector *) (x + i), *(HVX_Vector *) (x + i + 32));
        *(HVX_Vector *) (vec + i) = h;
        HVX_Vector lo, hi;
        hf_to_f32(h, &lo, &hi);
        *(HVX_Vector *) (back + i) = lo;
        *(HVX_Vector *) (back + i + 32) = hi;
        for (uint32 k = 0; k < 64; k++) sca[i + k] = f32_to_f16_bits(x[i + k]);
    }
    FILE *f = fopen(argv[2], "wb");
    fwrite(vec, 2, n, f);
    fwrite(sca, 2, n, f);
    fwrite(back, 4, n, f);
    fclose(f);
    printf("ok\n");
    return 0;
}
