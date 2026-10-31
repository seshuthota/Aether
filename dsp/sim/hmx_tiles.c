// Dequantizes weight groups into HMX fp16 tiles on the simulator.
// The host test sim_hmx_weight_tiles_match_gguf compares them with
// llama.cpp's gguf-py dequantizer, cast to float16.
//
//   hmx_tiles W OUT groups nb fmt
//
// W is `groups` groups of `nb` pieces in qx_hvx.h's layout. OUT is
// groups·nb tiles: 16 vectors of 64 f16 each, vector i then lane order.
#include <stdio.h>
#include <stdlib.h>

#include "../hmx_tiles.h"

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
    if (argc != 6) {
        printf("usage: hmx_tiles W OUT groups nb fmt\n");
        return 1;
    }
    uint32 groups = atoi(argv[3]), nb = atoi(argv[4]), fmt = atoi(argv[5]);
    if (!groups || !nb || fmt >= N_FORMATS) return 1;
    const HVX_Vector *w = load(argv[1], (size_t) groups * nb * piece_vectors(fmt) * 128);
    HVX_Vector *out = memalign(128, (size_t) groups * nb * 16 * 128);
    for (uint32 g = 0; g < groups; g++)
        hmx_weight_tiles(out + (size_t) g * nb * 16, w + (size_t) g * nb * piece_vectors(fmt), nb, fmt);
    FILE *f = fopen(argv[2], "wb");
    if (!f || fwrite(out, 2, (size_t) groups * nb * 16 * 64, f) != (size_t) groups * nb * 16 * 64) return 1;
    fclose(f);
    printf("ok\n");
    return 0;
}
