/* Calls the engine through include/aether.h exactly as an app would.
 *   cc -O2 -Iinclude examples/c/smoke.c -Ltarget/release -laether -o smoke
 *   LD_LIBRARY_PATH=target/release ./smoke MODEL "prompt" [cores] */
#include <stdio.h>
#include <string.h>
#include "aether.h"

static int print_piece(const uint8_t *bytes, size_t len, void *user) {
    (void)user;
    fwrite(bytes, 1, len, stdout);
    fflush(stdout);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL PROMPT [CORES]\n", argv[0]); return 2; }
    AetherSession *s = NULL;
    if (aether_open(argv[1], argc > 3 ? argv[3] : NULL, 512, &s) != AETHER_OK) {
        fprintf(stderr, "open: %s\n", aether_last_error());
        return 1;
    }
    AetherStats st;
    int chat = strstr(argv[1], ".gguf") != NULL;
    if (aether_generate(s, argv[2], chat, 0.0f, 0.9f, 64, 1, print_piece, NULL, &st) != AETHER_OK) {
        fprintf(stderr, "generate: %s\n", aether_last_error());
        aether_close(s);
        return 1;
    }
    printf("\n[prefill %u tokens %.1f ms, decode %u tokens %.1f ms]\n", st.prefill_tokens, st.prefill_ms, st.decode_tokens, st.decode_ms);
    aether_close(s);
    return 0;
}
