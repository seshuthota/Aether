// Reference tokenizer for stories15M: llama2.c's own encode(), nothing else.
//
//   llama2c_encode tokenizer.bin < corpus.txt > encoded.txt
//
// corpus.txt holds test strings separated by the byte 0x1E (ASCII record
// separator), so a string may itself contain newlines. For each string one
// line is printed: its token ids, space separated, encoded the way run.c
// encodes a prompt (BOS = 1 prepended, no EOS).
#define TESTING
#include "run.c"

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s tokenizer.bin < corpus\n", argv[0]); return 1; }
    Tokenizer t;
    build_tokenizer(&t, argv[1], 32000);

    // Read all of stdin.
    size_t cap = 1 << 20, len = 0, got;
    char* buf = malloc(cap);
    while ((got = fread(buf + len, 1, cap - len - 1, stdin)) > 0) {
        len += got;
        if (len + 1 == cap) { cap *= 2; buf = realloc(buf, cap); }
    }
    buf[len] = 0x1E; // terminate the last record

    int* ids = malloc((len + 3) * sizeof(int));
    char* start = buf;
    for (size_t i = 0; i <= len; i++) {
        if (buf[i] != 0x1E) { continue; }
        buf[i] = '\0';
        int n = 0;
        encode(&t, start, 1, 0, ids, &n);
        for (int k = 0; k < n; k++) { printf(k ? " %d" : "%d", ids[k]); }
        printf("\n");
        start = buf + i + 1;
    }
    free_tokenizer(&t);
    return 0;
}
