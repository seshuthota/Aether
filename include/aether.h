/* Aether: a from-scratch LLM inference engine. C interface.
 *
 * Load a model, stream generated text through a callback, close it.
 * Every function returns an error code (AETHER_OK = 0) instead of crashing;
 * aether_last_error() describes the last failure on the calling thread.
 * A session must not be used from two threads at the same time.
 */
#ifndef AETHER_H
#define AETHER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AETHER_OK          0
#define AETHER_ERR_ARGUMENT 1  /* null / non-UTF-8 argument, prompt too long */
#define AETHER_ERR_LOAD    2   /* model file missing or unreadable */
#define AETHER_ERR_PANIC   3   /* internal error (a bug); see aether_last_error */

typedef struct AetherSession AetherSession;

typedef struct {
    uint32_t prefill_tokens;
    double prefill_ms;
    uint32_t decode_tokens;
    double decode_ms;
} AetherStats;

/* Receives `len` bytes of UTF-8 text (a piece may end inside a multi-byte
 * character; the next piece completes it). Return non-zero to continue,
 * 0 to stop. */
typedef int (*AetherTokenCallback)(const uint8_t *bytes, size_t len, void *user);

/* Loads a .gguf file, Hugging Face checkpoint directory or llama2.c .bin.
 * cores: e.g. "2-7" (one thread per listed core), or NULL / "" for one
 * thread. max_seq: longest prompt + reply in tokens. */
int aether_open(const char *path, const char *cores, uint32_t max_seq, AetherSession **out);

/* Loads a --pure Q4_0 .gguf for decoding on the GPU (Vulkan). cores as in
 * aether_open: if given, a second copy of the model on those CPU cores runs
 * each prompt as one batch and hands its KV cache to the GPU (about 7x
 * faster prompts, twice the memory); NULL / "" runs the prompt through the
 * GPU token by token. Fails with AETHER_ERR_LOAD if the library was built
 * without the vulkan feature, there is no usable GPU, or a matrix is not
 * Q4_0. */
int aether_open_gpu(const char *path, const char *cores, uint32_t max_seq, AetherSession **out);

/* Generates a reply. chat != 0 wraps the prompt in the ChatML template.
 * temperature 0 = greedy. stats may be NULL. */
int aether_generate(AetherSession *session, const char *prompt, int chat,
                    float temperature, float top_p, uint32_t max_tokens, uint64_t seed,
                    AetherTokenCallback on_text, void *user, AetherStats *stats);

/* Prompt lookup decoding for later aether_generate calls: up to k draft
 * tokens copied from the last earlier match of the last `ngram` tokens,
 * checked in one batched pass. Faster when the reply repeats the prompt
 * (code edits, quoting). k = 0 turns it off; ngram >= 1 (2 is a good default).
 * CPU sessions only (a GPU session ignores it). */
int aether_set_lookup(AetherSession *session, uint32_t k, uint32_t ngram);

/* Message for the last error on this thread; valid until the next call. */
const char *aether_last_error(void);

/* Frees a session (NULL allowed). */
void aether_close(AetherSession *session);

#ifdef __cplusplus
}
#endif
#endif
