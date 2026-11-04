# Plan: a smaller KV cache on the CPU (f16, then Q8_0)

Written 2026-11-04 on branch `phone-prep`. Not started. It touches
`model/transformer.rs` and the attention kernels, and `plans/kquant-decode.md`
touches the matmul dispatch next to them, so it starts after `kquant-decode`
is merged.

## 1. Why

- **The CPU cache is f32.** `State::key_cache` / `value_cache` are
  `AlignedBuf` of f32 (`src/model/transformer.rs`). llama.cpp's default cache
  is f16, and our accuracy numbers have been compared against its f16 cache all
  along.
- **Decode slows with depth.** Qwen2.5-0.5B Q4_0 on the CPU: 180 tok/s at
  depth 0, 159 at depth 512 (results/phase-7.md, 2026-10-31). Attention reads
  every cached position every token, so its bytes grow with the context while
  the weights stay fixed. Halving the cache halves those bytes.
- **Memory.** Per position, K + V over all layers (calculated):

  | model | layers with a cache × KV width | f32 | f16 | at 4096 positions, f32 → f16 |
  | --- | --- | --- | --- | --- |
  | Qwen2.5-0.5B | 24 × 128 | 24,576 B | 12,288 B | 100.7 → 50.3 MB |
  | LFM3-1.7B | 6 full × 512 (+5 sliding rings of 512 rows) | 24,576 B | 12,288 B | 100.7 → 50.3 MB (+ rings) |
  | Gemma 4 E2B (planned) | 3 full × 512 + 12 sliding × 256 rings | — | — | `plans/gemma4.md` G5 |

- **It has been done once already.** The DSP keeps an f16 K/V cache
  (results/phase-l5.md, "f16 K/V cache", 2026-10-31). Perplexity moved by
  less than 0.1 on every file, and the CPU-side conversion cost showed up in
  the store (+90–123 µs on a 64-token prefix), not in decode.

## 2. Scope

- **In:** f16 K and V on the CPU for decode, the batched prompt pass,
  flash-decoding parts and the sliding rings; then Q8_0 K and V (llama.cpp's
  `-ctk q8_0 -ctv q8_0`) as a second step, measurement-gated.
- **Out:** the GPU (its cache is its own) and the DSP (already f16).
- **Kept:** f32 stays selectable (`--kv f32|f16|q8_0`, default decided in C4),
  and with f32 every model is bit-identical to today.

## 3. Steps

Each step is one commit with its check.

### C0. Baseline (phone)

`bench_suite --tg 64 --depth 0,512,2048,4096` on Qwen Q4_0 and LFM3 Q4_K_M,
cooled, battery, caps recorded; a profile at depth 2048 for the attention
share. Same session: `llama-bench -d 0,512,2048,4096` with its default f16
cache and with `-ctk q8_0 -ctv q8_0 -fa 1`.

### C1. Conversion kernels against numpy

f32 → f16 on store and f16 → f32 on load (NEON `fcvtn` / `fcvtl`; the phone
has fp16; the PC's scalar path uses the same rounding). Golden:
`numpy.astype(float16)`, tails and half-ulp cases included, as
`sim_f16_kv_convert_matches_numpy` did for the DSP.

### C2. The f16 cache through every attention path

The store in the decode step, the prompt pass, the sliding rings, LFM3's
conv rollback (positions only, no conv state), and every attention kernel
(`attention_head`, `attention_group`, `attention_group_part`, the window
variants) reading f16. Check: the existing attention goldens with the inputs
rounded to f16 first (numpy), within the existing bounds; decode against
batched prompt, as today.

### C3. Accuracy

PPL 20 × 512 and KL for Qwen Q4_0, Qwen Q4_K_M and LFM3 Q4_K_M with
`--kv f16`, against today's f32-cache digits and llama.cpp's (same f16
cache). LFM3 context 2048 against HF, as A5. Expect movement under 0.1, as on
the DSP; anything larger is a bug until shown otherwise.

### C4. Speed and the default

C0's runs again with `--kv f16`, alternated with f32 in the same session.
f16 becomes the default if it is not slower at depth 0 and faster from depth
512 up; the results write-up says which.

### C5. Optional: Q8_0 K and V

Blocks of 32 with one scale, as llama.cpp's Q8_0 cache. The score `q · k`
can then use the integer dot products (`sdot`, as the matmuls do) with q
rounded to Q8_0 per head. Check: llama.cpp's quantizer bytes for the store
(`tools/gen_quant.py`), PPL and KL against llama.cpp's `-ctk q8_0 -ctv q8_0`
run from C0. Kept only if C4's method shows a gain over f16.

## 4. What the review will check

- f32 mode bit-identical to today for every model.
- Every attention path reads the same format it was written in (sliding
  rings and rollback included).
- PPL and KL against llama.cpp's matching cache type, not just against our
  own f32.
- Same-session, alternated timings with conditions recorded.

## 5. References

- results/phase-l5.md, "f16 K/V cache" (the DSP's version)
- llama.cpp: `--cache-type-k`, `--cache-type-v` (`llama-bench -ctk/-ctv`),
  `ggml_cpy` to f16 / q8_0 in the KV store
