# P1 on the laptop: int8 prompt products for GemmaGpu (2026-11-14)

The phone is away, so P1's code is checked on the PC's GPU (Intel RPL-P)
and its accuracy gates are run there. The phone measurements
(`aether bench -dev gpu`, `profile_gemma_pp` on the Adreno 830, peak PSS)
are still to come; nothing here is a phone number.

Branch `prompt-speed`, on top of P0's `79d3876`. The change: a new shader,
`q4_0_gemm_q8r.comp`, the int8 product of `q4_0_gemm_q8.comp` over
`MatQ`'s row-major layout, so Gemma 4's prompt products get int8 dot
products without a second copy of the weights (the tuned layout,
`MatQ4_0T`, is a transpose: 1.05 GB of Q4_0 matrices here). Activations
go through the existing `quant_x.comp` (plain order), once per product
input. `Options::f32_gemm` keeps `qx_gemm` for the A/B. Q4_K_M has no Q4_0
matrix, so its products stay f32 and its numbers are unchanged; the
classifier (Q8_0) stays f32 as well.

## Accuracy (vk_gemma, PC, Intel RPL-P)

| Run | PPL | KL vs BF16 base | same top | prompt time |
| --- | --- | --- | --- | --- |
| Q4_0, 1 chunk, f32 products (`batch-f32`) | 24.4536 | 0.338552 | 82.353 % | 262 s |
| Q4_0, 1 chunk, int8 products (`batch`) | 24.5408 | 0.333100 | 82.745 % | 43 s |
| Q4_K_M, 4 chunks, f32 (int8 not used) | 138.0869 | 0.165110 | 84.706 % | 966 s |

- Q4_0: PPL +0.087 (+0.36 %), KL −0.0055 (better; the gate allows
  +0.01). The reference from `plans/gemma4-close.md`: 24.4536 / 0.33855.
- Q4_K_M: 138.0869 / 0.16511 against the recorded 138.0870 / 0.16511.
- RSS (peak, `/usr/bin/time -v`), Q4_0 1 chunk: 2,317,360 KiB f32 vs
  2,321,648 KiB int8 (+4.3 MB; the xq buffer is 2.4 MB). No weight copy.
- The tuned path: `vk_ppl` Qwen2.5-0.5B Q4_0, 4 chunks, 512, `batch`
  15.8560 (recorded 15.8560); `batch-f32` 15.8407. SmolLM2-135M Q4_0
  batch: 20.8721 (no earlier recording found; the code is untouched).
- `kernels::vulkan::tests::int8_products_match_goldens` now also checks
  `q4_0_gemm_q8r.comp` against the numpy/gguf-py goldens for every Q4_0
  case, set and accumulating, with a row stride longer than `rows`.
- `model::gpu::gemma` tests (LFM3 layout, long prompt against
  transformers, prompt vs decode) pass.

## Prompt profile (PC, `profile_gemma_pp`, Intel RPL-P)

Numbers per prompt, then per token. The PC's absolute speed is far below
the phone's (35 tok/s at 14 ms/token in the plan's profile): the per-layer
submissions cost it a lot, so only the ratios are useful.

| | 512 f32 | 512 int8 | 2048 f32 | 2048 int8 |
| --- | --- | --- | --- | --- |
| gate/up | 117.7 s (230 ms/tok) | 5.32 s (10.4) | 490.1 s (239) | 22.6 s (11.0) |
| down | 72.0 s (141) | 2.91 s (5.7) | 307.2 s (150) | 18.3 s (8.9) |
| q/k/v | 17.2 s (33.7) | 0.74 s (1.4) | 76.7 s (37.4) | 3.94 s (1.9) |
| wo | 17.9 s (35.0) | 0.73 s (1.4) | 78.6 s (38.4) | 4.60 s (2.2) |
| attention (+ merge) | 2.37 s | 2.37 s | 24.1 s | 24.2 s |
| not GPU | 1.08 s (0.5 %) | 0.96 s (7.1 %) | 4.37 s (0.4 %) | 5.59 s (6.7 %) |
| tok/s | 2.2 | 37.8 | 2.0 | 24.3 |

- gate/up 22×, down 25×, q/k/v and wo 25× at 512 on this GPU.
- Attention is now 17 % at 512 and 29 % at 2048 (it was 1–2 %): P2 is next.
- "Not GPU" is 7 % of the wall time now; P3 is next.
- The 2048 batch is 63 (`PARTS_BUDGET`), so the products lose size: P2's
  flash attention and a batch of 256 are what make 2048 scale.

One 2048 int8 profile run had an unwritten timestamp (attention
9.6e14 ms); the immediate rerun was clean (23.9 s). The Intel driver's
query pool, seen once.

## Commands

```
cargo run --release --features vulkan --example vk_gemma -- \
  testdata/gguf/gemma-4-E2B-it-Q4_0.gguf hostdata/gemma4/kl-base-bf16.bin 1 batch[-f32]
cargo run --release --features vulkan --example vk_gemma -- \
  testdata/gguf/gemma-4-E2B-it-Q4_K_M.gguf hostdata/gemma4/kl-base-bf16.bin 4 batch
cargo run --release --features vulkan --example vk_ppl -- \
  testdata/gguf/qwen2.5-0.5b-q4_0.gguf 4 512 batch[-f32]
cargo run --release --features vulkan --example profile_gemma_pp -- \
  testdata/gguf/gemma-4-E2B-it-Q4_0.gguf 512|2048 [f32]
```
