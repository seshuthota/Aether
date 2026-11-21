# P4b: the fused GeGLU + quant_x (2026-11-21, PC)

`geglu_q8.comp` writes the gated activations as `down`'s int8 codes, scale
and sums (the `quant_x` PAIRS layout) instead of `geglu.comp` writing the
f32 row and a second dispatch quantizing it. It runs where `down`'s product
is int8 (a prompt pass with `q4_0_gemm_q8`/`qx_gemm_q8`); decode and the
f32 / no-int-dot paths keep the f32 row.

## Checks

- `gemma_shaders_match_numpy`: 53 cases now. The 52 `gelu_` cases run
  `geglu` then `quant_x` (PAIRS) and compare every code, the f32 scale and
  both sums with `geglu_q8`'s output: **bit-identical**. A 224-value case
  (7 whole 32-value pieces in a 256-thread tile; the shader must stop at
  `nb`, not at the tile) is bit-identical too.
- `vk_gemma` Gemma 4 E2B Q4_0, 4 chunks, `batch`: **43.6921 / 0.342415 /
  79.608 %**, the recorded number to the digit. The Q4_K_M and LFM3 Q4_K_M
  numbers are P1b's (`../results/prompt-speed/p1b-kquant-host-2026-11-21.md`).
- `gemma4_prompt_matches_decode` and `gemma4_q4_k_m_prompt_matches_decode`
  pass.

## PC timing (sanity, not the phone)

`profile_gemma_pp` 512, Gemma 4 Q4_K_M, Intel RPL-P:

| | f32 (`f32_gemm`) | int8, fused |
| --- | --- | --- |
| tok/s | 2.9 | **60.3** |
| gate/up | 93.53 s | 3.71 s |
| down | 49.72 s | 2.97 s |
| q/k/v | 13.97 s | 0.60 s |
| wo | 12.94 s | 0.57 s |
| attention | 0.28 s | 0.28 s |

The int8 change is 20.6× here; the fusion's own share is the `down`
input's `quant_x` dispatch (about 1.5 % of the pass on the Adreno). The
phone must measure it: `profile_gemma_pp` 512/2048, old vs new, in
`plans/prompt-speed.md`'s phone block.
