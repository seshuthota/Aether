# P5: the prompt's attention as plain products, SUB = 8, the int8 PLE projection (2026-11-15)

Branch `prompt-speed-2` (aether-infer), on `measure-and-p6-design`.
iQOO 13 (SM8750, Adreno 830), Gemma 4 E2B Q4_0, `profile_gemma_pp` (one
warm pass after an untimed batch), on battery, screen on. Every run started
at 39–40 °C (zone0) with the 2.40/2.44 GHz caps (`scripts/phone_ab.py`:
variants alternated, the order reversed each round, cooled before each run).

Variants:

- `p3`: `c0e8cbe` (P3 as merged): one-pass attention (`attention_flash_gemma`,
  SUB 4), the per-layer batch projection in f32 `qx_gemm`.
- `new`: this branch, one-pass attention SUB 4, the per-layer projection int8.
- `sub8`: the same with 8-position tiles.
- `gemm`: the attention as products (`Options::attn_gemm`): f16 gathers,
  `S = K·Q` (f32 sums), masked softmax into P (f16), `O = P·V` (f16 sums of 16).
- `gemm16`: the same with the scores summed in f16 (dropped: no faster).

## 2048 tokens (`p5-attn-ab-2048-2026-11-15.txt`, 3 runs each, battery 62 → 60 %)

Medians (min–max), ms unless marked:

| | p3 | new | sub8 | gemm | gemm16 |
| --- | --- | --- | --- | --- | --- |
| tok/s | 171.4 (169–173) | 175.2 (173–178) | 172.2 (172–175) | **275.3 (272–289)** | 272.2 (269–274) |
| wall | 11951 | 11686 | 11894 | **7439** | 7523 |
| GPU stages | 11190 | 10983 | 11089 | 6843 | 6842 |
| rest (CPU, submits, waits) | 762 | 703 | 805 | 596 | 681 |
| attention, all | 4696 | 4697 | 4744 | **1101** | 1110 |
| – gathers | | | | 363 | 363 |
| – scores | | | | 286 | 296 |
| – softmax | | | | 205 | 204 |
| – values | | | | 248 | 247 |
| gate/up | 2888 | 2879 | 2894 | 2757 | 2754 |
| down | 1641 | 1633 | 1645 | 1512 | 1510 |
| per-layer proj (batch) | 222.0 | **45.6** | 45.7 | 44.4 | 44.3 |

- **The products are 4.3× faster than the one-pass shader** (4697 → 1101
  ms; ~0.31 TFLOP/s on the ~3.4e11 FLOP estimate, which counts no masked
  work). The pass goes 175 → 275 tok/s at 2048.
- **SUB = 8 does nothing** (4744 vs 4697 ms): halving the barriers was not
  the one-pass shader's limit.
- f16 score sums are no faster than f32 (296 vs 286 ms), so the scores stay
  f32 (the accurate one).
- The per-layer projection: 222 → 45.6 ms.
- The products' buffers: 30.0 MiB at seq_len 2048 (q 2, keys 2, values 2,
  scores 16, P 8 MiB); they grow as 14 KiB a position of `seq_len`
  (58 MiB at 4096, `aether serve`'s default; 114 MiB at 8192).

## The gathers tiled (`p5-attn-ab-2048-tiled-2026-11-15.txt`, 3 runs each)

q and the keys are transposed through 32 × 32 shared-memory tiles (reads
and writes contiguous); the values keep the simple gather.

| | flash (SUB 4) | products | products, tiled gathers |
| --- | --- | --- | --- |
| tok/s | 174.1 | 273.6 | **284.7** |
| attention, all | 4701 | 1099 | **798** |
| – gathers | | 362 | **62** |
| – scores / softmax / values | | 286 / 204 / 247 | 285 / 203 / 248 |

Attention is now 5.9× faster than the one-pass shader. One `tiled` run was
lost to an adb drop (exit 255, no output): its medians are of 2 runs.

Accuracy (host GPU, `vk_gemma batch`, 4 chunks, same build): one-pass
43.6280 / KLD 0.344709 / top 80.294 %; products 43.6921 / **0.342415** /
79.608 %. `gemma4_prompt_matches_decode` with f32 products: 5–7e-4 (one-pass
2–4e-4); 16 numpy cases worst 8.5e-4 of max |v|.

## 512 tokens (`p5-attn-ab-512-2026-11-15.txt`, 3 runs each, battery 59 → 58 %)

Medians (min–max), ms unless marked:

| | p3 | new | sub8 | gemm | tiled |
| --- | --- | --- | --- | --- | --- |
| tok/s | 232.1 (222–235) | 233.2 (226–236) | 229.9 (225–231) | 290.7 (290–293) | **300.4 (297–306)** |
| wall | 2206 | 2196 | 2227 | 1762 | **1705** |
| attention, all | 476.1 | 475.8 | 477.0 | 162.7 | **101.9** |
| – gathers | | | | 72.9 | **12.4** |
| – scores / softmax / values | | | | 36.9 / 25.9 / 26.9 | 36.8 / 25.8 / 26.8 |

The products are 4.7× faster than the one-pass shader at 512 (476 → 102
ms), and the pass 233 → 300 tok/s. SUB = 8 is again no gain (477.0 against
475.8). The tiled gathers are 6× the untiled ones (73 → 12 ms), as at 2048.
All three tiled rounds ran (the 2048 session lost one to an adb drop).

