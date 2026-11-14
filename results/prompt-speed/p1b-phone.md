# P1 final: the transposed layout (2026-11-14)

iQOO 13 (Adreno 830), Gemma 4 E2B Q4_0, cores 2–7, screen on, on battery
(86–90 %), three builds alternated: f32 (3a50c4f), the row-major int8
version (b95c260), and the transposed int8 version (31ccf58). Raw runs:
`p1b-layout-ab-2026-11-14.txt`, `p1b-profile-2026-11-14.txt`.

## Prompt and decode, `-r 2`, same caps (1.79/1.96 GHz)

| build | 128 | 512 | 1024 | 2048 | decode | Max RSS |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | 29.27 | 35.22 | 35.62 | 34.96 | 18.23 | 3,296,480 KiB |
| row-major int8 | 61.10 | 62.30 | 60.10 | 56.19 | 19.36 | 3,302,776 KiB |
| transposed int8 | **123.74** | **122.48** | **109.90** | **95.67** | 17.57 | 3,302,612 KiB |

- Transposed vs f32: 4.2× / 3.5× / 3.1× / 2.7×. vs row-major: 2.0× /
  2.0× / 1.8× / 1.7×.
- At 2.4–2.75 GHz (a 512-token single run): 149.9–154.2 tok/s.
- Decode: 17.57 here against 18.23 (f32) and 19.36 (row-major); the two
  qx_gemv builds differ by 6 % between themselves in this session, and at
  2.75 GHz the transposed version was 20.53 against 19.95. Unchanged
  within the spread; the decode path is a new shader (`q4_0_gemv_t`), and
  the PC's decode PPL is the recorded one to the digit.
- Max RSS: no second copy (the transposed layout pads rows to 4; the
  differences are page-cache noise).

## Profile, same caps (1.99/2.25 GHz)

| | 512 f32 | 512 int8 | 2048 f32 | 2048 int8 |
| --- | --- | --- | --- | --- |
| tok/s | 49.0 | **132.8** | 34.9 | **95.5** |
| gate/up | 4256 ms | 766 ms (5.6×) | 17568 ms | 3115 ms (5.6×) |
| down | 2275 ms | 418 ms (5.4×) | 15883 ms | 1725 ms (9.2×) |
| q/k/v | 683 ms | 125 ms (5.5×) | 3200 ms | 572 ms (5.6×) |
| wo | 580 ms | 107 ms (5.4×) | 4050 ms | 457 ms (8.9×) |
| attention + merge | 640 ms | 637 ms | 6752 ms | 6737 ms |
| not GPU | 1645 ms (15.7 %) | 1571 ms (40.7 %) | 8671 ms (14.8 %) | 7608 ms (35.5 %) |

Attention is unchanged and now the largest single stage at 2048 (30 %);
the CPU share is 35–41 %. P2 and P3 target exactly those.

## Accuracy (PC, and the phone's decode equivalence)

- Q4_0 1 chunk: PPL 24.5408, KL 0.333100 (the row-major int8 numbers; the
  f32 A/B is 24.4536 / 0.338552).
- 4-chunk decode: PPL 43.6074, KL 0.342819 — the recorded f32 values to
  the digit, same greedy text (" a young girl named Lily…").
- Q4_K_M 4 chunks: 138.0869 / 0.165110 (unchanged; no Q4_0 matrix).
- Shader goldens: `q4_0_gemv_t` against the numpy/gguf-py decode goldens;
  the int8 product goldens; LFM3 prompt/decode and transformers tests;
  `c_abi_gpu_gemma4`.

## Reading

The layout was worth 1.7–2.0× over the row-major version and 2.7–4.2×
over f32, with no extra memory. The products are no longer the wall:
attention and the CPU-side per-layer inputs are. P2 (flash attention,
batch 256) is next.
