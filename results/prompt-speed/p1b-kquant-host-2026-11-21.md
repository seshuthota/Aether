# P1b: the K-quant matrices on the int8 GPU prompt (2026-11-21, PC)

The phone is away this session. Everything below is the PC's Intel RPL-P
(`Intel(R) Graphics (RPL-P)`), one big job at a time; no phone numbers.

Branch `prompt-speed-kquant`, off `prompt-speed-2`. The change: every
non-Q4_0 matrix of a Q4_K_M file (Q4_K, Q6_K, Q5_0, Q8_0) is stored once in
`qx_gemm_q8`'s paired layout (`MatQT::new_paired`); the prompt's products
are `qx_gemm_q8` (int8 dot products, x quantized with `PAIRS`), mixed q/k/v
stacks split with its output stride, and decode uses the new
`qx_gemv_q8t.comp` on the same copy (x quantized too). Q4_0 stays in
`MatQ4_0T`/`q4_0_gemm_q8`, and `Options::f32_gemm` (or a GPU without int8
dot products) keeps `qx_gemm`/`qx_gemv` on `qx_gemv`'s layout.

The criterion for the layout was memory: option (a), one paired copy, is
**+86.9 MB** of layer weights for Gemma 4 E2B Q4_K_M (1,167.0 → 1,253.9 MB;
computed from the file's matrices at MatQ/MatQT's per-piece sizes). Option
(b), a second paired copy beside `MatQ`, would have been **+1,253.9 MB**;
not shipped.

## Accuracy (Gemma 4 E2B Q4_K_M, vk_gemma)

The recorded 138.0870 / 0.16511 (4 chunks) is the pre-P4 f32 number; see
`p4-sync-host-2026-11-21.md` for the P4 queue corruption this session found
first (the same build gives 392–405 on one chunk until the wait-before-submit
fix).

| mode | PPL | KL | same top | wall |
| --- | --- | --- | --- | --- |
| 1 chunk, `batch-f32` (old f32 products) | 50.9307 | 0.146345 | 85.882 % | 210.7 s |
| 1 chunk, `batch` (new int8) | 50.8264 | 0.148650 | 85.490 % | 36.8 s |
| 4 chunks, `batch-f32` | 137.7794 | 0.162738 | 84.706 % | 843.6 s |
| 4 chunks, `batch` | 137.9446 | 0.169960 | 85.294 % | 147.3 s |
| 1 chunk, decode (old path) | 51.0949 | 0.149804 | 85.882 % | 66.3 s |
| 1 chunk, decode (new paired GEMV) | 51.3065 | 0.152005 | 85.882 % | 54.4 s |

The decode KL moves +0.0022 on Gemma and +0.0010 on LFM3 (10.3576 /
0.057914 → 10.4212 / 0.058877), both inside the gate; decode is 18–20 %
faster on the PC with the int8 GEMV.

The int8 KL moves +0.0023 on one chunk and +0.0072 on four, inside the
plan's +0.01 gate; the PPL is +0.12 %.

The recorded 138.0870 / 0.16511 is P1's f32 prompt; the current f32 A/B
(`batch-f32`) is 137.7794 / 0.162738 because P3's GPU per-layer inputs
(its Q8_0 table projection, P5a) and P5b's attention products are in the
path now. The comparison that matters is on the same build.

## LFM3-1.7B-Q4_K_M

| mode | PPL | KL |
| --- | --- | --- |
| 4 chunks, `batch-f32` | 15.1771 | 0.063543 |
| 4 chunks, `batch` | 15.2087 | 0.064166 |

KL +0.0006 on LFM3; the prompt 539.7 → 91.2 s on the PC.

Peak RSS (`/usr/bin/time -v`) moved +4.9 MB on LFM3 (1,544,048 →
1,548,980 KiB) and +4.2 MB on Gemma 4 (2,290,080 → 2,294,300 KiB); the
matched GPU allocations are not all in the process RSS on the Intel
driver, so the computed weight bytes above are the GPU-side number.

## The Q4_0 files (must be bit-identical)

Gemma 4 E2B Q4_0, 4 chunks, `batch`: **43.6921 / 0.342415 / 79.608 %**,
the recorded number to the digit (the per-layer PLE products stay on the
Q4_0 int8 path, as in P1).

LFM3-1.7B-QAD-Q4_0, 20 chunks, `batch`: **27.1242 / 0.181109 / 79.902 %**,
the recorded number to the digit (567.0 s).

## Vk_ppl: the tuned path unchanged

Qwen2.5-0.5B Q4_0, 4 chunks, 512, `batch`: **15.8560** (recorded
15.8560), 175.6 tok/s; SmolLM2-135M Q4_0: **20.8721** (P1's 20.8721),
344.8 tok/s. The tuned path is unchanged.

## PC speed sanity (not the target)

`profile_gemma_pp` 512, Gemma 4 Q4_K_M, Intel RPL-P: f32 2.9 tok/s
(175.45 s) → int8 **60.3 tok/s** (8.50 s); gate/up 93.53 → 3.71 s, down
49.72 → 2.97, q/k/v 13.97 → 0.60, wo 12.94 → 0.57, attention 0.28 → 0.28.
The four-chunk wall is 843.6 → 147.3 s. The phone is the measurement that
counts (the commands are in the plan).
