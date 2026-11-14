# P1 on the phone: int8 prompt products in GemmaGpu (2026-11-14)

iQOO 13 (SM8750, Adreno 830), Gemma 4 E2B Q4_0, cores 2–7, screen on,
`aether` 3a50c4f (f32) against 26356b5 (int8), run alternately. The phone
was on battery for the first pair and charging for the rest (recorded
below); its vendor clock caps moved with temperature and charging, so the
pairs are grouped by cap.

## GPU prompt, `aether bench -dev gpu -p 128,512,1024,2048 -n 64 -r 2`

| build | caps perf/prime | start °C | 128 | 512 | 1024 | 2048 | decode | Max RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3a50c4f f32, on battery | 2.40/2.44 GHz | 38.4 | 31.60 | 37.40 | 38.10 | 35.95 | 19.33 | 3,355,532 KiB |
| 26356b5 int8, on battery | 1.99/2.25 GHz | 42.6 | 61.48 | 62.83 | 59.47 | 56.49 | 18.52 | 3,350,352 KiB |
| 3a50c4f f32, charging | 1.79/1.96 GHz | 43.8 | 29.10 | 35.39 | 36.07 | 30.35 | 16.39 | 3,291,180 KiB |
| 3a50c4f f32, charging | 1.79/1.96 GHz | 44.6 | 25.13 | 30.31 | 30.71 | 31.92 | 16.57 | 3,292,048 KiB |
| 26356b5 int8, charging | 1.79/1.96 GHz | 44.9 | 49.49 | 50.98 | 49.15 | 46.36 | 17.12 | 3,295,880 KiB |

- Same-cap pair (1.79/1.96 GHz, charging): **1.7× / 1.44× / 1.36× / 1.45×**
  (128/512/1024/2048).
- The int8 build at the *lower* 1.99/2.25 GHz caps already beats the f32
  build at 2.40/2.44 GHz by 1.9× / 1.7× / 1.6× / 1.6×: the first pair is
  a lower bound for the gain.
- Decode is the same path in both builds; its spread (16.4–19.3) is the
  cap/thermal state, not the change.
- Peak RSS: no second copy of the weights; the differences are page-cache
  noise, the only new buffer is `xq` (2.4 MB at batch 128).

## `profile_gemma_pp`, both builds, same caps (1.79/1.96 GHz, charging)

| 512 tokens | f32 | int8 | | 2048 tokens | f32 | int8 |
| --- | --- | --- | --- | --- | --- | --- |
| tok/s | 47.9 | 77.2 | | tok/s | 34.9 | 56.3 |
| gate/up | 4238.6 ms (8.3/tok) | 1327.0 ms (2.6/tok) | | gate/up | 17570 ms (8.6/tok) | 5953.5 ms (2.9/tok) |
| down | 2267.7 ms (4.4/tok) | 1975.5 ms (3.9/tok) | | down | 15887 ms (7.8/tok) | 10807 ms (5.3/tok) |
| q/k/v | 678.3 ms | 184.9 ms | | q/k/v | 3201.4 ms | 881.1 ms |
| wo | 578.2 ms | 403.5 ms | | wo | 4047.5 ms | 2477.6 ms |
| attention + merge | 640.0 ms | 626.0 ms | | attention + merge | 6575.9 ms | 6624.6 ms |
| not GPU | 1902 ms (17.8 %) | 1906 ms (28.8 %) | | not GPU | 8751 ms (14.9 %) | 8463 ms (23.3 %) |

- 1.61× at both lengths. Products: gate/up 3.2× (512) and 3.0× (2048),
  q/k/v 3.7×, wo 1.4–1.6×, down only 1.15× (512) and 1.47× (2048):
  `down`'s input (`hg`, 12288 wide at 512 tokens) pays a quantize of 25 MB
  read + 9.4 MB write inside the stage, and its row stride (6144 B) is the
  worst of the row-major shader's cases.
- Attention did not move (P2); it is 9–18 % of the wall. The per-layer
  inputs and submissions are unchanged in absolute terms (1.9 s / 8.5 s)
  and are now 23–29 % of the wall (P3).
- The batch is still 63 at 2048 (`PARTS_BUDGET`), so the products lose
  reuse there (P2).

## Accuracy

The phone's `vk_gemma` was not run: 2 GB of RAM were free (13.1 GB of
15.1 GB used) and the KL base is 2.67 GB. The PC's check is the gate
(`p1-host.md`): Q4_0 1 chunk PPL 24.4536 → 24.5408, KL 0.33855 → 0.33310;
Q4_K_M 4 chunks 138.0869 / 0.16511, unchanged. The int8 dot products are
exact integers on both GPUs and the quantizer is the same shader, so the
device's products agree with the PC's up to the f32 accumulation order.

## Reading

P1 is a real 1.5–1.9× but not the tuned path's 4×: the row-major layout
keeps the decode copy out (the transposed `MatQ4_0T` would add 1.05 GB),
and the Adreno pays for it in the weight loads (64 lanes read 16-byte
words at row-stride). The next decision is whether to:
1. keep the row-major shader (no memory, 1.6×) and get the rest from P2–P5;
2. swap the Q4_0 matrices to the transposed layout (size-neutral) and add
   a transposed decode GEMV (decode A/B needed);
3. add the 1.05 GB transposed copy (over the plan's 300 MB line).
