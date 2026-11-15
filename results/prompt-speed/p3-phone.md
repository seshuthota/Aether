# P3 on the phone: the per-layer inputs move to the GPU (2026-11-15)

`aether-infer` P3 (`ple_finish.comp`, `ple_table_to_q8_0`, the batch
projection) against P2 `88e0d5e`. iQOO 13, Gemma 4 E2B Q4_0, Adreno 830,
cores 2–7, screen on, one process per run, Q4_0 file. The phone was on the
charger at its 90 % limit (`dumpsys` reports AC powered false at 87–90 %),
so the vendor's clock cap moved between runs; every run's start
temperature and caps are in the logs
(`p3-ab-2026-11-15.txt`, `p3-profile-2026-11-15.txt`).

## The comparison that counts: one warm pass at the same caps

`profile_gemma_pp`, same 2.40/2.44 GHz caps, both runs started below 38.5 °C:

| | 512 P2 | 512 P3 | 2048 P2 | 2048 P3 |
| --- | --- | --- | --- | --- |
| tok/s | 152.8 | **232.6** (+52 %) | 124.6 | **173.1** (+39 %) |
| GPU stages, sum | 1997 ms | 2065 ms | 10960 ms | 11239 ms |
| per-layer proj (batch) | – | 55.0 ms | – | 222.0 ms |
| per-layer inputs | – | 1.6 ms | – | 8.0 ms |
| rest (CPU, submits, waits) | 1353 ms (40.4 %) | **136 ms (6.2 %)** | 5480 ms (33.3 %) | **590 ms (5.0 %)** |
| wall | 3.35 s | 2.20 s | 16.44 s | 11.83 s |

At 1024 the P3 pass is **200.8 tok/s** (4831 ms of GPU stages, 269.7 ms of
rest, 110.9 ms projection); that is the checkpoint the plan asked for
(≥ 150). The `-p 128,512,1024,2048 -r 2` runs, same instrument for both:

| build, start caps | 128 | 512 | 1024 | 2048 | decode | Max RSS |
| --- | --- | --- | --- | --- | --- | --- |
| P2, 3.53/4.20 GHz | 174.61 | 159.09 | 137.14 | 122.15 | 17.32 | 3.38 GB |
| P3, 2.40/2.44 GHz | **275.84** | **229.55** | **197.59** | **173.46** | 19.21 | 3.41 GB |
| P3, 2.75/2.65 GHz | **295.42** | **249.67** | **216.46** | 172.50 | 19.69 | 3.41 GB |
| single-size 1024, P2 2.92/3.07 | – | – | 150.47 | – | 20.53 | – |
| single-size 1024, P3 2.75/2.65 | – | – | **223.18** | – | 22.90 | – |

The P2 full run had the *higher* cap at the start (3.53/4.20 GHz, the
phone's full clocks on those cores; 2.40/2.44 is its thermal cap), so
+58 %/+44 %/+44 %/+42 % understates P3. The matched single-size pair at
1024 (P2 at 2.92, P3 at 2.75) is +48 %.

## What changed

- The prompt batch's per-layer projection (`[n_layers · ple_dim, dim]`,
  BF16 in the file) is one Q8_0 `qx_gemm` product over the whole batch:
  55 ms at 512, 222 ms at 2048 (128 GFLOP/s; ~1.9 % of the 2048 wall).
- `ple_finish.comp`: one workgroup per (token, layer) for the shared
  gain's RMSNorm of the 1/√dim-scaled projection, the token's table row
  (× √ple_dim) and 1/√2 — exactly the CPU's `ple_inputs` arithmetic after
  its projection, including the scale inside the norm.
- The 1.3 GB table stays on the CPU; the batch's rows are dequantized into
  a 9 MB buffer (~1 µs a token). Decode still runs `ple_inputs` there, one
  token; its projection is now Q8_0 too, which is the decode gain
  (20.5 → 22.9 tok/s in the matched 1024 pair, 17.3 → 19.2–19.7 in the
  full runs).
- The "rest" is now just the per-layer submissions: 1.9 ms a submission at
  both depths. That is P4's target.

## Accuracy (host GPU, `vk_gemma batch`)

- Gemma 4 Q4_0, 1 chunk: **PPL 24.4821, KLD 0.335790, same top 82.353 %**
  (P2: 24.6278 / 0.336379; P1: 24.5408 / 0.333100; llama.cpp Q4_0:
  24.4536 / 0.33855). Q8_0 on the table's projection did not cost
  anything measurable against the BF16 base.
- LFM3-QAD-Q4_0, 20 chunks: **27.1242 / 0.181109 / 79.902 %**, unchanged
  to the digit (P3 does not touch it).
- 52 gemma shader goldens, including the two new `plefinish_` cases;
  `gemma4_ple_matches_the_cpu` compares the GPU's `ple_in` with the CPU's
  `ple_inputs` from the same Q8_0 projection (worst < 1e-4 of max |v|);
  the five `model::gpu::gemma` tests and `c_abi_gpu_gemma4` (exact text)
  pass.

## Caveat on absolute numbers

Two P2 sessions at the same reported 2.40/2.44 caps differ by 47 % at 512
(104.0 tok/s on battery at 53–62 % in `p2-phone.md`; 152.8 tok/s here with
the phone at 88–90 %). `scaling_max_freq` is only the cap: the sustained
clock depends on the battery/thermal state, and the profiles are warm
single passes where the bench includes the first pass. Only the
same-session, same-instrument pairs above are comparable.
