# S6: Supra2.5 decode past 1000 tok/s

Plan: `aether-infer/plans/supra.md` (S6). `ab.sh`: two binaries alternating
in two rounds of opposite order, decode tg128 at depth 0, `-r 3`, cores
from 0xfc, each run after zone0 < 40 °C, screen held awake. iQOO 13 on
battery (AC powered: false), caps cpu2 2.40 / cpu7 2.44 GHz at every run.

## Step 1: a thread's whole row range in one kernel call (`d20a0df`)

`s1-rows.log`. Old `aether-supra` (`100b525`), new `aether-s1`. Outputs
identical (200 greedy tokens on each file).

| File | Threads | Old (tok/s) | New (tok/s) | Change |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | 1 | 221, 224 | 285, 285 | +28 % |
| | 2 | 402, 400 | 506, 508 | +26 % |
| | 6 | 925, 952 | 1054, 1041 | +10 % |
| Q8_0 | 1 | 270, 271 | 373, 376 | +38 % |
| | 2 | 453, 456 | 549, 544 | +20 % |
| | 6 | 650, 651 | 653, 649 | 0 |
| Q4_K_M (Q5_0 mix) | 1 | 182 | 223 | +22 % |
| | 2 | 332 | 384 | +16 % |
| | 6 | 777 | 813 | +5 % |

The Q4_K_M rows are one round each: the log stops after the new binary's
first round (the wireless adb link dropped).

## Step 2: the token broadcast once per product (`4d40a82`)

Each group of four rows re-broadcast every token block (4 `dup`s a block);
now the caller makes the broadcast once and the Q4_0 / Q5_0 kernels load
it. `s2-gemv8.log`: an 8-row kernel first (spilled registers; Q8_0 1 thread
422 -> 241): dropped. `s2-broadcast.log`: broadcast per thread in a
thread-local buffer, 6 threads −8 %: dropped. `s2-broadcast-shared.log`:

| File | Threads | Step 1 | Step 2 |
| --- | ---: | ---: | ---: |
| Q4_0 | 1 | 283, 286 | 326, 325 |
| | 2 | 511, 509 | 565, 569 |
| | 6 | 1065, 1066 | 1056, 1060 |
| Q4_K_M | 1 | 223, 225 | 241, 239 |
| | 6 | 832, 834 | 830, 822 |

## Step 3: decode blocks of at least 64 KB (`f3eda52`)

Rows went out in 16-row blocks (about 6 KB on Supra's 640-wide matrices),
eight a thread. Sweep of blocks a thread, Q4_0, 6 threads (one binary,
an environment variable, two rounds): 8 → 1088 / 1068, 4 → 1172 / 1139,
2 → 1175 / 1133, 1 → 1231 / 1212, 16 → 1021 / 1045. One block a thread
everywhere (`s3-contiguous.log`, `s3-lfm3.log`): Supra Q4_0 +15 % at 6
threads but Q8_0 −8 % at 2 and LFM3-1.7B −12 % at 6. The kept rule: as
many blocks a thread (1-8) as keep each at least 64 KB (`s3-floor-supra.log`,
caps 2.40 GHz, Q4_0 / Q8_0 at 6 threads: none 1060 / 691, 32 KB 1119 / 710,
64 KB 1179 / 734, 128 KB 1102-1191 / 724). `s3-final.log`, 6 threads:

| Model | Step 2 | Step 3 |
| --- | ---: | ---: |
| Supra Q4_0 | 1086, 1129 | 1235, 1232 |
| LFM3-1.7B Q4_K_M | 48.6, 49.7 | 48.1, 49.4 |

## The f16 KV cache (PR #9, `kv-f16`)

`kvf16-pr9.log`: PR #9's own binary (no S6 steps), Supra Q4_0, f32 and f16
alternated (the first f32 round ran throttled, 1.79 / 1.02 GHz, and is
discarded). 6 threads, depth 256: f32 684-709, f16 759-775 (+10 %); depth
0 even; 1 thread, depth 256: f16 about 3 % slower.

`kvf16-merged.log` (`kvf16-merged.sh`, 2026-11-27): `kv-f16` merged into
`supra` (`9a932b3`) against `supra` before it (`aether-s3`, `f3eda52`),
order s3 / merged f32 / f16 / f16 / merged f32 / s3, depths 0 and 256, cores
2-7. Q4_0 at caps 2.75 / 2.65 GHz throughout:

| Q4_0 | s3 (f32) | merged f32 | merged f16 |
| --- | ---: | ---: | ---: |
| 1 thread, d0 | 376, 375 | 373, 363 | 376, 374 |
| 1 thread, d256 | 263, 278 | 271, 272 | 263, 267 |
| 6 threads, d0 | 1280, 1267 | 1265, 1274 | 1283, 1301 |
| 6 threads, d256 | 768, 795 | 800, 802 | **943, 935** |

Q8_0 (first three runs at 2.75 GHz; the last two dropped to 2.40 and are
not compared): 6 threads d256 s3 553, merged f32 551, f16 622 / 621
(+13 %); 1 thread d256 f16 283-285 against 300-301 (−6 %). The merge
leaves f32 where it was; the f16 cache pays where attention reads a deep
cache on many threads, and costs a little on one.

Quality (`kvf16-ppl/`, wikitext-2 40 × 512, KL against llama.cpp's F16
logits, llama.cpp `d02b4c0d6`): F16 PPL 26.2073 (f32 cache) / 26.2076 (f16;
llama.cpp, whose cache is f16, 26.2076); KLD Q4_K_M 0.02644 / 0.02637, Q4_0
0.15550 / 0.15567; same top token Q4_0 76.5 / 76.4 %.

Closed (2026-11-28), found on the way: `kvcache::tests::
f16_cache_decode_matches_batch` failed on the phone with the f32 cache,
on Gemma 3 270M and then LFM3 (on `kv-f16` alone too). First read as a NEON
attention suspect; not a kernel fault. The phone's decode and prompt paths
differ by design (K-quant prompts quantize activations to Q8_0 for
`smmla`, decode to Q8_K for `sdot`), and a last-bit difference can round a
Q8_0 block's f16 scale the other way; the test's random tokens amplify it
(on the PC the f16 cache alone moves Gemma 3's logits by 0.26). The
`smmla` path equals dequant(Q8_0(x)) @ W from gguf-py on LFM3's matrices;
LFM3 Q4_K_M KL against BF16, 20 chunks, on the phone: 0.1150 with `smmla`,
0.1182 without (the PC: 0.1150). The test now bounds the f16 run against
the f32 run's own distance (aether-infer `084d49e`); plans/supra.md has the
detail.
