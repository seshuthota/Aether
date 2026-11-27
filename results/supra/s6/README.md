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
