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
