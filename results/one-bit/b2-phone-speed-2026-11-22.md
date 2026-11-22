# B2 speed on the phone: Bonsai 1.7B Q1_0 vs llama.cpp

aether-infer `one-bit` `2529567` (NEON `dot1`/`dot4` + i8mm `smmla`)
against llama.cpp `d02b4c0d6` (static Android build,
`armv8.7-a+dotprod+i8mm`), same `Bonsai-1.7B-Q1_0.gguf` (231.13 MiB),
iQOO 13 (I2401). Run with `scripts/phone_ab.py` from `prompt-speed-2`:
3 rounds, the variant order reversed every other round, one job per
process, cooled before every job (zone0 < 40 °C, `quiet_therm` < 37 °C,
performance-cap floor 2.40 GHz), screen held on, settings restored after.
6 threads, cores 2–7. On battery (56 %, discharging); no AC.

## Conditions

| | value |
| --- | --- |
| zone0 at each job's start | 35–38 °C |
| quiet_therm at each job's start | 34.7–36.4 °C |
| CPU caps (`cpu2` / `cpu6` `scaling_max_freq`) | 2.75 / 2.65 GHz, the top step, all session |
| battery | 56 %, discharging, at every job |
| exits | 0 for all 15 jobs |

## End-to-end, tok/s (median of 3 alternated rounds)

| engine | pp128 | tg64 |
| --- | --- | --- |
| Aether | **210.05** (210.05, 210.42, 203.32) | **53.15** (52.95, 53.15, 53.93) |
| llama.cpp | **235.18** (238.46, 235.18, 233.77) | **55.42** (55.42, 56.55, 55.00) |
| Aether / llama | 89.3 % | 95.9 % |

Against B1's scalar Q1_0 kernel (median pp128 5.29, tg64 4.01 on the same
phone and file): **~40× prompt, ~13× decode**. Aether is now within 4 %
of llama.cpp at decode and 11 % at this prompt size.

## Sign-expansion A/B, `ffn_gate` 6144×2048 (median of 3, single core)

| kernel | registers | `SIGNS` table | registers gain |
| --- | --- | --- | --- |
| `dot4` (t=1) | 3.89 GB/s (454 µs) | 3.42 GB/s (517 µs) | +13.7 % |
| `mmla4` (t=8) | 13.74 GB/s (1031 µs) | 12.54 GB/s (1129 µs) | +9.6 % |

The register expansion (`tbl` + `cmtst` + `orr`) is the measured winner;
the table variant stays behind `internals` as the alternative.

## The same kernel through the engine, `ffn_gate` (median of 3)

| threads | t=1 | t=128 |
| --- | --- | --- |
| 1 | 3.76 GB/s (9.6 ns/256) | 12.91 GB/s (2.8 ns/256) |
| 2 | 7.49 GB/s (4.8 ns/256) | 25.31 GB/s (1.42 ns/256) |

Only two threads were used here so the A/B would stay on core 7; this is
not the engine's multi-thread ceiling.

Raw log: `b2-phone-speed-2026-11-22.txt`.
