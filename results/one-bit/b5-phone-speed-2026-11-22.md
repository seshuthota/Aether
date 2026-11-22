# B5 phone speed: Ternary Bonsai 1.7B Q2_0 against Q1_0 and llama.cpp

aether-infer `one-bit` B5 (`aether-b5`, built from `aa8ddf4` + the Q2_0
work) against llama.cpp `d02b4c0d6`
(`/data/local/tmp/llama/llama-bench`), same GGUFs. iQOO 13 (I2401),
6 threads on cores 2–7, 3 alternated rounds, one job per process, cooled
before every job (zone0 < 40 °C, board < 37 °C, cap floor 2.40 GHz),
screen held on, settings restored. On battery (90 %, discharging).

## Conditions

The **second step only**: every one of the 24 jobs started and ended at
**2.40/2.44 GHz**, board 36.7–37.2 °C, zone0 37–42 °C. Cooling reached
board 36.9 °C and the caps still did not return to the top step
(2.75/2.65 GHz) in this session, so the absolute numbers are
second-step. All exits 0.

## pp128 / tg64, median of 3 rounds

| model | engine | pp128 | tg64 |
| --- | --- | --- | --- |
| Q1_0 1.7B | Aether | 192.16 (193.44, 191.97, 192.16) | 61.71 (61.71, 61.74, 61.50) |
| Q1_0 1.7B | llama.cpp | 206.19 (206.19, 205.10, 215.82) | 50.81 (50.81, 50.42, 51.48) |
| Q2_0 1.7B | Aether | **138.24** (136.79, 139.35, 138.24) | **43.27** (43.27, 43.26, 43.38) |
| Q2_0 1.7B | llama.cpp | 47.96 (47.96, 48.27, 47.84) | 37.90 (38.04, 37.90, 37.68) |

| | Aether / llama | Q2_0 / Q1_0 (Aether) |
| --- | --- | --- |
| Q2_0 pp128 | **288.2 %** | 71.9 % |
| Q2_0 tg64 | **114.2 %** | 70.1 % |

Q2_0 reads 484.2 MB per token against Q1_0's 242.4 MB (2.00×). At
43.27 tok/s that is 20.9 GB/s effective, against Q1_0's 61.71 tok/s ×
242.4 MB = 15.0 GB/s. llama.cpp's Q2_0 prompt path is much slower than
its own Q1_0 path (47.96 against 206.19), while its decode falls from
50.81 to 37.90.

A later rerun with the final binary (after the single-row `dot1` fix)
reached the top step for its first round and then fell back to the second,
so its cells are mixed (`b5-phone-speed-final-2026-11-22.txt`); it is not
used in the table, which is the clean all-second-step session. Its
second-step rounds agree within ~1 % on prompt and ~1–4 % on decode.

Raw: `b5-phone-speed-2026-11-22.txt`.
