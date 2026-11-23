# B5b phone speed: Ternary-Bonsai-2 2B/4B/9B, and the 1.7B reference

aether-infer `one-bit` at `1d65f78` (`aether-b5b`) against llama.cpp
`d02b4c0d6`, same GGUFs. iQOO 13 (I2401), 6 threads on cores 2–7,
`scripts/phone_ab.py`, 3 alternated rounds, one job per process, cooled
before every job, screen on. On battery 43 % → 35 %.

## Conditions

All 48 jobs started and ended at the **second step, 2.40/2.44 GHz**; the
top step was not reached. Board 36.7–39.4 °C at the start and
36.8–37.5 °C at the end; zone0 36–52 °C at the start (a few warm starts),
37–43 °C at the end. Every job exited 0. All cells are n=3.

## pp128 / tg64, median of 3 rounds

| model | engine | pp128 | tg64 |
| --- | --- | --- | --- |
| 2B | Aether | **104.62** (96.74, 104.62, 105.29) | **35.65** (35.65, 35.76, 35.54) |
| 2B | llama.cpp | 47.88 (47.85, 47.88, 47.88) | 30.45 (30.23, 30.45, 31.80) |
| 4B | Aether | **43.10** (42.32, 43.76, 43.10) | **15.89** (16.07, 15.89, 15.87) |
| 4B | llama.cpp | 18.59 (18.59, 18.66, 18.57) | 14.18 (14.18, 14.31, 14.17) |
| 9B | Aether | **24.97** (25.07, 24.97, 24.78) | 7.98 (7.97, 7.98, 8.01) |
| 9B | llama.cpp | 10.05 (10.03, 10.08, 10.05) | **7.99** (8.19, 7.99, 7.99) |
| 1.7B Q2_0 ref | Aether | **137.16** (137.16, 137.29, 135.13) | **43.27** (43.21, 43.27, 43.43) |
| 1.7B Q2_0 ref | llama.cpp | 48.07 (48.27, 47.84, 48.07) | 37.75 (35.79, 37.83, 37.75) |

| model | Aether / llama pp128 | Aether / llama tg64 | Aether GB/s | llama GB/s |
| --- | ---: | ---: | ---: | ---: |
| 2B | 218.5 % | 117.1 % | 19.10 | 16.31 |
| 4B | 231.8 % | 112.1 % | 19.08 | 17.03 |
| 9B | 248.5 % | 99.9 % | 20.31 | 20.34 |
| 1.7B ref | 285.3 % | 114.6 % | 20.95 | 18.28 |

The hybrid DeltaNet stack does not lose the Q2_0 picture: decode stays
at 112–117 % of llama.cpp for 2B/4B and at parity for 9B; prompt is
2.2–2.5× llama.cpp's Q2_0 prompt path, whose prompt kernels remain the
weak side (the 1.7B reference is 2.85× at prompt and only 1.15× at
decode).

Raw: `b5b-phone-speed-2026-11-23.txt`.
