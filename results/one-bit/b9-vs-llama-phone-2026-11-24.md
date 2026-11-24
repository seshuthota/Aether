# Aether (B9) vs llama.cpp, same session (phone, 2026-11-24)

Aether `aether-b9` (one-bit bfa0bf2) against `llama-bench-d02b4c0d6`, same
GGUFs. `scripts/phone_ab.py`, 3 alternated rounds, one job per process,
zone0 < 42 C, board < 39.5 C, cap floor 2.40 GHz, battery, screen timeout
raised for the session. Aether: `bench -dev cpu --cores 2-7 -t 6`; llama:
`taskset fc llama-bench -t 6`. Every one of the 36 runs started and ended at
2.40/2.44 GHz. Raw: `b9-vs-llama-phone-2026-11-24.txt`.

tok/s, three rounds, median:

| model | test | Aether | llama.cpp | Aether / llama |
|---|---|---|---|---|
| Bonsai 1.7B Q1_0 | pp128 | 190.47, 200.23, 200.37 → 200.23 | 204.30, 202.56, 201.80 → 202.56 | 0.99 |
| | tg64 | 81.96, 82.10, 82.01 → 82.01 | 50.45, 50.36, 50.29 → 50.36 | 1.63 |
| Bonsai 8B Q1_0 | pp128 | 42.22, 42.76, 42.82 → 42.76 | 44.98, 44.86, 44.58 → 44.86 | 0.95 |
| | tg64 | 21.17, 21.47, 21.50 → 21.47 | 12.61, 13.07, 12.92 → 12.92 | 1.66 |
| Ternary-Bonsai-2 4B Q2_0 | pp128 | 47.83, 47.75, 47.53 → 47.75 | 18.59, 18.63, 18.66 → 18.63 | 2.56 (llama's Q2_0 prompt path is unoptimised) |
| | tg64 | 19.44, 19.73, 19.39 → 19.44 | 14.20, 14.57, 14.49 → 14.49 | 1.34 |
