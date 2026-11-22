# B3 on the phone: Bonsai 4B and 8B Q1_0 vs llama.cpp (speed)

aether-infer `one-bit` `accd54d` (engine source `0e071bb`, no code change
since B2) against llama.cpp `d02b4c0d6` (static Android build,
`armv8.7-a+dotprod+i8mm`), same files: `Bonsai-4B-Q1_0.gguf` (540.09 MiB)
and `Bonsai-8B-Q1_0.gguf` (1.07 GiB). iQOO 13 (I2401), 6 threads, cores
2–7. Run with `scripts/phone_ab.py` from `prompt-speed-2`: 3 rounds, the
variant order reversed every other round, one job per process, cooled
before every job (zone0 < 40 °C, `quiet_therm` < 37 °C, performance-cap
floor 2.40 GHz), screen held on, settings restored afterwards. On battery
(90 % → 88 %, discharging); no AC.

## Conditions

| | value |
| --- | --- |
| jobs | 24, every exit 0 |
| zone0 at each job's start | 37–39 °C |
| quiet_therm at each job's start | 36.8–37.0 °C |
| CPU caps (`cpu2` / `cpu6` `scaling_max_freq`) | **2.40 / 2.44 GHz at every start and end** |
| battery | 90 → 88 %, discharging |
| screen / cores | on / 2–7, 6 threads |

The caps are the vendor's **second** step, not the top step B2 had
(2.75/2.65 GHz): the board stayed at 36.8–37.0 °C during this session and
the vendor holds 2.40/2.44 there. After the session, with the phone idle,
the board fell to 35.8 °C and the caps returned to 2.7456/2.6496 GHz. So
the absolute tok/s below are about 10–14 % under a top-cap run; the
Aether/llama ratios are measured at the same caps and are unaffected.

## End-to-end, tok/s (median of 3 alternated rounds)

| model | engine | pp128 | tg64 |
| --- | --- | --- | --- |
| 4B | Aether | **77.13** (75.26, 77.13, 77.35) | **21.44** (21.44, 21.23, 21.55) |
| 4B | llama.cpp | **82.58** (82.35, 82.58, 85.07) | **23.92** (23.92, 23.81, 24.03) |
| 8B | Aether | **41.08** (35.98, 41.33, 41.08) | **11.63** (11.71, 11.63, 11.52) |
| 8B | llama.cpp | **46.75** (46.77, 46.69, 46.75) | **13.13** (13.04, 13.13, 13.28) |

| | Aether / llama |
| --- | --- |
| 4B pp128 | 93.4 % |
| 4B tg64 | 89.6 % |
| 8B pp128 | 87.9 % |
| 8B tg64 | 88.6 % |

The 8B Aether pp128 first round (35.98) is 15 % below its next two
(41.33, 41.08); the other seven series vary by under 3 %. llama.cpp's
three 8B pp runs are 46.77 / 46.69 / 46.75, so the dip is on Aether's side
and is not a clock change (caps stayed 2.40/2.44); the median is used.

## Comparison with B2 and Gemma 4 E2B Q4_0

B2's 1.7B at the top 2.75/2.65 step was pp128 210.05 / tg64 53.15 against
llama.cpp 235.18 / 55.42 (89.3 % / 95.9 %). At the second step the 4B is
93.4 % / 89.6 % and the 8B 87.9 % / 88.6 % of llama.cpp: the Q1_0 kernels
stay in the same band as the model grows.

Absolute speed scales with parameters, not with the Q1_0 byte count:
4B tg64 21.44 and 8B 11.63 (1.84× for 2.03× the weight bytes). That is the
B2 finding again — at these speeds the model stream is not the limit; the
sign expansion and `sdot` are. For context, Gemma 4 E2B Q4_0 on the same
phone (2026-11-14, top caps, 6 threads) was Aether CPU 156 pp128 / 34.5
tg64 against llama.cpp 91 / 23.9, i.e. Aether was 171 % / 144 % of
llama.cpp. Bonsai Q1_0 is therefore slower than Gemma 4 E2B Q4_0 on the
CPU (4B: 77 / 21.44 against 156 / 34.5, with the caps caveat above) even
though it reads fewer bytes per token. The 1-bit format is a memory-size
win here, not yet a CPU-speed win; llama.cpp's Q1_0 CPU path is also
stronger relative to its Q4_0 path than ours is.

For llama.cpp's published 8B number (OpenCL, S25 Ultra: decode 19.6,
pp512 30.4 tok/s), our CPU decode is 11.63 and pp128 41.08 at 2.40/2.44;
different device, backend and prompt size, so only indicative.

Raw log: `b3-phone-speed-2026-11-22.txt`.
