# B4 CPU baseline on the phone (2026-11-22)

Bonsai Q1_0 decode and prompt speed, Aether `one-bit` at `31c4a20` against
llama.cpp `d02b4c0d6`, same GGUFs, 6 threads on cores 2–7, screen on,
`scripts/phone_ab.py`: one process per job, alternated order, cooled before
every job, start/end temperature, caps and battery logged.

**Clock step.** Every job ran at the vendor's second step, **2.40/2.44
GHz**, and every job started and ended on it. The top step (2.75/2.65) was
never reachable in this session: with the phone idle and the screen off,
`quiet_therm` (the board sensor the caps follow) sat at 37.9–38.5 °C, over
the cap's threshold. The room was warmer than during B2b (where the same
sensor read 35.4–37.0). The numbers below are therefore **not** the top-step
ones; the GPU comparison in `b4-phone-gpu-2026-11-22.md` is taken at the
same step.

Conditions: 36 jobs, every start and end at 2.40/2.44 GHz; zone0 39–42 °C
and board 38.5–39.2 °C at the start, zone0 40–50 °C and board 38.6–39.6 °C
at the end; battery 63 → 60 %, on battery.

## Results (median of 3 alternated rounds)

| model | engine | pp128 | tg64 |
| --- | --- | --- | --- |
| 1.7B | Aether | 186.16 (186.16, 185.63, 193.44) | 61.63 (62.57, 61.63, 61.48) |
| 1.7B | llama.cpp | 206.76 (192.77, 209.13, 206.76) | 51.01 (51.01, 50.20, 51.09) |
| 4B | Aether | 82.30 (82.07, 84.09, 82.30) | 27.82 (27.73, 27.91, 27.82) |
| 4B | llama.cpp | 85.33 (79.92, 85.49, 85.33) | 23.71 (23.71, 24.27, 22.88) |
| 8B | Aether | 41.59 (41.59, 37.18, 42.61) | 14.64 (14.64, 15.15, 14.03) |
| 8B | llama.cpp | 45.03 (45.03, 43.22, 45.91) | 13.03 (13.12, 13.03, 12.98) |

| | Aether / llama |
| --- | --- |
| 1.7B pp128 / tg64 | 90.0 % / **120.8 %** |
| 4B pp128 / tg64 | 96.4 % / **117.3 %** |
| 8B pp128 / tg64 | 92.4 % / **112.4 %** |

The same binary read 12.98 (8B tg64) and 53.88 (1.7B tg64) at the 2.40 step
in B2b, on starts whose SoC was hotter; cooling the SoC before each job
(zone0 < 43 °C here, rather than merely the board) raises the measured
rate by 13–14 % at the same cap. The 4B was not measured at the top step
before; B3's 21.44 was on a hot start.

Raw: `b4-cpu-baseline-2nd-2026-11-22.txt` (the first attempt,
`b4-cpu-baseline-aborted-2026-11-22.txt`, was stopped after one job when
the wireless adb transport dropped; its single 1.7B value, 199.86, is
consistent).
