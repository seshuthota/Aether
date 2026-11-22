# B2b on the phone: Q1_0 decode speed

aether-infer `one-bit` (B2b), iQOO 13 (I2401), 6 threads on cores 2–7,
against llama.cpp `d02b4c0d6`, same GGUFs. `scripts/phone_ab.py`: one job
per process, alternated order, cooled before every job, screen on, settings
restored. The vendor clock step was pinned by requiring the performance
core's `scaling_max_freq` to stay in one step.

The phone's cap hysteresis made one long session hard: the top step
(2.75/2.65 GHz) and the second step (2.40/2.44 GHz) swap at a board
temperature near 36–37 °C. Two sessions were used for the top step; every
job in the table below started and ended at 2.75/2.65 GHz and
`quiet_therm` was 35.4–37.0 °C at the start.

## Final top step (H1+H2), n noted per row

| model | engine | pp128 | tg64 |
| --- | --- | --- | --- |
| 1.7B | Aether | **227.49** (220.72, 234.26; n=2) | **68.33** (68.00, 68.66; n=2) |
| 1.7B | llama.cpp | 235.58 (232.58, 238.58; n=2) | 56.09 (55.92, 56.25; n=2) |
| 8B | Aether | **43.67** (43.67, 43.64, 43.74; n=3) | **16.53** (16.45, 16.53, 16.76; n=3) |
| 8B | llama.cpp | 45.29 (44.69, 45.29, 45.53; n=3) | 12.84 (13.24, 12.43; n=2) |

| | Aether / llama |
| --- | --- |
| 1.7B pp128 | 96.6 % |
| 1.7B tg64 | **121.8 %** |
| 8B pp128 | 96.4 % |
| 8B tg64 | **128.8 %** |

Condition for the two top sessions: 19 jobs, all caps 2.75/2.65 GHz,
zone0 36–39 °C and board 35.4–37.0 °C at the start, 37.5 °C the hottest
end, battery 78 % → 74 %, on battery. A third session was attempted for a
full 3×3 grid; the cap fell back to 2.40/2.44 GHz on the last 8B llama
job, so that row is n=2 rather than a mixed-cap median.

The top-step table is H1+H2 on cooled starts; the second-step table below
is **H1 only on hot starts**, so 12.98 → 16.53 mixes H2's gain with both
the clock step and the cooling, and is not a measurement of clock
sensitivity (B4 later measured 14.64 at the same 2.40/2.44 step after
cooling the SoC before each job). Likewise the 98.4 % parity below is
H1-only and hot-start; B4's cooled baseline at the same step was 112.4 %.
The controlled H2 result is the paired 4096×4096 kernel A/B, not this
model-level pair.

## Second step, H1 only, 3 full rounds (equal caps to B3)

This is the like-for-like comparison with B3, before H2's activation-sum
reuse was added:

| model | Aether | llama.cpp | Aether / llama |
| --- | --- | --- | --- |
| 1.7B pp128 / tg64 | 189.32 / 53.88 | 205.35 / 51.53 | 92.2 % / 104.6 % |
| 8B pp128 / tg64 | 43.27 / 12.98 | 46.75 / 13.19 | 92.6 % / 98.4 % |

Conditions: 24 jobs, every start and end at 2.40/2.44 GHz, zone0
37–38 °C and board 36.9–37.0 °C at the start, battery 83 % → 81 %.

Against B3 (same 2.40/2.44 step): 8B pp128 41.08 → 43.27 (+5.3 %),
tg64 11.63 → 12.98 (+11.6 %), with llama.cpp unchanged at 46.75 / 13.19.
B2's 1.7B at the top step was 210.05 / 53.15; the final top-step run is
227.49 / 68.33 (+8.3 % prompt, +28.6 % decode).

Raw: `b2b-phone-speed-2026-11-22.txt` (H1, 3 rounds),
`b2b-phone-speed-top-2026-11-22.txt` and
`b2b-phone-speed-top2-2026-11-22.txt` (H1+H2, top),
`b2b-phone-speed-mixedcaps-aborted-2026-11-22.txt` (aborted first
attempt).
