# Phase 1 — hardware and memory model (iQOO 13, SM8750)

Raw data: `results/phase-1/raw/` (each CSV starts with its conditions). Plots:
`read-sweep.png`, `kernels-core7.png`. Programs: `examples/bw_sweep.rs`,
`examples/bw_sustained.rs`, run by `scripts/phase1-runs.sh`.

## Conditions (all runs)

2026-10-21, **on battery** (adb over Wi-Fi; `USB/AC/Wireless powered: false`,
status discharging, level 90 → 81 %), screen kept on, `walt` governor, battery
saver off. Started below 44–48 °C on the hottest CPU zone.

**The vendor caps the clocks on battery**: under load the prime cores (6–7,
4.32 GHz max) were capped at 2.44 GHz and the performance cores (0–5, 3.53 GHz
max) at 2.40 GHz; with the screen off the caps fall to 1.02 / 1.79 GHz (which is
why runs keep the screen on). Every number here is "on battery caps". Plugged-in
or performance-mode numbers would be different and are not measured yet.

`CTR_EL0` reports a 64-byte data cache line (so `AlignedBuf`'s 64 is right).
This SoC has **no efficiency cores**: "prime vs efficiency" in the plan became
prime vs performance.

## The table

GB/s are decimal, bytes counted by the STREAM rule (each array element read or
written once counts once). Read = sum with 8 independent NEON accumulators.

| Measurement | Value | Source |
| --- | --- | --- |
| L1 read, one prime core (≤ ~90 KB) | **~102 GB/s** | sweep-read-core7 |
| L1 read, one performance core | ~61 GB/s | sweep-read-core0 |
| L2 read, one prime core (128 KB – 2 MB) | ~67 GB/s | sweep-read-core7 |
| L2 read, one performance core | ~44 GB/s | sweep-read-core0 |
| DRAM read, one prime core (≥ 64 MB) | **~61 GB/s** | sweep-read-core7 |
| DRAM read, one performance core | ~43 GB/s | sweep-read-core0 |
| DRAM read, 2 prime cores | ~68 GB/s | sweep-read-prime2-par |
| DRAM read, 6 performance cores | ~63 GB/s | sweep-read-perf6-par |
| DRAM read, all 8 cores | **~67 GB/s** (69.8 at the start of the sustained run) | sweep-read-all8-par |
| Aggregate L1, all 8 cores | ~400 GB/s | sweep-read-all8-par (64 KB per thread) |
| Sustained all-core DRAM read, 5 min | 70.2 GB/s first 30 s → **68.4 GB/s** last 30 s (−2.6 %) | sustained-all8 |
| Sustained one-prime-core read, 5 min | **47.5 GB/s** average (vs 61 fresh) | energy-core7 |
| Sustained one-perf-core read, 5 min | 32.3 GB/s average (vs 43 fresh) | energy-core0 |
| Energy per GB, prime core (5 min) | **0.066 J/GB** total, 0.055 above idle | energy-core7 |
| Energy per GB, performance core (5 min) | 0.082 J/GB total, 0.060 above idle | energy-core0 |
| All-core read power | 4.57 W (charge counter), 0.066 J/GB | sustained-all8 |
| Misaligned (+4 B) read, prime core | DRAM 59 vs 61 GB/s, L1 97 vs 102 (−3…5 %) | sweep-read-core7-off1 |
| copy / scale / add, prime core, DRAM | 42 / 43 / 46 GB/s (STREAM-counted) | sweep-{copy,scale,add}-core7 |

## What the numbers say

- **One prime core nearly saturates DRAM**: 61 of ~67–70 GB/s (~90 %). The plan
  expected one core to fall well short; on this SoC that is true only for a
  performance core (43 GB/s, ~64 %). Decode on one prime core is therefore
  already close to the memory limit *if the kernel keeps up* — the Phase 3
  GEMV reaches 41–49 GB/s of it.
- **Cache cliffs**: the L1 cliff sits just below 96 KB (Oryon's 96 KB L1D);
  the L2 plateau holds to ~2–3 MB per core. Between ~4 and ~8 MB a single core
  reads *slower* than from DRAM (46–57 GB/s vs 61), then recovers — not
  explained yet (candidates: the 12 MB L2 being shared by the cluster, the
  system-level cache, prefetcher behaviour). A good `simpleperf` question.
- **Copy/scale/add look slower than read** because of write-allocate: each
  written line is first read, so real traffic is 1.5× (copy, scale) or 1.33×
  (add) the counted bytes — 42 × 1.5 ≈ 63 GB/s, consistent with the read limit.
- **Alignment** costs 3–5 % — measurable, small; `AlignedBuf` is kept mainly so
  every benchmark starts from the same layout.
- **Thermals barely touch DRAM bandwidth**: over 5 minutes all-core reads lost
  only 2.6 % while the governor cut clocks from 2.4 / 2.44 GHz to 1.36 /
  1.02 GHz at up to 88 °C. A memory-bound load does not need full clocks. A
  *single* core, by contrast, lost ~22 % (61 → 47.5 GB/s average): one core
  does need its clock to issue loads fast enough.
- **Energy**: the prime core moved 47 % more bytes per second at 18 % more
  power, so it spent *less* energy per GB (0.066 vs 0.082 J/GB including the
  phone's fixed draw). Above idle the two are within the measurement's
  resolution (0.055 vs 0.060). For memory-bound work, finishing fast ("race to
  idle") wins on this phone.

## How energy was measured, and its limits

SELinux denies the shell user `/sys/class/power_supply/*`, so the plan's
`current_now` sysfs reads are impossible without root. Android's battery service
(`cmd battery get current_now|counter`, `dumpsys battery`) is readable, but on
this phone it refreshes only every 45–85 s
(`raw/battery-counter-cadence.csv`; under an 8-core spin `current_now` read
−475 mA while the charge counter fell at 477 mA, confirming mA / µAh units).
Energy is therefore measured from the **charge counter** between its first and
last step inside each 5-minute run, times the battery voltage, with a 5-minute
idle baseline before each run. Resolution is ~1 mAh per step (~3.6 % of a
5-minute prime-core run), and the idle baseline itself varied 0.53–0.70 W
between runs. Treat differences under ~10 % as noise.
