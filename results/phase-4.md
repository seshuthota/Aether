# Phase 4 — threading and core placement

Code: `src/pool.rs` (persistent pinned workers, epoch + spin/park handoff,
`for_blocks` dynamic scheduling), `kernels::*_on` (row-split matmuls),
parallel attention heads in `model/transformer.rs`. The `-c CORES` flag
(CLI, bench suite, C ABI) runs one thread per listed core.

## Correctness

Splitting by output rows leaves every row's sum to one thread in the same
order, so threaded results are **bit-identical** to single-threaded:
`kernels::tests::threaded_matmuls_are_bit_identical` checks every GEMV/GEMM
golden shape with 2, 3 and 8 threads, and the quantized path is checked
through `quantized_linear_matches_reference` with a 3-thread pool. All model
tests (llama2.c, HF, GGUF) pass with pools.

## Clean sweep (2026-10-22)

`scripts/phase4-sweep.sh results/phase-4/raw`, commit `e39d14c`: SmolLM2-135M
Q4_0, `--pp 64 --tg 64 --reps 3`, each run after the CPU cooled below 48 °C,
screen kept awake, on battery (level 81 → 74 %), twelve core sets in order
and then in reverse. Every run records the clock caps, the foreground app
and any playing media at its start (first lines of each `raw/*.txt`).

Conditions that were **not** clean: the caps were 2.40 / 2.44 GHz
(performance / prime) for the whole SmolLM2 speed sweep, but fell to
2.0–1.8 GHz during the Qwen runs and to 1.8–1.6 GHz during the five-minute
runs (the phone reached 72–78 °C in those). Some runs started with another
app in the foreground (a food-delivery app, Reddit, X; see the table); none
had media playing.

Decode tok/s, pass 1 / pass 2 (second pass in reverse order):

| Cores | tg64 | pp64 | start-of-run app |
| --- | --- | --- | --- |
| 7 | 160.0 / 158.9 | 221.9 / 221.8 | delivery app (p1) |
| 6–7 | 262.8 / 261.3 | 383.9 / 378.8 | delivery app (p1) |
| 5–7 | 276.2 / 279.9 | 397.8 / 398.9 | delivery app (p1) |
| 4–7 | 310.6 / 299.0 | 450.8 / 304.5 | delivery app (p1) |
| 3–7 | 326.2 / 327.1 | 442.3 / 487.0 | |
| 2–7 | 326.4 / 343.4 | 528.6 / 518.8 | |
| 1–7 | 347.8 / 338.0 | 490.7 / 531.2 | |
| 0–7 | 340.6 / 310.5 | 571.9 / 499.9 | |
| 0 | 83.3 / 81.8 | 104.7 / 111.0 | |
| 0–1 | 135.7 / 144.8 | 193.7 / 186.3 | caps 2.23 / 2.65 GHz in p2 |
| 2–5 | 219.3 / 200.2 | 320.2 / 319.9 | X (p2) |
| 0–5 | 247.5 / 184.4 | 379.8 / 325.4 | Reddit (p1), X (p2) |

- Decode rises with every performance core added, up to about 340 tok/s on
  1–7 / 2–7 / 0–7, and stays flat or dips beyond that: 0–7 is **not** slower
  than 2–7 (340.6 / 310.5 against 326.4 / 343.4). The two passes of one set
  differ by up to 25 % (0–5), 10 % (0–7), so differences of a few percent
  between the top sets are inside the noise.
- One prime core (160) is about 1.9× one performance core (83). Their caps
  were 2.44 and 2.40 GHz, so the clock explains little of this; the cause
  is not examined.
- Core 0 is the slowest single core, and 0–1 together reach 136–145.

Qwen2.5-0.5B Q4_0 (`--depth 0,512`; caps had already fallen to
2.0 → 1.8 GHz, so not comparable with the table above):

| Cores | tg64 @ d0 | tg64 @ d512 |
| --- | --- | --- |
| 6–7 | 86.6 / 71.4 | 71.2 / 59.3 |
| 2–7 | 117.3 / 125.1 | 118.1 / 105.6 |
| 0–7 | 128.8 / 123.5 | 104.2 / 94.6 |
| 0–5 | 78.8 / 88.4 | 66.9 / 68.7 |

## Equal shares against blocks, same day

The earlier claim (below) that eight cores were slower than two, and that
blocks fixed it, came from single short runs; the first (`model-speed.txt`) was taken on a
heavily throttled phone (caps of 1.79 GHz for performance and 1.02 GHz for
prime cores), and the later one (`model-speed-dynamic.txt`) recorded no
conditions at all. It was re-measured by building the last equal-share commit
(`67d334f~1`) with the current `bench_suite`, and running both builds
alternately on the same sets, three rounds
(`raw/ab-equal-vs-blocks.txt`; the old tree needed two edits to `bench_suite` to load the model through its older loader; caps 2.40 / 2.44 GHz and nothing in the
foreground except for round 3, where Reddit was). SmolLM2-135M Q4_0 decode,
tok/s, rounds 1 / 2 / 3:

| Cores | equal shares | blocks |
| --- | --- | --- |
| 6–7 | 254.8 / 254.3 / 256.8 | 260.7 / 261.6 / 259.5 |
| 2–7 | 264.4 / 267.8 / 133.4 | 342.3 / 343.9 / 330.6 |
| 0–7 | 269.6 / 241.4 / 239.6 | 345.7 / 237.4 / 317.1 |

- Equal shares do **not** get slower with more cores here: about 255 on 6–7,
  265 on 2–7 and 240–270 on 0–7. They simply gain nothing from the extra
  cores. (The 133.4 is a single outlier.)
- Blocks gain 25–30 % on 2–7 (about 340 against about 265), and 0–7 reaches
  345.7 once but 237.4 in round 2, so it is noisy.
- On two prime cores both are the same within 2.5 %, as expected: two equal
  threads need no balancing.

So the effect of blocks is real; the size originally claimed (98 → 228) was
probably mostly the throttled baseline.

## Interrupts (`raw/interrupts.csv`)

`/proc/interrupts` summed per CPU, read before and after each run.
- 60 s with nothing of ours running: cpu0 49,597, cpu1 12,371, cpu4 8,632,
  cpu5 10,047, cpu2 3,550, cpu3 2,961, cpu6 455, cpu7 29. Interrupts
  concentrate on cpu0, then cpu1, 4, 5.
- During the five-minute runs, cpu0 takes 246,000–298,000 (about 820–990 a
  second), more than twice any other core.
- If one interrupt costs about 10 µs (an assumption, not measured), cpu0
  loses about 1 % of its time. That cannot explain a 17–50 % slowdown.
- No run in this sweep shows cores 0–1 slowing the set down, so the
  hypothesis "interrupts on cores 0–1 make 0–7 slower than 2–7" is **not
  supported**; the earlier loss (228 against 275) is more likely noise or
  throttling, though its conditions were not recorded.

## Handing out a job (`raw/pool-handoff.txt`, `examples/pool_handoff.rs`)

Empty job, caller plus the other cores' workers, µs per hand-out, best of 5:

| Cores | spawn + join | condvar | pool.run |
| --- | --- | --- | --- |
| 6–7 | 70.3 | 48.2 | 0.147 |
| 4–7 | 425.0 | 183.9 | 0.863 |
| 2–7 | 505.8 | 276.2 | 1.262 |
| 0–7 | 659.0 | 440.9 | 1.881 |

The pool's hand-out grows with the number of threads and is 0.15–1.9 µs,
two to three orders of magnitude under the other two. On the PC the same
program gave 50–57, 3–5 and 0.2–0.3 µs. With 241 hand-outs per token,
1.26 µs on 2–7 is about 0.3 ms of the 2.9–3.1 ms per token (about 10 %),
by calculation.

## Energy per token (five minutes of decode each, `raw/*-sustained.txt`)

| Cores | start tok/s → end | monitor avg power | charge-counter power | J/token |
| --- | --- | --- | --- | --- |
| 7 | 132.9 → 132.9 | 1.82 W | 1.70 W | 0.013 |
| 6–7 | 219.4 → 182.6 | 1.94 W | 2.04 W | 0.011 |
| 2–7 | 245.2 → 211.0 | 1.88 W | 2.47 W | 0.011 |
| 0–7 | 190.8 → 206.6 | 1.66 W | 2.26 W | 0.011 |

All four were throttled (caps 1.56–1.96 GHz, up to 78 °C). The charge
counter covers only 168–254 s of each run and is coarse; the monitor's
power and the counter's disagree by up to 30 % (2–7). J/token is nearly the
same for every set with two or more cores; one prime core is a little more
costly per token. Differences between sets are not resolved by this
measurement.

## Not done

- Multi-thread GEMV GB/s (`gemv_bench` is single-threaded). The share of
  Phase 1's all-core bandwidth is calculated from tok/s × weight bytes
  instead: 340 tok/s × 75.8 MB ≈ 25.8 GB/s, 38 % of 67 GB/s.
- A sweep in steady caps for Qwen and the energy runs: the phone throttled
  during both.
- Why 0–7 swings between 237 and 346 in blocks mode.

## History: the first measurements

### The first version

`results/phase-6/raw/model-speed.txt` (commit `3788a28`, on battery caps):
static row split, every thread gets an equal share.

| Model, format | core 7 | cores 6–7 | cores 0–7 |
| --- | --- | --- | --- |
| stories15M f32 | 422 | 667 | 586 |
| SmolLM2-135M f32 | 39.2 | 65.4 | 57.1 |
| SmolLM2-135M Q8_0 | 116.7 | 162.0 | 97.0 |
| SmolLM2-135M Q4_0 | 123.9 | 179.6 | 98.4 |

(decode tok/s). **Eight cores were slower than two.** The profile
(`results/phase-6/raw/simpleperf-smollm2-q4.txt`, SmolLM2 Q4_0 on 8 cores)
shows why: 58 % of samples are workers spinning while they wait, 26 % the
actual kernel. With an equal split, the two 2.4 GHz prime cores finish
their share and then wait for the six slower performance cores at the end
of every matmul — about 200 times per token.

### The fix (commit `67d334f`)

- **Dynamic row blocks**: threads take small blocks of rows from a shared
  counter until none are left, so a faster core simply takes more blocks.
- **Caller on the fastest core**: the calling thread does all the serial
  work between matmuls (norms, residual adds, sampling), so the pool pins
  it to the fastest core of the set.
- **Parallel attention heads**.

SmolLM2-135M Q4_0 decode afterwards (`model-speed-dynamic.txt`, last four
lines, same conditions):

| Cores | before | after |
| --- | --- | --- |
| 6–7 (2 prime) | 179.6 | 214.9 |
| 4–7 | — | 229.5 |
| **2–7** | — | **274.7** |
| 0–7 (all) | 98.4 | 227.9 |

All eight cores went from 98 to 228 tok/s, and 2–7 became the best set.
Why leaving out cores 0–1 helps is **not established**: the commit's
explanation (Android routes most interrupts to cpu0–1) is a hypothesis; it
can be checked by summing `/proc/interrupts` per CPU.

The first six lines of `model-speed-dynamic.txt` come from an intermediate
build during that work and are not used here.

### Core sets on a bigger model

Qwen2.5-0.5B (`results/phase-7.md`): cores 2–7 decode Q4_0 1.6× faster than
the two prime cores alone (128.8 vs 78.8 tok/s), while llama.cpp gets the
same speed from both sets (125.0 vs 123.7): two of its threads already
saturate what decode needs, ours do not. Prefill gains less from more
cores at short prompts (more synchronisations per unit of work).

**Superseded**: these are single short runs, the first with caps down to
1.02 GHz and the second with unrecorded conditions; see the A/B and the clean sweep above.

## Re-run on Qwen2.5-0.5B after the later phases (2026-10-31; `results/plan-checks/raw/`)

`scripts/plan-checks.sh … energy sweep`, current code (`0b67b7f` +
`da29b98`), Qwen2.5-0.5B Q4_0, on battery (53 → 29 %, adb over Wi-Fi),
screen on, every run started below 45 °C.

**Speed per core set** (`sweep-*.txt`, `--pp 64 --tg 64 --reps 3`): tok/s,
pass 1 (in order, caps 2.40 / 2.44 GHz) / pass 2 (reverse order; caps 2.40 /
2.44 for its first three runs, 0–5, 2–5 and 0–1, then 2.75 / 2.65 GHz for
the other nine, so pass 2 is faster there by its clocks). Multi-thread GB/s
= tg64 × 278 MB of weights.

| Cores | tg64 | pp64 | GB/s (decode) | % of 67 GB/s |
| --- | --- | --- | --- | --- |
| 7 | 80.8 / 87.7 | 260 / 280 | 22–24 | 34–36 % |
| 6–7 | 150.4 / 156.8 | 416 / 581 | 42–44 | 62–65 % |
| 5–7 | 164.6 / 181.4 | 602 / 679 | 46–50 | 68–75 % |
| 4–7 | 166.6 / 192.0 | 693 / 768 | 46–53 | 69–80 % |
| 3–7 | 191.2 / 197.9 | 788 / 881 | 53–55 | 79–82 % |
| 2–7 | 193.3 / 198.7 | 905 / 1003 | 54–55 | 80–82 % |
| 1–7 | 181.4 / 196.3 | 990 / 968 | 50–55 | 75–81 % |
| 0–7 | 181.7 / 185.9 | 910 / 1069 | 51–52 | 75–77 % |
| 0 | 39.6 / 44.9 | 134 / 154 | 11–12 | 16–19 % |
| 0–1 | 76.0 / 87.0 | 262 / 296 | 21–24 | 32–36 % |
| 2–5 | 142.1 / 143.4 | 503 / 501 | 40 | 59 % |
| 0–5 | 180.7 / 178.4 | 708 / 598 | 50 | 75 % |

- Decode rises to ~195 tok/s at 3–7 / 2–7, 54–55 GB/s, about 80 % of the
  all-core DRAM read, and stays there or dips with cores 0–1 added. This
  is the multi-thread GEMV bandwidth the plan asks for, measured in the
  model rather than in `gemv_bench` (single-threaded).
- Prefill keeps gaining up to 0–7 (910–1069), as compute work should.
- One prime core decodes 2.0× one performance core (80.8 vs 39.6), as in
  the SmolLM2 sweep (1.9×); both caps were 2.40–2.44 GHz, so not the
  clock. Six performance cores (0–5, 178–181) come within 7–10 % of 2–7
  (193–199): with memory the limit, the prime cores add little.
- The two prime cores alone reach 150–157 tok/s, 2× the 2026-10-21 figure
  (78.8): repacked weights and the `sri` unpacks made the per-core work
  smaller (`results/phase-6.md`).

**Energy per token** (`energy-*.txt`, five minutes of 64-token decode
bursts each; energy from the battery charge counter over the window it
covers; whole phone, screen on):

| Cores | start tok/s | 5-min average | charge-counter power | J/token | max temp |
| --- | --- | --- | --- | --- | --- |
| 7 | 80.7 | 69.7 | 2.17 W (212 s) | 0.031 | 74.5 °C |
| 6–7 | 122.1 | 99.7 | 3.16 W (171 s) | 0.032 | 85.3 °C |
| 2–7 | 178.7 | 133.0 | 3.76 W (286 s) | 0.028 | 88.8 °C |
| 0–7 | 161.6 | 140.5 | 4.09 W (221 s) | 0.029 | 89.2 °C |

- Every set throttles: decode falls 14–26 % over five minutes; 6–8 cores
  reach 85–89 °C.
- J/token is 0.028–0.032 for every set; the spread is about the
  measurement's (the counter covers 171–286 s of each run, and the
  monitor's own power estimate differs from it by up to 24 %). More cores
  draw more power and finish sooner, in nearly equal proportion. If
  anything, 2–7 and 0–7 are slightly cheaper per token than the prime cores
  alone.
- Against the first energy run of Phase 7 (0.041 J/token, 86 tok/s
  average), the same model now costs a third less per token at 1.5× the
  sustained speed.

With this, 4.4 is complete except the optional rayon baseline.
