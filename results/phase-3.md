# Phase 3 — NEON f32 kernels

Raw data: `results/phase-3/raw/gemv-bench-core7-v2.csv` (`examples/gemv_bench.rs`,
one thread pinned to prime core 7, median of repeated calls, on battery
caps as in `results/phase-1.md`). `gemv-bench-core7.csv` is the same bench
before Phase 6's 4-row quantized kernels; the f32 rows agree within noise.

## Correctness

Every kernel is checked against numpy goldens (`tools/gen_kernels.py`: sizes
1, around multiples of 4/8/16/32, and the real layer shapes), on the host
(scalar) and on the phone (NEON): GEMV, GEMM, rmsnorm, softmax, swiglu, both
RoPE layouts. One attention head (added in Phase 7) passes on the host; its
NEON version has not run on the phone yet. stories15M's greedy
tokens stay identical to llama2.c with NEON kernels in place; per-layer
activations stay within tolerance (summing in a different order changes the
last bits, so bit-identity with llama2.c's scalar C is not expected).

## GEMV in three steps (f32, GB/s of weights read)

| Matrix (rows × cols, size) | scalar | 1 accumulator | 4 accumulators per row | 4 rows × 2 acc (`gemv`) |
| --- | --- | --- | --- | --- |
| stories15M wq 288×288 (324 KB, L2) | 5.3 | 17.8 | 41.6 | **53.5** |
| stories15M w1 768×288 (864 KB, L2) | 5.3 | 17.8 | 41.4 | **52.9** |
| stories15M classifier 32000×288 (36.9 MB) | 5.3 | 17.6 | **41.4** | 29.0 |
| Qwen2.5 up 4864×896 (17.4 MB) | 3.5 | 13.3 | **40.2** | 32.9 |
| Qwen2.5 down 896×4864 (17.4 MB) | 3.1 | 9.5 | 36.2 | **48.6** |
| synthetic 8192×8192 (256 MB, DRAM) | 3.0 | 9.3 | 36.0 | **40.6** |

What each step shows:

- **Scalar → 1 accumulator (×3.1–3.8)**: four lanes per instruction, but every
  FMA waits for the previous one (one dependency chain, ~4 cycles each).
- **→ 4 accumulators (×2.3–3.9)**: four independent chains keep the FMA unit
  busy. On L2-sized matrices this is now limited by loads, not latency.
- **→ 4 rows at once (`gemv`)**: each loaded piece of `x` feeds four rows, so
  loads per FMA fall from 2 to 1.25. It wins in L2 (53 GB/s) and on long
  rows (down, 8192²), but **loses on short rows of large matrices**
  (classifier, up: 288 or 896 columns). Four rows stream from four places
  in memory at once, and with rows that short each stream is only 1–3.5 KB
  long before jumping. The likely cause (not yet measured) is that the
  hardware prefetcher follows that pattern less well than one sequential
  stream. Not fixed: a choice of kernel by row length would recover it.

The DRAM case reaches 40.6 GB/s on one prime core, 67 % of that core's
measured DRAM read bandwidth (~61 GB/s, Phase 1) — the rest is the FMA and
reduction work interleaved with the loads.

## End to end

stories15M f32 decode on prime core 7 (`results/phase-6/raw/model-speed.txt`,
commit `3788a28`): **422 tok/s**, against the Phase 2 scalar baseline of
88.6 tok/s on the same core (`results/phase-2.md`) — ×4.8.

## GEMM: FLOPs per cycle (2026-10-28)

`examples/gemm_bench.rs`, one kernel and one shape per run on prime core 7,
under `simpleperf stat -e cpu-cycles:u` (`raw/gemm-cycles.txt`). Dividing
the floating-point operations by the cycles gives a number that does not
depend on the clock, which the phone changed during the plain runs
(`raw/gemm-core7.txt`, `raw/gemm-freq-core7.txt`: core 7 spent about half
the samples at 1.02 GHz). The cycles also count setup (filling the inputs),
so these are lower bounds.

| Kernel | 32×288×288 | 64×768×288 | 128×896×896 |
| --- | --- | --- | --- |
| scalar | 1.18 | 1.18 | 0.78 |
| NEON GEMV once per token | 11.5 | 11.5 | 11.1 |
| NEON GEMM, 4×4 tiles | **27.9** | **29.1** | **27.8** |

(tokens × columns × rows; FLOP/cycle, one multiply-add = 2 FLOPs.)

- **Against the peak.** *Assumption, not measured:* if each Oryon core has
  four 128-bit FMA pipes, its peak is 4 pipes × 4 lanes × 2 = 32 f32
  FLOP/cycle. The GEMM then reaches 87–91 % of it. At the 2.44 GHz battery
  cap that is 68–71 GFLOPS; the best plain run on core 7 measured 74 GFLOPS
  (`raw/gemm-core7.txt`), consistent with a slightly higher clock at that
  moment.
- **GEMV once per token reaches 11 FLOP/cycle**, a third of the peak: each
  weight is loaded for one multiply-add, so it is limited by loads. The
  tile reuses each loaded weight for 4 tokens, and that factor shows in the
  numbers (×2.5).

## Where the time goes in the NEON build (2026-10-28)

stories15M f32 decode, 256 steps, prime core 7, the same run as the Phase 2
profile (`raw/profile-neon.txt`, `raw/profile-neon-stat.txt`). It still
runs at 422.4 tok/s, the same speed as when this phase was first measured.

| Symbol | Share of cycles |
| --- | --- |
| `kernels::neon::gemv` | 89.2 % |
| `kernels::neon::attention_head` | 4.8 % |
| expf, sampler, allocator, sincosf, powf, … | < 1 % each |

Counters for the whole run, against Phase 2's scalar build:

| | scalar (Phase 2) | NEON |
| --- | --- | --- |
| cycles | 6.10 G | 1.29 G (×4.7 fewer) |
| instructions | 9.98 G | 2.60 G (×3.8 fewer) |
| instructions per cycle | 1.64 | 2.02 |
| L1D accesses | 2.34 G | 1.86 G |
| L1D refills | 5.4 M | 81.8 M |

- The matrix-vector product is still nearly all of the time, but now it
  streams 60.8 MB × 422 = **25.7 GB/s** of weights instead of 5.4 GB/s.
  That is 42 % of the ~61 GB/s one prime core measured in Phase 1, so
  this kernel is limited by memory, as Phase 1 predicted for decode.
- Attention grew from 1.1 % to 4.8 % only because everything else got
  faster.
- The L1D refills went up 15×. Likely cause (not measured further): at
  5.4 GB/s the hardware prefetcher had every line in place before the
  load; at 25.7 GB/s more loads arrive before their line does.
