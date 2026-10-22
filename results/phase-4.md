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

## The first version and what was wrong with it

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

## The fix (commit `67d334f`)

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

## Core sets on a bigger model

Qwen2.5-0.5B (`results/phase-7.md`): cores 2–7 decode Q4_0 1.6× faster than
the two prime cores alone (128.8 vs 78.8 tok/s), while llama.cpp gets the
same speed from both sets (125.0 vs 123.7): two of its threads already
saturate what decode needs, ours do not. Prefill gains less from more
cores at short prompts (more synchronisations per unit of work).

## Pending

- A clean sweep (tok/s vs thread count and core set, multi-thread GEMV
  GB/s as a share of Phase 1's all-core bandwidth, J/token per core set)
  with nothing else running on the phone.
- Check the interrupt hypothesis for cores 0–1.
