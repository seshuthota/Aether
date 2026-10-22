# Phase 7 — Qwen2.5-0.5B-Instruct, GGUF, benchmark suite

Raw data: `results/phase-7/raw/`. Programs: `examples/bench_suite.rs` (ours)
and llama.cpp's `llama-bench` (`d00685831`, built with the same NDK), both run
by `scripts/phase7-bench.sh` on the **same GGUF files** made by llama.cpp's
`convert_hf_to_gguf.py` + `llama-quantize`.

## Conditions

2026-10-21, iQOO 13 **on battery** (status discharging, 75 → 73 %), screen
kept on, every run started below 48 °C on the hottest CPU zone. Battery
clock caps apply (`results/phase-1.md`: ~2.4 GHz instead of 3.53 / 4.32 GHz).
llama-bench was pinned with `taskset` to the same cores and given one thread
per core (`-t 6` / `-t 2`), 3 repetitions; ours reports the median of 3.

Engine code: commit `0f56ed4`, i.e. **before NEON attention** (attention
scores were a scalar loop). NEON attention was measured the next day as an
A/B against this commit; see "Attention: A/B" below.

## Correctness (from the Phase 7 commits)

- Qwen2.5 from safetensors and from the f32 GGUF matches HF `transformers`
  at every layer (≤ 1.4e-4 relative) and for 48 greedy tokens.
- Tokenizer: 433 strings identical to HF `tokenizers`; NFC on 17,253 cases
  identical to Python's `unicodedata`; the ChatML prompt tokenizes exactly
  like `apply_chat_template`.
- Perplexity/KL against llama.cpp (wikitext-2, 20 × 512, base = f32 GGUF):

| Format | PPL ours | PPL llama.cpp | Mean KLD ours | Mean KLD llama.cpp | Same top ours / llama.cpp |
| --- | --- | --- | --- | --- | --- |
| f32 | 16.2745 | 16.2744 ± 0.667 | — | — | — |
| Q8_0 | 16.3404 ± 0.670 | 16.3425 ± 0.671 | 0.002366 | 0.002368 ± 0.000032 | 96.51 % / 96.88 ± 0.24 % |
| Q4_0 | 19.7441 ± 0.837 | 19.7283 ± 0.836 | 0.180924 | 0.181010 ± 0.002752 | 77.18 % / 76.98 ± 0.59 % |

Ours: `aether-host-qwen-{q8_0,q4_0}-ppl.txt`, run on the host (x86, so the
scalar kernels; accuracy does not depend on the machine) with `--kl --base
hostdata/gguf/qwen2.5-0.5b-f32.gguf`, the same base file llama.cpp used.
Every number agrees within llama.cpp's error bar except Q8_0's same-top rate
(1.5 of its standard errors apart, from tokens where two candidates are
nearly tied and summation order decides).

Qwen2.5-0.5B loses far less to Q4_0 than SmolLM2-135M did (PPL ×1.22 and
KLD 0.18 here, against ×1.41 and 0.34 in `results/phase-6.md`).

## Speed: ours vs llama.cpp

tok/s. `pp N` = prefill of an N-token prompt from an empty cache; `tg64 @ dD`
= 64 decode steps after D tokens are already in the KV cache.

### Q4_0 (278 MB of weights)

| Test | ours, cores 2–7 | llama.cpp, 2–7 | ours, cores 6–7 | llama.cpp, 6–7 |
| --- | --- | --- | --- | --- |
| pp64 | 92.0 | 536 ± 12 | 121.6 | 377 ± 3 |
| pp256 | 111.1 | 631 ± 4 | 112.1 | 370 ± 1 |
| pp512 | 165.0 | 432 ± 51 | 99.9 | 281 ± 4 |
| tg64 @ d0 | **128.8** | 125.0 ± 15.9 | 78.8 | 123.7 ± 0.1 |
| tg64 @ d256 | 113.2 | 123.2 ± 13.1 | 67.1 | 110.5 ± 0.4 |
| tg64 @ d512 | 62.8 | 122.6 ± 8.1 | 58.9 | 88.5 ± 0.1 |
| time to first token (64-token prompt) | 569 ms | — | 526 ms | — |
| peak RSS (`VmHWM`) | 348 MB | — | 349 MB | — |

### Q8_0 (525 MB of weights)

| Test | ours, cores 2–7 | llama.cpp, 2–7 | ours, cores 6–7 | llama.cpp, 6–7 |
| --- | --- | --- | --- | --- |
| pp64 | 194.9 | 555 ± 3 | 134.9 | 286 ± 71 |
| pp256 | 183.7 | 568 ± 5 | 125.1 | 304 ± 3 |
| pp512 | 163.1 | 426 ± 2 | 108.5 | 246 ± 1 |
| tg64 @ d0 | 88.8 | 96.4 ± 1.2 | 56.6 | 101.3 ± 0.2 |
| tg64 @ d256 | 77.3 | 90.0 ± 0.6 | 51.5 | 90.2 ± 0.5 |
| tg64 @ d512 | 72.7 | 81.9 ± 2.5 | 46.0 | 83.4 ± 0.2 |
| time to first token (64-token prompt) | 318 ms | — | 469 ms | — |
| peak RSS (`VmHWM`) | 589 MB | — | 590 MB | — |

## Where the time goes

- **Short-context decode is close to llama.cpp on 6 cores**: level on
  2026-10-21 (Q4_0 128.8 vs 125.0; Q8_0 88.8 vs 96.4), ~20 % behind in the
  same-day re-run of 2026-10-22 (111.5 vs 134.9 ± 6.6). Run-to-run and
  day-to-day spread is as large as the gap, so "within ~20 %" is what these
  numbers support. Decode reads every weight once per token, so it
  is bandwidth work: 278 MB × 128.8 tok/s ≈ 36 GB/s, about half the ~67 GB/s
  all-core DRAM read of Phase 1.
- **On 2 prime cores llama.cpp decodes ~1.6× faster** (123.7 vs 78.8). Two
  of its threads already saturate what decode needs; ours do not. Our Q4_0
  kernel spends instructions unpacking nibbles before every `sdot`
  (compute-bound on one core, `results/phase-6.md`); this llama.cpp build
  (`GGML_CPU_REPACK=ON`, `-march=armv8.7a+dotprod+i8mm`) repacks Q4_0
  weights at load into interleaved blocks for `smmla` and needs fewer.
- **Decode falls off with context faster in ours.** The table's 128.8 →
  62.8 tok/s (depth 0 → 512) did not reproduce: on 2026-10-22 the same
  commit gave 110.6 → 87.0 in three alternating runs, so the single 62.8 was
  mostly a bad run (likely clock or memory caps changing mid-test, which the
  shell cannot read). What the A/B below does show: at depth 512 scalar
  attention cost ~2.5 ms per token (11.5 vs 9.0 ms at depth 0), NEON
  attention ~1.5 ms (10.4 vs 9.0). llama.cpp loses less (134.9 → 117.5 the
  same day). One visible cause in ours: work is split by head, and 14 heads
  over 6 threads gives some threads 3 heads and others 2.
- **Prefill is 2.1–5.8× slower than llama.cpp.** Prefill is compute work (each
  weight is reused across all prompt tokens), and our quantized prefill
  still runs `sdot` row kernels per token group rather than a blocked
  integer GEMM; llama.cpp uses `smmla` (i8mm: one instruction multiplies a
  2×8 int8 block by an 8×2 block into 2×2 int32 sums) on repacked weights. That is item 6.5. For 64–256-token
  prompts Q8_0 prefills faster than Q4_0 in ours (no nibble unpacking) —
  the opposite of decode.
- **Cores 2–7 vs 6–7**: performance cores add bandwidth demand but each is
  slower; for decode 6 threads beat 2 by 1.6× in ours. Prefill at 512 tokens
  scales better than at 64 (more work per synchronisation).

## Sustained decode and energy (Q4_0, cores 2–7, 300 s)

`aether-qwen-q4_0-c2-7-sustained.txt`: 64-token decode bursts from position 0,
back to back for 5 minutes.

- 25,792 tokens in 301 s = **85.7 tok/s average**; 10-second windows: 112 tok/s
  for the first 40 s, ~97 by 100 s, then 60–89 with dips as the SoC
  throttles. **Max CPU temperature 84.5 °C.**
- Battery charge counter: 3.49 W average over 282 s (sampler estimate 3.37 W)
  → **0.041 J per token** (about 24 tokens per joule). That is total phone
  power with the screen on, not the CPU alone.

## Attention: A/B (2026-10-22)

`ab-attention.txt`: scalar attention (`0f56ed4`) and NEON attention
(`8ae105c`) run alternately on the same phone, three rounds, each run started
below 48 °C (battery 54 %, discharging; under load the prime cores were
capped at 1.96 GHz instead of the previous day's 2.44 GHz). Q4_0, cores 2–7,
median of three rounds, tok/s:

| | tg64 @ d0 | tg64 @ d256 | tg64 @ d512 |
| --- | --- | --- | --- |
| scalar attention | 110.6 | 96.8 | 87.0 |
| NEON attention | 111.5 | 103.8 | 95.7 |
| llama.cpp, same session (`llamacpp-qwen-q4_0-c2-7-tg-rerun.txt`) | 134.9 ± 6.6 | 114.0 ± 12.2 | 117.5 ± 8.5 |

NEON attention is +10 % at depth 512 and changes nothing at depth 0, as
expected. The `*-neonattn.txt` files are the first re-run of the full tables
that morning, before the A/B: their cores 6–7 decode came out at half the
previous day's (38.8 vs 78.8 tok/s for Q4_0), and so did its 256- and
512-token prefill (53.6 vs 112.1, 53.2 vs 99.9), while 64-token prefill
dropped only from 121.6 to 107.7. The prime-core cap under load was 1.96 GHz
(a 1.25× drop), which explains only part of that; the memory-bus frequency,
the other likely factor, is not readable from the shell
(`/sys/class/devfreq` is denied). The cause is not established. Those files
are kept as a record of how much conditions move, not as a comparison.

## Pending

- Split attention across positions as well as heads (14 heads over 6
  threads), then re-measure depth 512.
