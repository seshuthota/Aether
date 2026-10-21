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
scores were a scalar loop). The re-run with NEON attention is pending.

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

- **Short-context decode is level with llama.cpp on 6 cores** (Q4_0 128.8 vs
  125.0; Q8_0 88.8 vs 96.4). Decode reads every weight once per token, so it
  is bandwidth work: 278 MB × 128.8 tok/s ≈ 36 GB/s, about half the ~67 GB/s
  all-core DRAM read of Phase 1.
- **On 2 prime cores llama.cpp decodes ~1.6× faster** (123.7 vs 78.8). Two
  of its threads already saturate what decode needs; ours do not. Our Q4_0
  kernel spends instructions unpacking nibbles before every `sdot`
  (compute-bound on one core, `results/phase-6.md`); this llama.cpp build
  (`GGML_CPU_REPACK=ON`, `-march=armv8.7a+dotprod+i8mm`) repacks Q4_0
  weights at load into interleaved blocks for `smmla` and needs fewer.
- **Decode falls off with context in ours** (Q4_0, cores 2–7: 128.8 → 62.8
  tok/s from depth 0 to 512; llama.cpp 125 → 123). At depth 512 the step
  takes 8.2 ms longer (15.9 vs 7.8 ms), which is what the scalar attention loop costs:
  per layer, 14 heads × 513 positions × 64-long dot products, each a chain
  of dependent adds the compiler may not reorder. That is item 3.5's NEON
  attention (done since; re-measure pending).
- **Prefill is 2.1–5.8× slower than llama.cpp.** Prefill is compute work (each
  weight is reused across all prompt tokens), and our quantized prefill
  still runs `sdot` row kernels per token group rather than a blocked
  integer GEMM; llama.cpp uses `smmla` (i8mm, 8×8 int8 → int32 per
  instruction) on repacked weights. That is item 6.5. For 64–256-token
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

## Pending

- Re-run the speed tables with NEON attention (expect depth 512 to close most
  of the gap to depth 0).
