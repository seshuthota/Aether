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

## Same session against llama-bench, after the later phases (2026-10-31; `results/plan-checks/raw/`)

The tables above are from 2026-10-21, before repacked weights, `smmla`
prefill, K-quant kernels, flash-decoding and the pool changes
(`results/phase-6.md`). `scripts/plan-checks.sh … bench` re-ran both
engines on the same GGUF files and cores: `bench_suite --warmup 1` (median
of 3) and `llama-bench -p 64,256,512 -n 64 -d 0,256,512 -t N -r 3` (mean of
3, `taskset` to the same cores), two rounds with the order reversed in the
second, every run started below 45 °C. On battery (53 → 38 %), screen kept
on. The clock caps at the start of each run were 1.79 / 1.96 GHz
(performance / prime) for 21 of the 24 runs, both engines alike (the phone
was still warm from the energy runs before it); `aether-qwen-q8_0-c2-7-r1`
and two llama.cpp runs started at 2.00 / 2.25. Commit `0b67b7f` + the
imatrix writer (`da29b98`, which does not touch these paths).

tok/s, round 1 / round 2. llama.cpp's `pp N` is from depth 0.

| Qwen2.5-0.5B | pp64 | pp256 | pp512 | tg64 @ d0 | tg64 @ d256 | tg64 @ d512 |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_0 ours, 2–7 | 681 / 693 | 670 / 656 | 588 / 623 | **180 / 182** | 162 / 167 | 159 / 159 |
| Q4_0 llama.cpp, 2–7 | 541 / 556 | 510 / 543 | 422 / 425 | 153 / 152 | 136 / 137 | 123 / 123 |
| Q4_0 ours, 6–7 | 470 / 487 | 444 / 440 | 382 / 386 | 139 / 138 | 130 / 130 | 123 / 124 |
| Q4_0 llama.cpp, 6–7 | 357 / 333 | 371 / 328 | 284 / 248 | 124 / 108 | 111 / 97 | 101 / 88 |
| Q8_0 ours, 2–7 | 818 / 709 | 807 / 721 | 767 / 660 | 111 / 110 | 93 / 105 | 102 / 101 |
| Q8_0 llama.cpp, 2–7 | 613 / 623 | 638 / 654 | 479 / 480 | 99 / 99 | 93 / 93 | 84 / 79 |
| Q8_0 ours, 6–7 | 424 / 493 | 390 / 454 | 337 / 384 | 112 / 106 | 105 / 94 | 101 / 93 |
| Q8_0 llama.cpp, 6–7 | 325 / 326 | 307 / 312 | 245 / 246 | 101 / 102 | 92 / 91 | 84 / 83 |
| Q4_K_M ours, 2–7 | 551 / 557 | 557 / 532 | 516 / 512 | 112 / 112 | 107 / 107 | 104 / 104 |
| Q4_K_M llama.cpp, 2–7 | 167 / 146 | 169 / 150 | 155 / 139 | 91 / 86 | 81 / 81 | 72 / 75 |
| Q4_K_M ours, 6–7 | 333 / 336 | 329 / 326 | 292 / 292 | 78 / 78 | 74 / 75 | 72 / 72 |
| Q4_K_M llama.cpp, 6–7 | 80 / 83 | 84 / 84 | 77 / 77 | 61 / 60 | 56 / 56 | 50 / 52 |

Ours is ahead in every cell of both rounds: prefill by 1.2–1.6× (Q4_0,
Q8_0) and 3.3–4.3× (Q4_K_M: llama.cpp has no repacked K-quant path in this
build), decode by 1.0–1.3× at depth 0 and more at depth 512 (flash-decoding
by K/V head). The closest cells are Q8_0 decode on the two prime cores
(112 / 106 against 101 / 102), where both engines are near the memory
limit (below). The screen was on for these runs; that was the screen-on
check `results/phase-6.md` left open after its screen-off comparison.

Peak RSS of ours in these runs: Q4_0 344 MB, Q8_0 578–586 MB, Q4_K_M 457
MB (with a 512-token prompt and the KV cache; footprints per format are in
`results/phase-6.md`).

## Where the time and energy go

This is the written account the plan asks for, from the measurements in
`results/phase-1.md` … `phase-l5.md`; Qwen2.5-0.5B Q4_0 on cores 2–7 unless
stated.

**Decode is reading weights.** Each token reads every matrix once: 278 MB
for Q4_0 (the tied embedding is read as the classifier). At 180–199 tok/s
that is 50–55 GB/s, 75–82 % of the 67 GB/s all-core DRAM read measured in
Phase 1; Q8_0 at 110 tok/s reads 58 GB/s (86 %), and on just the two prime
cores 112 tok/s × 525 MB = 59 GB/s, so two cores already saturate memory for
Q8_0 and adding six gains nothing. Q4_0 and Q4_K_M need more cores because
unpacking 4-bit weights costs instructions: Q4_0 on two prime cores reaches
38 GB/s, Q4_K_M on six 44 GB/s (65 %, its scales are unpacked per
sub-block). The CPU-side time per token (2–7, Q4_0, from the profiles in
`results/phase-6.md`): the weight products ~60 %, attention ~11 % at depth
512 (it grows with the cache: 159 vs 180 tok/s from depth 0 to 512), pool
waits ~14 %, the rest (norms, RoPE, SwiGLU, activation quantization) a few
percent. So decode speed is bandwidth × the fraction of it the kernels can
use; the remaining headroom is ~20 %.

**Prefill is arithmetic.** Each weight is used for every prompt token, so
the work is `smmla` (int8 matrix multiply, 4 rows × 8 tokens per tile) on
weights repacked at load: ~58 % of prefill cycles in the `mmla4` kernel,
attention 17–19 %, waiting threads 6–8 %, quantizing activations 3 %.
Prefill falls from 681 tok/s at 64 tokens to 588 at 512 as attention over
the growing prompt adds up (quadratic in length).

**Heat sets the long-run speed.** Short runs above are at 1.79–2.75 GHz
caps; held for 5 minutes, decode settles lower: 180 → 133 tok/s average on
cores 2–7 (`results/phase-4.md`, energy), the SoC at 89 °C. Clock caps are
the single largest source of variation in every table here (up to 20 % between
runs an hour apart; `results/phase-6.md` "Measurement").

**Energy per token is nearly flat across core sets**: 0.028–0.032 J/token
(phone total, screen on) whether one prime core runs at 70 tok/s or six
cores at 133. More cores draw more power (2.2 → 3.8 W) but finish
proportionally sooner; the fixed cost (screen, DRAM, SoC) is a large share
at these sizes. Against the 2026-10-21 run (0.041 J/token at 86 tok/s
sustained), the kernel work cut energy per token by a third. The other
engines on the same phone (`results/backends/`, Qwen Q4_0, 5 min on battery):
GPU 0.029 J/token at 78 tok/s, Hexagon DSP 0.025 at 131 — the DSP is the
most efficient, and the CPU the fastest for short bursts.

**Memory** is the weights plus little else: peak RSS is the weight bytes
+ 15–30 MB for Q4_0 / Q8_0 / Q4_K_M (repacked copies replace, not add to,
the mapped file), 1.96 GB for f32 (`results/phase-6.md`).
