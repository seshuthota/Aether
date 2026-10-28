# Phase 6 — quantization (Q8_0, Q4_0, Q5_0, Q4_K, Q6_K)

Raw data: `results/phase-6/raw/`. Model: SmolLM2-135M (Hugging Face
safetensors, quantized in memory at load with `-q`, every matrix including the
embedding, like `llama-quantize --pure`).

## Accuracy against llama.cpp

Same text, same procedure as `llama-perplexity`: wikitext-2 test, first 20
chunks of 512 tokens, only the second half of each chunk scored (5,100 tokens).
KL divergence is against each engine's own f32 logits.
llama.cpp: `llama-perplexity` (`d00685831`) on the host CPU, on
`testdata/gguf/smollm2-135m-{q8_0,q4_0}.gguf` (all 211 matrices in the named
format, as `--pure` gives; norms f32).

| Format | Size (ours) | PPL ours | PPL llama.cpp | Mean KLD ours | Mean KLD llama.cpp | Same top token ours / llama.cpp |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | 538.1 MB | 19.6715 ± 0.812 | 19.6712 ± 0.812 | — | — | — |
| Q8_0 | 143.0 MB | 19.6707 ± 0.812 | 19.7008 ± 0.814 | 0.003904 | 0.003918 ± 0.000059 | 96.24 % / 96.16 ± 0.27 % |
| Q4_0 | 75.8 MB | 27.7150 ± 1.175 | 27.7412 ± 1.177 | 0.335544 | 0.335355 ± 0.004256 | 64.71 % / 64.77 ± 0.67 % |

Every pair agrees within llama.cpp's own error bar. The engines do not produce
identical bits, and should not: llama.cpp's f32 path on x86 sums in a
different order (AVX2 vs our scalar/NEON), and its `PPL(base)` in the KLD runs
(19.658, computed in batches of 4 sequences) differs from its own plain f32
run (19.671) by about as much as ours differs from it. Our Q8_0 PPL being a
hair *lower* than f32 is noise of the same size.

What the numbers say about the formats:

- **Q8_0** (8.5 bits per weight: 32 int8 + one f16 scale per block) is
  practically lossless: the top token agrees with f32 96 % of the time, and
  where it doesn't, the two candidates were nearly tied (KLD 0.004 nats).
- **Q4_0** (4.5 bits per weight) costs SmolLM2-135M 41 % perplexity and a
  third of its top-token choices. Small models suffer most from 4-bit weights
  (fewer parameters to absorb the rounding); Qwen2.5-0.5B loses much less
  (`results/phase-7.md`).

Files: `aether-host-{f32,q8_0,q4_0}.txt` (ours, `aether MODEL --ppl wiki.test.raw
--chunks 20 -q FMT --kl`), `llamacpp-{f32,q8_0,q4_0}.txt`.

## Kernels

Checked against numpy goldens of the same block math (`tools/gen_quant.py`,
which quantizes with the `gguf` package so the blocks are byte-identical to
llama.cpp's): the Q8_0/Q4_0 quantizers bit-exact, the dot products and
threaded `linear_on` within float tolerance (`kernels::tests`,
`quant::tests`).

The decode GEMV goes from 12 GB/s (dequantize to f32, then FMA) to 32 GB/s
for Q8_0 by quantizing the activation to Q8_0 too and using `sdot` (4 int8
products summed into an int32 lane per instruction). Q4_0 needs a nibble
unpack before each `sdot` and stays compute-bound on one core at 12–15 GB/s.

## Profile (SmolLM2-135M Q4_0 decode)

`simpleperf-smollm2-q4.txt`, taken before the Phase 4 scheduling fixes: on
8 cores 58 % of samples were workers spinning while waiting for the next job
(`ThreadPool::new::{closure}`), 26 % in `dot4_q4_0_q8_0`, 4 % in scalar
attention. On the 2 prime cores the kernel rises to 49 % and attention to
11 %. That spin share is what Phase 4's dynamic row blocks and pinning the
caller to the fastest core went after.

## Prefill with `smmla` (item 6.5, first step; `raw/prefill-ab/`)

`smmla` (i8mm) multiplies a 2×8 int8 block by another 2×8 block and adds the
2×2 int32 result: two weight rows × two tokens per instruction, so each
weight value loaded serves two tokens where `sdot` serves one. First version
(`mmla2_q8_0`, `mmla2_q4_0` in `src/kernels/neon_quant.rs`): 2 rows × 2
tokens, rows and tokens interleaved 8 bytes at a time inside the loop
(`zip1`/`zip2`), used by `quant_rows` whenever more than one token is
multiplied (prefill). Decode (one token) keeps the `sdot` kernels.

**Correctness.** Batched goldens (`testdata/quant_batch/`, 9 shapes with
odd row and token counts, t = 2…64; `dequant(X_q8_0) @ dequant(W)ᵀ` from
llama.cpp's quantizers): pass on the phone, threaded output bit-identical to
one thread (`quantized_linear_batch_matches_reference`). Perplexity on the
phone, wikitext-2, 20 × 512 (`ppl-q4_0.txt`, `ppl-q8_0.txt`):

| Format | ours, `smmla` (phone) | ours before (host, scalar) | llama.cpp |
| --- | --- | --- | --- |
| Q4_0 | 19.7499 | 19.7441 | 19.7283 ± 0.836 |
| Q8_0 | 16.3471 | 16.3404 | 16.3425 ± 0.671 |

**Speed.** Qwen2.5-0.5B, cores 2–7, 2026-10-24, on battery (51 %), screen on,
every run started below 48 °C; per round the new build, the previous commit
(`a077c63`, `sdot`) and llama-bench ran back to back (`scripts/prefill-ab.sh`).
Prefill tok/s, round 1 / round 2:

| | pp64 | pp256 | pp512 |
| --- | --- | --- | --- |
| Q4_0 `smmla` | 272 / 242 | 264 / 231 | 230 / 209 |
| Q4_0 `sdot` (before) | 208 / 176 | 196 / 169 | 188 / 159 |
| Q4_0 llama.cpp | 709 / 631 | 734 / 614 | 552 / 473 |
| Q8_0 `smmla` | 218 / 223 | 248 / 218 | 221 / 194 |
| Q8_0 `sdot` (before) | 222 / 224 | 252 / 224 | 225 / 204 |
| Q8_0 llama.cpp | 711 / 622 | 765 / 649 | 535 / 478 |

- Q4_0: +22–37 %. The nibbles are now unpacked once per pair of tokens
  instead of once per token.
- Q8_0: unchanged to −5 %. Per 32-value block the 2×2 kernel spends 8
  interleaving instructions and 4 `smmla` for 4 outputs; the 4-row `sdot`
  kernel spends 8 `sdot` and no interleaving for 4 outputs. The interleaving
  eats what `smmla` saves.
- Every build ran faster than in `results/phase-7.md` (old Q4_0 pp256 196
  against 111 then) and round 2 ran ~10 % slower than round 1, so only
  same-round comparisons hold.
- Gap to llama.cpp (its tok/s ÷ ours, same round): Q4_0 2.3–2.8× (the `sdot`
  build 2.9–3.7×), Q8_0 2.4–3.3×.

## Prefill with `smmla`, 4 rows × 4 tokens (item 6.5, second step; `raw/prefill-ab-4x4/`)

Two changes to the first version, which reloaded and interleaved both rows
and both tokens for every 2×2 output tile:

- **Activations interleaved once per matmul** (`interleave_pairs`): the
  Q8_0 activations of tokens 2p and 2p + 1 are stored as one `PairQ8` per
  block, already in the `[x0 0..8 | x1 0..8 | …]` order `smmla` takes, with
  their scales as `[d0, d1, d0, d1]`. The kernel only loads them.
- **4 rows × 4 tokens per tile** (`mmla4_q8_0`, `mmla4_q4_0`): per block the
  four rows are loaded (and for Q4_0 unpacked) and interleaved once, then
  used against two token pairs: 16 `smmla`s for 16 outputs, against 4
  `smmla`s and 8 interleaving instructions for 4 outputs before.

Edges (a row group or token-pair group past the end) repeat the last row or
pair; each output still has its own lane, integer sums and float additions,
so threads cannot change the result. The 2×2 kernel is gone; decode (one
token) still uses the `sdot` kernels.

**Correctness.** All 86 library tests pass on the phone, including the
batched goldens and the threaded = one-thread check. Perplexity on the
phone, wikitext-2, 20 × 512 (`ppl-q4_0.txt`, `ppl-q8_0.txt`):

| Format | ours, 4×4 | ours, 2×2 | llama.cpp |
| --- | --- | --- | --- |
| Q4_0 | 19.7486 | 19.7499 | 19.7283 ± 0.837 |
| Q8_0 | 16.3532 | 16.3471 | 16.3425 ± 0.671 |

The new order of float additions moves PPL by less than 0.01, far inside
the ± range.

**Speed.** Qwen2.5-0.5B, cores 2–7, 2026-10-28, on battery (31 % → 29 %,
discharging), screen kept on, every run started below 48 °C; per round the
new build, the previous commit (`37546a0`, 2×2) and llama-bench ran back to
back (`scripts/prefill-ab.sh`). Prefill tok/s, round 1 / round 2:

| | pp64 | pp256 | pp512 |
| --- | --- | --- | --- |
| Q4_0 4×4 | 370 / 257 | 334 / 250 | 273 / 224 |
| Q4_0 2×2 (before) | 241 / 187 | 230 / 178 | 213 / 166 |
| Q4_0 llama.cpp | 733 / 541 | 733 / 525 | 556 / 414 |
| Q8_0 4×4 | 401 / 288 | 326 / 280 | 287 / 249 |
| Q8_0 2×2 (before) | 268 / 200 | 251 / 189 | 222 / 173 |
| Q8_0 llama.cpp | 712 / 555 | 762 / 571 | 544 / 425 |

- Same-round gain over 2×2: Q4_0 +28–53 %, Q8_0 +29–50 %. Q8_0 now gains
  as much as Q4_0: the interleaving that ate the `smmla` saving is now done
  once per four rows (weights) or once per matmul (activations).
- Round 2 ran 25–30 % slower for every build, llama.cpp included, although
  each run started below 48 °C; only same-round comparisons hold.
- Gap to llama.cpp (its tok/s ÷ ours, same round): Q4_0 1.9–2.2×, Q8_0
  1.7–2.3× (2×2: 2.3–3.3×).

**Profile** (`profile-pp512-q4_0.txt`, `profile-pp512-q8_0.txt`;
`simpleperf record -e cpu-cycles:u`, pp512, cores 2–7): the `mmla4` kernel
58 % (Q4_0) / 52 % (Q8_0), attention 17 % / 19 % plus `expf` 3–4 %, pool
workers waiting 12 % / 14 %, quantizing activations 3 %. Per 32-value block
the Q8_0 loop is about 75 instructions, 16 of them `smmla`: 8 `zip`s
interleave the four weight rows, 8 `movi` + 4 `add` start and join two
integer chains per tile, the rest are loads, address arithmetic and the
float scaling.

**One chain per tile** (`raw/prefill-ab-chain4/`, against `0374996`, same
script, two rounds): each tile's four `smmla`s in one chain instead of two
chains of two, which drops 4 `movi` and 4 `add` per block; the four tiles
are independent, and the integer sum is exact, so the output is unchanged
bit for bit. Prefill tok/s, round 1 / round 2:

| | pp64 | pp256 | pp512 |
| --- | --- | --- | --- |
| Q4_0 one chain | 322 / 288 | 297 / 305 | 256 / 266 |
| Q4_0 two chains | 314 / 328 | 282 / 291 | 248 / 247 |
| Q4_0 llama.cpp | 623 / 625 | 618 / 630 | 461 / 467 |
| Q8_0 one chain | 392 / 379 | 320 / 323 | 276 / 275 |
| Q8_0 two chains | 335 / 335 | 308 / 310 | 268 / 266 |
| Q8_0 llama.cpp | 626 / 617 | 640 / 645 | 478 / 449 |

+3–7 % at pp256 and pp512 in both rounds, +13–17 % for Q8_0 at pp64; Q4_0
pp64 is noisy (+3 % / −12 %). Kept.

**Batched attention for prefill** (`raw/prefill-ab-attn-batch/`, against
`37b57ca`, two rounds): prefill called attention once per token, each call
sharing out only the 14 heads among 6 threads (at pp512: 24 × 512 calls,
each waiting for its slowest thread). Now one call per layer hands out all
t × 14 (token, head) pairs in blocks, last token first (it sees the most
positions), so the cheap items fill in at the end. Each head is computed as
before; `batched_prefill_matches_llama2c` now also runs on four threads and
must give the same activations bit for bit. Prefill tok/s, round 1 /
round 2:

| | pp64 | pp256 | pp512 |
| --- | --- | --- | --- |
| Q4_0 batched | 323 / 342 | 307 / 314 | 271 / 273 |
| Q4_0 per token | 315 / 332 | 302 / 302 | 265 / 258 |
| Q4_0 llama.cpp | 620 / 625 | 564 / 562 | 472 / 473 |
| Q8_0 batched | 389 / 403 | 332 / 334 | 305 / 301 |
| Q8_0 per token | 377 / 392 | 323 / 324 | 277 / 274 |
| Q8_0 llama.cpp | 611 / 621 | 620 / 649 | 478 / 479 |

+2–10 % in every cell, most at pp512 (Q8_0 +10 %), where attention is
largest. Gap to llama.cpp now 1.7–1.9× (Q4_0) and 1.5–1.9× (Q8_0).

## Repacked weights, NEON activation quantizer (item 6.5, third step; `raw/pending-checks/`)

Built while the phone was away, then checked on it on 2026-10-28 evening.
**Conditions differ from the morning**: the phone was charging (AC, 19 → 40 %),
clock caps 1.79 GHz (performance) / 1.96 GHz (prime) instead of ~2.4 GHz,
every run started below 48 °C. Absolute numbers are therefore not
comparable with the tables above; each change is an A/B against the commit
before it, alternating, two rounds (`scripts/prefill-ab.sh` with `NEW=`).

**Repack** (`31513f0`): quantized matrices stored four rows interleaved
8 bytes at a time with the four scales together (`BlockQ8_0x4`,
`BlockQ4_0x4`, llama.cpp's `4x8` layout; Q4_0 nibbles stored XOR 8),
repacked once in `Transformer::new` where i8mm is present, mapped pages
released after the copy. The `smmla` loop loses its 8 weight `zip`s per
block; the Q4_0 unpack is one shift or one mask (values ×16, undone exactly
by a fixed-point int→float conversion). Decode uses a matching `sdot` GEMV
(8 `sdot`s per block of four rows, as llama.cpp's `gemv_q4_0_4x8_q8_0`).
Prefill output is bit-identical to the row-major kernel (PPL unchanged to
the last digit: 19.7486 / 16.3532).

| tok/s, round 1 / round 2 | pp64 | pp256 | pp512 | tg64 @ d0 | tg64 @ d256 | tg64 @ d512 |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_0 repacked | 374 / 366 | 323 / 312 | 286 / 285 | 149 / 147 | 134 / 133 | 121 / 121 |
| Q4_0 row-major | 305 / 279 | 274 / 273 | 252 / 251 | 113 / 110 | 102 / 104 | 95 / 97 |
| Q4_0 llama.cpp | 554 / 553 | 546 / 544 | 419 / 418 | 151 / 140 | 135 / 131 | 121 / 114 |
| Q8_0 repacked | 400 / 404 | 333 / 335 | 301 / 299 | 97 / 94 | 87 / 86 | 82 / 83 |
| Q8_0 row-major | 349 / 350 | 307 / 309 | 286 / 275 | 89 / 89 | 82 / 84 | 78 / 76 |
| Q8_0 llama.cpp | 558 / 550 | 573 / 566 | 428 / 421 | 92 / 95 | 84 / 90 | 78 / 79 |

- Prefill: Q4_0 +13–31 %, Q8_0 +5–15 %.
- GEMV on one prime core (`gemv-bench.txt`, `_b_sdot` vs `_b_sdot_x4`):
  Q4_0 10.3–12.6 → 13.8–19.1 GB/s (256 MB matrix 10.5 → 19.1), Q8_0
  unchanged to +23 % (DRAM 29.3 → 33.3).
- Cost: load until the model is ready 192–194 ms for Q4_0 and 305–320 ms
  for Q8_0, against ~75 ms without repacking (`bench_suite` now times
  `Transformer::new` too). Peak RSS does not grow (Q4_0 343 vs 349 MB, Q8_0
  585 vs 590 MB in the A/B runs): the mapped pages are released as each
  matrix is copied. The repacked weights are heap memory, which Android's
  low-memory killer counts; mapped file pages it could simply drop.
- **Decode: Q4_0 +25–34 %, now level with llama.cpp** (it was compute-bound
  on the nibble unpack, item 6.4); Q8_0 +3–9 %, level with or ahead of
  llama.cpp.

**NEON activation quantizer** (`0d1510f`): Q8_0 quantization of the
activations, byte-identical to the reference by construction (exact max,
same scale math, `vcvtaq` rounds like `roundf`; checked against llama.cpp's
bytes on the phone). It runs on one thread before every quantized matmul, so
the other threads wait for it.

| tok/s, round 1 / round 2 | pp64 | pp256 | pp512 |
| --- | --- | --- | --- |
| Q4_0 NEON quantizer | 466 / 469 | 417 / 415 | 339 / 343 |
| Q4_0 scalar quantizer | 377 / 415 | 355 / 362 | 307 / 311 |
| Q4_0 llama.cpp | 608 / 624 | 622 / 631 | 472 / 472 |
| Q8_0 NEON quantizer | 537 / 518 | 443 / 455 | 368 / 374 |
| Q8_0 scalar quantizer | 440 / 448 | 374 / 375 | 331 / 326 |
| Q8_0 llama.cpp | 620 / 624 | 644 / 624 | 466 / 477 |

+10–24 %. **Gap to llama.cpp's prefill now 1.15–1.5×** (Q4_0 1.3–1.5×,
Q8_0 1.15–1.45×); it was 2.3–3.3× before this item's first step.

**Attention split across positions — tried and reverted** (`bdbf963`,
reverted in `655c34e`; `raw/pending-checks/chunked-decode*/`). Past 128
positions each head was computed in chunks (unnormalised sum and its max /
exp-sum per chunk, merged by `e^(m_i − m)` rescaling), so decode had
(head, chunk) items to share among threads instead of 14 heads. Correct
(numpy up to 1024 positions; llama2.c over the whole 256 context) but no
faster: tg64 at d0 / d256 / d512 unchanged, d1024 −2 % and d2048 −5 % for
Q4_0, +1–2 % for Q8_0. After the repack, attention no longer limits decode
at these depths: our d0 → d512 drop (−20 %) is llama.cpp's, and without
chunking we decode faster than llama.cpp at depth (Q4_0 109 vs 101–102
tok/s at d1024, 84 vs 78 at d2048). It changed PPL by 0.02–0.04 (rounding:
llama.cpp's own flash-attention on/off moves its PPL by 0.021 / 0.019 on
this phone, `llamacpp-ppl-*-fa-*.txt`). The long-context llama2.c test it
came with stays.

**NEON softmax — tried and reverted** (`85d1a12`, reverted in `351d83c`;
`raw/pending-checks/softmax-*/`): a vector `exp` (range reduction plus a
degree-5 polynomial, max relative error 1.9e-7) in attention's softmax.
Decode tg64 at d0–d512 −4 % to +1 % (one Q8_0 run −10 %), prefill −4 % to
+5 %: nothing measurable, so the scalar `expf` stays.

**After both reverts** (`351d83c` = repack + NEON quantizer): 89/89 tests on
the phone, PPL 19.7486 / 16.3532, the same digits as before the repack
(`ppl-*-final.txt`): both changes are bit-identical by design.

## Prefill attention by K/V head (2026-10-29; `raw/prefill-gap/`, `raw/prefill-ab-attn-group/`)

A fresh pp512 profile after the repack and the NEON quantizer (Qwen2.5-0.5B,
cores 2–7, phone on battery, caps 2.75 / 2.65 GHz,
`raw/prefill-gap/profile-pp512-*.txt`):

| Symbol | Q4_0 | Q8_0 |
| --- | --- | --- |
| `mmla4_*x4` (the matmuls) | 53.9 % | 49.1 % |
| `attention_head` | 22.6 % | 24.2 % |
| pool threads waiting | 6.1 % | 6.6 % |
| `expf` (softmax) | 4.5 % | 4.8 % |
| quantize + interleave activations | 3.4 % | 3.6 % |

Attention is about 2 % of the arithmetic (pp512: 24 layers × 14 heads ×
512 × ~256 positions × 64 × 2 multiply-adds ≈ 11 GFLOP, against ~500 G
integer operations in the matmuls) but over a fifth of the time.

**Change** (`attention_group`): in Qwen2.5 seven query heads read each K/V
head (grouped-query attention). Prefill now hands out (token, K/V head)
items, and one item computes all seven heads: each cached key and value
row is loaded once and used seven times while in L1, instead of being read
from L2 once per head. Each head's arithmetic is unchanged and in the same
order, so the output is bit-identical (`attention_group_matches_numpy`
checks this bit for bit against the per-head kernel; 90/90 tests on the
phone). Decode keeps one item per head: with only two K/V heads, grouping
would leave four of six threads idle.

**Result**, measured as cycles so the clock does not matter (pp512,
`simpleperf record`, same phone state, `profile-pp512-*-{base,new}.txt`):

| | Q4_0 | Q8_0 |
| --- | --- | --- |
| attention cycles, before → after | 7.94 G → 6.77 G (−15 %) | 7.76 G → 6.66 G (−14 %) |
| all cycles | −3 % | −5 % |

The timed A/B (`prefill-ab-attn-group/`) was too noisy to add much: round 2
ran warm and the clock dropped mid-run (one pp512 at 137 tok/s). Round 1,
the cool one: Q4_0 +7–10 % at pp64–pp1024, Q8_0 −3 % to +8 %.

So L2 traffic was only a small part of attention's cost. What remains is in
the inner loops: each score ends in a horizontal add of four vectors, and
the weighted sum of values loads and stores the output for every
multiply-add (16 loads, 16 FMAs, 16 stores per position and head).

**Then the inner loops** (`group_fixed`, head sizes 64 and 48 as const
generics; `raw/attn-blocked/`). One head's vectors now stay in registers:
the query while its scores are computed, four positions at a time with two
accumulators each, the four sums reduced by two pairwise adds instead of
four horizontal ones; and the output (16 registers) while the values are
summed, so each multiply-add loads only the value. Positions go in blocks
of 32 (8 KB of keys), the group's heads inside a block, so the block stays
in L1 for all of them. `attention_head` (decode) is the same code with one
head.

| Qwen2.5-0.5B Q4_0, cycles (two rounds each) | before (`6708f5b`) | after |
| --- | --- | --- |
| pp512: attention | 6.59 G (19.7 %) | 3.31 G (11.1 %), −50 % |
| pp512: all | 33.46 G | 29.83 G, −11 % |
| prefill to depth 1024: attention | 27.5 G | 13.7 G, −50 % |
| 64 decode steps at depth 1024: attention | 6.0 G | 4.7 G, −22 % |

The summation order changed, so the results are no longer the same bits as
before: PPL Q4_0 19.7486 → 19.7470, Q8_0 16.3532 → 16.3494 (`ppl-qwen-*.txt`),
a tenth of what llama.cpp's own flash-attention switch moves its PPL here.
92/92 tests on the phone (new numpy cases with head size 48, stories15M's;
`monitor::samples_until_stopped` failed once in a full run while the phone
was asleep, passed on every rerun: its first battery read was slow under
load).

## Q4_K_M files: Q5_0, Q4_K, Q6_K (2026-10-29; `raw/kquants/`)

`Q4_K_M` is the GGUF mix most people download. Made with
`llama-quantize … Q4_K_M` (`d00685831`, a static build: the host's
`LD_LIBRARY_PATH` points at another llama.cpp release), these two models
contain:

| Matrices | SmolLM2-135M | Qwen2.5-0.5B |
| --- | --- | --- |
| Q5_0 | q, k, o, gate, up, half of v (166) | q, k, o, gate, up, half of v (132) |
| Q4_K / Q6_K | `ffn_down`, 16 / 14 layers | `ffn_down`, 12 / 12 layers |
| Q8_0 | the other half of v, token embedding | the other half of v, token embedding |

K-quants pack 256 weights per super-block, so a row must be a multiple of
256 long. Only `ffn_down`'s rows (1536 and 4864) are; for the others (576,
896) llama.cpp falls back to Q5_0 (for Q4_K) and Q8_0 (for Q6_K). On small
models "Q4_K_M" is mostly Q5_0.

New formats (`src/quant.rs`):

- **Q5_0** (22 bytes per 32, 5.5 bits): Q4_0 plus a fifth bit per weight in a
  32-bit `qh`. Our quantizer gives llama.cpp's bytes exactly (12 cases,
  `quantizers_are_byte_identical_to_llama_cpp`), so `-q q5_0` works on any
  model as well.
- **Q4_K** (144 bytes per 256, 4.5 bits): 8 sub-blocks of 32, each with a
  6-bit scale and a 6-bit min, quantized against an f16 `d` / `dmin`.
- **Q6_K** (210 bytes per 256, 6.56 bits): 16 sub-blocks of 16 with int8
  scales.

The `gguf` package decodes Q4_K / Q6_K but cannot encode them, so their
goldens are random block bytes (`tools/gen_quant.py`, `testdata/quant_k/`):
every bit of the layout is exercised, and our decoders give gguf's values
exactly. The dot products quantize the activation to Q8_0 as for every
other format: a 32-weight activation block meets one Q4_K sub-block (`d_x ·
(d·sc·Σq·x − dmin·m·Σx)`, both sums exact integers) or two Q6_K ones.
llama.cpp uses Q8_K for K-quants (one f32 scale per 256), so the two
engines round the activation differently here.

**Accuracy** (wikitext-2, 20 × 512, KL against each engine's f32 logits;
ours on the phone, llama.cpp on the host):

| Model, format | PPL ours | PPL llama.cpp | Mean KLD ours | Mean KLD llama.cpp | Same top ours / llama.cpp |
| --- | --- | --- | --- | --- | --- |
| SmolLM2-135M Q4_K_M | 20.7201 | 20.7784 ± 0.851 | 0.051455 | 0.053319 ± 0.000979 | 85.86 % / 86.14 ± 0.48 % |
| SmolLM2-135M Q5_0 (`--pure`) | 21.6167 | 21.6071 ± 0.902 | 0.087297 | 0.087117 ± 0.001080 | 81.47 % / 81.67 ± 0.54 % |
| Qwen2.5-0.5B Q4_K_M | 16.7088 | 16.7657 ± 0.688 | 0.031828 | 0.033112 ± 0.000638 | 90.43 % / 90.10 ± 0.42 % |

PPL and top-token agreement match within llama.cpp's error bars. Our
Q4_K_M KLD is a little lower in both models (about two of llama.cpp's
standard errors). The likely reason is the activation rounding above: one
scale per 32 values is finer than one per 256. Q5_0, where both engines use
Q8_0 activations, agrees to 0.2 %. For scale: Q4_K_M costs Qwen2.5-0.5B
+2.7 % perplexity (KLD 0.03), Q4_0 +21 % (KLD 0.18, `results/phase-7.md`),
for a file 1.4× larger (380 vs 265 MB).

**Speed** (Qwen2.5-0.5B, cores 2–7, battery, `raw/kquants/speed*/`, one
round each, llama.cpp in the same run):

| | pp64 | pp256 | pp512 | tg64 @ d0 | tg64 @ d512 |
| --- | --- | --- | --- | --- | --- |
| Q4_K_M ours, Q5_0 row-major | 164.5 | 171.6 | 164.2 | 75.2 | 68.4 |
| Q4_K_M llama.cpp (same run) | 149.6 | 150.4 | 139.6 | 85.0 | 73.7 |
| Q4_K_M ours, Q5_0 repacked | **208.2** | **212.7** | **199.4** | **96.6** | **79.4** |
| Q4_K_M llama.cpp (same run) | 168.2 | 167.8 | 152.8 | 85.7 | 75.7 |
| Q4_0 ours (first run) | 441.1 | 377.1 | 335.6 | 153.4 | 124.0 |

Q4_K_M is much slower than Q4_0 in both engines, more than its 1.4× size
explains. A profile showed why (`profile-tg128-q4_k_m.txt`,
`profile-pp512-q4_k_m.txt`): decode spent 39 % in the Q5_0 kernel, which
spread each block's fifth bits into bytes with a table lookup and a bit
test, row by row; prefill spent 35 % there (and unpacked the same rows again
for every four tokens) and 47 % in the K-quant kernels, which have no
`smmla` version yet and run one token at a time.

**Q5_0 repacked four rows at a time** (`BlockQ5_0x4`): the nibbles laid out
as for Q4_0x4, and the fifth bits stored *inverted* and arranged so that the
k-th byte holds, in bit i, the bit for lane k of the i-th vector the kernel
builds. An inverted fifth bit is the sign bit of the 5-bit two's-complement
number q − 16, so `(nibble << 3) | (bit << 7)` read as int8 is exactly
8·(q − 16): per 16 weights a shift, a mask and one bit-select. The ×8 is
removed exactly when the integer sums are converted to float, so results are
bit-identical (repacked = row-major bit for bit in the tests, and the PPL
above reproduces to the last digit, `aether-phone-qwen-q4_k_m-repacked.txt`).
Decode +28 % / +16 %, prefill +21–27 %: **Q4_K_M now runs faster than
llama.cpp's on this phone** (decode +13 % at d0, +5 % at d512; prefill
+24–31 %), even though our K-quant matrices still go one token at a time.

**`smmla` for Q4_K / Q6_K** (prefill): each group of four rows is unpacked
once per matmul (`unpack_k4`) into 32-value blocks of int8 values (Q4_K: q
in 0..15; Q6_K: q − 32) with, per row, an f32 scale for each 16-value half
and a bias on Σx (Q4_K's −dmin·m, 0 for Q6_K). `mmla4_k` then runs every
token over the group: two `smmla` chains per tile (one per half, each with
its own scales) plus one FMA for the bias, whose Σx·d per token is computed
with the activation pairs. The unpacked group is scratch (no memory kept),
and its cost is shared by all tokens.

Accuracy: Qwen2.5-0.5B Q4_K_M PPL 16.7354, KLD 0.031719, same top 90.29 %
(`aether-phone-qwen-q4_k_m-mmla.txt`), against 16.7088 / 0.031828 / 90.43 %
with the per-token `sdot` kernels and llama.cpp's 16.7657 ± 0.688: only the
float summation order changed. 92/92 tests on the phone (the K-quant test
also checks that 1 and 3 threads give the same bits).

Speed: the timed A/B (`speed-kmmla/`) was unusable (the clock changed mid-run:
llama.cpp's own pp512 fell to 61 tok/s), so cycles again (pp512, two rounds
each, `profile-pp512-kmmla-*.txt`):

| | all cycles | K-quant matrices | pp512 in the same runs |
| --- | --- | --- | --- |
| before (`4bcfa25`, `sdot` per token) | 79.4 G | 40.9 G (51.5 %) | 173–175 tok/s |
| after (`mmla4_k`) | 46.8 G (−41 %) | ~10.5 G (`mmla4_k` 16 %, unpack in `quant_rows_mmla` ~5 %) | 331 tok/s |

Q4_K_M prefill is now about twice llama.cpp's on this phone (153–168 tok/s
in the cooler runs above).

## `smmla` tiles of 4 rows × 8 tokens (2026-10-29; `raw/tile8/`)

With attention halved, the `smmla` kernel is 61 % of Q4_0 pp512. Counters
on it (`raw/attn-blocked/counters-pp512-q4_0.txt`): 3.6 instructions per
cycle, 24 % of cycles stalled in the back end. It issues as fast as the
core allows; what limits it is how many instructions it needs. Its loop
(disassembly) was 58 instructions per block for 16 `smmla`s: 16 `smmla`, and
42 for weight loads, the nibble unpack, scale conversions, activation
loads and four scale-and-add steps, all shared by only 4 tokens.

Tiles now cover 4 rows × 8 tokens (`NP` = 4 token pairs, for every format
including the unpacked K-quants): the weights' loads, unpack and scale
conversion serve twice as many `smmla`s. The loop is 99 instructions for 32
`smmla`s (3.1 per `smmla` instead of 3.6). A first version kept the 2×4
accumulator array as an array of arrays and the compiler wrote it to the
stack every iteration (8 stores); a flat array keeps it in registers.

Whole-run cycles (pp64 + pp512, two rounds each, `stat-*.txt`) and the
timed prefill in the same runs:

| Qwen2.5-0.5B | cycles before → after | pp64 tok/s | pp512 tok/s |
| --- | --- | --- | --- |
| Q4_0 | 34.4 G → 31.5 G (−8 %) | 493 → 545 | 417 → 435 |
| Q8_0 | 31.0 G → 30.2 G (−2.5 %) | 564 → 545 | 441 → 439 |
| Q4_K_M | 51.7 G → 43.2 G (−16 %) | 288 → 336 | 282 → 365 |

Q8_0 has no unpack to share, so it gains least. Each output is computed
the same way whatever tile it is in, so results are bit-identical (Q4_0 PPL
19.7470 again, `ppl-qwen-q4_0.txt`; 92/92 tests).

## Threads waiting a third of prefill (2026-10-29; `raw/gap-2026-10-29/`, `raw/par-rows/`)

Measured as cycles, our prefill already cost less than llama.cpp's: the
cycles of a pp512 run minus those of a pp1 run (load and setup cancel),
all threads, two rounds each (`gap-2026-10-29/cycles/`):

| Qwen2.5-0.5B, 511 tokens | ours | llama.cpp |
| --- | --- | --- |
| Q4_0 | 9.8 G | 12.6 G |
| Q8_0 | 9.3 G | 15.0 G |
| Q4_K_M | 12.8 G | 41.3 G |

Yet its wall time was longer. `cpu-cycles:u` counts only cycles a thread
runs, and our pool's workers spin for a while and then sleep in the kernel.
With a build whose workers never sleep (`SPIN = u32::MAX`, measurement
only), every thread runs for the whole prefill, so the total is ~6 × wall
time in cycles: Q4_0 16.0 G, Q8_0 15.4 G, Q4_K_M 19.1 G. **Our threads waited
for 33–39 % of prefill.**

Where: every layer, the calling thread did the per-token steps between the
matmuls alone while the five workers waited — the RMSNorms, bias + RoPE,
SwiGLU (an `exp` for each of t × 4864 values), the residual adds, and inside
every quantized matmul the activations' Q8_0 quantization and pair
interleave. Each is short; together they were a sixth of the work, and
everything else waited for them.

**Change**: `pool::par_chunks_mut` splits a buffer into rows (tokens)
across the pool, and those steps use it. Each row is computed as before, so
results are bit-identical (Q4_0 PPL 19.7470 again, `par-rows/ppl-qwen-q4_0.txt`;
92/92 tests on the phone).

| 511 tokens, cycles | never-sleeping workers before → after | busy only, after |
| --- | --- | --- |
| Q4_0 | 16.0 G → 11.1 G (−31 %) | 10.2 G |
| Q8_0 | 15.4 G → 10.4 G (−32 %) | 9.8 G |
| Q4_K_M | 19.1 G → 13.9 G (−27 %) | 13.1 G |

Waiting is now 6–8 % of thread time. Busy cycles rose 3–5 % (more,
shorter jobs to hand out). The 20-chunk Q4_0 perplexity run took 43.0 s
instead of 86.0 s an hour earlier.

**Against llama.cpp, timed, same runs** (cores 2–7, two rounds, phone
screen off so clocks capped at 1.79 / 1.02 GHz for both engines,
`par-rows/timed/`):

| Qwen2.5-0.5B | pp64 | pp256 | pp512 |
| --- | --- | --- | --- |
| Q4_0 ours | 619 / 686 | 712 / 668 | 445 / 510 |
| Q4_0 llama.cpp | 621 / 613 | 625 / 604 | 453 / 447 |
| Q8_0 ours | 778 / 794 | 713 / 709 | 651 / 651 |
| Q8_0 llama.cpp | 577 / 620 | 612 / 634 | 318 / 449 |
| Q4_K_M ours | 401 / 320 | 477 / 501 | 494 / 335 |
| Q4_K_M llama.cpp | 166 / 166 | 164 / 157 | 73 / 67 |

**The prefill gap is closed**: level with llama.cpp for Q4_0 (0 to +14 %),
ahead for Q8_0 and far ahead for Q4_K_M. These ran with the screen off; a
check with the screen on (higher, uncapped clocks) is still to do.

## Q4_K decode: scales as vectors (2026-10-29; `raw/kdecode/`)

**Change** (`dot4_q4_k_q8_0`, decode): the 8 six-bit scales and 8 mins of a
super-block are decoded at once with 32-bit word operations
(`q4_k_scales_mins`, ggml's `utmp` trick) instead of a branchy call per
sub-block. Each sub-block's `sdot` result is reduced to one lane with
pairwise adds (`vpaddq_s32`), so the scales (d·sc·d_x) and the min term
(dmin·m·Σx·d_x) apply as two vector FMAs per four sub-blocks instead of one
scalar FMA and a scalar min sum each.

**Result** (Qwen2.5-0.5B Q4_K_M, cores 2–7, cycles of 128 decode steps =
tg128 − tg1, two rounds; `cycles.txt`):

| | cycles, 128 steps | tg128 tok/s (same runs) | `dot4_q4_k_q8_0` share |
| --- | --- | --- | --- |
| before (`9d8e8c1`) | 19.06 G / 18.78 G | 94.9 / 100.4 | 8.6 % |
| after | 18.09 G / 18.18 G (−4 %) | 107.8 / 107.6 | 6.3 % |

The kernel's own cycles fell by about a quarter; the whole step by 4 %
because Q4_K is only part of `ffn_down`. PPL 16.7330 (`ppl.txt`; 16.7354
before, only the float summation order changed).

**The same for Q6_K — tried and reverted** (`raw/kdecode-q6k/`): reducing
each Q6_K sub-block's scaled `sdot` to one lane and applying d·d_x as two
vector FMAs made 128 decode steps cost 18.60 / 18.43 G cycles against
18.10 / 18.06 G (+2.5 %). Q6_K's scales already apply as integer vector
multiplies, so there was only one scalar FMA per sub-block to save, and the
eight extra live accumulators per row cost more than that.

## Decode: workers falling asleep (2026-10-29; `raw/decode-wait/`, `raw/decode-spin/`)

A build whose workers never park (`SPIN = u32::MAX`) decoded Q4_0 much
faster than the normal one in some runs (156 vs 94 tok/s), so: how long is
`SPIN`, and how often do workers park during decode?

- `std::hint::spin_loop()` is an `isb` on aarch64, ~14 ns on a prime core:
  20 000 polls last **0.28 ms**, not the "tens of microseconds" the comment
  said. Decode never leaves the pool idle that long by itself (193 jobs per
  token, microseconds apart).
- A measurement-only build counting parks (not committed) still saw tens to
  hundreds of parks per 128 decode steps, and the caller waiting long
  enough to fall back to `yield` thousands of times: another thread had
  preempted one of ours, the others ran out of spins, parked, and then took
  long to wake.

**A/B** (`decode-spin/timed2.txt`): Qwen2.5-0.5B, cores 2–7, 4 rounds with
the order alternated, tg128 median of 2 reps each. The phone was hot (44–67
°C), so the numbers are low, but both builds share the conditions:

| tok/s (median of rounds) | SPIN 20 000 | SPIN 200 000 | parks / 128 steps |
| --- | --- | --- | --- |
| Q4_0 @ d0 | 108 | 127 | 0–543 → 0–45 |
| Q4_0 @ d512 | 83 | 106 | 0–84 → 0–8 |
| Q8_0 @ d0 / d512 | 95 / 69 | 96 / 69 | 24–838 → 0–54 |
| Q4_K_M @ d0 / d512 | 77 / 64 | 78 / 65 | 22–634 → 0–40 |

An earlier 3-round run (`decode-spin/timed.txt`) had Q4_0 @ d512 at 72–98
vs 121–136 tok/s. `SPIN` is now 200 000 (2.8 ms). The cost: after the last
job of a request, the workers spin 2.8 ms instead of 0.28 ms before they
sleep, a few milliseconds of five busy cores per request.

## Decode: q, k, v (and gate, up) as one pool job (2026-10-29; `raw/decode-fused/`)

**Change**: `linear_many_on` multiplies several matrices that read the same
input as one job: the activation is quantized once, and the threads take
blocks of rows from all the matrices (each block inside one matrix, aligned
as before). Decode uses it for q/k/v and for gate/up, so a Qwen2.5-0.5B
token is 121 pool jobs instead of 193; k and v (128 rows each) no longer
pay a hand-off of their own. Every row is computed as before: greedy
output is byte-identical for Q4_0, Q8_0, Q4_K_M; 92/92 tests on the phone.

**Result** (cores 2–7, 4 rounds with the order alternated, phone 46–63 °C;
`timed.txt`):

| tg128 tok/s, median of rounds | before (`392f9c8`) | after |
| --- | --- | --- |
| Q4_0 @ d0 / d512 | 133.1 / 113.0 | 133.9 / 114.5 |
| Q8_0 @ d0 / d512 | 95.8 / 70.1 | 97.9 / 69.9 |
| Q4_K_M @ d0 / d512 | 78.2 / 63.9 | 80.2 / 66.5 |

Cycles of the whole run fall 1–2 %. Small, as expected: the hand-off is
~1 µs and only 72 of them go per token (~0.1 ms of 8–15 ms).

## Decode GEMV: accumulators in registers, Q5_0 fifth bit with `sri` (2026-10-29; `raw/q5-gemv/`, `raw/decode-unroll-sri/`)

Q5_0 is most of a Q4_K_M file and took 31 % of its decode. `gemv_bench`
(now with Q5_0) on one prime core: the Qwen up matrix as Q5_0×4 took 530 µs
against 282 µs as Q4_0×4, for 22 % more bytes, so compute-bound. The
disassembly showed two things:

1. **`gemv4` kept its two accumulators on the stack**: `acc[b & 1]`
   indexed by a variable made every block a load, an FMA and a store
   (`ldr q4, [x10, x12]` … `str`). Now the loop takes two blocks per
   iteration, the even one into `acc[0]` and the odd one into `acc[1]`, as
   before (same sums in the same order: bit-identical). This applies to
   Q4_0 / Q8_0 / Q5_0.
2. **Q5_0's unpack was 5 instructions per vector** (nibble: shift, mask;
   fifth bit: shift, mask, or). `sri #1` (shift right and insert) puts the
   nibble, already at bits 4..7, under bit 7 of the shifted fifth-bit
   register in one instruction: 3 per vector. LLVM rewrote the
   `vsriq_n_u8` intrinsic back into and/or, so it is an `asm!` (`sri1`).

One prime core, round 2 of 2 (`unroll-sri-ab.txt`, µs per GEMV):

| matrix | format | before | unrolled | + `sri` |
| --- | --- | --- | --- | --- |
| up 4864×896 | Q8_0 / Q4_0 / Q5_0 | 145 / 165 / 288 | 124 / 167 / 276 | 124 / 141 / 210 |
| down 896×4864 | Q8_0 / Q4_0 / Q5_0 | 144 / 140 / 273 | 123 / 138 / 259 | 122 / 145 / 228 |
| 8192² (DRAM) | Q8_0 / Q4_0 / Q5_0 | 2160 / 2002 / 3949 | 1756 / 1849 / 3901 | 1747 / 1851 / 2886 |

Whole model, decode (Qwen2.5-0.5B, cores 2–7, 4 alternated rounds, phone
52–63 °C, `decode-unroll-sri/timed.txt`); greedy output byte-identical in
all three formats, 92/92 tests:

| tg128 tok/s, median | before (`8d3098e`) | after | run cycles |
| --- | --- | --- | --- |
| Q4_K_M @ d0 / d512 | 81.8 / 68.7 | 90.8 / 74.3 | −7 % |
| Q4_0 @ d0 / d512 | 133.5 / 108.8 | 131.8 / 112.1 | −1 % |
| Q8_0 @ d0 / d512 | 98.8 / 70.2 | 89.0 / 70.3 | 0 % |

With six cores Q4_0 and Q8_0 decode is limited by memory, so a faster
kernel on one core does not show there (the Q8_0 d0 median is pulled down
by two slow rounds; its cycles are unchanged). Q4_K_M gains 8–11 %.

**Q6_K with `sri #2`** (`raw/kdecode-q6k-sri/`): `q − 32` in 6-bit two's
complement is `q` with bit 5 flipped, so the int8 4·(q − 32) is built from
the two high bits (from `qh`, flipped once per load by XOR 0xAA, shifted to
bits 6..7) and the nibble inserted below them with `sri #2`: 3 instructions
per vector instead of 6, the ×4 folded into the float scale. The timed
A/B ran at a capped 1.5 GHz and was inconclusive, so profiles (tg256, two
rounds each): `dot4_q6_k_q8_0` 4.22 / 4.22 G cycles → 3.86 / 3.90 G (−8 %),
the whole run −1 to −3 %. The Q6_K rows are not repacked (four separate
row streams), so the kernel is close to its memory speed already. Tests
92/92 (the K-quant test checks against gguf's decoder).

## Night of 2026-10-29: where things stand, and a measurement caveat (`raw/night-2026-10-29/`, `raw/prefill-regress*/`)

`scripts/prefill-ab.sh`, Qwen2.5-0.5B, cores 2–7, two rounds; "start" is
`9d8e8c1` (before tonight's decode work), "now" is `66aed73`, tok/s:

| test | format | start | now | llama.cpp |
| --- | --- | --- | --- | --- |
| tg64 @ d0 | Q4_0 / Q8_0 / Q4_K_M | 156–157 / 99–102 / 83–87 | 158–163 / 101–102 / **104–105** | 138–143 / 91–96 / 86 |
| tg64 @ d512 | Q4_0 / Q8_0 / Q4_K_M | 129 / 62–82 / 40–65 | 100–102 / 77–84 / 73–87 | 121 / 76–77 / 66–68 |
| pp256 | Q4_0 / Q8_0 / Q4_K_M | 554–586 / 629–637 / 422–434 | 584–607 / 627–647 / 440–449 | 507–534 / 541–568 / 136–140 |
| pp512 | Q4_0 / Q8_0 / Q4_K_M | 528–564 / 570–583 / 418 | *431–440 / 460–466 / 289–294* | 366–401 / 369–404 / 60–82 |

Decode is ahead of llama.cpp in every format at d0 (Q4_K_M by 22 %). The
pp512 column and Q4_0 at d512 looked like a regression, so it was
bisected (`prefill-regress*/`), and **no commit is responsible**:

- The same binary (byte-identical) gives pp512 either ~530–565 or ~435–460
  tok/s from one process to the next, whichever commit it is and whatever
  its file name (`name-test.txt`). A process that starts slow stays slow
  for all its tests (tg16 after a slow pp512 is slow too).
- Run alone (`--pp 512`), six runs in a row were all fast (517–529); the
  slow mode shows up in multi-test runs after pp256 has warmed the phone.
- Not transparent huge pages: the phone has THP `always` but defrag
  `never`, and our process had 0 kB of `AnonHugePages` in every run.

**The cause** (`prefill-regress-q4_0/freq-samples.txt`): sampling cpufreq
every 0.25 s during such runs, the prime cores' (6–7) `scaling_max_freq`
drops from 1.69 GHz to **1.02 GHz** partway through the slow runs and
stays there; in the fast runs it never does. It is a thermal / power cap
on the prime cluster, applied after the sustained load of the earlier
tests. (Both caps are far below the 3.53 / 4.32 GHz maximums because the
screen is off overnight.) So timed A/B pairs on this phone can be off by
20 % in either direction depending on whether the cap hit; cycle counts
are unaffected, and timed comparisons should sample `scaling_max_freq`
or use more rounds.

**With the caps recorded** (`raw/caps-2026-10-29/prefill/`, screen kept on,
two rounds; each run's lowest caps in its file): runs at the full 2.4 GHz
caps against each other, tok/s:

| format | pp64 | pp256 | pp512 |
| --- | --- | --- | --- |
| Q4_0 ours / llama.cpp | 811–822 / 738 | 761–782 / 732 | 715–718 / 540 |
| Q8_0 ours / llama.cpp | 877 / 714–717 | 845 / 745–758 | 768 / 539–549 |
| Q4_K_M ours | 461–463 | 575–581 | 586–589 |

Every llama.cpp Q4_K_M run and one of each other pair were capped (prime
1.02 GHz) and are left out: e.g. our Q8_0 pp512 469 at the cap vs 768
without, llama.cpp's Q4_0 pp512 239 vs 540. With equal clocks our prefill
is 10–40 % ahead of llama.cpp for Q4_0 and Q8_0.

## K-quant prefill: rows unpacked with NEON (2026-10-29; `raw/unpack-k4/`, `raw/q4km-pp64/`)

At pp64, Q4_K_M ran slower than at pp256 (461 vs 578 tok/s). The pp64
profile (`q4km-pp64/profile-pp64.txt`) had `quant_rows_mmla` itself at 28 %:
the inlined `unpack_k4`, which unpacks each group of four K-quant rows once
per matmul (its cost does not shrink with fewer tokens, so at pp64 it
weighs 8× more than at pp512). It decoded value by value (`q4_k_values`,
`q6_k_values`) and scattered bytes one at a time.

**Change**: each 32-value sub-block is built as two int8 vectors with the
same loads as the decode kernels (`q4_k_sub`, `q6_k_sub`), the scales of a
super-block once (`q4_k_scales_mins`), and each 8-value chunk is stored
with one `vst1_s8`. Same values: PPL 16.7330 (`ppl.txt`), unchanged; 92/92
tests.

**Result** (cycles, `cycles.txt`, two rounds, reps 5):

| | pp64 | pp512 | timed pp64 tok/s |
| --- | --- | --- | --- |
| before (`66aed73`) | 15.3 / 15.4 G | 77.5 / 75.6 G | 457 / 418 |
| after | 11.1 / 10.9 G (−28 %) | 73.3 / 72.3 G (−5 %) | 651 / 541 |

**RoPE angles once per call, not per head** (`raw/rope-once/`): `rope_*`
recomputed `powf`, `sin` and `cos` for every head, though they depend only
on the position and the pair index. Now `rotations()` computes the
(sin, cos) table once per call and every head uses it: the same floats
(greedy output byte-identical in Q4_0 and Q4_K_M). Profile of Q4_0 pp64:
`powf` + `sincosf` + `rope_halves` 3.3 % → `powf` + `sincosf` + `rope_halves`
+ `bias_rope` 1.1 %. The cycle A/B (`cycles.txt`) is within its noise, as a
2 % change would be.

## Prefill: q/k/v and gate/up as one job each, too (2026-10-29; `raw/prefill-fused/`)

`linear_many_batch_on` is `linear_many_on` for `t` tokens: the shared input
is quantized and interleaved into pairs once instead of three (two) times,
and the threads take row blocks from all the matrices, so k and v (128 rows
each, 8 blocks for 6 threads) no longer leave threads idle at their end.
Same values: PPL Q4_0 19.7470, Q4_K_M 16.7330 (`ppl-*.txt`), 92/92 tests.

Cycles (`cycles.txt`, two rounds, reps 5): pp64 Q4_0 8.20 / 8.16 G → 7.85 /
7.85 G (−4 %), Q4_K_M 10.60 / 10.82 → 10.26 / 10.27 G (−4 %); pp512 −1 %.
A timed Q4_0 pp512 A/B with the order alternated (`pp512-order.txt`)
showed again how much the clock cap matters: every run with the prime
cores capped at 1.02 GHz did 455–491 tok/s, every uncapped one 580–588,
base and new alike.

## SwiGLU with a NEON `exp` (2026-10-29; `raw/swiglu-neon/`)

The polynomial `exp` written for attention's softmax (`85d1a12`, reverted
there for no gain) now computes SwiGLU's sigmoid, four lanes at a time:
`sigmoid(v)` needs only `e^(−|v|)` (`1/(1+e)` for v ≥ 0, `e/(1+e)` below),
which is in that `exp`'s range. SwiGLU runs over 4864 values per token and
layer, so scalar `expf` was 2–4 % of a profile.

Values change in the last bits (the numpy SwiGLU test still passes at
1e-6), and so does PPL, by 0.1–0.2 % (`ppl-*.txt`): Q4_K_M 16.7330 →
16.7531, Q4_0 19.7470 → 19.7110, Q8_0 16.3494 → 16.3771; llama.cpp's own
values are 16.7657 ± 0.688 (Q4_K_M), and these shifts are of the size
that any change of rounding gives after Q8_0 activation quantization.

Cycles of pp64 + tg64 (`cycles.txt`, two rounds): Q4_0 33.3 / 33.3 G →
30.7 / 31.6 G (−5 to −8 %), Q4_K_M 49.5 / 49.3 G → 47.9 / 47.1 G (−3 to
−4 %).

**NEON softmax in attention, second try** (`raw/softmax-neon-2/`): the same
vector softmax as `85d1a12` (which showed nothing on 2026-10-28, before
attention kept one head's vectors in registers and when timings were
noisier). Now, by cycles (Q4_0, two rounds): pp512 19.59 / 19.55 G →
18.66 / 18.83 G (−4 to −5 %); 64 decode steps at depth 1024 (plus the
untimed 1024-token prefill) 59.0 / 58.9 G → 53.7 / 52.2 G (−9 to −11 %),
timed tg64 @ d1024 66–75 → 99–128 tok/s. PPL Q4_0 19.7438, Q4_K_M 16.7599,
Q8_0 16.3546 (`ppl-*.txt`); the numpy attention tests pass.

## Flash-decoding by K/V head (2026-10-29; `raw/flash-decode/`)

Decode attention handed out whole query heads: 14 for 6 threads (three
threads did 3 heads, three did 2), and each of the 7 heads that share a
K/V head read its keys and values again. The earlier split across
positions (`bdbf963`, reverted) was per query head, so it kept the
rereads.

**Change**: past 64 positions (head size 64 or 48), decode's items are
(K/V head, chunk of 64 positions): 2 × 8 = 16 equal items at d512. Each
runs `attention_group_part`, the grouped kernel (`group_fixed`, one key
and value row loaded for all 7 heads) with the weights left unnormalised
and each head's (max, Σ e^(s − max)) returned; the caller merges a head's
parts in context order, `out = Σ e^(m_i − M)·acc_i / Σ e^(m_i − M)·s_i`.
The chunks depend only on the position, so one thread and six give the
same bits. Prefill on one thread goes token by token and keeps the
unchunked path (`attention_whole`), to stay bit-identical to the grouped
prefill on many threads (the long-context llama2.c test checks both).

**Correctness**: new test `attention_parts_merge_to_numpy` (chunks of 1, 7
and 64 positions merged, against numpy, up to 1024 positions); the
long-context llama2.c test (decode over 256 positions within 4.4e-6 of
llama2.c); 93/93 tests on the phone. PPL is unchanged (it runs prefill).

**Result** (Qwen2.5-0.5B Q4_0, cores 2–7, cycles of 64 decode steps =
tg65 − tg1 at each depth, two rounds, `cycles.txt`):

| depth | before (`f1b3b66`) | after | timed tok/s before → after |
| --- | --- | --- | --- |
| 256 | 4.76 / 4.77 G | 4.49 / 4.56 G (−5 %) | 164–167 → 175 |
| 512 | 5.34 / 5.44 G | 4.81 / 4.75 G (−11 %) | 147–152 → 163–167 |
| 1024 | 6.24 / 6.14 G | 5.11 / 4.99 G (−18 %) | 129–130 → 145–150 |

Against llama.cpp at depth (`flash-decode/vs-llamacpp/`, two rounds; every
run, ours and llama.cpp's, hit the prime-core cap of 1.02 GHz at some
point, so these are hot-phone numbers), tg64 tok/s:

| format | d0 ours / llama.cpp | d512 | d1024 |
| --- | --- | --- | --- |
| Q4_0 | 180–187 / 132–159 | 122–167 / 83–129 | 112 / 43–85 |
| Q8_0 | 110–111 / 93 | 92–100 / 68–72 | 75–83 / 50–64 |
| Q4_K_M | 89–117 / 83–85 | 55–108 / 58–65 | 50–73 / 34–50 |

Ours is ahead in 16 of the 18 pairs; with the caps moving mid-run the
spreads are wide, and the one Q4_K_M round that lost (55 at d512) was the
last, hottest run. Host tests 90/90.

## Prompt lookup decoding (plan section 13, "speculative decoding"; 2026-10-29; `raw/lookup/`)

Decode is now close to memory bandwidth (Q4_0 ~180 tok/s × 278 MB ≈ 50 GB/s),
so a faster kernel cannot give much more; reading the weights once for
several tokens can. Speculative decoding does that with a draft model, but
Qwen2.5-0.5B is the smallest of its family, so this is the model-free
variant: **prompt lookup**. After each certain token, up to K draft tokens
are copied from the most recent earlier place where the last N tokens
occurred (prompt or output), and the certain token plus the drafts run as
one batched pass (`forward_batch_logits`). Row r predicts the token after
input r, so drafts are accepted while each equals the argmax of the row
before it, and the first disagreement's argmax is the next certain token.
Cache entries of rejected drafts are simply overwritten later.
`Session::generate_lookup`, CLI `--lookup K [--ngram N]`, greedy only.

**Correctness**: on stories15M (f32) it reproduces llama2.c's greedy text
byte for byte (new test `lookup_decoding_gives_llama2c_greedy_text`).

**Speed** (Qwen2.5-0.5B Q4_0, cores 2–7, `--lookup 4`, n-gram 2; outputs in
`out-*.txt`):

| prompt | plain greedy | lookup | drafts accepted | same text |
| --- | --- | --- | --- | --- |
| "add comments to this function" (the model copies it back) | 170.4 tok/s | 256.1 (+50 %) | 61 / 84 | yes |
| "why is the sky blue" (the 0.5B model falls into a loop) | 171.4 | 243.9 (+42 %) | 216 / 296 | yes |
| short story, 194 tokens, two runs | 172.8 / 153.3 | 188.1 / 176.6 (+9 / +15 %) | 45 / 81 | diverges after ~80 tokens |
| same, n-gram 3 | | 166.3 / 165.1 | 5 / 44 | diverges |

The sky answer's gain is the model repeating one sentence, which lookup
accepts for free; the code case is the honest win (copying). On fresh prose
the gain is small, and a longer n-gram finds too few drafts to pay for its
passes.

**Why the story text diverges**: the verification pass computes logits by
the batched path, and on quantized models the two paths differ more than
one would think. `examples/logit_gap.rs` re-runs 100 greedy tokens in
batches (`logit-gap.txt`): on SmolLM2-135M f32 the paths agree within
5e-5, but on Q4_0 and Q8_0 by up to 0.4–0.65 in a logit, already for a
single batched token at the first position. The activations are quantized
to int8 before every matmul, so a float difference of 1e-7 (another
summation order) now and then flips one int8 value, and 24 layers carry
that on. The greedy margins (`margins-plain-greedy.txt`) go down to 0.02,
so a fork at about token 80 is expected. Lookup decoding therefore gives
"greedy under the batched path's numbers"; the same holds for flash-
decoding against the old whole-context decode (batch 1 column: 0 before,
up to 0.55 after), and for llama.cpp's own batched vs single-token paths.

## Sustained decode and energy, after the night's work (2026-10-29; `raw/sustained-2026-10-29/`)

The Phase 7 test again (Qwen2.5-0.5B Q4_0, cores 2–7, 64-token decode
bursts back to back for 300 s, on battery at 39 %, screen kept awake by a
key event every 50 s, start 41.5 °C; `bench_suite --sustain 300 --monitor`):

| | Phase 7 (2026-10-22) | now (`34c7597`) |
| --- | --- | --- |
| average | 85.7 tok/s | **94.8 tok/s** (28,480 tokens in 301 s) |
| 10-second windows | 112 first 40 s, then 60–89 | 121 first, then 72–137 |
| max CPU temperature | 84.5 °C | 80.3 °C |
| power (charge counter) | 3.49 W | 2.65 W (sampler: 3.16 W) |
| energy per token | 0.041 J | **0.028 J** (0.033 J by the sampler's power) |

The burst at the start (172.7 tok/s for tg64 from cold) falls to ~95 under
sustained load as the clock caps come down; per token the phone now spends
about a third less energy, since each token takes less time at a similar
power. The two power estimates differ (the charge counter covers a 277 s
window, the sampler the whole 300 s), so the energy figure is ±15 %.

**Sampling, and cheaper verification passes** (`raw/lookup/sampled.txt`,
`small-batches*.txt`, `after-narrow-tiles.txt`):

- `Sampler::sample_draft` checks a draft token by speculative sampling: with
  p the distribution `sample` would draw from (temperature, top-k/top-p,
  renormalised), the draft d is kept with probability p(d), and otherwise a
  token is drawn from p without d. The result is distributed exactly as p
  (test `draft_sampling_keeps_the_distribution`: 200,000 draws against the
  closed-form probabilities, within 0.005). Greedy is the temperature-0
  case of the same call.
- With sampling, the first measurement lost on prose (−8 to −13 %): a
  5-token batched pass cost ~2.3 decode steps. Timing small batches
  showed why (`small-batches.txt`): pp2 … pp8 all took 11–13 ms against a
  5.5 ms decode step, because the `smmla` tiles always covered 4 token
  pairs, so two tokens paid for eight. Tiles are now generic in their
  width (`Tiles<P>`, `mmla4::<F, P>`), and batches of 1–3 pairs use
  P = 1–3: pp2 11.3 → 5.8 ms, pp4 12.1 → 7.8 ms, pp5 12.4 → 9.6 ms. Each
  output's value does not depend on its tile partners, so PPL is unchanged
  (19.7438) and 94/94 tests pass.

After both (Qwen2.5-0.5B Q4_0, cores 2–7, two rounds, tok/s):

| prompt, sampler | plain | `--lookup 3` | `--lookup 4` |
| --- | --- | --- | --- |
| code (copying), greedy | 162–169 | 264–293 | 258–288 |
| code, t 0.7 top-p 0.9 | 156–163 | 149–245 | 195–240 |
| story, greedy | 146–174 | 166–172 | 164–177 |
| story, t 0.7 top-p 0.9 | 147–156 | 142–152 | 137–146 |

Lookup decoding pays when the output repeats the input (+60–75 % greedy,
+25–50 % sampled for the code task) and is about neutral on fresh prose
(within −10 % to +15 %, inside the run-to-run spread).
