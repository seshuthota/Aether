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

Not done yet: `smmla` for Q4_K / Q6_K; the remaining Q4_0 / Q8_0 prefill gap
(1.15–1.5×).
