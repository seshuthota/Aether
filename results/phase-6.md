# Phase 6 — quantization (Q8_0, Q4_0)

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
- Gap to llama.cpp: ~2.4–3× (was ~3.4–4× in the same session).

Not done yet (paused 2026-10-24 so the learning track can catch up):
interleave the activations once per call instead of once per row pair, a
4 rows × 4 tokens tile, then — if the profile still shows interleaving —
repack the weights at load as llama.cpp does.
