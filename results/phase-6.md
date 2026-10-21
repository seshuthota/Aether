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
