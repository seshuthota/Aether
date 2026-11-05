# Gemma 4 E2B on the CPU

Plan: `plans/gemma4.md`. Branch `gemma4` (worktree `../Aether-next`). The
phone was not connected (adb lost it over Wi-Fi) for this session, so
every number here is from the **PC** (x86_64, 16 threads, 23 GB) unless
marked otherwise; G7 (the phone run) is still to do.

## G0. References and files

### Versions

| What | Version / revision |
| --- | --- |
| llama.cpp | commit `d02b4c0d6` (PC build, Release) |
| HF transformers | 5.20.2 |
| HF tokenizers | 0.23.3 |
| torch | 2.14.1+cpu |
| `google/gemma-4-E2B-it` | `4adf15748de9e1a0097b6274cee9c15836c41c37` |

### Files

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `testdata/gguf/gemma-4-E2B-it-Q4_0.gguf` (ggml-org) | 2,841,481,184 | `8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52` |
| `testdata/gguf/gemma-4-E2B-it-Q4_K_M.gguf` (unsloth) | 3,106,738,272 | `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8` |
| `hostdata/gguf/gemma-4-E2B-it-BF16.gguf` (ggml-org) | 9,311,305,152 | `246c5fb19e64b941b531e86639f070bc67b00f20e9ec8556a6c02653321bc65d` |
| `hostdata/gemma-4-E2B-it/model.safetensors` | 10,246,621,918 | `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550` |
| `hostdata/gemma-4-E2B-it/tokenizer.json` | 32,155,733 | `ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2` |

The BF16 GGUF is kept in `hostdata/` (not `testdata/`), so the phone
runner does not copy 9.3 GB to the phone.

### The open question: the embedding scale

HF's modeling file rounds `√1536` to bf16 (39.19 → 39.25) **for a bf16
model**. The reference here runs the model in f32, where `embed_tokens`
multiplies by the f32 `√1536`, as llama.cpp does. The engine uses the f32
value; with it every layer matches HF within 1.0e-5 (G4), so in f32 the
two references agree and the question does not arise.

### How the HF reference fits in 23 GB

`tools/hf_dump_gemma4.py` builds transformers' own `Gemma4TextModel` in
f32 without the 262144 × 8960 per-layer table (`vocab_size_per_layer_input
= 1`), copies the checkpoint into it one tensor at a time (widening bf16
to f32 is exact), and passes `inputs_embeds` and the raw per-layer inputs
(the table rows of the prompt's tokens × √256, read with `safe_open`
slices). Peak 15.3 GB. The first version (whole state dict, then
`load_state_dict`) was killed by the OOM killer at 18 GB, without a message.

### llama.cpp baselines (PC, wikitext-2 test, 20 × 512, 5 threads)

| File | PPL | KL vs BF16 | same top token |
| --- | --- | --- | --- |
| BF16 | 184.2324 ± 13.98465 | | |
| Q4_K_M | 185.1681 ± 13.87906 | 0.178324 ± 0.005288 | 84.000 ± 0.513 % |
| Q4_0 | 57.4593 ± 3.33030 | 0.344301 ± 0.007424 | 77.922 ± 0.581 % |

Raw logs: `results/phase-gemma4/raw/llama-*.txt`.

- **The Q4_0 file is not a rounding of the BF16 weights.** Its PPL is a
  third of BF16's, and its KL against BF16 is twice Q4_K_M's. A rounded
  copy of a model does not beat the model by 3×; a different checkpoint
  (Google's quantization-aware Q4_0, as for Gemma 3) does. Q4_K_M
  (unsloth, from the BF16 weights) sits at 1.005 × BF16, as K-quants do.
  The general metadata of both GGUFs only names `google/gemma-4-E2B-it` as
  the base, so this is an inference from the numbers, not a stated fact.
- The PPL is high for all three (an instruction-tuned model on raw
  Wikipedia text without its chat template); only the comparisons matter.
- In `--kl-divergence` mode llama.cpp prints `PPL(base) = 161.29`, not the
  184.23 of the BF16 run itself on the same 20 chunks. The KL base file
  stores each position's log-probabilities in 16 bits over a range, which
  may lift the rarest target tokens (not checked further).

## G1. Config, loader, Q5_K rows

- Both files load with the shapes of the plan's section 2, checked per
  layer by `gguf::tests::gemma4_loads`: head size 512 on layers 4, 9, …,
  34 and 256 elsewhere (window 512), FFN 6144 for layers 0–14 and 12288
  after, KV source 13 (sliding) / 14 (full) for layers 15–34, RoPE 64 pairs
  at θ 1e6 (full; the other 192 factors in `rope_freqs` are 1e30) and 128
  at θ 1e4 (sliding), no K/V weights used for the shared layers (the
  Q4_K_M file has them; they are ignored, as in llama.cpp).
- The per-layer table stays mapped in its own format (Q4_0 in ggml-org's
  file, **Q5_K** in unsloth's) and one row per token is dequantized;
  repacking leaves it as it was (tested).
- Q5_K rows equal `gguf-py`'s dequantization exactly
  (`quant::tests::q5_k_matches_gguf`).

## G2. Tokenizer

`tokenizer/spm_bpe.rs`: special tokens split first, spaces to `▁`, the
whole text as one word, byte fallback for characters without a token, BPE
by merge rank. 440 of 440 strings (the shared corpus, chat markup,
newline runs, byte fallback, emoji, CJK) give HF `tokenizers`' ids
exactly; `llama-tokenize` agrees on spot checks.

## G3. Kernels

numpy goldens (`tools/gen_kernels.py`) for GELU-tanh × up, RMSNorm without
a gain (V), the soft-cap, attention at head sizes 256 and 512 (causal, one
KV head for 8 query heads) and the sliding window at 512. The old cases
are unchanged (md5). The scalar kernels pass on the PC.

NEON GELU and soft-cap (`x · sigmoid(2z)` and `(1 − e)/(1 + e)` with the
existing polynomial `exp`) are written but **not committed**: their golden
test runs only on the phone.

## G4. The forward pass

`gguf::tests::gemma4_matches_transformers`, BF16 GGUF against
transformers in f32 (bound 1e-3, LFM3's):

- the per-layer inputs and every layer's output on a 30-token prompt:
  worst relative error **1.0e-5**;
- logits by the batched pass and the one-token pass at all 30 positions;
- logits of a 600-token prompt at positions 0, 1, 510–514, 598, 599 (past
  the 512 window, through the batched pass in chunks);
- a 16-token greedy continuation: " a brave knight named Arthur. Arthur
  was known for his courage and his chivalry".

Run time 13 min, peak RSS 13.5 GB.

**Q4_0 greedy against `llama-completion --temp 0`** (128 tokens, PC,
`raw/greedy-Q4_0-pc.txt`): `code` and `fact` identical (both stop at the
same `<eos>`). `story` splits after 52 tokens and `long` (604-token
prompt, decoding past the 512 window) after 34. At each split our two
candidates are within a few hundredths of a logit (our logits on
llama.cpp's text):

| prompt | position | ours | llama.cpp's | gap |
| --- | --- | --- | --- | --- |
| story | 68 | " sound" 16.755 | " gentle" 16.736 | 0.019 |
| long | 638 | " and" 25.689 | " at" 25.679 | 0.010 |

Different summation orders in the Q4_0 dot products flip ties this close.

## G5. Accuracy on the real files

### Perplexity (PC, wikitext-2 test, 20 × 512)

| File | aether | llama.cpp | aether / llama.cpp | aether peak RSS |
| --- | --- | --- | --- | --- |
| Q4_0 | 57.4248 ± 3.32703 | 57.4593 ± 3.33030 | 0.99940 | 2.69 GB |
| Q4_K_M | 186.1763 ± 13.98238 | 185.1681 ± 13.87906 | 1.00544 | 2.67 GB |

Per chunk (`raw/aether-ppl-*-pc.err`, `raw/llama-ppl-*.txt`) the two go
back and forth: Q4_K_M chunk 1 is 52.82 against 53.02 (ours lower),
chunk 19 191.34 against 189.93 (ours higher). Both differences are far
inside the error bars. At a PPL near 200 the mean is carried by many
low-probability tokens, whose log-probabilities move most with rounding;
the Q4_0 file (PPL 57) agrees to 0.06 %, like LFM3's A5 (24.85 / 24.83).
The KL against our BF16 run (below) is the sharper check.

Peak RSS stays below the file size (2.84 / 3.11 GB). The per-layer table
(1.32 / 1.61 GB) is mapped and only the rows of tokens that occur are
read; the 295,216-token text touches some of them.

### KL against BF16

First 8 chunks of 512 (2,040 scored tokens). The 20-chunk runs were
OOM-killed when the test suite ran beside them, and a rerun did not fit the
night. llama.cpp's figures are its running values after chunk 8 in
`raw/llama-kl-*.txt`. Ours: `--kl --base` the BF16 GGUF, widened to f32
(`raw/aether-kl-*-pc.txt`).

| | aether | llama.cpp (after chunk 8) |
| --- | --- | --- |
| BF16 PPL | 183.4660 | 183.1428 (its own BF16 run) |
| Q4_K_M mean KL | 0.181374 | 0.18724 ± 0.00854 |
| Q4_K_M same top token | 82.990 % | 82.843 ± 0.835 % |
| Q4_0 mean KL | 0.345793 | 0.34587 ± 0.01319 |
| Q4_0 same top token | 77.745 % | 77.941 ± 0.918 % |

Every pair is inside llama.cpp's error bar. The Q4_0 KL equals llama.cpp's
to 1e-4 and the BF16 PPL to 0.2 %. Q4_K_M is the only measurable
difference: our KL is 0.7 error bars lower.

Each run took 70–76 min and peaked at 16 GB (two models in memory, the
base in f32).

### KL against llama.cpp's own base file (S2)

`aether --ppl FILE --kl-base FILE` reads the log-probabilities llama.cpp
saved with `--kl-divergence-base` from its BF16 run (`eval::BaseFile`), so
both engines are scored against the same numbers. Our tokens are checked
against the file's first: all 10,240 equal llama.cpp's. Q4_K_M, all
20 chunks (`raw/aether-klbase-Q4_K_M-pc.txt`, cores 0–15, 43 min, peak
RSS 2.54 GB):

| | aether | llama.cpp (`raw/llama-kl-Q4_K_M.txt`) |
| --- | --- | --- |
| PPL(base) from the file | 161.2919 | 161.291854 |
| PPL(Q4_K_M) | 186.1763 | 185.1681 ± 13.88 |
| Mean KL | 0.175124 | 0.178324 ± 0.005288 |
| Same top token | 83.980 % | 84.000 ± 0.513 % |

The base PPL equals llama.cpp's, so the file is read as llama.cpp reads
it. The file keeps only log-probabilities within 16 of the top one and
stores the rest at that floor, which is why its PPL (161.29) is below the
BF16 run's own 184.23. Our KL is 0.6 error bars below llama.cpp's and the
same top token agrees to 0.02 %. The PPL is our plain Q4_K_M PPL, 0.5 %
above llama.cpp's (as in the table above).

## G6. Chat

- `chat::gemma4` renders the four prompts HF `apply_chat_template` gives
  (with and without a system turn, with and without thinking).
- Stops on `<eos>` (1) and `<turn|>` (106).
- Reasoning follows each template's default: LFM3 on, Gemma 4 off
  (HF's `enable_thinking=False`). `--think` turns it on. `--hide-think`
  hides Gemma's `<|channel>…<channel|>` span.
- On Q4_0: "What is 2+2? Answer in one sentence." → "Two plus two equals
  four." With `--think --hide-think`, "What is 17*3?" → "17 * 3 = 51".
- The C ABI uses the same default; the app lists any `.gguf`, so Gemma 4
  needs no app change. Not yet tried in the app (no phone).

## Existing models unchanged

`main` (c85ff12) and `gemma4` built side by side on the PC: PPL over 4 ×
512 wikitext chunks, and the md5 of 32 greedy positions of "Once upon a
time" (batched prompt pass, then decode):

| Model | PPL, main | PPL, gemma4 | greedy text |
| --- | --- | --- | --- |
| smollm2-135m-q4_0 | 20.8966 ± 2.00131 | 20.8966 ± 2.00131 | same |
| smollm2-135m-q4_k_m | 16.2675 ± 1.52455 | 16.2675 ± 1.52455 | same |
| smollm2-135m-q8_0 | 15.5773 ± 1.46626 | 15.5773 ± 1.46626 | same |
| qwen2.5-0.5b-q4_0 | 15.8379 ± 1.51270 | 15.8379 ± 1.51270 | same |
| qwen2.5-0.5b-q4_k_m | 13.4590 ± 1.24034 | 13.4590 ± 1.24034 | same |
| qwen2.5-0.5b-q8_0 | 13.3717 ± 1.23962 | 13.3717 ± 1.23962 | same |
| LFM3-1.7B-QAD-Q4_0 | 16.4273 ± 1.61694 | 16.4273 ± 1.61694 | same |
| LFM3-1.7B-Q4_K_M | 15.2087 ± 1.47651 | 15.2087 ± 1.47651 | same |

The phone (NEON / i8mm paths) has not been compared yet.

## GPU (PC, plans/gemma4-close.md B1–B5)

`model::gpu::GemmaGpu`, separate from the uniform `GpuModel` (which is
unchanged). Run on the PC's Intel iGPU (RPL-P, Vulkan); the phone is still
owed (B6, B7). Q4_0, against llama.cpp's BF16 base file (S2),
`examples/vk_gemma.rs`:

| Chunk | GPU decode PPL / KL | GPU prompt pass PPL / KL | llama.cpp CPU PPL / KL |
| --- | --- | --- | --- |
| 1 | 24.4536 / 0.33855 | 24.4536 / 0.33855 | 24.5617 / 0.33477 |
| 2 | 41.5999 / 0.32370 | | 41.7681 / 0.32419 |
| 3 | 36.2937 / 0.35110 | | 36.4231 / 0.35142 |
| 4 | 43.6075 / 0.34282 | | 43.8254 / 0.34330 |

Same top token after 4 chunks: 79.902 %, as llama.cpp's. Our CPU path
gives 43.6083 after 4 chunks. The greedy continuation of "Once upon a
time, …" equals llama.cpp's and the CPU's. `aether --gpu --chat` answers
"What is 2+2? Answer in one sentence." with "Two plus two equals four.",
as the CPU does. (Raw: `raw/gpu-pc-Q4_0-{decode,batch}.txt`.)

Qwen2.5-0.5B Q4_0 through `vk_ppl` (4 chunks) is unchanged: batch
15.8560, decode 15.8023.

**The watchdog.** The first prompt pass was recorded as one command
buffer. On this iGPU it runs for tens of seconds, and i915 stopped it
partway through layer 25's gate/up product without an error (the kernel
log: "Fence expiration time out"). The rows past 11,264 kept layer 24's
values. Found by comparing the pass with decode layer by layer. The pass is
now one submission per layer, plus one for the classifier.

**Speed on this iGPU** (not the phone): decode 8–10 tok/s; the prompt
pass ~31 s for a 21-token prompt. The prompt products use `qx_gemm.comp`
(f32), which runs at ~11 GFLOP/s here (gate/up ≈ 420 ms a layer for a
64-token tile). The uniform path's fast prompt products read a second copy
of every matrix (f16 or int8 layout); for Gemma that is +1.5 GB of device
memory, beyond B6's budget. Whether it is worth it is B7's question, on
the Adreno.

## DSP

Refuses Gemma 4 with a message (plan part C).

## Not done

- **G7, the phone.** adb did not see the phone all session. To do: the
  kernel goldens on NEON (including the uncommitted GELU / soft-cap and
  the head-256/512 attention paths), the Gemma tests, Q4_0 PPL once,
  `bench_suite` against `llama-bench` cooled to 45 °C, peak RSS, the app.
- **Merge to main: not done.** The CPU code that runs on the phone (NEON
  and i8mm matmuls and attention at head sizes 256 and 512) has not run
  there yet.
- G8 optimisation and the G3 tiled attention variant for 256/512.
