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

**Past the window** (`examples/gemma_window.rs`, 640 wikitext tokens,
`raw/gpu-pc-Q4_0-window.txt`): logits at 12 positions from 0 to 639,
around the sliding window's 512. GPU decode against the GPU prompt pass:
1.7–3.5e-6 of the largest logit everywhere. GPU decode against the CPU
path: 1.1–2.7e-2, flat across 512 (the CPU rounds activations to 8 bits
before its products, the GPU does not), with the same top token at all 12
positions. A window off by one, or a shared layer reading the wrong
cache, would show as a jump from 512 on.

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

### Q4_K_M on the GPU, and S4 (F32 per-layer matrices in Q8_0)

No GPU shader reads an F32 matrix, so `GemmaGpu` loads Q4_K_M's 70 F32
per-layer matrices (`inp_gate`, `proj`) as Q8_0 (`Weights::ple_to_q8_0`,
llama.cpp's Q8_0 quantizer). The plan's gate runs the CPU path with that
conversion (`aether --ple-q8`). All rows are 20 chunks against llama.cpp's
BF16 base file:

| | PPL | Mean KL | Same top |
| --- | --- | --- | --- |
| CPU, F32 matrices (S2) | 186.1763 ± 13.98 | 0.175124 | 83.980 % |
| CPU, Q8_0 matrices (`--ple-q8`) | 187.7939 ± 14.10 | 0.181863 | 83.725 % |
| **GPU** (Q8_0 matrices) | 188.6380 ± 14.18 | **0.163587** | 84.490 % |
| llama.cpp CPU | 185.1681 ± 13.88 | 0.178324 ± 0.005288 | 84.000 % |

The gate as written fails narrowly on the CPU: KL rises by 0.0067, 1.3×
the KL error bar (PPL stays well inside its error bar). That CPU run
measures more than the conversion, though. Our CPU Q8_0 product rounds
the activations to 8 bits too (as llama.cpp's does), while the GPU keeps
them in f32. The GPU, the configuration that ships, has a KL below the
CPU's F32 run and below llama.cpp's. Its PPL is 1.3 % above the CPU's,
well inside the error bar. So the conversion stays. If KL ever matters at
this level, the alternative is an f16 product for these matrices (plan
risk 3). (Raw: `raw/aether-klbase-Q4_K_M-pleq8-pc.txt`,
`raw/gpu-pc-Q4_K_M-decode.txt`.)

The GPU's greedy continuation on Q4_K_M: " a young girl named Lily. Lily
had eyes the color of the deep ocean and a spirit as bright as the summer
sun". Decode is 6–8 tok/s on the iGPU; upload takes 8.2 s (the K-quant
repacking).

## DSP

Refused Gemma 4 until 2026-11-06 (C1–C7 on the phone: "DSP on the phone"
below). Done without the phone:

**C3, the element-wise steps on HVX** (`dsp/gemma_hvx.h`): GELU-tanh × up
(as g · σ(2z), the CPU's NEON identity, on the decode step's `vexp` /
`vrecip`, now in `dsp/vmath_hvx.h`), soft-cap, RMSNorm with and without
gain, the per-head q/k/v norm with RoPE on the first pairs and q × √hs,
and the post-norm add with a scale. On `hexagon-sim` (v79) against the
numpy goldens the CPU and GPU kernels pass
(`sim_gemma_elementwise_matches_numpy`):

| Kernel | Cases | Worst error / largest value | Bound |
| --- | --- | --- | --- |
| GELU × up | 21 | 1.64e-7 | 1e-6 |
| soft-cap 30 | 21 | 1.91e-7 | 1e-6 |
| RMSNorm, no gain (hs 256, 512) | 2 | 7.49e-8 | 1e-5 |
| q/k norm + RoPE + q√hs, v norm | 6 (2 head sizes × q, k, v) | 1.17e-7 | 1e-5 |
| post-norm add × scale | 2 | 1.70e-7 | 1e-5 |

**C0's probe** is ready for the phone: `dsp_maps_gemma_sized_buffers`
(ignored by default) maps 1.6 GB + 192 MB of shared buffers at once and has
the DSP read every word. C1, C2, C4 and C5 wait for its answer, which
decides how the weights are laid out.

## Phone (iQOO 13, Snapdragon 8 Elite, 2026-11-05)

Steps 1–4 of `plans/gemma4-phone-checks.md`, and step 5 except the greedy
text. The phone was on Wi-Fi adb; tests ran one at a time
(`--test-threads=1`). 157 tests in the build with `vulkan,hexagon`; the
first run needed a runner fix (this adb recreates symlinks instead of
following them, and `testdata/` mixes symlinks with real directories).

| Check | Result |
| --- | --- |
| Kernel goldens (`--lib kernels::`) | 18/18, `gemma_elementwise_matches_numpy` included: NEON GELU-tanh × up, NEON soft-cap, plain RMSNorm |
| `gemma4` filter | 5/5: `gemma4_loads`; `gemma4_matches_hf_tokenizers_on_corpus` (440 strings); `gemma4_template_matches_hf`; `c_abi_gpu_gemma4` — Q4_0 on the **Adreno** through the C ABI, answers "Two plus two equals four." and stops; `gemma4_matches_transformers` skips itself (hostdata is not pushed) |
| `spm_bpe chat::` | 5/5, the three chat templates and `think_filter_hides_split_spans` included |
| The whole suite (`--features vulkan,hexagon`) | **157 passed, 1 ignored, 0 failed**, 258 s |
| C0 (`dsp_maps_gemma_sized_buffers`, `--ignored`) | **Passes**: 1.6 GB + 192 MB mapped at once and read at 36.6 / 36.7 GB/s, checksum equal to the CPU's → C1, C2, C4, C5 use one weights buffer, as Qwen does |
| DSP refuses Gemma | still returns its message, no crash (in the suite) |

### CPU accuracy (wikitext-2 test, 20 × 512, cores 2–7)

| File | aether, phone | aether, PC | llama.cpp, PC |
| --- | --- | --- | --- |
| Q4_0 | **57.5076** ± 3.334 | 57.4248 ± 3.327 | 57.4593 ± 3.330 |
| Q4_K_M | **185.6800** ± 13.94 | 186.1763 ± 13.98 | 185.1681 ± 13.88 |

Peak RSS (VmHWM): **2.52 GB** for both files, below the files' 2.84 /
3.11 GB (the PC's 2.69 / 2.67 GB). Scored in 295 s and 235 s; an earlier
Q4_0 run without the RSS sampler took 172 s.

The Q4_0 gap to the PC (+0.14 %) splits, on the phone, in the same 20 × 512
run:
- **main's build** (GELU and soft-cap through the scalar `tanh`):
  **57.4775** — +0.09 % from the existing NEON / i8mm kernels against x86;
- **the NEON GELU / soft-cap on top**: +0.030 more, +0.05 %.

llama.cpp's PC run-to-run spread on Q4_0 was 0.06 %; whether the 0.09 %
part is inside its phone spread is what S0 (step 7) answers with its own
CPU run on this phone. Q4_K_M, whose mean is carried by rare tokens, is
+0.28 % from llama.cpp's PC number and −0.27 % from ours, well inside the
±13.9 error bar.

### Existing models, main against `gemma4-devices` (phone, 4 × 512, cores 2–7)

| File | main | `gemma4-devices` |
| --- | --- | --- |
| Qwen2.5-0.5B Q4_0 | 15.8323 | **15.8323** |
| LFM3-1.7B Q4_K_M | 15.1754 | **15.1754** |

To the digit. Neither model runs the changed kernels (GELU is Gemma's;
LFM3 and Qwen use SwiGLU), so this is the regression check and it holds.
Peak RSS 0.68 GB (Qwen) and 1.64 GB (LFM3), equal in both builds to
1.1 MB.

### GPU (Adreno 830, B6)

`examples/vk_gemma.rs` against the same BF16 base file as the PC:

| Run | phone | PC (iGPU) |
| --- | --- | --- |
| Q4_0 decode, 4 chunks | **43.6074 / KL 0.342819** | 43.6075 / 0.34282 |
| Q4_0 prompt pass, 1 chunk | **24.4536 / KL 0.338553** | 24.4536 / 0.33855 |
| Q4_K_M decode, 4 chunks | **138.0870 / KL 0.165109** | 138.0871 / 0.16511 |

Same top token at every chunk (79.902 % / 84.706 %; llama.cpp's CPU
79.902 %). The prompt pass equals the decode step (24.4536 both), as on
the PC. The phone is 2× the PC's iGPU for decode (134 s vs 274 s for
4 chunks), and the prompt pass is 12.7 s per chunk against 271.5 s.
`vk_gemma`'s own 40-position loop: 18.6–20.2 tok/s decode, ≈ 30 tok/s
prompt pass (1.6×). Peak PSS 3.41 GB while decoding, host side included.

Past the window (`gemma_window`, 640 tokens): decode vs prompt pass
1.8–3.6e-6 of the largest logit, GPU vs CPU 9.3e-3–2.6e-2 flat across 512,
the same top token at all 12 positions — the PC's picture, no jump at 512.

### llama.cpp on the phone (S0, partial)

- The two builds already there (`llama-ocl`, `llama-hex`, llama.cpp
  `d00685831`) **cannot load Gemma 4**: after "cannot properly format
  tensor name", `missing tensor 'blk.15.attn_k.weight'`. They predate the
  model, so their Gemma references do not exist; the plan's fallback says
  their accuracy reference is llama.cpp's CPU numbers, stated as such.
- A current llama.cpp (`d02b4c0d6`) cross-built here for Android (static,
  CPU, `armv8.7-a+dotprod+i8mm`) runs it: **PPL 57.4303 ± 3.328**
  (20 × 512, cores 2-7).
- Its greedy text on the phone reproduces the PC's llama.cpp text at the
  same splits (story char 246, long 355); `fact` ends at `<eos>`, `code`
  is identical but for a leading newline that `--no-display-prompt` adds.

### Greedy text (step 5's last piece, Q4_0, phone CPU)

The four prompts the PC's `raw/greedy-Q4_0-pc.txt` was made from are in
`raw/prompts/` now (the PC session kept them only in a scratch file, which
is why this check could not be repeated before).

| Prompt | Ours (phone) vs ours (PC) | Split | Decode-path gap |
| --- | --- | --- | --- |
| story | differs | pos 68: " gentle" vs " sound" | 0.106 (the PC's own gap there: 0.019, the other way) |
| fact | same | — | — |
| code | same | — | — |
| long | differs | pos 629: "The" vs "Mercury" | 0.047 |

At the story split our logits move by 0.03–0.15 between the PC and the
phone (Q4_0 rounds activations to int8, differently per ISA), so both
splits ride a gap smaller than the shift itself: near-ties, as the
checklist requires.

**The Q4_0 PPL gap, read as two ISA shifts.** Ours 57.5076 against
llama.cpp's phone 57.4303 is +0.135 %, past the 0.06 % run-to-run spread
the checklist asks for. Per chunk, neither engine drifts; each sits at a
constant offset from its own PC number, in opposite directions
(`raw/phone/`, the 20-chunk runs):

| | mean phone − PC, per chunk | range of the 20 | aggregate |
| --- | --- | --- | --- |
| ours | **+0.113** | +0.054 … +0.220 | 57.4248 → 57.5076 (+0.14 %) |
| llama.cpp | **−0.110** | −0.232 … −0.002 | 57.4593 → 57.4303 (−0.05 %) |

So the cross-engine gap on the phone is the sum of two same-model ISA
shifts: our x86 and NEON / i8mm paths round differently (int8 activations,
summation order), and so do llama.cpp's AVX-VNNI and NEON ones, with the
opposite sign. The 0.06 % bound was llama.cpp's spread between runs on one
machine, which says nothing about a comparison across ISAs. Within a
machine each engine reproduces itself — our GPU path, which does not round
activations, matches our PC to 1e-4 on the same base file — and main's
build on the phone is 57.4775, i.e. most of our shift is not the NEON
GELU / soft-cap (that adds +0.030). Read the bound as same-ISA.

### Speed (step 8)

CPU, Q4_0, 6 threads on cores 2-7, cooled to <= 45 C on the cpu zones before
every run, two rounds in opposite orders (`raw/phone/*-ours-r1.txt`,
`*-ours-r2.txt`, `*-llama-l1.txt`, `*-llama-l2.txt`):

| tok/s | pp64 | pp128 | pp512 | tg64 @ d0 | tg64 @ d512 |
| --- | --- | --- | --- | --- | --- |
| aether, round 1 | **157.1** | **153.5** | **123.8** | **36.4** | **30.0** |
| aether, round 2 | 139.4 | 136.5 | 110.5 | 35.8 | 28.1 |
| llama.cpp `d02b4c0d6`, round 1 | 87.6 | 92.8 | 84.8 | 29.4 | 24.3 |
| llama.cpp, round 2 | 90.0 | 92.5 | 82.2 | 29.8 | 24.5 |

Ours is 1.3-1.8x llama.cpp's prefill and 1.2x its decode, same session,
same file, both on the CPU (its `-lm none` keeps the weights resident as
`--mmap 0` used to; this llama.cpp renamed the flag). Our round 2 trailed
round 1 by 12 % after two llama.cpp runs and 25 minutes of heat; the
`--monitor` block of each file has the temperatures (the SoC touched
88 C during a round).

GPU, same file and sizes, one cooled run (`*-ours-gpu-r1.txt`):

| tok/s | pp64 | pp128 | pp512 | tg64 @ d0 | tg64 @ d512 |
| --- | --- | --- | --- | --- | --- |
| aether, Adreno 830 | 38.4 | 47.1 | 31.2 | 19.2 | 8.5 |

Peak RSS 3.15 GB; at depth 512 the per-token median is 60 ms but the p95 is
314 ms. **The Adreno is behind the CPU on this model on every size** -
1.9x at decode, 2.6-4x at prefill - so B7's second copy (f16/int8 prompt
matrices, +1.5 GB) would not close that gap and is not worth building now;
llama.cpp's OpenCL build cannot load Gemma 4, so there is no GPU-to-GPU
reference to compare against.

### Profile (step 9)

simpleperf, `cpu-cycles:u`, cores 2-7, 300 decode positions at a 7-token
prompt and a 512-token prefill (`raw/phone/profile-*.txt`; the
`perf.data` and binaries are not committed). Shares of cycles:

| | Q4_0 decode | Q4_K_M decode | Q4_0 prefill | Q4_K_M prefill |
| --- | --- | --- | --- | --- |
| weights GEMV (4-bit) | 51.5 % `gemv4_q4_0x4` | 51.7 % `dot4_q4_k_q8_k_x4` | 53.3 % `mmla4_q4_0x4` | 62.5 % `mmla4_k` |
| weights GEMV (other) | 16.4 % `gemv4_q8_0x4` | 10.7 % `dot4_q6_k_q8_k_x4` | 6.7 % `mmla4_q8_0x4` | 3.0 % `gemm_rows` |
| thread pool wait | **16.3 %** | **16.9 %** | 8.4 % | 6.4 % |
| activation quantizing | 4.4 % `quant_rows` | 3.3 % | 2.8 % `quant_rows_mmla` | 4.9 % |
| attention | 3.7 % `attention_group` | 2.9 % | 13.4 % | 10.5 % |
| f32 GEMV (norms, PLE row) | 2.6 % `gemv` | 8.2 % | - | - |

The order for A3 that this gives: the matrix products are half the time and
are already the mmla4 path; the **thread pool's wait/wake costs ~16 % of
decode** (the same share as a third of the weight streaming) and is the
cheapest first target; attention is only material in prefill (10-13 %); the
int8 activation quantization is 3-5 %. The per-token tok/s printed under
`simpleperf` is not comparable to step 8: sampling runs at ~13 kHz.

### The app (step 10, partial)

`scripts/android-native.sh` built `libaether.so` (2.15 MB, Vulkan + Hexagon)
and the JNI shim; `./gradlew assembleDebug` needed `android/local.properties`
with `sdk.dir` (it is gitignored, so a fresh worktree has none) and the APK
installed with `adb install -r`. Gemma 4 Q4_0 is in the app's files
(`run-as dev.aether.chat`). **The phone is locked**, so the three chat
checks of the checklist (CPU status line, GPU, the DSP's refusal) need the
user to unlock it; the C ABI's Gemma-on-GPU path is already covered by
`c_abi_gpu_gemma4`, which passed in step 3.

### DSP on the phone (C1–C7, 2026-11-06)

`aether-infer` branch `gemma4-dsp` (2db3b3f plus the `split` mode of
`dsp_gemma`), the checklist `plans/gemma4-dsp-phone-checks.md`, raw files
in `raw/dsp/`. 6 DSP threads, on battery.

**Existing models unchanged** (wikitext-2, 20 × 512, against REVIEW 3's
digits, `results/release/README.md`): SmolLM2 Q4_0 decode 27.7234, Qwen
Q4_0 decode 19.7275, Qwen Q4_K_M HMX batch 128 16.7155, each to the digit
with the same greedy text.

**Accuracy against llama.cpp's BF16 base file** (`dsp_gemma`):

| Run | PPL | KL | same top | reference |
| --- | --- | --- | --- | --- |
| Q4_0 decode, attention from DDR, chunk 1 | 24.4392 | 0.331835 | 83.53 % | GPU 24.4536 / 0.33855; llama.cpp CPU 24.5617 / 0.33477 |
| Q4_0 decode, streamed (default), chunk 1 | 24.6625 | 0.332917 | 83.14 % | |
| Q4_0 few-token pass, chunk 1 | 24.6625 | 0.332917 | 83.14 % | the streamed decode, to the digit |
| Q4_0 HMX prompt pass, chunk 1 | 24.4521 | 0.338651 | 82.35 % | GPU's prompt pass 24.4536 / 0.338553 |
| Q4_0 decode from DDR, 4 chunks | 43.7280 | 0.342967 | 79.71 % | GPU 43.6075 / 0.34282; llama.cpp 43.8254 / 0.34330 |
| Q4_K_M decode, streamed, 4 chunks | 139.3306 | 0.171054 | 85.10 % | GPU 138.0871 / 0.16511 |

Every run sits with the GPU and llama.cpp. Streamed and DDR attention
differ only in summation order; through Q8_0 activations that moves chunk
1's PPL by 0.9 % and KL by 0.001 (the PPL error bar is ± 6.4), not the
1e-4 the checklist guessed. Q4_K_M's KL is 0.006 above the GPU's: the DSP
rounds activations to Q8_0 before every product, as our CPU's `--ple-q8`
run does (0.1819 over 20 chunks), while the GPU keeps them in f32.
Q4_K_M's 1754 MB weights buffer mapped (C0 proved 1.6 GB + 192 MB);
its upload takes 31 s (the K-quant repacking), Q4_0's 0.9 s.

**Past the window** (`dsp_gemma … window`, 640 tokens): the largest logit
difference against the CPU path, over the largest logit, at 12 positions:
decode from DDR 1.3–2.7e-2, decode streamed and the few-token pass
1.3–2.7e-2 (equal to each other), the HMX prompt pass 0.9–2.6e-2. Flat
across 512, the same top token as the CPU at all 12: the GPU's picture.

**Greedy text** (`aether run -dev dsp --temp 0 -n 128`, the four prompts
of `raw/prompts/`, against `aether run` on the phone's CPU): `fact` the
same. The other three split, each at a near-tie measured with
`dsp_gemma … split` (the two tokens' logits over the shared prefix):

| Prompt | Split (DSP / CPU text) | CPU decode | DSP decode | DSP HMX prompt |
| --- | --- | --- | --- | --- |
| story, token 24 | " hair" / " a" | −0.159 | −0.126 | −0.030 |
| code, token 112 | " +" / "):" | −0.479 | −0.031 | +0.097 |
| long, token 630 | "Mercury" / "The" | +0.934 | +0.977 | +0.606 |

(A − B, the DSP text's token minus the CPU's.) The DSP's mixed prompt
pass + decode takes the side its HMX pass leans to inside gaps of 0.03–0.5,
the size of the paths' own logit shifts above (2.7e-2 × a largest logit
near 30). `long` is the CPU's run choosing against every path's decode;
the DSP gives the PC's text there. `--lookup 4` on `code` gives plain
decoding's text exactly.

**Memory**: peak PSS 2.05 GB decoding Q4_0 with the base file open (VmHWM
2.18 GB); the GPU's was 3.41 GB.

**Speed** (Q4_0, cooled below 44 °C on zone0 before each run, two rounds;
tok/s):

| | pp32 | pp64 | pp128 | tg64 @ d0 | tg64 @ d512 | tg64 @ d2048 |
| --- | --- | --- | --- | --- | --- | --- |
| DSP, round 1 | 68.8 | 122.2 | 192.9 | 24.9 | 21.8 | 21.4 |
| DSP, round 2 | | | | 24.1 | 21.0 | 20.5 |
| our CPU (step 8, round 1) | | 157.1 | 153.5 | 36.4 | 30.0 | |
| llama.cpp CPU (step 8, round 1) | | 87.6 | 92.8 | 29.4 | 24.3 | |
| our GPU (step 8) | | 38.4 | 47.1 | 19.2 | 8.5 | |

(Round 2's prefill: 192.4 at 128.) The DSP's prompt pass at 128 tokens is
the fastest of the four (1.26× our CPU, 2.1× llama.cpp's CPU); its decode
is 0.7× our CPU's and 0.8× llama.cpp's CPU, 1.3× the GPU. llama.cpp's
Hexagon build cannot load Gemma 4, so there is no DSP-to-DSP reference.

Where decode's 33 ms go (`raw/dsp/speed-depth-r1.txt`): gate/up 31.6 %,
the final norm + classifier 22.0 % (7.3 ms: 262,144 rows), down 16.5 %,
q/k/v 4.9 %, wo 4.3 %, the norm/add steps 3–4 % each, attention 2.3 %. So
the classifier is the first target, then the FFN products.

**C6's decision**: the few-token pass costs 1.09–1.11× a decode step for
2 tokens, 1.5× for 4, 2.8× for 8 (11.6 ms a token). Lookup stays opt-in:
it pays only where drafts are accepted (code, repeats).

## Not done

- **The app's three chat checks** (step 10): the device lock needs the user.
- **S0's GPU and NPU references for Gemma 4.** The phone's OpenCL and
  Hexagon builds cannot load the model; only its CPU backend (cross-built
  at `d02b4c0d6`) runs it, and that is the reference quoted above. A
  current OpenCL / HTP build for the phone would settle B7 and C7.
- **The 0.06 % bound's wording.** It is a same-ISA spread; a cross-ISA
  comparison needs its own bound (the table above is the evidence).
- **Merge to main.**
- G8 optimisation (A3 now has its order: the thread pool's wait path first)
  and the G3 tiled attention variant for 256/512.
