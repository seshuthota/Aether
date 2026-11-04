# LFM3-1.7B: hybrid conv/attention on the CPU

Plan: `plans/lfm3.md`. Branch `lfm3`. All numbers are from this phone
(iQOO 13, Snapdragon 8 Elite) unless marked PC.

## A0. References and files

### Versions

| What | Version / revision |
| --- | --- |
| llama.cpp | `b12061`, commit `d02b4c0d6` (needs >= b11960 for LFM3) |
| llama.cpp PC build | Release, `-DGGML_NATIVE=ON`, GNU 13.3.0, x86_64 |
| llama.cpp Android build | Release, NDK 27.3, aarch64 |
| HF transformers | 5.20.2 |
| HF tokenizers | 0.23.3 |
| torch | 2.14.1+cpu |
| LFM3 model repo | `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7` |
| LFM3 GGUF repo | `bad5729bba61d076fff20d2548e62526665af215` |

`transformers` has no built-in `lfm3` in 5.20.2; the reference runs the
repository's `modeling_lfm3.py` / `configuration_lfm3.py` with
`trust_remote_code=True` at the pinned revision. A tiny 4-layer LFM3
config runs a forward pass on the PC CPU: `lfm3_conv.py` only initialises
Triton on CUDA and falls back to the PyTorch `_conv` path on the CPU.
The pinned HF checkpoint is cached in `hostdata/LFM3-1.7B` (weights,
tokenizer and code); a bf16 forward on the CPU produces finite logits
and `The capital of France is` -> ` Paris`.

### Files

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `testdata/models/LFM3-1.7B/tokenizer.json` | 17,908,635 | `7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701` |
| `testdata/gguf/LFM3-1.7B-Q4_K_M.gguf` | 1,099,627,808 | `3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4` |
| `testdata/gguf/LFM3-1.7B-Q8_0.gguf` | 1,846,238,496 | `8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6` |
| `testdata/gguf/LFM3-1.7B-QAD-Q4_0.gguf` | 1,049,615,680 | `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452` |
| `hostdata/gguf/LFM3-1.7B-BF16.gguf` | 3,466,626,336 | `39158a79f56edb85e0c4252ce6447b8d7883ef82ffedd02f0cda7c84da5e9f68` |
| `hostdata/LFM3-1.7B/model.safetensors` | 3,457,635,448 | `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330` |

### Findings that A1-A4 depend on

- **RoPE.** `rope_freqs.weight` is `[1.0 × 32, 1e30 × 32]`; llama.cpp
  attaches it only to full-attention layers (`src/models/lfm3.cpp`,
  `rope_factors = is_swa ? nullptr : ...`). `ggml` divides the angle by
  the factor, so 1e30 means "do not rotate". Sliding layers use full
  rotation at theta = 1e4. This answers A0's open question: the factors do
  **not** apply to the sliding layers.
- **Sliding window edge.** HF's `sliding_window_overlay` masks
  `kv_idx > q_idx - sliding_window`, so a token at `i` sees
  `i - 255 ..= i` (256 positions including itself), not `i - 256 ..= i`.
- **Conv state.** HF's dynamic conv cache keeps 7 `u` rows including the
  current token; llama.cpp keeps the 6 previous rows and prepends the
  batch. The engine follows llama.cpp: 6 previous rows plus the current
  token, width 7, tap 6 on the current token.
- **GGUF tensors.** `shortconv.conv.weight` is F32 `[7, 2048]`;
  `attn_gate.weight` is Q4_K `[2048, 16]`; `token_embd` / `ffn_down` are
  Q6_K, `ffn_gate` / `ffn_up` Q4_K in Q4_K_M. `head_count_kv` is a
  24-element array with 0 on conv layers, and
  `attention.sliding_window_pattern` is an array of booleans.
- **Tokenizer.** `tokenizer.ggml.pre = "lfm2"`, BOS 124894, EOS 124900,
  `add_bos_token = false`, 140 control/user-defined added tokens. The
  tokenizer JSON has `ignore_merges: true` inside `model`, no normalizer,
  and the `\p{N}{1,3}` pre-tokenizer regex.

### Baselines on the PC

Wikitext-2, 20 chunks of 512, `llama-perplexity`.

| File | PPL | Notes |
| --- | --- | --- |
| BF16 | 23.4796 ± 1.05094 | `llama-perplexity -c 512 --chunks 20` |
| Q8_0 | 23.5508 ± 1.05490 | `llama-perplexity -c 512 --chunks 20` |
| Q4_K_M | 24.8278 ± 1.11523 | `llama-perplexity -c 512 --chunks 20` |
| QAD-Q4_0 | 27.1383 ± 1.24457 | `llama-perplexity -c 512 --chunks 20` |

KL base file from BF16: `results/phase-lfm3/raw/bf16-kld-base.bin`
(1.3 GB, `--save-all-logits`, 20 chunks of 512; generated, not committed).
A sanity run of Q4_K_M against it (`pc-kl-Q4_K_M.txt`) gives mean KLD
0.117968 ± 0.003084 and same-top 83.392 ± 0.521 %.

Raw logs: `pc-ppl-*.txt`, `pc-kl-Q4_K_M.txt`,
`phone-llamacpp-*-perf.txt` (screen on) and `phone-llamacpp-*-screenoff.txt`
(the low-cap session, kept only as a condition record).

### Baselines on the phone

`llama-bench -p 64,128,512 -n 64`, cores 2-7, on battery, cooled.

| File | pp64 | pp128 | pp512 | tg64 |
| --- | --- | --- | --- | --- |
| Q4_K_M | 178.61 ± 1.92 | 174.63 ± 3.28 | 154.27 ± 3.56 | 40.91 ± 5.76 |
| QAD-Q4_0 | 292.18 ± 28.72 | 298.99 ± 11.19 | 262.71 ± 0.69 | 50.71 ± 1.13 |

Conditions: on battery, 51-53 %, screen on (the 30-second screen
timeout was temporarily raised to 30 minutes and restored afterwards),
`cpu0` / `cpu6` caps 2.4 / 2.438 GHz at the start of each run,
`MemAvailable` ~6.2 GB. The phone heats during a 1.1 GB model run and
the caps can drop; the ± on pp512/tg64 includes that drift. The earlier
screen-off/low-power session (`cpu0` 1.7856, `cpu6` 1.0176 GHz) is kept
as `phone-llamacpp-*-screenoff.txt`; it is not comparable with this table.

Phone conditions at the start of the session (2026-11-03 00:20):
on battery (AC/USB/wireless all false), 53 %, ~34.4 C battery, max CPU
thermal zone 41.9 C, `MemAvailable` 6,293,652 kB, `/data` 238 G free.
The phone was left to cool before each timed run (`COOL_C=45` in
`scripts/lfm3-a0.sh`).

## A4. The forward pass

The mixed stack runs on the CPU: 13 short-conv layers with a 6-row state,
6 full-attention layers and 5 sliding layers. The sliding layers keep a
`256 + 256`-position ring (the second 256 is the prefill chunk; longer
prompts are cut into 256-token chunks), and each step materializes the
window in a scratch buffer before attention. Conv states are snapshotted
for the last 32 tokens and restored by `restore_conv` when speculative
drafts are rejected.

Checked against the pinned HF repository (`hostdata/lfm3/ref.safetensors`,
f32 eager, `tools/hf_dump_lfm3.py`): a 29-token prompt's full logits and
every layer's residual, logits at positions 0/1/254/255/256/257/300/318/319
of a 320-token prompt, and 16 greedy tokens all match (`< 1e-3` relative).
A separate Q4_K_M check rejects two drafts, restores the conv state and
gives the same next-token logits as plain decoding. The phone test binary
passes all 122 tests, including the 128-wide NEON attention, the LFM3
loader/tokenizer and the conv rollback (`raw/phone-tests.txt`).

## A5. Accuracy on the real files

llama.cpp reference PPL (wikitext-2, 20 x 512) is in the A0 table. The
engine's PC path is scalar on x86 (the NEON kernels are the phone path), so
a full 20-chunk PC run is slow but works:

| File | engine PPL | llama.cpp PPL |
| --- | ---: | ---: |
| Q4_K_M | 24.8542 ± 1.11750 | 24.8278 ± 1.11523 |
| Q8_0 | 23.5482 ± 1.05390 | 23.5508 ± 1.05490 |
| QAD-Q4_0 | 27.1557 ± 1.24588 | 27.1383 ± 1.24457 |
| BF16 (4 chunks) | 14.7419 ± 1.42442 | 14.7313 ± 1.42225 |

The per-chunk values track llama.cpp within 0.1-0.2 (e.g. Q4_K_M 10.3666
vs 10.3223, 21.1917 vs 21.1367, 25.6230 vs 25.5943). BF16 is the first four
chunks (the f32 PC path is slower); its chunks are 10.0984/13.7166/14.3846/
... against llama.cpp's 10.0955/13.6860/14.3638/... .

At context 2048 (one chunk, 1023 scored) the sliding window and the full
layers see different prefixes, and the engine's Q4_K_M PPL, 13.7917
± 1.29664 (phone: 13.7811), is 1.2 % below llama.cpp's 13.9561 ± 1.31213:
ten times the gap at context 512. To see whose number is off, the same
chunk at full precision, against HF transformers (f32, eager,
`tools/hf_ppl_lfm3.py`, 2026-11-04):

| ctx 2048, 1023 scored | PPL | KL against BF16 | same top |
| --- | ---: | ---: | ---: |
| HF transformers, f32 | 13.5848 | | |
| engine, BF16 file | **13.5848** | | |
| llama.cpp, BF16 file | 13.5986 | | |
| engine, Q4_K_M | 13.7917 | 0.052408 | 88.759 % |
| llama.cpp, Q4_K_M | 13.9561 | 0.055758 | 88.172 % |

The engine's BF16 run is HF's PPL to the fourth decimal past 512
positions, so the sliding ring, the window edge and the full layers'
RoPE are right at depth. The Q4_K_M gap is between two quantized paths,
not an error: each engine rounds the activations differently for the
K-quant dots (ours to Q8_0 per 32 values, llama.cpp to Q8_K per 256), and
ours ends up slightly closer to BF16 (KL each against its own BF16 run;
the two BF16 runs differ by 0.1 %). Raw: `aether-ppl-ctx2048-BF16.txt`,
`aether-kl-ctx2048-Q4_K_M.txt`, `phone-aether-ppl-ctx2048-Q4_K_M.txt`,
`pc-ppl-ctx2048-{BF16,Q4_K_M}.txt`, `pc-kl-ctx2048-Q4_K_M.txt`.

The same 20-chunk engine PPL on the phone (cores 2-7) gives Q4_K_M
24.8822 ± 1.11875, Q8_0 23.5744 ± 1.05532 and QAD-Q4_0 27.1489 ± 1.24542,
within 0.05 of llama.cpp. The phone run takes 75-90 s per file.

The engine KL against the BF16 file over all 20 chunks is mean KLD
0.115041, same-top 83.392 % (llama.cpp: 0.117968 / 83.392 %); the engine's
base PPL is 23.4865 against llama.cpp's 23.4796. A one-chunk cross-check
gave 0.058283 / 93.333 %; KL is chunk-size dependent. The full PC KL took
10,103 s on the scalar x86 path.

## A6. Chat, C ABI and the app

- `chat::lfm3` (BOS, optional system, `<think>` or an empty think block)
  matches `apply_chat_template(..., add_generation_prompt=True)` on four
  cases (`testdata/lfm3/chat.txt`).
- `--no-think` turns reasoning off in the prompt; `--hide-think` removes a
  `<think>...</think>` span from the printed reply, including when the tags
  are split across streamed pieces.
- LFM3 sampling defaults (temperature 0.1, top-k 50) are used when `-t/-k`
  are not given. Our sampler has no repetition penalty (llama.cpp's LFM3
  default is 1.1), documented in the CLI usage.
- The C ABI's `chat` flag now asks the loaded model for its template, so
  LFM3 gets BOS + `<think>` instead of Qwen's ChatML. The Android app
  already has the model picker; it uses `MAX_SEQ = 2048`.

## A7. Phone measurements

`scripts/lfm3-a7.sh`, 2026-11-03 19:40-20:09, commit `991d9d2`, cores
2-7 (6 threads), on battery (82 -> 78 %), screen on, cooled to 45 C before
each run, llama.cpp `b12061` in the same session. The order is reversed
in round 2 (r1: ours, llama; r2: llama, ours). Raw:
`results/phase-lfm3/raw/a7/`.

tok/s, round 1 / round 2:

| arm | pp64 | pp128 | pp512 | tg64@0 | tg64@256 | tg64@512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aether Q4_K_M | 160.5* / 114.7 | 174.6* / 123.1 | 159.2* / 131.7 | 29.8* / 26.2 | 17.0* / 26.1 | 25.9* / 26.0 |
| llama.cpp Q4_K_M | 132.6 / 133.2 | 129.2 / 127.6 | 130.1 / 130.3 | 44.2 / 44.3 | | |
| aether QAD-Q4_0 | 194.9 / 197.5 | 218.3 / 189.6 | 200.1 / 178.5 | 44.9 / 42.9 | 42.9 / 41.1 | 39.1 / 40.6 |
| llama.cpp QAD-Q4_0 | 204.6 / 208.7 | 198.8 / 168.2 | 195.0 / 195.8 | 50.9 / 49.6 | | |

\* The first arm ran with higher caps (`cpu0` / `cpu6` 2.75 / 2.65 GHz;
every other arm 1.79 / 1.96 GHz) and throttled during tg64@256 (p95
105 ms a token). Compare round 2 for Q4_K_M. The earlier evening arm
(05d7ed4, caps 2.4 / 2.44 GHz: pp512 164.9, tg64@0 30.7) is in git
history, not in the table.

- **Prompts**: level with llama.cpp. Q4_K_M round 2 is 0.86-1.01x,
  QAD-Q4_0 0.91-1.13x.
- **Decode is behind**, by far the most in Q4_K_M: 26.1 against 44.3 tok/s
  (0.59x); QAD-Q4_0 42.9 against 49.6 (0.86x). Against the bandwidth
  ceiling (file bytes x tok/s): Q4_K_M 1.09 GB x 26.1 = 28.5 GB/s for ours,
  48.3 GB/s for llama.cpp; QAD-Q4_0 1.04 GB x 42.9 = 44.7 GB/s against
  51.7 GB/s. llama.cpp's Q4_K_M decode is at the bandwidth of a Q4_0 file,
  ours is not: our Q4_K/Q6_K decode kernels are the cause, not the
  hybrid stack. Qwen2.5-0.5B's Q4_K_M (where we led llama.cpp by 22 %,
  phase-6) is mostly Q5_0, because its 896-wide rows are not a multiple
  of 256; LFM3 is the first file that runs on the K-quant GEMVs (see
  the profile below).
- **Depth**: decode falls 1-13 % from depth 0 to 512 (sliding window and
  six full layers), as expected for a model whose KV traffic is small next
  to its weights.
- **Memory**: peak RSS 1.13-1.19 GB; `MemAvailable` 5.4-6.7 GB at the start.

Sustained decode, 2 minutes (`sustain-*.txt`):

| file | first 11 s | last 11 s | average | max temp | power | J / token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 26.3 | 21.3 | 22.9 tok/s | 86.1 C | 3.70 W | 0.162 |
| QAD-Q4_0 | 40.5 | 34.1 | 35.7 tok/s | 88.4 C | 4.92 W | 0.138 |

Both lose 16-19 % in the first minute and then hold. The SoC sits near
88 C during every LFM3 run: a 1.1 GB model decoding is a thermal load on
this phone, which the 0.5B models never were. The power figures come
from two different sources (current x voltage for Q4_K_M, the charge
counter over 42 s for QAD-Q4_0), and the short-run monitor's averages
(0.5-4.4 W) vary too much between arms to use. The J / token figures are
rough.

### Where decode time goes

`simpleperf record -e cpu-cycles:u`, 300 greedy tokens of Q4_K_M, cores
2-7, 2026-11-04 (caps 2.75 / 2.65 GHz, 30.6 tok/s; `raw/a7/simpleperf.txt`):

| symbol | cycles |
| --- | ---: |
| `dot4_q4_k_q8_0` | 54.1 % |
| `dot4_q6_k_q8_0` | 32.9 % |
| thread pool (spin / run) | 8.3 % |
| `quant_rows` (activation to Q8_0) | 1.7 % |
| attention (all three paths) | 0.9 % |
| `short_conv_step` | 0.07 % |

87 % of decode is the two K-quant GEMVs; the conv layers are nearly free
next to them, as the plan expected, and so is attention at this depth. The
kernels are compute-bound, not bandwidth-bound: the activation is Q8_0, so
every 32-value sub-block of a Q4_K / Q6_K row ends in its own
`vcvtq_f32_s32` and float FMA with its own scale (Q6_K also a
`vmulq_n_s32` / `vmlaq_n_s32` by the sub-block scale), where an activation
quantized per 256-value super-block (llama.cpp's Q8_K, with block sums for
the `dmin` term) lets the sub-blocks accumulate in int32 and convert once
per super-block. That kernel work is its own step, after this plan.

### The app

Debug APK built and installed (`scripts/android-native.sh`,
`gradlew installDebug`), the Q4_K_M file in the app's shared folder, CPU,
cores 2-7, chat on: "Why is the sky blue?" streams a reasoning span, then
`</think>`, then the answer. 16 prompt tokens in 579 ms, 462 tokens at
29.8 tok/s; with lookup on, 441 tokens at 27.7 tok/s (the app samples at
temperature 0.7, so few drafts are accepted). The app process is at
1.30 GB PSS / 1.42 GB RSS (`dumpsys meminfo`), 145 MB of it native heap
(the KV cache, `MAX_SEQ = 2048`, and buffers). The app shows the reasoning
as plain text: the opening `<think>` is part of the prompt, so the reply
starts inside the span and only `</think>` appears. Splitting the reply
there (reasoning dimmed or folded) is an app change for later.
