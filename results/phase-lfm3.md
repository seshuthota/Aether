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
