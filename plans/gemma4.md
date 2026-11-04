# Plan: Gemma 4 E2B, per-layer embeddings and shared KV

Written 2026-11-04 on branch `phone-prep`, from the GGUF header of
`ggml-org/gemma-4-E2B-it-GGUF` (Q4_0) and llama.cpp's `src/models/gemma4.cpp`
(`~/llama.cpp` at `d02b4c0d6`, 2026-11-02). Not started; implementation by the
implementation track after the K-quant work (`plans/kquant-decode.md`) is merged.

## 1. Why this model

- **A production comparison exists.** The user ran Gemma 4 E2B in an Android
  app on this phone: 28.1 tok/s decode over 1227 tokens, 1.22 GB RAM (charging).
  llama.cpp runs the same model; this plan gives the engine a third point.
- **Two ideas the engine does not have yet**, both aimed at phones:
  - *Per-layer embeddings (PLE).* Each token also looks up a 256-wide vector
    for each of the 35 layers, in a 1.32 GB table. A lookup reads one row
    (about 5 KB in Q4_0, calculated), so the table can stay on flash and out of
    RAM. That is why the app reports 1.22 GB for a 2.84 GB file.
  - *Shared KV.* The last 20 of the 35 layers have no K/V projections and read
    the cache of an earlier layer, so the KV cache is 15 layers deep, not 35.
- **Bytes per token** (calculated from the Q4_0 file): 2.84 GB minus the PLE
  table = ~1.52 GB, of which the tied classifier (`token_embd`, Q8_0) is
  428 MB. At 28.1 tok/s that would be ~43 GB/s if the app's file were this one
  (it is not known which format the app uses). Our LFM3 Q4_K_M decode reads
  28.5 GB/s, llama.cpp's 48.3 GB/s (results/phase-lfm3.md A7).

## 2. The architecture, from the GGUF header and `gemma4.cpp`

| | Gemma 4 E2B | what the engine has (LFM3 / Qwen) |
| --- | --- | --- |
| layers, width | 35, 1536 | 24, 2048 |
| FFN | GELU (tanh), gate × up; width 6144 (layers 0–14) and 12288 (15–34) | SwiGLU, one width |
| attention | 8 query heads, **1 KV head** (MQA) | GQA |
| head size | **512** on full layers, 256 on sliding layers | 64 / 128, one per model |
| layer pattern | 4 sliding : 1 full, window 512 | LFM3 sliding window 256 |
| norms | sandwich (pre and post, attention and FFN); Q and K RMSNorm with gains; **V RMSNorm without gain** | sandwich, QK-norm (LFM3) |
| RoPE | θ 1e6 full (with `rope_freqs` factors, "proportional"), θ 1e4 sliding | partial RoPE via `rope_freqs` (LFM3) |
| attention scale | 1.0 (no 1/√d) | 1/√d |
| shared KV | layers 15–34 read the KV of layer 13 (sliding) or 14 (full) | — |
| per-layer embeddings | `per_layer_token_embd` [8960 = 35 × 256, 262144], `per_layer_model_proj`, per layer `inp_gate`, `proj`, `post_norm` | — |
| layer output scale | one scalar per layer (`layer_output_scale`) | — |
| embedding | scaled by √1536; classifier tied | tied (Qwen) |
| logits | soft-capped: 30 · tanh(x / 30) | — |
| tokenizer | `gemma4`: 262,144 tokens, 514,906 merges, BOS 2, EOS 1, no space prefix | byte-level BPE (Qwen, LFM3), SentencePiece (llama2.c) |
| chat | Gemma 4 template (in the GGUF) | ChatML, LFM3 |

Per layer, in order (`gemma4.cpp` lines ~180–340):

    x = attn_norm(h)
    q = rope(q_norm(Wq x))                      k, v only on layers 0–14:
    k = rope(k_norm(Wk x)), v = rms(Wv x)       (v = k if Wv is absent)
    a = h + post_attention_norm(Wo attention(q, k, v))
    f = a + post_ffw_norm(down(gelu(gate a) * up a))
    g = gelu(inp_gate f) * ple[layer]           256 wide
    h = (f + post_norm(proj g)) * layer_output_scale

and once per token, before the layers:

    ple = per_layer_proj_norm(per_layer_model_proj(embed) ...)
          combined with per_layer_token_embd[token] · √256

Open questions A0 must answer from the code, not by guessing:
- the exact combination in `project_per_layer_inputs` (scale, norm, sum);
- `rope_freqs` on full layers: which pairs rotate (the "proportional" form);
- the tokenizer: whether `gemma4` is SentencePiece-style BPE (merges by rank
  on pieces with ▁) or byte-level, and its pre-tokenizer;
- whether GELU is the tanh approximation (`ggml_gelu` is; check the HF config's
  `hidden_activation`).

## 3. Scope

- **In:** Gemma 4 E2B text decoding on the CPU from the official GGUFs (Q4_0,
  and the Q4_K_M once K-quant decode is merged), the chat template, the app.
- **Out:** images and audio (`mmproj`), the MoE 26B, E4B (same code, later
  check), the GPU and DSP backends (their own plans), the MTP draft head (G8,
  optional).
- **Kept:** every existing model stays bit-identical (Qwen/SmolLM2/LFM3 PPL and
  the tests, unchanged to the digit).

## 4. Steps

Each step is one commit with its check.

### G0. References and files (PC; adds nothing to the engine)

- GGUFs (downloaded 2026-11-04, `SHA256SUMS` next to them): `ggml-org`
  Q4_0 (`8e30dff3…`), `unsloth` Q4_K_M (`740185b2…`). Add the BF16
  (9.31 GB) for the KL base.
- HF reference: `google/gemma-4-E2B-it` is gated (licence click-through on the
  website; the user does that). Check the installed transformers has `gemma4`.
  **RAM:** ~5B parameters with the PLE table, so f32 needs ~20 GB, more than
  the PC's ~16 GB. Use bf16 for whole-model logits and f32 for per-layer
  goldens on short inputs (load, run, free), and record the choice.
- Baselines on the PC: `llama-perplexity` (20 × 512) on Q4_0, Q4_K_M and BF16,
  and the KL base from BF16.
- Baselines on the phone: `scripts/kquant-phone.sh` stage `gemma` (llama-bench,
  Q4_0 and Q4_K_M, pp 64/512, tg 64/256, depth 0/512), plus peak RSS with the
  PLE table mapped.

Done when: `results/phase-gemma4.md` G0 lists versions, hashes and baselines.

### G1. Config and weights

- `Config`: per-layer head size (512 / 256), per-layer FFN width, the
  attention scale, the KV source per layer, logit soft-cap, PLE width.
- `model/gguf.rs`: `gemma4`; `feed_forward_length` is an array;
  `shared_kv_layers` gives the KV source (llama.cpp: `n_layer − shared`, then
  −2 for sliding, −1 for full).
- PLE tensors stay mapped and untouched until looked up (no copy, no repack).

Check: every tensor's shape matches section 2; layers 15–34 have no `attn_k`;
existing tests unchanged.

### G2. Tokenizer

The `gemma4` vocab and merges, whatever A0 finds the algorithm to be. Check:
token ids equal HF `tokenizers` on the 433-string set (Phase 5) plus Gemma's
own special tokens; text round-trips.

### G3. Kernels, each against a numpy golden (`tools/gen_kernels.py`)

- GELU (tanh), the FFN with gate × up;
- RMSNorm without a gain (for V);
- soft-capping;
- attention with head size 512 and one KV head, scale 1.0 (the existing
  kernels assume ≤ 128: check registers and tails);
- the PLE row lookup from Q4_0 (dequantize one 8960-wide row).

### G4. The forward pass on the CPU

Per-layer embeddings, shared KV (a layer without K/V reads its source layer's
cache; the cache is allocated for the 15 source layers only), sliding ring of
512 (+ prompt chunk, as LFM3), output scale. Check: per-layer activations
against HF (f32, short input, G0) within the existing bounds; greedy text equal
to llama.cpp's on 3 prompts.

### G5. Accuracy on the real files

PPL 20 × 512 and KL against llama.cpp's for Q4_0 (and Q4_K_M after K5);
context 2048 against HF (bf16) as LFM3's A5 did. Record the KV-cache size at
2048 (calculated: 15 layers, one KV head).

### G6. Chat, C ABI, app

The Gemma 4 template from the GGUF; the C ABI asks the model for it (as A6);
the app selects it. Check: rendered template equals HF `apply_chat_template`.

### G7. Measure on the phone, against llama.cpp, same session

`bench_suite` and `llama-bench`, cooled, battery, caps recorded, two rounds
in reversed order (as A7). Peak RSS with the PLE table mapped, and the app's
PSS next to the 1.22 GB the other app showed. A profile of where decode time
goes.

### G8. Optional: the MTP draft head

`mtp-gemma-4-E2B-it.gguf` (0.10 GB) is a draft model made for this target.
The engine's draft-model speculation (`--draft`) is the place to try it, after
G7, with the same greedy-text test as LFM3.

## 5. What the review will check

- Every number in section 1 and 2 against the file and `gemma4.cpp`; the open
  questions answered from code.
- Existing models unchanged to the digit after every step.
- PLE never copied into RAM; peak RSS recorded.
- Shared-KV layers never write the cache.

## 6. References

- GGUF: https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF,
  https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF
- llama.cpp `src/models/gemma4.cpp`, `src/llama-arch.cpp` (`gemma4`),
  `src/llama-vocab.cpp` (the `gemma4` tokenizer)
- HF: `google/gemma-4-E2B-it` (gated), transformers' `gemma4` modeling file
