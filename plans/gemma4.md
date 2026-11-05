# Plan: Gemma 4 E2B on the CPU (per-layer embeddings, shared KV)

Written 2026-11-05 on `main` (`c85ff12`), replacing the 2026-11-04 sketch.
Every architectural fact below was read from the two GGUF files and from
llama.cpp `src/models/gemma4.cpp` / `src/llama-vocab.cpp` (`~/llama.cpp` at
`d02b4c0d6`, 2026-11-02) and transformers 5.20.2
(`models/gemma4/modeling_gemma4.py`, `configuration_gemma4.py`). Numbers marked
*calculated* are arithmetic from the files, not measurements.

Implementation goes on a new branch `gemma4` in `../Aether-next` (it holds the
two GGUFs). One commit per step, each with its check; a step is merged only
after its check passes.

## Status (2026-11-05, PC and the phone)

| Step | State |
| --- | --- |
| G0 | done: versions, hashes, llama.cpp PPL/KL baselines, the embed-scale question (`results/phase-gemma4.md`) |
| G1 | done: shapes for both files, Q5_K exact, table kept mapped (tested); peak RSS of a 20-chunk PPL run 2.52 GB on the phone and 2.69 / 2.67 GB on the PC, below the files' 2.84 / 3.11 GB |
| G2 | done: 440/440 strings equal HF `tokenizers` |
| G3 | done: scalar and NEON kernels against numpy, `gemma_elementwise_matches_numpy` passes on the phone (NEON GELU-tanh × up, soft-cap, plain RMSNorm) with the other 17 kernel tests; no tiled 256/512 attention variant yet (G8) |
| G4 | done: every layer vs HF within 1.0e-5, batched and one-token, past the 512 window; Q4_0 greedy vs llama-completion splits only at near-ties (≤ 0.019 logits) |
| G5 | PPL on the PC (Q4_0 0.06 %, Q4_K_M 0.5 % from llama.cpp) and on the phone (Q4_0 57.5076, Q4_K_M 185.6800); llama.cpp's phone CPU is 57.4303, i.e. ours is +0.135 %, which the results file splits and flags for investigation; greedy text on the phone splits only at near-ties (gaps 0.106 / 0.047); KL within llama.cpp's bars on the PC and the GPU |
| G6 | template, stops, hidden reasoning, C ABI default done; template and the C ABI's GPU Gemma path checked on the phone (`c_abi_gpu_gemma4`); app not tried |
| G7 | Done (2026-11-06, cooled, two rounds): ours 157/154/124 tok/s pp 64/128/512 and 36.4 / 30.0 tg at depth 0 / 512, against llama.cpp's 88/93/85 and 29.4 / 24.3; the Adreno does 38-47 pp and 19.2 / 8.5 tg, i.e. behind the CPU on every size |

## 1. Why this model, and what we compare against

- **A production app runs it on this phone.** The user's run: 28.1 tok/s
  decode over 1227 tokens, TTFT 654 ms for 12 tokens, 1.22 GB RAM (charging;
  file format unknown).
- **llama.cpp on this phone, same session as K5** (caps 2.40/2.44 GHz, 6
  threads, `results/phase-kquant.md`):

  | | pp64 | pp512 | tg64 | tg256 | tg64 @ d512 | tg256 @ d512 |
  | --- | --- | --- | --- | --- | --- | --- |
  | Q4_0 | 117.91 | 106.11 | 30.54 | 28.57 | 24.34 | 21.73 |
  | Q4_K_M | 88.09 | 75.76 | 29.42 | 27.18 | 21.93 | 21.48 |

- **Two ideas the engine does not have**, both made for phones:
  - *Per-layer embeddings (PLE):* each token also looks up a 256-wide vector
    for each of the 35 layers, from a 1.32 GB table. One lookup reads one row
    (5,040 B in Q4_0, 6,160 B in Q5_K, calculated), so the table can stay on
    flash, mapped and mostly untouched.
  - *Shared KV:* layers 15–34 have no K/V of their own and read the cache of
    layer 13 (sliding) or 14 (full). The cache is 15 layers deep, not 35.

### Bytes per decoded token (calculated)

| | Q4_0 file (`ggml-org`) | Q4_K_M file (`unsloth`) |
| --- | --- | --- |
| file | 2,825.7 MB | 3,090.9 MB |
| PLE table (lookups only) | 1,321.2 MB, Q4_0 | 1,614.8 MB, **Q5_K** |
| everything else, read per token | 1,504.5 MB | 1,476.1 MB |
| of which the tied classifier `token_embd` | 427.8 MB, Q8_0 | Q4_K |
| of which `per_layer_model_proj` | 27.5 MB, BF16 (55 MB once widened to f32) | same |
| of which `inp_gate` + `proj` (35 each) | Q4_0 | **F32**, 110 MB |

At llama.cpp's 30.54 tok/s the Q4_0 file reads ~46 GB/s. Our LFM3 Q4_K_M
decode now reads ~53 GB/s (K5); at that rate the Q4_0 file would give
~34 tok/s (a projection, not a target: this is a learning project).

## 2. The architecture

### Shape (from the Q4_0 GGUF header)

| key | value |
| --- | --- |
| layers, width | 35, 1536 |
| FFN width (`feed_forward_length`, an **array**) | 6144 for layers 0–14, 12288 for 15–34 |
| query heads, KV heads | 8, **1** (multi-query) |
| head size | **512** on full layers (`key_length`), **256** on sliding (`key_length_swa`) |
| layer pattern (`sliding_window_pattern`) | 4 sliding : 1 full; full = 4, 9, 14, 19, 24, 29, 34 |
| sliding window | 512 |
| shared KV (`shared_kv_layers`) | 20: layers 15–34; source = 13 for sliding, 14 for full |
| RoPE θ | 1e6 full, 1e4 sliding |
| RoPE on full layers | `rope_freqs` [256]: 64 factors of 1, then 192 of 1e30 = only pairs 0–63 rotate, angles `θ^(−2i/512)` (HF: "proportional", `partial_rotary_factor` 0.25) |
| RoPE layout | NeoX (`(x[i], x[i + d/2])`, our `RopeLayout::Halves`) |
| RMS ε | 1e-6 |
| attention scale | **1.0** (no 1/√d; the Q/K norms do the scaling) |
| activation | GELU, tanh approximation (`gelu_pytorch_tanh`), gate × up |
| logit soft-cap | 30: `30 · tanh(x / 30)` |
| PLE width (`embedding_length_per_layer_input`) | 256 (8960 = 35 × 256) |
| vocab | 262,144; classifier tied to `token_embd` |

Per layer (`blk.N.`): `attn_norm`, `attn_q`, `attn_k`/`attn_v` (only in
layers 0–14 of the Q4_0 file; the Q4_K_M file carries them in all 35, and
layers 15–34 must **ignore** them), `attn_q_norm`, `attn_k_norm`,
`attn_output`, `post_attention_norm`, `ffn_norm`, `ffn_gate`/`up`/`down`,
`post_ffw_norm`, `inp_gate` [1536 → 256], `proj` [256 → 1536], `post_norm`,
`layer_output_scale` (one f32). Global: `token_embd`, `per_layer_token_embd`,
`per_layer_model_proj` [1536 → 8960], `per_layer_proj_norm` [256],
`rope_freqs`, `output_norm`.

Note the projections are not square: `attn_q` is 1536 → 2048 (sliding, 8 ×
256) or 1536 → 4096 (full, 8 × 512), and `attn_output` maps it back to 1536.
The engine assumes `n_heads · head_dim = dim` today.

### The forward pass (`gemma4.cpp` lines 161–448)

Once per token, before the layers:

    e   = token_embd[tok] · √1536                        (llama.cpp: sqrtf(1536) in f32)
    p   = per_layer_model_proj · e / √1536               [8960]
    p   = RMSNorm_256(p) · per_layer_proj_norm           (each of the 35 chunks of 256)
    ple = (p + per_layer_token_embd[tok] · √256) / √2    [35, 256]

Per layer `l`, with `h` the residual:

    x  = RMSNorm(h) · attn_norm
    q  = rope(RMSNorm_head(Wq x) · q_norm)
    if l < 15:  k = rope(RMSNorm_head(Wk x) · k_norm)
                v = RMSNorm_head(Wv x)                   (no gain; v = Wk x if Wv is absent)
                store k, v in layer l's cache
    else:       read the cache of layer 13 (sliding) or 14 (full)
    a  = h + RMSNorm(Wo · attention(q, K, V, scale 1.0)) · post_attention_norm
    f  = a + RMSNorm(down(gelu(gate x') · up x')) · post_ffw_norm,   x' = RMSNorm(a) · ffn_norm
    g  = gelu(inp_gate f) · ple[l]                       [256]
    h  = (f + RMSNorm(proj g) · post_norm) · layer_output_scale

Then `RMSNorm(h) · output_norm`, the tied classifier, and the soft-cap.

### The tokenizer (`llama-vocab.cpp` 540–570, 715–725, 2413, 3589)

`tokenizer.ggml.model = gemma4` is **SentencePiece-style BPE**, not GPT-2
byte-level:
- the text is split only on newlines (`[^\n]+|[\n]+`); runs of newlines that
  are a token on their own (`\n` is id 107, `\n\n` 108, …) are taken whole;
- spaces become `▁` (U+2581); no space prefix (`add_space_prefix` false);
- merges by rank (514,906 of them) on raw UTF-8 characters, no byte
  remapping;
- characters with no token fall back to `<0xXX>` byte tokens (256 of them,
  token type 6);
- BOS (2) is always added (llama.cpp forces `add_bos`); EOS 1; and the
  chat's end-of-turn `<turn|>` (id 106) is a stop token.
- Token types in the file: 261,864 normal, 256 byte, 17 control, 7
  user-defined.

### Chat template (in the GGUF, 18,569 characters)

    <bos><|turn>user\n{text}<turn|>\n<|turn>model\n

(`<|turn>` 105, `<turn|>` 106, `\n` 107), with `<|turn>system\n…<turn|>\n`
for a system prompt. Thinking is switched on by `<|think|>` (98) at the top
of the first system turn; the model then writes
`<|channel>thought\n…<channel|>`, hidden as LFM3's `ThinkFilter` does. The exact strings are checked against HF
`apply_chat_template` in G6.

## 3. What the engine has today, and what changes

| Area | Today (`src/`) | Gemma 4 needs |
| --- | --- | --- |
| `model/config.rs` `Config` | one `head_dim`, one `hidden_dim`, `kv_dim` = heads × head_dim, `LayerKind::{Attention, Conv}` with window and `Rope { theta, pairs, layout }` | per-layer head size, q width and FFN width; a KV source per layer; attention scale; embedding scale; soft-cap; V norm; activation (SwiGLU / GELU-tanh); PLE width |
| `model/gguf.rs` `load` | `llama`, `qwen2`, `lfm3`; checks `attn_q` is [dim, dim]; `feed_forward_length` a number | `gemma4`; per-layer shapes; the FFN array; optional K/V; the PLE tensors; `rope_freqs` → `pairs` (the LFM3 code already counts factors < 1e29: 64) |
| `weight.rs` | F32/Q8_0/Q4_0/Q5_0/Q4_K/Q6_K mapped; F16/BF16 **widened to f32**; repack copies matrices into RAM | the PLE table **stays mapped, never repacked or widened**, with a one-row lookup for Q4_0, **Q5_K** (new) and BF16 |
| `quant.rs` | Q4_K, Q6_K blocks | `BlockQ5_K` + dequantize one row (lookup only, no matmul kernel) |
| `model/transformer.rs` `State` / `Batch` | `q`, `xb` sized `dim`; `hb` `hidden_dim`; caches at `cache_off[l] · kv_dim` | buffers at the largest per-layer width (q 4096, hidden 12288); cache offsets in floats with per-layer `kv_dim` (512 / 256); no cache for shared layers; a `[35, 256]` PLE buffer |
| `forward_traced` / `batch_impl` | one layer body for all models | shared-KV read, V norm, GELU, the PLE block, the output scale, soft-cap; both the decode and the batched path (perplexity and prefill use the batch path) |
| attention kernels (`kernels/neon.rs`) | fast `group_fixed` for head 48/64/128; others a general `dot_row` loop; scale 1/√hs built in; flash-decoding only for 48/64/128 | a scale parameter; head 256 and 512 (512 = 128 NEON registers per row, so a tiled variant, not `group_fixed::<128>`) |
| `kernels` | `swiglu`, `rmsnorm`, `rmsnorm_heads` | `gelu_tanh_mul`, `rmsnorm_heads` without a gain, `softcap` |
| `tokenizer/` | `bytelevel` (GPT-2), `llama2c` (SentencePiece scores) | an SPM-BPE mode (above) |
| `chat.rs`, `session.rs`, `ffi.rs` | ChatML, LFM3 | the Gemma 4 template, stop tokens, the app's model list |
| GPU (`model/gpu.rs`), DSP (`model/dsp.rs`) | all-attention models | refuse `gemma4` with a clear error (out of scope) |

Memory (calculated, f32 cache as today): full layers with a cache (4, 9, 14):
3 × (512 + 512) × 4 B = 12 KB per position, 50 MB at 4096 positions; the 12
sliding layers with a cache: rings of 512 + 256 rows × 256 × 2 × 4 B =
18.9 MB, fixed. The f16 cache (`plans/kv-cache.md`) halves both.

## 4. Scope

- **In:** Gemma 4 E2B text decoding on the CPU from the two GGUFs: Q4_0
  (`ggml-org`) and Q4_K_M (`unsloth`); the batched prompt pass; perplexity
  and KL; the chat template; the C ABI and the app.
- **Out:** images, audio and video (`mmproj`); E4B, 26B MoE, 31B (same code
  for E4B, a later check); GPU and DSP backends; the MTP draft head is G9,
  optional.
- **Kept:** every existing model bit-identical: Qwen / SmolLM2 / LFM3 PPL to
  the digit and every test unchanged, after every step.

## 5. References (correctness from these only)

| What | Reference | Used in |
| --- | --- | --- |
| tokens | llama.cpp `llama-tokenize` on the GGUF (no gating); HF `tokenizers` when the repo is accessible | G2 |
| kernels | numpy (`tools/gen_kernels.py`) | G3 |
| Q5_K rows | `gguf-py` `dequantize` | G1 |
| per-layer activations, logits | HF transformers 5.20.2, `google/gemma-4-E2B-it` (gated: the user accepts the licence on the website) | G4 |
| greedy text, PPL, KL | llama.cpp `llama-completion`, `llama-perplexity` (`--kl-divergence`) at `d02b4c0d6` | G4, G5 |
| speed | `llama-bench` on the phone, same session | G8 |
| chat template | HF `apply_chat_template` (or the GGUF template rendered by llama.cpp's `--jinja`) | G6 |

**PC memory** (23 GB, ~12 GB free): the model is 5.1 B parameters, 2.35 B of
them the PLE table. HF in f32 with everything else in f32 and the PLE table
kept in bf16 (a lookup; bf16 → f32 is exact, and the weights are bf16
originally) is ~14 GB (calculated). The dump script loads, runs a short input,
writes the taps to `hostdata/gemma4/ref.safetensors`, and exits, as
`tools/hf_dump_lfm3.py` did. Our BF16 GGUF run on the PC keeps the PLE table
mapped in BF16 (the lookup reads BF16 rows) and widens the rest: ~10 GB.

## 6. Steps

### G0. References and baselines (PC; adds nothing to the engine)

- Files: the two GGUFs (`SHA256SUMS` in `testdata/gguf/`); add the BF16
  GGUF (`ggml-org`, 9.3 GB) for the KL base and the f32 activation check.
- HF: accept the licence; download `google/gemma-4-E2B-it` to
  `hostdata/gemma-4-E2B-it`; check `transformers` 5.20.2 loads it as above.
- `tools/hf_dump_gemma4.py`: taps per layer for 3 short prompts (≤ 64
  tokens, including one longer than 512 tokens for the sliding window: use
  600): `embed`, `ple` (the [35, 256] after the combine), per layer `q`, `k`,
  `v` (after norm and RoPE), `attn_out`, `resid_attn`, `ffn_out`,
  `ple_out`, `resid`, and the logits after the soft-cap.
- **Open question for G0:** HF casts `√hidden` to bf16 (39.19 → 39.25, the
  modeling file's comment on `embed_scale`); llama.cpp multiplies by
  `sqrtf(1536)` in f32. Record which the HF taps show; the engine follows
  llama.cpp (the GGUF reference) and the tap check allows for it.
- PC baselines: `llama-perplexity` 20 × 512 on wikitext-2 for BF16, Q4_0,
  Q4_K_M; the KL base from BF16 (`--kl-divergence-base`) and KL of Q4_0 and
  Q4_K_M against it; `llama-completion --temp 0 -n 128` greedy text for 3
  prompts; `llama-tokenize` ids for the tokenizer corpus.
- Phone baselines exist (section 1); add peak RSS from `llama-cli` and from
  `/proc/<pid>/status` for both files.

Done when `results/phase-gemma4.md` G0 lists versions, hashes, the baselines
and the answer to the open question.

### G1. Config, loader, Q5_K rows

- `Config`: per-layer `LayerKind::Attention` gains `head_dim`, `q_dim`,
  `kv_source: Option<usize>`; `hidden_dims: Vec<usize>`; `attn_scale`,
  `embed_scale`, `logit_softcap: Option<f32>`, `v_norm: bool`,
  `activation: Activation`, `ple_dim`. Existing models fill them so that
  nothing changes (scale 1/√d, SwiGLU, no PLE).
- `gguf.rs`: the `gemma4` arch; `feed_forward_length` as an array;
  `key_length_swa`; `kv_source` as llama.cpp (`n_layer − shared`, then −2
  sliding / −1 full); K/V tensors skipped for shared layers even if present.
- Weights: `AttentionWeights.wk`/`wv` optional; `LayerWeights.ple:
  Option<PleWeights { inp_gate, proj, post_norm, out_scale }>`; `Weights.ple:
  Option<PleTable { table, proj, proj_norm }>`. The table is excluded from
  `repack` and `convert` (an explicit test).
- `quant.rs`: `BlockQ5_K` and `dequantize_q5_k`; a row lookup for mapped
  Q4_0 / Q5_K / BF16.

Checks: every tensor shape against section 2 for both files; Q5_K rows equal
`gguf-py`'s dequantized rows (golden from `tools/gen_quant.py`); peak RSS after
load does not include the table (`/proc/self/status` on the PC); all existing
tests and PPL unchanged.

### G2. Tokenizer

`tokenizer/spm_bpe.rs` (name decided in the step): load tokens, types and
merges from the GGUF; newline split; `▁`; merge by rank; byte fallback;
special tokens matched before merging (`<|turn>`, `<turn|>`, `<|think|>`, …);
decode back to text (`▁` → space, `<0xXX>` runs → UTF-8).

Checks: ids equal `llama-tokenize` (and HF when available) on the 433-string
corpus (`tools/gen_tokenizer_corpus.py`) plus Gemma's special tokens, emoji,
CJK, runs of newlines and spaces; text round-trips; load time and memory of
the 514,906 merges recorded.

### G3. Kernels, each against a numpy golden

- `gelu_tanh_mul` (gate × up), scalar and NEON (`tanh` by the same
  approximation on both, checked against numpy's exact `tanh` within a bound);
- `rmsnorm_heads` without a gain (V);
- `softcap`;
- attention with a scale parameter (1.0), head size 256 and 512, one KV head
  shared by 8 query heads: a tiled NEON variant for 256/512 (the output row
  in blocks that fit the 32 registers), the sliding window, and the
  flash-decoding parts (`attention_group_part`, `merge_parts`) for 256/512;
- the PLE combine (projection scale, chunked RMSNorm, add, 1/√2).

Checks: goldens within the existing bounds (`2e-6 · (√hs + √n)`); the old
head sizes bit-identical.

### G4. The forward pass

Decode (`forward_traced`) and batch (`batch_impl`) both, in this order, each
checked before the next:

1. Embedding × √1536, the PLE inputs, layers 0–14 (own K/V) with sliding
   rings of 512 (+ `PREFILL_CHUNK`) and full caches, V norm, GELU, PLE block,
   output scale.
2. Layers 15–34 reading their source's cache (the source has already stored
   this position: layer 13/14 run first). Shared layers never write.
3. Final norm, tied classifier, soft-cap.

Checks:
- BF16 GGUF on the PC against the HF taps, every tap, every layer, relative
  error < 1e-3 (LFM3's bound), including the 600-token prompt (sliding
  window wrap);
- decode equal to batch, as the existing `batched_matches_decode` tests;
- Q4_0 greedy text equal to `llama-completion --temp 0` on the 3 prompts for
  128 tokens (or to the first divergence explained by float order, recorded);
- a test that layers 15–34 leave the cache unchanged.

### G5. Accuracy on the real files (PC, then a phone spot check)

PPL 20 × 512 and KL against our BF16 run, for Q4_0 and Q4_K_M, next to
llama.cpp's (G0): expect agreement as LFM3's A5 (24.85 vs 24.83); context
2048 against HF bf16 as A5. On the phone, the Q4_0 PPL once (batched path).

### G6. Chat, C ABI, app

`chat::gemma4(system, user, thinking)`; stop on `<turn|>` and `<eos>`;
thinking hidden by the existing filter with Gemma's markers; the C ABI asks
the model for its template (as A6); the app lists Gemma 4. Checks: rendered
prompt equals HF `apply_chat_template` for system / no system / thinking;
a multi-turn chat in the app.

### G7. First phone run

`bench_suite` (cores 2–7, `--pp 64,128,512 --tg 64 --depth 0,512`), cooled
to 45 °C, battery, caps recorded, alternated with `llama-bench` in the same
session (two rounds, reversed). Peak RSS and the app's PSS next to the
1.22 GB the other app showed. A `simpleperf` profile: where decode time goes
(classifier, FFN, attention at head 512, PLE lookup page faults).

### G8. Optimisation, measurement-gated (one commit per idea, each measured)

Candidates, ordered by the profile, not decided yet:
- `per_layer_model_proj` as BF16 (a `bfdot` GEMV; the phone has bf16)
  instead of widening to f32: 55 → 27.5 MB per token;
- the Q4_K_M file's F32 `inp_gate` / `proj` (110 MB per token): quantize at
  load (`--format`) only if PPL stays within the step's bound;
- the full layers' head-512 attention at depth (7 of 35 layers read it);
- PLE lookups: `madvise(WILLNEED)` on the next token's rows during the
  previous token's classifier (llama.cpp's `llama_prefetch_rows`);
- the f16 KV cache from `plans/kv-cache.md` applies here too.

### G9. Optional: the MTP draft head

`mtp-gemma-4-E2B-it.gguf` (0.10 GB, `gemma4-assistant` arch: reads the
target's post-norm hidden state). Needs a hidden-state hand-off our `--draft`
does not have; plan it after G8 if decode is the limit.

## 7. Risks

- **Two new formats in the Q4_K_M file** (Q5_K table, F32 PLE matrices):
  G1 handles the first, G8 the cost of the second.
- **Head 512** is 4× any head the NEON kernels handle; the general loop works
  but will be slow at depth. G3 adds the tiled kernel before the forward pass
  depends on it.
- **The embedding scale** differs between HF (bf16-rounded) and llama.cpp
  (f32): G0 answers it before G4's bounds are set.
- **PC memory** for the HF reference: f32 with the PLE table in bf16 is
  ~14 GB; close the IDE and browsers during G0's dump.
- **RSS.** The engine repacks quantized matrices into owned memory and drops
  the mapping; the PLE table must stay mapped, or RSS grows by 1.3–1.6 GB.

## 8. What the review will check

- Every number in sections 1–3 against the files and the cited code.
- Existing models unchanged to the digit after every step.
- PLE table never copied, repacked or widened; peak RSS recorded.
- Shared-KV layers never write the cache; K/V tensors in the Q4_K_M file's
  shared layers are ignored.
- Decode equals batch; GPU and DSP refuse the model cleanly.
- Timings same-session, alternated, with caps.

## 9. Sources

- GGUF: https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF,
  https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF
- llama.cpp `d02b4c0d6`: `src/models/gemma4.cpp`, `src/models/gemma4-assistant.cpp`,
  `src/llama-vocab.cpp` (`LLAMA_VOCAB_PRE_TYPE_GEMMA4`), `src/llama-model.cpp`
  (rope type)
- transformers 5.20.2: `models/gemma4/modeling_gemma4.py`
  (`project_per_layer_inputs`, `Gemma4RMSNorm(with_scale=False)`,
  `layer_scalar`), `configuration_gemma4.py` (`gelu_pytorch_tanh`,
  `proportional` RoPE)
- HF model: `google/gemma-4-E2B-it` (gated)
