# Plan: LFM3-1.7B, a hybrid architecture and a bigger model

Written 2026-11-02, after the flash-attention merge. Implementation by the user, review by
Claude after each step. Work on branch `lfm3` in `../Aether-impl`; merge into `main` only
after A7's checks pass. Model: <https://huggingface.co/LiquidAI/LFM3-1.7B> (LFM Open License
v1.0), GGUFs in `LiquidAI/LFM3-1.7B-GGUF`.

## 1. Why this model

Every model so far (stories15M, SmolLM2, Qwen2.5) is the same Llama-style decoder: 24-ish
identical attention + MLP layers. LFM3 is a **hybrid**: of its 24 layers only 11 attend, and
5 of those see only the last 256 positions. The other 13 are **gated short convolutions**: a
fixed-size state of 6 tokens instead of a KV cache. It is also 3.5× Qwen2.5-0.5B's size, the
first model whose weights (~1.1 GB at Q4_K_M) make the phone's memory bandwidth bind decode
hard.

What it teaches:
- **Why hybrids exist: the KV cache.** Per cached position LFM3 keeps 6 full-attention layers
  × 4 KV heads × 128 × (k + v) = 6144 values, the same as Qwen2.5-0.5B (24 × 2 × 64 × 2) for a
  model 3.5× bigger. The sliding layers cost a fixed 256 positions, the conv layers 6 tokens.
  At a 32K context (f16): ~400 MB, against ~1.6 GB if all 24 layers attended fully.
- **Decode's bandwidth ceiling.** One token reads every weight once. ~1.1 GB per token at
  ~60 GB/s is at most ~55 tok/s. Qwen2.5-0.5B Q4_0 ran at 158–180 tok/s.
- **Partial RoPE, per-layer RoPE bases, QK-norm, output gates, sandwich norms**: five small
  variations that the reference implementations differ on in detail. That makes this a good
  exercise in reading a model's code rather than guessing.

## 2. The architecture, from `modeling_lfm3.py` and the GGUF header

`config.json`: hidden 2048, 24 layers, 16 query heads, 4 KV heads, head_dim 128, MLP 7680,
vocab 128000, tied embeddings, norm eps 1e-5, sliding window 256, conv width 7, no conv bias.
1.73B parameters: embedding 262M, MLP 1132M, conv layers 218M, attention layers 115M.

Layer order (`layer_types`, C = conv, F = full attention, S = sliding attention):

    C C F C S C F C S C F C S C F C S C F C S C F C

**Every layer** (sandwich norm, `use_sandwich_norm: true`):

    h = h + post_op_norm( mixer( op_norm(h) ) )         GGUF: attn_norm, post_attention_norm
    h = h + post_ffn_norm( mlp( ffn_norm(h) ) )         GGUF: ffn_norm, post_ffw_norm
    mlp(x) = w2( silu(w1 x) ⊙ w3 x )                    GGUF: ffn_gate, ffn_up, ffn_down

All norms are RMSNorm with a gain (`weight * x / sqrt(mean(x²) + eps)`). After the last layer:
`output_norm` (HF calls it `embedding_norm`), then the tied classifier (`token_embd`).

**Conv layer** (mixer = `Lfm3ShortConv`; GGUF `shortconv.in_proj` [2048 → 6144],
`shortconv.conv` [2048 channels × 7], `shortconv.out_proj` [2048 → 2048]):

    [b, c, x] = split3( in_proj(h) )          # in that order: B, C, x
    u_t       = b_t ⊙ x_t
    y_t[ch]   = Σ_{k=0..6} w[ch, k] · u_{t−6+k}[ch]     # causal, depthwise; w[ch, 6] × the current token
    out       = out_proj( c_t ⊙ y_t )

The state between steps is the last 6 `u` vectors per conv layer (13 × 6 × 2048 floats,
~640 KB in f32). A prompt is a convolution over the whole batch plus the 6 cached rows.

**Attention layer** (`Lfm3Attention`; GGUF `attn_q`, `attn_k`, `attn_v`, `attn_output`,
`attn_q_norm` [128], `attn_k_norm` [128], `attn_gate` [2048 → 16]):

    q = q_proj(h) as [16, 128];  q = rmsnorm_per_head(q) * q_norm      # QK-norm, gain shared by heads
    k = k_proj(h) as [4, 128];   k = rmsnorm_per_head(k) * k_norm
    v = v_proj(h) as [4, 128]
    q, k = rope(q, k)                      # see below; in f32
    o = attention(q, k, v), scale 1/√128, causal; sliding layers: only the last 256 positions
    o[head] *= sigmoid( gate_proj(h) )[head]                           # one scalar per head
    out = o_proj(o)

**RoPE** (rotate-half layout, pairs `(i, i + 64)`):
- Full layers: θ = 1e6, `partial_rotary_factor 0.5` in LFM3's *zero-tail* form:
  `inv_freq[i] = θ^(−2i/128)` for i < 32 and **0** for i = 32..63. So pairs (i, i+64) with
  i < 32 rotate, with the frequencies of a full 128-wide head; dims 32..63 and 96..127 are
  untouched. This is **not** the usual partial RoPE (rotate the first 64 dims with
  frequencies of a 64-wide head); getting it wrong still produces text.
- Sliding layers: θ = 1e4, full rotation (no partial factor).
- The GGUF has `rope.dimension_count = 128`, `rope.freq_base_swa = 10000` and a
  `rope_freqs.weight` [64] tensor. llama.cpp presumably encodes the zero tail through those
  frequency factors; A0 reads its LFM3 code to find out whether they apply to the sliding
  layers too. HF's code is the definition; if the two disagree, that is a finding to report.

**Sliding window**: which positions does a token at i see: `i − 255 ..= i` or
`i − 256 ..= i`? Taken from transformers' `create_sliding_window_causal_mask`, checked in A3
by a golden that is wrong under the other reading.

**Tokenizer** (`tokenizer.json`, GGUF `tokenizer.ggml.pre = "lfm2"`): byte-level BPE like
Qwen2's, with three differences:
1. The regex has `\p{N}{1,3}` (digits in groups of up to 3) where Qwen2's has `\p{N}`.
2. No normalizer (Qwen2 applies NFC).
3. `"ignore_merges": true`: a pre-token that is itself in the vocabulary becomes that one
   token without running the merges.

140 added tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, …). BOS 124894, EOS 124900, and
`add_bos_token: false` in the GGUF. The chat template is ChatML and adds `<think>` to the
generation prompt (reasoning on by default).

## 3. Scope

In:
- The CPU path (NEON, threads), decode and batched prefill, for the official GGUFs:
  Q4_K_M, Q8_0, the QAD Q4_0 (quantization-aware distilled weights, so *not* llama.cpp's
  quantization of the BF16 file) and BF16.
- Tokenizer, chat template for `aether chat`, the C ABI and the app.
- External verification (section 5) and a CPU benchmark against llama.cpp.

Out (each is its own later plan):
- The GPU (A8 lists what it needs: head size 128, conv, gate, QK-norm, sliding window).
- The DSP.
- Our own quantization of LFM3 (`aether quantize`): use the official files first.
- Contexts beyond what the phone's RAM allows; A7 records what it is.

## 4. Steps

Each step is one commit with its check. Today's models must stay bit-identical through all
of them (Qwen/SmolLM2 PPL and the existing tests, unchanged to the digit).

### A0. References and files (PC; adds nothing to the engine)

- llama.cpp: `~/llama.cpp` is b8660; LFM3 needs **b11960 or newer**. Update and rebuild the
  PC and the Android (`build-android`) builds. Read its LFM3 graph for the RoPE question
  above and the conv-state layout.
- Download into `testdata/` (gitignored): the Q4_K_M, Q8_0, QAD-Q4_0 and BF16 GGUFs (~7.5 GB
  together) and `tokenizer.json`.
- HF reference in `tools/.venv`: check whether the installed transformers has `lfm3` built
  in. If not, `trust_remote_code` runs the repo's Python files: pin the revision (commit
  hash), and note that `lfm3_conv.py` (Triton) must fall back to the PyTorch path on the PC's
  CPU. f32 on the PC needs ~7 GB of RAM.
- Baselines on the PC: `llama-perplexity` on wikitext-2 (20 × 512) for each GGUF, and the
  `--kl-divergence-base` file from BF16.
- Baselines on the phone: `llama-bench -p 64,128,512 -n 64` on cores 2–7, battery, cooled,
  for Q4_0 and Q4_K_M. Record the phone's free RAM.

Done when: `results/phase-lfm3.md` A0 lists versions, file hashes and the baselines.

### A1. Config and weights for a mixed stack

Today `Config` assumes identical layers and `head_size = dim / n_heads`, and `LayerWeights`
has one shape.
- `Config`: `head_dim` stored (not derived), a per-layer kind
  (`Attention { window: Option<usize>, rope: Rope }` / `Conv { width }`), per-kind RoPE
  (θ, rotated pairs), flags for QK-norm, the output gate and sandwich norms. llama/qwen2
  fill them so that nothing changes for them.
- `LayerWeights`: the mixer as an enum (attention or conv weights), optional QK-norm gains,
  gate and post-norms.
- `model/gguf.rs`: `lfm3`. `head_count_kv` is an **array** per layer (0 = conv),
  `sliding_window_pattern` is an array of booleans.

Check: the Q4_K_M file loads, and every tensor's shape matches section 2. Existing tests
are unchanged.

### A2. Tokenizer

- The `lfm2` pre-tokenizer: Qwen2's regex with `\p{N}{1,3}`, no NFC, `ignore_merges`.
- Check: the same 433-string test as Qwen against HF `tokenizers` with LFM3's
  `tokenizer.json`, plus multilingual strings (the model covers 16 languages) and digit runs
  (`1234567`, which split 123|456|7).

### A3. New kernels, each against a numpy golden (`tools/gen_kernels.py`)

- **Short conv**: `u` for a batch with 6 cached rows in front, conv weights [ch, 7]; cases
  with the batch shorter than, equal to and longer than the width, and a decode step.
- **RMSNorm per head** (QK-norm): `[n, heads, 128]`, gain shared across heads.
- **RoPE with a rotated prefix and per-call θ**: LFM3's zero tail. The golden is computed
  from HF's formula, so a "usual partial RoPE" implementation fails it.
- **Sigmoid gate per head** applied to the attention output.
- **Sliding-window attention**: positions > 256 back masked out. Goldens at positions
  255/256/257 and with a cache of a few thousand, so off-by-one at the window edge fails.
- **Head size 128** in the NEON attention (today it has fixed paths for 64 and 48 and a
  generic fallback): a fixed path for 128, tested like the others.

Check: the goldens pass on the host and the phone.

### A4. The forward pass on the CPU

- Decode (`forward`) and batched prefill (`forward_batch`) through the mixed stack.
- The KV cache only for attention layers. Sliding layers keep a 256-position ring (memory:
  5 layers × 256 positions instead of the whole context). Conv layers keep their 6-row
  state, saved and restored like the KV cache wherever the engine rewinds (speculative
  decoding rolls back rejected drafts: the conv state must roll back too). The ring needs
  256 + the longest draft slots: k drafts written into a 256-slot ring overwrite positions
  that are needed again once the drafts are rejected.
- Check against HF transformers (f32, PC): logits for a few prompts, including one longer
  than 256 tokens (so sliding layers drop positions) and token by token against batched, all
  within the bounds used for Qwen. Per-layer hidden states via `forward_traced`, if a layer
  diverges.

### A5. Accuracy on the real files (PC and phone)

- PPL on wikitext-2 (20 × 512) for each GGUF against `llama-perplexity`, same tokens.
- KL against the BF16 model. On the phone an f32 copy is ~7 GB, which probably does not fit
  next to everything else. Options, decided in A0 once the free RAM is known: read
  llama.cpp's `--kl-divergence-base` file (its own logits from BF16, computed on the PC), or
  run our KL on the PC.
- A context-2048 run: the first check where the sliding window and the full layers behave
  differently.

### A6. Chat, C ABI and the app

- ChatML with `<think>`. Show or hide the thinking span. Sampling defaults from
  `generation_config.json`: temperature 0.1, top-k 50, repetition penalty 1.1. The engine
  has no repetition penalty yet: add it, or document that ours differs.
- Speculative decoding: prompt lookup works if A4's rollback covers the conv state.
- The app: model picker, and a check that a 1.1 GB model loads and runs within the app's
  memory.

### A7. Measure on the phone (CPU), against llama.cpp, same session

- `llama-bench`-style pp64/128/512 and tg64 for Q4_0 and Q4_K_M, cores 2–7, battery,
  cooled, order reversed in round 2.
- Decode against the bandwidth ceiling: bytes per token (from the file) / time per token, as
  for Qwen in Phase 7. Expect the ceiling to bind much more than at 0.5B.
- Energy per token and temperature over a 2-minute decode: a bigger model on a phone is a
  thermal question.
- Where the time goes (`simpleperf`): the conv layers should be nearly free next to the
  MLP; check.

Close: `results/phase-lfm3.md`, `IMPLEMENTATION.md` status, merge into `main`.

### A8. Later: the GPU (its own plan)

What the Vulkan path needs, so that the CPU work does not close doors:
- **Head size 128.** `gpu.rs` asserts `hs <= 64`; the attention shaders' shared-memory
  tiles and lanes are sized for 64.
- A conv shader with the 6-row state, the gate, QK-norm (fits `qkv_post.comp`), RoPE with a
  rotated prefix, and the sliding window in flash and decode attention (the window bounds
  the positions walked, which helps).
- Sandwich norms: two more RMSNorm dispatches per layer, or fused.

## 5. What the review will check

- **Correctness from references only**: HF transformers logits, `tokenizers`, numpy goldens,
  llama.cpp PPL/KL. No second LFM3 implementation of our own as an oracle.
- **The details that run but are wrong**: RoPE's zero tail and per-layer θ, the order of
  B/C/x in `in_proj`, the conv tap order (w[ch, 6] on the current token), the window edge,
  the gate applied before `o_proj`, and the post-norms applied before the residual add.
  Each has a golden that fails under the plausible wrong reading.
- **Old models unchanged**: Qwen/SmolLM2/stories15M tests and PPL to the digit after every
  step.
- **State on rewind**: KV ring, the conv state and positions all roll back together.
- **Measurement hygiene**: same-session pairs, battery, cooled, conditions in each raw file.
- **House rules**: `src/lib.rs` untouched on impl, no `cargo fmt`, comments in the style of
  the surrounding code.

## 6. References

- LiquidAI, LFM3-1.7B model card, `config.json`, `modeling_lfm3.py` (read 2026-11-02;
  pin the revision in A0).
- LFM2 technical report (Liquid AI, 2025): the gated short-convolution block LFM3 inherits.
- Su et al., *RoFormer* (2021): RoPE. Gemma 3 report (2025): per-layer-type RoPE bases and
  local/global attention. Qwen3 / OLMo 2: QK-norm. Gemma 2: sandwich norms.
- Beltagy et al., *Longformer* (2020) and Mistral 7B (2023): sliding-window attention.
- In this repo: `src/model/gguf.rs`, `src/model/transformer.rs`,
  `src/tokenizer/bytelevel.rs` (the Qwen2 regex), `src/kernels/neon.rs` (attention head
  sizes), `plans/flash-attention.md` (the plan format).
