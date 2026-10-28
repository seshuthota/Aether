# Phase 5 — SmolLM2-135M (GQA, safetensors, byte-level BPE)

Code: `format/json.rs` (hand-written JSON), `format/mmap.rs`,
`format/safetensors.rs`, `model/hf.rs` (Hugging Face Llama/Qwen2
checkpoints), `tokenizer/bytelevel.rs` (+ `unicode_tables.rs`, `nfc.rs`).
References: Hugging Face `transformers` (activations, generation) and
`tokenizers`, through `tools/hf_dump.py` and `tools/gen_unicode_tables.py`.

## What SmolLM2 adds over stories15M

- **Grouped-query attention**: 9 query heads share 3 key/value heads, so
  the KV cache is a third of the size.
- **Tied embeddings**: the output classifier reuses the token embedding.
- **Rotate-half RoPE** (θ = 100,000): Hugging Face pairs element `i` with
  `i + head_size/2`, where llama2.c pairs adjacent elements.
- **Byte-level BPE** from `tokenizer.json`: a GPT-2 pre-tokenizer regex,
  every byte mapped to a printable character, merges applied by rank.
- **safetensors + bf16**: weights memory-mapped, bf16/f16 widened to f32
  exactly.

## Verification

| Check | Result | Test |
| --- | --- | --- |
| Every layer's activations vs `transformers` | within 5.5e-5 relative error | `model::hf::tests` |
| Greedy generation vs `generate(do_sample=False)` | 48 tokens identical | `model::hf::tests` |
| Tokenizer vs HF `tokenizers` | 433 strings encode and decode identically (scripts, emoji, ZWJ sequences, flags, code, chat markup) | `tokenizer::bytelevel::tests` |
| JSON parser | escapes, surrogate pairs, error positions | `format::json::tests` |
| bf16 → f32 widening | exact | `format::safetensors::tests` |

These pass on the host and on the phone (85/85 on 2026-10-22).

## Speed

Measured with the threading work: `results/phase-4.md` (SmolLM2-135M
f32 39 tok/s on one prime core; Q4_0 275 tok/s on cores 2–7).

## SmolLM2-360M (item 5.6)

Same code path, larger shape: 15 query heads sharing 5 K/V heads, 32
layers, dim 960, hidden 2560, θ = 100,000. Its `tokenizer.json` is byte for
byte the 135M one, so the tokenizer check above covers it. Goldens:
`tools/hf_dump.py testdata/models/SmolLM2-360M testdata/smollm2-360m`
(2026-10-28); test `smollm2_360m_matches_transformers`.

On the host (2026-10-28): 48 greedy tokens identical to `transformers`;
worst relative error per activation 1.0e-4 (`resid_ffn`), 3.2e-5
(`attn_out`), 1.1e-5 (`ffn_out`), ≤ 6.7e-6 elsewhere. The largest is about
twice 135M's 5.5e-5, still a tenth of the 1e-3 bound; both models sum in
f32, and this one is deeper and wider.

On the phone (2026-10-28): the same test passes. Speed, cores 2–7, phone
charging (clock caps 1.79 / 1.96 GHz, so lower than the battery runs
elsewhere), quantized in memory from the safetensors
(`results/phase-6/raw/pending-checks/smollm2-360m-*.txt`):

| Format | weights | pp64 | pp256 | tg64 @ d0 | tg64 @ d256 |
| --- | --- | --- | --- | --- | --- |
| f32 | 1447 MB | 239 | 222 | 35.6 | 34.7 |
| Q8_0 | 385 MB | 471 | 385 | 124 | 94 |
| Q4_0 | 204 MB | 420 | 352 | 183 | 132 |

Peak RSS is 2.1 GB in every format: the safetensors path widens bf16 to f32
before quantizing, so the f32 copy exists during load. A GGUF file of the
quantized model would avoid it.
