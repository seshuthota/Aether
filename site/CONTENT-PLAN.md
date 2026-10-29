# Content plan: Phases 2–8

Phases 0–6 and 8 have their pages, Phase 7 its overview and seven topic
pages, and the work after Phase 8 (Q4_K_M files, speculative decoding, the
GPU) has its own section, `beyond/`. The engine is built and verified through Phase 8
and beyond, so every topic below has real code to point at and real numbers
measured on this phone. This is the list of pages still to write, in
reading order (some entries below are now written; check `js/pages.js`).

Each entry: **what the reader learns** · *interactive idea* · `where it
lives in the engine` / measured numbers to use. ★ marks pages that would
make good films (plan with `3b1b-explainer`, build with `manimgx`).

Numbers come from `results/phase-N.md`; anything not measured on this phone
is labelled an illustration.

---

## Start here (additions)

1. **★ Life of a token** — one keypress in the app to one word on screen,
   through every part of the engine: tokenizer → embedding → 24 layers →
   logits → sampler → UTF-8 → screen. The capstone page; every step links to
   its own page below. · *A clickable pipeline: each stage shows its input,
   output and time share.* · `android/` → `ffi.rs` → `session.rs` →
   `model/transformer.rs`; prefill 35 tokens in ~200 ms, decode ~100 tok/s.
2. **Code map** (written: `start/code-map.html`, as tables; the hover tree is still open) — which file does what, in the order a token meets them. ·
   *A tree of `src/` where hovering a file shows its one-line job and the
   page that explains it.*
3. **How we know it's right** (written: `start/right.html`) — the external-reference rule: numpy,
   llama2.c, Hugging Face, llama.cpp; golden files; why a second copy of our
   own code proves nothing. · `tools/`, `testdata/`, `src/testutil.rs`; 85
   tests on the phone.

## Phase 2 · A whole model in plain Rust (stories15M)

1. **What's inside a model file** — llama2.c's 7-number header, the weight
   order, why the classifier can share the embedding. · *A byte map of
   stories15M.bin: click a region to see which matrix it is and its shape.*
   · `model/llama2c.rs`.
2. **Text to tokens (SentencePiece-style BPE)** — vocabulary, scores, the
   merge loop, BOS. · *Type a word and step through the merges.* ·
   `tokenizer/llama2c.rs`; checked against llama2.c's own encoder.
3. **Embeddings** — a token id is just a row number. · *Pick a token, see its
   288 numbers as a strip of colours.*
4. **RMSNorm** — why activations are rescaled before every block. · *Drag a
   vector's values, watch the normalised output keep its direction.*
5. **★ Attention, step by step** — query, keys, values; scores; dividing by
   √d; softmax; the weighted sum. · *A 5-token toy: edit the query, watch
   the score row and the output move.* · `kernels/scalar.rs::attention_head`.
6. **Softmax and overflow** — why the max is subtracted first. · *The same
   scores with and without it; one produces `inf`.*
7. **★ RoPE: position as rotation** — pairs of numbers rotated by an angle
   that grows with position, fast pairs and slow pairs; the dot product then
   depends only on distance. · *Two arrows rotating as you slide the
   position; their dot product stays fixed when both move together.* ·
   `rope_adjacent`.
8. **The MLP and SwiGLU** — gate, up, down; the SiLU curve. · *Plot SiLU vs
   ReLU; feed a value through gate × up.*
9. **The residual stream** — `x += block(x)`: each layer adds a correction to
   one running vector.
10. **★ The KV cache** — why keys and values of past tokens are kept, and what
    that costs. · *Calculator: layers × positions × KV width × 2 × bytes, for
    stories15M, SmolLM2 and Qwen2.5 at any context length.* ·
    `State::key_cache`.
11. **Choosing the next token** — greedy, temperature, top-k, top-p; the
    random number generator. · *A toy distribution with sliders for each
    knob.* · `model/sampler.rs` (xorshift, same as llama2.c, so runs are
    bit-identical).
12. **Bit-identical, and why that's unusual** — float addition isn't
    associative; matching llama2.c bit for bit needs the same order of
    operations. · *Add the same 6 numbers in different orders; the last
    digits change.* · `results/phase-2.md`.
13. **First speed and profile** — 88.6 tok/s on a prime core; the profile
    says matmul is nearly everything.

## Phase 3 · Using the vector unit (NEON)

1. **SIMD and NEON registers** — 4 floats per register, `vld1q`, `vfmaq`. ·
   *Animated lanes: one instruction, four multiplies.* · `kernels/neon.rs`.
2. **★ Dependency chains: latency vs throughput** — why one accumulator
   wastes the FMA unit and four don't. · *A pipeline timeline: 1 chain vs 4
   chains, cycles ticking.* · measured 5.3 → 17.8 → 41.6 → 53.5 GB/s.
3. **GEMV in three steps** — the measured table, including the surprise: the
   4-row kernel loses on short rows of big matrices (29 vs 41 GB/s). ·
   `results/phase-3.md`.
4. **Tails: where SIMD bugs live** — sizes that aren't multiples of 4/8/16,
   and why the test goldens use them.
5. **Prefill vs decode** — the same matrix, multiplied by 1 vector or by T
   vectors; arithmetic intensity grows with T. · *Roofline dot that moves
   right as you add prompt tokens.*
6. **★ GEMM: tiles and register blocking** — the 4×4 micro-kernel: 8 loads,
   16 FMAs; cache blocking. · *Tile animation over a matrix.* ·
   `neon::gemm_rows`.
7. **Testing floats against numpy** — float64 references, relative error,
   bounds that grow with √n.
8. **Dispatch** — `cfg(target_arch)`, runtime feature detection, scalar
   fallback on the PC; intrinsics vs `asm!` (`fcvtl`, and `smmla` next).
9. **★ Measuring a change on a phone that won't sit still** — the NEON
   attention story: a single run said scalar attention halved decode; an
   alternating A/B said it cost 2.5 ms per token. Caps, heat, a video playing
   in another app. · *Two noisy series; show how alternation cancels drift.*
   · `results/phase-7.md`, "Attention: A/B".

## Phase 4 · Many cores

1. **Why threads help a memory-bound job** — one core reaches 61 GB/s, but
   decode also has compute and serial parts.
2. **A thread pool from scratch** — persistent pinned workers, an epoch
   counter, spin then park; why not spawn per matmul (≈200 matmuls per
   token). · *Timeline of one job handoff.* · `pool.rs`.
3. **Splitting work without changing the answer** — rows per thread, each
   sum in one thread, so results are bit-identical; 64-byte blocks against
   false sharing.
4. **★ Fast and slow cores: when 8 is slower than 2** — equal shares make
   prime cores wait for performance cores at every matmul; dynamic blocks
   fix it: 98 → 228 tok/s. · *Simulator: 2 fast + 6 slow workers, static vs
   dynamic split, watch the idle time.* · `results/phase-4.md`.
5. **Amdahl's law and the serial part** — norms, adds and sampling run on one
   thread; put that thread on the fastest core. · *Amdahl slider with this
   engine's serial fraction.*
6. **Picking cores** — 2–7 vs 6–7 vs 0–7, and energy per token per set
   (once the clean sweep is done).
7. **Rust concurrency used here** — atomics, Acquire/Release ordering,
   `unsafe impl Send` for a shared output pointer, and the reasoning that
   makes it sound.

## Phase 5 · Real model files (SmolLM2)

**Written (2026-10-29)**, as four pages: `phase-5/safetensors.html` (1–3),
`config.html` (4–6), `bytelevel.html` (7–8), `matching-hf.html` (9). Still
open from the ideas below: the interactive hex view, the bit editor, the
MHA/GQA/MQA toggle and the step-by-step encoder.

1. **safetensors** — 8-byte length, a JSON header, raw tensors; parsing JSON
   by hand. · *Hex view of a real header with the fields highlighted.* ·
   `format/safetensors.rs`, `format/json.rs`.
2. **★ mmap: loading without copying** — the file becomes memory; pages are
   read the first time they're touched (why the app's first answer is
   slower: 262 vs 205 ms prefill). · `format/mmap.rs`, `results/phase-8.md`.
3. **f32, f16, bf16** — sign, exponent, mantissa; why bf16 → f32 is exact.
   · *A bit editor for each format.*
4. **Config-driven models** — `config.json` decides the shapes; one code
   path for three model families.
5. **Grouped-query attention** — 9 query heads sharing 3 KV heads; the KV
   cache shrinks 3×. · *Head-mapping diagram, toggled between MHA, GQA and
   MQA.*
6. **Two RoPE layouts** — adjacent pairs vs rotate-half: same rotation,
   different pairing, and what converters do to the weights.
7. **★ Byte-level BPE** — bytes mapped to printable characters, the
   pre-tokenizer regex, merges by rank. · *Encode "Hello 👋 world" step by
   step; see the emoji become several tokens.* · `tokenizer/bytelevel.rs`;
   433 strings identical to HF.
8. **Unicode for tokenizers** — code points, UTF-8 bytes, letter/number
   categories, and generated tables. · `tools/gen_unicode_tables.py`.
9. **Checking against `transformers`** — hooks that dump every layer,
   tolerance 5.5e-5, and the gotcha: a repetition penalty hidden in
   `generation_config.json`.

## Phase 6 · Quantization

1. **Why quantize** — the decode ceiling again: 538 → 143 → 76 MB for
   SmolLM2, and what that does to tokens per second.
2. **★ Block quantization** — Q8_0 (32 int8 + one f16 scale = 34 bytes),
   Q4_0 (32 nibbles, offset 8 = 18 bytes). · *Type or randomise 32 numbers,
   quantize, see each one's error.* · `quant.rs`; byte-identical to llama.cpp.
3. **Rounding to f16** — round-to-nearest-even, subnormals, and the bug a
   debug build caught (an overflow release builds hid).
4. **Two ways to multiply quantized weights** — dequantize to f32, or
   quantize the activation and use integer dots: 12 → 32 GB/s.
5. **`sdot`** — four int8 products summed into one int32 lane per
   instruction. · *Lane diagram.* · `kernels/neon_quant.rs`.
6. **Why Q4_0 is slower per byte** — unpacking nibbles costs instructions;
   on one core Q4_0 is compute-bound (12–15 GB/s).
7. **★ Perplexity** — the average surprise of the model on real text,
   exponentiated; chunks of 512, scoring the second half. · *A 5-word
   sentence with the model's probabilities; compute PPL by hand.* ·
   `eval.rs`.
8. **KL divergence and top-token agreement** — how far the quantized model's
   choices drift from f32. · *Two distributions with a slider; KL updates.*
   · our numbers vs llama.cpp's in `results/phase-6.md`.
9. **Small models suffer more** — Q4_0 costs SmolLM2-135M ×1.41 perplexity,
   Qwen2.5-0.5B ×1.21.

## Phase 7 · A real chat model (Qwen2.5, GGUF)

1. **GGUF** — header, typed metadata, tensor table, alignment; one file
   holds weights, tokenizer and chat template. · *Hex/metadata explorer.* ·
   `format/gguf.rs`.
2. **Reading the architecture from metadata** — llama vs qwen2, QKV biases,
   RoPE θ = 1,000,000.
3. **Unicode normalization (NFC)** (written: `phase-7/nfc.html`; the
   string widget is still open) — "é" as one code point or two; why a
   tokenizer must pick one. · *Type a string, see both forms' bytes.* ·
   `tokenizer/nfc.rs`; 17,253 cases identical to Python.
4. **Chat templates** (written: `phase-7/chat.html`) — ChatML, special tokens, stop tokens; why an instruct
   model answers only when asked in its format. · `chat.rs`.
5. **Benchmarking like llama-bench** (written: `phase-7/benchmarking.html`) — prefill vs decode, depth, time to
   first token, peak memory, synthetic tokens.
6. **★ Where the time goes** (written: `phase-7/time.html`, with item 8's `smmla` and repacking) — decode moves ~36 GB/s of weights; prefill is
   compute; why llama.cpp is 2–6× faster at prefill (repacked weights,
   `smmla`). · `results/phase-7.md`.
7. **Heat and energy** (written: `phase-7/heat.html`, with the plot) — 5 minutes of decoding: 112 → 65–75 tok/s at
   84.5 °C, 3.5 W, 0.041 J per token. · *Plot of the sustained trace.*
8. **`smmla` and weight repacking (preview)** — a 2×8 by 8×2 int8 block
   product per instruction; the next speed-up.

## Phase 8 · On the phone, in an app

**Written (2026-10-29)**, as two pages: `phase-8/c-abi.html` (1) and
`app.html` (2, 3, the engine thread from 4, and 5). Still open: APK
packaging (`jniLibs`, 16 KB alignment), model storage and `run-as`, and the
app's own memory (not measured).

1. **A C interface** — why C is the common language; opaque handles, error
   codes, and why a Rust panic must never cross into C (`catch_unwind`). ·
   `ffi.rs`, `include/aether.h`.
2. **JNI** — Kotlin → C → Rust and back; byte arrays instead of Java's
   "modified UTF-8"; calling back into Kotlin per token. ·
   `aether_jni.c`, `Aether.kt`.
3. **Streaming text** — a character can be split across tokens; a decoder
   that holds back incomplete bytes. · *Feed an emoji byte by byte.* ·
   `Utf8Stream`.
4. **Android basics for engine people** — APK, `jniLibs`, app storage,
   `run-as`, one engine thread, keeping the screen awake.
5. **Memory on a phone** — 348 MB peak for Qwen2.5 Q4_0; mapped pages and
   the low-memory killer.

---

## After Phase 8

`beyond/q4km.html` (Q4_K_M: Q5_0, Q4_K, Q6_K, accuracy, speed) is written,
alongside the speculative and GPU pages.

## Glossary terms to add

Phase 2: checkpoint, BOS/EOS, BPE, merge, embedding, RMSNorm, residual
stream, query/key/value, head, softmax, RoPE, SiLU, SwiGLU, temperature,
top-k, top-p. Phase 3: intrinsic, lane, accumulator, tail, micro-kernel,
tiling, feature detection. Phase 4: thread pool, spin-wait, park, load
balancing, static/dynamic scheduling, Amdahl's law, memory ordering.
Phase 5: safetensors, JSON, mmap, f16, GQA, tied embeddings, pre-tokenizer,
byte-level, code point, UTF-8. Phase 6: block, scale, nibble, dequantize,
sdot, perplexity, KL divergence, nats. Phase 7: GGUF, metadata, NFC,
chat template, ChatML, special token, TTFT, peak RSS, repacking, smmla,
i8mm. Phase 8: C ABI, opaque handle, panic, JNI, APK, callback.

## Suggested films (★ above), in priority order

1. Attention, step by step · 2. The KV cache · 3. Life of a token ·
4. Dependency chains · 5. Block quantization · 6. When 8 cores are slower
than 2 · 7. Perplexity · 8. RoPE · 9. Byte-level BPE · 10. Measuring on a
phone that won't sit still.
