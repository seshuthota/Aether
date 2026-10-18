# From-Scratch Android Inference Engine in Rust

**Complete Learning and Implementation Plan**

_Target: Snapdragon Android devices • Language: Rust • Approach: Everything from scratch in an empty project_

## 1\. Philosophy and Goals

**The objective is learning.** This is a long journey with a steep learning curve, taken to understand how LLM inference really works by building something this complex from scratch. A fast, usable engine at the end is a bonus, not the goal. We are not trying to beat llama.cpp or anything else.

This plan is designed for deep understanding rather than speed of completion. You will learn the concepts thoroughly before (and while) implementing each piece. Every component of the engine will be written from scratch in an empty Rust project. No reliance on existing inference engines for the core logic.

**The engine is built specifically for phones**, not for big GPUs. Phone hardware is small and constrained, and those constraints drive every design decision:

- **Memory bandwidth**, not compute, limits token generation, and phones have far less of it than desktop GPUs.
- **RAM is small and shared** with the OS and other apps; Android will kill a process that uses too much.
- **Thermals**: a phone has no fan. Performance in the first 10 seconds is not performance after 5 minutes.
- **Energy**: every token costs battery. Joules per token is a first-class metric.
- **Heterogeneous CPU cores** (prime / performance / efficiency) that behave very differently.
- **CPU first**: every phone has ARM NEON; GPU and NPU paths come later, if at all.

Performance targets in this plan are **measurements to understand, not gates to pass**. A phase is complete when the engine is correct and you can explain the numbers you measured: where the time goes, and what limits it. If a kernel reaches 40% of bandwidth and you can explain why, that's a finished phase.

Existing open-source implementations (llama2.c, Hugging Face `transformers`, llama.cpp) are used **only as external references** — sources of golden outputs to check against and performance numbers to compare with. We never write a second implementation of our own just to test the first one.

Primary goals:

- Understand why inference is hard on mobile (memory bandwidth, cache, thermal limits, energy, heterogeneous cores, decode vs prefill).
- Write high-quality ARM NEON kernels that approach the practical limit of the hardware (for decode, that limit is memory bandwidth).
- Build a correct, measurable decoder-only transformer engine that runs on a real Snapdragon phone.
- Establish rigorous testing and benchmarking habits from the very first kernel.
- Create a foundation that can later be extended to GPU (Vulkan) and potentially NPU without rewriting the architecture.

## 2\. Prerequisites

Before starting the phases:

- Comfortable writing Rust (ownership, traits, unsafe basics). If needed, complete the official Rust book chapters on ownership and smart pointers.
- Android device with Snapdragon chip + developer mode + USB debugging.
- Android NDK installed, plus `adb` on the host.
- Basic understanding of neural networks (what a linear layer, attention, and residual connection are).
- Host-side reference tooling (not part of the engine):
  - Python with `torch`, `transformers`, `safetensors`, `tokenizers`, `numpy`.
  - A clone of [llama2.c](https://github.com/karpathy/llama2.c) (built on host).
  - A clone of [llama.cpp](https://github.com/ggml-org/llama.cpp), built on host **and** cross-compiled for the phone (used for golden logits, perplexity/KL numbers, GGUF conversion, and `llama-bench` comparisons).

## 3\. Correctness Strategy: External References

Every correctness check compares our engine against output produced by an established open-source implementation. Golden data is generated on the host by scripts in `tools/` and stored in `testdata/`.

| **What is being checked**                        | **Reference**                                                                  | **How**                                                                                                    |
| ------------------------------------------------ | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Individual kernels (matmul, RMSNorm, softmax, …) | numpy / PyTorch                                                                | Script generates random inputs + expected outputs; Rust test loads both and compares                       |
| stories15M forward pass + generation             | llama2.c (`run.c`)                                                             | Patch `run.c` with a few `fwrite` calls to dump per-layer activations and logits as raw f32                |
| SmolLM2 / Qwen2.5 forward pass                   | HF `transformers`                                                              | Forward hooks dump per-layer activations and logits to a `.safetensors` file (reuses our own parser)       |
| Tokenizer                                        | llama2.c tokenizer / HF `tokenizers`                                           | Encode/decode a few thousand strings (incl. Unicode, emoji, whitespace edge cases); require exact match    |
| Quantized formats + accuracy                     | llama.cpp (`llama-perplexity`, `--kl-divergence`)                              | Same GGUF file, same text: our perplexity and KL vs f32 should match llama.cpp's within noise              |
| Performance comparison (learning tool)           | llama.cpp (`llama-bench`) on the same phone                                    | Same model, quant type, threads, thermal conditions — used to find and study techniques, not as a target   |

Tolerances: report max absolute error and max relative error for every comparison. As a starting point, f32 kernels vs numpy should agree to ~1e-5 relative; full-model f32 logits vs reference to ~1e-3 relative (summation order differs); greedy-decoded tokens should match exactly for at least the first few dozen tokens.

Dump at **every** layer boundary (after embedding, after each norm, after attention, after MLP, final logits), not just the final output — when something breaks, the first diverging layer tells you where the bug is.

## 4\. Project Setup (Phase 0)

### What to Learn

- Cross-compilation for Android (`aarch64-linux-android` target).
- How to measure time accurately on device (`std::time::Instant` uses `CLOCK_MONOTONIC`; good enough for microbenchmarks with enough iterations).
- Memory layout of tensors (row-major vs column-major, strides, alignment).

### What to Implement

1. `git init`, then an empty Cargo project with a library and a binary.
2. `.cargo/config.toml` pointing the `aarch64-linux-android` linker at the NDK clang (or use `cargo-ndk`).
3. A cargo **runner** script (`target.aarch64-linux-android.runner = "scripts/adb-run.sh"`) that `adb push`es the binary to `/data/local/tmp` and runs it. This makes `cargo run`, `cargo test`, and `cargo bench` execute **on the phone** with no extra steps.
4. Simple `Tensor` struct (owned data, shape, dtype placeholder).
5. Basic timing harness (warm-up, N iterations, median + min + spread).
6. `tools/` directory for host-side Python reference scripts; `testdata/` for golden files (keep large files out of git, regenerate via script).
7. `results/` log (markdown) for benchmark numbers, committed alongside the code that produced them.
8. `notes/` directory for the learning journal (see section 14).

### Success Criteria

`cargo test --target aarch64-linux-android` runs a trivial test on the phone and reports the result on the host terminal.

## 5\. Phase 1 — Hardware & Memory Model

### What to Learn

- Snapdragon core topology: prime / performance / efficiency cores, their cache sizes, and which CPU numbers they map to (`/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq`).
- Cache hierarchy (L1, L2, L3/SLC) and DRAM bandwidth limits.
- Why LLM decode is almost always memory-bandwidth bound (every weight is read once per token).
- Why a **single core usually cannot saturate DRAM bandwidth** — and therefore why decode needs multiple threads.
- Importance of alignment (16-byte for NEON, 64-byte for cache lines).
- Roofline model (arithmetic intensity vs bandwidth).
- Profiling with `simpleperf` (Android's `perf`).
- Power and energy on a phone: reading battery current and voltage from `/sys/class/power_supply/battery/` (`current_now`, `voltage_now`; units and sign vary by device, and the phone must be unplugged). Energy = power × time.
- Thermal behaviour: how the frequency governor lowers clock speeds as the SoC heats up (`/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`, `/sys/class/thermal/`).

### What to Implement

1. Aligned allocator wrapper.
2. Bandwidth benchmark (copy, scale, add, and a pure **read** / sum reduction — the read case is what matters for decode).
3. Working-set sweep: run the read benchmark from ~4 KB to ~256 MB and plot GB/s vs size; the drops reveal L1/L2/L3/DRAM boundaries.
4. Run the sweep pinned to (a) one prime core, (b) one efficiency core, (c) all performance cores in parallel (use `sched_setaffinity` via `libc`).
5. A small power/thermal monitor that samples battery current, voltage, core frequencies and temperatures in a background thread while a benchmark runs, and reports average power and energy (joules) for the run.

### Testing & Benchmarking

- Record single-core vs multi-core DRAM read bandwidth and compare against the SoC's theoretical bandwidth.
- Run a long (5 min+) test while logging core frequencies and temperature; note how quickly throttling kicks in and how far bandwidth drops.
- Compare energy per GB read on a prime core vs an efficiency core. (Faster isn't always more efficient — this matters for which cores you choose later.)
- Record benchmark conditions every time: which cores, screen on/off, charging or not, starting temperature (`/sys/class/thermal/`).

### Success Criteria

A table in `results/` with: L1/L2/L3 effective bandwidth, single-core DRAM read bandwidth, all-core DRAM read bandwidth, throttled sustained bandwidth, and energy per GB on prime vs efficiency cores. These numbers are the yardstick for every later kernel.

## 6\. Phase 2 — Scalar End-to-End on a Real Tiny Model (stories15M)

Get a real, trained model producing correct text **before** optimizing anything. This gives a correctness oracle that runs on the phone, a baseline number, and a real workload to profile.

### What to Learn

- Full decoder-only layer structure: RMSNorm → attention (Q/K/V projections, RoPE, scaled dot-product, softmax, output projection) → residual → RMSNorm → SwiGLU MLP → residual.
- RMSNorm mathematics and numerical stability.
- RoPE (Rotary Position Embeddings) — how it encodes position. Note the **two layout conventions**: adjacent pairs `(x[2i], x[2i+1])` vs halves `(x[i], x[i + d/2])` ("rotate_half"). llama2.c uses adjacent pairs; HF `transformers` uses rotate_half. Mixing them up is the classic "runs fine, outputs garbage" bug.
- Softmax and its numerical issues (max subtraction).
- SiLU / SwiGLU activation.
- KV cache: why it exists and how it turns per-token cost from O(n²) to O(n).
- Autoregressive generation loop; greedy decoding first, then temperature / top-k / top-p sampling.

### What to Implement

1. Loader for the llama2.c model format (a small header of ints followed by raw f32 weights) — `stories15M.bin` from llama2.c's TinyStories models.
2. Loader + encoder/decoder for llama2.c's `tokenizer.bin` (32k vocab, score-based BPE merges — a gentle introduction to tokenization).
3. Straightforward scalar implementations of every op, written as the engine's first version (not as a separate reference).
4. Pre-allocated KV cache.
5. Generation loop with greedy decoding, then sampling.

### Testing & Benchmarking

- Patch llama2.c's `run.c` to dump per-layer activations and logits for a fixed prompt; compare layer by layer.
- Greedy decoding must produce the **exact same token sequence** as llama2.c for the same prompt.
- Tokenizer encode/decode must match llama2.c exactly on a test corpus.
- Record tokens/sec on the phone (scalar, single thread) — this is your baseline.
- Run `simpleperf` and record where time goes (expect matmul to dominate).

### Success Criteria

Coherent TinyStories text on the phone, token-for-token identical to llama2.c under greedy decoding.

## 7\. Phase 3 — NEON Kernels (f32)

### What to Learn

- GEMM vs GEMV. Decode is GEMV (one token × weight matrix) and is bandwidth-bound; prefill is GEMM and can be compute-bound.
- Tiling and blocking to fit in cache (matters for GEMM / prefill).
- ARM NEON basics: 128-bit registers, `vld1q_f32`, `vfmaq_f32`, horizontal reductions, multiple accumulators to hide FMA latency.
- `std::arch::aarch64` in Rust: keeping `unsafe` contained, `#[target_feature(enable = "...")]`, and runtime detection with `is_aarch64_feature_detected!`.
- Check which NEON intrinsics are stable on your toolchain: base NEON is stable, but some extensions (dotprod, i8mm, fp16) have historically been nightly-only. If one you need is unstable, decide early between nightly and `core::arch::asm!`.

### What to Implement

1. NEON f32 GEMV (start 4-wide, then 4–8 independent accumulators, then unrolling).
2. NEON f32 GEMM for prefill (simple register-blocked micro-kernel + cache blocking).
3. NEON versions of RMSNorm, softmax, SiLU/SwiGLU, RoPE where profiling shows they matter.
4. Microbenchmark harness reporting **GB/s** (for GEMV) and **GFLOPS** (for GEMM).
5. Swap the new kernels into the Phase 2 engine.

### Testing & Benchmarking

- Correctness: each kernel against golden numpy outputs (absolute + relative error), across shapes: square, tall, wide, GEMV-like, and sizes that are **not** multiples of the vector width (tail handling bugs live here).
- End-to-end: stories15M must still match llama2.c's tokens exactly and its logits within tolerance.
- GEMV performance: report effective bandwidth as a **percentage of the Phase 1 single-core DRAM read bandwidth**. Aim for ≥ 70%, but use the number to learn: if it's lower, find out why (`simpleperf`, instruction counts, cache misses).
- GEMM performance: report GFLOPS vs the core's theoretical peak FMA throughput.

### Success Criteria

stories15M still produces identical greedy tokens; the tokens/sec speedup over the Phase 2 baseline is recorded; and you can explain what limits your GEMV — how close it gets to measured single-core bandwidth, and where the rest goes.

## 8\. Phase 4 — Threading and Core Placement

Threading is required for decode, not just prefill, because one core cannot saturate DRAM bandwidth (see Phase 1).

### What to Learn

- Splitting GEMV across threads by output rows (no reductions needed between threads).
- Heterogeneous cores: a thread on an efficiency core makes every parallel matmul wait for it. Pin threads to prime + performance cores.
- Thread pool design: persistent workers, spin-then-park synchronization, minimal per-op overhead (a decode step issues hundreds of small ops).
- False sharing and cache-line alignment of per-thread data.

### What to Implement

1. Custom persistent thread pool with core pinning (`sched_setaffinity`). (Write it yourself for learning; `rayon` is fine to benchmark against.)
2. Parallel GEMV and GEMM.
3. Thread-count and core-set configuration flags.

### Testing & Benchmarking

- Results must be identical to single-threaded (row-split GEMV doesn't change summation order).
- Sweep thread counts and core sets; plot tokens/sec vs threads.
- Multi-threaded GEMV bandwidth as a percentage of Phase 1 **all-core** bandwidth.
- Energy per token for different core sets (e.g. prime + performance vs performance only). The fastest configuration is often not the most energy-efficient one.

### Success Criteria

You can explain the tokens/sec vs thread-count curve and the energy-per-token vs core-set trade-off, and how close multi-threaded GEMV gets to measured all-core bandwidth.

## 9\. Phase 5 — Modern Architecture (SmolLM2-135M / 360M)

Step up to a model with the features every current small model has, while staying small enough to iterate fast.

### What to Learn

- **Grouped-Query Attention (GQA)**: fewer K/V heads than Q heads; shrinks the KV cache and changes indexing.
- Tied embeddings (output projection reuses the embedding matrix).
- Safetensors format (8-byte header length + JSON header + raw tensor bytes) and bf16 → f32 conversion.
- HF-style RoPE (rotate_half) and reading `config.json` for hyperparameters (RoPE theta, norm epsilon, head counts).
- Byte-level BPE tokenization (GPT-2 style): byte-to-unicode mapping, pre-tokenization, merge ranks, special tokens.
- Memory mapping (`mmap`) for weight files.

### What to Implement

1. Safetensors parser (from scratch) + `mmap` loader.
2. `config.json` parsing → model hyperparameters (minimal hand-written JSON parser, or `serde_json` if you decide JSON parsing isn't part of the learning goal).
3. GQA attention.
4. Byte-level BPE tokenizer from `tokenizer.json`.
5. Generalize model code so stories15M and SmolLM2 run through the same code path driven by config.

### Testing & Benchmarking

- Golden per-layer dumps from HF `transformers` (forward hooks → `.safetensors`); compare layer by layer.
- Greedy tokens match HF `generate(do_sample=False)` for a fixed prompt.
- Tokenizer exactly matches HF `tokenizers` on a corpus including Unicode, emoji, code, and odd whitespace.

### Success Criteria

SmolLM2 generates coherent text on the phone matching HF greedy output; tokenizer matches HF exactly.

## 10\. Phase 6 — Quantization

### What to Learn

- Why quantization is essential on mobile: decode speed is roughly proportional to bytes of weights read per token.
- Block-wise quantization: llama.cpp's `Q8_0` (blocks of 32 int8 values + one f16 scale) and `Q4_0` (blocks of 32 4-bit values + one f16 scale, value = (q − 8) × scale). Matching llama.cpp's block layouts lets you load its GGUF files and compare against it directly.
- Two ways to compute a quantized matmul — make this choice explicitly:
  - (a) Dequantize weights to f32 inside the kernel, then FMA. Simpler; often compute-bound on dequant.
  - (b) Quantize the **activation vector** to int8 blocks on the fly, then use integer dot products (`vdotq_s32` / `sdot`), and on newer Snapdragons `smmla` (i8mm) for GEMM. Much faster; this is what llama.cpp does.
- Weight repacking / interleaving for SIMD-friendly access.
- Measuring quantization quality: perplexity and KL divergence of logits vs the f32 model.

### What to Implement

1. Offline quantizer tool (Rust binary) producing `Q8_0` and `Q4_0` weights.
2. `Q8_0` GEMV, approach (a) first, then approach (b) with dotprod.
3. `Q4_0` GEMV (nibble unpacking with NEON).
4. Quantized GEMM for prefill (dotprod, then i8mm if your SoC supports it).
5. Perplexity + KL-divergence evaluation mode over a fixed text file.

### Testing & Benchmarking

- Kernel correctness: quantized GEMV output vs golden output computed by a numpy script that implements the same block format.
- Accuracy: run `llama-perplexity` (with `--kl-divergence`) on the same model, same quant type, same text; our perplexity and KL should match llama.cpp's within noise.
- Performance: tokens/sec and GB/s vs Phase 1 bandwidth (bytes counted at the quantized size); memory footprint vs f32.

### Success Criteria

`Q8_0` and `Q4_0` perplexity/KL matching llama.cpp on the same text, with quantized GEMV still at a high fraction of measured bandwidth.

## 11\. Phase 7 — Real Small Model (Qwen2.5-0.5B) + GGUF

### What to Learn

- GGUF format: header, key-value metadata, tensor info, alignment, data section.
- Qwen2 architecture specifics: biases on Q/K/V projections, tied embeddings, RoPE theta = 1e6, GQA with very few KV heads.
- GGUF weight conventions: llama.cpp's converter **permutes Q/K weights for Llama-architecture models** (so it can use adjacent-pair RoPE) but not for Qwen2 (rotate_half / "NeoX" style). Your RoPE choice must follow the file's architecture.
- Qwen tokenizer: ~151k vocab byte-level BPE with a regex-based pre-tokenizer using Unicode categories. Writing a hand-coded pre-tokenizer that reproduces the regex is a sub-project of its own — budget time for it.
- Chat templates and special tokens.
- Full generation metrics: tokens/sec (prefill and decode separately), time-to-first-token, peak memory (RSS).
- Android memory pressure: the low-memory killer, and why `mmap`ed read-only weights are friendlier than heap copies.

### What to Implement

1. GGUF parser (from scratch) + `mmap` loader, supporting f16/bf16/`Q8_0`/`Q4_0` tensors.
2. Qwen2 architecture support (QKV bias, RoPE variant chosen from metadata).
3. Qwen tokenizer + chat template.
4. Comprehensive benchmark suite (prefill tok/s at several prompt lengths, decode tok/s at several context lengths, TTFT, peak RSS, joules per token, and a sustained 5-minute run showing tokens/sec over time as the phone heats up).

### Testing & Benchmarking

- Convert the model with llama.cpp's `convert_hf_to_gguf.py` (and `llama-quantize` for `Q4_0`), so our engine and llama.cpp read the **same file**.
- Golden per-layer dumps from HF `transformers` for the f16/bf16 model; greedy tokens match.
- Perplexity/KL matches llama.cpp for each quant type.
- Tokenizer exactly matches HF `tokenizers`.
- `llama-bench` on the same phone with the same quant, threads, and thermal state, as a learning tool: where there's a large gap, profile both engines and read the relevant llama.cpp code to understand which technique explains it.

### Success Criteria

Usable chat-like generation on device, with numbers documented (including energy and sustained performance), and a written explanation of where your engine's time and energy go.

## 12\. Phase 8 — Android Integration

### What to Learn

- Building a Rust `cdylib` for Android (`cargo-ndk` makes this easy).
- C ABI design (opaque handles, explicit create/destroy, error codes, no panics across FFI — use `catch_unwind`).
- JNI basics and calling a native `.so` from Kotlin.
- Streaming tokens back to the UI thread; running inference off the main thread.

### What to Implement

1. Clean C ABI around the engine (load model, tokenize, generate with a token callback, free).
2. Thin JNI layer + minimal Kotlin app that loads a model from app storage and streams generated text.

### Success Criteria

An installable APK that loads Qwen2.5-0.5B `Q4_0` and streams a chat response, with tokens/sec matching the command-line binary.

## 13\. Later Phases (After Core is Solid)

- Better quantization (K-quant style formats, importance matrices).
- i8mm-specific weight repacking for faster prefill.
- Flash-decoding / tiled attention for long contexts.
- Vulkan Compute kernels for matmul and attention (Adreno GPU).
- Speculative decoding or other advanced techniques (optional).
- NPU exploration via Hexagon-MLIR or QNN (research-level).

To keep these possible without a rewrite: keep model/graph code separate from kernel code behind a small set of op functions (`matmul`, `rmsnorm`, `rope`, `attention`, …). Don't design a full backend abstraction until you actually start on Vulkan.

## 14\. Testing & Benchmarking Methodology (Ongoing)

Apply these practices from Phase 0 onward:

- Correctness first: every kernel and every model is checked against an **external reference** (numpy, llama2.c, HF `transformers`, llama.cpp) via golden files, with automated comparison in `cargo test` running on the phone.
- Compare at every layer boundary, not just final logits.
- Test odd sizes (non-multiples of vector width and block size) explicitly.
- Warm-up runs before timing.
- Report median and spread of multiple runs; note variance.
- Measure on the real device with pinned cores and recorded thermal conditions.
- Track hardware metrics (GB/s as % of measured bandwidth, GFLOPS), application metrics (tokens/sec for prefill and decode, TTFT, peak RSS), and mobile metrics (joules per token, sustained tokens/sec after several minutes).
- Profile with `simpleperf` before optimizing anything.
- Keep `results/` in git so every number is tied to the commit that produced it.

**Learning journal.** At the end of each phase, write `notes/phase-N.md` in your own words:

- The concepts the phase covered, explained as if teaching someone else.
- What you expected vs what you measured, and why they differed.
- Bugs that took longest to find, and what they taught you.
- Open questions to come back to.

If you can't explain something in the journal, that's the signal to dig deeper before moving on — that's the real success criterion for every phase.

## 15\. Recommended Order and Time Allocation

Do not skip ahead. The early phases build the intuition required for later ones.

| **Phase** | **Focus**                                        | **Rough Effort** |
| --------- | ------------------------------------------------ | ---------------- |
| 0–1       | Setup + on-device testing + memory model         | 1–2 weeks        |
| 2         | Scalar end-to-end on stories15M                  | 2–3 weeks        |
| 3         | NEON f32 kernels                                 | 2–4 weeks        |
| 4         | Threading + core placement                       | 1–2 weeks        |
| 5         | GQA + safetensors + BPE tokenizer (SmolLM2)      | 2–4 weeks        |
| 6         | Quantization (Q8_0, Q4_0, dotprod/i8mm)          | 3–6 weeks        |
| 7         | GGUF + Qwen2.5-0.5B + benchmark suite            | 3–5 weeks        |
| 8+        | Android app + later phases                       | Ongoing          |

## 16\. Final Notes

This is a long journey with a steep learning curve, and the understanding you build along the way is the point. The biggest learning happens in Phases 2–4 (a working model, then making it run well on small, constrained hardware). Resist the temptation to optimize before you have a correct end-to-end model and a profile showing where the time goes. The timeline is a rough guide, not a deadline: take longer wherever the learning is.

When you finish Phase 7 you will understand every major piece of a from-scratch inference engine built for phones: memory bandwidth, thermals, energy, heterogeneous cores, quantization and kernels. A working engine on your Snapdragon phone is the bonus. That knowledge transfers extremely well to using or contributing to existing engines later.

_Next step: Start Phase 0 and Phase 1. When you have on-device `cargo test` working and the bandwidth table filled in, move to the stories15M end-to-end build and refine the later phases based on what you learned._
