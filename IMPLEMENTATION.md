# Implementation track

The plan (`From_Scratch_Android_Inference_Engine_Plan.md`) says *what* to build and why.
This file is the build order: every concrete piece, where it lives, what checks it, and its status.
It is written for two parallel tracks:

- **Implementation track** — builds the engine ahead, phase by phase, on branch `impl`.
- **Learning track** — goes through the concepts at its own pace on `master`, using the
  finished, verified code and measured numbers as material.

Status legend: `[x]` done · `[~]` in progress · `[ ]` not started · `(phone)` needs the device ·
`(unplugged)` needs the phone off USB power (adb over Wi-Fi).

---

## 0. Where the engine is today (2026-10-22)

All nine phases have working, verified code; `master` has everything up to `dc636d9`.
85 tests pass on the phone and 83 on the host. Each phase's numbers, conditions and raw data are in
`results/phase-N.md`.

| Phase | What works | Verified against | Key numbers (phone, on battery) | Open |
| --- | --- | --- | --- | --- |
| 0 Setup | Cargo project, phone runner, timing harness, `Tensor` | — | — | `notes/phase-0.md` (user) |
| 1 Hardware | `AlignedBuf`, pinning, sysfs/battery readers, sampler, bandwidth kernels | — (measurements) | DRAM read 61 GB/s one prime core, 67 GB/s all cores; 0.066 J/GB prime | — |
| 2 Scalar | llama2.c loader + tokenizer, forward pass, KV cache, sampler, CLI | llama2.c, bit-identical | stories15M 88.6 tok/s on one prime core | — |
| 3 NEON | NEON GEMV (3 steps), GEMM, batched prefill, NEON attention | numpy goldens; llama2.c tokens | GEMV 53 GB/s in L2, 41 GB/s from DRAM; stories15M 422 tok/s | kernel choice for short rows; GEMM GFLOPS |
| 4 Threads | pinned pool, dynamic row blocks, caller on fastest core | bit-identical to 1 thread | SmolLM2 Q4_0 275 tok/s on cores 2–7 | clean sweep; cores 0–1 hypothesis |
| 5 SmolLM2 | JSON, mmap, safetensors, GQA, rotate-half RoPE, byte-level BPE | `transformers` (≤ 5.5e-5), `tokenizers` (433 strings) | — | 360M |
| 6 Quantization | Q8_0/Q4_0 byte-identical to llama.cpp, `sdot` kernels, PPL/KL mode | llama.cpp quantizers, `llama-perplexity` | Q8_0 GEMV 32 GB/s; PPL matches llama.cpp | `smmla` GEMM (6.5) |
| 7 Qwen2.5 + GGUF | GGUF reader, Qwen2 (biases, θ 1e6), NFC + Qwen tokenizer, ChatML, bench suite | `transformers`, `tokenizers`, `llama-perplexity`, `llama-bench` | Q4_0 decode 111–129 tok/s (cores 2–7), 0.041 J/token | prefill 2–6× behind llama.cpp |
| 8 App | C ABI (`include/aether.h`), JNI shim, Kotlin app | app speed = CLI speed | ~95–100 tok/s streaming chat | — |

Biggest open items: quantized prefill GEMM with `smmla` (6.5), attention split across positions as
well as heads, SmolLM2-360M (5.6), a clean Phase 4 sweep.

Reference tooling in place: llama.cpp host + Android builds (`~/llama.cpp`, `d00685831`), patched
llama2.c (`third_party/`), `tools/.venv` with numpy, torch, transformers, tokenizers, gguf.

---

## S. Setup (before or alongside Phase 1)

- [x] **S1. Serialize phone access.** `flock` in `scripts/adb-run.sh` so two instances never run on
      the phone at once (they would corrupt each other's timings and heat the SoC).
- [x] **S2. Python reference env.** `tools/requirements.txt` + `uv venv tools/.venv` (gitignored):
      numpy, torch (CPU wheel), transformers, safetensors, tokenizers, gguf, matplotlib. ~2 GB.
- [x] **S3. llama2.c.** (cloned `350e04f`; `tools/stories15m_goldens.sh` applies the patch and regenerates goldens) Clone to `third_party/llama2.c` (gitignored), build `run` on host, apply
      `tools/llama2c-dump.patch` (fwrite of per-layer activations + logits as raw f32).
- [~] **S4. Model fetch script.** (stories15M + tokenizer.bin downloaded; script pending) `tools/fetch_models.sh` → `testdata/models/` (gitignored):
      stories15M.bin + tokenizer.bin (~60 MB), SmolLM2-135M (~270 MB), SmolLM2-360M (~720 MB),
      Qwen2.5-0.5B (~1 GB), each only when its phase starts.
- [x] **S5. llama.cpp for Android.** (`~/llama.cpp/build-android/bin`: llama-bench, llama-cli, llama-perplexity) Cross-build `llama-bench`, `llama-cli`, `llama-perplexity`
      with the same NDK (needed from Phase 6; build when Phase 6 starts).
- [x] **S6. Miri** (`rustup component add --toolchain nightly miri`) to check every `unsafe` module
      on the host.
- [x] **S7. Golden-file format.** (llama2.c record dumps + `src/testutil.rs`; runner syncs `testdata/` to the phone) One tiny shared format for kernel/activation dumps until the
      safetensors parser exists: raw little-endian f32 + a `.json`/text sidecar with shape. Readers in
      `src/testutil.rs` (test-only).

Disk budget: env 2 GB + models ~2.5 GB (f32/bf16) + GGUFs ~1.5 GB + llama.cpp Android build ~0.5 GB
+ golden dumps <1 GB ≈ 7.5 GB of the 14 GB free.

---

## Target code layout

Grows phase by phase; nothing is created before its phase.

```
src/
  aligned.rs          AlignedBuf (64-byte aligned f32 storage)                      P1
  bench.rs            timing harness                                                 P0
  tensor.rs           shape/strides                                                  P0
  sys/                affinity.rs (sched_setaffinity, sched_getcpu)                  P1
                      sysfs.rs (cpufreq, thermal zones, battery current/voltage)     P1
                      cpu.rs (CTR_EL0 cache-line size, feature detection)            P1/P3
  monitor.rs          background power/thermal sampler → joules, traces             P1
  kernels/            scalar.rs (matmul, rmsnorm, softmax, rope, silu)               P2
                      neon.rs (gemv, gemm, vector ops)                               P3
                      quant.rs (Q8_0, Q4_0 blocks; dequant, sdot, smmla kernels)     P6
  pool.rs             persistent pinned thread pool                                  P4
  model/              config.rs, transformer.rs (forward), kv_cache.rs, sampler.rs   P2→P5
                      llama2c.rs (loader)                                            P2
  format/             mmap.rs, json.rs, safetensors.rs                               P5
                      gguf.rs                                                        P7
  tokenizer/          llama2c.rs (score-based BPE)                                   P2
                      bytelevel.rs (GPT-2 byte-level BPE from tokenizer.json)        P5
                      pretok.rs + unicode tables (Qwen regex, hand-written)          P7
  ffi.rs              C ABI (cdylib)                                                 P8
src/bin/ or examples/ bw_sweep, generate, quantize, perplexity, bench_suite
tools/                Python golden generators, llama2.c patch, plotting, fetch
android/              Kotlin app + JNI                                               P8
results/phase-N.md    numbers + conditions, committed with the code that made them
```

Rule from the plan, kept throughout: model code calls a small set of op functions
(`matmul`, `rmsnorm`, `rope`, `attention`, `softmax`, `swiglu`); kernels never know about models.

---

## Phase 1 — Hardware & memory model

Goal: the yardstick table. This SoC has **no efficiency cores** (cores 0–5 performance 3.53 GHz,
6–7 prime 4.32 GHz), so "prime vs efficiency" in the plan becomes **prime vs performance**.

- [x] **1.1 AlignedBuf.** Done and Miri clean (slices, `Deref`, `Send`/`Sync`); `CTR_EL0` reads 64 B on the phone.
- [x] **1.2 Core pinning.** `sys::affinity::pin_current_thread(&[cores])`, `current_cpu()` via the
      `libc` crate (first dependency). Test: pin to each core, `sched_getcpu` agrees (phone).
- [x] **1.3 sysfs readers.** cpufreq cur/max per core, thermal zones (type + temp, m°C),
      battery `current_now`/`voltage_now` with unit/sign detection, charging status. Missing or
      unreadable files are reported, not fatal (phone).
- [x] **1.4 Sampler.** (battery via Android's battery service + charge counter: sysfs battery is SELinux-denied; see results/phase-1.md) Background thread, ~100 ms period, stop flag; returns traces + average power,
      energy (J), min/max freq, max temp. CSV output (phone).
- [x] **1.5 Bandwidth kernels.** (`src/membench.rs`) read (NEON, 4–8 independent accumulators — the Phase 0 sum was a
      latency-bound dependency chain, not bandwidth), copy, scale, add; bytes counted per kernel.
      Aligned vs +4-byte-shifted read comparison.
- [x] **1.6 Working-set sweep** `examples/bw_sweep.rs`: 4 KB → 256 MB log-spaced, fixed total bytes per point,
      `--cores` flag: one prime (7), one performance (0), all 6 performance, all 8 in parallel with
      scoped threads each on its own pinned chunk. CSV with a conditions header (phone).
- [x] **1.7 Plot** `tools/plot_sweep.py` (log x axis, cache cliffs annotated).
- [x] **1.8 Sustained run** (`examples/bw_sustained.rs`; `scripts/phase1-runs.sh sustained`) ≥5 min all-core read with the sampler: bandwidth, freq, temp over time
      (phone).
- [x] **1.9 Energy per GB** (`bw_sustained --idle-s`; `scripts/phase1-runs.sh energy`), prime vs performance core (phone, unplugged — adb over Wi-Fi; USB
      charging makes `current_now` meaningless).
- [x] **1.10 `results/phase-1.md`**: L1/L2/SLC effective GB/s, single-core DRAM read, all-core
      DRAM read, throttled sustained, J/GB per core class, conditions.

## Phase 2 — Scalar end-to-end on stories15M

- [x] **2.1 References.** S3 + S4: llama2.c built and patched; dumps for a fixed prompt
      (embedding, each norm, attention out, MLP out, logits per position) → `testdata/stories15M/`.
- [x] **2.2 Loader** `model/llama2c.rs`: 7×i32 header (dim, hidden_dim, n_layers, n_heads,
      n_kv_heads, vocab_size — negative means unshared classifier — seq_len), then weights in
      llama2.c order.
- [x] **2.3 Tokenizer** `tokenizer/llama2c.rs`: tokenizer.bin (max len, then score/len/bytes per
      token), score-based merges, byte fallback `<0xNN>`, BOS, leading-space rule as in `run.c`.
- [x] **2.4 Scalar ops** (checked through the layer dumps; numpy kernel goldens come with 3.1) `kernels/scalar.rs`: matmul (W·x), rmsnorm, softmax (max-subtracted),
      rope (adjacent pairs, llama2.c), silu/swiglu; checked against numpy goldens (S7).
- [x] **2.5 Forward pass** + preallocated KV cache (`[layer][pos][kv_dim]`), run-state buffers
      allocated once.
- [x] **2.6 Sampler**: argmax, temperature, top-p, top-k; xorshift RNG identical to llama2.c so
      seeded sampling can match too.
- [x] **2.7 `generate` CLI** (the `aether` binary, plus `-c CORES`): `--model --tokenizer --prompt --steps --temp --topp --seed`, prints
      tok/s.
- [x] **2.8 Correctness** (bit-identical on host and phone; `results/phase-2.md`): per-layer vs dumps (~1e-5 rel), greedy tokens identical for the full
      run, tokenizer identical on a test corpus (encode via a tiny C harness around llama2.c's
      `encode`).
- [x] **2.9 Baseline** (88.6 tok/s prime / 58.4 perf, on battery caps): scalar single-thread tok/s on the phone + `simpleperf` profile →
      `results/phase-2.md` (phone).

## Phase 3 — NEON f32 kernels

- [x] **3.1 Kernel golden generator** `tools/gen_kernels.py`: random inputs + numpy outputs for
      square, tall, wide, GEMV-shaped and odd sizes (tails).
- [x] **3.2 NEON GEMV**: 4-wide → 4–8 accumulators → unrolled; tail handling. GB/s as % of
      Phase 1 single-core read.
- [x] **3.3 Batched forward for prefill**: process T prompt tokens at once (the Phase 2 loop is one
      token at a time), so prefill becomes GEMM.
- [x] **3.4 NEON GEMM**: packed panels + register-blocked micro-kernel + cache blocking. GFLOPS vs
      theoretical peak FMA.
- [~] **3.5 NEON vector ops** (attention done: +10% Qwen decode at depth 512, A/B in results/phase-7.md; rest not needed yet) where the profile says so: rmsnorm, softmax (needs a vector exp),
      silu/swiglu, rope.
- [x] **3.6 Dispatch** (dotprod/fp16 intrinsics stable on 1.99; i8mm `vmmlaq_s32` is not → `asm!` for smmla): `is_aarch64_feature_detected!`, `#[target_feature]`, scalar fallback on
      the host (x86 tests still run). Check dotprod/i8mm intrinsic stability on Rust 1.99 now
      (decides nightly vs `asm!` for Phase 6).
- [~] **3.7 Swap in, re-verify** (verified: tokens identical; results/phase-3.md pending): stories15M tokens still identical; speedup + profile →
      `results/phase-3.md` (phone).

## Phase 4 — Threading & core placement

- [x] **4.1 Thread pool** `pool.rs`: persistent workers pinned to a core set, epoch counter,
      spin-then-park, cache-line-padded per-worker state; scoped job submission.
- [x] **4.2 Parallel GEMV** by output rows (chunks aligned to 64 B), parallel GEMM, attention
      heads in parallel.
- [x] **4.3 Flags** (`-c LIST`: one thread per listed core): `--threads`, `--cores`. Results bit-identical to single-thread (test).
- [~] **4.4 Sweeps** (running: scripts/model-speed.sh): tok/s vs threads and core sets; multi-thread GEMV GB/s as % of Phase 1
      all-core; J/token per core set (unplugged) → `results/phase-4.md`. Optional: rayon baseline.

## Phase 5 — SmolLM2 (GQA, safetensors, byte-level BPE)

- [x] **5.1 mmap** (`libc::mmap`, read-only) + **safetensors parser** (8-byte len, JSON header,
      data); bf16 → f32.
- [x] **5.2 JSON**: minimal hand-written parser for `config.json`, safetensors headers,
      `tokenizer.json` (decision: hand-written, unless you prefer `serde_json`).
- [x] **5.3 Config-driven model**: one code path for stories15M and SmolLM2 (GQA head mapping,
      tied embeddings, rope theta, norm eps, rope layout = rotate_half for HF).
- [x] **5.4 Byte-level BPE** from `tokenizer.json`: byte↔unicode map, pre-tokenizer as declared
      there, merge ranks, special tokens, decode.
- [x] **5.5 HF goldens** `tools/hf_dump.py`: forward hooks → `.safetensors` per-layer dumps (read
      by our own parser), greedy `generate(do_sample=False)` tokens, tokenizer corpus (Unicode,
      emoji, code, odd whitespace) → `testdata/smollm2/`.
- [~] **5.6 Verify** (135M done on host and phone; 360M not yet) 135M and 360M: layers within tolerance, greedy tokens equal, tokenizer exact;
      tok/s → `results/phase-5.md` (phone).

## Phase 6 — Quantization

- [x] **6.1 Block formats** (byte-identical to llama.cpp via the `gguf` package; tools/gen_quant.py) Q8_0 (32×i8 + f16 scale = 34 B) and Q4_0 (32×4-bit + f16 scale =
      18 B, value = (q−8)·d), byte-identical to llama.cpp's reference quantizers.
- [~] **6.2 `quantize` tool** (changed: models are quantized in memory at load, `-q q8_0|q4_0`, like `llama-quantize --pure`; a file writer comes with GGUF in Phase 7) (Rust binary): safetensors → our quantized weights.
- [x] **6.3 Q8_0 GEMV (a)** dequant-to-f32 + FMA; **(b)** activations quantized to Q8_0 on the
      fly + `sdot` (`vdotq_s32`).
- [x] **6.4 Q4_0 GEMV** (4-row kernels; Q4_0 still compute-bound on one core): NEON nibble unpack + `sdot`.
- [~] **6.5 Quantized GEMM** (sdot rows-across-tokens version done; smmla not yet) for prefill: `sdot`, then `smmla` (i8mm) with repacked weights.
- [x] **6.6 Perplexity + KL mode** (`--ppl FILE --kl`, llama-perplexity's procedure) over wikitext-2 test (cached), with f32 logits as base.
- [~] **6.7 Verify** (f32, Q8_0 and Q4_0 PPL/KLD match llama.cpp; results/phase-6.md pending): kernels vs numpy goldens of the same block math; PPL/KL vs `llama-perplexity
      --kl-divergence` on the same weights and text (S5); GB/s at quantized size, memory footprint →
      `results/phase-6.md` (phone).

## Phase 7 — Qwen2.5-0.5B + GGUF

- [x] **7.1 GGUF parser** (header, typed KV metadata, tensor infos, alignment, data) + mmap;
      F32/F16/BF16/Q8_0/Q4_0 tensors. Files from llama.cpp's `convert_hf_to_gguf.py` +
      `llama-quantize`, so both engines read the same bytes.
- [x] **7.2 Architecture from metadata**: llama (Q/K permuted by converter → adjacent-pair RoPE)
      vs qwen2 (NeoX/rotate_half), QKV bias, rope theta 1e6, tied embeddings.
- [x] **7.3 Qwen tokenizer** (433 strings identical to HF tokenizers; NFC on 17,253 cases): ~151k byte-level BPE; hand-written pre-tokenizer equivalent to the
      Qwen2 regex, with Unicode category tables generated by `tools/gen_unicode_tables.py`;
      special tokens; ChatML chat template.
- [x] **7.4 Benchmark suite** (`scripts/phase7-bench.sh`): prefill tok/s at several prompt lengths, decode tok/s at several
      context lengths, TTFT, peak RSS (`VmHWM`), J/token, 5-minute sustained tok/s trace.
- [~] **7.5 Verify** (HF safetensors + GGUF f32: 48 greedy tokens identical; PPL/KLD for f32, Q8_0, Q4_0 match llama.cpp; results/phase-7.md written; NEON attention A/B done): HF per-layer + greedy tokens (f16/bf16), PPL/KL vs llama.cpp per quant type,
      tokenizer exact vs HF; `llama-bench` on the phone under the same conditions →
      `results/phase-7.md` with a written account of where time and energy go (phone).

## Phase 8 — Android app

- [x] **8.1 cdylib + C ABI** (`include/aether.h`, `examples/c/smoke.c`; round-trip test reproduces llama2.c text) `ffi.rs`: opaque handle, create/load/generate(callback)/free, error
      codes, `catch_unwind` at every boundary.
- [x] **8.2 JNI + Kotlin app** (`scripts/android-native.sh`, `android/README.md`) in `android/`: load Qwen2.5-0.5B Q4_0 from app storage, generate off
      the main thread, stream tokens to the UI.
- [x] **8.3 Verify** (app within run-to-run spread of the CLI; results/phase-8.md): installable APK; tok/s matches the CLI.

---

## How the two tracks share one repo and one phone

- Implementation works in a git worktree on branch `impl` (`../Aether-impl`), so it never touches
  the learning track's working tree. A phase is merged to `master` only when its verification
  items pass and its `results/phase-N.md` exists.
- One phone: `adb-run.sh` takes a lock (S1). Benchmarks record conditions (cores, screen, charging,
  start temperature) and wait for the SoC to cool between runs.
- The learning track owns `site/`, `videos/`, and `notes/phase-N.md` (the user's journal). The
  implementation track owns `results/` and the engine code; the site cites `results/` numbers.
- Every number in `results/` is tied to the commit that produced it.

## Risks known now

- **Disk** (14 GB free) — fetch models per phase, delete superseded dumps.
- **Energy needs adb over Wi-Fi**, phone unplugged; plugged-in current readings are charging current.
- **Shared phone** — a benchmark during the other track's run is invalid; the lock handles
  overlap, not heat. Check temperature at start.
- **dotprod/i8mm intrinsics** may still need nightly or `asm!` — checked in 3.6, before Phase 6.
- **Qwen pre-tokenizer** needs Unicode category tables — a sub-project inside Phase 7.
- **Prefill needs a batched forward** — a structural change to the Phase 2 engine in 3.3.
