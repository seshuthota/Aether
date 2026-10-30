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
| 3 NEON | NEON GEMV (3 steps), GEMM, batched prefill, NEON attention | numpy goldens; llama2.c tokens | GEMV 53 GB/s in L2, 41 GB/s from DRAM; stories15M 422 tok/s; GEMM 28–29 FLOP/cycle (~90 % of an assumed peak) | kernel choice for short rows |
| 4 Threads | pinned pool, dynamic row blocks, caller on fastest core | bit-identical to 1 thread | SmolLM2 Q4_0 275 tok/s on cores 2–7 | clean sweep; cores 0–1 hypothesis |
| 5 SmolLM2 | JSON, mmap, safetensors, GQA, rotate-half RoPE, byte-level BPE | `transformers` (≤ 5.5e-5; 360M ≤ 1.0e-4), `tokenizers` (433 strings) | SmolLM2-360M Q4_0 183 tok/s (charging) | — |
| 6 Quantization | Q8_0/Q4_0/Q5_0 byte-identical to llama.cpp, Q4_K/Q6_K read from GGUF, `sdot`/`smmla` kernels, PPL/KL mode | llama.cpp quantizers, `gguf` dequantizers, `llama-perplexity` | Q8_0 GEMV 32 GB/s, Q4_0 repacked 19 GB/s (one core); PPL matches llama.cpp; Qwen Q4_K_M decode 104–111 vs llama.cpp 86 | — |
| 7 Qwen2.5 + GGUF | GGUF reader, Qwen2 (biases, θ 1e6), NFC + Qwen tokenizer, ChatML, bench suite | `transformers`, `tokenizers`, `llama-perplexity`, `llama-bench` | Q4_0 decode 111–129 tok/s (cores 2–7), 0.041 J/token; after the repack level with llama.cpp; prefill level or ahead (2026-10-29) | — |
| 8 App | C ABI (`include/aether.h`), JNI shim, Kotlin app | app speed = CLI speed | ~95–100 tok/s streaming chat | — |

Night of 2026-10-29 (results/phase-6.md from "Q4_K decode: scales as vectors" on): Q4_K /
Q5_0 / Q6_K kernels with `sri` unpacks, K-quant rows unpacked with NEON, q/k/v and gate/up as one
pool job, workers spin 2.8 ms before parking, RoPE angles once per call, SwiGLU and attention
softmax on NEON. Qwen2.5-0.5B, cores 2–7, uncapped clocks: decode ahead of llama.cpp in every
format (Q4_0 tg64 158–180 vs 138–143, Q4_K_M 104–111 vs 86); prefill 10–40 % ahead for Q4_0 /
Q8_0, 3–7× for Q4_K_M. 92 tests on the phone. Timed runs need the clock caps recorded
(`prefill-ab.sh` does): a thermal cap on the prime cores (1.69 → 1.02 GHz) moves them by 20 %.

Day of 2026-10-29 (later phases, plan section 13): flash-decoding by K/V head (decode −11 / −18 %
at d512 / d1024); speculative decoding, both prompt lookup (`--lookup K`, +60–75 % when the reply
copies the prompt) and a draft model (`--draft PATH`, 0 to +29 %: SmolLM2-135M decodes only 2× faster
than 360M); prompt lookup in the C ABI and the app; prefill tail tiles (−6 % cycles on odd lengths).
**Vulkan on the Adreno 830** (results/phase-l3.md, `--features vulkan`): Q4_0 GEMV / GEMM shaders and
a whole decode step on the GPU (`aether --gpu`), PPL matching llama.cpp on SmolLM2-135M and
Qwen2.5-0.5B. The GPU reads DRAM as fast as the CPU (~66 GB/s), so decode is not faster (103 vs 158
tok/s sustained), but it takes ~2.3 W instead of 4.5 W: 0.021–0.024 J/token vs 0.029, 68 vs 89 °C.
99 tests on the phone with `--features vulkan` (96 without).

**The NPU** (2026-10-29 / 30, results/phase-l5.md, `--features hexagon`): our own code on the
Hexagon DSP, unsigned. The whole decode step in one FastRPC call (HVX across 6 threads, weights by
DMA into VTCM): Qwen2.5-0.5B 6.75 ms a token (123 tok/s) against llama.cpp's NPU backend 74, PPL
matching llama.cpp. Matrix × matrix products on HMX for prompts: Qwen's layer products at 1512
tok/s for 128 tokens (llama.cpp's NPU prefill ~330 for the whole model). Prefill through the whole
model on the DSP is written and waits for the phone (L.5 step 5b).

Since then (2026-10-30, all verified on the phone and merged): prefill on the DSP (HMX, 1608 /
778 tok/s at 128 tokens), the app's CPU / GPU / DSP selector (DSP sessions work from an app),
prompt lookup on the DSP (+6–28 %), Q4_K_M files on the DSP (PPL matching llama.cpp).

### Next steps (2026-10-30)

1. **Q4_K_M prompts on HMX.** The DSP runs Q4_K_M prompts in 8-token passes on HVX: 272 tok/s
   against 755 for Q4_0 on HMX. `hmx.c` dequantizes only Q4_0 into fp16 tiles; the other formats'
   dequantization (Q5_0, Q8_0, Q4_K, Q6_K) can be checked on the simulator first.
2. **GPU lookup that pays on the Adreno.** The few-token shader costs about a whole dispatch per
   extra token there (`results/phase-l3/raw/vk-gemv-n-adreno.txt`); a shader that shares each
   weight load across the tokens is needed before lookup is turned back on for GPU sessions.
3. **GPU prefill**: the texel-buffer path for the activations (~1.7× behind llama.cpp's OpenCL
   on long prompts; L.3).
4. **DSP attention at long positions.** Profiled
   (`dsp_ppl … depth 0,512,2048`, results/phase-l5.md): the whole slowdown
   is the attention stage (Qwen 0.26 → 3.46 → 17.14 ms at depths 0 / 512 /
   2048). The wide value loop (four accumulators) was measured next: −0.38 ms
   of attention at depth 512, nothing useful at 2048, PPL within a few
   hundredths (27.7623 / 19.7716 decode). Next: stride padding where the
   key stride is a power of two, and l2fetch of the next key and value
   blocks, each measured at 512 and 2048, then the position split and one
   VTCM read per K/V head. f16 KV only if depth 2048 is still steep.
5. **Verification leftovers** (all need the phone): 4.4 sweeps, 6.7 / 7.5 GB/s and footprint
   write-ups, a same-session `llama-bench` rerun for an exact CPU comparison, a screen-on
   comparison with the user present.
6. **Learning track** (with the user): pages and films for what the implementation track built.

### Waiting for the phone (written 2026-10-30, committed untested on `gpu-pending`)

Items 9–13 ran on the phone on 2026-10-30, 22:00–23:15 (results/phase-l5.md "Step 6"):
all three engines in the app, DSP sessions from an app work; GPU lookup slower (off in the
C ABI); DSP lookup +6–28 % with the few-token pass by DMA; Q4_K_M on the DSP at llama.cpp's
PPL after a VTCM slot fix (`c68da0a`).

Steps 1–8 ran on the phone on 2026-10-30, 19:20–20:40 (results/phase-l5.md "Step 5b" and
"Every engine", results/phase-l3.md step 11; one bug found and fixed, `3ab2966`):

1. [x] Tests: hexagon 112/112 (three runs, after the VTCM lock), vulkan 113/113.
2. [x] `dsp_gemm -- 128 6`: 1656 tok/s (was 1543).
3. [x] DSP decode with transposed keys: PPL 27.7876 / 19.7560 (reordered adds; v3 27.7429 /
   19.7584), Qwen over a 512-token run 72 → 86 tok/s.
4. [x] DSP prefill: PPL 27.6893 / 19.7174 (llama.cpp 27.7412 / 19.7283); 1608 / 778 tok/s at 128.
5. [x] `aether --dsp`: agrees with the CPU for 2–40 tokens, then f16 vs int8 rounding picks a
   different word at a near-tie (the PPL in 4 is the check).
6. [x] GPU f16 Q4_0: 590–817 GFLOP/s (f32 147–228); Qwen prompt 812 tok/s at 128 (was 267);
   PPL 27.6924 / 19.7190. The app keeps the prompt on the CPU; its hybrid session no longer
   builds the f16 matrices (`5003b5e`: −196 MB, faster load).
7. [x] Q4_K_M on the GPU: PPL equal to the PC's; prompt 563 tok/s at 128; decode 12.3 ms of GPU
   time a token (Q4_0 9.0, unchanged).
8. [x] `backends-bench.sh` (`results/backends/raw`), and `SUSTAIN=300` on battery for Qwen Q4_0:
   CPU 107 tok/s 0.030 J/token, GPU 78 / 0.029, DSP 131 / 0.025.
9. [x] **The app's CPU / GPU / DSP selector** (replaces the GPU checkbox). Installed on the
   phone (2026-10-30, with `5003b5e`); `libcdsprpc.so` is in `/vendor/etc/public.libraries.txt`.
   `qwen2.5-0.5b-instruct-q4_0.gguf` (internal) and `qwen2.5-0.5b-q4_k_m.gguf` (external files
   dir, a base model: Chat off) are on the phone. Try each of CPU, GPU and DSP on the instruct
   model, then GPU on the Q4_K_M one: the reply streams and the status line shows load time and
   prefill / decode tok/s. Open question: can an app process start an unsigned DSP session from
   its files directory? If not, the DSP option shows the error and CPU and GPU still work.
10. [x] **Prompt lookup on the GPU** (written 2026-10-30 after the run above, PC-verified;
   results/phase-l3.md step 12). `cargo test … --features vulkan --lib qx_gemv_n gpu_lookup`;
   `--example vk_gemv_n` (few-token shader against one decode step: on the Intel GPU 2 tokens
   cost 3.7–4.3 steps; on the Adreno, where decode is bound by memory, it should be far less);
   `aether GGUF --gpu -c 2-7 -t 0 --lookup 4` against without, on a prompt the reply copies
   (results/phase-6/raw/lookup/, on the phone `/data/local/tmp/aether/prompt-*.txt`) and on
   fresh prose: tok/s and drafts
   accepted. The app's lookup checkbox in GPU mode. If lookup is slower on the Adreno, turn it
   off for GPU sessions in the C ABI again.
11. [x] **Prompt lookup on the DSP** (written 2026-10-30 after the run above; the HVX product
   for several tokens verified on the Hexagon simulator, the rest phone-only). `cargo test …
   --features hexagon --lib few_tokens dsp_lookup` (the few-token pass's logits must equal the
   decode step's bit for bit); `dsp_ppl SMOLLM 20 512 6 1f 1 few` and the same on Qwen (PPL
   must equal the decode PPL of item 3: 27.7876 / 19.7560; prints the pass's time for 1–8
   tokens against one decode step); `aether GGUF --dsp -t 0 --lookup 4` against without on
   the copying prompts and on prose; the app's lookup checkbox in DSP mode. The pass reads
   weights by l2fetch, not DMA: if 2 tokens cost much more than one decode step, DMA is the
   next thing to try.
12. [x] **Q4_K_M on the DSP** (written 2026-10-30; the products of every format verified on the
   simulator, the model phone-only). `cargo test … --features hexagon --lib few_tokens` now
   also runs SmolLM2 Q4_K_M (few-token logits = decode logits). `dsp_ppl` on
   `smollm2-135m-q4_k_m.gguf` and `qwen2.5-0.5b-q4_k_m.gguf` (decode, and `… 1 few`): llama.cpp
   20.7784 / 16.7657, our CPU 20.7201 / 16.7088. Decode tok/s against Q4_0 (bytes per weight
   ~1.4× Q4_0's, so expect it slower), the stage profile, and with DMA mask 0 against 1f.
   `aether qwen…q4_k_m.gguf --dsp`: the text, and the prompt speed (8 tokens a call on HVX:
   no HMX prefill for these formats yet). The app's DSP mode on the Q4_K_M file.
13. [x] Then: the app's result in results/phase-l5.md, tick L.5 / L.3 below,
   `git branch -f impl gpu-pending`, merge.

### Phone checks of the host-built commits (2026-10-28): done

All ran on the phone the same evening (charging, clock caps 1.79 / 1.96
GHz); results in `results/phase-6.md` ("Repacked weights …") and
`results/phase-5.md`. Kept: weight repack (`31513f0`; Q4_0 decode +25–34 %,
level with llama.cpp) and the NEON activation quantizer (`0d1510f`; prefill
+10–24 %, gap to llama.cpp 1.15–1.5×). Reverted for no measurable gain: NEON
softmax (`85d1a12`) and attention split across positions (`bdbf963`). 89/89
tests; PPL unchanged. 3.7 (GEMM FLOP/cycle and the NEON-build profile) done
the same night, results/phase-3.md.

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
- [x] **S4. Model fetch script.** (`tools/fetch_models.sh [part|all]`, skips existing files; also makes the GGUFs with llama.cpp and fetches wikitext-2; URLs checked, stories15M md5 equal to ours) `tools/fetch_models.sh` → `testdata/models/` (gitignored):
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
- [~] **3.5 NEON vector ops** (attention done: +10% Qwen decode at depth 512, A/B in results/phase-7.md; NEON softmax: no gain on 2026-10-28, kept on 2026-10-29 after the attention rewrite (decode at d1024 −10 % cycles); rest not needed yet) where the profile says so: rmsnorm, softmax (needs a vector exp),
      silu/swiglu, rope.
- [x] **3.6 Dispatch** (dotprod/fp16 intrinsics stable on 1.99; i8mm `vmmlaq_s32` is not → `asm!` for smmla): `is_aarch64_feature_detected!`, `#[target_feature]`, scalar fallback on
      the host (x86 tests still run). Check dotprod/i8mm intrinsic stability on Rust 1.99 now
      (decides nightly vs `asm!` for Phase 6).
- [x] **3.7 Swap in, re-verify** (verified: tokens identical, ×4.8 over scalar; GEMM 28–29 FLOP/cycle on core 7, NEON-build profile: gemv 89 %, 25.7 GB/s; results/phase-3.md): stories15M tokens still identical; speedup + profile →
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
- [x] **5.6 Verify** (135M and 360M match transformers on host and phone; 360M tok/s in results/phase-5.md) 135M and 360M: layers within tolerance, greedy tokens equal, tokenizer exact;
      tok/s → `results/phase-5.md` (phone).

## Phase 6 — Quantization

- [x] **6.1 Block formats** (byte-identical to llama.cpp via the `gguf` package; tools/gen_quant.py) Q8_0 (32×i8 + f16 scale = 34 B) and Q4_0 (32×4-bit + f16 scale =
      18 B, value = (q−8)·d), byte-identical to llama.cpp's reference quantizers.
- [x] **6.2 `quantize` tool** (models are also quantized in memory at load, `-q q8_0|q4_0`; 2026-10-30: `aether-quantize [--imatrix FILE] IN.gguf OUT.gguf q8_0|q4_0|q5_0|q4_k_m`, `format::gguf::Writer`; output byte-identical to `llama-quantize --pure [--imatrix]` and to its Q4_K_M mix (with and without imatrix) for SmolLM2 and Qwen2.5, results/phase-6.md) (Rust binary): f32/f16 GGUF → quantized GGUF.
- [x] **6.3 Q8_0 GEMV (a)** dequant-to-f32 + FMA; **(b)** activations quantized to Q8_0 on the
      fly + `sdot` (`vdotq_s32`).
- [x] **6.4 Q4_0 GEMV** (4-row kernels; Q4_0 still compute-bound on one core): NEON nibble unpack + `sdot`.
- [x] **6.5 Quantized GEMM** (smmla tiles of 4 rows × 8 tokens on weights repacked four rows at a time, NEON activation quantizer, per-token steps across the pool: prefill level with llama.cpp for Q4_0, ahead for Q8_0 and Q4_K_M, 2026-10-29, results/phase-6.md) for prefill: `sdot`, then `smmla` (i8mm) with repacked weights.
- [x] **6.6 Perplexity + KL mode** (`--ppl FILE --kl`, llama-perplexity's procedure) over wikitext-2 test (cached), with f32 logits as base.
- [~] **6.7 Verify** (f32, Q8_0 and Q4_0 PPL/KLD match llama.cpp; results/phase-6.md written): kernels vs numpy goldens of the same block math; PPL/KL vs `llama-perplexity
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

## Later phases (plan section 13)

- [x] **L.1 Q4_K_M files** (2026-10-29; results/phase-6.md "Q4_K_M"): read llama.cpp's most common
      GGUF mix. On these small models that is mostly **Q5_0** (K-quants need rows a multiple of 256;
      576 and 896 are not), plus Q4_K / Q6_K in `ffn_down` and Q8_0. Done: Q5_0 (quantizer
      byte-identical to llama.cpp, `sdot`, `smmla`, repacked four rows at a time), Q4_K and Q6_K
      decoders (identical to gguf's on random blocks) and `sdot` kernels; PPL/KLD match
      llama-perplexity for SmolLM2-135M Q4_K_M and Q5_0 and Qwen2.5-0.5B Q4_K_M; `smmla` for the
      K-quants on rows unpacked once per matmul (pp512 cycles −41 %); Q4_K decode scales as vectors
      (decode cycles −4 %); Q5_0 unpack with `sri` (Q4_K_M decode +8–11 %); Q6_K with `sri`
      (its cycles −8 %); K-quant rows unpacked with NEON (pp64 cycles −28 %).
- [x] **Measurement: bimodal timed runs** (2026-10-29): explained — the prime cores' clock cap
      drops 1.69 → 1.02 GHz partway through some runs (results/phase-6.md, "Night of").
      `prefill-ab.sh` now records the lowest caps seen during each run ("# lowest clock caps").
- [x] **L.2 Importance matrices / our own K-quant quantizer** (2026-10-29, host only, on
      branch gpu-pending; results/phase-6.md "Importance matrices"): our own Q4_K / Q6_K
      quantizers and imatrix-weighted Q4_0 / Q5_0 / Q4_K / Q6_K, byte-identical to ggml-base's
      (`imatrix_quantizers_are_byte_identical_to_llama_cpp`); llama-imatrix files read
      (`format::imatrix`), `aether -q q4_0|q5_0 --imatrix FILE` byte-identical to
      `llama-quantize --pure --imatrix`, and every tensor of its imatrix Q4_K_M re-quantized to
      the same bytes; our own imatrix (`Imatrix::collect`) within 1.7e-3 of llama-imatrix's.
      PPL / KLD match llama-perplexity; the imatrix cuts KLD 11–33 % at the same size.
      Not done: writing imatrix files, and choosing a Q4_K_M mix ourselves (`-q` stays
      `--pure`). Lib tests pass on the phone too (2026-10-29).
- [x] **Flash-decoding** (plan section 13, 2026-10-29; results/phase-6.md "Flash-decoding by K/V
      head"): decode attention as (K/V head, 64-position chunk) items, merged per head; decode
      cycles −11 % at d512, −18 % at d1024.
- [x] **L.4 Speculative decoding** (2026-10-29; results/phase-6.md "Prompt lookup decoding"):
      model-free prompt lookup (`--lookup K`), verified in one batched pass, greedy or by
      speculative sampling (`Sampler::sample_draft`); `smmla` tiles of 1–3 pairs for small
      batches. +60–75 % on a copying task, about neutral on fresh prose. Draft model too
      (`--draft PATH`, `Drafter`; results/phase-6.md "Speculative decoding with a draft
      model"): SmolLM2-135M for 360M, 0 to +29 % at 81–95 % acceptance, because the 135M
      model decodes only 2× faster, and a K = 3 pass costs ~2.9 target steps.
- [x] **Lookup in the C ABI and the app** (2026-10-29): `aether_set_lookup(session, k, ngram)`
      (include/aether.h), `Aether.setLookup` in Kotlin, a "lookup" checkbox in the app (K = 3).
      `c_abi_round_trip` checks the greedy text with lookup on; tried in the app on the phone.
- [~] **L.3 Vulkan** (Adreno; results/phase-l3.md, `--features vulkan`):
  - [x] Device, buffers, compute kernels (`kernels::vulkan`, shaders/ via `build.rs` + NDK
        `glslc`, `ash` for the C API), `vk_bandwidth`: GPU reads DRAM at ~66 GB/s (CPU: ~67),
        an empty dispatch + barrier costs 1.7 µs. Test `copy_shader_copies`.
  - [x] Q4_0 GEMV shader against the llama.cpp goldens (`q4_0_gemv_matches_reference`):
        62 GB/s on the classifier (CPU 50), level with the CPU on layer matrices.
  - [x] Q4_0 GEMM (prefill) against the batch goldens (`q4_0_gemm_matches_reference`):
        ~260 GFLOP/s in f32, a third to a half of the CPU's int8 `smmla` path.
  - [x] A whole decode step on the GPU (`model::gpu::GpuModel`, one recorded command buffer
        per token): PPL matches llama.cpp (SmolLM2-135M 27.6910 vs 27.7412 ± 1.177, Qwen2.5-0.5B
        19.7193 vs 19.7283 ± 0.836; `vk_ppl`).
  - [x] Decode speed and energy (`vk_decode`): per-stage GPU timestamps; attention split in
        64-position chunks + merge (d1024 5.6 → 2.0 ms). Sustained 120 s: GPU 103 tok/s at
        ~2.3 W = 0.021–0.024 J/token, 68 °C; CPU 158 tok/s at 4.5 W = 0.029 J/token, 89 °C.
        Limit: GPU/bus clocks drop between bursts (no Vulkan control; sysfs needs root).
  - [x] CLI switch `aether MODEL.gguf --gpu` (prompt token by token; `Text::from_gguf`).
  - [x] The app: `aether_open_gpu` (C ABI), `GpuSession`, a "GPU" checkbox; tried on the phone
        (decode 90–92 tok/s vs 155 on the CPU). The "lookup" checkbox tried too (works).
  - [x] Prefill on the CPU, KV cache handed to the GPU: `aether --gpu -c CORES`
        (`GpuSession::open_hybrid`, `GpuModel::load_kv`, `Transformer::kv_cache`). 345-token
        prompt 3.9–4.5 s → 0.55–0.61 s (same as CPU-only); same greedy text as GPU-only.
        Not in the app yet. A batched GPU prefill (the step-3 GEMM) is still open.
  - [x] Step 6, GPU decode efficiency: the CPU's gap between tokens is not the cost (8 steps
        per submit: 7.6–8.6 ms per token, same as one); the matrix products are ~75 % of GPU
        time at 30–40 GB/s. Stacked wq/wk/wv and w_gate/w_up (`Kernel::bind_at` views):
        q/k/v −20 %, gate/up −6 % of GPU time, PPL identical. Tried and dropped: an unrolled
        row loop (slower everywhere), split-K across lane groups (never faster).
  - [x] Branch `gpu-pending` (2026-10-29, built while the phone was away, then verified on the
        phone: 106/106 lib tests, the same PPL, the app; results/phase-l3.md step 7):
        - Vulkan asks for 32-lane subgroups where a device may pick fewer (Intel), so every GPU
          test and `vk_ppl` runs on the PC. GPU PPL on Intel equals the Adreno's to 4 decimals.
        - Fused into the GEMVs: the residual add (wo, down), the RMS norm (q/k/v, classifier).
          Also fusing the norm into gate/up and the SwiGLU into down made decode 8 % slower on
          the phone (every workgroup redoes them), so those two run as their own kernels again.
          `add.comp` removed. Logits in host-cached memory.
        - Batched prompt pass on the GPU: `GpuModel::forward_batch` (up to 128 tokens; the
          step-3 GEMM with ACCUM/GLU; shaders take a token index). PPL through it equals the
          token-by-token path (SmolLM2 27.6910, Qwen2.5 19.7193). `vk_ppl … batch`, `vk_prefill`.
        - `aether_open_gpu(path, cores, max_seq, out)`: with cores, the CPU runs the prompt; the
          app's GPU mode uses that. `GpuSession` without cores runs the prompt batched on the GPU.
        - Greedy argmax on the GPU (`argmax.comp`, `GpuModel::greedy`): the CPU no longer reads
          152k logits per token at temperature 0 (~1 ms). CLI GPU decode 77–80 → 97 tok/s,
          after a CPU prompt 63–66 → 96–101 (no gap left).
  - [x] Prompt lookup on the GPU (`GpuModel::forward_few`, the decode shader for up to 8
        tokens, `GpuSession::generate_lookup`): correct, but 2.4× slower on the Adreno (each
        extra token costs about a whole dispatch); the C ABI ignores lookup for GPU sessions.
        A shader that shares the weights across tokens on the Adreno would be needed.
  - [~] GPU prefill as fast as llama.cpp's OpenCL (1,387 vs our 267 tok/s at pp256; step 8).
        f16 shaders on the phone (step 11): 771 at pp256, 812 at 128; the gap is ~1.7×. Left:
        the texel-buffer path for the activations.
        How its Adreno Q4_0 kernel (`mul_mat_Ab_Bi_8x4.cl`) differs from `q4_0_gemm.comp`:
        - No shared memory and no barriers. The weights are repacked once, transposed in groups
          of 4 nibbles (a `ushort` holds 4 consecutive inputs of one row) and scales likewise, so
          neighbouring threads read neighbouring rows: each load is coalesced straight from
          memory, and every weight is used by 8 tokens in registers.
        - The activations are transposed to fp16 and read as a 1-D image buffer
          (`read_imageh`): through the texture cache, 8 tokens a load, shared by every thread
          that needs them.
        - fp16 arithmetic (`half8` accumulators, double rate on Adreno) and a small tile per
          thread: 8 tokens × 4 rows = 32 halves (16 registers) against our 64 f32 accumulators,
          so many more waves are resident to hide latency. 128-lane subgroups ("full").
        A Vulkan version: weights repacked as above at load; activations transposed to fp16 in a
        uniform texel buffer (`texelFetch`, the texture path); `float16_t` arithmetic
        (`VK_KHR_shader_float16_int8`), accumulating each 32-input block in f16 and adding it
        into f32 to bound the error over long rows (llama.cpp accumulates whole rows in f16).
        Check against the batch goldens with an fp16 bound, then PPL through `vk_ppl … batch`.
        Written 2026-10-30 while the phone was away, checked on the PC's Intel GPU only
        (results/phase-l3.md step 9): `shaders/q4_0_gemm_h.comp`, `to_f16t.comp`, `MatQ4_0T`,
        `Info::fp16` (features turned on where the device has them), used by
        `GpuModel::forward_batch` unless `Options::f32_gemm`. Activations through a storage
        buffer for now (the texel-buffer path is a next step if the phone wants it). Test
        `q4_0_gemm_h_matches_reference` (rel ≤ 1.7e-3, bound 4 · 2⁻¹¹); PPL through it SmolLM2
        27.6961 (f32 27.6910), Qwen2.5 19.7256 (f32 19.7193; llama.cpp 27.7412 / 19.7283).
        To do on the phone: `vk_gemm` (f16 vs f32 GFLOP/s), `vk_ppl … batch`, `vk_prefill`,
        and whether the app's GPU mode should now run the prompt on the GPU instead of the CPU.
  - [x] Q4_K_M files on the GPU, decode and prompt (2026-10-30; phone numbers in
        results/phase-l3.md step 11: PPL equal to the PC's, prompt 563 tok/s at 128 in f16):
        `shaders/qx_gemv.comp` takes every format of the mix (Q4_0, Q5_0, Q8_0, Q4_K, Q6_K) with
        the Q4_0 shader's lane layout; `MatQ` lays each 32-value piece out as nibble pairs (int8s
        for Q8_0), K-quant scales per super-block, so each reads close to its file size. Test
        `qx_gemv_matches_reference` (all five, 21 golden cases, against numpy). `GpuModel`
        keeps the tuned Q4_0 shader for Q4_0 matrices and splits q/k/v or gate/up into parts
        when their formats differ (Q4_K_M has `attn_v` in Q8_0 on half the layers). PPL on the
        PC: SmolLM2 20.6876, Qwen2.5 16.7167 (llama.cpp 20.7784 ± 0.85 / 16.7657 ± 0.69).
        Prompt pass: `shaders/qx_gemm.comp` (q4_0_gemm's f32 tiles, any format, output row
        stride for parts); test `qx_gemm_matches_reference` (20 cases); PPL through it the same
        as the decode step. On the Intel GPU it is no faster than decode (f32 tiles ~15
        GFLOP/s there). To do: phone numbers; an f16 multi-format GEMM if the phone needs it.
- [ ] **L.5** NPU (Hexagon v79 DSP), from scratch (agreed 2026-10-29). Toolchain: Hexagon SDK
      6.4.0.2 in `~/.hexagon-sdk/6.4.0.2` (public release `snapdragon-toolchain/hexagon-sdk`):
      Hexagon clang for the DSP side, `qaic` for the FastRPC glue. llama.cpp's `ggml-hexagon` is the
      reference for platform use (and its numbers to measure against: pp ~330, tg ~74 tok/s,
      results/phase-l3.md step 8), not code to copy. Unsigned DSP code runs on this phone (step 8).
  - [x] Step 1, hello (results/phase-l5.md; ~146 µs per call, copies at 2.5 GB/s): a DSP library (`dsp/`, C) with one call that adds two arrays with HVX in
        a buffer shared with the CPU (`rpcmem` / `fastrpc_mmap`); the Rust side opens the
        unsigned `cdsp` session through `libcdsprpc.so` (dlopen, as `ash` does for Vulkan),
        checks the result, and times a round trip.
  - [x] Step 2, memory (DRAM via HVX 46–47 GB/s at 4–5 threads; DMA into VTCM in step 4b): how fast HVX threads read DRAM (cf. CPU and GPU ~66 GB/s), and VTCM.
  - [x] Step 3, decode GEMV (`vrmpy`, rel err ≤ 4.8e-7 vs llama.cpp bytes; ~10.5 ms of products per token): Q4_0 × f32 (or ×Q8_0) on HVX across the DSP's hardware threads,
        checked against llama.cpp's bytes / numpy as the GPU kernels were; bandwidth per shape.
  - [x] Step 4, a whole decode step on the DSP (one call per token; PPL 27.7429 / 19.7584 vs
        llama.cpp 27.7412 / 19.7283; Qwen 10.5 ms a token, 81 tok/s, vs llama.cpp NPU 74),
        PPL against llama.cpp. (A request queue as `dspqueue` was not needed: one FastRPC call
        per token is ~150 µs.)
  - [x] Step 4b, weights by DMA into VTCM (products ~36 → ~60 GB/s; Qwen 6.75 ms a token,
        123 tok/s; SmolLM2 271 tok/s; PPL unchanged). Attention at long positions is
        profiled (results/phase-l5.md, depths 0 / 512 / 2048): the growth is the attention stage.
  - [x] Step 5a, Q4_0 matrix-matrix product on HMX (fp16 tiles in VTCM; HVX threads dequantize
        while one drives HMX; Qwen's layer products 1512 tok/s at 128 tokens, 2413 at 256).
  - [x] Step 5b, prefill on the DSP through the whole model with HMX, PPL again (phone
        2026-10-30: PPL 27.6893 / 19.7174, 1608 / 778 tok/s at 128; results/phase-l5.md). Written
        while the phone was away (2026-10-30):
        - `model_prefill` (dsp/model.c): up to `batch` tokens per call, every layer; products on
          HMX through `hmx_rows` (all threads, shared barrier), norms / RoPE / KV store a run of
          tokens per thread, causal attention a run of (token, head) pairs, SwiGLU by blocks.
          Logits for every token on HMX (`all_logits`, perplexity) or the last on HVX.
          `DspModel::new(.., batch, all_logits)`, `DspModel::prefill`; `dsp_ppl … BATCH`
          (perplexity through the prefill, plus prefill tok/s at 32 / 64 / 128 / BATCH).
        - Attention: keys stored transposed (a row of every position per key element), so 32
          positions' scores are one vector of multiply-adds with no reduction per position.
          For the decode slowdown at long positions (72 tok/s over a 512-token run).
        - HMX dequantization: each worker l2fetches its share of the next chunk's Q4_0 bytes
          (one thread was waiting on DRAM, ~0.8 GB/s).
        - `DspSession` (prompt by prefills of 128, then decode), `aether MODEL --dsp`,
          `aether_open_dsp(path, dir, max_seq, out)` in the C ABI and `include/aether.h`. Not in
          the app's UI yet (JNI and a checkbox, after the CLI works).
        To check on the phone: all tests; `dsp_ppl` batch 128 on SmolLM2 and Qwen against
        llama.cpp's PPL (fp16 activations in the products may cost a little); the greedy text;
        decode PPL unchanged with the transposed keys; `dsp_gemm` again (prefetch); `--dsp`.
  - [x] Prompt lookup on the DSP (2026-10-30; phone: +6–28 %, text unchanged, results/phase-l5.md step 6):
        `model_few` (dsp/model.c) runs up to 8 tokens through every layer with every token's
        logits. Its products (`q4_0_rows_n_on`, dsp/q4_0_hvx.h) split each block of weights
        into nibbles once and use it for every token, adding the same block sums in the same
        order as the decode step, so the logits are the decode step's. The HVX product code
        moved into `dsp/q4_0_hvx.h` so `dsp/sim/rows_n.c` can run it on the SDK's simulator:
        `sim_q4_0_rows_n_matches_reference` checks it against llama.cpp's Q4_0 bytes (the
        quant_batch goldens) for 1, 3 and 8 tokens on the PC. `DspModel::few`,
        `DspSession::generate_lookup`, `aether --dsp --lookup K`, the C ABI's lookup for DSP
        sessions, `dsp_ppl … few`.
  - [x] Q4_K_M on the DSP (2026-10-30; phone: PPL 20.7241 / 16.7381, Qwen 92–98 tok/s): products for Q5_0,
        Q8_0, Q4_K and Q6_K (dsp/qx_hvx.h), each laid out like Q4_0 (32-row groups, a piece
        as whole vectors; `repack_qx`), checked on the simulator against llama.cpp's bytes
        (`sim_qx_rows_n_matches_reference`, 20 shapes, 1/3/8 tokens). A product is up to
        three runs of one format each (q/k/v and gate/up may mix formats). Decode uses DMA
        for every format (`qx_rows`); the few-token pass runs them too, and is the prompt
        pass for files not all Q4_0 (HMX prefill is Q4_0 only).

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
