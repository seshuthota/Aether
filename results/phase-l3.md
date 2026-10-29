# L.3: Vulkan compute on the Adreno GPU

Later phase from plan section 13 ("Vulkan Compute kernels for matmul and
attention"). Built with `--features vulkan`; the default build does not
change. Raw output in `phase-l3/raw/`.

## Step 1: the device, and how fast it reads memory (2026-10-29)

`kernels::vulkan` (the module sits under `kernels` for now: `src/lib.rs`
is being edited on master by the learning track) opens the GPU, allocates
buffers in memory both the CPU and the GPU can see, and runs compute
shaders from `shaders/*.comp` (GLSL, compiled to SPIR-V by `build.rs` with
the NDK's `glslc`) timed by GPU timestamps. Vulkan's C API comes from the
`ash` crate (declarations only; it loads the system's `libvulkan.so`).
Test `copy_shader_copies` (a buffer through the GPU and back, odd size).

What the device reports (`vk_bandwidth`, `raw/vk-bandwidth-g512.txt`):

| | |
| --- | --- |
| GPU | Adreno (TM) 830, Vulkan 1.3.284, driver 0x8032004e |
| subgroup (wave) size | 64 |
| subgroup operations | basic, vote, arithmetic, ballot, shuffle, shuffle relative, quad, rotate (no clustered) |
| max workgroup | 1024 invocations |
| memory | heap 0: 15,110 MB device-local (the phone's RAM), heap 1: 4 GB protected |
| memory types a storage buffer may use | 0 (device-local only) and 6 (device-local + host-visible + coherent + cached) |
| timestamp tick | 52.08 ns (19.2 MHz) |

So "GPU memory" is the phone's RAM: one heap, the same LPDDR5X the CPU
reads. A buffer is one allocation that both sides use, with no copies.

Bandwidth (GB/s, median of 5 trials of ~20 ms after 0.3 s warm-up; a
copy counts its bytes twice):

| buffer | read, 256 groups | read, 512 | read, 2048 | copy, 512 | copy, 2048 |
| --- | --- | --- | --- | --- | --- |
| 1 MB | 48.9 | 41.7 | 16.5 | 64.0 | 29.3 |
| 4 MB | **67.9** | 67.1 | 48.2 | 70.6 | 62.6 |
| 16 MB | 65.5 | 66.5 | **68.5** | 67.2 | 67.4 |
| 64 MB | 65.1 | 66.2 | 66.2 | 65.4 | 63.0 |
| 256 MB | 40.6 | 55.7 | 62.5 | 27.9 | 62.9 |

- **The GPU reads DRAM at ~66 GB/s, as fast as all eight CPU cores
  (~67 GB/s, Phase 1).** They share one memory controller, so for decode
  (one pass over the weights per token) the GPU cannot be faster than the
  CPU. Where it can help: prefill (compute-bound), and taking the work off
  the CPU (heat and energy).
- An empty dispatch with a barrier after it costs **1.7 µs**, plus about
  18 ns per workgroup (256 groups 6.1 µs, 2048 groups 40 µs). A decode
  step of Qwen2.5-0.5B is ~170 matrix products and other ops; at a few µs
  each that is well under a millisecond, small next to the CPU pool's
  ~120 handoffs.
- Small buffers want few workgroups (each must have work); large ones
  more. The 256 MB copy at 512 groups (28 GB/s) and read at 256 groups
  (41 GB/s) are low, and not yet explained.
- The first version of the read shader wrote one 16-byte sum per
  invocation: 2 MB of writes per dispatch at 512 groups, which made small
  reads look slow. It now writes only when the sum is −1, which this data
  never gives.

## Step 2: a Q4_0 matrix-vector shader (2026-10-29; `raw/vk-gemv-r*.txt`)

`shaders/q4_0_gemv.comp`, `MatQ4_0` (upload), `Gpu::q4_0_gemv`. llama.cpp's
18-byte blocks are split at upload into the 16 nibble bytes (one `uvec4`
load) and the f16 scales, so every load is aligned. The activation stays
f32 (no int8 re-quantization as on the CPU), so the reference is the "a"
golden: dequant(W) @ x in float64. Test `q4_0_gemv_matches_reference`
runs llama.cpp's own `w_q4_0.bin` bytes for all 12 shapes, at R = 4, 8, 12.

How it got to the bandwidth limit (GB/s of Q4_0 bytes, Qwen2.5-0.5B
shapes):

1. **One row per 64 lanes**, half a block (8 bytes) per lane, the 64
   partial sums added in shared memory (6 barriers): 15–25 GB/s, half the
   CPU's. A 896-column row has only 56 half-blocks, so most lanes did one
   load and then waited on barriers.
2. **32 lanes per R rows, a whole block per lane**, partial sums added by
   `subgroupShuffleXor` butterflies (no barriers), and each lane's 32 x
   values loaded once for all R rows: 29–46 GB/s, level with the CPU.
   (A first try used `subgroupClusteredAdd`. Adreno 830 does not support
   clustered subgroup ops (`supported_operations` lacks CLUSTERED), and
   its driver's shader compiler crashed with SIGSEGV in
   `vkCreateComputePipelines` instead of reporting that.)
3. **R as a specialization constant**, so each matrix gets its best R. R
   floats in an array were 15 % slower than R/4 `vec4`s (one shuffle
   moves four sums), so the accumulators are `vec4`s.

| matrix (rows × cols) | GPU R=4 | R=8 | R=12 | R=16 | CPU, 6 threads |
| --- | --- | --- | --- | --- | --- |
| wq/wo 896 × 896 (0.45 MB) | **22–29** | 20–25 | 20–24 | 18 | 29–31 |
| wk/wv 128 × 896 (0.06 MB) | **8–9** | 6 | 4.5 | 3.5 | 10 |
| gate/up 4864 × 896 (2.5 MB) | 41 | **47** | 43 | 37 | 39–45 |
| down 896 × 4864 (2.5 MB) | **43** | 40–42 | 38 | 32 | 41 |
| classifier 151936 × 896 (77 MB) | 46 | 58 | 61 | **62** | 50–51 |

- On the big classifier the GPU reaches 62 GB/s, 94 % of what its read
  test gets (66) and 24 % above the CPU. The layers' matrices are about
  level with the CPU. Small ones (wk/wv: 64 KB, 7.5 µs) are dominated by
  the fixed cost of a dispatch and too few workgroups to fill the GPU.
- Summed over one decode step's matrices, GPU ≈ 24 × 208 µs + 1.23 ms ≈
  6.2 ms against the CPU's ≈ 6.5 ms for the same matrices one at a time
  (its real decode step, with fused jobs, is ~5 ms). As step 1 predicted:
  both read the same memory at the same speed, so the GPU does not make
  decode faster. What it can do is take the work off the CPU, and prefill.

## Step 3: a Q4_0 matrix-matrix shader for prefill (2026-10-29; `raw/vk-gemm.txt`)

`shaders/q4_0_gemm.comp`, `Gpu::q4_0_gemm`. Test `q4_0_gemm_matches_reference`
runs llama.cpp's Q4_0 bytes of the 11 batch cases (odd row and token
counts) against X @ dequant(W)ᵀ in float64 (`out_q4_0_a`, now also written
by `tools/gen_quant.py` for the batch cases, together with the raw weight
bytes; the existing goldens are byte-identical).

Prefill is compute-bound, so this is where a GPU should help. The steps
(GFLOP/s = 2 · rows · cols · tokens / time; gate/up, 4864 × 896, 256
tokens):

| version | GFLOP/s |
| --- | --- |
| 1. GEMV-style: 32 lanes × 4 rows × T tokens, x reread from cache per row | 65 |
| 2. 64 × 64 tile in shared memory, 4 × 4 outputs per thread, `dot()` | 180 |
| 2b. two blocks per barrier pair (all 32 KB of shared memory) | 166 |
| 3. 8 × 8 outputs per thread (64 threads) | 207 |
| 4. outer products: tiles stored by column, `vec4` multiply-adds | **264** |
| 4b. activations read so neighbouring threads read neighbouring floats | 169 (stores to shared memory then collide in one bank) |

Now (median of 5, GPU vs CPU `linear_batch_on`, int8 + `smmla`, six
threads; the CPU's numbers moved between runs with its clock caps):

| tokens | wq/wo 896 × 896 | gate/up 4864 × 896 | down 896 × 4864 | CPU (same three) |
| --- | --- | --- | --- | --- |
| 64 | 157 | 215 | 160 | 205 / 478 / 506 |
| 256 | 210 | 264 | 209 | 277–389 / 419–779 / 421–769 |
| 512 | 249 | 263 | 251 | 688 / 732 / 647 |

The GPU in f32 reaches ~260 GFLOP/s, a third to a half of the CPU's
int8 `smmla` path. The limit is in the inner loop, not the loads (vec4
activation loads changed nothing). It is likely occupancy: 64 accumulators
per thread leave few waves to hide shared-memory latency. Ways to go
further are known but not taken yet: f16 arithmetic
(`VK_KHR_shader_float16_int8` is there, at double rate, at a cost in
accuracy), int8 dot products (`VK_KHR_shader_integer_dot_product`, like
the CPU), and smaller per-thread tiles with more threads. There is no
cooperative-matrix extension on this driver.

## Step 4: a whole decode step on the GPU (2026-10-29; `raw/vk-ppl-*.txt`)

`model::gpu::GpuModel` runs [`Transformer::forward`]'s decode step as
shaders, for `--pure` Q4_0 GGUFs. The new shaders are `rmsnorm`, `qkv_post`
(biases, RoPE, KV-cache store), `attention` (one workgroup per query
head, grouped-query), `add` and `swiglu`, around the Q4_0 GEMV. All
buffers are shared memory. One token is ~14 dispatches per layer with a
barrier after each step, recorded **once** into a command buffer that is
resubmitted for every token (`Gpu::record` / `Gpu::submit`; descriptor
sets are bound up front with `Kernel::bind`). What changes per token lives
in a `params` buffer: the position, and the RoPE (sin, cos) table,
computed on the CPU by the same function the CPU path uses, so the GPU's
own `sin`/`cos` at large angles never matter. The token's embedding row
is dequantized by the CPU straight into the input buffer.

**Correctness** against llama.cpp (`llama-perplexity`, same GGUF files,
wikitext-2, 20 × 512, `examples/vk_ppl.rs` feeding every token through
the GPU decode step):

| model (`--pure` Q4_0) | GPU | llama.cpp | ours, CPU |
| --- | --- | --- | --- |
| SmolLM2-135M | **27.6910 ± 1.174** | 27.7412 ± 1.177 | 27.7150 |
| Qwen2.5-0.5B (biases, 7:1 grouped-query attention, halves RoPE) | **19.7193 ± 0.835** | 19.7283 ± 0.836 | 19.7438 |

Both agree within llama.cpp's error bar. The GPU's activations stay f32
(no int8 rounding before each matrix product), which may be why both
GPU numbers are a little lower than the CPU's. Greedy text: "The capital
of France is Paris. It is the largest city in the world…" (Qwen).

**Speed** (first, before any tuning): 98 tok/s for Qwen at short context,
52 tok/s averaged over positions 0–510; SmolLM2-135M 106–122 tok/s. The
CPU does ~180 and ~400. Step 5 looks at why.

## Step 5: where the GPU's decode time goes, and energy per token (2026-10-29; `raw/energy-gpu-vs-cpu.txt`)

`examples/vk_decode.rs` records the step with a GPU timestamp after every
stage (`Gpu::record_timed`, `GpuModel::profile`). Qwen2.5-0.5B Q4_0, ms
per token summed over the 24 layers:

| stage | depth 0 | depth 256 | depth 1024 |
| --- | --- | --- | --- |
| gate/up | 4.20 | 3.62 | 5.10 |
| down | 2.30 | 1.99 | 2.71 |
| classifier | 2.46 | 2.10 | 3.09 |
| q/k/v + wo | 1.41 | 1.25 | 1.60 |
| attention (one workgroup per head) | 0.51 | 1.58 | **5.59** |
| rmsnorm, add, swiglu, bias + RoPE | 1.07 | 0.98 | 1.08 |
| total on the GPU | 11.95 | 11.52 | 19.16 |

Two findings:

1. **The matrices run at about half their stand-alone speed.** The
   classifier takes 2.46 ms here and 1.23 ms in `vk_gemv`. The reason is
   the clocks. With the GPU idle for a few milliseconds before a single
   dispatch, the same GEMV is 1.5–2.7× slower (`vk_gemv`; classifier
   1.23 → 1.66 / 2.70 / 3.34 ms after 1 / 5 / 20 ms idle; gate/up 60 →
   95 µs). The GPU's (and memory bus's) frequency governors scale with
   recent load, and decode is bursty: one token's dispatches, then the
   CPU's turn. Vulkan has no way to ask for clocks, and the sysfs
   controls (`/sys/class/kgsl/kgsl-3d0/*`) are not readable without
   root. Run to run, the same decode measurement varies from 50 to 105
   tok/s.
2. **Attention with one workgroup per head leaves the GPU mostly idle.**
   It is now split like the CPU's flash-decoding: workgroups (head,
   64-position chunk) write (max, sum, partial output), and a merge
   shader combines them (`attention.comp`, `attention_merge.comp`). The
   dispatch is recorded once for every chunk the cache can hold, and
   chunks past the position return at once. Depth 1024: 5.59 → 1.95 ms.
   PPL is unchanged (19.7193, 27.6910), and the average speed over the
   PPL runs went from 52.5 to 80.5 tok/s (Qwen) and from 87.6 to
   122.4 (SmolLM2-135M).

The GPU timestamps themselves cost ~2–3 ms per token, so speeds are
measured without them.

**Sustained decode and energy** (Qwen2.5-0.5B Q4_0, 120 s each, 64-token
bursts from position 0, on battery, each run starting below 42 °C,
`raw/energy.sh`):

| | tok/s | power (charge counter / sampler) | energy per token | max temp |
| --- | --- | --- | --- | --- |
| CPU, cores 2–7 | **157.6** | 4.55 / 4.28 W | 0.029 J | 88.8 °C |
| GPU, CPU polls the fence | 86.4 | 2.37 / 1.93 W | 0.027 J | 64.6 °C |
| GPU, CPU sleeps on the fence | 102.7 | 2.17 / 2.48 W | **0.021–0.024 J** | 67.6 °C |

- **The GPU path uses about half the power and stays 20 °C cooler.** At
  0.65× the CPU's speed it spends less energy per token. On a phone that
  is a real choice: the CPU when speed matters, the GPU for long
  generations or to leave the CPU to the app.
- Polling the fence made the GPU path slower as well as costlier. The
  polling core probably competes with the GPU for the shared power and
  thermal budget. `GpuModel` now sleeps on the fence by default.
- The charge-counter windows are short (42–89 s, the counter moves in
  steps), so the energy figures are ±15 %.
