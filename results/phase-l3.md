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

`aether MODEL.gguf --gpu` now decodes on the GPU (`--features vulkan`;
the prompt goes through token by token, since there is no batched GPU
pass yet). Qwen2.5-0.5B Q4_0, "Why is the sky blue?", greedy: 98 tok/s
decode on the GPU against 161 on the CPU, with a 35-token prompt in 541
against 59 ms. The two greedy texts agree for the first ~30 tokens, then
fork where two tokens are nearly tied. The GPU keeps activations in f32,
while the CPU rounds them to int8 before each product.

**In the app** (2026-10-29, phone charging, 44–49 °C): `aether_open_gpu`
in the C ABI (`GpuSession` in the library, shared with `aether --gpu`;
test `c_abi_gpu_session`), `Aether.openGpu` in Kotlin, and a "GPU"
checkbox. `libaether.so` is now built with `--features vulkan`. Qwen2.5-0.5B
Q4_0 chat, "Why is the sky blue?", t 0.7 (driven over adb):

| app mode | prefill (35 tokens) | decode |
| --- | --- | --- |
| CPU, cores 2–7 | 56 ms | 155.5 tok/s |
| CPU + lookup | 53 ms | 159.6 tok/s (prose: about neutral, as measured before) |
| GPU | 319–442 ms (token by token) | 90–92 tok/s |

**Prefill on the CPU, decode on the GPU** (2026-10-29, phone charging,
44–50 °C). Feeding the prompt through the GPU's decode step one token at a
time is slow, so `aether --gpu -c CORES` now also loads a CPU copy of the
model. That copy runs the prompt as one batch (`forward_batch`), and its KV
cache rows `0..n` are copied into the GPU's cache. Both caches use the same
`[layer, pos, kv_dim]` layout, about 24 KB per token for Qwen2.5-0.5B.
Decoding then continues on the GPU at position `n`. The weights are in
memory twice (+278 MB for Qwen2.5-0.5B Q4_0), so this is an option, not
the default.

Qwen2.5-0.5B Q4_0, greedy, a 345-token prompt (the start of wiki.test.raw),
two runs each (raw: `raw/hybrid-prefill.txt`):

| mode | prefill, 345 tokens | decode, 55 tokens |
| --- | --- | --- |
| `--gpu` (prompt token by token on the GPU) | 3873 / 4474 ms | 80.6 / 82.0 tok/s |
| `--gpu -c 2-7` (prompt on the CPU) | 606 / 548 ms | 63.3 / 64.4 tok/s |
| `-c 2-7` (all on the CPU) | 602 / 543 ms | 157.8 / 157.9 tok/s |

- The handoff is correct: on the short prompt ("Why is the sky blue?", 114
  tokens), `--gpu` and `--gpu -c 2-7` print the same greedy text, token for
  token. A wrong cache row would change the text within a few tokens.
- Prefill is 7× faster and equals CPU-only, as expected: it is the same
  code. The copy itself is ~8 MB and takes a negligible share of the time.
- Decode after the CPU prefill was 63–75 tok/s against 81–82 for GPU-only.
  The CPU threads are not the cause: they spin for ~2.8 ms and then park,
  and one prefill core (`-c 7`, 67 tok/s) did no better than six. More
  likely the GPU clocks dropped during the ~0.5 s it sat idle (as in step
  5). In GPU-only mode, the long token-by-token prefill keeps the GPU busy
  and its clocks up. A 55-token decode is short, so this start-up loss
  weighs heavily.
- Rerun on battery (not charging), cooler (37–38 °C), the modes
  alternating: decode 84.4 / 96.5 tok/s GPU-only against 75.0 / 93.5 with
  the CPU prefill. The gap shrank from ~20 % to 3–11 %. Most of it came
  from the warm, charging phone; what is left is within the run-to-run
  spread and still to be explained.
- Per-token times (`aether --gpu --times`; raw:
  `raw/hybrid-token-times.txt`), cooler still (34–35 °C, battery): decode
  83.7 / 89.1 tok/s GPU-only against 101.6 / 103.0 with the CPU prefill.
  The CPU-prefill runs hold 8–10 ms per token from the first token on,
  with one 23–27 ms token (the first token in one run, mid-way in the
  other). GPU-only runs sit at 11–12 ms per token after their 3.5 s of
  GPU prefill. So there is no slow start-up phase to fix. The earlier gap
  came from the phone's state (warm, charging). On a cool phone, the GPU
  decodes faster when it has not just been busy with the prompt.

## Step 6: what limits GPU decode (2026-10-29, on battery, 34–41 °C)

GPU decode of Qwen2.5-0.5B Q4_0 takes ~9–11 ms per token. Streaming its
278 MB of weights at the GPU's measured 66 GB/s would take ~4.2 ms. Where
does the rest go?

**The CPU's part is small.** `vk_decode split` times the step (submit and
wait) apart from the CPU's work after it. Copying the 152k logits out of
uncached GPU memory and taking the argmax costs 0.3–1.1 ms per token.
`vk_decode queued` records the whole step 8 times in one submit, so the
GPU runs 8 forward passes with no CPU between them
(`GpuModel::with_repeat`, timing only). That gives 7.6–8.6 ms per token,
the same as one step per submit (6.8–10.6 ms in plain runs). So the gap
between tokens is not what makes the GPU slow. The work inside the step
is. (Wall times swing ±20 % between runs: the GPU's clock state, which
an unrooted phone cannot fix.)

**The matrix products are ~75 % of the step,** and run at 30–40 GB/s
there (profile at depth 256: gate/up 2.6–3.3 ms, down 1.5–1.9, classifier
1.4–1.9 ms). On its own (`vk_gemv`, a barrier between dispatches as in
the model), the same kernel moves 41–47 GB/s on the 2.4 MB matrices and
62 GB/s on the 77 MB classifier. A dispatch costs ~10 µs beyond its bytes
(a 2.4 MB matrix at 66 GB/s would take 37 µs and takes ~50). That is a
large share for small matrices: wq reaches 28 GB/s, wk and wv 8.

Three changes tried on the GEMV (raw: `raw/vk-gemv-splitk.txt`,
`raw/vk-gemv-stacked.txt`, `raw/stacked-ab.txt`):

| change | stand-alone | kept? |
| --- | --- | --- |
| unroll the R-row loop (all R loads issued first) | slower everywhere (classifier 62 → 38 GB/s), probably register pressure | no |
| split-K: G groups of 32 lanes share rows (more workgroups for `down`) | never faster (down 50 → 48 / 45 GB/s at G = 2 / 4) | no |
| stack matrices of the same width: wq/wk/wv (1152 rows), w_gate/w_up (9728 rows) | q+k+v 32.2 → 20.4 µs, gate+up 104.6 → 93.3 µs | **yes** |

The stacked matrices are uploaded one under the other. Their outputs
share one buffer, and the shaders after them bind views at offsets into
it (`Kernel::bind_at`, offsets aligned to the device's
`minStorageBufferOffsetAlignment`). So `qkv_post`, `attention` and
`swiglu` are unchanged. Each row's arithmetic is unchanged as well, and
the GPU perplexities are identical to the last digit (Qwen2.5-0.5B
19.7193, SmolLM2-135M 27.6910; llama.cpp 19.7283 and 27.7412).

In the model, the gain is smaller than on its own. Measured against the
stages that did not change (down + classifier, as a clock reference),
over three alternating runs: q/k/v 0.192 → 0.154 (−20 %), gate/up 0.885
→ 0.832 (−6 %). That is ~0.3 ms of a ~9.5 ms token, about 3–4 %, which
the clock swings hide in plain wall times.

What is left: the matrix products run at about 60 % of the bandwidth
the GPU can reach, and each of the ~240 remaining dispatches costs a
fixed ~10 µs. Fusing the small stages (rmsnorm into the next product,
the residual add into wo and down) would remove ~100 dispatches.

## Step 7: the gpu-pending branch on the phone (2026-10-29, on battery, 33–42 °C)

The branch `gpu-pending` was built while the phone was away: fusions into
the GEMV, a batched GPU prompt pass, CPU prompt in the app. Correctness
had been checked on the PC's Intel GPU only. Raw:
`raw/gpu-pending-verify.txt`, `raw/vk-prefill.txt`, `raw/fusion-ab.txt`,
`raw/cli-modes.txt`, `raw/unfuse-ab.txt`, `raw/unfuse-ppl.txt`,
`raw/cli-hybrid-ab.txt`.

**Correctness on the Adreno.** 105/105 lib tests pass (`--features
vulkan`). GPU perplexity through the decode step and through the batched
pass is identical to the PC's and to step 4: SmolLM2-135M 27.6910,
Qwen2.5-0.5B 19.7193.

**Batched GPU prefill** (`vk_prefill`, Qwen2.5-0.5B Q4_0): 16 tokens
66 tok/s, 64 tokens 241, 128 tokens 251. The GEMM's tiles are 64 tokens
wide, so 16 tokens cost nearly as much as 64 (241 vs 263 ms). The matrix
products are nearly all of it: gate/up ~45 %, SwiGLU + down + add ~39 %.
At 64 tokens that is ~63 GFLOP in 0.26 s, about 240 GFLOP/s, a small
fraction of what the GPU can compute. The CPU's batched prefill does
750–800 tok/s, so the CPU keeps the prompt.

**Two fusions made decode slower.** Profile at depth 256, three runs
each, against the build before the fusions (`stacked`):

| stage | before (ms) | all fused (ms) | change |
| --- | --- | --- | --- |
| rmsnorm + q/k/v | ~0.14 + 0.55–0.57 | 0.77–0.78 | −0.1 |
| wo + add | 0.45–0.46 + 0.15 | 0.49 | −0.1 |
| rmsnorm + gate/up | ~0.14 + 2.98–3.10 | 3.57–3.58 | **+0.5** |
| SwiGLU + down + add | 0.10 + 1.78–1.85 + 0.15 | 2.46–2.47 | **+0.4** |
| GPU time per token | 9.31–9.63 | 10.37–10.38 | +8 % |

A fused GEMV loads its input in every subgroup. With the RMS norm fused,
each of the gate/up workgroups sums the 896 squares again before it
starts. With the SwiGLU fused, every subgroup of `down` computes all
4,864 `exp`s again. For q/k/v (1,152 rows) the redundant norm costs less
than the dispatch it saves; for gate/up (9,728 rows) it costs more.

**Un-fused again** (gate/up after a separate `rmsnorm`; `down` after a
separate `swiglu`, still adding into x; the norm stays fused into q/k/v
and the classifier, the add into wo and down). Same three-run profile:

| | GPU ms per token | step (submit + wait), depth 0 / 256 |
| --- | --- | --- |
| before the fusions | 9.31 / 9.63 / 9.60 | 10.47–10.57 / 10.50–10.66 |
| all fused | 10.38 / 10.37 / 10.37 | 11.72–13.38 / 11.50–11.55 |
| un-fused gate/up and down | **9.69 / 9.57 / 9.12** | 10.36–10.43 / 10.73–10.74 |

So the build is back where it was before the fusions. The two fusions
that are kept save ~0.2 ms, which is smaller than the run-to-run spread.
PPL is unchanged (27.6910, 19.7193).

**Greedy argmax on the GPU.** Copying Qwen's 151,936 logits to the CPU
and taking their argmax cost 0.9–1.2 ms per token (`vk_decode split`).
That is ~10 % of a token. Now two small dispatches after the classifier
(`argmax.comp`: 64 workgroups, then one) leave the index in a 4-byte
buffer (`GpuModel::greedy`), and `GpuSession::generate` reads only that
when the temperature is 0. Ties go to the lowest index, as on the CPU;
`gpu_argmax_matches_the_logits` checks it against the CPU's argmax of
the same logits after a batch and 16 steps. 106/106 lib tests pass on the
phone.

**The CLI** (`aether --gpu --times -t 0`, 345-token prompt, 55 tokens,
three alternating rounds, cooled below 40 °C before each run):

| mode | prefill tok/s | decode tok/s | before this step |
| --- | --- | --- | --- |
| `--gpu` (batched GPU prompt) | 240 | 97.4–97.7 | 77–80 |
| `--gpu -c 2-7` (CPU prompt, six cores) | 761–820 | 95.9–101.0 | 63–66 |
| `--gpu -c 7` (CPU prompt, one core) | 218–220 | 66.5–86.3 | — |
| `-c 2-7` (CPU only) | 750–773 | 166–169 | |

GPU decode got ~25 % faster (un-fusing plus the argmax). The gap after a
CPU prompt that `raw/cli-modes.txt` showed earlier today (63–66 against
77–80) is gone: the six-core CPU prompt is followed by 12–15 ms for its
first ~5 tokens, then ~9.6 ms per token. After a one-core prompt (1.6 s
of one prime core at full load) decode holds ~15 ms per token for 15–40
tokens, then drops to ~10 ms in one step. A step like that looks like a
clock governor changing level, not our code: per token the work is the
same. An unrooted phone does not show which clock it was.

**The app** (rebuilt from this branch; the GPU mode runs the prompt on
the CPU): 41-token prompt in 46–52 ms, GPU decode 89.3 / 92.2 tok/s with
sensible text; CPU mode 172.1 / 172.8 tok/s. That was measured before the
un-fusing and the argmax.

What is left for GPU decode: the CPU still decodes ~1.7× faster. The
matrix products run at 30–40 GB/s of the GPU's 66.

## Step 8: every engine on this phone, against llama.cpp (2026-10-29, 21:20, battery 85 %)

Qwen2.5-0.5B Q4_0, two rounds in opposite orders, cooled below 40 °C
before each run, screen kept awake (clock caps 2.75 / 2.65 GHz in round 1,
2.40 / 2.44 in round 2). llama.cpp `d00685831` built three ways: CPU,
OpenCL (its Adreno backend, linked against the phone's `libOpenCL.so`)
and Hexagon (its NPU backend, Hexagon SDK 6.4.0.2), `llama-bench -t 6
--cpu-mask 0xfc --mmap 0 -r 2`. Ours: `bench_suite -c 2-7` and `aether
--gpu` (345-token prompt, 55 tokens; our GPU prompt pass is batched by
128, so it is compared with pp256 / pp512). tok/s, round 1 / round 2
(raw: `raw/engines-vs-llamacpp.txt`):

| engine | pp64 | pp256 | pp512 | decode |
| --- | --- | --- | --- | --- |
| **ours, CPU** | 749 / 832 | 973 / 852 | 855 / 770 | 192 / 194 (tg64) |
| llama.cpp, CPU | 930 / 831 | 937 / 843 | 739 / 661 | 172 / 166 |
| **ours, GPU** | | 267 / 240 (345 tok) | | 106.5 / 101.6 |
| **ours, CPU prompt + GPU decode** | | 876 / 804 (345 tok) | | 109.9 / 106.0 |
| llama.cpp, OpenCL GPU | 689 / 646 | 1387 / 1216 | 1347 / 1194 | 58 / 49 |
| llama.cpp, Hexagon NPU | 295 / 296 | 330 / 327 | 313 / 311 | 74.5 / 74.0 |

- **CPU**: we decode 11–17 % faster and prefill 4–16 % faster from 256
  tokens on; llama.cpp is ahead at pp64 in round 1 (930 vs 749), level in
  round 2.
- **GPU decode**: ours is ~2× llama.cpp's OpenCL (106 vs 58).
- **GPU prefill**: llama.cpp's OpenCL is ~5× ours (1,387 vs 267) and
  faster than either CPU. Its Adreno Q4_0 kernel (`mul_mat_Ab_Bi_8x4.cl`)
  reads the activations as an fp16 image (the texture path) and multiplies
  and accumulates in fp16, 8 × 4 outputs per thread; ours is f32 and
  reaches ~240 GFLOP/s (step 7). This is the largest gap left.
- **NPU** (llama.cpp only): runs the whole model on the Hexagon DSP with
  correct text ("The capital of France is Paris. It is the largest city in
  France…"); 74 tok/s decode, ~330 tok/s prefill. Its prompt products run
  on HMX (weights expanded to fp16 tiles in VTCM), decode on HVX. During
  its runs the phone heated to 68 °C (its CPU threads poll), against
  47–57 °C for the other engines.
- **Unsigned code on the NPU works on this phone**: llama.cpp opens a
  `cdsp` session with `DSPRPC_CONTROL_UNSIGNED_MODULE` and loads its own
  `libggml-htp-v79.so`. So L.5 can run our own DSP code without signing.

## Step 9: an f16 prefill shader, on the PC only (2026-10-30)

The phone was away; everything here ran on the PC's Intel GPU (RPL-P),
whose numbers say nothing about the Adreno's speed. The shader follows
the way llama.cpp's Adreno kernel is organized (step 8): no shared
memory, 4 rows × 8 tokens per thread, weights repacked so neighbouring
threads read neighbouring words (`MatQ4_0T`), activations transposed to
f16 first (`to_f16t.comp`, with the SwiGLU for down), arithmetic in f16
within each 32-input block, then summed in f32.

**Correct**: `q4_0_gemm_h_matches_reference` on the batch goldens, rel
error 0 to 1.67e-3 (f16's 2⁻¹¹ is 4.9e-4; bound 4 · 2⁻¹¹). Perplexity
through `GpuModel::forward_batch` (`vk_ppl … batch`, 20 chunks of 512):

| model | f32 products | f16 products | llama.cpp |
| --- | --- | --- | --- |
| SmolLM2-135M | 27.6910 | 27.6961 | 27.7412 ± 1.177 |
| Qwen2.5-0.5B | 19.7193 | 19.7256 | 19.7283 ± 0.836 |

**On the PC** (`vk_gemm 256`, Intel; GFLOP/s, f32 → f16): wq/wo 14 → 79,
gate/up 16 → 109, down 14 → 80; SmolLM2's perplexity run 217.8 → 50.0 s.
The phone's numbers are the ones that count (llama.cpp's OpenCL: 1,387
tok/s at pp256 against our 267).

## Step 10: Q4_K_M files on the GPU, decode (2026-10-30, PC only)

A llama.cpp Q4_K_M file of these models is a mix of formats (results/phase-6.md):

| Matrix | Format |
|---|---|
| q, k, o, gate, up | Q5_0 |
| v | Q5_0 or Q8_0, by layer |
| down | Q4_K or Q6_K, by layer |
| embedding (= classifier) | Q8_0 |

`shaders/qx_gemv.comp` reads all of these plus Q4_0. Its work split is
`q4_0_gemv.comp`'s: lanes over 32-value pieces, rows in registers, shuffles
to add.

At upload (`MatQ`), every piece is laid out like a Q4_0 block: 16 bytes,
byte k = value k | value k+16 << 4, or 32 int8s for Q8_0. The extra bits
come after:
- Q5_0: one uint of fifth bits per piece.
- Q6_K: two uints of 2-bit highs per piece.

The K-quants' scales stay per super-block, unpacked to bytes, so each
format reads close to its file size (Q4_K 0.58 bytes a weight against 0.56;
Q6_K 0.83 against 0.82).

`GpuModel` keeps the tuned Q4_0 shader for Q4_0 matrices. When q/k/v or
gate/up differ in format, they are split into parts that write their rows
at a byte offset of the shared output. The prompt pass stays Q4_0-only.

**Test:** `qx_gemv_matches_reference` runs all five formats against
X @ dequant(W)ᵀ in float64 from numpy:
- Q4_0, Q5_0 and Q8_0 use llama.cpp's quantizer bytes (12 cases).
- Q4_K and Q6_K use random blocks dequantized by the gguf package (9 cases,
  every token). `tools/gen_quant.py` now also writes `out_q4_k_a` and
  `out_q6_k_a`.
- Bound 1e-6 · √cols, as for Q4_0.

**Perplexity** through the GPU decode step (`vk_ppl`, wikitext-2, 20 × 512,
Intel RPL-P):

| Model | GPU (ours) | CPU (ours, phone) | llama.cpp |
|---|---|---|---|
| SmolLM2-135M Q4_K_M | 20.6876 | 20.7201 | 20.7784 ± 0.851 |
| Qwen2.5-0.5B Q4_K_M | 16.7167 | 16.7088 | 16.7657 ± 0.688 |
| SmolLM2-135M Q4_0 (regression) | 27.6910 | | unchanged from step 7 |
| Qwen2.5-0.5B Q4_0 (regression) | 19.7193 | | unchanged from step 7 |

Our CPU quantizes the activations to Q8_0; the GPU keeps them in f32. Both
are within llama.cpp's error bars.

**Speed** (Intel, only to see that the cost follows the bytes): Qwen decode
33.3 ms a token for Q4_K_M (392 MB) against 22.8 ms for Q4_0 (278 MB), a
ratio of 1.46 for 1.41× the bytes. The Adreno numbers wait for the phone.

`bench_suite --backend cpu|gpu|dsp` and `scripts/backends-bench.sh` now
run the same measurements on every engine.

**Prompt pass for Q4_K_M** (same day): `shaders/qx_gemm.comp` is
`q4_0_gemm.comp`'s tiling (64 rows × 64 tokens, f32) with each piece
dequantized from `MatQ`'s layout, plus an output row stride, so a q/k/v
part writes into the stacked row. A Q4_0 model keeps `q4_0_gemm` /
`q4_0_gemm_h`; any other model now runs `forward_batch` with `qx_gemm`
for every part.

Test `qx_gemm_matches_reference`: 11 batch goldens (Q4_0, Q5_0, Q8_0) and
9 K-quant goldens, bound 1e-6 · √cols.

Perplexity through the batched pass (`vk_ppl … batch`, PC):

| Model | Shader | PPL |
|---|---|---|
| SmolLM2 Q4_K_M | `qx_gemm` | 20.6876 (decode step: 20.6876) |
| Qwen2.5 Q4_K_M | `qx_gemm` | 16.7167 (decode step: 16.7167) |
| SmolLM2 Q4_0 | `q4_0_gemm_h` | 27.6961 (unchanged) |
| Qwen2.5 Q4_0 | `q4_0_gemm_h` | 19.7256 (unchanged) |

Speed on the Intel GPU (`vk_gemm 128`, GFLOP/s):

| Shape | `q4_0_gemm` f32 | `q4_0_gemm_h` f16 | `qx_gemm` Q4_0 | `qx_gemm` Q5_0 |
|---|---|---|---|---|
| wq/wo | 14 | 46 | 8 | 13 |
| gate/up | 16 | 94 | 9 | 15 |
| down | 14 | 47 | 8 | 14 |

`qx_gemm` on Q5_0, most of a Q4_K_M file, runs at the f32 Q4_0 shader's
speed. On this GPU the f32 tiles are slow: Qwen Q4_K_M's prompt runs at
19 tok/s at 128 tokens (`vk_prefill`), below its decode (30 tok/s token by
token). On the Adreno the f32 Q4_0 shader beat decode (251–267 against
~100 tok/s, step 7), so whether `qx_gemm` helps is a phone question.

Next, if it does not help: an f16 multi-format shader like
`q4_0_gemm_h`. (`qx_gemm`'s Q4_0 case is slower than its Q5_0 case; a
mixed file has no Q4_0 matrices, and Q4_0 models do not use it.)

**An f16 prompt shader for every format** (same day, PC only):
`shaders/qx_gemm_h.comp` is `q4_0_gemm_h.comp`'s organization (4 rows × 8
tokens a thread, no shared memory, f16 sums per 32-value piece added into
f32) for all five formats. It needs its own copy of each matrix (`MatQT`),
laid out as `MatQ4_0T` is: per piece, the words of 4 neighbouring rows side
by side.
- Low 4 bits: 4 words, value 8h + j in bits 4j of word h.
- Q5_0: one more word of fifth bits.
- Q6_K: two more words of 2-bit highs.
- Q8_0: 8 words of int8s.
- Scales: one f16 per piece (Q4_0, Q5_0, Q8_0). The K-quants' are
  multiplied out per piece and rounded to f16: Q4_K's d · scale and dmin ·
  min, Q6_K's two d · scale.

`GpuModel` uses it for the prompt of any model that is not all Q4_0 when
the GPU has fp16 (`batch-f32` keeps `qx_gemm`), one part per format as in
decode.

Test `qx_gemm_h_matches_reference`, on the same goldens as `qx_gemm`:
- 32-value formats: max error < 4 · 2⁻¹¹ of the largest output, as for
  `q4_0_gemm_h`; measured up to 1.7e-3.
- K-quant random blocks: one output (1x256) is a sum that cancels, with
  error 2.3e-2 of itself, because Q4_K's mins make a whole row lean one
  way. So each output is checked against Σ|w|·|x| (W from the gguf
  package's dequantization), the scale its f16 roundings grow with: bound
  2⁻¹¹, measured up to 0.53 · 2⁻¹¹.

Perplexity through the f16 batched pass (`vk_ppl … batch`, Intel):

| Model | `qx_gemm_h` (f16) | `qx_gemm` (f32) | llama.cpp |
|---|---|---|---|
| SmolLM2 Q4_K_M | 20.6939 | 20.6876 | 20.7784 ± 0.851 |
| Qwen2.5 Q4_K_M | 16.7150 | 16.7167 | 16.7657 ± 0.688 |

Speed on the Intel GPU (`vk_gemm 128`, GFLOP/s; f16 columns exclude the
activations' transpose):

| Shape | `q4_0_gemm_h` | `qx_gemm_h` Q4_0 | `qx_gemm_h` Q5_0 | `qx_gemm` Q5_0 |
|---|---|---|---|---|
| wq/wo | 46 | 51 | 33 | 14 |
| gate/up | 94 | 107 | 70 | 15 |
| down | 47 | 51 | 33 | 15 |

Q5_0 costs a third more than Q4_0 here: one more word per piece, and one
more shift, mask and OR per weight for the fifth bit.

Qwen Q4_K_M prompt (`vk_prefill`, Intel):

| Tokens | f32 `qx_gemm` | f16 `qx_gemm_h` | token by token |
|---|---|---|---|
| 128 | 19 tok/s | 83 tok/s | ~30 tok/s |

So on this GPU the prompt pass now beats decode. Whole-file perplexity
runs went from 238 s / 738 s to 67 s / 169 s (SmolLM2 / Qwen). The Adreno
numbers wait for the phone.
