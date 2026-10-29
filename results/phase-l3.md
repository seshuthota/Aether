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

Next: a Q4_0 matrix-matrix shader for prefill, where the GPU's arithmetic
should matter, then a whole forward pass on the GPU.
