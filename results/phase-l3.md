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

What the device reports (`vk_bandwidth`):

| | |
| --- | --- |
| GPU | Adreno (TM) 830, Vulkan 1.3.284, driver 0x8032004e |
| subgroup (wave) size | 64 |
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

Next: a Q4_0 matrix-vector shader, checked against the llama.cpp goldens
the CPU kernels use, and its speed against the CPU GEMV.
