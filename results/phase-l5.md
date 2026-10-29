# L.5: our own code on the Hexagon NPU

Later phase from plan section 13 ("NPU exploration"). The NPU of the
Snapdragon 8 Elite is the Hexagon v79 compute DSP (cDSP), with vector
units (HVX, 128-byte vectors) and a matrix unit (HMX). Everything that runs
on it here is ours: `dsp/aether_dsp.c` (C, built with the Hexagon SDK
6.4.0.2's clang) and `src/kernels/hexagon.rs`. The SDK provides the
toolchain, QuRT (the DSP's OS) headers and `qaic`, which generates the
DSP-side glue from `dsp/aether_dsp.idl`. Built with `--features hexagon`.
Raw output in `phase-l5/raw/`.

## Step 1: a library of ours on the DSP (2026-10-29; `raw/dsp-hello.txt`)

The CPU side loads the phone's `libcdsprpc.so` with `dlopen`, marks the
session as running an unsigned module, and opens
`file:///libaether_dsp.so?aether_dsp_skel_handle_invoke&_dom=cdsp`. The
library is embedded in our binary and written next to it, and that
directory is put first in `ADSP_LIBRARY_PATH`, where the DSP's loader
looks. Each call is one `remote_handle64_invoke`: a method number, and the
input and output buffers. We pack those ourselves (`Dsp::call`) instead of
linking `qaic`'s generated CPU-side stub.

- **It runs, unsigned.** Opening the session and loading the library
  takes 33–37 ms. `dsp_answers_a_ping` and `dsp_adds_arrays_with_hvx`
  pass on the phone; the HVX float add (`Q6_Vsf_vadd_VsfVsf`) gives
  exactly `a + b` in every lane, tails included.
- **A call costs ~146 µs** (median of 2,000; p10 136, p90 160) with no
  work in it, after asking for the highest clocks with no sleep
  (`HAP_power_set_DCVS_v3`). Without that request it was 393 µs. That is
  about the time of a whole decode token on the CPU (~5 ms) divided by 35,
  so a design with one call per matrix product (~170 per token) cannot
  work; one request per token, or a queue, is needed (step 4).
- **Copying arguments is slow**: c = a + b with the buffers copied in and
  out moves 2.5 GB/s at 4 MB per array, against ~66 GB/s that the CPU and
  GPU read from DRAM. Weights must live in memory the DSP maps once
  (step 2), not travel with each call.
- **Crash found and fixed**: unloading `libcdsprpc.so` (`dlclose`) after a
  session crashed the process, as its listener thread outlives the
  session. The library now stays loaded.

## Step 2: shared memory, and how fast the DSP reads it (2026-10-29; `raw/dsp-bandwidth.txt`)

**Shared buffers** (`Dsp::shared`): `rpcmem_alloc2` (system heap,
CPU-cached) gives memory with a file descriptor; `fastrpc_mmap` maps it
into the DSP once. A call then passes only the descriptor, and the DSP
finds its own address for it (`HAP_mmap_get`). Nothing is copied.

**The DSP's cache is not coherent with the CPU's writes.** The first
shared add returned zeros for the last third of the output: the DSP's
data cache is write-back, and those results had not reached memory when
the call returned. Reading the CPU's writes works: the DSP reads see them
(`dsp_reads_every_word` passes with the CPU's writes possibly still in its
cache). So the DSP side now writes back what it wrote
(`qurt_mem_cache_clean(…, FLUSH)`) and drops stale lines of what the CPU
wrote before reading it (`INVALIDATE`). `dsp_adds_shared_buffers` passes,
three rounds of new data, three runs in a row.

**Read bandwidth** (`read_bw`: HVX loads summed as integers, four in
flight per thread, split over DSP threads, timed on the DSP; GB/s):

| buffer | 1 thread | 2 | 3 | 4 | 5 | 6 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 KB (L2) | 93–135 | 186–270 | 223–270 | 237–347 | 294–350 | 323–365 | 272–336 |
| 4 MB | 16 | 33–35 | 51–55 | 65–67 | 67–68 | 66–67 | 77–81 |
| 256 MB (DRAM) | **13.8** | 27.0 | 38.1 | 45.9 | **46.9** | 45.9 | 39.2 |
| 256 MB, `l2fetch` 16 KB ahead | **35.6** | 27.8 | 37.9 | 40.1 | 37.3 | 35.9 | 30.5 |

- The DSP has 6 HVX contexts (`qurt_hvx_get_units`). More threads than
  that (8) read slower.
- From DRAM, one thread reads ~14 GB/s, and adding threads scales almost
  linearly up to ~46–47 GB/s at 4–5 threads. That is ~70 % of the
  ~66 GB/s the CPU and GPU reach on this phone.
- Software prefetch (`l2fetch`) makes one thread 2.6× faster (13.8 →
  35.6 GB/s) but slows several threads: their prefetches compete.
- So a Q4_0 matrix-vector product reading the 278 MB of Qwen2.5-0.5B's
  weights from DRAM with HVX could take ~6 ms per token at best on the
  DSP, against ~4.2 ms at the CPU's bandwidth.
- **Not measured yet**: DMA from DRAM into VTCM (the DSP has 8 MB). A
  first chain of 1-D descriptors reported every descriptor done but gave
  timings that do not scale with the bytes moved (four 1 MB descriptors:
  30 ms; 16,384 of 16 KB: 4.4 ms). That is not understood, so no DMA
  number is reported. (Step 4b: DMA into VTCM in the products, ~60 GB/s.)

## Step 3: a Q4_0 matrix-vector product on HVX (2026-10-29; `raw/dsp-gemv-v*.txt`)

**Layout** (`Dsp::q4_0_matrix`, repacked once by the CPU into shared
memory): rows in groups of 32; per group and 32-column block, four
128-byte vectors whose 32-bit lane j holds 4 quant bytes of row j, then
one vector of the 32 rows' f32 scales (640 bytes per 32 × 32 weights,
against Q4_0's 576: the scales are f32).

**Arithmetic**: x is quantized to Q8_0 by the CPU (as on our CPU path and
llama.cpp's). `vrmpy` multiplies each lane's 4 unsigned bytes by 4 signed
bytes held in a scalar register, so eight of them give each of the 32
rows its integer dot product over a block, one row per lane: no
reduction across lanes. Q4_0 stores q + 8, so each lane starts at
−8 × (the block's sum of x) and the nibbles go in unsigned. Then
sum × row scale × x scale, accumulated in float.

**Correct**: `dsp_q4_0_gemv_matches_reference` runs llama.cpp's own Q4_0
weight bytes and Q8_0 activation bytes (12 shapes, 1 × 32 to
896 × 4864) against dequant(x) · dequant(W) in float64: relative error
≤ 4.8e-7 on 1 and 5 threads, with and without prefetch.

**Speed** (Qwen2.5-0.5B's shapes, random weights, DSP time per product,
median of 15; per token = 24 layers × 4 products + the classifier):

| version | 1 thread | 2 | 4 | 6 | best GB/s (classifier) |
| --- | --- | --- | --- | --- | --- |
| v1: one accumulator | 48.6 ms | 25.7 | 14.3 | 10.8 | 31.9 |
| v2: two groups at once, nibbles in separate sums | 43.9 | 23.3 | 12.8 | 10.0–10.3 | 32–35 |
| v3: v2 + `l2fetch` of the next two groups | **10.9** | 11.1 | 10.8–10.9 | 10.5–10.6 | 40–41 |

- v1's loop took ~16 packets per block: all eight `vrmpy` added into one
  register, each waiting for the last. v2 has four independent chains
  (23 packets for two blocks, 16 of them `vrmpy`), close to the floor of
  one `vrmpy` per packet.
- Without prefetch every thread waits on memory: time halves as threads
  double. With it, one thread reads the classifier at 40 GB/s and more
  threads add nothing, as in step 2 (their prefetches compete). Either
  way the products take ~10.5 ms per token: ~306 MB at ~29 GB/s on
  average, against 46–47 GB/s that the DSP can read.
- The small matrices (q/k/v 0.6 MB, wo 0.5 MB) take 20–30 µs whatever the
  thread count: mostly starting and joining threads in every call. Over
  48 such products a token, that is ~1.2 ms. Step 4 needs persistent
  threads.
- For comparison: a whole token takes ~5.2 ms on our CPU and 13.5 ms on
  llama.cpp's NPU backend (step 8 of phase-l3).

## Step 4: the whole decode step on the DSP (2026-10-29; `raw/dsp-ppl-v*.txt`, `raw/dsp-decode-threads.txt`)

`dsp/model.c`, `src/model/dsp.rs`, `examples/dsp_ppl.rs`. One FastRPC
call per token (`model_decode`) runs every layer, the classifier and the
argmax on the DSP. The CPU uploads the model once (`model_load`) into one
shared buffer: the matrices in step 3's layout (q/k/v and gate/up stacked,
as on the GPU), norms and biases in f32. A second shared buffer holds the
state: activations, KV cache, logits, and a table of time per stage.
Per token the CPU writes only the token's embedding and the RoPE table
(sin, cos per pair, computed as on the CPU path).

**How the threads share a step**: 6 DSP threads (one per HVX context) run
the whole step together and wait at a barrier (`qurt_barrier`) wherever
the next stage needs every thread's results. Each thread takes a run of
row groups of each matrix product. Activations are f32 and are quantized
to Q8_0 before every product, as on our CPU path and llama.cpp's.

**Correct** (wikitext-2, 20 chunks of 512, the second half of each scored,
as `llama-perplexity`; `--pure` Q4_0 GGUFs):

| | SmolLM2-135M | Qwen2.5-0.5B |
| --- | --- | --- |
| llama.cpp (CPU) | 27.7412 | 19.7283 |
| ours, CPU | — | 19.7438 |
| ours, GPU (phase-l3) | 27.6910 | 19.7193 |
| ours, DSP v1 (scalar `expf`) | 27.7301 | 19.7237 |
| ours, DSP v3 (vector exp) | **27.7429** | **19.7584** |

The error bars are ±1.18 and ±0.84. Greedy text equals the GPU's for
SmolLM2 and the CPU's for Qwen ("The capital of France is Paris. It is
the largest city in the world…").
`dsp_vector_exp_and_swiglu_match_libm` checks the vector e^x and SwiGLU
against the standard library's f64 exp: relative error below 1e-6 and 2e-6.
Changing v2's scalar `expf` for the vector exp moved Qwen's chunk 1 from
10.61 to 10.72 and the final value by +0.035. That is the size of the
spread between our CPU, our GPU and llama.cpp, which compute the same
thing in different orders; Q8_0 rounding of the activations turns tiny
differences into whole steps.

**Speed** (Qwen2.5-0.5B, DSP time per token at positions 0–63, from
`take_profile`):

| stage | v1 | v2 | v3 |
| --- | --- | --- | --- |
| attn norm + quantize | 1548 µs | 218 | 208 |
| q/k/v | 522 | 476 | 445 |
| bias + RoPE + KV store | 597 | 171 | 176 |
| attention (+ quantize from v2) | 2419 + 1324 | 2069 | 564 |
| wo | 393 | 367 | 331 |
| ffn norm + quantize | 1489 | 185 | 183 |
| gate/up | 3935 | 3889 | 3606 |
| SwiGLU (+ quantize from v2) | 1388 + 7001 | 1814 | 605 |
| down | 2048 | 2009 | 1835 |
| classifier | 2460 | 2456 | 2284 |
| argmax, rest | 893 | 217 | 214 |
| **per token** | **25.97 ms** | **13.88** | **10.46** |
| tok/s, with the CPU's part | 36.0 | 64.7 | 81.3 |

- v1: every norm and quantize ran on thread 0 while the others waited.
  `roundf` and the f16 conversion are library calls on the DSP, once per
  value: 7 ms a token in the quantize before `down` alone (4864 values per
  layer).
- v2: rounding inline (truncate, then step away from zero when the exact
  fraction is at least ½, which is `roundf`). Every thread quantizes its
  own blocks: the norms sum all of x in each thread, and attention and
  SwiGLU quantize their own outputs. That removed four of the eleven
  barriers per layer. The argmax is split across threads.
- v3: SwiGLU and the softmax on HVX, with an e^x of our own (x = n·ln2 +
  r, Taylor series to r⁸, 2ⁿ into the exponent bits) and 1/x by three
  Newton steps. The attention's dot products and its sum over V are
  vectors too (one vector op per 32 floats; one reduction per position).
- SmolLM2-135M: 3.96 ms a token (207 tok/s).
- Against the others on Qwen: llama.cpp's NPU backend 13.5 ms (74 tok/s),
  our GPU ~10 ms, our CPU ~5.2 ms.
- Wall time per token is ~1.8 ms more than the DSP's (12.3 ms on Qwen):
  the call (~150 µs), the embedding and RoPE table, and `dsp_ppl`
  reading all 151936 logits for its own argmax check. Greedy decoding
  could take just the DSP's `next` and not read the logits.

**What limits it now**: the matrix products, 8.5 of the 10.5 ms. All
matrices are ~306 MB per token, so that is ~36 GB/s, against the 46–47
GB/s that HVX reads reach (step 2). Thread count barely matters
(`raw/dsp-decode-threads.txt`): 2 threads 12.8 ms, 4 11.2, 5 10.6,
6 10.5, 8 (more than the 6 HVX contexts) 11.1. With `l2fetch` off, the
products are latency-bound: 27.0 / 15.3 / 11.4 ms on 2 / 4 / 6 threads.
Next: DMA into VTCM (step 2's open item), or prefetch spread over the
threads so they do not compete.

Attention grows with position: over the perplexity run (positions 0–510)
the DSP does 63 tok/s on Qwen, against 81 at positions 0–63.

## Step 4b: the weights by DMA into VTCM (2026-10-30; `raw/dsp-gemv-v4-*.txt`, `raw/dsp-ppl-v4-*.txt`, `raw/dsp-decode-dma-threads.txt`)

Step 4's products read the weights from DRAM through the L2 with
`l2fetch`, at ~36 GB/s. Now each DSP thread has two slots in VTCM (the
DSP's 8 MB of on-chip memory). The DMA engine copies the thread's next
pair of row groups (2 × 32 rows, all their blocks) into one slot while
the vector unit computes on the pair in the other. The copies bypass the
L2. `Fetch::Dma` in `Dsp::q4_0_gemv`; per product in the model (bit k of
`DspModel::dma`, all five by default).

**How the DMA engine is driven**: a 16-byte descriptor (source,
destination, size, a `done` bit the engine sets). Every hardware thread
has its own queue: `dmstart` begins a chain, `dmlink` appends to its
tail, and the thread polls `done` (with `dmpoll`). The descriptor layout
for v79 and the use of `dmlink` follow llama.cpp's backend (the SDK's
header is older). Each thread keeps a ring of 64 descriptors, so none is
rewritten while the engine may still read it.

**The bug that hid this in step 2**: VTCM is asked for with
`HAP_compute_res_acquire`. When all the worker threads asked at once,
one got it and the others waited inside the call for ever: the call
never returned. A probe (each thread writing where it was into shared
memory, flushed, read by the CPU while the call hung) showed five of six
threads inside the acquire and the sixth at the barrier after its rows.
A matrix test whose first call used 1 thread never raced, which is why it
worked there. Now VTCM is asked for once, by the thread FastRPC calls in
on, before the workers start (`model_load`, and `q4_0_gemv` with DMA).
Step 2's odd DMA timings (a single thread, no race) remain unexplained;
that test was not rerun.

**Correct**: `dsp_q4_0_gemv_matches_reference` runs the DMA path too (1
and 6 threads; 108/108 tests on the phone). Perplexity is unchanged to
the last digit, 27.7429 and 19.7584: only how the bytes arrive changed.

**Matrix products** (Qwen2.5-0.5B's shapes, random weights, DSP time,
best of 1–6 threads):

| matrix | MB | l2fetch | DMA | GB/s, l2fetch → DMA |
| --- | --- | --- | --- | --- |
| q/k/v | 0.6 | 21 µs | 21 µs | 31 → 31 |
| wo | 0.5 | 18 | 24 | 28 → 21 |
| gate/up | 5.4 | 180 | 100 | 30 → 55 |
| down | 2.7 | 105 | 63 | 26 → 43 |
| classifier | 85.1 | 2073 | 1311 | 41 → 65 |
| per token | | 11.0 ms | 6.5 ms | |

65 GB/s is the DRAM speed the CPU and GPU reach (~66 GB/s). The small
matrices are thread start and join in this test (the model starts its
threads once per token).

**Decode** (6 threads, DSP time per token at positions 0–63):

| | l2fetch | DMA |
| --- | --- | --- |
| Qwen2.5-0.5B | 10.44 ms (81 tok/s) | **6.75 ms (123 tok/s)** |
| SmolLM2-135M | 3.95 ms (198 tok/s) | **2.76 ms (271 tok/s)** |

On Qwen the five products now take 5.1 ms for ~306 MB, ~60 GB/s:
gate/up 2.09 ms, classifier 1.35, down 1.13, q/k/v 0.28, wo 0.23. Thread
count (`raw/dsp-decode-dma-threads.txt`): 2 threads 8.30 ms, 3 7.53,
4 7.12, 5 6.93, 6 6.79.

Against the others on Qwen decode: llama.cpp's NPU backend 13.5 ms (74
tok/s), our GPU ~10 ms, our CPU ~5.2 ms (192 tok/s). The DSP's ~1.7 ms
left over beyond the products is the norms, RoPE, attention and SwiGLU
(~1.5 ms) and 9 barriers per layer.

Over the perplexity run (positions 0–510) Qwen does 72 tok/s: attention
grows with position. Next for decode: attention over the whole cache in
vectors of positions, not one reduction per position.
