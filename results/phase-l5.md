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

## Step 5a: matrix-matrix products on HMX (2026-10-30; `raw/dsp-hmx-gemm.txt`)

For a prompt the products are matrix × matrix: every weight is used by
all the prompt's tokens, so the work is arithmetic, not reading DRAM.
The DSP has a matrix unit for that, HMX. It multiplies 32 × 32 tiles of
fp16 that must be in VTCM, adding into its own accumulator, and writes
a 32 × 32 result tile back to VTCM. HMX is not in the SDK's documents;
the tile layouts and instructions follow llama.cpp's backend.

`Dsp::hmx_gemm` (dsp/hmx.c): the activations (up to ~320 tokens) are
turned into fp16 tiles in VTCM's first 4 MB bank. The weights stay in
our Q4_0 layout in DRAM. HVX threads dequantize chunks of 32-row groups
into fp16 weight tiles in one of two 1.5 MB buffers in the second bank,
while thread 0 drives HMX on the chunk in the other buffer and turns the
result tiles into f32 rows. Q4_0's 32 rows of a group sit in the 32
lanes of a vector, which is the layout of a weight tile's 32 outputs, so
no shuffling across lanes is needed.

**Three things that went wrong first**:
- The DSP hung, and stayed hung for every later call until the phone
  was rebooted. HMX has to be powered on (`HAP_power_set_HMX`) like
  HVX; its instructions on an unpowered unit never finish.
- Every output was 1.0 too large. The output "scales" are a 32-bit word
  per column, fp16 scale in the low half and fp16 bias in the high half;
  `0x3c00` splatted into both halves set a bias of 1.0.
- With 2048 inputs (64 tiles in one streamed `:deep` load) the sums were
  wrong; 28 tiles were right. Loads are now at most 32 tiles each.
  (llama.cpp's backend loads one tile pair per instruction.)

**Correct**: `dsp_hmx_gemm_matches_reference`, against numpy's float64
X·dequant(W)ᵀ on the 11 batch goldens (odd row and token counts), at 1,
2 and 6 threads: relative error under 3 · 2⁻¹¹ (fp16's rounding of X, W
and the result). 109/109 tests on the phone.

**Speed** (Qwen2.5-0.5B's four layer products, random weights, DSP
time; "tok/s" is 24 layers of products alone, no attention or classifier):

| tokens, threads | q/k/v | wo | gate/up | down | tok/s |
| --- | --- | --- | --- | --- | --- |
| 128, 1 | 735 µs | 494 | 7417 | 4074 | 419 |
| 128, 4 | 335 | 296 | 2726 | 1913 | 1012 |
| 128, 6 | 308 | 239 | 1656 | 1325 | 1512 |
| 32, 6 | 153 | 105 | 1594 | 805 | 502 |
| 256, 6 | 408 | 309 | 1842 | 1862 | 2413 |

Dequantizing the weights is most of the work: on one thread with it
skipped, 128 tokens take 443 µs on gate/up (5 TFLOP/s) against 7417. So
the cost per matrix is nearly fixed and the tokens share it: 32 tokens
run at 502 tok/s, 256 at 2413. llama.cpp's NPU prefill of the whole
model is ~330 tok/s. Next (step 5b): prefill through the whole model on
the DSP, and perplexity again.

## Step 5b: the prompt through the whole model on the DSP (2026-10-30; `raw/dsp-ppl-v5-*.txt`, `raw/dsp-hmx-gemm-prefetch.txt`, `raw/dsp-cli-vs-cpu/`)

Written while the phone was away, run on it the same evening (on battery
from step 3 on, 40–57 °C).

`model_prefill` (dsp/model.c) runs up to 128 tokens per FastRPC call
through every layer:
- The products run on HMX through step 5a's code, now shared by all
  threads.
- Norms, RoPE and the KV store run a batch of tokens per thread.
- Attention runs a batch of (token, head) pairs per thread.
- The logits come out for every token (perplexity) or for the last one.

Two changes came with it:
- **Keys stored transposed:** row e of a layer's cache holds key element e
  for every position. So 32 positions' scores are one vector of
  multiply-adds, with no sum across lanes per position. This is aimed at
  decode at long positions.
- **HMX dequantization prefetches:** each worker l2fetches its share of
  the next chunk's Q4_0 bytes.

**A bug the first phone run found**: `dsp_hmx_gemm_matches_reference`
failed once in the full test run, with one output 10⁴ off, and passed when
run alone. Each FastRPC call runs on its own DSP thread, and the tests call
from several threads at once. The gemv (DMA mode), the HMX product, decode
and prefill all use the one VTCM, the per-thread DMA rings and `dma_gen`,
so two calls at once overwrote each other's data. `dsp_busy` / `dsp_free`
now hold a lock around their worker threads (`3ab2966`). The engine makes
one call at a time, but an app could make two. After the fix, the full run
passed 112/112 three times (and 113/113 with `--features vulkan`).

**Correct**, perplexity (wikitext-2, 20 × 512, the second half of each
chunk scored):

| Path | SmolLM2-135M | Qwen2.5-0.5B |
| --- | --- | --- |
| DSP prompt pass, batches of 128 (HMX, fp16 products) | **27.6893** | **19.7174** |
| DSP decode step, token by token (transposed keys) | 27.7876 | 19.7560 |
| DSP decode step before (v3, step 4) | 27.7429 | 19.7584 |
| GPU f32 (phase-l3) | 27.6910 | 19.7193 |
| llama.cpp (CPU) | 27.7412 ± 1.18 | 19.7283 ± 0.84 |

- The HMX path is within 0.05 of llama.cpp. fp16 products cost nothing
  measurable.
- The decode step's number moved when only the order of the attention's
  adds changed. SmolLM2 moved by +0.045 and Qwen by −0.002, the same size as
  earlier shifts from reordering (scalar `expf` → vector exp moved Qwen by
  0.035). The activations are rounded to Q8_0 before every product, so a
  value near a rounding edge can flip and change the logits a little.

**`aether --dsp` against the CPU** (greedy, 96 positions): the same text
for SmolLM2's short prompt. The other three runs agree for 2–40 tokens,
then pick a different word at a near-tie and go their own way. The DSP's
Qwen continuation ("the largest city in the world and the most populous…")
is word for word the PC GPU's f16 one, so two f16 paths agree and the
CPU's integer path differs. "The same text as the CPU" was the wrong test;
perplexity is the right one.

**Speed** (DSP time, prompt from position 0):

| Tokens | SmolLM2-135M | Qwen2.5-0.5B |
| --- | --- | --- |
| 32 | 1205 tok/s | 376 tok/s |
| 64 | 1636 | 596 |
| 128 | 1608 | 778 |
| whole perplexity run | 872 | 443 |

llama.cpp's NPU prefill is ~330 tok/s on Qwen (phase-l3 step 8).

- **`dsp_gemm -- 128 6`** with the prefetch: 1656 tok/s for Qwen's layer
  products, up from 1512–1543. q/k/v went from 308 to 172 µs and wo from
  239 to 155 µs. gate/up and down barely moved: on those the matrix unit
  is the limit, not DRAM.
- **Decode over a 512-token run** (the change the transposed keys were
  for): Qwen 72 → 86 tok/s, SmolLM2 175. At short positions: 297 / 112
  tok/s. Qwen's number was taken at 57 °C; on a cool phone it was 123.

## Every engine, one benchmark (2026-10-30; `../backends/raw/`, `../backends/raw-sustain/`)

`scripts/backends-bench.sh` runs `bench_suite` on the CPU (cores 2–7), the
GPU and the DSP for each model. It waits for the CPU sensors to drop below
45 °C before each run. Medians of 3, tok/s, Qwen2.5-0.5B Q4_0:

| | CPU | GPU | DSP |
| --- | --- | --- | --- |
| pp64 | **777** | 691 | 679 |
| pp128 | 669 | **813** | 799 |
| pp512 | 526 | **693** | 641 |
| tg64 at depth 0 | **136** | 99 | 131 |
| tg64 at depth 512 | **125** | 89 | 90 |
| time to first token, 64-token prompt | 116 ms | 93 ms | 94 ms |
| peak RSS | 344 MB | 838 MB | 373 MB |
| hottest CPU sensor in the run | 75.7 °C | 54.9 °C | 55.2 °C |

The GPU's RSS includes its f16 copy of the matrices for the prompt pass
(phase-l3 step 11).

**Five minutes of decode, on battery** (`SUSTAIN=300`, Qwen2.5-0.5B Q4_0,
screen kept on; power from the battery's charge counter, whole phone):

| | tok/s average | power | J/token | hottest |
| --- | --- | --- | --- | --- |
| CPU | 107.3 | 3.23 W | 0.030 | 85 °C |
| GPU | 78.2 | 2.26 W | 0.029 | 58 °C |
| **DSP** | **130.6** | 3.28 W | **0.025** | 70 °C |

The CPU starts at 188 tok/s on a cool phone, heats up and slows to an
average of 107 tok/s. The DSP holds 131. So over minutes the DSP is the
fastest engine and the most efficient one: about 17 % less energy per token
than the CPU. The GPU draws the least power and stays coolest, but it is
slow enough that its energy per token about equals the CPU's.

The power monitor's numbers from the short runs (5–20 s) are not
meaningful: the charge counter updates too rarely for them.

## Step 6: prompt lookup and Q4_K_M on the DSP, and the app (2026-10-30, evening; `raw/lookup/`, `raw/dsp-q4km-*.txt`)

Written the same afternoon while the phone was away, run in the evening
(phone on battery, 37–62 °C).

**A pass of a few tokens.** Speculative decoding checks a certain token
and up to 7 drafts in one call (`model_few`): every layer for up to 8
tokens, each group of weights read once and used for every token. Its
sums are added in the decode step's order, so its logits equal the decode
step's bit for bit (`few_tokens_get_the_decode_logits`, Q4_0 and Q4_K_M).
First version fetched weights by `l2fetch`; with DMA into VTCM as decode
does (Qwen2.5-0.5B Q4_0, DSP time):

| tokens | 1 | 2 | 4 | 8 |
| --- | --- | --- | --- | --- |
| one decode step | 6.4 ms | | | |
| pass, `l2fetch` | 13.9 | 14.4 | 17.0 | 28.0 |
| pass, DMA | 6.5 | 7.8 | 12.1 | 23.0 |

**Lookup** (`aether --lookup 4`, Qwen2.5-0.5B-Instruct Q4_0, greedy, the
phase-6 prompts; decode tok/s):

| prompt | CPU plain → lookup | GPU plain → lookup | DSP plain → lookup |
| --- | --- | --- | --- |
| add comments to a function (copies it) | 151 → 224 | 69 → 28 | 119 → 149 |
| why is the sky blue | 137 → 180 | 50 → 28 | 111 → 142 |
| short story | 164 → 198 | 69 → 24 | 111 → 118 |

On the DSP and the GPU the text with lookup equals the text without in
all three. (The CPU's story differs after ~80 tokens, as in phase 6: its
batched path rounds differently.) On the GPU lookup is a loss: on the
Adreno each extra token of the few-token shader costs about a whole
one-token dispatch (`../phase-l3/raw/vk-gemv-n-adreno.txt`), so the C ABI
now ignores lookup for GPU sessions.

**Q4_K_M** (llama.cpp's mix: q/k/o/gate/up Q5_0, v Q5_0 or Q8_0, down
Q4_K or Q6_K, classifier Q8_0). Each format has its own HVX product
(`dsp/qx_hvx.h`), checked on the SDK's simulator against llama.cpp's
bytes before the phone. Perplexity (wikitext-2, 20 × 512):

| | DSP decode | DSP few-token pass | our CPU | llama.cpp |
| --- | --- | --- | --- | --- |
| SmolLM2-135M Q4_K_M | 20.7241 | 20.7241 | 20.7201 | 20.7784 ± 0.85 |
| Qwen2.5-0.5B Q4_K_M | 16.7381 | 16.7381 | 16.7088 | 16.7657 ± 0.69 |

Decode: SmolLM2 256 tok/s (Q4_0 271), Qwen 92–98 tok/s (Q4_0 123; the
weights are ~1.4× the bytes). The prompt runs in passes of 8 tokens on
HVX (no HMX prefill for these formats): 272 tok/s against 755 for Q4_0
on HMX, 241 tokens.

**A bug the new test found.** The first phone run of Q4_K_M gave PPL
47,530 through decode and 16.30 through the few-token pass. Only q/k/v
broke, and only with DMA: in this file q/k/v is a Q5_0 run then a Q8_0
run with no barrier between them, and each thread's VTCM slots were
placed by the product's size, so a thread on the Q8_0 run copied over a
neighbour's slots still in use. Each thread now has a fixed eighth of
VTCM. Q4_0 numbers unchanged (27.7876 / 19.7560, 124 tok/s).

**The app** (Qwen2.5-0.5B-Instruct Q4_0 and Qwen2.5-0.5B Q4_K_M, the
app's sampling at temperature 0.7). All three engines answer. The app
opens an unsigned DSP session from its own files directory: FastRPC
falls back from the default device node (denied to apps) to the HAL and
creates the user PD in 28 ms; the denials around it (kernel
optimisations, folder watchers, properties) are not needed.

| run | prompt | decode |
| --- | --- | --- |
| DSP, "Why is the sky blue?" | 35 tok, 462 tok/s | 183 tok, 112 tok/s |
| DSP, repeat a sentence, lookup off / on | 50 tok, 568 tok/s | 14 tok, 116 / 146 tok/s |
| CPU, Q4_K_M base, lookup on | 6 tok | 512 tok, 171 tok/s |
| GPU, Q4_K_M base | 6 tok, 243 tok/s | 219 tok, 53 tok/s |

The DSP on Q4_K_M ran in the app (session opened and closed with no
errors) but its status line was not captured.

## Decode profile at depth (2026-10-30; `raw/dsp-profile-depth-*.txt`)

`dsp_ppl MODEL depth 0,512,2048` fills one cache in order, clears
`DspModel::take_profile`, and times the next 64 decode steps. The stage
times are the DSP's qtimer. Tokens are the start of wikitext-2. Nothing
in the step changed; this is the baseline for the long-context work.
`attend` is untouched, so perplexity was not rerun.

Phone on battery, screen woken. zone0 (`aoss-0`) and the hottest CPU
sensor, and the scaling cap (`scaling_cur/scaling_max`):

| | Qwen2.5-0.5B Q4_0 | SmolLM2-135M Q4_0 |
| --- | --- | --- |
| depth 0 | 38–42 °C, caps 2.75 / 2.65 GHz | 40–45 °C, caps 2.40 / 2.44 GHz |
| depth 512 | 47–48 °C, same caps | 47 °C, same caps |
| depth 2048 | 53–55 °C, same caps | 51–53 °C, same caps |

Prime-core cap was already 2.65 GHz (Qwen) and 2.44 GHz (SmolLM2) at the
start, and cpu7 sat near 1.0 GHz during the DSP work. The matrix-product
stages did not slow down as the phone warmed (Qwen gate/up 2092 → 1877 µs,
classifier 1337 → 1209 µs), so the growth below is not the clock cap.

DSP time per token, and the attention stage inside it (µs):

| depth (positions timed) | Qwen ms/token | Qwen attention | SmolLM2 ms/token | SmolLM2 attention |
| --- | --- | --- | --- | --- |
| 0 (0–63) | 6.56 (126 tok/s) | 263 | 2.59 (297 tok/s) | 279 |
| 512 (512–575) | 9.84 (84 tok/s) | 3464 | 6.29 (136 tok/s) | 3911 |
| 2048 (2048–2111) | 23.08 (39 tok/s) | 17142 | 17.91 (51 tok/s) | 15553 |

Every other stage stays within a few tens of microseconds of its depth-0
time, apart from the small product speed-up above. From depth 0 to 512
the step grows 3.28 ms on Qwen and 3.70 ms on SmolLM2, of which attention
is 3.20 and 3.63 ms. From depth 0 to 2048 the step grows 16.5 and 15.3 ms,
and attention is all of it (17.1 and 15.6 ms of the step: 74 % and 87 %).

Attention grows with position: about 6.3 µs per extra position on Qwen
between the depth-0 and depth-512 windows, and 8.9 µs per position from
512 to 2048; SmolLM2 stays near 7.1 then 7.6 µs. Depth 0 matches the
earlier short runs (Qwen 6.57 ms, SmolLM2 2.57 ms), so those runs were
the flat part of this curve.

**Why SmolLM2 costs more per head.** The stage time is the busiest
thread, and `decode_thread` hands out whole query heads. Qwen has 14
heads and 24 layers, so two of the six threads get 3 heads and the stage
is 72 head-visits (3 × 24). SmolLM2-135M has 9 heads and 30 layers, so
the busiest threads get 2 heads: 60 head-visits. At depth 512 that is
48 µs per head on Qwen (3464 µs / 72) and 65 µs per head on SmolLM2
(3911 / 60). A head is the same shape on both, 64 floats, so a visit at
depth 512 reads the same 256 KB of K and V. That is about 5.3 GB/s on
the Qwen thread and 3.9 GB/s on the SmolLM2 thread.

The difference is how those visits share a K/V head. Qwen has 2 K/V
heads and groups of 7; the busy thread's 3 heads are the same K/V head,
read three times. SmolLM2 has 3 K/V heads and groups of 3; the busy
thread reads one K/V head twice, and the other threads are on the other
two K/V heads at the same time. One layer's live K/V is 512 KB on Qwen
and 768 KB on SmolLM2 (2 against 3 heads). The per-head gap narrows at
depth 2048 (238 µs against 259 µs), where a layer is 2.0 MB against
3.0 MB and neither thread is rereading a small cache.

## Wide value loop (2026-10-30; `raw/dsp-profile-depth-vloop-*.txt`, `raw/dsp-ppl-vloop-*.txt`)

`attend` scales the scores by 1/sum in one vector pass, then sums values
with four accumulators. Same f32 values, different add order. Decode,
prefill and the few-token pass all call `attend`.

Perplexity, wikitext-2, 20 × 512, against the transposed-key run
(27.7876 / 19.7560 decode, 27.6893 / 19.7174 prefill) and llama.cpp
(27.7412 / 19.7283):

| | SmolLM2-135M | Qwen2.5-0.5B |
| --- | --- | --- |
| decode | 27.7623 | 19.7716 |
| prefill, 128 tokens a call | 27.6943 | 19.7205 |

A few hundredths, inside the spread. 114/114 phone tests, including
`few_tokens_get_the_decode_logits`.

Same profile as above, on battery, screen woken. zone0 38 °C at the
start of Qwen, 49 °C at depth 512, 53 °C at depth 2048 (baseline was
38 / 48 / 54). Caps 2.75 / 2.65 GHz. Products at depth 512 match the
baseline to ~10 µs (gate/up 2092 against 2095).

| depth | Qwen ms/token | Qwen attention | SmolLM2 ms/token | SmolLM2 attention |
| --- | --- | --- | --- | --- |
| 0 | 6.54 (was 6.56) | 248 (was 263) | 2.56 (was 2.59) | 255 (was 279) |
| 512 | 9.47 (was 9.84) | 3087 (was 3464) | 5.92 (was 6.29) | 3548 (was 3911) |
| 2048 | 24.14 (was 23.08) | 17680 (was 17142) | 17.47 (was 17.91) | 15108 (was 15553) |

The clean gain is depth 512: −377 µs of attention on Qwen and −363 µs
on SmolLM2, and the step moves by the same amount. That is about 5 µs
off a 48 µs head, which is what a dependency chain of a few microseconds
can be. At depth 2048 SmolLM2 is −445 µs and Qwen is +538 µs. The
baseline's Qwen depth-2048 products were themselves ~10 % quicker than
its own depth-512 products (gate/up 1877 against 2095 µs), so that one
window was a fast clock; this run's products sit on the depth-512
numbers (gate/up 2098). Either way the loop is not where the 17 ms goes.

## Key-stride padding (2026-10-30; `raw/dsp-profile-pow2-*.txt`, `raw/dsp-profile-pad-*.txt`, `raw/dsp-ppl-pad-*.txt`)

The depth profile above uses context 2112, so `att_stride` is 2112 floats
(8448 bytes). 2112 is not a power of two, and a pad would not change that
run. Context 512 and 4096 do round to a power of two (2048 and 16384
bytes between key elements). The pad adds one vector, 32 floats, only in
that case, and the state buffer grew to match (`dsp.rs` and `model_load`).

Compared at context 4096, before and after, same caps (cpu0 1.79 GHz,
cpu7 1.96 GHz) and the same gate/up time (2095 µs on Qwen), so the DSP
was not on a different clock. Attention, µs/token:

| depth | Qwen, stride 4096 | Qwen, 4128 | SmolLM2, 4096 | SmolLM2, 4128 |
| --- | --- | --- | --- | --- |
| 512 | 3815 (10.17 ms) | 2902 (9.28 ms) | 3678 (6.04 ms) | 3393 (5.77 ms) |
| 2048 | 17725 (24.21 ms) | 17372 (23.85 ms) | 15382 (17.67 ms) | 14737 (17.09 ms) |

At depth 512 the pad removes 912 µs on Qwen and 284 µs on SmolLM2: set
conflicts were a real piece there, about a quarter of Qwen's attention.
At depth 2048 it is 353 µs and 645 µs, a few percent of a 15–17 ms
stage. An extra window at depth 4032 (stride still the power of two)
moved 36000 → 35190 µs on Qwen. Conflicts are not the long-context slope.

Perplexity is the wide-loop number to the last printed digit, as it
should be: the values did not change, only their spacing. Decode
27.7623 / 19.7716, prefill 27.6943 / 19.7205. 114/114 tests.

## Key and value prefetch (2026-10-31; `raw/dsp-profile-depth-l2fetch-*.txt`, `raw/dsp-ppl-l2fetch-*.txt`)

`attend` asks L2 for the key chunk four chunks ahead (one vector on
each of the head's rows) and, once per 32 positions, for one block of
this head's value rows four blocks ahead. The first 128 positions are
not fetched. A stride that does not fit l2fetch's 16-bit field is
skipped; 2112 floats is 8448 bytes, so this run does fetch. The
arithmetic is unchanged.

Same profile as the wide-loop table: context 2112, so the stride pad
is off and the only difference is the prefetch. On battery, screen
woken. Qwen zone0 41 °C at the start (hottest cpu 43 °C), 52 °C at
depth 512, 58 °C at depth 2048. Caps were already 2.40 / 2.44 GHz.
Qwen gate/up stays 2099 / 2089 / 2097 µs against 2090 / 2092 / 2098 in
the wide-loop run, so the DSP clock matches even though the CPU cap
does not. SmolLM2 gate/up matches at depth 512 (581 µs). At depth 2048
it is 605 µs against 531 in the wide-loop window, which was the fast
one; this window is the slower clock.

| depth | Qwen ms/token | Qwen attention | SmolLM2 ms/token | SmolLM2 attention |
| --- | --- | --- | --- | --- |
| 0 | 6.59 (was 6.54) | 254 (was 248) | 2.58 (was 2.56) | 267 (was 255) |
| 512 | 8.45 (was 9.47) | 2058 (was 3087) | 4.77 (was 5.92) | 2352 (was 3548) |
| 2048 | 14.23 (was 24.14) | 7774 (was 17680) | 9.99 (was 17.47) | 7435 (was 15108) |

Depth 0 does not issue a fetch (the window is shorter than the lead),
and it does not move. At depth 512 the fetch removes 1.03 ms on Qwen
and 1.20 ms on SmolLM2. At depth 2048 it removes 9.91 ms and 7.67 ms,
about half of that stage. The step moves by the same amount. The
dependent adds were 0.4 ms and the power-of-two conflicts were 0.35 ms
at this depth, so the half that moved is DRAM latency: the bytes were
already in the loop, waiting.

What is left still scales with the bytes. Qwen's busy thread is 72
head-visits, so 2058 µs is 29 µs per head at depth 512 and 7774 µs is
108 µs per head at depth 2048. A head reads 256 KB and 1 MB of K and V
at those depths, which is about 9 GB/s at both. SmolLM2's
60 visits are 39 µs and 124 µs per head, about 7 GB/s then 8.5 GB/s.
The same reasons as in the baseline: the busy SmolLM2 thread has two
heads on one K/V head and the other threads are on the other two K/V
heads (a layer is 768 KB at depth 512, against Qwen's 512 KB and a
group of 7 reread by the busy thread). The gap narrows at depth 2048,
where a layer is 3.0 MB against 2.0 MB.

Perplexity is again the same digits: decode 27.7623 / 19.7716, prefill
27.6943 / 19.7205. Context 512 does pad the stride (544 floats, 2176
bytes), so this run also checks the prefetch together with the pad.
114/114 phone tests, including `few_tokens_get_the_decode_logits`.

The two experiments split the depth-2048 stage. Conflicts are a few
hundred microseconds. Latency was about half, and this prefetch covers
it on these contexts. The rest is the K/V bytes, reread by every query
head on the thread (three times on Qwen's busy thread, twice on
SmolLM2's). A single read of each K/V head into VTCM is aimed at that
remainder, not at the latency. f16 KV stays off until that read has
been measured: depth 2048 is still 7.8 ms of attention on Qwen.

## One read of each K/V block (2026-10-31; `raw/dsp-profile-depth-stream-*.txt`, `raw/dsp-ppl-stream-*.txt`)

Past 256 positions, attention streams the cache. A block is 256
positions of one K/V head, DMA'd once into that thread's VTCM slot
(keys packed, values in order, source read through L2). Every query
head of that K/V head uses the block, and the thread keeps a running
max, sum and output per query head. Blocks merge in order after a
barrier. The pieces are (K/V head, block), spread over all six threads.
A whole head is not copied: at depth 2048 that is 1 MB, and at 4096 it
is 2 MB, and the products already use VTCM on either side of attention.
Shorter contexts stay on the prefetch path. Decode, prefill and the
few-token pass call the same function.

Same depth profile as the prefetch section, context 2112, on battery.
Qwen zone0 53 °C at the start (hottest cpu 56 °C), 61 °C before depth
512, 64 °C before depth 2048. Caps cpu0 1.79 GHz, cpu7 1.96 GHz. Qwen
gate/up is 2195 / 2115 / 2126 µs. The depth-0 figure is the high one;
512 and 2048 match each other, and they sit next to the prefetch run's
2089 / 2097 µs. SmolLM2 gate/up is 584 / 573 / 573 µs against 576 / 581
/ 605. The clock is not what moved the long windows.

| depth | Qwen ms/token | Qwen attention | SmolLM2 ms/token | SmolLM2 attention |
| --- | --- | --- | --- | --- |
| 0 | 6.83 (was 6.59) | 270 (was 254) | 2.60 (was 2.58) | 273 (was 267) |
| 512 | 7.91 (was 8.45) | 1546 (was 2058) | 3.78 (was 4.77) | 1506 (was 2352) |
| 2048 | 10.39 (was 14.23) | 4008 (was 7774) | 5.97 (was 9.99) | 3689 (was 7435) |

Depth 0 is the prefetch path (the window ends at position 63) and does
not move. At depth 512 the stream removes 0.51 ms on Qwen and 0.85 ms
on SmolLM2. At depth 2048 it removes 3.77 ms and 3.75 ms, and the step
moves by the same amount. Qwen's distinct K and V at that depth is
about 48 MB (24 layers, 2 K/V heads, 1 MB each). At the 46–47 GB/s HVX
read, those bytes are about 1 ms. The measured stage is 4.0 ms, so the
bytes are no longer read once per query head, and what is left is more
than the copy.

The crossover is below 512. A same-session pair, prefetch against the
stream, caps cpu0 2.00 GHz and cpu7 2.25 GHz, gate/up flat at about
2100 µs on Qwen and 578 µs on SmolLM2 (`raw/dsp-profile-depth-prefetch-128-*.txt`,
`raw/dsp-profile-depth-stream-128-*.txt`). Attention, µs/token:

| depth | Qwen prefetch | Qwen stream | SmolLM2 prefetch | SmolLM2 stream |
| --- | --- | --- | --- | --- |
| 128 | 808 (7.21 ms) | 907 (7.17 ms) | 875 (3.21 ms) | 726 (3.01 ms) |
| 256 | 1191 (7.55 ms) | 1289 (7.56 ms) | 1389 (3.75 ms) | 1052 (3.33 ms) |
| 512 | 2094 (8.52 ms) | 1489 (7.76 ms) | 2265 (4.66 ms) | 1506 (3.79 ms) |

At 128 and 256 a layer is one or two blocks. Qwen has two K/V heads, so
two threads do the work and each runs all seven query heads; attention
is about 100 µs slower and the step does not move. SmolLM2 has three
K/V heads and is already faster at 128, by 0.20 ms then 0.42 ms.
Streaming starts at 256 positions: Qwen's depth-128 window stays on the
prefetch, and from there SmolLM2's step drops. Both models are faster
at 512.

Perplexity, 20 chunks of 512, the second half scored. The block merge
changes the add order. Decode 27.7779 / 19.7503 (was 27.7623 / 19.7716),
prefill 27.6890 / 19.7196 (was 27.6943 / 19.7205). The moves are a few
hundredths, inside the ±1.18 / ±0.84 error. 114/114 phone tests,
including `few_tokens_get_the_decode_logits`. That test now compares
eight tokens at position 256, where the merge is two blocks, and decode
still matches the few-token pass bit for bit. f16 KV stays off. Depth
2048 attention is 4.0 ms, still the long stage next to 2.1 ms of gate/up.

## Where the 4 ms goes (2026-10-31; `raw/dsp-profile-depth-split-*.txt`, `raw/dsp-profile-depth-value-*.txt`, `raw/dsp-few-at-2048-*.txt`)

Before changing the loop, depth 2048 was split with per-thread cycle
counters (`raw/dsp-profile-depth-split-qwen.txt`,
`raw/dsp-profile-depth-split-smol.txt`). Qwen's step was 10.36 ms,
attention 4024 µs, gate/up 2105 µs, zone0 56.5 °C, cpu0 at its 2.40 GHz
cap. SmolLM2's step was 5.99 ms, attention 3710 µs, gate/up 573 µs.
The busiest thread and the average match on total cycles (Qwen 8.12M
vs 8.12M, SmolLM2 7.48M vs 7.46M), so the work is spread. The average
barrier is higher than the busiest thread's because a thread with less
merge work waits there. Busiest thread, then average, microseconds:

| part | Qwen busy | Qwen avg | SmolLM2 busy | SmolLM2 avg |
| --- | --- | --- | --- | --- |
| DMA wait | 531 | 531 | 631 | 636 |
| score + softmax | 1345 | 1238 | 1001 | 1028 |
| value accumulation | 1776 | 1641 | 1613 | 1633 |
| barriers + merge | 200 | 439 | 300 | 240 |

Exp, timed inside the score, is 0.35M cycles on Qwen's busiest thread,
about 0.17 ms, and 0.29M on SmolLM2. It is not the long piece. `vexp`
clamps each lane to [−80, 80], splits `x` into `n·ln2 + r`, takes the
Taylor series of `e^r` through `r^8`, and writes `2^n` into the exponent
bits. The cross-block merge uses scalar `expf` twice per later block,
and that time sits in the barrier bucket.

DMA was already double-buffered: the next block's copy is issued before
the current block is computed, and the measured wait is what the copy
still sticks out by. Barriers and the merge are the small bucket, so
the block stayed 256. The long bucket was the value mix. Holding every
query head's accumulators at once, or two heads, spilled the schedule
and made Qwen's attention 9.2 ms, then 6.5 ms. One head with the scores
sitting in VTCM was 6.2 ms: the scalar load of the score from VTCM
stalled the chain. The loop that shipped keeps one head's four
accumulator pairs as named vectors, so they stay in the register file,
and the scores stay in a stack buffer. The add order per head is the
one the single-head loop used.

Same depth profile afterwards, on battery. Qwen zone0 53.8 °C at the
start, 60.7 °C before depth 512, 63.8 °C before depth 2048, 64.2 °C
after. Caps cpu0 1.79 GHz, cpu7 1.96 GHz. Gate/up is 2100 / 2102 / 2099
µs on Qwen and 575 / 573 / 577 µs on SmolLM2.

| depth | Qwen ms/token | Qwen attention | SmolLM2 ms/token | SmolLM2 attention |
| --- | --- | --- | --- | --- |
| 0 | 6.59 (was 6.83) | 256 (was 270) | 2.57 (was 2.60) | 265 (was 273) |
| 512 | 7.60 (was 7.91) | 1292 (was 1546) | 3.63 (was 3.78) | 1341 (was 1506) |
| 2048 | 9.30 (was 10.39) | 2994 (was 4008) | 5.57 (was 5.97) | 3273 (was 3689) |

Depth 0 is still the prefetch path. At depth 2048 the new split, busiest
thread then average, is:

| part | Qwen busy | Qwen avg | SmolLM2 busy | SmolLM2 avg |
| --- | --- | --- | --- | --- |
| DMA wait | 485 | 487 | 1302 | 1303 |
| score + softmax | 1338 | 1229 | 957 | 989 |
| value accumulation | 837 | 757 | 599 | 609 |
| barriers + merge | 154 | 338 | 236 | 189 |

Value fell from 1.78 ms to 0.84 ms on Qwen and from 1.61 ms to 0.60 ms
on SmolLM2. Score did not move. Qwen's long piece is now the score,
1.34 ms of the 3.0 ms stage, and exp is still about 0.18 ms of that.
SmolLM2's long piece is now the DMA wait, 1.30 ms. The value loop no
longer covers the copy, and the wait is the part that sticks out of the
double-buffer that was already there. Totals stay balanced (Qwen 5.93M
cycles on the busiest thread against 5.93M average). f16 KV stays off:
on Qwen the bytes are still the smaller bucket.

Perplexity did not move. Decode 27.7779 / 19.7503, prefill 27.6890 /
19.7196, the same as the streamed run, inside the ±1.18 / ±0.84 error.
114/114 phone tests, including `few_tokens_get_the_decode_logits`.

A few-token pass at depth 2048 does not share one K/V DMA across its
tokens. Each token calls the same `attend_token` as a decode step, so
the streamed attention on the busiest thread scales with the token
count (Qwen 2837 µs for one, 22664 µs for eight). The call gets cheaper
per token because the matrix products read each weight once for the
whole pass. Median of five, call microseconds and that attention:

| tokens | Qwen call | Qwen attention | Qwen per token | SmolLM2 call | SmolLM2 attention | SmolLM2 per token |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 9375 (1.01×) | 2828 | 9375 | 5612 (1.03×) | 3018 | 5612 |
| 2 | 13548 (1.46×) | 5673 | 6774 | 9453 (1.73×) | 6024 | 4726 |
| 4 | 23463 (2.53×) | 11323 | 5866 | 17460 (3.20×) | 12050 | 4365 |
| 8 | 45658 (4.93×) | 22664 | 5707 | 33761 (6.18×) | 24097 | 4220 |

One decode step at the same position is 9264 µs on Qwen (attention
2837 µs) and 5461 µs on SmolLM2 (attention 3002 µs). Zone0 was 61 °C,
battery discharging.
