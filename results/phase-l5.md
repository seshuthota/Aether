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
  number is reported; it is open for step 3 or later.
