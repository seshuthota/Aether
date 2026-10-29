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
