# Gemma 4: everything still owed on the phone

Rewritten 2026-11-05, after the PC work of `plans/gemma4-close.md` (its
status table in section 0). Everything below needs the phone. Run from
`../Aether-next` on branch **`gemma4-devices`**: it holds `gemma4` (the CPU
work) and the PC-verified GPU, DSP-kernel and app work on top.

Order matters: the CPU checks gate the first merge. C0 is quick and decides
the DSP's design, so it runs early even though the DSP work comes later.

| # | What | Plan step | Rough time | Gates |
| --- | --- | --- | --- | --- |
| 0 | Setup, push the GGUFs | — | 15–40 min (6 GB) | — |
| 1 | NEON kernel goldens | A1 | 5 min | committing the NEON GELU / soft-cap |
| 2 | Gemma tests | A1 | 5 min | merge |
| 3 | Whole suite, all backends | A1, B, C3, D | 30 min | merge |
| 4 | **C0**: the DSP maps 1.6 GB | C0 | 2 min | C's design |
| 5 | CPU accuracy, memory, greedy | A1 | 60 min | merge |
| 6 | GPU accuracy and memory | B6 | 30 min | GPU merge |
| 7 | llama.cpp's GPU and NPU on Gemma | S0 | 30 min | B7 / C7 references |
| 8 | Speed: CPU, then GPU | A1 (G7), B7 | 90 min | — (measurement) |
| 9 | Profile the CPU | A2 | 30 min | orders A3 |
| 10 | The app on CPU and GPU | D | 20 min | — |

If the evening is short: 0–5 and 10 are the core (merge the CPU work, know
C0's answer, see Gemma in the app); 6–9 can follow another day.

## 0. Before starting

- `adb devices` lists the phone. Check the SoC temperature before any timed
  run; `scripts/adb-run.sh` serializes runs.
- **The first run pushes ~6 GB.** `scripts/adb-run.sh` syncs `testdata/`,
  and `testdata/gguf/` holds both Gemma GGUFs (2.8 + 3.1 GB). Do it on USB
  if possible.
- `testdata/*` in this worktree are symlinks into `../Aether-impl`. Check
  that `adb push --sync` follows them: the files on the phone must be
  regular files of the right size (`adb shell ls -lL /data/local/tmp/aether/testdata/gguf`).
- The working tree has the **uncommitted** NEON GELU and soft-cap
  (`src/kernels/neon.rs`, `mod.rs`, `scalar.rs`). Step 1 is their check.
  Commit them only if it passes.

## 1. Kernels on NEON (numpy goldens)

```
cargo test --release --target aarch64-linux-android --lib kernels::
```

| Test | What it checks | Bound |
| --- | --- | --- |
| `gemma_elementwise_matches_numpy` | NEON GELU-tanh × up, NEON soft-cap, plain RMSNorm | rel 1e-6 / 1e-6 / 1e-5 |
| `attention_matches_numpy`, `attention_parts_merge_to_numpy`, `attention_group_matches_numpy`, `attention_window_matches_numpy` (new cases: head 256 / 512, causal 8 × 1, window 512) | NEON attention at head size 256 and 512, causal 8 × 1 heads, sliding window 512 | the existing `2e-6 · (√hs + √n)` |
| every other kernel test | unchanged | as before |

If the GELU or soft-cap bound fails, the likely spot is near zero: the
soft-cap's `(1 − e)/(1 + e)` loses relative precision there. The bound is
relative to the largest value, so it should still pass. Look at the failing
case before loosening anything. (The DSP's HVX soft-cap uses the same
identity and passed on the simulator at 1.9e-7.)

## 2. Gemma tests on the phone

```
cargo test --release --target aarch64-linux-android --lib gemma4
cargo test --release --target aarch64-linux-android --lib -- spm_bpe chat::
```

- `gemma4_loads`: shapes, KV sources, RoPE pairs, the table format, the
  table unchanged by repacking (both files).
- `gemma4_matches_hf_tokenizers_on_corpus`: 440 strings.
- `gemma4_template_matches_hf`, `think_filter_hides_split_spans`.
- `gemma4_matches_transformers` skips itself on the phone (its BF16 file
  and reference live in `hostdata/`). It passed on the PC.

## 3. The whole suite, all backends

```
cargo test --release --target aarch64-linux-android --features vulkan,hexagon
```

Expect the previous count (123) plus the new tests, all passing. New since
`main` and worth a look in the output:
- `c_abi_gpu_gemma4`: Gemma 4 Q4_0 on the **Adreno** through the C ABI,
  opened with cores as the app does; must answer "Two plus two equals
  four." and stop. This is the first Gemma run on the phone's GPU.
- `gemma_shaders_match_numpy`, `attention_per_token_matches_reference`
  (head 256 / 512, window): the new shaders on the Adreno.
- The DSP must still refuse Gemma 4 with its message, not crash.
- `dsp_maps_gemma_sized_buffers` is ignored here; it is step 4.

## 4. C0: can the DSP map Gemma's weights?

```
cargo test --release --target aarch64-linux-android --features hexagon --lib dsp_maps_gemma_sized_buffers -- --ignored --nocapture
```

Maps 1.6 GB + 192 MB of shared buffers at once; the DSP reads every word
and its checksum must equal the CPU's. Record the result and the GB/s it
prints in `results/phase-gemma4.md` (DSP section).
- **Passes:** C1, C2, C4, C5 go ahead with one weights buffer, as for Qwen.
- **Fails** (allocation, mapping, or a wrong checksum): try 1.2 GB and
  0.8 GB to find the limit, then pick the plan's fallback (split weights
  over several buffers, or the 428 MB classifier on the CPU) before any
  C code is written.
- If the phone hangs (DSP code can need a reboot), note it; that is an
  answer too.

## 5. CPU accuracy, memory and greedy text

- Q4_0 perplexity, 20 × 512, cores 2–7:
  `aether testdata/gguf/gemma-4-E2B-it-Q4_0.gguf --ppl testdata/wikitext/wikitext-2-raw/wiki.test.raw -c 2-7`.
  Compare with the PC's 57.4248 (ours) and llama.cpp's 57.4593. LFM3's
  phone and PC numbers differed in the 2nd–3rd decimal (NEON / i8mm
  rounding), so expect the same here.
- Existing models unchanged on the phone: Qwen2.5-0.5B Q4_0 and LFM3
  Q4_K_M PPL (4 chunks) built from `main` and from `gemma4-devices`, to the
  digit. The PC check covered only the x86 paths.
- Greedy text: the 4 prompts of `results/phase-gemma4/raw/greedy-Q4_0-pc.txt`
  against `llama-completion --temp 0` on the phone. Any split must be a
  near-tie (check the logit gap).
- Peak RSS (`/usr/bin/time -v` or VmHWM in `/proc/<pid>/status`) for both
  files. The PC peak was 2.67–2.69 GB, below the file size.

**After 1–5 pass:** commit the NEON kernels, write the phone section of
`results/phase-gemma4.md`, tick G3/G5/G6 in `plans/gemma4.md`, and merge
`gemma4-devices` into `main` (the GPU and DSP code in it is verified on
the PC and either refuses or is checked by step 3).

## 6. The GPU on the phone (B6)

On the Adreno 830, through `examples/vk_gemma.rs` (PPL/KL against
llama.cpp's BF16 base file; `hostdata/gemma4/kl-base-bf16.bin` must be
pushed, 1 GB):

- Q4_0 decode, 4 chunks. PC: 43.6075 / KL 0.34282 (llama.cpp 43.8254 /
  0.34330). Then the prompt pass (`batch`), 1 chunk: must equal decode as
  on the PC (24.4536 / 0.33855).
- Q4_K_M decode, 4 chunks. PC: 138.0871 / KL 0.16511.
- `examples/gemma_window.rs`: GPU decode vs prompt pass vs CPU at
  positions 0–639 (past the window). Compare with the PC's table in
  `results/phase-gemma4.md`.
- Memory: PSS of `vk_gemma` while decoding (`dumpsys meminfo <pid>` or
  `/proc/<pid>/smaps_rollup`). Expected ~1.5 GB of device buffers + the
  151 MB cache at 4096 positions (*calculated*). If a buffer allocation
  fails, the plan's B6 fallback is to split per layer.
- **Watch for the watchdog.** On the PC's Intel GPU a whole prompt pass
  in one submission was cut off silently (now one submission per layer).
  If the Adreno's batch PPL differs from decode, check `dmesg`/`logcat`
  for a GPU fault or timeout first.
- **Prompt speed.** On the PC the f32 `qx_gemm` prompt pass ran at ~11
  GFLOP/s. Time `forward_batch` on the Adreno. If it is far behind
  decoding the prompt token by token, B7 decides on the f16/int8 second
  copy (+1.5 GB).

## 7. llama.cpp's GPU and NPU on Gemma 4 (S0)

The references for B7 and C7. llama.cpp's OpenCL build is in
`/data/local/tmp/llama-ocl` (as in `scripts/gpu-pp-vs-llamacpp.sh`); the
Hexagon (HTP) build is the one of phase L3
(`results/phase-l3/raw/llamacpp-npu-*.txt`; find its directory with
`adb shell ls /data/local/tmp`). For both Gemma files and both backends:

- `llama-bench -m … -p 64,128,512 -n 64 -ngl 99 --mmap 0 -r 3`;
- `llama-perplexity -m … -f wiki.test.raw --chunks 4 -ngl 99`;
- the load log: does every op run on the device, or do some fall back to
  the CPU (the log lists the graph splits)?

If a backend cannot run Gemma 4, note it; then that backend's references
are llama.cpp's CPU numbers, stated as such.

## 8. Speed (G7, B7)

**CPU (G7).** `bench_suite` at cores 2–7 with `--pp 64,128,512 --tg 64
--depth 0,512`, alternated with `llama-bench` in the same session (two
rounds, the second reversed; `scripts/kquant-phone.sh gemma` runs
llama.cpp's side). Cool to 45 °C before each run, on battery, and record
the clock caps. llama.cpp's numbers from the K5 session (Q4_0 tg64 30.54,
pp512 106.11) are the reference, but only same-session numbers count.

**GPU (B7).** `bench_suite --gpu` (or `vk_gemma`'s timing) against step 7's
OpenCL numbers, same session, alternated, cooled: pp64/128/512, tg64 at
depth 0 and 512, with power and J/token as in L.3. Decide here:
- whether the prompt pass needs the f16/int8 second copy of the matrices;
- whether the CPU-side PLE inputs (`ple_inputs`, ~0.5 ms a token,
  *calculated*) limit GPU decode (plan risk 5);
- whether lookup on the GPU (B5) is worth building.

## 9. Profile the CPU (A2)

`simpleperf` on Q4_0 and Q4_K_M decode and prefill, cores 2–7. Split the
time into: classifier (262,144 rows), FFN, attention by head size, the PLE
block, the PLE inputs (BF16 GEMV + table row), soft-cap, page faults on
table rows. This table orders A3's optimisations (plan section 4); nothing
in A3 is built before it.

## 10. The app (D)

Build with `scripts/android-native.sh` and `./gradlew assembleDebug`
(both build on the PC; the APK is from `34fdb4e`). Push the Q4_0 GGUF to
the app's files directory, then:
- **CPU**: chat a few turns; the status line says "CPU 2-7".
- **GPU**: the status line must say "GPU" (not "prompt on CPU": Gemma's
  GPU path runs its own prompt pass); chat a few turns. Expect a slow
  first reply if the Adreno's prompt pass is as slow as the PC's.
- **DSP**: must show the refusal message, not crash.
- Note the PSS on CPU and GPU next to the 1.22 GB the production app
  showed.

## After the session

- `results/phase-gemma4.md`: phone sections for CPU, GPU and C0.
- `plans/gemma4-close.md` status table: A1, B6, B7, S0, C0, D.
- `IMPLEMENTATION.md` §0 and `CLAUDE.md` "Current state" on `main` after
  the merge.
- Next work, by the results: A3 (ordered by step 9), C1–C5 (shaped by
  step 4), B5 / the GPU second copy (decided by step 8).
