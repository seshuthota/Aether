# Gemma 4: the checks still owed on the phone

Written 2026-11-05 after the night's PC run (`results/phase-gemma4.md`).
Everything below waits for the phone. Branch `gemma4` is merged into
`main` only when all of 1–5 pass. Run from `../Aether-next`.

## Before starting

- `adb devices` lists the phone. Check the SoC temperature before any timed run.
- **The first run pushes ~6 GB.** `scripts/adb-run.sh` syncs `testdata/`,
  and `testdata/gguf/` now holds both Gemma GGUFs (2.8 + 3.1 GB). Over
  Wi-Fi adb that takes a while. Do it once on USB if possible.
- `testdata/*` in this worktree are symlinks into `../Aether-impl`. Check
  that `adb push --sync` follows them: the files on the phone must be
  regular files of the right size.
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
case before loosening anything.

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

Expect the previous count (123) plus the new tests, all passing. The GPU and
DSP paths must refuse Gemma 4 with their message, not crash.

## 4. Accuracy and memory on the phone

- Q4_0 perplexity, 20 × 512, cores 2–7. Compare it with the PC's 57.4248
  (ours) and llama.cpp's 57.4593. LFM3's phone and PC numbers differed in
  the 2nd–3rd decimal (NEON / i8mm rounding), so expect the same here.
- Existing models unchanged on the phone: Qwen2.5-0.5B Q4_0 and LFM3
  Q4_K_M PPL (4 chunks) from `main` and from `gemma4`, to the digit. The PC
  check covered only the x86 paths.
- Greedy text: the 4 prompts of `results/phase-gemma4/raw/greedy-Q4_0-pc.txt`
  against `llama-completion` on the phone. Any split must be a near-tie
  (check the logit gap).
- Peak RSS (`/usr/bin/time -v` or `/proc/<pid>/status` VmHWM) for both
  files. The PC peak was 2.67–2.69 GB, below the file size.

## 5. Speed (G7)

`bench_suite` at cores 2–7 with `--pp 64,128,512 --tg 64 --depth 0,512`,
alternated with `llama-bench` in the same session (two rounds, the second
reversed). Cool to 45 °C before each run, on battery, and record the clock
caps. llama.cpp's numbers from the K5 session (Q4_0 tg64 30.54, pp512
106.11) are the reference, but only same-session numbers count.

## 6. The app

Push the Q4_0 GGUF to the app's files directory, chat for a few turns, and
note the PSS next to the 1.22 GB the production app showed.

## After all pass

- Commit the NEON kernels, and write the phone section of
  `results/phase-gemma4.md`.
- Tick G3/G5/G6/G7 in the status table of `plans/gemma4.md`.
- Merge `gemma4` → `main`.
