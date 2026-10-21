# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Aether is a from-scratch LLM inference engine in Rust for Snapdragon Android phones (CPU / ARM NEON first). `From_Scratch_Android_Inference_Engine_Plan.md` is the source of truth for scope, phase order, and success criteria — read the relevant phase before working on anything.

**Current state:** Phase 0 done (Cargo project, on-device test runner, `src/bench.rs`, `src/tensor.rs`); `notes/phase-0.md` is still the user's to write. Phase 1 started: `src/aligned.rs` (`AlignedBuf`, piece 1). Build order and per-item status: `IMPLEMENTATION.md`.

## Two tracks (agreed 2026-10-21)

- **Implementation track** builds the engine ahead, whole phases at a time, in the git worktree `../Aether-impl` on branch `impl`. It follows `IMPLEMENTATION.md`, keeps its status boxes current, verifies every phase against the external references, and writes `results/phase-N.md`. A phase is merged into `master` only after its verification passes.
- **Learning track** works on `master` with the user: explains concepts, walks through the verified code and the measured numbers, runs and modifies experiments, and keeps `site/` and `videos/` up to date. It does not rewrite engine code the implementation track has built, and does not edit the `impl` branch.
- One phone: `scripts/adb-run.sh` serializes runs. Check the SoC temperature before trusting a benchmark; the other track may just have heated it.

`site/` is a static learning-notes site (no build step; see `site/README.md`). Pages are listed once in `site/js/pages.js` (menu, breadcrumb, prev/next and home cards are generated from it) and glossary terms live in `site/js/glossary.js`. When a concept is taught, add or update its page so it stays useful to a complete beginner.

Explainer films live in `videos/<phase>-<topic>/` and are copied to the page's folder in `site/` as `<topic>.mp4` + poster `.jpg`. Plan every film with the `3b1b-explainer` skill (puzzle first, picture before the name, one new element per line, motion while its line is spoken, animated recap), then build it with the `manimgx` skill. Films are narrated, with no on-screen captions: `SCRIPT` in `scene.py` is the voice-over, `self.vo("id")` / `self.hold()` from `videos/narrate.py` make each beat last as long as its recorded clip in `audio/<id>.wav` (estimated from word count until recorded), and `narrate.py` mixes the clips in (`mux`), writes subtitles (`vtt`) and checks clips against the script with Whisper on Groq (`check`, keys in `.env`). Write voice lines to be heard: numbers spelled as spoken, "D-RAM" not "DRAM". Anything that is not a measurement from this phone is labelled as an illustration on screen. `videos/phase-1-bandwidth/` is the reference film.

## Commands

Target phone: iQOO 13, Snapdragon 8 Elite (SM8750), connected over adb. Cores 0–5 are 3.53 GHz performance cores, cores 6–7 are 4.32 GHz prime cores (all Oryon; no efficiency cores). CPU features include dotprod, i8mm, bf16, fp16; no SVE.

```
cargo test                                                    # on the host (PC)
cargo test --release --target aarch64-linux-android          # on the phone, via the runner
cargo test --release --target aarch64-linux-android --lib tensor::   # one module / one test: pass a name filter
cargo run --release --target aarch64-linux-android --example bench_sum
```

- `.cargo/config.toml` sets the NDK clang (`aarch64-linux-android31-clang`, i.e. min Android 12) as linker and `scripts/adb-run.sh` as runner. It is **gitignored** (absolute, machine-specific linker path); a new setup copies `.cargo/config.toml.example` and edits the path.
- The runner pushes the binary to `/data/local/tmp/aether/`, runs it with `adb shell`, deletes it, and returns the exit code. `/sdcard` cannot execute binaries.
- Always benchmark with `--release` (the release profile keeps debug symbols for `simpleperf`). Doc-tests run on the host, not the phone.
- Benchmarks: use `aether::bench::bench` and wrap inputs/results in `black_box`; the compiler otherwise collapses repeated pure calls (observed: 10 identical sums ran as 1).

## How to work in this repo

- **This is a learning project.** The user is building it to understand inference deeply; a fast engine is a bonus, not the goal. Performance numbers are things to measure and explain, not gates to pass.
- **Step by step, concept by concept** (learning track). Explain the concept first, check understanding, then go through one small piece at a time with the user. Don't jump ahead of the phase being learned. The implementation track builds ahead in small, verified commits, one `IMPLEMENTATION.md` item each.
- **Engine code is written from scratch** — no existing inference engine for core logic.
- **No duplicate implementations for testing.** Correctness comes from external references only (see the plan's "Correctness Strategy"): numpy/PyTorch for kernels, llama2.c for stories15M, HF `transformers`/`tokenizers` for SmolLM2/Qwen, llama.cpp for quantization formats, perplexity/KL, and `llama-bench`. Never propose writing a second implementation of our own as a test oracle.
- **Mobile constraints drive design:** memory bandwidth, small shared RAM, thermals, energy per token, heterogeneous (prime/performance/efficiency) cores.

## Planned layout and workflow (from the plan, Phase 0)

- Cargo project with a library (the engine) and binaries; target `aarch64-linux-android`, linker from the Android NDK via `.cargo/config.toml`.
- A cargo runner script (`scripts/adb-run.sh`) that pushes binaries to `/data/local/tmp` and runs them, so `cargo test`/`cargo run`/benchmarks execute on the phone.
- `tools/` — host-side Python scripts that generate golden reference data; `testdata/` — golden files (large files regenerated, not committed).
- `results/` — benchmark log committed alongside the code that produced the numbers, with recorded conditions (cores, thermal state, power).
- `notes/phase-N.md` — the user's learning journal per phase.
