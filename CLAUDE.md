# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Aether is a from-scratch LLM inference engine in Rust for Snapdragon Android phones (CPU / ARM NEON first). `From_Scratch_Android_Inference_Engine_Plan.md` is the source of truth for scope, phase order, and success criteria — read the relevant phase before working on anything.

**Current state (2026-11-07):** the engine is its own repository, **`../aether-infer`** (crate `aether-infer`, imported as `aether`; its own `CLAUDE.md`, plans and docs; on GitHub as `seshuthota/aether-infer`, private for now). It has everything the plan built: Phases 0–8 and the later phases (K-quants and imatrix, i8mm, flash-decoding, speculative decoding, Vulkan GPU, Hexagon NPU), LFM3-1.7B, Gemma 4 E2B on CPU / GPU / DSP, the `Model` / `Session` API, a C ABI, an Android AAR, llama.cpp-style CLI subcommands and `aether serve`. This repo keeps the history (`IMPLEMENTATION.md`, `plans/`) and all measured numbers (`results/phase-N.md`, `results/release/`). The split: `plans/open-source-library.md`; the engine's next steps: `aether-infer/plans/release-and-llamacpp-compat.md`. The learning track (site, films, `notes/phase-N.md`) is at Phase 1.

## This repo: the learning journal

- Engine work happens in `../aether-infer`, not here. Here we explain concepts, walk through the engine's verified code and measured numbers (its source is under `crates/aether/` there; the site's older pages name the paths as they were here, e.g. `src/model/transformer.rs` is `crates/aether/src/model/transformer.rs`), run and modify experiments, and keep `site/` and `videos/` up to date.
- `src/` is the user's own learning code (crate `aether-learn`), one module per exercise. It depends on the engine by path (`aether = { path = "../aether-infer/crates/aether", package = "aether-infer", features = ["internals"] }`), so an exercise can compare against `aether::kernels` or time with `aether::bench`.
- One phone: `scripts/adb-run.sh` serializes runs (`/tmp/aether-phone.lock`, shared with the engine repo). Check the SoC temperature before trusting a benchmark; engine work may just have heated it.

`site/` is a static learning-notes site (no build step; see `site/README.md`). Pages are listed once in `site/js/pages.js` (menu, breadcrumb, prev/next and home cards are generated from it) and glossary terms live in `site/js/glossary.js`. When a concept is taught, add or update its page so it stays useful to a complete beginner.

Explainer films live in `videos/<phase>-<topic>/` and are copied to the page's folder in `site/` as `<topic>.mp4` + poster `.jpg`. Plan every film with the `3b1b-explainer` skill (puzzle first, picture before the name, one new element per line, motion while its line is spoken, animated recap), then build it with the `manimgx` skill. Films are narrated, with no on-screen captions: `SCRIPT` in `scene.py` is the voice-over, `self.vo("id")` / `self.hold()` from `videos/narrate.py` make each beat last as long as its recorded clip in `audio/<id>.wav` (estimated from word count until recorded), and `narrate.py` mixes the clips in (`mux`), writes subtitles (`vtt`) and checks clips against the script with Whisper on Groq (`check`, keys in `.env`). Write voice lines to be heard: numbers spelled as spoken, "D-RAM" not "DRAM". Anything that is not a measurement from this phone is labelled as an illustration on screen. `videos/phase-1-bandwidth/` is the reference film.

## Commands

Target phone: iQOO 13, Snapdragon 8 Elite (SM8750), connected over adb. Cores 0–5 are 3.53 GHz performance cores, cores 6–7 are 4.32 GHz prime cores (all Oryon; no efficiency cores). CPU features include dotprod, i8mm, bf16, fp16; no SVE.

```
cargo test                                                    # the learning crate, on the host (PC)
cargo test --release --target aarch64-linux-android          # on the phone, via the runner
cargo test --release --target aarch64-linux-android scalar_array::   # one module / one test: pass a name filter
```

The engine's own commands (its test suite, `aether bench`, examples such as `bench_sum`) run in `../aether-infer`; see its `CLAUDE.md`.

- `.cargo/config.toml` sets the NDK clang (`aarch64-linux-android31-clang`, i.e. min Android 12) as linker and `scripts/adb-run.sh` as runner. It is **gitignored** (absolute, machine-specific linker path); a new setup copies `.cargo/config.toml.example` and edits the path.
- The runner pushes the binary to `/data/local/tmp/aether/`, runs it with `adb shell`, deletes it, and returns the exit code. `/sdcard` cannot execute binaries.
- Always benchmark with `--release` (the release profile keeps debug symbols for `simpleperf`). Doc-tests run on the host, not the phone.
- Benchmarks: use `aether::bench::bench` (the engine's, through the path dependency) and wrap inputs/results in `black_box`; the compiler otherwise collapses repeated pure calls (observed: 10 identical sums ran as 1).

## How to work in this repo

- **This is a learning project.** The user is building it to understand inference deeply; a fast engine is a bonus, not the goal. Performance numbers are things to measure and explain, not gates to pass.
- **Step by step, concept by concept.** Explain the concept first, check understanding, then go through one small piece at a time with the user. Don't jump ahead of the phase being learned. Engine changes go to `../aether-infer` in small, verified commits.
- **Engine code is written from scratch** — no existing inference engine for core logic.
- **No duplicate implementations for testing.** Correctness comes from external references only (see the plan's "Correctness Strategy"): numpy/PyTorch for kernels, llama2.c for stories15M, HF `transformers`/`tokenizers` for SmolLM2/Qwen, llama.cpp for quantization formats, perplexity/KL, and `llama-bench`. Never propose writing a second implementation of our own as a test oracle.
- **Mobile constraints drive design:** memory bandwidth, small shared RAM, thermals, energy per token, heterogeneous (prime/performance/efficiency) cores.

## Layout

- `src/` — the user's learning code (crate `aether-learn`); `Cargo.toml` depends on `../aether-infer`.
- `site/`, `videos/` — the learning site and the explainer films.
- `notes/phase-N.md` — the user's learning journal per phase; `notes/impl-track/` — the implementation track's notes from before the split.
- `results/` — every measured number, with recorded conditions (cores, thermal state, power); the engine repo's docs link here.
- `plans/`, `IMPLEMENTATION.md`, `From_Scratch_Android_Inference_Engine_Plan.md` — the plan and its history.
- `scripts/` — the phone runner (`adb-run.sh`), the QEMU runner, and the site's check / serve scripts.
