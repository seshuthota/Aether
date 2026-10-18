# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Aether is a from-scratch LLM inference engine in Rust for Snapdragon Android phones (CPU / ARM NEON first). `From_Scratch_Android_Inference_Engine_Plan.md` is the source of truth for scope, phase order, and success criteria — read the relevant phase before working on anything.

**Current state:** Phase 0 nearly done. The Cargo project (lib `aether` + bin), on-device test runner, timing harness (`src/bench.rs`) and `Tensor` (`src/tensor.rs`, f32 row-major) exist. Folders `tools/`, `testdata/`, `results/`, `notes/` exist (empty). Still to do: first commit and `notes/phase-0.md`.

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
- **Step by step, concept by concept.** Never one-shot a feature or a phase. Explain the concept first, check understanding, then implement one small piece at a time with the user. Don't jump ahead of the current phase.
- **Engine code is written from scratch** — no existing inference engine for core logic.
- **No duplicate implementations for testing.** Correctness comes from external references only (see the plan's "Correctness Strategy"): numpy/PyTorch for kernels, llama2.c for stories15M, HF `transformers`/`tokenizers` for SmolLM2/Qwen, llama.cpp for quantization formats, perplexity/KL, and `llama-bench`. Never propose writing a second implementation of our own as a test oracle.
- **Mobile constraints drive design:** memory bandwidth, small shared RAM, thermals, energy per token, heterogeneous (prime/performance/efficiency) cores.

## Planned layout and workflow (from the plan, Phase 0)

- Cargo project with a library (the engine) and binaries; target `aarch64-linux-android`, linker from the Android NDK via `.cargo/config.toml`.
- A cargo runner script (`scripts/adb-run.sh`) that pushes binaries to `/data/local/tmp` and runs them, so `cargo test`/`cargo run`/benchmarks execute on the phone.
- `tools/` — host-side Python scripts that generate golden reference data; `testdata/` — golden files (large files regenerated, not committed).
- `results/` — benchmark log committed alongside the code that produced the numbers, with recorded conditions (cores, thermal state, power).
- `notes/phase-N.md` — the user's learning journal per phase.
