# Plan: Aether as an open-source inference library

Written 2026-11-05. Facts below were read from the tree at `gemma4-devices`
(`dd16acc`) and from GitHub. Decisions taken the same day are in section 9.
Section 8 says which steps start now (additive, no file moves) and which
wait for the Gemma merge (plans/gemma4-phone-checks.md), so the moves do
not collide with the phone work.

## 0. Where it stands

**Already true:**
- `github.com/seshuthota/Aether` is **public** and the notes site deploys
  from it (`.github/workflows/pages.yml`). It has **no LICENSE**, which
  legally means "all rights reserved": people can read the code but not
  reuse it.
- A C ABI exists (`include/aether.h`, `src/ffi.rs`: open on CPU, GPU or DSP,
  generate, lookup, errors as codes plus a message), and the Android app
  uses it. This is the closest thing to a stable interface today.
- Dependencies are small: `libc`, plus `ash` (optional, Vulkan
  declarations).
- CPU features are detected at run time (`is_aarch64_feature_detected!`).
  One build therefore runs on any arm64 phone, and the scalar path runs on
  x86.
- `.env` (the Groq keys) was never committed; checked over all history.
  Nothing from the Qualcomm SDK is in the tree.

**In the way of others using it:**

| Area | Today | Why it matters |
| --- | --- | --- |
| Crate | One crate `aether` (rlib + cdylib), 17 `pub mod`s, everything public | No line between API and internals; any refactor breaks users |
| API | Three session types with different `open` signatures (`Session::open(path, tokenizer, max_seq, format, cores)`, `GpuSession::open(path, max_seq)`, `DspSession::open(path, max_seq, dir, threads)`) and different `generate`s | Backend choice should be an option, not a type |
| Errors | `io::Error` in loaders and sessions, `String` in `kernels::vulkan` / `hexagon`, `assert!` / `unwrap` / `panic!` on library paths | A library must return errors and never abort its caller |
| Output | A few `eprintln!` in the library (`format::imatrix`) | Libraries log through a hook, not stderr |
| Repo contents | `site/` 19 MB, `videos/` 40 MB, `results/` 1,502 files (111 with `/home/curious` paths), `notes/`, `plans/`, `CLAUDE.md`, `.claude/`, `agent/`, a tracked `.pyc` | The engine is hard to find in the learning workspace |
| Examples | 27, most of them experiments (`vk_gemv`, `dsp_bandwidth`, `vk_few` …) | Users need 3–4 that show the API |
| Build | `vulkan` needs `glslc` at build time (PATH or the NDK's); `hexagon` needs the Hexagon SDK and `qaic` (`build.rs`) | Fine, but undocumented for outsiders |
| Docs | No README at the root, no LICENSE, no CONTRIBUTING, no CI for the code (only the Pages deploy) | First impressions; nobody can tell what works |
| Name | `aether` is taken on crates.io (a WebAssembly language) | Needs a crate name |

## 1. What we take from llama.cpp, and what we don't

**No code is copied, from llama.cpp or any other engine.** That is already
the project's rule (`CLAUDE.md`: engine code is written from scratch, and
other engines are only references to check against). The open-source
version keeps it, and CONTRIBUTING states it.

What we do take is **shapes and interfaces**, the way many servers offer
an OpenAI-compatible API without sharing any code with OpenAI:
- **The file format.** We read GGUF, so models from Hugging Face work
  as-is. We already do this, from the format's description.
- **The layering.** A library, a C API, tools on top, and bindings by
  others. That is how llama.cpp is used (libllama, its C API, the CLI /
  server / quantize / perplexity / bench tools). It is a common pattern,
  not llama.cpp's invention.
- **Interfaces where compatibility helps users:**
  - CLI flags close to llama2.c's and llama.cpp's, so numbers can be
    compared run for run (as `aether` does today);
  - later, perhaps an OpenAI-compatible HTTP server, so existing clients
    work unchanged.
- **Measurements:** llama.cpp stays the reference for perplexity, KL and
  speed. Comparing against it is not copying it.

What we do **not** take: ggml (a general tensor and graph library), its
code, its kernels, its API names. Our models are written by hand per
architecture, on purpose: that is how the project teaches.

The position to claim is narrow and real: **phone-first inference.** That
means CPU (NEON / i8mm), Adreno (Vulkan) and Hexagon (HVX / HMX) in one
library, verified against external references at every step, with the
measured numbers published.

## 2. Repository split (decided)

- **The engine:** a new public repo, **`github.com/seshuthota/aether-llm`**
  (the name is free on GitHub). It holds the library, C API, CLI tools,
  shaders, DSP code, the Android library, tests, reference generators,
  docs and benchmark summaries.
- **The learning journal:** this repo, `seshuthota/Aether`, **unchanged**:
  site, films, notes, plans, raw results, `CLAUDE.md`, and the user's own
  learning code (`src/neon_array.rs` and the like). Nothing is moved out
  of it; it keeps its full history and the site keeps deploying.

**Low hassle by design:** no file moves happen in this repo. The engine
repo is made from this repo's history with `git filter-repo`, keeping
only the engine's paths and moving them into the new layout in the same
pass. Its 310 commits keep their messages, which tell the from-scratch
story and carry the verification numbers. Afterwards:
- the implementation track works in a clone of `aether-llm` instead of
  `../Aether-impl`;
- this repo's engine code stays as a frozen snapshot, and its site pages
  link to the engine repo's code;
- the learning track keeps working here as it does now.

## 3. Layout of the engine repo

A small Cargo workspace. Three crates are enough; more is a maintenance
cost with one maintainer.

```
Cargo.toml                  workspace
crates/
  aether/                   the library: formats (gguf, safetensors), quant,
    src/                    tokenizers, chat templates, models, sampler, sessions;
      backend/cpu/          kernels: scalar + NEON (today kernels/{scalar,neon,neon_quant})
      backend/vulkan/       feature "vulkan" (today kernels/vulkan + model/gpu*)
      backend/hexagon/      feature "hexagon" (today kernels/hexagon + model/dsp)
    shaders/  dsp/          built by its build.rs, as today
  aether-capi/              cdylib + include/aether.h: the stable C ABI (today ffi.rs)
  aether-cli/               aether (generate / chat), quantize, imatrix,
                            perplexity, bench (today main.rs, src/bin, bench_suite)
android/                    an AAR (Kotlin wrapper + JNI) and the sample chat app
bench/                      the micro-benchmarks and llama.cpp comparisons
                            (today's vk_* / dsp_* examples and scripts/)
tools/                      Python reference generators, with requirements.txt
docs/                       getting started, models, backends, architecture, benchmarks
```

The C ABI moves to its own crate so that Rust users don't build a cdylib,
and so the ABI changes only with intent.

## 4. The library API (the main code work)

One entry point; the backend becomes an option:

```rust
let model = aether::Model::load("gemma-4-E2B-it-Q4_0.gguf", &LoadOptions {
    backend: Backend::Gpu,          // Cpu { cores }, Gpu, GpuWithCpuPrompt { cores }, Dsp
    context: 4096,
    ..Default::default()
})?;
let mut session = model.session();
let stats = session.generate("Why is the sky blue?", &GenerateOptions {
    chat: true, temperature: 0.7, max_tokens: 256, ..Default::default()
}, |text| { print!("{text}"); true })?;
```

- **One `Backend` enum** replaces the three session types. Internally it is
  the enum `ffi.rs` already uses.
- **A low-level layer** for people building servers and bindings, i.e. the
  `llama_decode` equivalent: `tokenize` / `detokenize`,
  `eval(tokens, pos) -> logits`, sampling as a separate step, and KV
  save / restore later.
- **One `aether::Error`** (a kind plus a message). No panics across the API:
  `assert!` stays only for internal invariants, never for bad input. No
  `eprintln!` in the library; logging goes through an optional callback.
- **Everything else becomes `pub(crate)`.** Kernels, formats and quant stay
  internal at first; they become public only when someone asks, with docs.
- **The C API mirrors the Rust API:** `aether_version()`, plus tokenize /
  eval / logits calls next to `generate`.

Versioning: the C ABI is the stable surface (semver from 1.0). The Rust API
stays 0.x while it settles.

## 5. Build, platforms, correctness, CI

- **Features:**
  - default = CPU only: builds on any Rust target, fast paths on arm64;
  - `vulkan`: needs `glslc` at build time and a Vulkan loader at run time;
  - `hexagon`: needs the Hexagon SDK. The docs say which phones load an
    unsigned DSP module.
- **Correctness stays external-reference-only**, the project's rule, and it
  goes into CONTRIBUTING:
  - numpy / PyTorch goldens for kernels;
  - HF `transformers` / `tokenizers` for models and tokenizers;
  - llama.cpp for quantization formats, perplexity and KL.
- **CI (GitHub Actions), for what a runner can do:**
  - `cargo test` on x86 with the small goldens (stories15M, SmolLM2-135M),
    cached or regenerated from `tools/`;
  - `cargo clippy`;
  - a cross-compile check for `aarch64-linux-android` (with and without
    `vulkan`);
  - the `hexagon-sim` kernel tests, only if the SDK can be installed in CI
    (license check).
- **No phone in CI.** The phone suite and the big-model checks (Qwen,
  LFM3, Gemma 4 PPL / KL against llama.cpp) are a release checklist, run
  from `bench/` scripts. Each release publishes its numbers.
- **Formatting (decided: adopt rustfmt).** A `rustfmt.toml` close to the
  current style (`max_width = 160`, the long one-line signatures the code
  uses), and one reformat commit in the engine repo right after the split.
  From then on CI runs `cargo fmt --check`. This repo keeps its current
  style, and its "no cargo fmt" rule stays for the learning code.

## 6. Docs and community files

- **README:**
  - what it is (phone-first, three backends, from scratch, verified);
  - status;
  - a 5-line quick start (CLI and Rust);
  - a supported-models table (llama2.c, SmolLM2, Qwen2.5, LFM3, Gemma 4 E2B,
    and the formats);
  - a backends table with measured speed next to llama.cpp's, stated
    honestly (where llama.cpp is faster, say so);
  - a link to the learning site ("how this was built"), which no other
    engine has.
- **LICENSE, CONTRIBUTING** (the correctness rule, how to add a model,
  how to run the phone suite), **CHANGELOG**, issue templates.
- **docs/:**
  - backends: requirements per device, Hexagon signing;
  - architecture: sessions → models → kernels, adapted from the site's
    pages;
  - benchmarks: summaries of `results/`, linking the raw data in the
    journal repo.

## 7. Legal checks before the first release

- **License (decided): MIT OR Apache-2.0**, the Rust ecosystem's
  default. Users pick either:
  - MIT is short and permissive: do anything, keep the notice;
  - Apache-2.0 is also permissive, and adds an explicit patent grant (a
    contributor cannot later sue users over patents in their
    contribution), which companies often require;
  - offering both lets the crate combine with almost anything, including
    GPL-2.0 projects, which Apache-2.0 alone does not.

  Files: `LICENSE-MIT`, `LICENSE-APACHE`, and `license = "MIT OR
  Apache-2.0"` in every `Cargo.toml`. Contributions come in under the
  same terms (a line in CONTRIBUTING).
- **Unicode tables** (`tokenizer/unicode_tables.rs`) are generated from
  Unicode data: add the Unicode license notice.
- **Hexagon:**
  - our C includes SDK headers, and the `qaic` glue is generated at build
    time; none of the SDK is in the repo;
  - shipping a **prebuilt** `libaether_dsp.so` (e.g. in the AAR) needs a
    read of the SDK license's redistribution terms first. Until then,
    users build it with their own SDK.
- **Model files are never shipped.** The docs link to Hugging Face; Gemma
  has its own terms of use.
- **Test references** (llama.cpp and HF outputs) are generated by `tools/`,
  not shipped.

## 8. Steps, and what can start now

**Now, before the phone run (additive, no file moves).** On a new branch
`library` off `gemma4-devices`, in its own worktree (`../Aether-lib`), so
tonight's run in `../Aether-next` is untouched. These steps add files and a
layer on top; they do not rename or reformat what the phone checks
exercise, so tonight's fixes merge in without conflicts.

| Step | What | Check |
| --- | --- | --- |
| L1 | LICENSE-MIT, LICENSE-APACHE, README, CONTRIBUTING (correctness rule, no copied code), CHANGELOG, the Unicode notice | — |
| L2 | `aether::Error` and the `Model` / `Backend` / `Session` API as a new module over today's `Session` / `GpuSession` / `DspSession` (nothing removed yet) | Greedy text through the new API equals the CLI's, on CPU and GPU (PC); bad input returns an error, not a panic |
| L3 | 3–4 user-facing examples on the new API (generate, chat, logits, C) | They build and run on the PC |
| L4 | CI workflow for the host: tests with the small models, clippy, an Android cross-compile check | Green on a fresh clone |
| L5 | `rustfmt.toml` chosen and tried on a copy (not applied) | The diff it would make is reviewed |

**After the Gemma merge (the moves):**

| Step | What | Check |
| --- | --- | --- |
| R1 | Make `aether-llm` with `git filter-repo`: engine paths only, moved into the workspace layout (section 3); then the rustfmt commit | PPL and greedy text identical on all backends (vk_ppl, dsp_ppl, `--ppl`), all tests pass, the app runs |
| R2 | The CLI and the C ABI rebuilt on the L2 API; the old session types and modules become `pub(crate)`; panics and prints out of library paths | The same checks; C ABI tests |
| R3 | Experiments to `bench/`; docs/ (backends, architecture, benchmarks) | Examples build in CI |
| R4 | Android AAR + sample app | The app builds from the AAR alone |
| R5 | Tag v0.1.0, publish the crate `aether-infer`, release notes with measured numbers; link `aether-llm` from this repo's README and site | A stranger's quick start works in a clean container |

R1 is mostly mechanical (an hour or two); R2 is the bulk. Both are
engineering rather than new inference concepts, so they fit between
learning phases.

## 9. Decisions (taken 2026-11-05)

1. **Repos:** a new engine repo, `seshuthota/aether-llm`, made with
   filtered history; this repo stays the learning journal, unchanged.
2. **License:** MIT OR Apache-2.0 (section 7).
3. **Names:** the project stays "Aether"; the GitHub repo is `aether-llm`.
   On crates.io both `aether` and `aether-llm` are taken (`aether-llm` by
   another project's "Aether AI agent framework", which would confuse
   people). The crate is **`aether-infer`** (free on 2026-11-05), with
   `[lib] name = "aether"` so code still reads `use aether::Model`.
   `aether-lm` and `aether-edge` were also free.
4. **Formatting:** rustfmt in the engine repo from the split on (section 5).
5. **No copied code** from llama.cpp or any engine; compatible formats and
   interfaces only (section 1).

## 10. Risks

- **API churn while the engine still grows** (A3 optimisations, C1–C5 on
  the DSP). R1 is a move only. R2's API sits above the backends, so backend
  work keeps going underneath.
- **One maintainer.** Keep the public promise small: what is supported,
  what is verified, what is experimental (the DSP's Gemma path until C7).
- **Hexagon on locked phones:** unsigned modules load only on some devices.
  State it, and keep CPU / GPU as the portable paths.
- **The learning goal:** the restructure must not stall the phases. It is
  scheduled work with its own checks, done between phases.
