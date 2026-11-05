# Plan: Aether as an open-source inference library

Written 2026-11-05. Facts below were read from the tree at `gemma4-devices`
(`dd16acc`) and from GitHub. Nothing here is built yet: it starts after
`gemma4-devices` merges (plans/gemma4-phone-checks.md), so the move does
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

## 1. What "like llama.cpp" should mean here

llama.cpp is four things:
1. **ggml:** a tensor library with backends.
2. **libllama:** model loading, the KV cache, sampling, and a C API.
3. **Tools:** CLI, server, quantize, perplexity, bench.
4. **Bindings** that other people write on top.

Aether should copy 2–4 and **not** 1. Our models are written by hand per
architecture, on purpose (that is how the project teaches). A general graph
library would be a different project.

The position to claim is narrow and real: **phone-first inference.** That
means CPU (NEON / i8mm), Adreno (Vulkan) and Hexagon (HVX / HMX) in one
library, verified against llama.cpp and HF at every step, with the measured
numbers published.

## 2. Repository split

Two repos:

- **The engine** (new, public): the library, C API, CLI tools, shaders,
  DSP code, the Android library, tests, reference generators, docs, and
  benchmark summaries.
- **The learning journal** (this repo, as today): site, films, notes,
  plans, raw results, `CLAUDE.md`. It depends on the engine (git submodule,
  or a Cargo git dependency), and its pages link to the engine's code.

How the engine repo gets its history:
- **(a) Recommended:** `git filter-repo` keeping only the engine's paths.
  The 310 commits stay; their messages tell the from-scratch story and
  carry the verification numbers.
- **(b)** A fresh first commit. Simpler, but loses that story.

The old history stays public in this repo either way, so filtering is
about a clean layout, not about hiding anything.

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
- **Formatting:** the repo does not use `cargo fmt` today. The split is the
  one moment to adopt a `rustfmt.toml` close to the current style (one
  reformat commit). Otherwise CI does not check formatting.

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

- **License:**
  - MIT OR Apache-2.0 is the Rust norm, and llama.cpp is MIT;
  - Apache-2.0 adds a patent grant;
  - the user decides.
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

## 8. Steps

| Step | What | Check |
| --- | --- | --- |
| R0 | Decisions: license, name, repo split / history, formatting (section 9) | — |
| R1 | Workspace split, files moved, no logic change | PPL and greedy text identical on all backends (vk_ppl, dsp_ppl, `--ppl`), all tests pass, the app runs |
| R2 | `Model` / `Backend` / `Error` API; the CLI and the C ABI rebuilt on it; panics and prints out of library paths | The same checks; C ABI tests; a test that bad input returns an error |
| R3 | Examples down to 3–4 (generate, chat, logits, C); experiments to `bench/` | Examples build in CI |
| R4 | README, LICENSE, CONTRIBUTING, docs/, CI | CI green on a fresh clone; a stranger's quick start works (try it in a clean container) |
| R5 | Android AAR + sample app | The app builds from the AAR alone |
| R6 | Create the public engine repo (filtered history), tag v0.1.0, publish the crate and the release notes with measured numbers | — |

R1–R2 are the bulk (a few sessions). They are engineering, not new
inference concepts, so they fit between learning phases, not in place of
them.

## 9. Decisions for the user

1. **Repos:** a new engine repo with filtered history (recommended), or
   restructure this repo in place and move the learning material out?
2. **License:** MIT OR Apache-2.0 (recommended, the Rust norm), or MIT
   only, as llama.cpp?
3. **Name:** keep "Aether" as the project name with a free crate name
   (e.g. `aether-llm`; to be checked), or pick a new name?
4. **Formatting:** adopt rustfmt at the split, or keep the current
   hand-formatted style?

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
