# Hand-off: what kquant-decode can close before the phone is back

Written 2026-11-04 on `kquant-decode` at `1d48657`, after the K3 review fixes
(`review-k0-k3.md`). The phone comes back this evening. Until then, close
everything whose answer does not depend on phone timing. Three parts, in this
order; Part 2 runs in the background while Part 1 is worked on.

House rules (unchanged): one commit per item with its check; `src/lib.rs`
untouched; no `cargo fmt`; comments in the style of the surrounding code;
correctness only from external references (llama.cpp, gguf, numpy goldens,
HF); never a second implementation of our own as an oracle. `feedback.txt`,
`review-k0-k3.md` and this file stay untracked.

---

## Part 1. Run the NEON code on the PC under QEMU

### Why

Both bugs in the K3 review were in code only the phone executes:
`repack_weights()` is false on x86 and every NEON kernel is
`#[cfg(target_arch = "aarch64")]`. So the host suite (121 passing) proves
nothing about `dot4_q4_k_q8_k`, `dot4_q6_k_q8_k`, the x4 kernels,
`unpack_k4`, the `smmla` path or the dispatch with repacked weights. QEMU user
mode runs aarch64 Linux binaries on x86 with `-cpu max`, which implements
dotprod (`sdot`, including the indexed form) and i8mm (`smmla`) exactly as the
ISA defines them. That makes every correctness check on the phone list
runnable now; only the timings need the phone.

QEMU proves correctness, not speed. Never record a time measured under QEMU.

### 1a. Toolchain (no sudo)

1. Rust target: `rustup target add aarch64-unknown-linux-musl`. Use musl
   (static, no glibc sysroot needed), not `-android` (needs bionic's
   `/system/bin/linker64`) and not `-gnu` (needs an aarch64 glibc and gcc we
   do not have).
2. Linker: first try Rust's bundled one, in `.cargo/config.toml` (gitignored,
   machine-specific):
   ```toml
   [target.aarch64-unknown-linux-musl]
   linker = "rust-lld"
   rustflags = ["-C", "link-self-contained=yes"]
   runner = "scripts/qemu-run.sh"
   ```
   If `rust-lld` does not find the CRT objects, fall back to the NDK's clang
   (`$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang
   --target=aarch64-linux-musl`) with `link-self-contained=yes`. Write down
   which one worked. Add the same block, with placeholder paths, to
   `.cargo/config.toml.example`.
3. QEMU without installing it: in a new, empty directory (e.g.
   `~/.local/opt/qemu-deb/`), `apt-get download qemu-user-static` (no root
   needed for `download`), then `dpkg-deb -x qemu-user-static_*.deb
   ~/.local/opt/qemu`. The binary is `~/.local/opt/qemu/usr/bin/qemu-aarch64-static`.
   Record its version (`--version`). The package is a downloaded artifact:
   extract it in its own directory and do not run anything else from it.
   If Ubuntu's package is older than QEMU 7 (i8mm support), say so and stop;
   do not build QEMU from source without asking.
4. `scripts/qemu-run.sh` (committed), like `scripts/adb-run.sh` in spirit:
   ```sh
   #!/bin/sh
   # Cargo runner for aarch64-unknown-linux-musl: runs the binary under QEMU
   # user mode with dotprod and i8mm (and no SVE, like the phone's Oryon).
   exec "${QEMU_AARCH64:-$HOME/.local/opt/qemu/usr/bin/qemu-aarch64-static}" -cpu max,sve=off "$@"
   ```
   `sve=off` keeps the CPU like the phone's (dotprod, i8mm, bf16, fp16, no
   SVE).

### 1b. Make the crate build for the musl target

- `cargo build --release --target aarch64-unknown-linux-musl --lib --tests`.
- Code behind `target_os = "android"` (in `session.rs`, `ffi.rs`,
  `model/dsp.rs`, `kernels/vulkan.rs`, `kernels/hexagon.rs`) is off on musl,
  which is what we want; the `vulkan` / `hexagon` features stay off.
- If something does not build only because the target is Linux and not
  Android (affinity, `sys::`), fix it with the narrowest `cfg`
  (`any(target_os = "android", target_os = "linux")`) and say which. Do not
  change engine behaviour on Android.

### 1c. Prove the harness can see the bugs (negative controls)

Before trusting a pass, show that QEMU fails where the phone would have:
1. Check out `572f689` (K3 before the fixes), run
   `cargo test --release --target aarch64-unknown-linux-musl --lib
   k_quant_linear_matches_reference -- --nocapture`. Expected: the panic
   `a K-quant row needs Q8_K activations` (bug 1).
2. Check out `ad57c09` (bug 1 fixed, bug 2 not). Same test. Expected: the
   Q6_K x4 cases fail with a large relative error (bug 2).
3. Back to the tip. Same test passes.

Also print, once, what the engine detects under QEMU
(`neon_quant::has_dotprod()`, `has_i8mm()`, `k::repack_weights()`); all must
be true, otherwise the x4 and `smmla` paths are not being run.

Record the three outcomes (raw output in
`results/phase-kquant/raw/qemu-negative-controls.txt`). If 1 or 2 does
not fail, the harness is not exercising the phone path: find out why before
going on.

### 1d. The correctness checks, at the tip

Run under QEMU, raw output to `results/phase-kquant/raw/qemu-tests.txt`:
- `--lib quant::` (quantizer bytes against llama.cpp, scalar and NEON dots,
  repack round trips);
- `--lib kernels::` (every kernel golden, including
  `k_quant_linear_matches_reference`: row-major and x4, decode and `smmla`
  prefill, 1 and 3 threads);
- `--lib session::` and `--lib model::gguf::` (the speculative/rollback tests,
  batched against decode, `lfm3_conv_state_rolls_back`,
  `lfm3_draft_decoding_keeps_the_greedy_text` — the latter forks only at a
  near tie on aarch64, which QEMU should reproduce);
- then the whole `--lib` suite, with `--test-threads=1` if memory is short
  (the PC has ~16 GB free), skipping `lfm3_matches_transformers` as on the
  host.

QEMU is roughly 10–50× slower than native; a full-model test may take long.
Run in the background and report the wall time; skip a test only with a
reason written down.

Expected: everything that passes on the phone passes here. Integer `sdot` /
`smmla` and IEEE f32 FMA are deterministic, so results should match the phone
bit for bit; the phone run tonight still confirms that once.

### 1e. Update the phone checklist

In `results/phase-kquant.md`, "Phone checks pending": mark every
correctness-only item that QEMU has now covered as "done under QEMU
(raw/qemu-tests.txt)", and leave on the phone list only what needs the real
CPU: `kquant_bench`, `bench_suite`, `simpleperf` and the annotated listings,
the bandwidth ceiling, peak RSS, sustained runs, and one confirming
`--lib k_quant` run on the phone. Add a short "QEMU harness" section (what it
is, the toolchain, the negative controls, what it can and cannot prove).

Commits: `scripts/qemu-run.sh` + config example; any `cfg` fix; the results
section and raw logs.

---

## Part 2. K2 accuracy on the PC (background, sequential)

### Why

K2 changed the activation rounding for K-quant matrices (Q8_0 → Q8_K), so
PPL and KL are allowed to move, and the plan asks by how much. The host runs
the scalar kernels without repacking, which is exactly K2's arithmetic; K3
only regroups the same integers (checked under QEMU in Part 1). So these
numbers answer K2's accuracy questions now, and the phone only needs a spot
check.

### How

- Build once: `cargo build --release` on the host at the tip.
- Reuse the **exact invocations** from the earlier runs; read them from the
  raw files' headers and from `results/phase-lfm3.md` A5 /
  `results/phase-6.md` (the `aether MODEL --ppl …` lines, `--ctx`,
  `--chunks`, `--kl`, `--base`). Same text (`testdata/wikitext/wikitext-2-raw/
  wiki.test.raw`), same chunking, same base, so only K2 differs. Do not
  change flags between before and after.
- One run at a time (`nice`), in the background; the LFM3 f32 base alone is
  ~7 GB. Write each to `results/phase-kquant/raw/pc-k2-*.txt` with the date,
  commit and the command line at the top.
- Approximate times from before: LFM3 Q4_K_M ctx 2048 one chunk with KL
  ~45 min; LFM3 20 × 512 full KL ~2.8 h; Qwen much faster.

### Runs, in this order

| # | run | before (Q8_0 activations) | llama.cpp |
| --- | --- | --- | --- |
| 1 | LFM3 Q4_K_M, ctx 2048, 1 chunk, PPL + KL | 13.7917, KL 0.052408, same top 88.759 % | 13.9561, 0.055758, 88.172 % |
| 2 | Qwen2.5-0.5B Q4_K_M, 20 × 512, PPL + KL | 16.7088, KL 0.031828, 90.43 % (phase 6, `sdot` path) | 16.7657 ± 0.688 |
| 3 | LFM3 Q4_K_M, 20 × 512, PPL + KL | 24.8542, KL 0.115041, 83.392 % | 24.8278, 0.117968, 83.392 % |
| 4 | control: Qwen2.5-0.5B Q4_0, 20 × 512, PPL | the phase-6 PC value | — |

Run 4 has no K-quant matrix, so it must be **unchanged to the digit**; if
it moves, something outside the K-quant path changed: stop and find it.

(If the Qwen/LFM3 "before" numbers in the table differ from what the raw
files say for the exact same command, the raw files win; correct the table.)

### Write-up

A "K2 accuracy (PC)" subsection in `results/phase-kquant.md`: the table with
the after column, the shift toward or away from llama.cpp, and the KL against
BF16 (A5 found our Q8_0 rounding slightly closer to BF16 at ctx 2048; say
whether Q8_K gives that up, and by how much). Tick the matching phone items
as "PC done; phone spot check only".

---

## Part 3 (only after Part 1 passes). K4 design note, then maybe the kernel

### The layout question to settle first

K3's repacked layout interleaves rows **4 bytes** at a time (for `sdot`, one
lane per row). `smmla` wants row pairs interleaved **8 bytes** at a time
(`[row a 8 bytes | row b 8 bytes]`, see `zip8` / `interleave4`). So a batched
kernel cannot read the K3 layout directly without shuffles. Options:
- **A.** Keep K3's 4-byte layout for decode and keep prefill's de-interleave
  (today, `unpack_k4`). No new code.
- **B.** An 8-byte layout (llama.cpp's `block_q4_Kx8` with blocklen 8, used by
  both `ggml_gemv_q4_K_8x8_q8_K` and `ggml_gemm_q4_K_8x8_q8_K`) serving both
  paths: decode then needs one `vpaddq` per pair of lanes (as llama.cpp does),
  prefill uses `smmla` directly with Q8_K activation pairs.
- **C.** Store both layouts: costs a second copy of the weights (~1.1 GB on
  LFM3). Rejected unless the numbers force it; say why.

Write this as a short "K4 design" section in `results/phase-kquant.md`
with the trade-off, referencing llama.cpp's
`ggml/src/ggml-cpu/arch/arm/repack.cpp` (`ggml_gemm_q4_K_8x8_q8_K`,
`ggml_gemm_q6_K_8x8_q8_K`) and `repack.cpp` (`block_q4_Kx8`). The choice
between A and B is the phone's (pp and tg numbers); do not merge B as the
default before that.

### If time remains: implement B behind a switch

- Its own commits, after the design note; A stays the default.
- New repacked variants and a Q8_K activation-pair type for `smmla`
  (two tokens interleaved 8 bytes at a time per 256-value block, the two
  `d`s, and `bsums` for Q4_K's min term).
- Checks, all under QEMU and on the host scalar path: the `_k` goldens
  (`out_q4_k_k.npy`, `out_q6_k_k.npy` are `[t, d]`, so they cover batched
  cases), the repack round-trip test for the new layout, and decode through
  the same layout. The batched cases with odd token counts (t = 3, 5, 7, 11,
  13) must pass, as for the existing tiles.
- A switch to select it (env var or a `Transformer` option) so the phone can
  measure A against B in one session.

---

## What to report back

1. Part 1: toolchain used (target, linker, QEMU version), the three
   negative-control outcomes, the test totals under QEMU with wall times, any
   `cfg` changes, and the shortened phone checklist.
2. Part 2: the filled table (after column), with the run-4 control, and the
   commits.
3. Part 3: the design note's recommendation and whether B was implemented
   and verified.
4. Anything that failed or was skipped, with the output.
