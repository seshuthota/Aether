# K-quant decode at the bandwidth ceiling

Plan: `plans/kquant-decode.md`. Branch `kquant-decode`. All numbers are from
the phone (iQOO 13, Snapdragon 8 Elite) unless marked host.

## K0. Baseline and a kernel benchmark

### The benchmark

`examples/kquant_bench.rs` loads the Q4_K_M file and the QAD-Q4_0 file with
`load_any` (file layout, no repack), keeps one matrix per distinct
(format, rows, cols) and times `kernels::linear_on` on it: activation
quantized to Q8_0, then the decode kernels. The weight bytes are the file's,
so the K-quant block patterns are real. The Q4_0 file supplies the same
shape in the format the engine already decodes at the ceiling.

Output columns: `median_us`, `median_gbps` (weight bytes / median time),
`best_gbps`, and `ns_per_256` = median time / (rows × cols / 256), the time
to stream and dot one 256-weight super-block of one row, including the
per-call activation quantization. `--cores` is the single-thread arm
(default 7), `--pool` the multi-thread arm (default 2-7).

```
cargo run --release --target aarch64-linux-android --example kquant_bench -- \
    testdata/gguf/LFM3-1.7B-Q4_K_M.gguf
```

### The shapes in the file

LFM3 Q4_K_M has 136 Q4_K and 18 Q6_K tensors. `ffn_down` is **both**: Q6_K in
the first and last eighth of the layers, Q4_K in the middle (llama.cpp's
Q4_K_M mix). The benchmark therefore covers it twice, as it does every
distinct (format, shape).

| format | tensor | rows | cols | weight bytes |
| --- | --- | ---: | ---: | ---: |
| Q6_K | `token_embd` (tied classifier) | 128000 | 2048 | 215,040,000 |
| Q4_K | `ffn_gate` / `ffn_up` | 7680 | 2048 | 8,847,360 |
| Q6_K | `ffn_down` (first/last layers) | 2048 | 7680 | 12,902,400 |
| Q4_K | `ffn_down` (middle layers) | 2048 | 7680 | 8,847,360 |
| Q4_K | `shortconv.in_proj` | 6144 | 2048 | 7,077,888 |
| Q4_K | `shortconv.out_proj` / `attn_q` | 2048 | 2048 | 2,359,296 |
| Q4_K | `attn_k` | 512 | 2048 | 589,824 |
| Q6_K | `attn_v` | 512 | 2048 | 860,160 |
| Q4_K | `attn_gate` | 16 | 2048 | 18,432 |

### Host smoke (not the baseline)

The full run is `raw/k0-host-smoke.txt` (2026-11-04, x86_64, 16 CPUs, scalar
path: the NEON kernels do not run there). It proves the harness and the shape
set; its GB/s are roughly 25× below the phone's and are not comparable with
A7. Host 6-thread medians, for orientation only:

| case | K-quant | Q4_0 at the same shape |
| --- | ---: | ---: |
| `ffn_gate` (7680 × 2048) | 1.64 GB/s | 4.43 GB/s |
| `ffn_down` Q6_K (2048 × 7680) | 1.25 GB/s | 4.43 GB/s |
| `ffn_down` Q4_K (2048 × 7680) | 1.62 GB/s | 4.43 GB/s |
| `token_embd` Q6_K (128000 × 2048) | 1.29 GB/s | — |

### Ceiling

Same-session reads at the benchmark's working-set sizes, and the sustained
rate over the whole 1.1 GB file:

```
cargo run --release --target aarch64-linux-android --example bw_sweep -- \
    --kernel read --cores 7 --min-kb 32 --max-mb 1024 --steps 4 --target-mb 512 --samples 7
cargo run --release --target aarch64-linux-android --example bw_sweep -- \
    --kernel read --cores 2-7 --parallel --min-kb 32 --max-mb 1024 --steps 4 --target-mb 512 --samples 7
cargo run --release --target aarch64-linux-android --example bw_sustained -- \
    --cores 2-7 --parallel --seconds 30 --size-mb 1024
```

Phase 1 measured 61 GB/s on one prime core and 67 GB/s on all cores; A7 put
the Q4_0 decode at 44.7 GB/s, so the file read is not the limit the kernels
are hitting.

### Phone baseline (pending)

The phone was not attached when this was written. To fill this section:

```
cargo run --release --target aarch64-linux-android --example kquant_bench -- \
    testdata/gguf/LFM3-1.7B-Q4_K_M.gguf > results/phase-kquant/raw/k0-phone.txt

cargo run --release --target aarch64-linux-android --example bench_suite -- \
    testdata/gguf/LFM3-1.7B-Q4_K_M.gguf -c 2-7 --pp 64,128,512 --tg 64 \
    --reps 3 --json --monitor
```

Record with each run: battery, screen state, `cpu0`/`cpu6` caps, temperature
before the run (A7's `wait_cool` in `scripts/lfm3-a7.sh` is the model).
Profile as in A7 (`raw/a7/simpleperf.txt`), then the annotated listing:

```
adb shell simpleperf record -e cpu-cycles:u -o /data/local/tmp/aether/perf.data -- \
    /data/local/tmp/aether/aether testdata/gguf/LFM3-1.7B-Q4_K_M.gguf \
    -i 'The history of the printing press' -t 0 -n 300 -c 2-7
adb shell simpleperf report -i /data/local/tmp/aether/perf.data --sort symbol
```

and `report_html.py` on a pulled `perf.data` + binary cache for the
instruction-level annotation of `dot4_q4_k_q8_0` and `dot4_q6_k_q8_0`
(simpleperf documents annotated source/disassembly through `report_html.py`).

K0's "done when" is satisfied by this section once it has: per-shape GB/s
(1 and 6 threads), the ceiling at the same working set, the profile with the
annotated listing, and the end-to-end `tg64` baseline with conditions.

## K1. Cheap fixes inside the Q8_0 activation format

The activation format is unchanged, so every result is bit-identical: the
integer sums are the same, only where and how often they are computed
changes.

### What changed

- **Q4_K's activation sums are computed once per matmul.** `load_xk` used
  to compute Σ q · d_x for the eight activation blocks of every super-block
  for every group of four weight rows (16 `vaddlvq` per super-block per
  group). `neon_quant::sums_qd` now computes the whole vector once, next to
  `xd`, in `linear_batch_on` / `linear_many_batch_on`, and `dot4_q4_k_q8_0`
  loads the eight values per super-block. For a 7680-row matrix this removes
  1919 of every 1920 repetitions of that work.
- **Q6_K no longer computes the sums at all.** `q6_k_block` never used
  `XK::sv` (Q6_K has no min term); `load_xk` was still computing it. The
  field is gone and Q6_K pays nothing.
- The host path uses `quant::sum_qd` (scalar); the phone path uses
  `neon_quant::sums_qd` (`vaddlvq_s8`). Both produce the same f32 values.

### Deferred, with reasons

- **Q6_K's FMA chain.** Splitting the eight sub-block FMAs into separate
  accumulators changes the f32 addition order, so it cannot be bit-identical.
  `dot4_q6_k_q8_0` already interleaves the block loop over four rows (four
  independent chains), so the annotated listing has to show whether the
  remaining 8-deep chain costs anything before it is changed. If it does, it
  belongs with K2/K3, where results are allowed to move.
- **Q4_K scale/min decode.** `q4_k_scales_mins` + `u8x8_to_f32` are per row
  and per super-block; there is no value shared across the four rows. Grouping
  the four rows' decodes before the row loop is possible but is guesswork
  without the annotated listing. Deferred to the phone profile.

### Checks

- Host: `cargo check --all-targets` and the three K-quant tests pass
  (`k_quants_match_gguf`, `k_quant_linear_matches_reference`,
  `k_quant_mix_with_imatrix_matches_llama_quantize`). The host runs the scalar
  kernels, so these do not exercise the changed NEON code.
- Phone: the K-quant tests, PPL unchanged to the digit (LFM3 and Qwen), and
  `kquant_bench` / `tg64` before and after. See the pending list below.

## K2. Q8_K activations for K-quant matmuls

The K-quant matmuls now use llama.cpp's activation format: one f32 scale
per 256 values and `bsums[16]`. The sub-block scales apply in int32, so a
whole super-block converts once instead of once per 32 values. This step
changes results (the rounding is llama.cpp's).

### Format and quantizer

- `quant::BlockQ8_K`: `d: f32`, `qs: [i8; 256]`, `bsums: [i16; 16]`, 292
  bytes, ggml's layout.
- `quant::quantize_q8_k` follows `quantize_row_q8_K_ref` operation for
  operation: `max` is the signed value of largest magnitude (first on ties),
  `iscale = -127 / max`, `d = 1 / iscale`, and rounding is ggml's
  `nearest_int` (ties to even), not `roundf` — the two differ on exact
  halves.
- Reference: `tools/gen_q8k.py` calls llama.cpp's own
  `quantize_row_q8_K_ref` through ctypes from `libggml-base.so` (the same
  approach as `gen_quant_imatrix.py`; no C shim needed) and writes
  `x_q8_k.bin` plus `out_q4_k_k.npy` / `out_q6_k_k.npy` (float64
  `dequant(X_q8_K) @ dequant(W)ᵀ`) next to the `_b` files.
  `cargo test k_quant` compares our quantizer byte for byte.

### Kernels

- `dot4_q4_k_q8_k`: per super-block, the eight sub-blocks' `sdot` sums are
  multiplied by their 6-bit scales in int32 (`vmlaq_n_s32`); the min term is
  `bsums · mins` with `vmlal_s16`; then two converts and two FMAs per
  super-block per row. Before: one convert and one FMA per sub-block, plus
  16 `vaddlvq` for the min term's activation sums.
- `dot4_q6_k_q8_k`: the 16 sub-blocks' `sdot` sums are multiplied by their
  int8 scales in int32 and converted once per super-block per row. The
  dependent 8-FMA chain is gone with the format.
- Single-row `dot_q4_k_q8_k` / `dot_q6_k_q8_k` for tails; scalar versions
  in `quant::scalar` for the host.
- Removed: the Q8_0 K-quant kernels and K1's `sums_qd` hoist (the Q8_K
  path has `bsums`, so the hoist has nothing to feed). The K1 code remains
  only at commit `5116346`; its phone measurement would have to be taken
  from that commit.

### Dispatch

- Decode (`t = 1`) and batched products on CPUs without i8mm quantize `x`
  to Q8_K once per matmul when a K-quant matrix is present, and to Q8_0
  once for the other formats.
- Prefill with i8mm still uses Q8_0 + `unpack_k4`; K4 decides whether to
  move it to the new layout/format. Until then batched and single-token
  K-quant results differ by the activation rounding, and the tests use the
  matching reference (`_b` for the `smmla` path, `_k` otherwise).

### Checks

- Host: `cargo test --release --lib k_quant` passes — quantizer bytes equal
  llama.cpp's, scalar Q8_K dots match the float64 `_k` reference, and the
  model path (`linear_batch_on` / `linear_on`, one and three threads) is
  within the bound.
- `cargo check --target aarch64-linux-android --all-targets` passes; the
  NEON kernels are not run on the host.
- Host, one chunk of LFM3 Q4_K_M (ctx 512, 255 tokens scored): PPL
  **10.3471**. Before K2 the same chunk was 10.3666; llama.cpp's is 10.3223.
  The Q8_K rounding moved us toward llama.cpp, as the plan expected.
- Pending: the full 20-chunk PPL/KL on the phone (and the Qwen file); see
  the list below.

## K2 accuracy (PC)

The host runs the scalar kernels without repacking, which is exactly K2's
arithmetic; K3 only regroups the same integers (checked under QEMU). These
runs use the same text, chunking and base as the earlier numbers, so only
K2 differs. One at a time, `nice`, with the command at the top of each
`raw/pc-k2-*.txt`.

```
target/release/aether MODEL --ppl testdata/wikitext/wikitext-2-raw/wiki.test.raw \
    --chunks N [--ctx 2048] --kl --base BASE
```

| # | run | before (Q8_0) | after (Q8_K) | llama.cpp |
| --- | --- | --- | --- | --- |
| 1 | LFM3 Q4_K_M, ctx 2048, 1 chunk, PPL + KL | 13.7917, KL 0.052408, 88.759 % | **14.0234, KL 0.054800, 89.345 %** | 13.9561, 0.055758, 88.172 % |
| 2 | Qwen2.5-0.5B Q4_K_M, 20 × 512, PPL + KL | 16.7088, KL 0.031828, 90.43 % | **16.7186, KL 0.032851, 90.137 %** | 16.7657, 0.033112, 90.10 % |
| 3 | LFM3 Q4_K_M, 20 × 512, PPL + KL | 24.8542, KL 0.115041, 83.392 % | **24.9094, KL 0.118284, 83.078 %** | 24.8278, 0.117968, 83.392 % |
| 4 | control: Qwen2.5-0.5B Q4_0, 20 × 512, PPL | 19.7470 (phase 6, stale) | **19.7441**; K1 (pre-K2) also **19.7441** | — |

Run 1: PPL and KL moved **toward** llama.cpp — the PPL gap shrank from
0.164 to 0.067 and KL from 0.052408 to 0.054800 (llama.cpp 0.055758) —
while same-top moved from 88.759 % to 89.345 % (llama.cpp 88.172 %). Against
BF16 the KL rose from 0.052408 to 0.054800: the cost side A5 predicted for
adopting llama.cpp's Q8_K rounding.

Run 2: every number moved toward llama.cpp — PPL gap 0.0569 → 0.0471, KL
gap 0.001284 → 0.000261, same-top gap 0.33 → 0.04 points.

Run 3: KL moved toward llama.cpp (gap 0.0029 → 0.0003) but PPL moved away
(0.0264 → 0.0816) and same-top fell 0.314 points; against BF16 the KL rose
0.115041 → 0.118284. A mixed result, not the uniform move at ctx 2048.

Run 4 has no K-quant matrix. The phase-6 value 19.7470 predates the LFM3
merge, so the proper before is the pre-K2 tree: `5116346` (K1) gives
**19.7441**, the same digits as the tip (`raw/pc-k2-qwen-q4_0-k1.txt`). K2
and K3 therefore leave the Q4_0 path bit-identical; the phase-6 number was
stale, not a regression.

## K3. Repacked K-quant rows for decode

Q4_K and Q6_K are repacked four rows at a time at load, with the four rows
interleaved **four bytes at a time**, so one `sdot` result lane is one row.
N = 4 (the existing group size) rather than llama.cpp's 8: it fits the
current `group` / `ROW_BLOCK` machinery, and with a Q8_K activation the
lane-per-row form needs no pairwise adds. N = 8 remains a phone experiment
(the plan asks to try both; without the phone the simpler one is in).

### Layouts

- `BlockQ4_Kx4` (592 bytes for four rows, +16 over four row-major blocks):
  the 6-bit scales and mins are pre-decoded and transposed
  (`scales[j*4 + r]`), so the kernel applies them without unpacking;
  `qs[128*c + 16*v + 4*r..+4]` is row `r`'s four bytes of chunk `c`.
- `BlockQ6_Kx4` (840 bytes, exactly four row-major blocks): `ql`/`qh` keep
  the row-major bytes, interleaved four at a time per 128-value half;
  `scales[s*4 + r]` is 16-value sub-block `s` of row `r`.
- `Weight::Q4_Kx4` / `Q6_Kx4`, `Weight::repack` arms (mapped pages released,
  so the file is not held twice), `dequantize_q4_kx4` / `q6_kx4`.

### Kernels

- `dot4_q4_k_q8_k_x4`: eight `sdot`s per sub-block pair, each lane one row;
  scales per row in int32, mins from `bsums`; two converts and two FMAs per
  super-block. No horizontal reduction at all (the four lanes of `acc` are
  the four rows).
- `dot4_q6_k_q8_k_x4`: 16 sub-blocks, each four `sdot`s; int8 scales per
  row in int32; one convert and one FMA per super-block.

### Prefill

`quant_rows_mmla` still uses `smmla` with Q8_0 activations. `unpack_k4` now
de-interleaves the repacked rows into `KBlock4` (the same per-matmul unpack
as before, now from the repacked buffer), so prefill is unchanged in
results. K4's direct prefill path remains measurement-gated.

### Bit-identity

**Not bit-identical to K2 in the last bits.** The integer sums are the same,
but K2's row-major kernel converts four partial int32 lanes per row and sums
them in f32 at the end, while the lane-per-row kernel converts one total
integer per row. The plan's K3 check says "bit-identical"; that is only true
for the integers, not for the f32 grouping. PPL/KL must be compared on the
phone, not promised identical.

### Checks

- Host: `repacked_k_rows_dequantize_like_row_major` passes (Q4_K and Q6_K
  rows dequantize exactly like row-major); the scalar
  `dot4_*_q8_k_x4` kernels match the float64 `_k` goldens; the full suite
  passes.
- `cargo check --target aarch64-linux-android --all-targets` passes. The
  NEON x4 kernels and the `unpack_k4` de-interleave only run on the phone.

## K4. Prefill on the new layout (measurement-gated)

The plan's decision is "by measurement": read the repacked K layout directly
(no `unpack_k4`) and use Q8_K activations, llama.cpp's
`gemm_q4_K_8x8_q8_K`; or keep Q8_0 + `smmla`.

Current state: prefill keeps `smmla` with Q8_0 and **de-interleaves** the
repacked rows in `unpack_k4` (the same per-matmul unpack prefill paid
before, now from the repacked buffer). Results are unchanged; the cost is
the copy. The direct path needs a batched kernel, and the N=4 layout's four
bytes per row does not fit `smmla`'s 2×8 matrix without extra shuffles, so
the choice is a real one and needs the phone's pp numbers. It is not
implemented blind.

On the phone: compare pp64/128/512 with K3's bridge. If the de-interleave
costs more than a few percent, the direct path is the next step; if not,
K3's bridge is enough and the batched kernel is not worth its risk.

## K5. Measure and close (pending)

Not run: the phone was not available. The order is:

```
scripts/lfm3-a7.sh results/phase-kquant/raw/k5
```

(two rounds, reversed, Q4_K_M and QAD-Q4_0, with a two-minute sustained
decode), then K0's `bw_sweep` / `bw_sustained` at the same working set, the
profile with the annotated listing, and the Qwen Q4_K_M decode before/after. The target is Q4_K_M decode near the QAD-Q4_0 file's
44.7 GB/s, roughly 40+ tok/s. `IMPLEMENTATION.md` and the merge into `main`
wait for those numbers.

## QEMU harness (correctness without the phone)

The two K3 bugs lived in code only the phone runs: `repack_weights()` is
false on x86 and the NEON kernels are `#[cfg(target_arch = "aarch64")]`.
QEMU user mode runs the aarch64 Linux build on the PC, with `-cpu max,sve=off`
so it has dotprod (`sdot`, including the indexed form) and i8mm (`smmla`),
and no SVE like the Oryon. It proves **correctness**, not speed: no timing
from QEMU is recorded.

- Toolchain: `rustup target add aarch64-unknown-linux-musl`; linker
  `rust-lld` with `-C link-self-contained=yes` (the target's
  `lib/self-contained` CRT objects), runner `scripts/qemu-run.sh`.
- QEMU: `qemu-user-static` 8.2.2, downloaded with `apt-get download` and
  extracted under `~/.local/opt/qemu` (no root).
- The engine detects `dotprod=true i8mm=true repack_weights=true` under it,
  so the x4 and `smmla` paths really run.

Negative controls (`raw/qemu-negative-controls.txt`): `572f689` panics with
`a K-quant row needs Q8_K activations`; `ad57c09` fails the Q6_K x4 case
with `rel 1.58e0`; the tip passes. That is the harness seeing both fixed
bugs, not a vacuous pass.

One test change: `session::tests::lfm3_draft_decoding_keeps_the_greedy_text`
judged a fork by its index (`d >= 8`). K2's Q8_K rounding moved a 4.6e-4
near tie from token 8 to token 7, so the test now checks the gap only —
the gap is the invariant; the index was incidental.

Under QEMU (`raw/qemu-tests.txt`):

| run | result | wall |
| --- | --- | ---: |
| `--lib quant::` | 7 passed | 0:30 |
| `--lib kernels::` | 17 passed | 0:33 |
| `--lib session::` | 5 passed after the gap-only test change | 6:06 |
| `--lib model::gguf::` (skip `lfm3_matches_transformers`) | 6 passed | 9:50 |
| `lfm3_matches_transformers` alone | 1 passed | 52:09 |
| full `--lib` (skip `lfm3_matches_transformers`) | 121 passed, 1 failed | 37:18 |

`model::gguf::` with all seven tests was SIGKILLed (signal 9) while the
LFM3 f32/BF16 match test ran beside the Part 2 accuracy run. Run alone
after that memory was free, `lfm3_matches_transformers` passes under QEMU
too (29 short + 9 long logits, every layer tap and 16 greedy tokens match
HF). So all seven `model::gguf::` tests are covered.

The full suite's one failure is QEMU-only:
`sys::cpu::tests::cache_line_matches_aligned_buf` reads CTR_EL0 as 32 bytes
under `-cpu max`, while the Oryon reports 64; the engine's 64-byte
`AlignedBuf` is a multiple of both. It is a host-CPU property test, not
engine logic, and the phone run passes. No engine change.

Host and QEMU runs are separate; the phone still confirms the same bits
once.

## K4 design: the layout question

K3 interleaves four rows **4 bytes** at a time, which makes each `sdot`
result lane one row. `smmla` wants row pairs **8 bytes** at a time
(`[row a 8 bytes | row b 8 bytes]`, see `zip8` / `interleave4`), so a
batched kernel cannot read the K3 layout without shuffles. Three options:

- **A. Keep K3's 4-byte layout for decode, de-interleave in prefill**
  (today's `unpack_k4`). No new code; prefill pays the per-matmul copy it
  already paid before K3.
- **B. An 8-byte layout for both paths**, llama.cpp's `block_q4_Kx8` /
  `block_q6_Kx8` with blocklen 8 (`ggml/src/ggml-cpu/repack.cpp`, used by
  `ggml_gemv_q4_K_8x8_q8_K` and `ggml_gemm_q4_K_8x8_q8_K` in
  `arch/arm/repack.cpp`). Decode then needs one `vpaddq` per row pair (as
  llama.cpp does) and prefill uses `smmla` directly with Q8_K activation
  pairs. One layout, no de-interleave.
- **C. Store both layouts.** A second copy of the weights (~1.1 GB on
  LFM3), rejected unless the measurements force it.

Recommendation: **A stays the default until the phone's pp and tg numbers
say otherwise.** If the prefill de-interleave costs more than a few percent
of pp64/128/512, B is the next step: new repacked variants, a Q8_K
activation-pair type for `smmla` (two tokens interleaved 8 bytes at a time
per 256-value block, both `d`s, and `bsums` for Q4_K's min term), and the
`_k` batched goldens (t = 3, 5, 7, 11, 13) as its checks. B is not merged
as the default before the phone comparison.

## Review fixes (2026-11-04)

Two bugs in the phone-only path, found before the phone checklist:

- **Repacked K-quants did not ask for Q8_K.** `linear_batch_on` /
  `linear_many_batch_on` tested only the row-major `Q4_K` / `Q6_K`, so on the
  phone (where `Weights::repack` turns them into `Q4_Kx4` / `Q6_Kx4`) `xk`
  stayed `None` and `dot4` panicked on the first decode matmul. Fixed with
  `needs_q8_k(w)`, which lists all four variants; `is_k_quant` uses it too
  (`ad57c09`).
- **Q6_K x4 dropped the 0xAA flip.** The lane kernel built `4·q` read as
  int8 instead of `4·(q − 32)`, so every Q6_K value was off by 32 in q units.
  The flip is back, hoisted out of the loops, with the doc comment updated
  (`9a7b59d`). The host scalar x4 kernel was already correct, which is why
  the host tests passed.

Performance items from the same review:

- **Lane `sdot`s.** Both x4 kernels now load 16 activation bytes once and
  use `vdotq_laneq_s32` (stable on Rust 1.99) for four `sdot`s, instead of a
  `vld1q_dup_s32` before every one. Bit-identical; the before/after is for
  `kquant_bench --repack` on the phone.
- **No Q8_0 for K-only matmuls.** `quant_rows` now takes the column count;
  `linear_batch_on` / `linear_many_batch_on` build `xq`/`xd` only when a
  non-K-quant matrix is present (or `smmla` prefill needs the pairs). An
  LFM3 q/k/v or gate/up decode group quantizes Q8_K alone. Bit-identical.
- **Q6_K x4 `ql`/`qh` reloads.** The 16 sub-blocks re-load `ql` twice and
  `qh` four times per super-block; whether LLVM merges them is for the
  annotated listing. If not, restructure by `(n, j)` as
  `q6_k_block_q8_k` does. Added to the phone list.
- **N = 8** was not tried; the K3 section says why (4 fits the existing
  group and lane form). It remains the first experiment if the 4-row kernel
  is short of the ceiling.

Cleanups: `at` is `4 * v` for every `v` in 0..8 (the `if` was redundant) in
all five places and the layout comment; `(nbk, nk)` in the golden test is
one name now.

## Phone checks pending

The phone was not available during implementation. Correctness-only items
are now covered under QEMU (`raw/qemu-tests.txt`); what remains needs the
real CPU. Run these in one session when it is back, in this order. Each
step is a commit; for a step's own before/after, check out its commit
(`25bed76` K0, `5116346` K1, `79439bc` K2, `a1ac2a9` K3 with the review
fixes) and measure that tree.

- [ ] K0: `kquant_bench` on the phone, 1 and 6 threads (`raw/k0-phone.txt`).
- [ ] K0: `bw_sweep` / `bw_sustained` ceiling at the same working set.
- [ ] K0: `bench_suite --tg 64`, cooled, battery, caps recorded.
- [ ] K0: `simpleperf` symbol report + annotated listing of both kernels.
- [ ] K1 (only at `5116346`; K2 replaced these kernels):
      `cargo test --release --target aarch64-linux-android --lib k_quant`.
- [ ] K1 (only at `5116346`): LFM3 Q4_K_M PPL 20 × 512 unchanged to the
      digit (24.8822 on the phone).
- [ ] K1 (only at `5116346`): Qwen2.5-0.5B Q4_K_M PPL unchanged.
- [ ] K1 (only at `5116346`): `kquant_bench` and `tg64` after, against K0.
- [ ] K1 (only at `5116346`): annotated listing: did the 16 `vaddlvq` per
      group disappear, and is `q4_k_scales_mins` / `u8x8_to_f32` visible?
- [x] K2: quantizer bytes, scalar and NEON dots, model path — done under
      QEMU (`raw/qemu-tests.txt`: `quant::` 7, `kernels::` 17).
- [x] K2: LFM3 Q4_K_M PPL 20 × 512 — PC done: **24.9094**, KL 0.118284,
      83.078 % (llama.cpp 24.8278 / 0.117968 / 83.392 %); phone spot check
      only.
- [x] K2: LFM3 Q4_K_M context 2048 — PC done: **14.0234**, KL 0.054800,
      89.345 % (llama.cpp 13.9561 / 0.055758 / 88.172 %); phone spot check
      only. KL against BF16 rose 0.052408 → 0.054800, the cost side A5
      predicted.
- [x] K2: Qwen2.5-0.5B Q4_K_M — PC done: **16.7186**, KL 0.032851,
      90.137 % (llama.cpp 16.7657 / 0.033112 / 90.10 %); phone spot check
      only.
- [ ] K2: `kquant_bench` and `tg64` after, against K0/K1 (phone only).
- [x] K2: the speculative and batched-vs-decode tests pass — done under QEMU
      (`session::` 5, `model::gguf::` 6).
- [x] K3: row-major and x4 kernels, decode and prefill — done under QEMU
      (`kernels::` 17, including `k_quant_linear_matches_reference`).
- [ ] K3: LFM3 Q4_K_M PPL against K2 (same integers, different f32
      grouping; record the digits). Phone only: the PC is K2 arithmetic.
- [ ] K3: `kquant_bench --repack` against the row-major lines, and `tg64`;
      if the x4 kernels are short of the ceiling, try N = 8 (phone only).
- [ ] K3: peak RSS stays ~1.1-1.2 GB (the mapped file is released; phone).
- [x] K3: prefill (`unpack_k4` from the repacked rows) gives the same
      batched results — done under QEMU (`kernels::`, `model::gguf::`).
- [x] K3 fix: an LFM3 Q4_K_M greedy run does not panic (repacked K-quants
      get Q8_K) — done under QEMU (`model::gguf::`, `session::`); the bench
      still runs on the phone.
- [x] K3 fix: the Q6_K x4 goldens pass (the 0xAA flip) — done under QEMU;
      the phone spot-checks the PPL.
- [ ] K3: `kquant_bench --repack` before/after the lane-`sdot` rewrite
      (phone only).
- [ ] K3: annotated listing of `dot4_q6_k_q8_k_x4`: are the `ql`/`qh`
      loads merged across the 16 sub-blocks? If not, restructure by
      `(n, j)` (phone only).
- [ ] Phone: one confirming `--lib k_quant` run at the final tip.
