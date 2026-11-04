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
- Pending: host 1-chunk LFM3 Q4_K_M PPL (running) and the phone PPL/KL
  before and after; see the list below.

## Phone checks pending

The phone was not available during implementation. Run these in one session
when it is back, in this order:

- [ ] K0: `kquant_bench` on the phone, 1 and 6 threads (`raw/k0-phone.txt`).
- [ ] K0: `bw_sweep` / `bw_sustained` ceiling at the same working set.
- [ ] K0: `bench_suite --tg 64`, cooled, battery, caps recorded.
- [ ] K0: `simpleperf` symbol report + annotated listing of both kernels.
- [ ] K1: `cargo test --release --target aarch64-linux-android --lib k_quant`.
- [ ] K1: LFM3 Q4_K_M PPL 20 × 512 unchanged to the digit (24.8822 on the phone).
- [ ] K1: Qwen2.5-0.5B Q4_K_M PPL unchanged.
- [ ] K1: `kquant_bench` and `tg64` after, against K0.
- [ ] K1: annotated listing: did the 16 `vaddlvq` per group disappear, and is
      `q4_k_scales_mins` / `u8x8_to_f32` visible?
- [ ] K2: `cargo test --release --target aarch64-linux-android --lib k_quant`
      (quantizer bytes, scalar and NEON dots, model path).
- [ ] K2: LFM3 Q4_K_M PPL 20 × 512: today 24.8822 (phone); llama.cpp
      24.8278. Expect closer to llama.cpp, not necessarily equal.
- [ ] K2: LFM3 Q4_K_M context 2048: today 13.7811 (phone) / 13.7917 (PC),
      KL 0.052408; llama.cpp 13.9561 / 0.055758. Record the KL either way;
      A5 found our Q8_0 rounding slightly closer to BF16, which is the cost
      side of this step.
- [ ] K2: Qwen2.5-0.5B Q4_K_M PPL against llama.cpp's (phase 6).
- [ ] K2: `kquant_bench` and `tg64` after, against K0/K1.
- [ ] K2: the speculative and batched-vs-decode tests still pass with their
      stated tolerances (prefill is still Q8_0 on i8mm).
