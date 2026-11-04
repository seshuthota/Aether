# Plan: K-quant decode at the bandwidth ceiling

Written 2026-11-04, after the LFM3 merge. Work on branch `kquant-decode` in `../Aether-impl`;
merge into `main` only after K5's checks pass. Review after each step.

## 1. Why

LFM3-1.7B Q4_K_M decodes at 26.1 tok/s on cores 2–7 against llama.cpp's 44.3, same session
and clock caps (`results/phase-lfm3.md` A7, round 2). Prompts are level (0.86–1.01×). The
profile (`raw/a7/simpleperf.txt`) puts 87 % of decode in two functions:

| symbol | cycles |
| --- | ---: |
| `dot4_q4_k_q8_0` | 54.1 % |
| `dot4_q6_k_q8_0` | 32.9 % |
| thread pool | 8.3 % |
| everything else | 4.7 % |

Against the bandwidth ceiling (file bytes × tok/s): ours 28.5 GB/s, llama.cpp 48.3 GB/s, our
QAD-Q4_0 decode 44.7 GB/s. So the phone can stream this file at ~45–48 GB/s, and our K-quant
kernels are compute-bound well below it. Not thermal: per-token times are flat within a run
(median 37.3 ms, p95 39.9 ms) and higher caps (2.75 / 2.65 GHz) only reach 30.6 tok/s.

LFM3 is the first model whose decode runs on the K-quant GEMVs: Qwen2.5-0.5B's Q4_K_M is
mostly Q5_0 (896-wide rows are not a multiple of 256), so these kernels were verified
(`tools/gen_quant.py` goldens) but never tuned.

What it teaches:
- **Compute-bound vs bandwidth-bound**, measured: the same bytes per token, two speeds.
- **Activation format as a design choice.** A per-32 activation scale forces a float
  convert and FMA per 32 values; a per-256 scale (Q8_K) lets a whole super-block sum in
  integers. The rounding changes too, and PPL/KL show by how much.
- **Weight layout for SIMD.** Interleaving rows so each lane of a dot product belongs to
  one row removes the horizontal reductions, and decoding scales once per group of rows
  amortizes their cost.

## 2. What llama.cpp does differently

`~/llama.cpp` (`d02b4c0d6`, b12061, the build A7 ran), on this phone (dotprod + i8mm):
`ggml/src/ggml-cpu/repack.cpp` picks `q4_K_8x8_q8_K` for Q4_K and `q6_K_8x8_q8_K` for Q6_K;
the decode kernels are `ggml_gemv_q4_K_8x8_q8_K` / `ggml_gemv_q6_K_8x8_q8_K` in
`arch/arm/repack.cpp`.

| | ours | llama.cpp |
| --- | --- | --- |
| weights | file layout, 4 rows per call | repacked at load: 8 rows interleaved 8 bytes at a time |
| activation | Q8_0: f16 scale per 32 | Q8_K: f32 scale per 256, `bsums[16]` (Σq per 16) |
| sub-block scales (Q4_K) | 6-bit unpack per row, → f32, float FMA | unpacked once per 8 rows, int32 multiply |
| min term (Q4_K) | Σx·d_x per 32, 16 `vaddlvq` per super-block per 4 rows (`load_xk`) | `bsums` once per token, `vmlal_s16` with the mins |
| per-row result | `sdot` lanes mix one row's values → `vpaddq` reductions | each lane is one row: no reductions |
| float work | Q4_K: 2 cvt + 4 FMA per row per super-block; Q6_K: 8 cvt + 8 FMA, one dependent chain | 1 cvt + 1 FMA per 64 values per 4 rows |

Q6_K in ours also chains all 8 sub-block FMAs of a super-block through one accumulator
(`q6_k_block`), so each waits for the last (4-cycle FMA latency).

## 3. Scope

In:
- The CPU decode path (`quant_rows`, `dot4`, `dot1`) for Q4_K and Q6_K, NEON with dotprod.
- The activation format those matmuls use, including `linear_many_on` (q/k/v and gate/up
  share one quantized `x`, and a Q4_K_M layer can mix Q4_K, Q6_K and other types).
- Batched prefill (`quant_rows_mmla`, `unpack_k4`) only as far as K4 decides: decode first.
- LFM3 Q4_K_M as the main benchmark; Qwen2.5-0.5B Q4_K_M as the second (its Q4_K / Q6_K
  matrices change too).

Out:
- The GPU and DSP K-quant paths.
- Other K-quants (Q2_K, Q3_K, Q5_K): none of our files use them. The layout work should not
  close the door.
- The thread pool's 8.3 % (spin and hand-off): a separate look once the kernels are faster,
  since its share will grow.
- `aether quantize` producing Q4_K / Q6_K.

## 4. Steps

Each step is one commit with its check. Models without Q4_K / Q6_K matrices (stories15M,
SmolLM2 and Qwen Q8_0/Q4_0, LFM3 QAD-Q4_0) stay bit-identical through all of them: their
tests and PPL unchanged to the digit. Files with K-quant matrices change only where a step
says so.

### K0. Baseline and a kernel benchmark

- `examples/kquant_bench.rs` (or a K-quant section in `gemv_bench.rs`): one GEMV at LFM3's
  real shapes and types, read from the Q4_K_M file so the bit patterns are real:
  `ffn_gate`/`ffn_up` (2048 → 7680), `ffn_down` (7680 → 2048), `shortconv.in_proj`
  (2048 → 6144) and the tied classifier, per type found. Reports GB/s of weights and ns per
  super-block, 1 thread (core 7) and 6 threads (cores 2–7), with `black_box` as usual.
- The ceiling to compare with: `bw_sweep` / `bw_sustained` at the same working-set size, and
  the Q4_0 GEMV at the same shape.
- `simpleperf report --sort symbol` and an annotated listing of `dot4_q4_k_q8_0` /
  `dot4_q6_k_q8_0`, so each later step can say which instructions it removed.
- Record the end-to-end decode again (`bench_suite --tg 64`, cores 2–7, battery, cooled,
  caps written down) as this plan's baseline.

Done when: `results/phase-kquant.md` K0 has the per-shape GB/s, the ceiling, the profile and
the baseline tok/s. Expected: Q4_K / Q6_K GEMVs far below the Q4_0 GEMV and the read ceiling.

### K1. Cheap fixes inside today's format

No change to the activation format, so results stay **bit-identical** (the sums are the
same integers; only where and how often they are computed changes).
- Compute each activation block's Σq · d once per matmul, next to `xd`, instead of in
  `load_xk` for every group of 4 rows.
- Q6_K: break the FMA chain. Sum the 8 sub-blocks with separate accumulators (or pairwise,
  as `q4_k_block` does with `vpaddq`) and fold them once per super-block.
- Q4_K: hoist what `q4_k_block` can share across its 4 rows; check in the annotated listing
  that `q4_k_scales_mins` and `u8x8_to_f32` are not the cost.

Check: all quant tests, the LFM3 and Qwen PPL **unchanged to the digit**; K0's bench and
decode tok/s before/after. This step measures how much of the gap is overhead around the
`sdot`s and how much is the format.

### K2. Q8_K activations for K-quant matmuls

- `BlockQ8_K` in `quant.rs` with ggml's layout: `d: f32`, `qs: [i8; 256]`,
  `bsums: [i16; 16]`; and our quantizer for it.
- **Reference**: llama.cpp's own `quantize_row_q8_K_ref` (`ggml/src/ggml-quants.h`,
  exported by `libggml-base`). A small C program in `tools/` (like `llama2c_encode.c`)
  linked against `~/llama.cpp/build/bin/libggml-base.so` writes the reference bytes for the
  activations in `tools/gen_quant.py`'s K-quant cases (the `gguf` Python package encodes
  Q8_0 but not Q8_K). Our quantizer must match those bytes exactly, including the sign
  convention (ggml scales by the signed max, `iscale = -127 / max`).
- Goldens: `out_q4_k_k.npy`, `out_q6_k_k.npy` = dequant(X_q8_K) @ dequant(W)ᵀ in float64
  (gguf dequantizes the weights; Q8_K dequantizes as `d · q`), next to today's `_b` files.
- Kernels: `dot4_q4_k_q8_k`, `dot4_q6_k_q8_k`, and the single-row versions. Within a
  super-block the sub-block scales multiply the int32 `sdot` sums in integers; one convert
  and FMA per super-block (or per 64 values, as llama.cpp does, if that measures faster).
  Q4_K's min term from `bsums` and the mins in int16/int32 (`vmlal_s16`).
- Dispatch: a K-quant matrix gets Q8_K activations. `linear_many_on` quantizes `x` once
  per format that its matrices need (Q8_0 and/or Q8_K).
- Scalar versions in `quant::scalar` for the host, against the same goldens.

Check: goldens on host and phone; then PPL / KL, which now **change** (the rounding is
llama.cpp's):
- LFM3 Q4_K_M 20 × 512: today 24.8542 (PC) / 24.8822 (phone); llama.cpp 24.8278. Expect
  closer to llama.cpp, not necessarily equal (other ops still round differently).
- LFM3 Q4_K_M context 2048: today 13.7917, KL 0.052408; llama.cpp 13.9561, KL 0.055758.
  Expect ours to move toward llama.cpp's; record the KL either way. The A5 finding (our
  Q8_0 rounding was slightly closer to BF16) is the cost side of this step: write it down.
- Qwen2.5-0.5B Q4_K_M PPL against llama.cpp's, as in phase 6.
- Decode tok/s and K0's bench before/after.

### K3. Repacked K-quant rows for decode

- Repack Q4_K and Q6_K at load, as `repack_q4_0` / `repack_q5_0` already do for the
  `smmla` kernels (`Weights::repack`, `k::repack_weights`): N rows interleaved so each
  `sdot` result lane belongs to one row; scales and mins of the N rows stored together so
  they are decoded once per group. N = 4 (our existing group size) or 8 (llama.cpp's): try
  both in K0's bench and keep the faster one; write down why.
- New `Weight` variants (`Q4_Kx4`/`Q4_Kx8`, …), `dequantize_*` for a repacked row, and the
  test that repacking only moves bytes (`repacked_rows_dequantize_like_row_major`).
- Memory: repacking happens once at load into a new buffer. Check peak RSS: the 1.1 GB file
  must not be held twice (today's repack path; confirm it frees or maps per matrix).

Check: K2's goldens through the repacked kernels (bit-identical to K2's results: same
integers, same rounding); PPL to the digit against K2; decode tok/s and the bench.

### K4. Prefill on the new layout

Today's `smmla` prefill unpacks K-quant blocks per tile (`unpack_k4`) and uses Q8_0
activation pairs (`PairQ8`). After K3 the weights are repacked anyway, so:
- Decide, by measurement, whether prefill uses the repacked K layout directly (no
  `unpack_k4`) and Q8_K activations (llama.cpp's `gemm_q4_K_8x8_q8_K`).
- If prefill keeps Q8_0 while decode uses Q8_K, batched and single-token logits differ by
  more than today: the speculative tests (`lfm3_draft_decoding_keeps_the_greedy_text`) and
  the batched-vs-decode checks must still pass with their stated tolerances. Prefer one
  format for both.

Check: pp64/128/512 not slower than today (level with llama.cpp now); the batched goldens;
PPL (which runs batched) as in K2.

### K5. Measure and close

- `scripts/lfm3-a7.sh` against llama.cpp, same session, two rounds in reversed order,
  battery, cooled, caps recorded: pp and tg64@0/256/512 for Q4_K_M and QAD-Q4_0.
- Bandwidth: file bytes × tok/s against K0's ceiling. Target: Q4_K_M decode near our
  QAD-Q4_0's 44.7 GB/s, i.e. roughly 40+ tok/s.
- Sustained 2-minute decode (tok/s, temperature, power), and the profile again: what is
  left, and is the thread pool now the next item?
- Qwen2.5-0.5B Q4_K_M decode before/after (phase 6 had us ahead of llama.cpp; check it
  stays so).

Close: `results/phase-kquant.md`, `IMPLEMENTATION.md` status, merge into `main`.

## 5. What the review will check

- **Correctness from references only**: llama.cpp's `quantize_row_q8_K_ref` bytes, gguf's
  dequantizers, numpy/float64 goldens, llama.cpp PPL/KL. No second implementation of our
  own as an oracle.
- **Bit-identical where promised**: K1 and K3 change no result; non-K-quant models never
  change.
- **Explained changes where expected**: K2's PPL/KL shift, with numbers before and after.
- **Each speedup attributed**: bench, profile and annotated listing per step, so the
  results say which change bought what.
- **Measurement hygiene**: same-session pairs, battery, cooled, caps and conditions in each
  raw file.
- **House rules**: `src/lib.rs` untouched on impl, no `cargo fmt`, comments in the style of
  the surrounding code.

## 6. References

- llama.cpp `d02b4c0d6`: `ggml/src/ggml-common.h` (`block_q8_K`, `block_q4_K`,
  `block_q6_K`), `ggml/src/ggml-quants.c` (`quantize_row_q8_K_ref`),
  `ggml/src/ggml-cpu/repack.cpp` (layout choice, `block_q4_Kx8` repacking),
  `ggml/src/ggml-cpu/arch/arm/repack.cpp` (`ggml_gemv_q4_K_8x8_q8_K`,
  `ggml_gemv_q6_K_8x8_q8_K`), `ggml/src/ggml-cpu/arch/arm/quants.c`
  (`ggml_vec_dot_q4_K_q8_K`, `ggml_vec_dot_q6_K_q8_K`: the non-repacked versions).
- Arm A64 ISA reference: `sdot`, `smmla`, `vmlal_s16`, `vpaddq`. Instruction costs on
  Oryon are measured (K0's bench and listing), not taken from another core's guide.
- In this repo: `src/kernels/neon_quant.rs` (`q4_k_block`, `q6_k_block`, `load_xk`),
  `src/kernels/mod.rs` (`quant_rows`, `linear_many_on`, `quant_rows_mmla`),
  `src/quant.rs` (formats, `repack_q4_0`), `tools/gen_quant.py`,
  `results/phase-lfm3.md` A5/A7, `plans/lfm3.md` (the plan format).
