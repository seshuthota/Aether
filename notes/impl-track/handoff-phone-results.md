# Hand-off: the K-quant phone session (2026-11-04, 22:16–23:52)

Run by `scripts/kquant-phone.sh` (branch `phone-prep`, `../Aether-next`):
each step on its own commit in a temporary worktree, cooled to 45 °C before
every timed run, on battery (90 → 78 %), adb over Wi-Fi, screen on. Raw
files, each with its conditions header (caps, temperature, battery):
`results/phase-kquant/raw/phone/`. The first two attempts failed on script
bugs (`session-1.log`); the k0 data is from the second, the rest from the
third.

**Read every timing with its caps.** The vendor moved the caps between
1.79/1.96, 2.00/2.25 and 2.75/2.65 GHz during the session. Compare kernels
by their ratio to the Q4_0 line of the same `kquant_bench` run, and decode
runs only at equal caps (below).

## What the phone measured

### Decode and prompt (`*-tg64.txt`, `bench_suite`, LFM3 Q4_K_M, cores 2–7)

At equal caps, 2.00/2.25 GHz:

| | tg64 d0 / d256 / d512 | pp64 / pp128 / pp512 | peak RSS |
| --- | --- | --- | --- |
| K0 `25bed76` | 28.8 / 28.4 / 25.8 | 134.2 / 140.1 / 146.2 | 1,188,804 kB |
| K2 `79439bc` | 28.7 / 26.4 / 25.8 | 133.8 / 141.9 / 146.6 | 1,188,428 kB |
| tip `c2469c3` | **40.6 / 39.1 / 34.9** | 133.4 / 138.6 / 142.5 | 1,211,464 kB |

K1 `5116346` ran at 1.79/1.96 (27.6 / 26.8 / 26.7; pp 114.1 / 126.0 /
130.3), so it is not in the table.

### K5: against llama.cpp, same session, both at 1.79/1.96 GHz (`k5-*`)

| | ours r1 / r2 | llama-bench r1 / r2 |
| --- | --- | --- |
| tg64 d0 | 38.8 / 38.6 | 44.24 / 44.03 |
| pp64 | 118.7 / 118.1 | 132.41 / 134.59 |
| pp128 | 126.7 / 125.0 | 127.56 / 129.82 |
| pp512 | 125.8 / 125.0 | 130.13 / 130.62 |

Decode is at 88 % of llama.cpp (A7: 26.2 vs 44.3, 59 %); prompts at 90–98 %.

### Kernels relative to Q4_0, 6 threads (`k0-bench`, `k1-bench`, `k2-bench`)

| | K0 | K1 | K2 |
| --- | --- | --- | --- |
| Q4_K `ffn_gate` | 0.90 | 0.94 | 0.96 |
| Q6_K `ffn_down` | 0.79 | 0.79 | 0.91 |
| Q6_K `token_embd` (215 MB) | 0.45 | 0.58 | 0.66 |

(`token_embd` is divided by the `ffn_gate` Q4_0 line; there is no Q4_0
line at that shape.) The ~9 MB matrices partly stay in cache: repacked Q4_K
`ffn_gate` reads 75 GB/s against the 65–67 GB/s DRAM ceiling
(`k0-ceiling-6`, `k0-sustained`: 66.77 GB/s over 30 s). `token_embd` is
the line that measures DRAM.

### The lane-`sdot` rewrite, before/after alternated (`lane-*`)

Round 2, both arms at 2.00/2.25 GHz, repacked (x4) kernels, 6 threads:

| | `9a7b59d` | `3febb3a` | change |
| --- | --- | --- | --- |
| Q4_K `ffn_gate` | 59.6 GB/s | 75.3 | +26 % |
| Q4_K `ffn_down` | 53.2 | 70.0 | +32 % |
| Q6_K `ffn_down` | 44.9 | 36.1 | **−20 %** |
| Q6_K `token_embd` | 44.0 | 35.1 | **−20 %** |

Round 1 shows the same ratios. **The rewrite regressed the Q6_K x4
kernel by 20 %**, and the tip has it.

### Profiles (`k0-profile`, `k1-profile`, `tip-profile`, with `perf.data` and the binary)

| | K0 | tip |
| --- | --- | --- |
| Q4_K kernel | 53.41 % (`dot4_q4_k_q8_0`) | 39.81 % (`dot4_q4_k_q8_k_x4`) |
| Q6_K kernel | 33.68 % (`dot4_q6_k_q8_0`) | 40.46 % (`dot4_q6_k_q8_k_x4`) |
| pool worker | 6.74 % | 10.07 % |

The file's weight bytes are 716 MB Q4_K (66 %) and 374 MB Q6_K (34 %), so
per byte the Q6_K x4 kernel now costs ~2× the Q4_K x4 kernel.

### Accuracy on the phone

- `--lib k_quant`: 3/3 at K1 and at the tip.
- LFM3 Q4_K_M 20 × 512: **24.8822** at K1, K2 and the tip; context 2048
  **13.7811** at K2; Qwen Q4_K_M **16.7599** at K1, K2 and the tip (the
  same as phase 6's later runs).
- **Note:** on the phone `--ppl` runs through the batched `smmla` path
  (its own Q8_0 pairs, `unpack_k4` for repacked rows), so these numbers
  confirm the prompt path is unchanged; they do not exercise K2's Q8_K
  decode arithmetic. That is covered by the QEMU goldens and the PC PPL.
  The CLI has no decode-path PPL mode.

### Gemma 4 E2B in llama.cpp (`gemma-*`, for `plans/gemma4.md` G0; caps 2.40/2.44)

| | pp64 | pp512 | tg64 | tg256 | tg64 @ d512 | tg256 @ d512 |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_0 | 117.91 | 106.11 | 30.54 | 28.57 | 24.34 | 21.73 |
| Q4_K_M | 88.09 | 75.76 | 29.42 | 27.18 | 21.93 | 21.48 |

## Tasks

1. **Fix the Q6_K x4 regression.** In `dot4_q6_k_q8_k_x4`, `3febb3a`
   replaced four `dup4` broadcasts with one `vld1q_s8` plus
   `vdotq_laneq_s32::<lane>` in a `dot!` macro. The instruction count did
   not grow; look at the code generated (`tip-profile/perf.data` and
   `tip-profile/aether`, `report_html.py` from the NDK's `simpleperf/`,
   or `objdump -d`) against `9a7b59d`'s. Candidates:
   - the review's point 5: `ql` loaded twice and `qh` four times per
     super-block; restructure by `(n, j)` so each load serves its sub-blocks;
   - the four `sdot`s chained on one accumulator (`s16`): two accumulators;
   - `vshlq_u8` by a runtime `6 − 2k` inside the macro: hoist or make const.
   
   Bit-identical (same integers). Check with `kquant_bench --repack`
   against `9a7b59d` (alternated, as `scripts/kquant-phone.sh` stage `lane`
   does: `STAGES=lane` with the new commit in place of `3febb3a`), then tg64.
2. **Record the session in `results/phase-kquant.md`**: K0 baseline, K1–K3
   phone numbers, the lane table, K5, with caps beside every timing; tick
   the checklist items this covers.
3. **K4: decide A.** The de-interleave costs 2.8 % at pp512 (146.6 → 142.5)
   and 2.3 % at pp128; the plan's bar was "more than a few percent", so the
   4-byte layout stays and B is not built.
4. **Still open for the phone:** the instruction-level listings (K0 `vaddlvq`
   question, K3 `ql`/`qh` merge question) from the pulled `perf.data`; N = 8
   after task 1.
5. **Optional PC check** (offered to the user, not started): KL of our
   Q4_K_M against llama.cpp's own Q4_K_M logits (`--kl-divergence-base`
   made from the Q4_K_M file), K1 against the tip, Qwen first. It measures
   directly whether Q8_K brought our decode arithmetic closer to llama.cpp's.
