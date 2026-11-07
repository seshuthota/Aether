# LFM3-1.7B on the GPU and the DSP

Plan: `plans/lfm3-devices.md` in the engine repo. Branch `lfm3-devices`
off `main` at `f96f127`. All phone numbers are from the iQOO 13
(SM8750), battery, screen on, `scripts/adb-run.sh`'s lock; temperature
was not below the compare-script's 45 °C for every run (the PPL runs do
not time).

## L0: references and guard baselines

- `llama.cpp` `d02b4c0d6` (`llama-perplexity`), LFM3-1.7B-BF16.gguf:
  20 × 512 wikitext-2, PPL **23.4796 ± 1.05094**. The base file is
  `hostdata/lfm3/kl-base-bf16.bin`, 1.3 GB,
  sha256 `63122ebba1cb860eea2c4a3568ca366d8a3213c56d0280cffc775dcfda9b7d93`.
- Phone copy: the first 8 chunks, 522,272,724 B,
  sha256 `bfcfd53dcf87591b66241c540b58036578b9378b9c424bcf85f79acb81360a98`
  (`hostdata/lfm3/kl-base-bf16-8.bin`).
- With the base, the engine's CPU Q4_K_M gives PPL/base 1.055828
  (≈18.2242) and KLD **0.085114**, same top 85.196 % (8 chunks); the
  20-chunk CPU run is 24.8542 / KL 0.115041 (`phase-lfm3.md`).

Guards rerun on the branch (unchanged to the digit unless noted):

| Guard | Command | Result |
| --- | --- | --- |
| Gemma 4 GPU Q4_0, 1 chunk | `vk_gemma MODEL KLBASE 1 decode` | 24.4536 / KL 0.338552 |
| Gemma 4 DSP Q4_0, DDR | `AETHER_DSP_STREAM=0 dsp_gemma MODEL KLBASE 1 decode` | **24.4392 / 0.331835** |
| Gemma 4 DSP Q4_0, streamed | `dsp_gemma …` | **24.6625 / 0.332917** |
| Qwen2.5-0.5B GPU Q4_0, 4 chunks | `vk_ppl MODEL 4 512 batch` / decode | 15.8560 / 15.8023 |
| Host suite | `cargo test --release --workspace --features aether-infer/vulkan,aether-infer/hexagon -- --skip gemma4_matches_transformers --test-threads=1` | green (library 183 tests; see the run log) |
| clippy | host and Android | clean, `-D warnings` |

## S1/S2: the paths

- PLE is optional on both per-layer paths; `geglu.comp` takes a
  specialization constant (GELU/SiLU); `Config::uniform()` now means the
  uniform path's assumptions, so LFM3 routes to the per-layer paths.
- GPU: conv layers (in_proj, `short_conv.comp`, out_proj), 16 gate rows
  after v in the qkv row, a 624 KB f32 conv state, 11 attention layers'
  cache only (not 24).
- DSP: `desc` gains kind/width/taps/conv state and a 32-row padded gate
  run; HVX `short_conv_range` and `gate_heads_hvx`; the C loader and pass
  run both kinds; the taps are transposed to `[tap][channel]`.

## GPU (Adreno 830)

- LFM3 Q4_K_M, `vk_gemma`, 8 chunks (2040 scored): **PPL 18.3443,
  KL 0.092680**, same top 85.049 % (base PPL 17.2604). CPU on the same
  8 chunks: 18.2242 / 0.085114; the GPU is +0.7 % PPL and +0.0076 KL,
  the f32-activation vs Q8/Q8_K-activation and summation-order gap.
- LFM3 Q8_0, same: **PPL 17.2892, KL 0.000704**, same top 99.069 %
  (base 17.2604); Q8_0 is essentially BF16.
- Decode: Q4_K_M **24.7 tok/s**, Q8_0 **8.7 tok/s** at the start of the
  context (Q8_0 moves 1.8 GB a token). Phone CPU Q4_K_M is 45.7 tok/s
  (`results/release/`); the GPU is not faster, as the bandwidth ceiling
  predicted.
- Prompt: Q4_K_M 8 chunks in 79 s (4096 tokens through the prompt pass,
  batches of 128) ≈52 tok/s, against the phone CPU's ~139 tok/s
  (`results/release/`). The conv layers and the per-layer submissions are
  the cost; B7's stage profile is still owed.
- GPU stage profile (`bench/examples/profile_lfm3`, 16 steps, Adreno 830):
  gate/up **31.7 %**, down **25.8 %**, classifier **17.9 %**, in_proj
  **7.8 %**, q/k/v **4.3 %**, out_proj **3.3 %**, wo **2.7 %**, attention
  + merge **1.1 %**, norms **4.2 %**; the shaders sum to **2.10 ms/token**
  (477 tok/s) while the measured decode step is **~34 ms/token**: ~32 ms
  is CPU submit/fence/wait overhead per token, not shader time. The same
  profile on the Intel host sums to 10.75 ms/token.
- `bench/compare.sh`, LFM3 Q4_K_M, `-p 64 -n 32`, two rounds, starts
  49–55 °C: llama.cpp CPU pp64 **125.3** (89.7–160.9), tg32 **44.1**
  (43.8–44.5); aether GPU pp64 **66.1** (65.8–66.4), tg32 **29.1**
  (28.5–29.7). The GPU is 1.5x slower at decode and 1.9x at pp64.
- `aether run -dev gpu` works: 4-token prompt + 7 decode tokens, peak RSS
  **2.32 GB**.
- Host Vulkan: LFM3 layers 0–2 against an exact f32 reference built from
  the dequantized matrices match at 1e-6; the 29-token prompt pass against
  the decode step is <2e-3; the 320-token HF prompt passes with the Q8_0
  bound.

## DSP (Hexagon v79)

- LFM3 Q4_K_M, `dsp_ppl` 20 × 512, batch 192 (HMX prompt): **24.8928**
  (llama.cpp 24.8278, CPU 24.8542). Q8_0: **23.4725** (llama.cpp 23.5508,
  CPU 23.5482). QAD-Q4_0: **27.1759** (llama.cpp 27.1383, CPU 27.1557).
  The batch path is within ~0.3 % of the CPU.
- LFM3 Q4_K_M, 8 chunks with the KL base, `dsp_gemma … batch`:
  **PPL 18.4554, KL 0.099698**, same top 84.412 % (GPU 18.3443 / 0.092680,
  CPU 18.2242 / 0.085114). The f16 cache and Q8_0 activations account for
  the difference.
- Batch vs one-token decode, chunk 1: batch **10.3570** (matches the GPU's
  10.3576), streamed decode **10.9642**, DDR decode **16.6889** (after the
  C loader/gate fixes; the first, stale binary's numbers are not used).
  These were a bug, since fixed (C3 below).
- Speed, Q4_K_M: decode 64 positions at depth 0 **26.1 tok/s**
  (35.7 ms/token), depth 512 **25.8**, depth 2048 **25.1** (the cooled run
  below; an earlier run at 68–70 °C gave 14.5–15.0). `aether run -dev dsp`
  29.7 tok/s for 7 tokens; prefill 32/64/128/192 **88/163/282/370 tok/s**;
  peak RSS `aether run -dev dsp` **1.17 GB**.
- Depth profile: before depth 0 48.4 °C, after depth 2048 67.2 °C (aoss),
  hottest CPU 65.2 °C; the decode rate falls 26.1 → 25.1 tok/s as the
  context grows, so the KV reads are not the leading cost yet.
- `aether run -dev dsp` works; `dsp_ppl` 20 chunks 35.7 s (286 tok/s
  through the DSP).

## C3: the one-token decode gap, closed

The gap above was a race in `dsp/gemma.c`, not the Q8_0 activations. The
output gate (`gate_heads_hvx`) ran on all 16 heads of a token in that
token's thread right after attention, with no barrier, while other
threads were still writing their heads and quantizing them for `wo`. Some
heads reached `wo` ungated. The DDR path, with no barrier after
attention, lost the most; the stream path ends attention with a barrier
and lost less; the all-HMX pass does not quantize there and was right.
Fix: each thread gates its own heads (engine `840f12c`). The per-stage
HMX experiments and the two-level Q8_0 activation had only moved the
threads' timing.

Phone, LFM3 one-token decode after the fix (PPL runs, not timed; the phone
reached ~80 °C by the last run):

| Check | Before | After | Reference |
| --- | --- | --- | --- |
| Q4_K_M chunk 1, streamed | 10.9642 | **10.3672** | HMX prompt 10.3570, GPU 10.3576 |
| Q4_K_M chunk 1, DDR | 16.6889 | **10.4069** | the same |
| Q4_K_M 20 × 512 | | **24.8266** | llama.cpp 24.8278, CPU 24.8542 |
| QAD-Q4_0 20 × 512 | | **27.1670** | llama.cpp 27.1383, CPU 27.1557 |
| Q8_0 20 × 512 | | **23.5837** | llama.cpp 23.5508, CPU 23.5482 |
| Q4_K_M 8 chunks, KL base | | **18.3469 / KL 0.093721**, same top 85.049 % | GPU 18.3443 / 0.092680, CPU 18.2242 / 0.085114 |
| Q4_K_M context 2048, 1 chunk | | **13.7980** | PC: llama.cpp 13.9561, our CPU 13.7917 with Q8_0 activations (14.0234 with Q8_K) |
| Gemma 4 Q4_0 guard, streamed / DDR | 24.6625 / 24.4392 | **24.6625 / 0.332917, 24.4392 / 0.331835** | unchanged to the digit |

Decode stays 26 tok/s (Q4_K_M), 34.8 (QAD-Q4_0), 20.3 (Q8_0), first 64
positions. The 2048 context is one chunk, so its 1.1 % below llama.cpp is
within a single chunk's spread (±1.3); the HMX prompt pass at 2048 on a
cool phone would pin it down (done below).

### Context 2048 through the HMX prompt pass (2026-11-07)

`dsp_ppl LFM3-1.7B-Q4_K_M.gguf 1 2048 6 1f 192`, then the same with batch
1, engine `a309a74` (C3's fix in), on battery, started at 39 °C and 44 °C
on the hottest CPU zone. Script `phase-lfm3-devices/ctx2048-dsp.sh`; output
and conditions next to it.

| path | PPL, 1 × 2048 (1023 scored) | time |
| --- | ---: | ---: |
| HMX prompt pass, batch 192 | **13.8022 ± 1.29796** | 7.7 s (267 tok/s) |
| one-token decode | **13.7980 ± 1.29790** | 81.5 s (25.1 tok/s) |
| our CPU, Q8_0 activations (`phase-lfm3.md`, PC / phone) | 13.7917 / 13.7811 | |
| our CPU, Q8_K activations (`phase-kquant.md` K2) | 14.0234 | |
| llama.cpp CPU (PC) | 13.9561 | |
| BF16: HF transformers f32 / ours | 13.5848 / 13.5848 | |

The decode path reproduces C3's 13.7980 to the digit, and the HMX pass is
0.03 % from it. Both are within 0.1 % of our CPU run with the same
activation rounding (Q8_0 per 32 values, which the DSP uses), so the 1.1 %
below llama.cpp is the activation rounding `phase-lfm3.md` already
explained (llama.cpp rounds to Q8_K per 256), not a DSP error. The
"CPU 14.0234" in the C3 table was the CPU after the K2 change to Q8_K
activations, so it was the wrong row to compare with. Speeds in the same
run: decode 28.7 tok/s for the 59 positions after the prefill (batch run) and
26.3 for positions 0–63 (batch 1); prefill 32/64/128/192 89/164/287/376 tok/s, as recorded before.


## App and serve (phone)

- The APK built, installed and ran LFM3 Q4_K_M on the GPU: with
  `aether_set_thinking(0)` and `aether_set_hide_think(1)` the reply starts
  with the answer (`The sky appears blue due to ...`), not a thinking span;
  prefill 17 tok in 966 ms, **decode 178 tok at 27.8 tok/s**.
- `c_abi_gpu_lfm3` on the phone (aarch64) passes with thinking off.
- `aether serve -dev gpu` on the phone, Qwen2.5-0.5B Q4_0,
  `tools/serve_check.py` without llama-server: **0 failures** (stream =
  plain, JSON shape, multi-turn reuse 51/66, 4 clients, errors). With LFM3
  the same check's multi-turn reuse is 0: the API deliberately restarts a
  chat for a running-state model when the template does not continue the
  cache exactly (`api.rs` test), so the check's >10 expectation applies to
  stateless models.

## Still owed

1. ~~The DSP one-token decode discrepancy~~: closed (C3 above).
2. GPU QAD-Q4_0 on the phone; GPU against L0's OpenCL build if it loads
   LFM3 (the table above labels llama.cpp as CPU).
3. The submit/fence overhead the B7 profile exposes (~32 ms of the ~34 ms
   decode step): the next GPU optimization, not a stage shader.
