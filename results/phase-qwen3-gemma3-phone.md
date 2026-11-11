# Qwen3 and Gemma 3 on the phone (plan D5, E3)

aether-infer `503ec46` (main, after the `qwen3-gemma3` merge), llama.cpp
`d02b4c0d6` (`llama-android`, Clang 18, the same commit as the PC checks).
iQOO 13 (SM8750), Android 16 (`BP2A.250605.031.A3`), on AC power, screen
held on by the runner. 2026-11-11.

## The phone test suite

`cargo test --release --target aarch64-linux-android --features
aether-cli/vulkan,aether-cli/hexagon --workspace -- --test-threads=1`: all
pass (226 library tests with the GPU and DSP ones, the C ABI, the CLI). The
only skips are the host-only files (`hostdata/`). It took three test fixes
(`503ec46`): the PC's decode and batch kernels are bit-identical, the
phone's are not (NEON / i8mm rounding), and three tests had assumed so —

- a reused chat turn against one fresh batch: the reply's KV entries come
  from decode steps; SmolLM2 Q8_0's logits differ by 0.55 (1.65e-2
  relative) and a 0.03 near tie flips the greedy text. The test now checks
  the reused cache against a fresh session replaying the same split (bit
  for bit, on the phone too) and the fresh batch within 5e-2;
- llama.cpp's penalty goldens: at step 12 of "repeat" llama.cpp's " smile"
  is 0.0055 behind our " heavy" on the phone;
- a test that read `dsp/gemma.c` at run time.

The C ABI test had failed the same way before Part D (`52abf6c`): an old
test assumption, not a regression.

## Perplexity and KL, 5 chunks of 512

`wiki.test.raw`, 1275 tokens scored (the second half of each chunk), 6
threads. llama.cpp's `--kl-divergence-base` file from the same Q8_0 GGUF is
the reference, so the KL is between the two engines' arithmetic on the same
weights. Logs: `phase-qwen3-gemma3-phone/ppl/`.

| Model (Q8_0) | llama.cpp CPU | Aether CPU | Aether Adreno | Aether Hexagon |
| --- | ---: | ---: | ---: | ---: |
| Gemma 3 270M | 48.2027 | 48.3605 (KL 0.0044, top 96.5 %) | 48.0097 (KL 0.0040, top 96.8 %) | 48.0167 (KL 0.0040, top 96.9 %) |
| Gemma 3 1B | 25.1669 | 25.0958 (KL 0.0029, top 98.0 %) | 25.1910 (KL 0.0022, top 98.4 %) | 25.1893 (KL 0.0022, top 98.3 %) |
| Qwen3 0.6B | 19.5426 | 19.5405 (KL 0.0016, top 97.7 %) | not supported | not supported |
| Qwen3 1.7B | 15.6079 | 15.5342 (KL 0.0020, top 98.0 %) | not supported | not supported |

Every Aether number is within 0.5 % of llama.cpp's. The GPU (`vk_gemma`,
the prompt pass in batches of 128) and the DSP (`dsp_gemma`, the HMX prompt
pass) are Gemma 3's first runs on those devices; their greedy text matches
each other ("…a young woman named Elara. She was known for her beautiful
singing voice…" for the 1B). Decode, 40 positions: 270M 82 tok/s on the
Adreno, 126 tok/s on the DSP; 1B 31 and 41 tok/s.

"Mean PPL(base)" (44.79 for the 270M) is below llama.cpp's own estimate
(48.20) because the base file stores clamped log-probabilities; llama.cpp
reading its own file prints the same 44.785842, so it is the format, not
our reader.

**Qwen3 runs on neither device.** It has Q/K norms, so it is not
`Config::uniform` and goes to the per-layer paths (`GemmaGpu`, `dsp/gemma.c`),
which both require Gemma's post-attention / post-FFN norms (`dsp_ppl`:
"layer 0 lacks a post-norm"); the uniform paths have no Q/K norm, and the
GPU's also asserts `head_size <= 64` (Qwen3: 128). The plan's note that the
per-layer paths "accept these shapes" was not run until now.

## Speed

`bench/compare.sh`, the method of `phase-llama32-phone/`: the CPU, 6
threads on cores 2–7 (`-C 0xfc`), `-p 64,256,512 -n 128 -d 0,512 -r 5`, two
rounds in opposite engine order, every run started below 42 °C (all at
41 °C; they end at 65–74 °C). Medians in tok/s; the min–max, JSON and
conditions are in `phase-qwen3-gemma3-phone/`.

| Test | 270M llama.cpp | 270M Aether | 1B llama.cpp | 1B Aether | 0.6B llama.cpp | 0.6B Aether | 1.7B llama.cpp | 1.7B Aether |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pp64 | 1188.1 | 1821.0 | 385.4 | 439.0 | 572.7 | 651.8 | 218.8 | 243.4 |
| pp256 | 1430.1 | 1821.7 | 418.7 | 417.1 | 495.6 | 551.6 | 219.0 | 205.2 |
| pp512 | 1143.9 | 1360.6 | 383.7 | 360.7 | 409.1 | 397.9 | 196.8 | 170.2 |
| tg128 | 182.7 | 182.8 | 54.7 | 54.6 | 82.1 | 85.0 | 31.7 | 29.1 |
| pp64 @ d512 | 239.7 | 563.4 | 132.4 | 223.6 | 184.5 | 249.3 | 103.6 | 131.9 |
| pp256 @ d512 | 728.2 | 909.4 | 269.9 | 284.0 | 241.3 | 220.8 | 125.4 | 111.6 |
| pp512 @ d512 | 712.2 | 894.1 | 263.4 | 276.6 | 207.8 | 192.6 | 114.6 | 98.2 |
| tg128 @ d512 | 145.5 | 139.6 | 48.9 | 46.6 | 50.1 | 57.6 | 24.6 | 27.2 |

(Gemma 3 is the "it" file of each size, Qwen3 the base-named Q8_0 GGUFs.)

- **Decode is level**: 0.92–1.15× across the four models; Aether leads at
  depth 512 on Qwen3 (1.11–1.15×) and trails slightly on Gemma 3
  (0.95–0.96×).
- **Short prompts are Aether's**: pp64 1.11–1.53×, and 1.27–2.35× at depth
  512 (llama.cpp's pp64 @ d512 on Gemma 3 is unusually slow: 240 tok/s on the
  270M against 728 at pp256).
- **Long prompts on Qwen3 are llama.cpp's**: pp512 0.86–0.97×, pp256 /
  pp512 at depth 512 0.86–0.93×, the same shape as the Llama 3.2 3B
  (`phase-llama32-phone-3b`): head 128 attention in the prompt pass. Gemma 3
  (head 256 on the 1B, but a 512-token sliding window on 5 of 6 layers) is
  level or ahead there.
- The Qwen3 1.7B's Aether runs have low outliers (pp512 109.4, pp256 @
  d512 52.4) in one repetition each; the medians are not moved by them.
