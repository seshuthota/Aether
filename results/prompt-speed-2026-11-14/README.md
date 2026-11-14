# Gemma 4 E2B prompt and decode speed: Aether, llama.cpp, LiteRT-LM (2026-11-14)

Why: the user saw Aether's prompt processing behind other engines in
chat apps (Aether's app, PocketPal = llama.cpp, Edge Gallery = LiteRT-LM).
Those screenshots had 12-token prompts, which measure fixed overhead, not
prompt speed. This measures 128–2048-token prompts directly.

iQOO 13 (SM8750), on battery (65% → 57%), screen on, each run started
below 45 °C. CPU runs on cores 2–7, 6 threads (`taskset fc` for the others).
- Aether: lfm3-audio build (CPU), and the same source with `vulkan,hexagon`;
  `aether bench -p 128,512,1024,2048 -n 64 -r 2`; GGUF Q4_0 / Q4_K_M.
- llama.cpp d02b4c0d6 `llama-bench`, same GGUFs. The phone's OpenCL and
  Hexagon llama.cpp builds (older) fail to load Gemma 4.
- LiteRT-LM: `litert_lm_main` already on the phone (Nov 9 build) with Edge
  Gallery's own `gemma-4-E2B-it.litertlm` (commit 6e5c4f1, CPU/GPU; no
  Qualcomm NPU build exists for it). Its `--benchmark_prefill_tokens` is
  ignored by this build, so it reads real prompts: wikitext-2 test text of
  127 / 511 / 1023 / 2047 Gemma tokens + "Summarize the text above in one
  sentence." (136–2056 tokens with its chat template). Aether and
  llama-bench use synthetic tokens of the same lengths.

Prompt tok/s (prompt length), decode tok/s, peak RSS (`toybox time -v`):

| Engine, backend, weights | 128 | 512 | 1024 | 2048 | decode | RSS |
| --- | --- | --- | --- | --- | --- | --- |
| Aether CPU Q4_0 | 156 | 131 | 107 | 88 | 34.5 | 1.88 GB |
| Aether CPU Q4_K_M | 86 | 81 | 74 | 60 | 28.2 | 1.88 GB |
| Aether DSP (Hexagon) Q4_0 | 150 | 131 | 118 | 105 | 25.2 | 1.78 GB |
| Aether GPU (Vulkan) Q4_0 | 27 | 35 | 35 | 34 | 18.3 | 3.32 GB |
| llama.cpp CPU Q4_0 | 91 | 87 | 79 | 68 | 23.9 | 4.46 GB |
| llama.cpp CPU Q4_K_M (warm rerun) | – | 63 | – | 54 | 26.4 | 4.59 GB |
| LiteRT-LM CPU | 41 | 81 | 95 | 107 | 16–21 | 1.4–2.1 GB |
| LiteRT-LM GPU | 184 | 216 | 429 | 380 | 18–24 | 0.68–0.86 GB* |

\* GPU memory is not all in RSS. LiteRT GPU's engine init took 9.5–14 s per
run (with its cache warm); Aether loads in about 1 s.

The first llama.cpp Q4_K_M run (in `cpu.txt`) is not used: the phone had
1 GB free (other apps held models), its numbers rose with prompt length
and spread ±9; the warm rerun above replaces it.

Reading:
- Aether leads decode on every backend (CPU 34.5 against 16–26).
- On the CPU Aether leads prompts up to 1024 tokens, but its prompt speed
  falls with length (156 → 88) while LiteRT's rises (41 → 107): at 2048
  LiteRT's CPU is ahead. The DSP falls the same way (150 → 105).
- LiteRT's GPU is the prompt leader: 380–430 tok/s at 1–2k tokens, 3.6–4×
  Aether's best (DSP 105–118) and 11–12× Aether's Vulkan (34–35 tok/s,
  flat with length, about twice its own decode).
