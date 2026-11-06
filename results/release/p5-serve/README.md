# `aether serve` (plan P5) checks, 2026-11-06

`tools/serve_check.py` (aether-infer, branch `p5-serve`) against `llama-server`
(llama.cpp `d02b4c0d6`), both on qwen2.5-0.5b-q4_0.gguf, greedy, 48 tokens.

| File | Where | aether serve | llama-server |
| --- | --- | --- | --- |
| `pc-cpu.txt` | PC, i7-13620H | `-np 4 -t 8` (commit 60a4071, before `cache_prompt`) | `-c 4096 -t 8` (4 slots) |
| `pc-gpu-intel.txt` | PC, Intel iGPU, Vulkan | `-ngl 99` (before `cache_prompt`, spacer requests) | CPU, as above |
| `phone-cpu.txt` | iQOO 13, CPU | `-np 4 -t 8`: slots on cores {6,2} {7,3} {0,4} {1,5} | `-c 4096 -t 8`, on the phone |
| `phone-gpu.txt` | iQOO 13, Adreno 830 | `-ngl 99`, one slot | CPU, on the phone |
| `phone-dsp.txt` | iQOO 13, Hexagon | `-dev dsp`, one slot | CPU, on the phone |

All runs: 0 failures (JSON keys of responses and chunks as llama-server's,
prompt token counts equal, stream = plain, a second chat turn reusing 51 of
66 prompt tokens, N clients at once = one at a time, the 400s / 401).
Greedy text equal to llama-server's: phone CPU 4/6 chat + 2/3 text, Adreno
3/6 + 2/3, DSP 3/6 + 3/3; the rest part at near-ties after 24-140 characters.

The first phone CPU run (before `cache_prompt: false`) failed "stream =
plain" on 2 text prompts: the streamed request landed on the slot that held
the whole prompt, ran its last token as a decode step over the reused cache
instead of in one batched prefill, and took the other branch of a near-tie.
Now the compared requests run fresh; reuse is checked by the multi-turn
case.

The `timing` line of each phone file is one 64-token chat (24 tokens reused,
11 prefilled) with llama-server loaded idle beside it: one sample, not a
benchmark. Decode: CPU slot (2 cores) 74 tok/s, Adreno 95, DSP 128. SoC
(cpu-1-0-0) 47-51 °C at the end of each run, 44 °C at the start, on USB power.
