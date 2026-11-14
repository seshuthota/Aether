# Prompt speed: raw runs and conditions (2026-11-14)

`plans/prompt-speed.md` section 6: Aether CPU / DSP / GPU, llama.cpp's CPU
and LiteRT-LM's CPU and GPU on Gemma 4 E2B Q4_0, one file per run with the
conditions. The comparison is `scripts/prompt-speed.sh` in aether-infer.

**P0's check (section 0's table within 5% on a cool phone) did not run
today.** The phone was at 37–44 % after an hour and a half of runs, its
clock caps fell from 2.745 GHz while cool to 1.55–1.99 GHz under load, and
a 3-minute CPU run ended at 61–64 °C (zone0). The same build then gives
170–182 tok/s at 128 tokens, but 49–81 at 1024–2048 and 14–34 decode:
the runs measure the vendor's throttling, not the engine. The check is
pending a charged, rested phone.

The files:

- `baseline-2026-11-14.txt`: `scripts/prompt-speed.sh` started at 18:01,
  stopped after the CPU run (the phone was not going to cool between the
  engines). Warm-up 164.93 tok/s at 128; the timed run
  95.59 / 90.16 / 91.92 / 63.91 and decode 22.91, clock caps 1.55/1.02 GHz.
- `cpu-2026-11-14-1830.txt`: the same build at full cool caps
  (2.400/2.438 GHz), started at 39.2 °C, ending at 61.1 °C:
  170.66 / 53.12 / 80.73 / 62.70 and decode 13.76. The 128-token test ran
  before the caps fell; the rest after.

Conditions common to both: iQOO 13 (SM8750), on battery (44 → 37 %), screen
on, cores 2–7, 6 threads, `aether` at `3a50c4f` built with
`--features vulkan,hexagon`, gguf `gemma-4-E2B-it-Q4_0.gguf`. The
reference table (`../prompt-speed-2026-11-14/`, 156 / 131 / 107 / 88,
decode 34.5) was taken at 57–65 % battery on a rested phone with an older
build; its 128-token number is below today's 170–182 at full caps, so the
builds are not the same either.

To finish P0: charge the phone, leave it idle until zone0 is below 35 °C
(and the CPU zones below 38 °C), then run

    scripts/prompt-speed.sh ../Aether/results/prompt-speed/<file>.txt

with the phone unplugged if the table is to be compared with the old one
(it was taken on battery).
