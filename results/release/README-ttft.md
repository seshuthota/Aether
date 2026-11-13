# Time to first token: `aether serve` against `llama-server`

Plan `next-six.md`, T6/W3; the method is `release-and-llamacpp-compat.md`
section 7. PC and phone rows below.

- Both servers: `testdata/gguf/smollm2-360m-instruct-q8_0.gguf`
  (sha256 `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`),
  `-c 4096 -t 8 -np 1` on the PC, `-c 4096 -t 6 -np 1` on the phone.
- `bench/ttft.py`: `/v1/completions`, `stream: true`, greedy,
  `max_tokens 16`, `cache_prompt: false`, 1 warm-up + 5 measured runs per
  cell, the engine order alternating by run. The prompt is the first N
  tokens of `testdata/wikitext/wikitext-2-raw/wiki.test.raw`, built with
  the server's own tokenizer and checked to be N tokens on **both**
  servers (64, 256, 512, 2048). Both report the same `prompt_tokens`.
- Time from the request to the first non-empty content chunk, ms.

## PC

i7-13620H, 16 threads, 23 GB, Linux x86_64. Other host work (an Android
emulator, a Gradle and a Kotlin daemon) was running for both engines; the
runs alternate, so the comparison is paired but the absolute numbers
include that load.

| tokens | aether median (min–max) | llama-server median (min–max) |
| ---: | ---: | ---: |
| 64 | 990.74 (976.41–1007.77) | 170.40 (156.93–358.02) |
| 256 | 4033.23 (3989.04–4072.81) | 655.64 (638.61–703.32) |
| 512 | 8416.38 (8275.62–8428.46) | 1322.96 (1300.12–1366.55) |
| 2048 | 36197.73 (35756.38–36401.28) | 6276.30 (6197.03–6376.65) |

Aether's 6× gap is the x86 scalar path (plan section 2, item 1), not the
server: on the phone's NEON path P5 measured 660–778 prompt tok/s against
llama.cpp's 815–958.

Files: `ttft-pc.json`, `ttft-pc.md`, `ttft-pc-aether.log`,
`ttft-pc-llama.log`.

## Phone

iQOO 13 (SM8750), Android, both servers on the phone and the client on the
PC over `adb forward`. Cores 2–7 (`0xfc`), 6 threads, `-np 1`, screen held
awake (`scaling_max_freq` read at both ends). Battery, not charger.

- Start: 41.5 °C (hottest `cpu-*` thermal zone), battery 59 %
  discharging, cpu0 2.400 GHz / cpu6 2.4384 GHz caps.
- End: 51.1 °C, battery 59 %, caps **1.7856 / 1.9584 GHz** — the phone
  throttled during the run.
- Peak sampled zone: 87.2 °C; a separate per-zone probe during a pp512 run
  put the hotspot on `cpu-1-1-1` (73.7 °C). The clock caps follow it down.
- Recorded run: `ttft-phone.*`. The first two attempts are kept as
  `ttft-phone-run{1,2}.*`: llama's worker threads inherited an all-core
  mask (0xff) there, so they are **not** the comparison. In run 3 every
  server thread is in `0xfc` (`taskset -p` on every tid checked before
  the run).

| tokens | aether median (min–max) | llama-server median (min–max) |
| ---: | ---: | ---: |
| 64 | 90.13 (67.20–158.66) | 86.37 (80.50–97.40) |
| 256 | 289.15 (269.40–342.32) | 330.40 (303.98–425.36) |
| 512 | 719.08 (677.66–757.59) | 718.24 (634.57–792.43) |
| 2048 | 5884.09 (5361.23–6247.77) | 6603.13 (5731.91–6762.49) |

Aether is level at 512, 4 % behind at 64, and ahead at 256 (12 %) and 2048
(11 %). Every cell alternated the engines, so the paired comparison holds
while the phone heats; the absolute numbers for the later cells are on the
lowered caps. The 64 and 256 cells are the least throttled; 512/2048
include the throttle.

Files: `ttft-phone.json`, `ttft-phone.md`, `ttft-phone-temps.txt`,
`ttft-phone-start.txt`, `ttft-phone-end.txt`, `ttft-phone-aether.log`,
`ttft-phone-llama.log`. Run 1/2 raw: `ttft-phone-run1*.json`,
`ttft-phone-run2.json`.
