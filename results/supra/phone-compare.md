# Supra2.5 on the phone: aether against llama.cpp

Plan: `aether-infer/plans/supra.md` (S5). Raw: `phone-compare/` (JSON per run,
`conditions.txt` with the clock caps, screen and battery before and after
each run), `phone-threads.log`.

## Conditions

- 2026-11-27, iQOO 13, wireless adb, **on battery** (AC powered: false,
  100 %), screen held awake (WAKE_CMD); clock caps cpu0 2.40 / cpu7 2.44
  GHz before and after every run.
- `bench/compare.sh`: two rounds in opposite engine order, each run after
  zone0 < 40 °C (runs ended at 46-52 °C); `-p 64,256 -n 128 -d 0,256 -r 3
  -t 6`, `-C 0xfc` (cores 2-7) for both.
- aether `100b525` (+ the uncommitted SupraChatTest only), llama-bench
  `d02b4c0d6` (NDK build, stripped). The fixed GGUFs (`tools/supra_gguf.py`).

## Six threads (tok/s, median, min-max over 6 repetitions)

| File | Test | llama.cpp | aether | ratio |
| --- | --- | ---: | ---: | ---: |
| Q4_0 (36 MiB) | pp64 | 4342 (4072-4540) | 4394 (4318-4432) | 1.01× |
| | pp256 | 3925 (3884-3958) | 4216 (4163-4226) | 1.07× |
| | tg128 | 984 (953-1004) | 970 (903-996) | 0.99× |
| | pp64 @ d256 | 1902 (1810-2043) | 3019 (2934-3120) | 1.59× |
| | pp256 @ d256 | 2821 (2772-2898) | 3014 (2921-3024) | 1.07× |
| | tg128 @ d256 | 426 (412-447) | 681 (674-687) | 1.60× |
| Q4_K_M = Q5_0 mix (52 MiB) | pp64 | 1052 (991-1114) | 4087 (4077-4096) | 3.89× |
| | pp256 | 1059 (1041-1093) | 3945 (3898-3966) | 3.72× |
| | tg128 | 614 (537-625) | 783 (752-790) | 1.28× |
| | pp64 @ d256 | 848 (794-895) | 2875 (2692-2890) | 3.39× |
| | pp256 @ d256 | 919 (906-926) | 2852 (2713-2875) | 3.10× |
| | tg128 @ d256 | 428 (390-433) | 575 (561-580) | 1.34× |
| Q8_0 (68 MiB) | pp64 | 4338 (4316-4358) | 4486 (4449-4493) | 1.03× |
| | pp256 | 4054 (4043-4194) | 4142 (3762-4178) | 1.02× |
| | tg128 | 633 (610-657) | 661 (656-679) | 1.04× |
| | pp64 @ d256 | 1938 (1795-2056) | 3051 (3015-3065) | 1.57× |
| | pp256 @ d256 | 2777 (2712-2929) | 2977 (2871-3034) | 1.07× |
| | tg128 @ d256 | 354 (338-365) | 493 (410-506) | 1.39× |

- Decode at depth 0: about 970-980 tok/s on Q4_0, 780 on the Q5_0 mix, 660
  on Q8_0; the two engines are even there except on Q5_0 (llama.cpp has no
  repacked Q5_0 path).
- At depth 256 aether is 1.3-1.6× faster in decode and short prompts (its
  attention); llama.cpp's Q5_0 prompt is 3-4× slower.

## Threads (decode, tg128 at depth 0, -r 3, `-C 0xfc`, cooled before each engine)

| Threads | aether Q4_0 | llama.cpp Q4_0 | aether Q8_0 | llama.cpp Q8_0 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 224 | 478 | 273 | 444 |
| 2 | 401 | 805 | 457 | 694 |
| 3 | 546 | 791 | 577 | 657 |
| 4 | 703 | 938 | 633 | 650 |
| 6 | 963 | 979 | 595 ± 114 | 656 |

- **aether's decode is half llama.cpp's on one or two threads.** At 6 both
  read the weights at DRAM speed and meet; below that our per-core GEMV
  is the limit. Q8_0 tops out at 4 threads for both.
- simpleperf, aether Q4_0, 1 thread on core 2, tg256 (209 tok/s, 4.8 ms
  a token): `gemv4_q4_0x4` 63 % (the layers' 32 MB at about 10.5 GB/s),
  `quant_rows` 20 % (the tied Q8_0 classifier read from the mapped table,
  11 MB), `attention_chunked` 9.5 %. llama.cpp reads the same 43 MB a token
  at about 20 GB/s on one core.
