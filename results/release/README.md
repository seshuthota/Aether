# Release checks on the phone (2026-11-06)

iQOO 13 (Snapdragon 8 Elite) over Wi-Fi adb, on battery (80% at the start,
76% at the end of the DSP runs). `aether-infer` at `b32a227` plus the
uncommitted `aether run` sampling flags and the DSP reuse test, built with
`vulkan,hexagon` into `target-phone/` (the host suite held `target/`).

## REVIEW 3: `dsp_ppl` (`raw/review3/`)

wikitext-2, 20 × 512, 6 DSP threads, DMA mask 0x1f. Every run equals the
recorded digits (`results/phase-l5.md`) and its greedy text.

| run | PPL now | recorded | zone0 at start |
| --- | --- | --- | --- |
| SmolLM2 Q4_0, decode | 27.7234 ± 1.17484 | 27.7234 | 38.4 °C |
| Qwen2.5-0.5B Q4_0, decode | 19.7275 ± 0.83552 | 19.7275 | 39.2 °C |
| SmolLM2 Q4_0, prefill batch 128 | 27.6933 ± 1.17452 | 27.6933 | 41.5 °C |
| SmolLM2 Q4_K_M, HMX batch 128 | 20.6872 ± 0.84729 | 20.6872 | 40.7 °C |
| Qwen2.5-0.5B Q4_K_M, HMX batch 128 | 16.7155 ± 0.68569 | 16.7155 | 40.3 °C |

## The app and the AAR

`android/app` built from the `android/aether` module, Qwen2.5-0.5B
Q4_K_M, "Why is the sky blue?", chat on, temperature 0.7:

| build | CPU 2-7 | GPU | DSP |
| --- | --- | --- | --- |
| `vulkan hexagon` | replied; prefill 608 tok/s, decode 122.5 | replied; 341 / 87.7 | replied; 405 / 90.2 |
| `vulkan` (the release AAR's features) | replied; 591 / 121.2 | replied; 538 / 76.9 | "Error: the dsp backend needs a build with the `hexagon` feature" |

The vulkan-only `aether-release.aar` was built; the app installed was
built from the same module, not from that `.aar` file.

## P4 on the phone

`cargo test --release --target aarch64-linux-android --features vulkan,hexagon -- reuse cross_threads`:
5 passed (`the_next_turn_reuses_the_cache`, `gpu_generation_and_reuse`,
`model_and_session_cross_threads`, and the new `dsp_reuse_across_threads`).
On the DSP the second chat turn reused 40 tokens, prefilled 14 from
position 40 on another thread, and gave a fresh session's text.

Gemma 4 E2B Q4_K_M with `aether run -dev dsp`: refused at load ("the DSP
path is a later plan"), no crash.

## P2: the matrix against llama.cpp (`compare-iqoo13-cpu/`)

**Partial: round 1 of 2 only** (stopped at 9 PM when the phone was taken
out). `table-round1.md` is one round, so not yet the method's result.
llama.cpp `d02b4c0d6` (CPU build, `/data/local/tmp/llama-android`),
`-t 6 --cpu-mask 0xfc`; aether `-t 6 -C 0xfc`;
`-p 64,256,512 -n 128 -d 0,512 -r 5`. Cool-down to < 45 °C on the CPU
zones (idle sits at 43-46 °C on battery; the PMIC zone reads 52 °C and is
left out). The app process was alive but idle in the background during
the run.

Round 1 in short (median tok/s, aether vs llama.cpp): Qwen Q4_0 decode
184 vs 187, prompt 660-778 vs 815-958 (llama.cpp faster); Qwen Q4_K_M
prompt 376-422 vs 157-161, decode 98 vs 93; LFM3 Q4_K_M within about
±25% either way; Gemma 4 Q4_K_M decode 32.6 vs 14.4, prompt 55-75 vs
30-40.

## Update 2026-11-07: the matrix finished

CPU round 2 ran 2026-11-06 23:19–00:29 IST (binaries unchanged, md5
checked); `compare-iqoo13-cpu/table.md` covers both rounds. The LFM3 cells
swung widely in round 2 and need a re-run. The Adreno matrix (Qwen Q4_0,
Q4_K_M; `llama-ocl -ngl 99 --mmap 0` vs `aether bench -dev gpu`) ran both
rounds, the last on 2026-11-07 (one cell cut off overnight was redone):
`compare-iqoo13-gpu/table.md`. Scripts: `round2.sh`, `gpu-resume.sh`.
Summary and caveats: `aether-infer/plans/HANDOFF-2026-11-07.md` section 5.

## Update 2026-11-07 (later): LFM3 re-run, a third Adreno round, the GPU prompt path

Same binaries as the matrix (aether md5 `e11a8e1e`, llama.cpp `d02b4c0d6` /
`d00685831`), same args, cool-down below 45 °C. Script:
`rerun-2026-11-07.sh`.

**LFM3-1.7B Q4_K_M on the CPU**, alone, two new rounds in opposite order
(`compare-iqoo13-cpu-lfm3/`). Both engines are steady now (aether decode
45.0–47.5 tok/s across every repetition, against 14.8–34.7 in the
disturbed round 2), and the picture changes: decode is level or slightly
ours (tg128 45.7 vs 41.6; at depth 512 37.1 vs 36.9), and llama.cpp leads
the prompt by 5–29 % (pp64 154.7 vs 120.1, pp512 139.1 vs 132.6; 4–10 % at
depth 512). `compare-iqoo13-cpu/table.md` now takes its LFM3 rows from
this run (the other models' rows are unchanged); the table with the
2026-11-06 LFM3 cells is kept as `table-2026-11-06.md`, and those JSON
files stay in `compare-iqoo13-cpu/`.

**Adreno, round 3** (llama.cpp first, as odd rounds), for the wide decode
ranges. Round 3 was the quiet one: decode ranges are tight on both
engines (aether Q4_0 91–94 tok/s, llama.cpp 62–64), while round 1 was low
for both. Median tg128 per round, depth 0:

| | round 1 | round 2 | round 3 |
| --- | --- | --- | --- |
| aether Q4_0 | 70.5 | 77.3 | 92.8 |
| llama.cpp Q4_0 | 27.1 | 59.4 | 63.5 |
| aether Q4_K_M | 55.2 | 70.6 | 73.3 |
| llama.cpp Q4_K_M | 10.6 | 10.2 | 2.5 (2–9) |

Aether is ahead in every round; over all three rounds tg128 is 91.0 vs
59.4 (Q4_0) and 72.1 vs 9.8 (Q4_K_M). `compare-iqoo13-gpu/table.md` covers
the three rounds; the two-round table is `table-rounds12.md`.

**Which prompt path `aether bench -dev gpu` measures**: the whole-GPU one.
`-dev gpu` is `Backend::Gpu` (only `--cpu-prompt` makes it
`GpuWithCpuPrompt`), a multi-token `Session::eval` runs
`GpuModel::forward_batch`, and every aether GPU row in the JSON says
`Vulkan`, `n_batch` 256, `n_gpu_layers` 99. The 267 tok/s that looked
inconsistent is step 7 of `results/phase-l3.md` (the f32 shader); the later
steps there (f16 and int8 products, tiled prompt attention, 256-token
passes) reach 1307 / 989 tok/s at pp512 in `vk_pp` (Q4_0 / Q4_K_M), the
bench's 1315 / 959. The GPU prompt rows need no relabel.

**P3: `bench/migrate-check.sh` on the phone** (`migrate-check-phone/`:
`run.sh`, the adb wrappers it passes as `LLAMA` / `AETHER`, `output.txt`).
aether at `main` `e5e0bbf` against llama.cpp `d02b4c0d6`, `-t 6`, no
f32 file on the phone, so no quantize step:

| | greedy text, 32 tokens | perplexity, 8 × 512 (llama.cpp / aether) | pp64 / tg16 (llama.cpp / aether) |
| --- | --- | --- | --- |
| Qwen Q4_0 | 1 of 4 equal, the rest part after 56–156 characters | 18.5542 / 18.5571 | 721 / 665, 183 / 178 |
| Qwen Q4_K_M | 2 of 4 equal, the rest part after 80–87 characters | 15.5057 / 15.5163 | 158 / 550, 95 / 118 |

Perplexity within 0.07 %. The greedy splits are near-ties, as on the PC
(5 of 8 equal there, parting after 56–142 characters): the two engines'
arithmetic differs in the last bits.

## Left for the next phone session (as of 2026-11-06; items 1-2 since done, see above)

1. **Matrix round 2** (opposite order) for the four CPU models, then
   `bench/compare.py` over both rounds into `table.md`.
2. **Adreno matrix**: Qwen Q4_0 and Q4_K_M, llama.cpp `llama-ocl`
   (`-ngl 99 --mmap 0`) against `aether bench -dev gpu`. Not started.
3. **Hexagon against llama.cpp**: `llama-hex` (`d00685831`) now aborts
   on this phone with `ggml-hexagon.cpp:156: HTP0 dspqueue_write failed:
   0x0000000c` (same command family ran on 2026-10-29); the first try
   hung in `fastrpc_dspsignal_wait` and was killed. Needs a look (a
   reboot, or a fresh llama.cpp Hexagon build) before a DSP row exists.
4. SmolLM2-360M (in the plan's "Next" models) has no GGUF here yet.
5. Gemma 4's full DSP path (C1, C2, C4–C7) is engine work, not a check.

The driver used: `bench/compare.sh` with `TEMP_CMD` reading the
`cpu*` thermal zones and `COOL_C=45`.

## Update 2026-11-07 (evening): round 4 on the Adreno, and the screen

**Adreno round 4** (aether first; aether at `main` `a309a74` pushed as
`aether-r4`, md5 `0e1ea1fb`; llama.cpp `llama-ocl` unchanged; script
`gpu-round4.sh`, table `compare-iqoo13-gpu/table-round4.md`, kept apart
from the three-round `table.md` because the aether binary differs). The
prompt rows repeat rounds 1–3 within a few per cent (Q4_0 pp512 1284,
Q4_K_M 966). Decode did not settle: Q4_0 tg128 95.7 (93–102), Q4_K_M
tg128 75.7, but Q4_K_M tg128 @ d512 43.6 (34–52) against 68.3 before, and
a re-run of that cell minutes later at 43 °C gave 38.8 ± 8.8 at depth 0.
llama.cpp's OpenCL Q4_K_M fell to 5.6 tok/s decode and 47–65 prompt.

**The cause: the screen.** `compare.sh` and the round scripts call
`adb shell` directly, without `scripts/adb-run.sh`'s keep-awake, so the
screen times out (30 s) early in a run. With it off the phone lowers its
CPU caps (on battery: cpu0 1.79 / cpu7 1.02 GHz, against 2.00 / 2.25 GHz
on; read from `scaling_max_freq`), and the GPU decode step is mostly CPU
submit and wait (`phase-lfm3-devices.md`'s B7 profile). Anything else that
wakes the screen mid-run (a person, another session's `adb-run.sh`) moves
the clocks again. Same cell, aether Qwen Q4_K_M on the GPU, alternating,
each from below 45 °C (`screen-check/out.txt`):

| pass | screen held on: tg128 / @ d512 | screen off: tg128 / @ d512 |
| --- | --- | --- |
| 1 | 88.1 ± 8.4 / 82.5 ± 3.8 | 79.0 ± 1.7 / 71.7 ± 1.0 |
| 2 | 80.3 ± 4.4 / 84.8 ± 11.1 | 51.1 ± 13.6 / 37.6 ± 5.2 |

So every matrix number so far (CPU and Adreno, both engines) was taken
with the screen state unheld, mostly off. Both engines ran under the same
conditions in each round, so the comparisons are not biased toward one
engine, but the absolute numbers are on the lowered caps for an unknown
part of each run, and the swings between rounds (LFM3 round 2, Adreno
round 1) fit the screen flipping. Fix (engine `6ec88a6`): `compare.sh`
takes `WAKE_CMD` (held during every run) and `STATE_CMD` (caps and
screen recorded before and after each run); `docs/benchmarks.md` asks for
the screen held on. The matrix needs re-running that way before its
numbers are published.
