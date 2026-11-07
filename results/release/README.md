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
