# P1 review fixes (2026-11-15)

Follow-up to `p1b-phone.md`: the conv layers' int8 products, the decode
workgroup count, and the NaN hole in the test helper. `p2` below is
`31ccf58` (before these fixes); `p3` the fixed tree.

## Decode workgroups (iQOO 13, Adreno 830, Gemma 4 E2B Q4_0)

`q4_0_gemv_t` is 64 threads (64 rows) a workgroup, but the decode step
dispatched `groups_for(rows, 4 · gemv_r(rows))`, 4× too many at the gate/up
and down shapes (and 2× at 12288 rows). Now `rows.div_ceil(64)`.

| run | build | start °C → end °C | caps at start | tg64 tok/s |
| --- | --- | --- | --- | --- |
| first A/B | p3 | 36.1 → 50.7 | 2.75/2.65 GHz | 18.10 |
| first A/B | p2 | 41.5 → 50.7 | 2.40/2.44 GHz | 17.74 |
| first A/B, matched | p2 | 43.0 → 50.7 | 2.00/2.25 GHz | 17.61 |
| first A/B, matched | p3 | 42.3 → 54.9 | 2.00/2.25 GHz | **20.16** |
| second A/B, cool | p2 | 38.0 → 52.3 | 2.40/2.44 GHz | 20.66 |

The only adjacent pair at the same stated caps is 2.00/2.25 GHz:
**17.61 → 20.16 tok/s (+14.5 %)**. The old build also reaches 20.66 at
2.4 GHz when the phone starts at 38 °C, so the device's thermal state moves
decode by ±15 %: the earlier −3.6 % was that noise, not the grouping.
A cool p3 run at 2.4/2.44 GHz was not obtained — the phone's wireless adb
dropped after the p2 run above (`adb connect` refused on 44747 and 5555;
the phone still answers ping). Re-run when it is back:

    ./aether-p3 bench -m gemma4/gemma-4-E2B-it-Q4_0.gguf -dev gpu --cores 2-7 -p 512 -n 64 -r 3

## LFM3-1.7B-QAD-Q4_0 prompt against llama.cpp's BF16 base

`vk_gemma testdata/gguf/LFM3-1.7B-QAD-Q4_0.gguf hostdata/lfm3/kl-base-bf16.bin 20 batch`
on the laptop's Intel RPL-P GPU (the phone's adb was gone), batch 128,
5100 tokens scored, 578 s. Base: llama.cpp (build 0.28.0)
`--save-all-logits` on `hostdata/gguf/LFM3-1.7B-BF16.gguf`, wikitext-2 test,
ctx 512, 20 chunks. llama.cpp's own KL on the same file and base:

| run | PPL | Mean KLD vs BF16 | Same top p |
| --- | --- | --- | --- |
| llama.cpp CPU QAD-Q4_0 | 27.1383 ± 1.24457 | 0.183190 ± 0.004556 | 79.922 % |
| aether CPU QAD-Q4_0, PC (`phase-lfm3.md`) | 27.1557 ± 1.24588 | — | — |
| aether CPU QAD-Q4_0, phone (`phase-lfm3.md`) | 27.1489 ± 1.24542 | — | — |
| aether GPU prompt QAD-Q4_0 (int8, this tree) | **27.1311 ± 1.24489** | **0.180309** | **80.157 %** |

The base's PPL as our reader scores it is 23.4666 (llama.cpp prints
23.4796); it is the same file in both KL columns. Ours is marginally closer
to BF16 than llama.cpp's, and the 0.02 PPL gap to the CPU path is the same
quantization noise that the CPU/llama.cpp pair shows. Raw:
`/tmp/lfm3-qad-kl.log`, `p1c-lfm3-gpu-2026-11-15.txt`.

The new test `lfm3_q4_0_prompt_matches_the_cpu` compares the GPU prompt with
the CPU engine position by position: with the conv layers on the wrong
layout (the bug this fixes) position 0 is off by 0.93; with the int8 conv
products it is ≤ 4.3e-2 (Q4_0 activations vs the CPU's f32 kernels), bound
5e-2.

## Tests

- All 15 `kernels::vulkan::tests` pass with the NaN-strict `max_errors` and
  the fixed `q4_0_gemv_t` groups (`rows.div_ceil(64)`).
- Host `kernels::` (x86/scalar): 16 pass.
- `model::gpu::gemma` (4 tests, incl. the new LFM3 one), `c_abi_gpu_gemma4`,
  `fmt`, `clippy` (host + Android): pass.
- PC Gemma 4 Q4_0 decode PPL after the grouping fix: 24.4536, unchanged.
- The full `ci-local.sh` was stopped by request after 30 min in
  `model::gguf::tests::gemma4_matches_transformers` (not one of the changed
  areas); `--quick` passes.
- **Pending:** `cargo test --release --target aarch64-linux-android
  -p aether-infer --lib kernels::` on the phone (NEON/i8mm), blocked on the
  dropped adb connection.
