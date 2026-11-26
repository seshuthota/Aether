# M4: LFM3-VL's tower attention on HMX (phone, 2026-11-27)

aether-infer `276c79d` (branch `multimodal-devices`); plan:
`aether-infer/plans/multimodal-devices.md` M4.

Conditions: iQOO 13 over USB, **charging**; screen awake; zone0 39–58 °C
at the starts of the test runs, 40.7–44.2 °C in the CLI runs; CPU caps
2.40 / 2.44 GHz.

## The encoder test (`lfm3_vl_vision_dsp_matches_transformers`)

Each image: the CPU encoder, the DSP encoder traced, twice more untraced,
then the DSP encoder with the CPU's attention (`hmx_attention = false`).
Three runs (the first before the 768-patch threshold, so all four images
on HMX attention). DSP times untraced, ms:

| Image | Patches | HMX attention | CPU attention | RMS (HMX att.) | RMS (CPU att.) | CPU encoder RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1024 | 438, 452; 437, 441; 476, 437 | 552; 534; 545 | 3.28e-2 | 2.81e-2 | 4.15e-2 |
| 1 | 1024 | 537, 441; 446, 472; 438, 440 | 536; 536; 539 | 3.01e-2 | 2.72e-2 | 3.28e-2 |
| 2 | 512 | 197, 200 (run 1) | 198; 202; 203 | 3.62e-2 | 3.25e-2 | 8.31e-2 |
| 3 | 256 | 332, 123 (run 1) | 104; 105; 106 | 2.90e-2 | 2.55e-2 | 3.07e-2 |

Runs 2 and 3 put images 2 and 3 on the CPU's attention (threshold), so
their "HMX" columns there are the CPU path.

Inside the attention, per 1024-patch layer (instrumented build, before
the transposed O): gather 1.2–1.5 ms, quantize and repack K 3.0–3.8, S on
HMX 8.8–9.3, softmax 3.3–3.9 (NEON; 11–15 scalar), O = P V on HMX 17.4–18.4,
scatter 0.7. Oᵀ = Vᵀ Pᵀ with P as Q8_0 weights replaced the 18 ms O.

## Image chat (`phone-m4-vl.sh`, `phone-2026-11-27-attention-vl.log`)

LFM3-VL-2B Q4_K_M, Q8_0 mmproj, `scene.png` (1024 patches), 275-token
prompt, 15 tokens, greedy, two rounds alternated:

| Route | Prompt ms | Was (M4, 2026-11-26) |
| --- | --- | --- |
| CPU | 2448, 2311 | |
| DSP (HMX encoder) | 1744, 1648 | 1.76–1.79 s |
| CPU, GPU encoder | 1991, 1925 | |
| DSP, GPU encoder | 1523, 1519 | 1.53–1.56 s |

`AETHER_DSP_PROFILE`, three runs: "image encoded on the DSP in 774 /
605 / 607 ms (1024 patches)" (M4: 681–728). Same reply on every route.
