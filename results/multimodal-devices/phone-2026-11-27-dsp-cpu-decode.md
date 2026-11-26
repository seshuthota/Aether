# M4: the DSP's prompt with the CPU's decode (phone, 2026-11-27)

aether-infer branch `multimodal-devices` (commit after `757756d`); plan:
`aether-infer/plans/multimodal-devices.md` "M4: the DSP's prompt, the
CPU's decode". CLI `aether run|speak ... -dev dsp --cpu-decode --cores 2-7`.

Conditions: iQOO 13 over USB, **charging**; screen awake; zone0
41.5–49.6 °C at the run starts (no cooling between runs); caps 2.40 /
2.44 GHz.

## LFM3 text (one run each, 8-token raw prompt, 48 tokens, greedy)

| Route | Prompt | Decode | Peak RSS |
| --- | --- | --- | --- |
| CPU | 175 ms | 50.8 tok/s | 1.29 GB |
| DSP | 353 ms | 27.3 tok/s | 1.17 GB |
| DSP + CPU decode | 342 ms | 48.0 tok/s | 2.36 GB |

## LFM3-VL image chat (`phone-2026-11-27-dsp-cpu-decode-vl.log`)

`scene.png` (1024 patches), 275-token prompt, 63 tokens, greedy; the
same reply on every route. Two rounds:

| Route | Prompt ms | Decode tok/s | Total s | Peak RSS |
| --- | --- | --- | --- | --- |
| CPU | 2671, 2709 | 49.2, 48.9 | 3.95, 4.00 | 1.53 GB |
| DSP | 1534, 1676 | 25.6, 25.4 | 4.00, 4.15 | 1.36 GB |
| DSP + CPU decode | 1670, 1556 | 48.7, 48.7 | 2.97, 2.85 | 2.53–2.59 GB |
| DSP + CPU decode, GPU encoder | 1567, 1529 | 48.2, 48.8 | 2.87, 2.82 | 2.84–2.85 GB |
| CPU, GPU encoder | 2265, 2214 | 49.1, 49.3 | 3.55, 3.49 | 1.81 GB |

## LFM3-Audio voice chat (`phone-2026-11-27-dsp-cpu-decode-speech.log`)

`speak --chat -s 5`, sampled (the card's settings), two rounds. Sampled
replies part between routes after the first words (different numerics).

| Question, route | Real time | Prompt | First audio |
| --- | --- | --- | --- |
| text (29 tokens), CPU | 1.60x, 1.66x | 291, 320 ms | 532, 470 ms |
| text, DSP | 0.96x, 0.98x | 414, 412 ms | 679, 678 ms |
| text, DSP + CPU decode | 1.66x, 1.64x | 408, 402 ms | 557, 552 ms |
| jfk clip + text (170 tokens), CPU | 1.71x, 1.66x | 1454, 1450 ms | 1599, 1604 ms |
| jfk clip + text, DSP | 1.16x, 1.04x | 894, 908 ms | 1151, 1168 ms |
| jfk clip + text, DSP + CPU decode | 1.86x, 1.65x | 719, 720 ms | 862, 871 ms |

## Gemma 4 E2B Q4_0 (one run each, 11-token raw prompt, greedy)

| Route | Prompt | Decode | Peak RSS |
| --- | --- | --- | --- |
| CPU | 91 ms | 34.9 tok/s (59 tokens) | 1.69 GB |
| DSP | 460 ms | 23.2 tok/s | 1.68 GB |
| DSP + CPU decode | 462 ms | 29.3 tok/s | 3.22 GB |

The DSP and DSP + CPU decode texts are the same; the CPU's parts from them
at the second sentence. The CPU decode here is below a CPU session's,
probably the CPU copy's weight pages faulting in during the first steps
(not measured further).

## Test

`dsp_prompt_cpu_decode_matches_the_cpu` passes: "DSP prompt -> CPU decode
through eval: 4.49e-2 of max over 8 steps".
