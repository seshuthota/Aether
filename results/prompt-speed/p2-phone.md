# P2 on the phone: flash attention, batch 256 (2026-11-15)

`aether-infer` `db5f372` + `88e0d5e` (the multi-head workgroup). iQOO 13,
Gemma 4 E2B Q4_0, cores 2–7, screen on, on battery, 53–62 %, one process
per prompt batch test, `-r 2`. P1 = `8446fd4`.

## Prompt and decode, matched 2.40/2.44 GHz caps

Two runs each, alternating, cooled to < 40 °C before every run
(`p2-ab-2026-11-15.txt`). The P1 run at the start of the session was at
2.75/2.65 GHz and is listed for reference.

| build / caps | 128 | 512 | 1024 | 2048 | decode | Max RSS |
| --- | --- | --- | --- | --- | --- | --- |
| P1, 2.75/2.65 GHz | 113.95 | 114.69 | 113.99 | 92.42 | 20.11 | 3.33 GB |
| P1, 2.40/2.44 GHz | 105.26 | 104.04 | 101.36 | 87.98 | 18.73 | 3.31 GB |
| P2, 2.40/2.44 GHz | **165.02** | **115.57** | **117.38** | **111.14** | 18.59 | 3.35 GB |
| P2, 2.40/2.44 GHz | **168.06** | **118.77** | 113.78 | **112.66** | 19.86 | 3.35 GB |

Matched gains: 128 **+57 %**, 512 **+11–14 %**, 1024 **+12–16 %**, 2048
**+26–28 %**; decode is unchanged (18.6–19.9 against 18.7).

## Per stage, same caps (`profile_gemma_pp`, one pass)

P1 numbers are from `p1b-profile-2026-11-14.txt` (2.00/2.25 GHz); P2 from
`p2-ab-2026-11-15.txt` (2.40/2.44 GHz), so the P2 attention gain is if
anything understated.

| | 512 P1 | 512 P2 | 2048 P1 | 2048 P2 |
| --- | --- | --- | --- | --- |
| tok/s | 132.8 | **134.0** | 95.5 | **123.1** (140.0 at 2.75 GHz) |
| attention | 609 ms | **478 ms** | 6470 ms | **4696 ms** (4224 at 2.75) |
| gate/up | 766 | 711 | 3115 | 2879 |
| down | 418 | 396 | 1725 | 1635 |
| GPU stages | 2285 | 1989 | 13844 | 10934 |
| rest (CPU, submits, waits) | 1571 (41 %) | 1832 (48 %) | 7608 (35 %) | 5702 (34 %) |
| wall | 3.86 s | 3.82 s | 21.45 s | 16.64 s |

At 512 the CPU rest is now the largest part (1832 of 3822 ms) — P3's
target; at 2048 it is the products (gate/up 2879, down 1635) and the
attention (4696).

## What changed the attention number

- The first one-pass shader took one query head a workgroup:
  attention 669 / 7691 ms at 512 / 2048 (16 values a thread, 4/2 tokens) —
  slower than P1's per-token path, because each of Gemma 4's eight query
  heads read the same K/V.
- 32 values a thread (8/4 tokens) was worse still: 826 / 8756 ms; more
  registers per thread without sharing K/V across heads.
- 256 threads over pairs of (token, query head), all `n_heads / n_kv`
  query heads of a K/V head in one workgroup (8 for Gemma 4, 4 for LFM3),
  is the one that works: 478 / 4696 ms.

## Accuracy (host GPU, unchanged by the workgroup layout)

- Gemma 4 Q4_0, 1 chunk: PPL 24.6278, KLD 0.336379, same top 81.961 %
  (P1: 24.5408 / 0.333100; the +0.35 % is the f16 K/V staging).
- LFM3-QAD-Q4_0, 20 chunks: PPL 27.1242, KLD 0.181109, same top 79.902 %
  (P1: 27.1311 / 0.180309; llama.cpp: 27.1383 / 0.183190).
- 30 numpy goldens (head sizes 128/256/512, 16×4 and 8×1 GQA, 256-token
  tiles at depth 2000, windows), worst 3.9e-4 of max |v| (bound 1.95e-3);
  LFM3 prompt-vs-decode, Q4_0-vs-CPU and the 320-token transformers test.
