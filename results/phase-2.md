# Phase 2 — scalar end-to-end on stories15M

Commit: the commit that adds this file. Raw output: `results/phase-2/raw/`.

## Correctness (reference: llama2.c `350e04f`, patched with `tools/llama2c-dump.patch`)

Goldens: `tools/stories15m_goldens.sh` (prompt "Once upon a time", 64 positions).

| Check | Host (x86-64, glibc) | Phone (SM8750, bionic) |
| --- | --- | --- |
| 14 activation kinds × 6 layers × 64 positions (embed, attn_norm, q, k, v, attn, attn_out, resid_attn, ffn_norm, swiglu, ffn_out, resid_ffn, final_norm, logits) | max abs error **0** (bit-identical) | max abs error **0** (bit-identical) |
| Greedy tokens (`-t 0`), 64 positions | identical | identical |
| Seeded sampling (`-t 0.8 -p 0.9 -s 42`), 64 positions | identical | identical |
| Tokenizer, 323 strings (`tools/gen_tokenizer_corpus.py`: whitespace, accents, CJK, emoji, control bytes, random mixes) | identical ids | identical ids |
| Generated text of the CLI vs llama2.c `run` (greedy and seeded) | byte-identical | — |

Bit-identical (not just within tolerance) because every op follows llama2.c's
order of floating-point operations and neither compiler contracts `a*b+c` into
an FMA. That stops being true in Phase 3: NEON kernels sum in a different order,
so from then on the comparison is by tolerance.

## Baseline speed (phone)

Conditions: 2026-10-21, on battery (USB not powered, status discharging, level
89–90%), screen on, `walt` governor. **Clock caps on battery were far below the
hardware maximum**: under load the prime cores were capped at 2.44 GHz (max 4.32)
and the performance cores at 2.40 GHz (max 3.53); with the screen off the caps
drop further (1.02 / 1.79 GHz). Battery saver off. The counted 2.38 GHz average
below confirms the cap was in force.

`aether stories15M.bin -t 0 -n 256 -i "Once upon a time" -c CORE` (222 positions
before BOS ends the story), three runs each:

| Core | Aether (scalar, 1 thread) | llama2.c `run`, same phone, NDK clang -O3 |
| --- | --- | --- |
| 7 (prime) | **88.3 / 88.8 / 88.6 tok/s** | 84.9 / 84.7 / 84.6 tok/s |
| 0 (performance) | **58.6 / 58.4 / 58.3 tok/s** | 58.1 / 58.1 / 57.4 tok/s |

Run-to-run spread is under 1%.

## Where the time goes (simpleperf, core 7)

`simpleperf record -e cpu-cycles:u` (kernel events are not available to the
shell user; `cpu-clock` is not supported on this device):

| Symbol | Share of cycles |
| --- | --- |
| `kernels::scalar::matmul` | 97.6% |
| `kernels::scalar::attention_head` | 1.1% |
| sampler, expf, powf, rmsnorm, allocator, … | < 0.4% each |

`simpleperf stat` over the same run: 6.10 G cycles (2.38 GHz average), 9.98 G
instructions (IPC 1.64), 2.34 G L1D loads with only 5.4 M L1D misses.

## Explanation (calculated from the numbers above)

- Multiply-adds per token: 6 layers × (4·288² + 3·288·768) + 32000·288
  (classifier, tied to the embedding) = 15.19 M. Each weight is used once, so
  a token reads 15.19 M × 4 B = **60.8 MB** of weights.
- At 88.6 tok/s that is 60.8 MB × 88.6 = **5.4 GB/s**, far below what memory
  can deliver (Phase 1 sweep). So the scalar matmul is *not* bandwidth-bound.
- Cycles per multiply-add: 6.10 G × 0.976 / (222 × 15.19 M) ≈ **1.77**. The loop
  `acc += w[j] * x[j]` has one accumulator: every add waits for the previous
  add's result. That is the same dependency chain as the Phase 0 sum, and the
  same fix applies in Phase 3 — several independent accumulators, 4 lanes each.
- Few L1 misses despite streaming 60 MB per token: the hardware prefetcher
  fetches the next lines long before the slow, latency-bound loop asks for them.
- The prime core is 1.52× faster than a performance core although their caps
  were nearly equal (2.44 vs 2.40 GHz). Not yet explained; Phase 1's per-core
  sweeps and a `simpleperf stat` on core 0 are the next things to look at.
