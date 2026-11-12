# Time to first token: `aether serve` against `llama-server` (PC)

Plan `next-six.md`, T6/W3; the method is `release-and-llamacpp-compat.md`
section 7. **PC only so far; the phone row is owed.**

- Host: i7-13620H, 16 threads, 23 GB, Linux x86_64. Other host work was
  running (an Android emulator, a Gradle and a Kotlin daemon) for both
  engines; the runs alternate, so the comparison is paired but the absolute
  numbers include that load.
- Both servers: `testdata/gguf/smollm2-360m-instruct-q8_0.gguf`
  (sha256 `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`),
  `-c 4096 -t 8 -np 1`, one after the other on the same machine.
- `bench/ttft.py`: `/v1/completions`, `stream: true`, greedy,
  `max_tokens 16`, `cache_prompt: false`, 1 warm-up + 5 measured runs per
  cell, the engine order alternating by run. The prompt is the first N
  tokens of `testdata/wikitext/wikitext-2-raw/wiki.test.raw`, built with
  the server's own tokenizer and checked to be N tokens on **both**
  servers (64, 256, 512, 2048).
- Time from the request to the first non-empty content chunk, milliseconds.

| tokens | aether median (min–max) | llama-server median (min–max) |
| ---: | ---: | ---: |
| 64 | 990.74 (976.41–1007.77) | 170.40 (156.93–358.02) |
| 256 | 4033.23 (3989.04–4072.81) | 655.64 (638.61–703.32) |
| 512 | 8416.38 (8275.62–8428.46) | 1322.96 (1300.12–1366.55) |
| 2048 | 36197.73 (35756.38–36401.28) | 6276.30 (6197.03–6376.65) |

Both servers report the same `prompt_tokens` in `usage` (64/256/512/2048),
so the prompts are equal. The 5-run spreads are tight on both (the three
llama outliers at 64 tokens are the first measured run).

**Why aether is 6× behind on the PC:** the host is x86, where Aether has
only the scalar kernels; llama.cpp has AVX2/AVX-512. This is plan section
2, item 1, not a server overhead: the same code on the phone's NEON path
gave 660–778 prompt tok/s against llama.cpp's 815–958 in P5. The phone row
will say what a user actually feels on the target.

Files: `ttft-pc.json` (conditions and every row), `ttft-pc.md` (the table
above), `ttft-pc-aether.log`, `ttft-pc-llama.log`.
