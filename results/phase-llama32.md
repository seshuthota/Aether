# Part D — the architecture table, Llama 3.2

`plans/architectures-and-audio.md`, part D, on `aether-infer` branch
`arch-table-llama` (from `fdb7b38`). Run 2026-11-09 and -10 on the PC
(13th Gen Intel Core i7-13620H, 16 threads, 23 GiB) and on an iQOO 13
(Snapdragon 8 Elite). References: `transformers` 5.22.1 (torch 2.15.0,
CPU) for the goldens, `llama.cpp` `d02b4c0d6` for perplexity, KL and
speed.

## D0 — `Arch` table, no behaviour change

`model/config.rs` gains `Arch` (a family's fixed facts: norm placement,
Q/K norms, QKV bias, RoPE variant and scaling, tied embeddings) and
`Family` names one `Arch`; `gguf.rs` reads the table once per file,
`hf.rs` takes the same table where it applies. The per-family booleans are
gone; `Config` gained only `rope_freqs` (empty when the file has none).

Checked by the model goldens of every family: `model::` in a debug build,
`--test-threads=1` (long: the Qwen3.5 and Gemma 4 goldens take hours in a
debug build), plus the release suite.

## D1 — goldens

`tools/hf_dump_llama3.py <checkpoint> testdata/llama3.2-1b [f16]` runs
transformers' `llama` on the CPU and saves, in one `ref.safetensors`:
every layer's residual at the 22 positions of a short prompt as a matrix,
all 128 256 logits of those positions, 16 greedy tokens after them, the
tokenizer corpus (433 strings, `encoded.txt` / `decoded.txt`), and — with
`f16`, matrices rounded as torch rounds them — the same for the f16 GGUF.
It also builds an 8448-token prompt from the wikitext corpus and saves the
logits at positions 0, 1, 8000, 8191, 8192, 8193, 8400 and 8447: on both
sides of `original_max_position_embeddings` 8192.

## D2 — `llama3` RoPE frequency scaling

GGUF files carry the divisors in `rope_freqs` (`pairs` of `dim/2`, the
`1e30` sentinel of unrotated pairs read as 1); HF `config.json` carries
`rope_scaling: {rope_type: llama3, factor: 32}` in its `rope_theta`
form. Both feed `Config::rope_freqs`, applied per pair. Logits:

| Check | Worst relative error | Bound |
| --- | --- | --- |
| HF f32 checkpoint, 22 positions, every layer, 128k logits | < 1e-3 | 1e-3 |
| f16 GGUF (our `convert_hf_to_gguf.py --outtype f16`) vs torch-f16 | < 1e-3 | 1e-3 |
| Q8_0 GGUF (unsloth) vs the f32 reference, same greedy tokens | 3.5e-2 | 5e-2 |
| 8448-token prompt, positions 0 … 8447 (both regimes), release + 16 threads | < 1e-3 | 1e-3 |

The Q8 number is the 8-bit weights, not the scaling: an f32 or f16
reference error would be at 1e-3 with a wrong scaling table.

## D3 — tokenizer and chat template

`llama-bpe`: the regex pre-tokenizer (`GPT2` with contractions and
`\p{N}{1,3}`), byte-level, then merges over single-byte pieces. The 433
string corpus matches HF `tokenizers`, encode and decode; `add_special`
puts BOS (`128000`) first, and `chunk_bos` gives `llama-perplexity`'s
BOS-at-every-chunk rule. Chat: the template's "Cutting Knowledge Date"
header with its fallback date (26 Jul 2024), one turn per message,
`<|eot_id|>` and an open assistant turn — checked against HF
`apply_chat_template`. End to end, `aether run -cnv` on the 1B answers
coherently.

## D4 — perplexity and KL against llama.cpp

Q8_0 GGUFs, `wiki.test.raw` (289 077 tokens), 512-token chunks, 8 threads
both engines, CPU.

| Model | Chunks | Aether PPL | llama.cpp PPL | mean KLD | same top p |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B-Instruct Q8_0 | 20 | **15.5982** ± 0.6362 | 15.6106 ± 0.6371 | **0.000683** | 98.63 % |
| Llama-3.2-3B-Instruct Q8_0 | 10 | **12.2059** ± 0.6740 | 12.1926 ± 0.6730 | **0.000385** | 99.02 % |

PPL within 0.11 % of llama.cpp; the KL is between the two engines running
the same Q8 weights, so it measures arithmetic, not quantization (both
engines' last-bit order). Both the 1B and the 3B load and run; the 3B
exercises 28 layers, 24 query heads over 8 K/V, width 128.

Our CPU speed for the record: 1B Q8 prefill 7.5 tok/s, decode 3.3 tok/s
(8 threads, scalar x86-64 path; llama.cpp's 18.6 s per 512-token pass).
The phone is the target and the numbers that matter.

## On the phone (iQOO 13, SM8750), 1B Q8_0

The same file, 5 chunks of 512 on `wiki.test.raw` (1275 tokens scored), the
second half of each chunk, as `llama-perplexity` does. Screen on, 6 threads
(cores 2–7, `-C 0xfc`) for the CPU runs.

| Backend | PPL (5 chunks) | Greedy text | Notes |
| --- | --- | --- | --- |
| Aether CPU (6 threads) | 12.4319 | coherent | prefill 187.6 tok/s, decode 47.3 tok/s (41-token prompt, `-cnv`) |
| Aether Adreno (Vulkan) | 12.4305 | coherent | 383 tok/s prefill, 39 tok/s decode; 6.6 s for the five chunks |
| Aether Hexagon (HMX) | 12.4016 | coherent | 9.7 tok/s through the DSP (263 s), 21.4 tok/s on 64 decode steps, classifier 7.0 ms/token of 45.1 |
| llama.cpp CPU (6 threads) | 12.4048 | — | the reference for the same chunks |

All four within 0.25 % of each other: the family's arithmetic (the 128k
vocabulary, tied output, the pre-tokenizer) is the same on the three
backends. The 8192-position regime is verified by the PC check only; the
GPU's and DSP's rotations come from the same `rotations_div`, but a run
past 8192 positions on the phone is owed.

Speed, `bench/compare.sh` (two rounds in opposite order, cooled below
42 °C — the phone's idle floor with the screen on, the PMIC zone excluded —
screen held on, 5 repetitions, `-t 6 -C 0xfc`), medians with min–max over
every repetition, `phase-llama32-phone/` has the JSON and conditions:

| Test | llama.cpp t/s | Aether t/s | Aether / llama.cpp |
| --- | ---: | ---: | ---: |
| pp64 | 353.3 (240.0–392.6) | 343.7 (283.2–350.5) | 0.97× |
| pp256 | 285.8 (229.0–342.6) | 328.2 (323.4–330.0) | 1.15× |
| pp512 | 309.3 (211.8–358.7) | 285.1 (214.2–291.1) | 0.92× |
| tg128 | 45.6 (42.3–46.8) | 44.6 (41.8–46.5) | 0.98× |
| pp64 @ d512 | 212.6 (198.8–226.1) | 233.2 (215.0–235.3) | 1.10× |
| pp256 @ d512 | 215.6 (198.1–240.4) | 222.6 (216.8–227.2) | 1.03× |
| pp512 @ d512 | 202.7 (162.0–213.2) | 209.5 (202.3–211.5) | 1.03× |
| tg128 @ d512 | 37.7 (32.4–38.0) | 42.4 (40.5–42.9) | 1.12× |

The phone throttles within a run (the caps fall from 2.4/2.44 GHz to
1.55/1.40 GHz as the SoC heats), which is the spread in the parentheses;
both engines ran under the same rule, in both orders.

## 3B Q8_0 on the phone

5 chunks of 512, same text and procedure; the CPU on 6 threads.

| Where | PPL | Greedy text |
| --- | ---: | --- |
| Aether, CPU | 9.3339 | — |
| Aether, Hexagon (HMX) | 9.3462 | coherent |
| llama.cpp, CPU (reference) | 9.3437 | — |
| Aether, PC (x86, reference) | 9.3471 | — |
| Adreno | — | **not supported**: `GpuModel` asserts `head_size <= 64`, the 3B has 128-wide heads; the attention shaders need a wider layout (work, not a run) |

The DSP carries the 3B: 8.7 tok/s decode, 3.7 tok/s through the five
chunks (684 s), 3.8 tok/s prompt at 8 tokens. Its perplexity lands on the
CPU's to 0.03 %, as the 1B's did.

Speed, the same method (`phase-llama32-phone-3b/`):

| Test | llama.cpp t/s | Aether t/s | Aether / llama.cpp |
| --- | ---: | ---: | ---: |
| pp64 | 131.5 (131.0–132.3) | 130.7 (129.9–131.5) | 0.99× |
| pp256 | 112.3 (105.5–123.2) | 108.3 (85.0–113.2) | 0.96× |
| pp512 | 101.2 (99.1–102.7) | 88.2 (86.7–94.1) | 0.87× |
| tg128 | 17.7 (17.2–18.0) | 17.8 (17.2–18.0) | 1.00× |
| pp64 @ d512 | 68.1 (61.1–81.7) | 65.1 (62.4–70.9) | 0.96× |
| pp256 @ d512 | 70.5 (65.9–76.3) | 59.8 (56.0–60.7) | 0.85× |
| pp512 @ d512 | 69.8 (65.3–73.0) | 57.2 (56.3–57.9) | 0.82× |
| tg128 @ d512 | 14.5 (13.5–14.7) | 15.8 (15.1–15.9) | 1.09× |

Prefill at 512 tokens is where llama.cpp pulls ahead on this model (13–18 %);
decode is level (1.00×) or ahead at depth 512 (1.09×).

## Owed

- **Positions past 8192 on the phone**: the CPU, GPU and DSP runs above are
  512-token chunks, so the 8192-point regime has only the PC check; the
  three backends' rotations come from the same `rotations_div`.
- **The 3B on the Adreno**: blocked by the shader (`head_size <= 64`), not
  by a run: the attention shaders need a layout for 128-wide heads first.
- **The phone suite** (`cargo test --release --target aarch64-linux-android`)
  with `AETHER_REQUIRED_GOLDENS` naming the llama3.2 goldens: the model
  goldens have run on the PC only.
- Qwen3 dense (D5) and the Qwen3.5 chat-model check (D6) are untouched.
