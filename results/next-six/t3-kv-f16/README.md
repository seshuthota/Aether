# T3: the f16 KV cache on the CPU (plans/next-six.md, plans/kv-cache.md in aether-infer)

Branch `kv-f16`. PC: i7-13620H, 6 threads (`-t 6`), plugged in. llama.cpp
`d02b4c0d6`. The phone steps (C0, C4: speed and the default) wait for the
phone; everything here is the PC.

## C1, C2: kernels and the cache

- f32 → f16 store against numpy `astype(float16)`, f16 → f32 load: bit for
  bit on 14 lengths plus a fixed case (ties both ways, subnormals, overflow,
  -0).
- Every attention kernel with f16 K/V against numpy on the f16-rounded
  inputs, within the f32 kernels' bounds; bit-equal to the f32 kernel on
  the widened rows.
- Whole models, batched prompt against one token at a time, each cache
  format: Gemma 3 270M Q8_0, LFM3 1.7B Q4_K_M, Qwen2.5 0.5B Q4_0 over 600
  positions, bit for bit with f32 and with f16. stories15M (f32 weights; its
  batch matmul sums in another order) 2.8e-6 with f32, 2.1e-4 with f16 (the
  two paths' keys round to neighbouring halves).
- Full host suite on the branch: all pass (225 + 14 + 23 + 3).

## C3: perplexity and KL, wikitext-2, 20 chunks × 512

KL against the f32/bf16 bases (`hostdata/imatrix/qwen2.5-0.5b-f32.kld`,
`hostdata/lfm3/kl-base-bf16.bin`). Script `c3/run.sh`, raw output `c3/`.

| model | KV cache | llama.cpp PPL | KLD | same top | Aether PPL | KLD | same top |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 0.5B Q4_0 | f32 | 19.7283 | 0.181010 | 76.98% | 19.7254 | 0.181293 | 77.24% |
| | f16 | 19.7283 | 0.181010 | 76.98% | 19.7366 | 0.181282 | 77.16% |
| | q8_0 | 20.0143 | 0.197042 | 76.45% | – | | |
| Qwen2.5 0.5B Q4_K_M | f32 | 16.7508 | 0.032762 | 90.29% | 16.7394 | 0.033116 | 90.18% |
| | f16 | 16.7508 | 0.032762 | 90.29% | 16.7414 | 0.032944 | 89.73% |
| | q8_0 | 16.9846 | 0.045390 | 88.65% | – | | |
| LFM3 1.7B Q4_K_M | f32 | 24.8278 | 0.117968 | 83.39% | 24.8663 | 0.118724 | 83.28% |
| | f16 | 24.8278 | 0.117968 | 83.39% | 24.8431 | 0.119875 | 82.94% |
| | q8_0 | 24.9045 | 0.116862 | 83.49% | – | | |

- Aether f16 against Aether f32: PPL +0.011, +0.002, −0.023; KLD within
  ±0.0012. The plan's bound was movement under 0.1.
- Aether f16 against llama.cpp f16: PPL within 0.016 on all three; KLD
  within 0.0019.
- llama.cpp's `-ctk f32` does store f32 (CPU KV buffer 48 MiB against
  24 MiB for f16, Qwen Q4_0, one chunk, `-lv 4`), yet its results equal its
  f16 run to the last digit: its CPU flash attention (`-fa on`) apparently
  computes at f16 precision either way. Aether's f32 row is a true f32
  cache.
- llama.cpp's q8_0 cache costs more on Qwen (PPL +0.29 / +0.23, KLD +0.016
  / +0.013) than on LFM3 (+0.08, KLD −0.001). That is the bar for C5.
- Time for the 20 chunks (same PC, 6 threads): Aether 324 s (Qwen Q4_0),
  818 s (Qwen Q4_K_M), 2713 s (LFM3) with f32, the same ±2% with f16;
  llama.cpp 20–65 s. Not looked into here (the PC is not the target); noted
  for a later look at Aether's batched prompt path on x86.

## C3: Qwen3-TTS with f16 caches (talker and code predictor)

`f16_kv_drift_against_qwen_tts` (raw: `tts/drift-f16.log`), the four golden
cases, teacher-forced on the reference's codes, greedy agreement:

| talker + predictor | KV cache | first codes | predictor codes | free-running greedy, frames equal before the first difference |
| --- | --- | --- | --- | --- |
| f32 (checkpoint) | f32 | 173/173 (identical codes, `generation_matches_qwen_tts`) | identical | all |
| f32 (checkpoint) | f16 | 166/173 (96%) | 2585/2595 (99.6%) | 28, 6, 14, 6 of 50, 25, 49, 49 |
| Q4_K_M at load (no imatrix) | f32 | 147/173 | 1418/2595 | 0, 0, 0, 1 |
| Q4_K_M at load (no imatrix) | f16 | 148/173 | 1420/2595 | 0, 0, 0, 0 |

f16 caches on the f32 model move the talker's logits by at most 5.7e-4
(relative) and free-running greedy leaves the reference after 6–28 frames,
at near ties (a code sequence is chaotic once one code differs). On Q4_K_M,
what the phone runs, the f16 cache changes nothing measurable: the
quantized weights already leave the reference at the first frame with
either cache.

Intelligibility, `tools/tts_quality.py` (24 clips, Whisper small; scripts
`tts/gen-f16.sh`, scores `tts/scores-f16.json`), the f32-cache rows from
docs/text-to-speech.md:

| Audio from | English letter errors | Chinese character errors |
| --- | --- | --- |
| The shipped Q4_K_M GGUF, f32 cache | 0.18% | 0% |
| The shipped Q4_K_M GGUF, **f16 cache** | 0.18% | 1.1% (1 character: 发 heard as 罚, zh1 seed 3) |
| f32 checkpoint, f32 cache | 0.83% | 0% |
| f32 checkpoint, **f16 cache** | 0.18% | 0% |

Sampled speech differs clip by clip whenever any logit moves, so single
errors either way are noise at 24 clips; nothing here separates f16 from
f32.

Speed on the PC (`aether speak`, shipped GGUF, 6 threads, same sentence,
alternated f32/f16/f32/f16): 0.25/0.24/0.24/0.24× real time. The PC carried
a load average near 7 from other processes during this session (the
drift test's Q4_K_M f16 timings, 5× slower than its f32 run, are that load,
not the cache); speed is C4's, on the phone.
