# T1: LFM3-VL-2B, images in chat (plans/next-six.md in aether-infer)

Branch `lfm3-vl`. Model `LiquidAI/LFM3-VL-2B` at `23387be` (HF) and
`LiquidAI/LFM3-VL-2B-GGUF` at `1add221`. References: transformers f32 on the
CPU with the checkpoint's remote code (`tools/hf_dump_lfm3_vl.py`), and
llama.cpp `d02b4c0d6` (llama-server, mtmd). PC: i7-13620H, 8 threads.

The four V0 images (drawn by the dump script, `testdata/lfm3-vl/images/`),
each with "Describe the image in one sentence." and reasoning off:

| image | size | grid | image tokens | prompt |
| --- | --- | --- | --- | --- |
| scene | 512×512 | 32×32 | 256 | 276 |
| chart | 512×512 | 32×32 | 256 | 276 |
| strip | 1024×128 | 8×64 | 128 | 148 |
| icon | 40×40 (to 256×256) | 16×16 | 64 | 84 |

## V1: pixels

`image::patches` with LFM3-VL's bounds gives HF's `pixel_values` exactly
(worst difference 0 on all four, both temporal copies).

## V3: the encoder against transformers

Relative to each tap's largest value, worst over the normed embedding, the
14 blocks and the projector rows:

| mmproj | scene | chart | strip | icon |
| --- | --- | --- | --- | --- |
| f32 (converted from HF, host only) | 2.2e-5 | 1.2e-5 | 2.3e-5 | 8.6e-6 |
| F16 (published) | 2.2e-5 | 1.2e-5 | 2.1e-5 | 7.0e-6 |

## V4: the LM with the image rows

| LM + mmproj | last logits (rel.) | 32 greedy tokens = HF |
| --- | --- | --- |
| bf16 (converted, widened to f32) + f32 | 1.9e-6 .. 2.9e-6 | 4 of 4 |
| Q8_0 (published) + F16 | 1.9e-2 .. 3.9e-2 | 3 of 4: the strip leaves HF at step 17 ("with" → "by"), our top-two gap 0.31 |

(HF's dump used `min_new_tokens=32`, so the end token is masked for the
32 steps in both.)

## V5: surfaces

- API (`Session::chat`, bf16 + f32): each reply a prefix of HF's greedy;
  prompt lengths equal; a second turn reuses > 256 tokens and answers
  "yellow" for the sun; a changed picture restarts.
- C ABI (`aether_chat_vision`, Q8_0 + F16): the icon's answer is HF's.
- `aether serve` + the `openai` package 3.49.0 (`tools/serve_vision_check.py`,
  Q8_0 + F16): prompt_tokens 276/276/148/84; 3 of 4 answers HF's (the strip
  as in V4); streamed = plain; turn 2 cached 307 of 326 tokens, "The sun is
  yellow."; a remote URL is a 400.
- `aether run --image scene.png` (Q8_0 + F16): HF's text.

## V6: llama.cpp as the second reference

llama-server `d02b4c0d6`, `--jinja`, `enable_thinking: false`, greedy, 32
tokens (`tools/llamacpp_lfm3_vl.py`; outputs in `v6-*.json`). Prompt tokens
276/276/148/84 for both pairs, as HF and Aether.

| files | scene | chart | strip | icon |
| --- | --- | --- | --- | --- |
| Q8_0 + F16 | leaves HF at step 24 | leaves HF at step 20 | leaves HF at step 17, **the same text as Aether's** | HF's |
| bf16 + f32 | HF's | leaves HF at step 20 | HF's | HF's |

Every place llama.cpp leaves HF is a near-tie in HF's own f32 logits
(`tools/hf_gap_lfm3_vl.py`): scene step 24 " mountains" over " and" by
0.10; chart step 20 " the" over " " by 0.31; strip step 17 " with" over
" by" by 0.28. Aether at bf16 + f32 gives HF's text on all four; with the
published Q8_0 + F16 it leaves HF only at the strip's tie, where llama.cpp
leaves it the same way.

## PC speed (not the target; for scale)

`aether run` Q8_0 + F16, scene: prefill 276 tokens (image encoding
included) 19.0 s, decode 12.3 tok/s; the same model on 241 text tokens
prefills in 15.6 s, so the encoder is a small part. The phone numbers are
V7.
