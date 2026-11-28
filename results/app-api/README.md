# App API: the prompt read in chunks (aether-infer plans/app-api.md, A6)

`bench/examples/prompt_chunks.rs` (aether-infer `app-api`): one session
reads an N-token prompt from position 0 whole, then in chunks (`eval` per
chunk, the engine prefill the API's `chat_with_progress` runs), alternated
whole / chunked / chunked / whole. iQOO 13 (SM8750), cores 2-7, on
battery, wireless adb.

LFM3-1.7B Q4_K_M, 2048 tokens, 4 reps (started at 38 °C, ended at 60 °C
with caps 1555200 / 1689600 kHz; alternation keeps the pairs comparable):

    chunk 256: whole 135.6 124.4 121.9 122.2 | chunked 128.1 127.9 122.8 122.0 | -0.6 %
    chunk 512: whole 122.6 112.5 112.1 106.1 | chunked 122.6 112.0 111.6 109.1 | +0.5 %

Supra2.5-65M Q4_0, 500 tokens, 6 reps (started at 39.9 °C, caps 2400000):

    chunk 256: whole 3041.3 3131.7 3114.2 3109.4 3128.6 3112.5 | chunked 2857.3 2913.0 2929.5 2847.6 3029.6 2989.2 | -5.7 %

Reading: on a model whose prompt takes seconds the chunks cost nothing
measurable; on Supra the second pass costs about 9 ms of a 160 ms prompt.
