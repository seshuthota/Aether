# Plan: flash attention for the GPU prompt pass

Written 2026-11-02, after step 25 (results/phase-l3.md). Implementation by the user, review by
Claude after each step. Work on `gpu-pending` / `impl` in `../Aether-impl`, as before; merge
into `main` only after the phone checks in F4 pass.

## 1. Why

The prompt pass (`forward_batch`) computes attention in two dispatches per layer:

1. `attention_tile.comp`: workgroup (head h, chunk c, 32 tokens) scores the 64 cached positions
   of chunk c for 32 tokens and writes, per token, head and chunk, a partial result
   `(m, l, out[0..hs])` into the `parts` buffer.
2. `attention_merge.comp`: workgroup (h, token) reads all chunks of that token and head back and
   combines them: `out = Σ e^(m_c − M) out_c / Σ e^(m_c − M) l_c`.

This splitting (flash-decoding) is right for **decode**: one token has only `n_heads`
workgroups' worth of work unless the positions are split. For a **prompt** of 128 tokens there
is already plenty of parallel work in the tokens, and the split costs:

- **The `parts` traffic.** Per token, head and chunk: hs + 2 = 66 floats written, then read
  again. Qwen2.5-0.5B (14 heads), a 128-token batch at depth 4096: 128 × 14 × 64 × 66 × 4 B =
  30 MB written and read per layer, ~1.4 GB per batch over 24 layers. The keys and values
  themselves are about as much (an estimate with no caching, to be checked against F0's profile).
- **A second dispatch and barrier per layer** (24 per pass), plus the merge's own exp() per chunk.
- **The batch size shrinks with the context.** `parts` is sized for the whole cache:
  `batch = PARTS_BUDGET (32 MB) / (4 · n_heads · seq_len/64 · (hs + 2))`, capped at 128
  (`src/model/gpu.rs:460-462`):

  | Qwen2.5-0.5B, seq_len | chunks | parts per token | batch |
  | --- | --- | --- | --- |
  | 1024 | 16 | 59 KB | 128 |
  | 4096 | 64 | 237 KB | 128 |
  | 8192 | 128 | 473 KB | 70 |
  | 32768 (the model's own) | 512 | 1.9 MB | 17 |

  With 17-token batches, the GEMMs (64-token tiles) run a quarter full. That hurts the whole
  pass, not just attention.

Measured so far: the prompt speed falls about 20 % from 128 to 512 tokens (step 21, open in
IMPLEMENTATION.md). Step 22 found attention was 28 % of the GPU's prompt time at 512 tokens
before the tile shader.

**Flash attention** (Dao et al. 2022; FlashAttention-2, Dao 2023) keeps one running
`(m, l, acc)` per token and walks all positions in one pass. The online softmax is already in
`attention_tile.comp`. What changes is that a workgroup no longer stops at 64 positions: it
walks every position its tokens can see, divides by `l` at the end and writes the finished
output to `xb2`. The `parts` buffer and the merge disappear from the prompt path.

## 2. The math (for the notes)

For one token, query q, positions 0..p, scores `s_j = q·k_j / √hs`:

    out = Σ_j e^(s_j − M) v_j / Σ_j e^(s_j − M),   M = max_j s_j

M is not known until the end. Keep the running max m, the sum `l = Σ e^(s_j − m)` and the
weighted sum `acc = Σ e^(s_j − m) v_j`. When a new score s arrives with `m' = max(m, s)`:

    l   ← l · e^(m − m') + e^(s − m')
    acc ← acc · e^(m − m') + e^(s − m') v

Both quantities stay relative to the current max, so nothing overflows, and at the end
`out = acc / l`. The merge shader applies the same rule to whole chunks. Doing it in one pass
just means applying it to every position as it arrives.

FlashAttention-2's refinement (step F7a) handles a tile of positions at a time: first all the
tile's scores, then the tile's max, then **one** rescale of acc per tile instead of one per
position.

## 3. Scope

In:
- A new prompt attention shader, `shaders/attention_flash.comp`, used for `forward_batch`
  (n > 1, not the few-token pass).
- `parts` sized for decode and the few-token pass only, so the batch no longer depends on
  `seq_len`.
- Measuring at depth (prompt after 1K–8K cached tokens), which we cannot do today.
- A numpy reference for causal, grouped-query attention over a batch.

Out (each is its own later topic, listed in F8):
- Decode attention. It stays flash-decoding (split + merge); F8 only measures it at depth.
- The few-token pass (speculative verify, ≤ 8 tokens). It keeps `attention.comp`.
- CPU and DSP attention.
- An f16 KV cache. It would halve the key and value bytes, but touches the post/RoPE shader,
  the decode shaders and `load_kv`.
- Head sizes over 64. The current shaders all assert `hs <= 64`, and both models have 64.

## 4. Steps

Each step is one commit and has a check. Steps F0 and F1 add nothing new to the engine. They
give us a baseline and a test, so that F2 has something to be measured and checked against.

### F0. Baseline at depth (measure first)

What:
- `examples/vk_pp.rs`: add `depth D` (repeatable, e.g. `depth 0 depth 1024 depth 4096`),
  like llama-bench's `-d`. Fill the cache with D tokens through `forward_batch` (untimed),
  then time the pp N runs from position D. Load with `seq_len = max depth + max N`. Today the
  loader passes `1024`.
  - A timed run must restart at D each time. The cache positions ≥ D are simply overwritten,
    so nothing needs clearing.
- Print `m.batch` (already in the header) so the batch shrinking is visible at large depths.
- With `profile`, the attention and merge stage times per depth.

Runs (phone, battery, `scripts/opt-ab.sh` new stage `depth`, or by hand):
- Qwen2.5-0.5B Q4_0 and Q4_K_M, pp512 at depth 0 / 1024 / 2048 / 4096, plus one profiled run.
- External reference, same session: llama.cpp's OpenCL `llama-bench -p 512 -d 0,1024,2048,4096`
  (as in step 21).

Done when: `results/phase-l3.md` step 26 has tok/s and the attention + merge share per depth.
That table is what F5 is measured against.

### F1. A reference for causal grouped-query attention, and a GPU test with today's shaders

What:
- `tools/gen_kernels.py`: new cases `attention_causal_<nh>x<nkv>x<hs>_p<pos0>_t<nt>`:
  - q `[nt, nh, hs]`, k and v `[pos0 + nt, nkv, hs]`, out `[nt, nh, hs]`.
  - Token i is at position pos0 + i and sees positions 0..=pos0 + i.
  - Query head h reads key/value head `h / (nh / nkv)`.
  - Computed in float64 by numpy, as the existing `attention_<n>x<hs>` cases are.
  - Shapes:
    - (14, 2, 64), Qwen.
    - (9, 3, 64), SmolLM2.
    - (4, 4, 17), odd head size, no grouping.
  - pos0 ∈ {0, 1, 63, 64, 1000, 4000}, nt ∈ {1, 31, 32, 33, 128}. These cross the 16-position
    sub-tiles, the 32-token workgroups and the 64-position chunks.
  - Keep the number of large cases small; testdata/kernels is regenerated, not committed.
- `src/kernels/vulkan.rs` tests: a test that runs the attention shaders directly on these cases,
  the way `qx_gemm_matches_reference` runs its shader:
  - Upload q, k, v into buffers laid out as `forward_batch` lays them out:
    - q at `tok * qstride + h * hs`.
    - k and v at `layer_off + pos * kv_dim + kvh * hs`.
  - Dispatch, read the output and compare with numpy.
  - Run it first on `attention_tile.comp` + `attention_merge.comp`. That proves the harness on
    code that is already verified, before the new shader exists.
- The bound must be argued, not tuned:
  - Keys and values pass through f16 in shared memory, a relative rounding of 2⁻¹¹ each.
  - The output is a weighted average of values, so its error is about that of one value plus
    the scores' error carried through exp.
  - Start from about `4 · 2⁻¹¹` relative to max |v|, and write down why in the test's comment,
    as `attention_matches_numpy` does.

Check: the test passes on the host (Intel) and on the phone for today's shaders.

### F2. `shaders/attention_flash.comp`

Start from a copy of `attention_tile.comp`: same thread layout (4 threads per pair of tokens, 16
values each), same f16 sub-tiles of SUB = 16 positions in shared memory, same online softmax.
Change:

- **Workgroups:** `(h, g)`, the 32 tokens `32g .. 32g + 31` of the batch. No chunk dimension.
  Dispatch `[n_heads, ceil(nt / 32), 1]`.
- **Loop:** over positions `0 .. last`, where `last = pos + tok0 + nq − 1`, in sub-tiles of SUB.
  - All threads of the workgroup must run the same number of iterations, because of the
    barriers. The bound is per workgroup, so it is uniform.
  - Tokens that are not live keep `live[u] = false`. They must not `return` before the loop.
- **Mask:** `tj <= at[u]` only matters in the last sub-tiles, those within 32 positions of the
  end. Keeping the check everywhere is correct. Removing it from earlier tiles is F7b.
- **Output:** after the loop, thread (t, part) writes `acc[u][i] / l[u]` to
  `o[(tok * n_heads + h) * hs + d]`, the layout `attention_merge.comp` writes into `xb2`.
  - Push constants: `n_heads, n_kv, hs, layer_off, qstride, ntok`.
  - Bindings: q, kc, vc, o, params (`pos`).
- **Edge cases:**
  - `m = −∞` at start: `exp(m − m')` is `exp(−∞) = 0`, which is fine as long as m' is finite.
    Every token sees position 0, so the first sub-tile always gives a finite m. Note this in
    a comment.
  - The subgroup shuffles (xor 1, 2) need subgroups of at least 4 lanes and the four threads
    of a token in one subgroup. That already holds for attention_tile.comp. Keep its comment.
- The header comment says what changed from attention_tile.comp and why (§1), as the existing
  shaders do.

Check: F1's test, now also run on `attention_flash.comp`, passes on host and phone with the same
bound.

### F3. Wire it into `GpuModel`

- `src/model/gpu.rs`:
  - `Bound`: add `attn_flash` (bind q = `qkv`, `kc`, `vc`, o = `xb2`, `params`).
  - `record_pass`: for `n > 1 && !few` (and not an old-path option), dispatch `attn_flash` with
    `[n_heads, nt.div_ceil(32), 1]`, `end(r, "attention")`, and **skip** the merge dispatch.
    Decode and few-token passes are unchanged.
  - `Options`: add `attn_chunked: bool` for the step-22 path (attention_tile + merge), kept for
    A/B like `attn_per_token`. The default is flash.
  - `parts` and `batch`: when neither `attn_chunked` nor `attn_per_token` is set, `parts` only
    has to hold `FEW` tokens (decode and the few-token pass), and the batch is 128 whatever
    `seq_len` is. With either old option, keep today's sizing.
- `examples/vk_pp.rs`: flag `chunked` (with `attn1` as is). Print which attention ran in the
  header.
- `examples/vk_ppl.rs`: a `batch-chunked` mode, so KL can be compared on the same file.
- `src/lib.rs` is not to be changed.

Check: `cargo test` on the host. Then the full suite on the phone
(`--features vulkan,hexagon`), and all tests must pass as before, including
`c_abi_gpu_session`.

### F4. Verify against the references

1. **Kernel:** F1/F2's numpy test (done above).
2. **Model, short context:** `vk_ppl … 20 512 batch BASE.gguf` on the phone for SmolLM2 Q4_0,
   Qwen Q4_0, Qwen Q4_K_M. The PPL and KL must match step 22's `vk-kld-*-batch-tile.txt` to
   within run noise (the order of additions changes, nothing else).
3. **Model, long context:**
   - `vk_ppl` with CTX 2048 and 4096 (Qwen; SmolLM2 up to its context), flash and `chunked`.
   - Against llama.cpp: `llama-perplexity -c 2048 / 4096 --kl-divergence` with its base made
     from the same f32 GGUF on the host (`~/llama.cpp/build-cuda/bin/`).
   - Our KL and llama.cpp's should be the same size. Flash and chunked should agree with each
     other to within noise.
   - This is the first time the GPU prompt path is checked beyond 512 tokens.
4. **Greedy text:** the `vk_ppl` greedy line is the same as with `chunked` on the phone, or
   it differs at a near tie that is explained, as in step 24.

Done when all four pass. Raw output goes in `results/phase-l3/raw/` (`vk-kld-*-flash*.txt`,
`vk-tests-flash-phone.txt`).

### F5. Measure on battery

- `scripts/opt-ab.sh`: a stage `flash`: `vk_pp` flash against `chunked`, Q4_0 and Q4_K_M,
  pp512 at depth 0 / 1024 / 2048 / 4096, two rounds, the order reversed in round 2, as the
  other stages.
- One profiled run each, for attention + merge ms against F0.
- Write step 27 in `results/phase-l3.md`, with the conditions at the top of each raw file as
  always.

What to expect, so a surprise is noticed:
- At depth 0 and pp ≤ 128 there are few workgroups: 14 × 4 = 56 for Qwen at a 128-token
  batch. Flash may be slower there than chunked, which has `chunks` times more workgroups.
- At depth 4096 it should win: no parts traffic, no merge.
- If it loses at depth too, profile before changing anything.

### F6. If short prompts lose: more workgroups

Only if F5 shows it. Pick by measurement, one at a time:

a. **16 tokens per workgroup** (U = 1) at small batches: twice the workgroups, but each key and
   value read from shared memory serves one token instead of two. Step 22 measured one token
   per thread as slower (193 against 121 ms at 512 tokens, both f32), but that was with chunks
   giving plenty of workgroups.
b. **Split only when short of workgroups:** a push constant `span` (positions per workgroup).
   With `span ≥ last + 1`, write `o` directly. With a smaller span, write `parts` and merge
   (that is today's path with longer chunks). Choose the span on the host so that
   `n_heads · groups · splits` reaches a target (measure the target: 2×, 4× the number of
   Adreno shader processors' workgroup slots).
c. **Batches over 128**, now that `parts` does not limit them: try 256 for long prompts.
   GEMM tiles are 64 tokens, so it doubles the attention workgroups and halves the
   per-pass overheads.

### F7. Optimizations, each its own A/B (F5's stage), in this order

a. **One rescale per sub-tile (FlashAttention-2):**
   - Each thread pair computes the SUB scores of its tokens first; the 4 lanes of a token
     already share a score through the shuffles.
   - Then take the sub-tile max, rescale `acc`/`l` once, and add `e^(s − m) v` per position.
   - This removes 15 of 16 `acc · r` multiplies (4 vec4s per position) and exp() calls.
   - It costs SUB × U = 32 score registers per thread. Watch for register spills, which showed
     up as the 4-token variant's slowdown in step 22.
b. **The causal mask only where it matters:** sub-tiles entirely below `pos + tok0` need no
   check. Split the loop into the unmasked part and the last ≤ 2 sub-tiles.
c. **Grouped-query sharing:** Qwen's 7 query heads per KV head each load the same keys and
   values. A workgroup per (KV head, 32 tokens) holding several query heads would cut
   global K/V reads up to 7×, but needs more registers per thread (or fewer tokens per
   workgroup). Do this only if F5's profile says global K/V reads matter (compare attention ms
   with bytes / measured bandwidth from `vk_bandwidth`).

Each one: F1's test, the phone suite, `vk_ppl` KL unchanged, then the battery A/B. Keep it only
if it wins.

### F8. Long-context decode (measure; change only if it falls)

Decode keeps attention.comp + merge, dispatched for every chunk the cache can hold
(`seq_len / 64`). The dispatch is recorded once, so chunks past the position return at once.
At seq_len 32768 that is 512 × 14 workgroups per layer, most of them empty, and the merge
loops over up to 512 chunks per head.

- `vk_decode` at depth 0 / 1024 / 4096 / 8192 (it already takes depths) with seq_len 8192+.
- If the decode speed falls at depth more than the extra K/V bytes explain, make each decode
  workgroup cover more positions (256 instead of 64: fewer splits and a shorter merge). That
  is flash-decoding with a longer split, the same `span` idea as F6b.

### F9. Close

- `IMPLEMENTATION.md`: the next-steps item. What was measured and kept, and what stays open.
- `results/phase-l3.md`: steps 26–27 (+ F6–F8 if done).
- Comments: `attention_tile.comp` and `attention_merge.comp` say they are now the `chunked`
  option and decode, respectively.
- Phone suite passes, then merge into `main` (`git merge --no-ff impl`).

## 5. What the review will check

At each step:

- **Correctness from references only:** numpy goldens, llama.cpp's KL and PPL. No second
  attention of our own written to compare against.
- **Barriers in uniform control flow:** no early `return` before a `barrier()` in any thread of a
  workgroup that still loops.
- **Indices:**
  - q at `tok * qstride + h * hs`.
  - K/V at `layer_off + pos * kv_dim + (h / (n_heads / n_kv)) * hs`.
  - o at `(tok * n_heads + h) * hs`.
  - These must match `attention_tile.comp` and `attention_merge.comp`.
- **The last workgroup of a batch** (nq < 32) and tokens past `ntok`: not written, and not
  counted in the max.
- **`parts` / `batch` sizing:** the old options must still get today's buffer; flash must not
  read `parts`.
- **Tests:**
  - The test bound has a written argument.
  - The test covers pos0 at chunk and sub-tile edges.
  - Cases come from `gen_kernels.py`, not hand-written numbers.
- **Measurement hygiene:**
  - Same-session pairs, on battery, cooled.
  - Order reversed in round 2.
  - Conditions in each raw file.
  - Numbers from different sessions are not compared.
- **House rules:**
  - `src/lib.rs` untouched on impl.
  - No `cargo fmt`.
  - Comments in the style of the surrounding shaders (what and why, measured numbers with
    their step).

## 6. References

- Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*,
  2022.
- Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*,
  2023 (per-tile rescale, parallelism over the sequence).
- Flash-decoding (Dao, Haziza, Massa, Sizov, 2023): the split + merge our decode and today's
  prompt shader use.
- Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018: the running max and
  sum.
- In this repo:
  - `shaders/attention_tile.comp` (step 22), `shaders/attention.comp`,
    `shaders/attention_merge.comp`.
  - The CPU flash-decoding: `src/kernels/neon.rs` `attention_group_part`, tested by
    `attention_parts_merge_to_numpy`.
