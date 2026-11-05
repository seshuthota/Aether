# Plan: close Gemma 4 on every backend (CPU finish, GPU, DSP)

Written 2026-11-05 on branch `gemma4` (`63bcaa9`). Every engine fact
below was read from `src/model/gpu.rs`, `src/model/dsp.rs`, `dsp/model.c`
and the shaders at that commit. Every llama.cpp fact is from `~/llama.cpp`
at `d02b4c0d6`. Numbers marked *calculated* are arithmetic, not
measurements.

## 0. Where Gemma 4 stands, and what "closed" means

| Backend | Today | Closed when |
| --- | --- | --- |
| CPU | Verified on the PC (`results/phase-gemma4.md`); phone checks owed (`plans/gemma4-phone-checks.md`) | Phone checks pass, merged, G8 measured, G9 decided |
| GPU (Vulkan, Adreno 830) | Refuses Gemma 4 (`GpuModel::with`) | Decode, prompt pass and lookup run Gemma 4, PPL/KL match llama.cpp's, speed measured against llama.cpp's OpenCL |
| DSP (Hexagon v79) | Refuses Gemma 4 (`DspModel::new`) | Decode, HMX prefill and lookup run Gemma 4, PPL/KL match llama.cpp's, speed measured against llama.cpp's Hexagon backend |
| App / C ABI | CPU path only by default; untested with Gemma 4 | The app runs Gemma 4 on each of CPU / GPU / DSP |

**Out of scope:**
- LFM3 on the GPU or DSP. Its conv layers are a separate plan, though the
  per-layer groundwork in part S makes it easier.
- Images and audio.
- E4B and the larger Gemma 4 models: the same code, a later check.

Order: **A** (CPU, needs the phone) can run in parallel with **S** (shared
groundwork, PC). Then **B** (GPU) and **C** (DSP), in either order. **D**
(app) comes last.

### Status (2026-11-05, branch `gemma4-devices`)

| Step | State |
| --- | --- |
| S1 | Not needed as a new type: the GPU reads each layer's shapes from `Config`'s per-layer accessors (`head_dim_of`, `q_dim`, `kv_dim_of`, `hidden_of`, `kv_source_of`, `window`, `rope`), which the CPU already uses |
| S2 | Done (`7e67efa`): Q4_K_M PPL(base) 161.2919 = llama.cpp's, KL 0.1751 vs 0.1783 ± 0.0053 |
| S3 | Done by reuse: `transformer::ple_inputs` is `pub(crate)` and the GPU calls it per token |
| S4 | Done: the CPU gate failed narrowly on KL (0.1819 vs 0.1751, activation rounding included); the GPU with Q8_0 matrices gives KL 0.1636, below both the F32 CPU and llama.cpp, so the conversion stays (`results/phase-gemma4.md`) |
| B1 | Done (`4f91d07`): attention at head size 256 / 512 and the window, decode and prompt alike |
| B2 | Option (c): the prompt pass uses B1's attention + merge pair |
| B3 | Done (`49b90aa`) |
| B4, B5 (prompt pass) | Done on the PC (`33a9499`): Q4_0 decode and prompt pass, Q4_K_M decode, PPL/KL at llama.cpp's (`results/phase-gemma4.md`, GPU) |
| B5 (lookup) | Not started: Gemma's `few` is 1, so lookup decodes plainly |
| B6, B7 | Need the phone. B7 also decides the f16/int8 second copy for prompt speed |
| C0 | Probe written (`dsp_maps_gemma_sized_buffers`, `--ignored`); runs on the phone |
| C3 | Done on `hexagon-sim`: `dsp/gemma_hvx.h`, 52 golden cases, worst 1.9e-7 of the largest value |
| C1, C2, C4–C7 | Wait for C0 (it decides the weight layout); C2's streamed attention and VTCM sizing need the phone anyway |
| D, A | Need the phone |

## 1. Why the GPU and DSP refuse today: every assumption Gemma 4 breaks

| Assumption in the code | Where | Gemma 4 E2B |
| --- | --- | --- |
| One head size; GPU attention needs hs ≤ 64 (4 threads × 16 values per token) | `gpu.rs:505`, `attention*.comp` | 256 (sliding) and 512 (full) |
| DSP: hs = dim / heads, hs % 64 == 0, hs ≤ 256 | `dsp/model.c:174,186` | dim / heads = 192; 512 on full layers |
| One FFN width (`hidden_dim`), SwiGLU | `gpu.rs:498,931`; DSP `D_HIDDEN`, "SwiGLU + quantize" | 6144 / 12288 per layer, GELU-tanh |
| Every layer has its own K/V; cache offset `l · seq_len · kv_dim` | `gpu.rs:894`; DSP `D_KC`/`D_VC` | layers 15–34 read layer 13 or 14's cache and store nothing |
| No sliding window | both | 28 of 35 layers see the last 512 positions |
| Score scale 1/√hs | all attention shaders; `model.c:387,501` | 1.0 (the CPU path pre-scales q by √hs) |
| No q/k norm, no v norm | `qkv_post.comp`; DSP "bias + RoPE + KV store" | q/k RMSNorm with gain, v RMSNorm without |
| One RoPE (θ, all pairs); DSP gets one sin/cos table per token | `qkv_post.comp`; `D_ROT` | θ 1e4 / 128 pairs (sliding), θ 1e6 / 64 of 256 pairs (full) |
| Matrices in Q4_0, Q5_0, Q8_0, Q4_K, Q6_K | `MatQ::new`, `DspFormat::of` | Q4_K_M file: `inp_gate` / `proj` are **F32** (70 matrices, 110 MB) |
| Embedding × 1, no extra per-layer input, no layer scale, no soft-cap | both | × √1536, the PLE block in every layer, `layer_output_scale`, soft-cap 30 |

None of these is a large change by itself. Together they touch every stage
of both backends' layer step, which is why each is its own part.

## 2. Design decisions (shared by B and C)

1. **The per-layer inputs are computed on the CPU, per token, and written
   into the device's state.** Both backends already look up the token
   embedding on the CPU (`gpu.rs:984`, `dsp.rs:299`). The PLE inputs are
   that row plus one table row (5–6 KB) and a 27.5 MB BF16 projection
   (`ple_inputs`, verified). That GEMV costs ~0.5 ms a token at the phone's
   ~55 GB/s (*calculated*); A2's profile gives the real cost. The 1.3 GB
   table never goes to the device. Moving the projection onto the device is
   a later optimisation, made only if B7 / C7 show it matters.
2. **Sliding layers keep a full-length cache, and attention starts at
   `max(0, pos − 511)`.** No ring on the device. The memory at 4096
   positions, all f32 K+V (*calculated*): 12 sliding layers × 256 + 3 full
   × 512 = 4,608 floats per position for K, the same for V, 36.9 KB a
   position, **151 MB** at 4096. The ring is a later memory optimisation.
3. **Shared layers get a per-layer cache offset** (`kv_source_of(l)`) and
   skip the K/V products and the store.
4. **F32 matrices (Q4_K_M's `inp_gate` / `proj`) become Q8_0 at load** for
   both backends: 110 → 29 MB (*calculated*), with an accuracy gate (S4).
   The CPU keeps F32 unless G8 adopts the same, under the same gate.
5. **Attention scale 1.0 is folded into q**, as on the CPU: q × √hs after
   its norm and RoPE, so the shaders' `1/√hs` gives 1.0. No shader gets a
   scale parameter.
6. **One description of each layer** goes to both backends, replacing the
   scalars: (hs, q_dim, kv_dim, hidden, kv_source, window, rope θ / pairs,
   flags). S1 builds it from `Config`.

## 3. How correctness is checked (external references only)

- **llama.cpp's PPL on the phone, same file.** Its CPU backend always.
  Also its OpenCL and Hexagon backends if they run Gemma 4 (S0 finds out).
- **KL against llama.cpp's BF16 base.** A new reader (S2) for the file
  `llama-perplexity --kl-divergence-base` writes, so any of our backends
  can be scored against llama.cpp's own BF16 logits on the phone. The
  BF16 model (9.3 GB, 16 GB peak on our side) does not fit the phone, so
  `--base` cannot be used there. An 8-chunk base is 8 × 512 × 262,144 ×
  2 B = 2.1 GB (*calculated*).
- **Greedy text against `llama-completion`** on the 4 prompts of
  `results/phase-gemma4/raw/greedy-Q4_0-pc.txt`. Every split must be a
  near-tie, with the logit gap recorded.
- **numpy goldens** for every new shader or HVX kernel, through the
  existing `tools/gen_kernels.py` cases (head 256 / 512, window, GELU,
  soft-cap, plain norm). These already exist for the CPU.
- **Existing models unchanged on each backend:** Qwen2.5-0.5B and
  SmolLM2-135M PPL through `vk_ppl` / `dsp_ppl`, before and after each
  step, to the digit.
- **Bounds:** LFM3's. Our PPL within the run-to-run spread of llama.cpp's
  (the CPU got 0.06 % on Q4_0), KL within llama.cpp's error bar. A larger
  gap is investigated, not accepted.

**Where checks can run without the phone:**
- The Vulkan tests run on the PC: 11 of 12 `kernels::vulkan` tests pass
  through the host's Vulkan driver (2026-11-05). The 12th needs the
  `tools/.venv` symlink in this worktree. Subgroup sizes differ (NVIDIA /
  Intel 32, Adreno 64), so every shader is checked on the phone as well.
- HVX kernels run on the PC in `hexagon-sim` (v79; see the
  hexagon-simulator memory). QuRT threads, DMA, VTCM, HMX and FastRPC need
  the phone.

## 4. Part A: finish the CPU (needs the phone)

**A1. Phone checks and merge.** All of `plans/gemma4-phone-checks.md`: the
NEON kernel goldens (commit the GELU / soft-cap only if they pass), the
Gemma tests, the whole suite with `--features vulkan,hexagon`, Q4_0 PPL,
existing models to the digit, greedy text, peak RSS, the speed runs
against `llama-bench`, and the app. Then merge `gemma4` → `main`.

**A2. Profile decode and prefill** (`simpleperf`, Q4_0 and Q4_K_M, cores
2–7). Split the time into: classifier (Q8_0 / Q4_K, 262,144 rows), FFN,
attention by head size, the PLE block, the PLE inputs (BF16 GEMV + table
row), soft-cap, page faults on table rows. A3 is ordered by this table,
not by guesses.

**A3. G8 optimisations, one commit each, measured same-session A/B,
PPL unchanged within its bound.** The candidates from `plans/gemma4.md`
G8:
- `per_layer_model_proj` as a BF16 GEMV (`bfdot`; the phone has bf16)
  instead of widening it to f32: 55 → 27.5 MB per token.
- Q4_K_M's F32 `inp_gate` / `proj` as Q8_0 (shared with S4's gate).
- A tiled NEON attention for head size 512 (the output row in register
  blocks; G3 planned it). Only if A2 shows attention at depth matters.
- `madvise(WILLNEED)` on the next token's table rows.
- The f16 KV cache (`plans/kv-cache.md`) applies here, as a separate plan.

**A4. G9, the MTP draft head: decide, don't build blindly.** It needs a
hidden-state hand-off `--draft` does not have. Build it only if A2/A3
leave decode as the limit and prompt lookup's gains on Gemma are small;
record the decision with the numbers.

## 5. Part S: shared groundwork (PC)

**S0. llama.cpp's GPU and NPU on Gemma 4** (phone, in A1's session). Run
`llama-bench` and `llama-perplexity` with the OpenCL and Hexagon backends
on both files. Record:
- whether every op runs on the device or some fall back to the CPU (the
  log lists the splits);
- PPL and pp / tg speed.

These are B's and C's speed references. If a backend cannot run Gemma 4,
note it; then that backend's accuracy reference is llama.cpp's CPU PPL.

**S1. One layer description for the devices.** Add `Config::layer_desc(l)`
with the fields of section 2, item 6. The GPU and DSP read their shapes
from it in place of the scalars (`c.head_size()`, `c.hidden_dim`,
`c.kv_dim()`). Existing models give the same values, so their PPL stays
to the digit (`vk_ppl` / `dsp_ppl` on Qwen and SmolLM2, before and after).

**S2. KL against llama.cpp's base file.** Read the format from
`tools/perplexity/perplexity.cpp` (the writer), not from memory, and add
`--kl-base FILE` to `aether --ppl`, `vk_ppl` and `dsp_ppl`. Checks on the
PC:
- the CPU path scored against the BF16 base gives llama.cpp's own
  `PPL(base)` (161.29 over 20 chunks, from its 16-bit log-probs);
- Q4_0 / Q4_K_M give KL within the error bar of `--base`'s figures
  (0.3458 / 0.1814 over 8 chunks).

**S3. The PLE inputs for device sessions.** A function on the CPU side
that writes `[n_layers, ple_dim]` per token (decode) and per row (prompt
batches) into a device buffer, reusing `ple_inputs` / `ple_finish`. On the
PC, its output equals the CPU path's `ple` tap bit for bit (it is the same
code).

**S4. F32 → Q8_0 at load for the devices, gated.** Quantize `inp_gate` /
`proj` with our verified Q8_0 quantizer (byte-identical to llama.cpp's).
Gate: the CPU path with those 70 matrices in Q8_0 must give Q4_K_M PPL and
KL against BF16 within the error bars of the F32 run on the PC. If it
fails, the backends need an f16 matrix path instead (a larger step; noted
as risk 3).

## 6. Part B: the GPU (Vulkan)

Each step: shader goldens on the PC first, then on the phone; existing
models' `vk_ppl` unchanged to the digit.

**B1. Attention for head size 256 and 512, decode** (`attention.comp`,
`attention_merge.comp`). Today one workgroup of 64 threads holds a token's
hs ≤ 64 values. For hs 256 / 512, each thread holds hs / 64 values (4 or
8). The score is a sum over threads (a subgroup reduction, then shared
memory across subgroups on hardware with 32-wide subgroups). Add the
window start. Goldens: the existing head-256 / 512 and window cases.

**B2. Attention for the prompt pass** (`attention_flash.comp`). Its layout
is 4 threads × 16 values × 2 tokens per thread. At hs 512 that is 8× the
registers per token. Options:
- (a) fewer tokens per workgroup;
- (b) split the head across more threads, with a reduction per score;
- (c) for hs > 64, the older tile + merge pair, made hs-generic in B1.

Start with (c), measured. (a) or (b) only if the profile says prompt
attention matters at Gemma's sizes. Goldens: causal 8 × 1 at 256 / 512,
window 512.

**B3. The new element-wise shaders.**
- `qkv_post.comp`: q/k RMSNorm with gain, v RMSNorm without gain, RoPE
  with per-layer θ and pairs (pairs < hs / 2 leaves the rest unrotated),
  the q pre-scale, and no K/V store on shared layers.
- `geglu.comp`, beside `swiglu.comp`.
- `softcap.comp` (or a flag on the argmax / logits read).
- The PLE block: `inp_gate` GEMV (existing), GELU ⊙ ple[l] (a new
  element-wise shader), `proj` GEMV (existing), post-norm (`rmsnorm.comp`),
  add, × `out_scale`.

Goldens: the CPU's numpy cases (gelu, softcap, plainnorm), plus a PLE-block
case from `tools/gen_kernels.py` (new: numpy of the block's formula).

**B4. Decode end to end.** `GpuModel::with` builds per-layer buffers from
S1: per-layer q / kv / hidden widths, so each layer's matrices are bound
with its own row count. The KV cache is laid out by owning layer, with
offsets from `kv_source`. The PLE inputs come from S3, and logits are
soft-capped. The F32 matrices come from S4. Checks:
- `vk_ppl` on Q4_0 and Q4_K_M against llama.cpp's PPL;
- KL against the BF16 base (S2);
- greedy text on the 4 prompts, splits only at near-ties;
- GPU decode logits against GPU batch logits, as the existing GPU tests
  do, at positions 0–600 (past the window).

**B5. The prompt pass and lookup** (`forward_batch`, `forward_few`). The
same per-layer plumbing in the batched path, the window in the prompt
shaders (B2), and lookup turned on for Gemma only if B7 shows a gain.
Checks: prompt-pass PPL equals decode PPL within the existing GPU bound;
lookup keeps the greedy text.

**B6. Memory.** Device buffers for the non-table weights: 1.5 GB (Q4_0 /
Q4_K_M, *calculated* from `results/phase-gemma4.md`), plus the cache
(151 MB at 4096) and the activations. Measure the process's PSS on the
phone. If the Adreno's allocation limit refuses a 1.5 GB buffer set, split
it per layer (the code already allocates per matrix).

**B7. Speed against llama.cpp's OpenCL** (S0), same session, cooled,
caps recorded, alternated, two rounds: pp64 / 128 / 512, tg64 at depth 0 /
512. Power, J/token and temperature as in L.3 (the GPU's case there was
energy, not speed).

## 7. Part C: the DSP (Hexagon)

**C0. Feasibility first: the weights in one shared buffer.** The DSP path
uploads all matrices into one FastRPC-shared buffer. Gemma 4's non-table
weights are ~1.5 GB, against Qwen's ~0.4 GB, and the DSP's address space
is 32-bit. On the phone, map and touch a 1.6 GB buffer from the DSP
before anything else. If it fails, split the weights over several buffers
(more `desc` entries), or keep the classifier (428 MB in Q8_0, the
largest matrix) on the CPU and hand back the last hidden state. This step
decides C's shape, so it comes first.

**C1. `desc` per layer.** Per-layer entries from S1: hs, q_dim, kv_dim,
hidden, kv_source, window, flags. Remove `hs = dim / heads`. Relax the
check at `model.c:186` to hs % 64 == 0 and hs ≤ 512 with the VTCM
budget below. Two RoPE tables per token (`D_ROT` × 2: sliding and full),
written by the CPU as today.

**C2. Attention at head size 512 in VTCM.** Today `Attn.per = g · hs + 2g`
floats per partial, and each thread's K/V block is DMA'd into its VTCM
slots (`ATTN_BLOCK` 256 positions). At hs 512 with one KV head, a block of
256 positions of K and V in f16 is 2 × 256 × 512 × 2 B = **512 KB**, twice
hs 256's (*calculated*). Fit it to `VTCM_SHARE` per thread: halve
`ATTN_BLOCK` for hs 512 if needed. Check: `hexagon-sim` goldens for the
attention arithmetic, then the phone for the DMA / VTCM path. Add the
window start, and the q pre-scale in the "bias + RoPE + KV store" stage.

**C3. HVX element-wise kernels**, on `hexagon-sim` against the numpy
goldens first:
- GELU-tanh × up (as `x · sigmoid(2z)`, the CPU's NEON identity, on the
  existing HVX `exp`);
- soft-cap;
- RMSNorm without gain;
- q/k norm with gain;
- the PLE block's element-wise part, ⊙ and × `out_scale`.

**C4. Decode end to end.** The new stages in `model_decode`: per-layer
widths, shared-KV reads (no store), GELU in place of SwiGLU, the PLE block
after the FFN residual, soft-cap before the argmax. The PLE inputs come
from S3, through the shared state. Checks: as B4, through `dsp_ppl` and
the DSP decode tests (decode vs few vs prefill rows, as
`model::dsp::tests` do today).

**C5. The HMX prefill.** Per-layer widths in `hmx_rows`. The activation
bank is 3 MB; a 12288-wide FFN row in f16 is 24.6 KB a token, so the
`down` input fits **≤ 122 tokens** (*calculated*; Qwen's limit of 320
came from the same bank). Cap Gemma's prefill batch accordingly (e.g. 96)
and measure. Checks: prefill logits equal the decode step's (the existing
`hmx_prefill` tests), and PPL through the prefill.

**C6. Lookup / few-token pass** for Gemma. Turn it on only if C7 shows a
gain. The greedy text must be unchanged.

**C7. Speed against llama.cpp's Hexagon backend** (S0), same session, as
B7, with power.

## 8. Part D: the app and C ABI

- `aether_open_gpu` / `aether_open_dsp` accept Gemma 4 once B / C pass.
  Their error messages are tested for the cases still refused (e.g.
  F32 matrices if S4 fails).
- The app's CPU / GPU / DSP selector with Gemma 4: a few turns of chat on
  each, PSS recorded next to the 1.22 GB of the production app, and the
  reasoning hidden (`<|channel>` filter) on every backend.
- `results/phase-gemma4.md`: a section per backend; `IMPLEMENTATION.md`
  section 0 updated.

## 9. Risks

1. **DSP address space (C0).** 1.5 GB in one shared buffer may not map.
   The fallback (split buffers, or the classifier on the CPU) changes C's
   structure, so C0 runs before any other C step.
2. **Adreno buffer limits (B6).** Large allocations or a 262,144-row
   classifier dispatch may exceed limits (`maxStorageBufferRange`). Query
   them first; split per matrix and per row block.
3. **S4's gate fails.** Then F32 matrices need an f16 GEMV on both
   devices, a separate step each.
4. **Prompt attention at hs 512 on the GPU (B2).** The register budget
   may force a slower layout; prompt speed may trail llama.cpp's OpenCL
   further than for Qwen.
5. **The CPU-side PLE GEMV becomes the bottleneck** for device decode.
   Measured in B7 / C7; the fix is a BF16 or Q8_0 projection on the
   device.
6. **llama.cpp's backends may not run Gemma 4 fully (S0).** Then the
   speed comparison is against llama.cpp's CPU, and stated as such.

## 10. Branches and order of work

- **A** on `gemma4` in `../Aether-next`, until merged.
- **S** on a new branch `gemma4-devices`, off `gemma4` and rebased onto
  `main` after A1's merge. S0 joins A1's phone session; S1–S4 need only
  the PC.
- **B** and **C** on `gemma4-devices`, one commit per step, each after its
  check. B is mostly testable on the PC (Vulkan shaders); C needs the
  phone from C0 on, with HVX arithmetic on the simulator.
- Merge to `main` per backend, after its accuracy and speed steps
  (B4–B7, or C4–C7) pass on the phone, and after existing models are
  unchanged on that backend.

| Step | Needs the phone | Rough size |
| --- | --- | --- |
| A1 | yes | one session |
| A2–A4 | yes | small, one commit per idea |
| S0 | yes (with A1) | small |
| S1–S4 | no | small each |
| B1–B3 | PC first, then phone | medium (B2 largest) |
| B4–B5 | PC first, then phone | medium |
| B6–B7 | yes | small |
| C0 | yes | small, decisive |
| C1–C3 | simulator, then phone | medium |
| C4–C5 | yes | large (the biggest part of the plan) |
| C6–C7 | yes | small |
| D | yes | small |
