# Review: kquant-decode K0–K3 (commits 25bed76 … 86d59fa)

Reviewed 2026-11-04 against `plans/kquant-decode.md`. Host suite passes (121), but
two bugs sit in code that only the phone executes (NEON / `repack_weights()` is
false on the host), so neither is caught yet. Fix both **before** the phone
checklist: K3's commit would crash on the first decode token and give wrong Q6_K
results.

House rules still apply: `src/lib.rs` untouched, no `cargo fmt`, comments in the
style of the surrounding code, one commit per fix with its check.

---

## Must fix

### 1. Repacked K-quant matrices never get Q8_K activations → panic in decode

**Where:** `src/kernels/mod.rs`
- line 567 (`linear_batch_on`):
  `let xk = matches!(w, Weight::Q4_K(_) | Weight::Q6_K(_)).then(|| quantize_q8_k_activations(pool, x, t));`
- line 600 (`linear_many_batch_on`, aarch64):
  `ws.iter().any(|w| matches!(w, Weight::Q4_K(_) | Weight::Q6_K(_)))`
- line 602 (`linear_many_batch_on`, other targets): same match.

**What happens:** on the phone `k::repack_weights()` is true (i8mm + dotprod), so
`Transformer::new` → `Weights::repack` turns every Q4_K / Q6_K matrix with a row
count divisible by 4 into `Weight::Q4_Kx4` / `Weight::Q6_Kx4`. Those variants are
not in the three matches above, so `xk` is `None`, and `dot4` reaches
`Weight::Q4_Kx4(b) => neon_quant::dot4_q4_k_q8_k_x4(&b[...], k())`, where
`k()` is `xk.expect("a K-quant row needs Q8_K activations")` → **panic on the
first decode matmul** of LFM3 Q4_K_M (and of Qwen Q4_K_M's K-quant matrices).

Also hit by:
- `examples/kquant_bench.rs --repack` (it calls `linear_on` on repacked weights);
- the aarch64 block of `k_quant_linear_matches_reference` (`linear_on(p, &mut one, …, &r)`
  on the repacked weight), so the phone test run would fail here.

The host never sees it: `repack_weights()` is false there and the x4 test block is
`#[cfg(target_arch = "aarch64")]`.

**Fix:** one helper used at all three sites, matching the repacked variants too
(note `is_k_quant` already lists all four, but it is `#[cfg(target_arch = "aarch64")]`,
so either make a target-independent helper or move it):

```rust
/// True for the formats whose dot products take Q8_K activations.
fn needs_q8_k(w: &Weight) -> bool {
    matches!(w, Weight::Q4_K(_) | Weight::Q6_K(_) | Weight::Q4_Kx4(_) | Weight::Q6_Kx4(_))
}
```

and `matches!(w, …)` / `ws.iter().any(|w| matches!(…))` → `needs_q8_k(w)` /
`ws.iter().any(needs_q8_k)`. Consider having `is_k_quant` call it, so there is
one list.

**Check:** on the phone, `k_quant_linear_matches_reference` (its x4 cases: the
4×256 and 64×1536 goldens) passes; `kquant_bench --repack` runs; an LFM3 Q4_K_M
greedy run produces text.

### 2. `dot4_q6_k_q8_k_x4` drops the 0xAA flip → every Q6_K value off by ±128

**Where:** `src/kernels/neon_quant.rs`, `dot4_q6_k_q8_k_x4`, the inner loop:

```rust
let qh = vld1q_u8(g.qh.as_ptr().add(n * 128 + 16 * ((v0 % 8) + j)));
let lo = if high { vandq_u8(ql, vdupq_n_u8(0xF0)) } else { vshlq_n_u8::<4>(ql) };
let q = sri2(vshlq_u8(qh, vdupq_n_s8(6 - 2 * k as i8)), lo);
```

**What happens:** the trick (documented on `q6_k_block_q8_k`, ~line 403) builds the
int8 `4·(q − 32)` by putting q's 6 bits at bits 2..7; that is only `4·(q − 32)`
if q's bit 5 (the upper bit of each 2-bit `qh` pair, which lands at bit 7) is
flipped. The row-major kernel does this once per load:

```rust
let flip = vdupq_n_u8(0xAA);
let qh = [veorq_u8(vld1q_u8(...), flip), veorq_u8(vld1q_u8(...), flip)];   // line 422-423
```

The x4 kernel loads `qh` without the XOR, so it computes `4·q` read as int8
(= `4q` for q < 32, `4q − 256` for q ≥ 32) instead of `4q − 128`: every value
is wrong by 128 in the int8 domain (32 in q units). All Q6_K matrices on the
phone are wrong after K3: in LFM3 Q4_K_M that includes `ffn_down` in several
layers and the tied classifier / `token_embd` (check which tensors are Q6_K in
K0's shape list).

The scalar `dot4_q6_k_q8_k_x4` in `quant.rs` is correct (it decodes
`(lo | hi << 4) − 32` directly), which is why the host test passes.

**Fix:** XOR the loaded `qh` with 0xAA, as the row-major kernel does:

```rust
let qh = veorq_u8(vld1q_u8(g.qh.as_ptr().add(n * 128 + 16 * ((v0 % 8) + j))), vdupq_n_u8(0xAA));
```

(Hoist `vdupq_n_u8(0xAA)` out of the loops; see also point 5 about reusing `qh`
loads.) Update the doc comment of `dot4_q6_k_q8_k_x4` to say it uses the same
`4·(q − 32)` construction as `q6_k_block_q8_k`, including the flip.

**Check:** on the phone, `k_quant_linear_matches_reference` Q6_K x4 cases pass
within the existing bound (`1e-6 · √n`); LFM3 Q4_K_M one-chunk PPL is close to
K2's (not hundreds).

### Verify both fixes together

These two are exactly the kind of bug the plan's references are for; the
existing goldens already catch both on the phone. After fixing:
1. Host: full suite (`--test-threads=1 --skip lfm3_matches_transformers`, the PC is
   short of RAM), unchanged.
2. Phone: `cargo test --release --target aarch64-linux-android --lib quant` and
   `--lib kernels::tests::k_quant` before the full checklist.
3. Then the step-by-step phone checklist in `results/phase-kquant.md` (for K3,
   check out the commit with these fixes, not 572f689).

---

## Should fix (performance, measured by K0's bench)

### 3. One load per `sdot` for the activation broadcast

`dup4(xb.qs.as_ptr().add(...))` (`vld1q_dup_s32`) runs before every `sdot` in
both x4 kernels: in Q4_K 64 broadcast loads per super-block per group of 4 rows
(`a0`, `a1` for each `v`), in Q6_K 64 as well. AArch64 has the indexed form
`sdot vd.4s, vn.16b, vm.4b[lane]` = `vdotq_laneq_s32(acc, w, x, lane)`, which
multiplies each 4-byte group of `w` by the 4-byte group `lane` of `x` — the same
broadcast, from a register. So load 16 activation bytes once (`vld1q_s8`) and use
lanes 0..3 for four `sdot`s. Check that the intrinsic is stable on our Rust
(1.99); if it is not, it is one `asm!` like `smmla` (see `fn smmla`). Results are
bit-identical (same integer products). Measure with `kquant_bench --repack`
before/after.

### 4. Decode still quantizes to Q8_0 for K-quant-only matmuls

`linear_batch_on` / `linear_many_batch_on` always build `xq` (Q8_0) and `xd`, even
when every matrix needs only Q8_K, because `quant_rows` derives `nb` from `xq`.
Pass the column count (or `nbk`) instead and build `xq` only when a non-K-quant
matrix is present (or `xp` is needed for `smmla`). Small (the A7 profile had
`quant_rows` activation work at 1.7 %), but it is pure waste on every LFM3 decode
matmul. Bit-identical.

### 5. Repeated `ql` / `qh` loads in `dot4_q6_k_q8_k_x4`

Per super-block the row-major Q6_K kernel loads each `ql` vector once (used for
the low and the high nibble) and each `qh` vector once (used for four 2-bit
fields). The x4 kernel loops over the 16 sub-blocks and loads `ql` and `qh` again
for each, so `ql` is loaded 2× and `qh` 4× per super-block. The compiler may merge
them after unrolling; check the annotated listing (`simpleperf annotate` or
`objdump -d`). If not merged, restructure the loop by `(n, j)` and derive the
four sub-blocks' values from one `ql`/`qh` pair, as `q6_k_block_q8_k` does.
Bit-identical.

### 6. 4 vs 8 rows per group

The plan asked to try both N = 4 and N = 8 in K0's bench. K3 kept 4. That is fine
as long as `results/phase-kquant.md` says the 8-row variant was not tried and
why; or try it after 3–5, when the 4-row kernel's own overhead is gone, so the
comparison is fair.

---

## Cleanups (no behaviour change)

- `let at = if v < 4 { 4 * v } else { 16 + 4 * (v - 4) };` is `4 * v` for every
  `v` in 0..8. It appears in `repack_q4_k`, `dequantize_q4_kx4`,
  `scalar::dot4_q4_k_q8_k_x4` (as `vv`), `dot4_q4_k_q8_k_x4` and `unpack_k4`'s
  `Q4_Kx4` arm; the `BlockQ4_Kx4` doc comment describes the same thing in two
  cases. Replace with `4 * v` and simplify the comment ("row `r`'s bytes
  `32c + 4v .. +4`").
- `let (nbk, nk) = (n / QK_K, n / QK_K);` in `k_quants_match_gguf`: two names for
  one value.

---

## Checked and fine (no action)

- `quantize_q8_k` matches `quantize_row_q8_K_ref` (signed max, first on ties,
  `iscale = -127/max`, `min(127)`, `d = 1/iscale`, all-zero block); the test
  compares bytes against llama.cpp's own function through ctypes
  (`tools/gen_q8k.py`).
- Q8_K dot products: the Q4_K min term via `bsums` and the int32 sub-block
  scaling are right; int32 ranges are safe (Q6_K super-block worst case ~1.3·10⁸,
  Q4_K ~3·10⁷).
- `BlockQ4_Kx4` / `BlockQ6_Kx4` layouts: the Q6_K index arithmetic matches
  ggml's dequantizer (checked by hand), and `repacked_k_rows_dequantize_like_row_major`
  covers it on the host; sizes asserted at compile time.
- `unpack_k4`'s `Q4_Kx4` / `Q6_Kx4` arms rebuild row-major blocks correctly
  (`q4_k_sub` reads only `qs`, so the zeroed `scales` there are fine).
- GPU and DSP models build from un-repacked `Weights` (`GpuModel::with`,
  `DspModel::new`), so the new variants do not reach `MatQ::new` / `DspFormat::of`.
- The K1 write-up (bit-identical; the Q6_K chain deferred) and the K3 bit-identity
  correction (same integers, different f32 grouping) are accurate.
