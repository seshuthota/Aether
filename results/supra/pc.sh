#!/usr/bin/env bash
# Supra2.5 on the PC (aether-infer plans/supra.md, S2): wikitext-2 PPL and KL,
# llama.cpp and aether on the same files. Run from aether-infer.
set -euo pipefail
OUT=${OUT:-../Aether/results/supra/pc}
L=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
W=testdata/wikitext/wikitext-2-raw/wiki.test.raw
G=testdata/gguf
BASE=$OUT/f16.kld
mkdir -p "$OUT"
{ git log --oneline -1; git -C "$HOME/llama.cpp" log --oneline -1; date -Is; nproc; } > "$OUT/conditions.txt"
"$L/llama-perplexity" -m $G/Supra2.5-65M-fixed.F16.gguf -f $W -c 512 --chunks 40 -t 8 --kl-divergence-base "$BASE" > "$OUT/llama-F16.log" 2>&1
"$L/llama-quantize" testdata/gguf/Supra2.5-65M-fixed.F16.gguf "$OUT/mix-q4_0.gguf" Q4_0 8 > "$OUT/quantize-mix-q4_0.log" 2>&1
for m in F16 Q8_0 Q4_K_M Q4_0 mix; do
  f=$G/Supra2.5-65M-fixed.$m.gguf; [ $m = mix ] && f=$OUT/mix-q4_0.gguf
  target/release/aether perplexity -m "$f" -f $W -c 512 --chunks 40 --kl-divergence-base "$BASE" --kl-divergence > "$OUT/aether-$m.log" 2>&1
  [ $m = F16 ] || "$L/llama-perplexity" -m "$f" -f $W -c 512 --chunks 40 -t 8 --kl-divergence-base "$BASE" --kl-divergence > "$OUT/llama-$m.log" 2>&1
done
# The published (mislabelled) files, llama.cpp's own tokenization.
"$L/llama-perplexity" -m $G/Supra2.5-65M.F16.gguf -f $W -c 512 --chunks 40 -t 8 > "$OUT/llama-published-F16.log" 2>&1
rm -f "$OUT/mix-q4_0.gguf" "$BASE"
