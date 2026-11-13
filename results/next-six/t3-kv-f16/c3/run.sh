#!/bin/bash
# T3 C3: perplexity and KL, 20 x 512 wikitext-2, llama.cpp and Aether, per KV cache format.
# Runs one job at a time; each job's output goes to $OUT/<name>.txt.
set -u
R=/home/curious/Documents/hermes-projects/aether-infer
KV=/home/curious/Documents/hermes-projects/aether-infer-kv
OUT=${OUT:-/tmp/claude-1000/-home-curious-Documents-hermes-projects-aether-infer/db08fe39-a46e-4005-afd9-05b02a7c3a54/scratchpad/c3}
mkdir -p "$OUT"
WIKI=$R/testdata/wikitext/wikitext-2-raw/wiki.test.raw
LP=~/llama.cpp/build/bin/llama-perplexity
AE=$KV/target/release/aether
declare -A BASE=(
  [qwen2.5-0.5b-q4_0]=$R/hostdata/imatrix/qwen2.5-0.5b-f32.kld
  [qwen2.5-0.5b-q4_k_m]=$R/hostdata/imatrix/qwen2.5-0.5b-f32.kld
  [LFM3-1.7B-Q4_K_M]=$R/hostdata/lfm3/kl-base-bf16.bin
)
for m in qwen2.5-0.5b-q4_0 qwen2.5-0.5b-q4_k_m LFM3-1.7B-Q4_K_M; do
  f=$R/testdata/gguf/$m.gguf
  b=${BASE[$m]}
  for t in f16 f32 q8_0; do
    o=$OUT/llamacpp-$m-$t.txt
    [ -s "$o" ] && grep -q "Final estimate" "$o" && continue
    echo "== llama.cpp $m $t $(date +%T)"
    $LP -m "$f" -f "$WIKI" -c 512 --chunks 20 -t 6 -ctk $t -ctv $t -fa on --kl-divergence-base "$b" --kl-divergence > "$o" 2>&1
    grep -E "Final estimate|Mean    KLD|Same top" "$o"
  done
  for t in f32 f16; do
    o=$OUT/aether-$m-$t.txt
    [ -s "$o" ] && grep -q "KLD\|Final" "$o" && continue
    echo "== aether $m $t $(date +%T)"
    $AE perplexity -m "$f" -f "$WIKI" --kl-divergence-base "$b" --kl-divergence --chunks 20 -t 6 -ctk $t -ctv $t > "$o" 2>&1
    tail -12 "$o"
  done
done
echo "== done $(date +%T)"
