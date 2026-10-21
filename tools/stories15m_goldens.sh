#!/usr/bin/env bash
# Regenerates every stories15M golden file from llama2.c (the reference).
#
#   tools/stories15m_goldens.sh
#
# Needs: third_party/llama2.c cloned, testdata/models/stories15M/{stories15M,tokenizer}.bin.
# Writes testdata/stories15M/:
#   greedy.dump    activations at every layer boundary + logits + tokens,
#                  greedy decoding (-t 0) of PROMPT for STEPS positions
#   sampled.dump   tokens only matter here: -t 0.8 -p 0.9 -s 42
#   corpus.txt     tokenizer test strings (0x1E separated)
#   encoded.txt    llama2.c's encoding of each corpus string
set -euo pipefail
cd "$(dirname "$0")/.."

L2C=third_party/llama2.c
MODEL=testdata/models/stories15M/stories15M.bin
TOK=testdata/models/stories15M/tokenizer.bin
OUT=testdata/stories15M
PROMPT="Once upon a time"
STEPS=64
mkdir -p "$OUT"

# Apply the dump patch once (a clean checkout has no "aether_dump").
if ! grep -q aether_dump "$L2C/run.c"; then
  git -C "$L2C" apply "$PWD/tools/llama2c-dump.patch"
fi
make -s -C "$L2C" run
cc -O2 -o "$L2C/encode" -I "$L2C" tools/llama2c_encode.c -lm

AETHER_DUMP="$OUT/greedy.dump"  "$L2C/run" "$MODEL" -z "$TOK" -t 0 -n $STEPS -i "$PROMPT" > "$OUT/greedy.txt"
AETHER_DUMP="$OUT/sampled.dump" "$L2C/run" "$MODEL" -z "$TOK" -t 0.8 -p 0.9 -s 42 -n $STEPS -i "$PROMPT" > "$OUT/sampled.txt"

tools/.venv/bin/python -I tools/gen_tokenizer_corpus.py "$OUT/corpus.txt"
"$L2C/encode" "$TOK" < "$OUT/corpus.txt" > "$OUT/encoded.txt"

ls -la "$OUT"
