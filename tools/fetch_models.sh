#!/usr/bin/env bash
# Downloads the models and data the tests and benchmarks use, and makes the
# GGUF files with llama.cpp's own tools (so both engines read the same bytes).
# Files that already exist are skipped, so it is safe to re-run.
#
#   tools/fetch_models.sh [stories15m|smollm2-135m|smollm2-360m|qwen2.5|gguf|wikitext|all]
#
# Fetch only what the phase being worked on needs: the full set is ~6 GB.
#
#   stories15m     testdata/models/stories15M/       ~60 MB  (llama2.c, Phase 2)
#   smollm2-135m   testdata/models/SmolLM2-135M/     ~270 MB (Phase 5)
#   smollm2-360m   testdata/models/SmolLM2-360M/     ~720 MB (Phase 5)
#   qwen2.5        testdata/models/Qwen2.5-0.5B-Instruct/ ~1 GB (Phase 7)
#   gguf           f32 GGUFs in hostdata/gguf/ (host-only, ~2.5 GB) and
#                  Q8_0 / Q4_0 / Q5_0 GGUFs in testdata/gguf/ (`llama-quantize
#                  --pure`) plus llama.cpp's usual Q4_K_M mix, for SmolLM2-135M
#                  and Qwen2.5 (Phases 6-7)
#   wikitext       testdata/wikitext/wikitext-2-raw/wiki.test.raw (perplexity)
#   imatrix        hostdata/imatrix/ (host-only, ~1.3 GB; needs gguf and
#                  wikitext): llama-imatrix importance matrices on the second
#                  half of wiki.test.raw (perplexity scores the first chunks),
#                  and llama-quantize --imatrix output (--pure Q4_0 / Q5_0,
#                  Q4_K_M) for SmolLM2-135M and Qwen2.5 (L.2)
#
# Needs: curl, unzip; for gguf also tools/.venv (tools/requirements.txt) and a
# host build of llama.cpp (LLAMA_CPP, default ~/llama.cpp, tools in build/bin;
# LLAMA_BIN overrides that directory). Build it with -DBUILD_SHARED_LIBS=OFF if
# LD_LIBRARY_PATH may point at another llama.cpp's libraries.
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA_CPP=${LLAMA_CPP:-$HOME/llama.cpp}
LLAMA_BIN=${LLAMA_BIN:-$LLAMA_CPP/build/bin}
HF=https://huggingface.co

# fetch URL FILE: download URL to FILE unless FILE exists (via a .part file,
# so an interrupted download is not mistaken for a finished one).
fetch() {
  [ -f "$2" ] && return
  mkdir -p "$(dirname "$2")"
  echo "fetch $2"
  curl -sSfL -o "$2.part" "$1" && mv "$2.part" "$2"
}

# hf_model REPO DIR FILES...: files of a Hugging Face model repo.
hf_model() {
  local repo=$1 dir=$2; shift 2
  for f in "$@"; do fetch "$HF/$repo/resolve/main/$f" "$dir/$f"; done
}

HF_FILES=(config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt model.safetensors)

stories15m() {
  fetch "$HF/karpathy/tinyllamas/resolve/main/stories15M.bin" testdata/models/stories15M/stories15M.bin
  fetch https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin testdata/models/stories15M/tokenizer.bin
}
smollm2_135m() { hf_model HuggingFaceTB/SmolLM2-135M testdata/models/SmolLM2-135M "${HF_FILES[@]}" special_tokens_map.json; }
smollm2_360m() { hf_model HuggingFaceTB/SmolLM2-360M testdata/models/SmolLM2-360M "${HF_FILES[@]}" special_tokens_map.json; }
qwen25() { hf_model Qwen/Qwen2.5-0.5B-Instruct testdata/models/Qwen2.5-0.5B-Instruct "${HF_FILES[@]}"; }

# gguf_of HF_DIR NAME: NAME-f32.gguf (host-only), NAME-{q8_0,q4_0,q5_0}.gguf
# and NAME-q4_k_m.gguf.
gguf_of() {
  local src=$1 name=$2 f32=hostdata/gguf/$2-f32.gguf
  mkdir -p hostdata/gguf testdata/gguf
  if [ ! -f "$f32" ]; then
    echo "convert $f32"
    tools/.venv/bin/python "$LLAMA_CPP/convert_hf_to_gguf.py" "$src" --outtype f32 --outfile "$f32"
  fi
  for q in q8_0 q4_0 q5_0; do
    local out=testdata/gguf/$name-$q.gguf
    [ -f "$out" ] && continue
    echo "quantize $out"
    # --pure: every matrix in this format, the embedding included (no
    # mixed K-quant choices), which is what our in-memory -q does too.
    "$LLAMA_BIN/llama-quantize" --pure "$f32" "$out" "${q^^}"
  done
  # The mix llama.cpp picks per matrix (on these models mostly Q5_0, with
  # Q4_K / Q6_K where rows are a multiple of 256).
  local out=testdata/gguf/$name-q4_k_m.gguf
  [ -f "$out" ] || { echo "quantize $out"; "$LLAMA_BIN/llama-quantize" "$f32" "$out" Q4_K_M; }
}
gguf() {
  smollm2_135m; qwen25
  gguf_of testdata/models/SmolLM2-135M smollm2-135m
  gguf_of testdata/models/Qwen2.5-0.5B-Instruct qwen2.5-0.5b
}

wikitext() {
  local out=testdata/wikitext/wikitext-2-raw/wiki.test.raw
  [ -f "$out" ] && return
  # The same archive llama.cpp's scripts/get-wikitext-2.sh uses.
  fetch "$HF/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" testdata/wikitext/wikitext-2-raw-v1.zip
  unzip -q -o testdata/wikitext/wikitext-2-raw-v1.zip -d testdata/wikitext
  rm testdata/wikitext/wikitext-2-raw-v1.zip
}

# imatrix_of NAME: NAME.imatrix.gguf (100 chunks of 512),
# NAME-10.imatrix.gguf (10 chunks, for the collection test) and
# NAME-{q4_0,q5_0,q4_k_m}-im.gguf.
imatrix_of() {
  local m=$1 d=hostdata/imatrix f32=hostdata/gguf/$1-f32.gguf
  mkdir -p $d
  if [ ! -f $d/calib.txt ]; then
    local w=testdata/wikitext/wikitext-2-raw/wiki.test.raw
    tail -n +$(( $(wc -l < $w) / 2 )) $w > $d/calib.txt
  fi
  for n in 100 10; do
    local out=$d/$m.imatrix.gguf
    [ $n = 10 ] && out=$d/$m-10.imatrix.gguf
    [ -f $out ] || { echo "imatrix $out"; "$LLAMA_BIN/llama-imatrix" -m $f32 -f $d/calib.txt -o $out --chunks $n -c 512 > $out.log 2>&1; }
  done
  for q in q4_0 q5_0; do
    local out=$d/$m-$q-im.gguf
    [ -f $out ] || { echo "quantize $out"; "$LLAMA_BIN/llama-quantize" --pure --imatrix $d/$m.imatrix.gguf $f32 $out "${q^^}" > $out.log 2>&1; }
  done
  local out=$d/$m-q4_k_m-im.gguf
  [ -f $out ] || { echo "quantize $out"; "$LLAMA_BIN/llama-quantize" --imatrix $d/$m.imatrix.gguf $f32 $out Q4_K_M > $out.log 2>&1; }
}
imatrix() {
  imatrix_of smollm2-135m
  imatrix_of qwen2.5-0.5b
}

case "${1:-all}" in
  stories15m) stories15m ;;
  smollm2-135m) smollm2_135m ;;
  smollm2-360m) smollm2_360m ;;
  qwen2.5) qwen25 ;;
  gguf) gguf ;;
  wikitext) wikitext ;;
  imatrix) imatrix ;;
  all) stories15m; smollm2_135m; smollm2_360m; qwen25; gguf; wikitext ;;
  *) sed -n 2,25p "$0"; exit 1 ;;
esac
