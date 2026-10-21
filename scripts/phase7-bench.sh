#!/usr/bin/env bash
# Phase 7 benchmarks on the phone: aether's bench_suite and llama.cpp's
# llama-bench on the same GGUF files, cores and settings, then a 5-minute
# sustained decode for tokens/s over time and energy per token.
#
#   scripts/phase7-bench.sh results/phase-7/raw
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir}; mkdir -p "$OUT"
COOL_C=${COOL_C:-48}
LLAMA_BENCH=${LLAMA_BENCH:-$HOME/llama.cpp/build-android/bin/llama-bench}
REMOTE=/data/local/tmp/aether

wait_cool() {
  while :; do
    local max
    max=$(adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type 2>/dev/null) in cpu*) cat $z/temp 2>/dev/null;; esac; done' | sort -n | tail -1 | tr -d '\r')
    [ "${max:-0}" -lt $((COOL_C * 1000)) ] && return
    sleep 15
  done
}
conditions() {
  echo "# $(date -Iseconds), commit $(git rev-parse --short HEAD)"
  adb shell dumpsys battery | grep -E 'powered: true|status:|level' | sed 's/^ */# /' | tr -d '\r' || true
}
ours() { # name, args...
  local name=$1; shift
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  { conditions; cargo run -q --release --target aarch64-linux-android --example bench_suite -- "$@" 2>&1 | grep -v pushed; } > "$OUT/$name.txt"
  echo ">>> $name"; grep -E '^(pp|tg|ttft)|RSS|sustained|energy' "$OUT/$name.txt"
}
theirs() { # name, mask, model, args...
  local name=$1 mask=$2 model=$3; shift 3
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  { conditions; flock /tmp/aether-phone.lock adb shell "cd $REMOTE && taskset $mask ./llama-bench -m testdata/$model $*" 2>&1 | tr -d '\r'; } > "$OUT/$name.txt"
  echo ">>> $name"; grep -E '\| *(pp|tg)' "$OUT/$name.txt" || tail -3 "$OUT/$name.txt"
}

adb push "$LLAMA_BENCH" $REMOTE/llama-bench >/dev/null
# Keep the screen on (screen-off lowers the clock caps); see phase1-runs.sh.
( while :; do sleep 60; adb shell taskset 1 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT

for q in q4_0 q8_0; do
  M=gguf/qwen2.5-0.5b-$q.gguf
  ours "aether-qwen-$q-c2-7" testdata/$M -c 2-7
  theirs "llamacpp-qwen-$q-c2-7" fc $M -p 64,256,512 -n 64 -d 0,256,512 -t 6 -r 3
  ours "aether-qwen-$q-c6-7" testdata/$M -c 6-7
  theirs "llamacpp-qwen-$q-c6-7" c0 $M -p 64,256,512 -n 64 -d 0,256,512 -t 2 -r 3
done
ours "aether-qwen-q4_0-c2-7-sustained" testdata/gguf/qwen2.5-0.5b-q4_0.gguf -c 2-7 --pp 64 --depth 0 --sustain 300
