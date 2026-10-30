#!/usr/bin/env bash
# The same benchmark on every engine of the phone: bench_suite on the CPU
# (cores 2-7), the GPU and the DSP, for each model, with the conditions
# recorded and the SoC cooled before each run: prefill tok/s, decode tok/s
# at several depths, time to first token for a 64-token prompt, peak RSS,
# and (SUSTAIN=seconds, phone unplugged, adb over Wi-Fi) tok/s over time and
# energy per token.
#
#   scripts/backends-bench.sh results/phase-l5/raw/backends [MODEL.gguf ...]
#
# Models default to the Q4_0 and Q4_K_M files of SmolLM2-135M and
# Qwen2.5-0.5B. The DSP reads Q4_0 only, so it skips the others.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir}; shift; mkdir -p "$OUT"
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(testdata/gguf/smollm2-135m-q4_0.gguf testdata/gguf/qwen2.5-0.5b-q4_0.gguf testdata/gguf/smollm2-135m-q4_k_m.gguf testdata/gguf/qwen2.5-0.5b-q4_k_m.gguf)
COOL_C=${COOL_C:-45}
SUSTAIN=${SUSTAIN:-0}
FLAGS=(--pp 64,128,256,512 --tg 64 --depth 0,256,512 --reps 3 --json --monitor)
[ "$SUSTAIN" -gt 0 ] && FLAGS+=(--sustain "$SUSTAIN")

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

cargo build -q --release --target aarch64-linux-android --features "vulkan hexagon" --example bench_suite
# Keep the screen on (screen-off lowers the clock caps); see phase1-runs.sh.
( while :; do sleep 60; adb shell taskset 1 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT

for m in "${MODELS[@]}"; do
  for b in cpu gpu dsp; do
    [ $b = dsp ] && [[ $m != *q4_0* ]] && continue
    extra=(); [ $b = cpu ] && extra=(-c 2-7)
    name="$(basename "$m" .gguf)-$b"
    wait_cool; adb shell input keyevent KEYCODE_WAKEUP
    { conditions; cargo run -q --release --target aarch64-linux-android --features "vulkan hexagon" --example bench_suite -- "$m" --backend $b "${extra[@]}" "${FLAGS[@]}" 2>&1 | grep -v pushed; } > "$OUT/$name.txt"
    echo ">>> $name"; grep -E '^(pp|tg|ttft)|RSS|sustained|energy' "$OUT/$name.txt" || tail -3 "$OUT/$name.txt"
  done
done
