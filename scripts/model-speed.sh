#!/usr/bin/env bash
# Prefill and decode speed of every model × weight format × core set on the
# phone, with the conditions of each run. Output: results/<OUT>.
#
#   scripts/model-speed.sh results/phase-6/raw/model-speed.txt
#
# Each run: greedy decoding (-t 0) of a fixed prompt, so every format does
# the same amount of work; the phone is woken and allowed to cool first.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output file}
COOL_C=${COOL_C:-48}
mkdir -p "$(dirname "$OUT")"

PROMPT_SHORT="Once upon a time"
PROMPT_LONG="The history of the city begins in the early medieval period, when a small settlement grew up around a river crossing. Over the following centuries the town became a center of trade, attracting merchants from across the region, and its markets, guild halls and churches still shape the streets today. In the nineteenth century the arrival of the railway"

wait_cool() {
  while :; do
    local max
    max=$(adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type 2>/dev/null) in cpu*) cat $z/temp 2>/dev/null;; esac; done' | sort -n | tail -1 | tr -d '\r')
    [ "${max:-0}" -lt $((COOL_C * 1000)) ] && return
    sleep 15
  done
}

{
  echo "# model-speed $(date -Iseconds), commit $(git rev-parse --short HEAD)"
  adb shell dumpsys battery | grep -E 'powered: true|status|level' | sed 's/^ */# /' | tr -d '\r' || true
  adb shell 'for c in 0 6; do echo "# cpu$c cap $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq) kHz"; done' | tr -d '\r' || true
} > "$OUT"

run() { # label, model args...
  local label=$1; shift
  wait_cool
  adb shell input keyevent KEYCODE_WAKEUP
  local line
  line=$(cargo run -q --release --target aarch64-linux-android -- "$@" 2>&1 >/dev/null | grep -E '^prefill' || echo "FAILED")
  echo "$label | $line" | tee -a "$OUT"
}

S15=testdata/models/stories15M/stories15M.bin
SMOL=testdata/models/SmolLM2-135M
for q in f32 q8_0 q4_0; do
  for cores in 7 6-7 0-7; do
    run "stories15M $q cores $cores" "$S15" -t 0 -n 256 -i "$PROMPT_SHORT" -q "$q" -c "$cores"
    run "SmolLM2-135M $q cores $cores" "$SMOL" -t 0 -n 256 -i "$PROMPT_LONG" -q "$q" -c "$cores"
  done
done
