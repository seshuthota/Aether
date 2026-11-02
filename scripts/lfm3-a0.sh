#!/usr/bin/env bash
# A0 baselines for plans/lfm3.md: llama.cpp's perplexity on the PC and
# llama-bench on the phone, for the official LFM3 GGUFs.
#
#   scripts/lfm3-a0.sh pc|phone|all [OUT]
#
# The PC stage also writes the BF16 logits file used by A5's KL check.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:-all}
OUT=${2:-results/phase-lfm3/raw}
mkdir -p "$OUT"

LLAMA_BIN=${LLAMA_BIN:-$HOME/llama.cpp/build/bin}
LLAMA_ANDROID=${LLAMA_ANDROID:-$HOME/llama.cpp/build-android/bin}
WIKI=testdata/wikitext/wikitext-2-raw/wiki.test.raw
REMOTE=/data/local/tmp/aether
BF16=hostdata/gguf/LFM3-1.7B-BF16.gguf
GGUF=(testdata/gguf/LFM3-1.7B-Q4_K_M.gguf testdata/gguf/LFM3-1.7B-Q8_0.gguf testdata/gguf/LFM3-1.7B-QAD-Q4_0.gguf)
PHONE_GGUF=(testdata/gguf/LFM3-1.7B-Q4_K_M.gguf testdata/gguf/LFM3-1.7B-QAD-Q4_0.gguf)

llama_perplexity() { # file, name
  echo ">>> ppl $2"
  "$LLAMA_BIN/llama-perplexity" -m "$1" -f "$WIKI" -c 512 --chunks 20 \
    > "$OUT/pc-ppl-$2.txt" 2>&1
  grep -E 'Final estimate|perplexity' "$OUT/pc-ppl-$2.txt" | tail -2 || tail -5 "$OUT/pc-ppl-$2.txt"
}

pc() {
  for m in "${GGUF[@]}"; do
    n=$(basename "$m" .gguf); llama_perplexity "$m" "${n#LFM3-1.7B-}"
  done
  echo ">>> BF16 perplexity and KL base"
  "$LLAMA_BIN/llama-perplexity" -m "$BF16" -f "$WIKI" -c 512 --chunks 20 \
    --save-all-logits "$OUT/bf16-kld-base.bin" > "$OUT/pc-ppl-BF16.txt" 2>&1
  grep -E 'Final estimate|perplexity|saving all logits' "$OUT/pc-ppl-BF16.txt" | tail -3 || tail -5 "$OUT/pc-ppl-BF16.txt"
}

wake() {
  adb shell input keyevent KEYCODE_WAKEUP
  adb shell wm dismiss-keyguard
  adb shell input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1 || true
}
wait_cool() {
  local cool=${COOL_C:-40} max
  while :; do
    max=$(adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type 2>/dev/null) in cpu*) cat $z/temp 2>/dev/null;; esac; done' | sort -n | tail -1 | tr -d '\r')
    [ "${max:-0}" -lt $((cool * 1000)) ] && return
    sleep 15
  done
}
# The vendor drops the caps to 1.78/1.02 GHz in its low-power state even when
# the screen is on; wait up to ten minutes for the performance caps to return.
wait_caps() {
  local deadline=$((SECONDS + 600)) c0 c6
  while [ $SECONDS -lt $deadline ]; do
    c0=$(adb shell cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null | tr -d '\r')
    c6=$(adb shell cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq 2>/dev/null | tr -d '\r')
    [ "${c0:-0}" -ge 2400000 ] && [ "${c6:-0}" -ge 1800000 ] && return
    sleep 15
  done
}
conditions() {
  echo "# $(date -Iseconds), commit $(git rev-parse --short HEAD)"
  adb shell dumpsys battery | grep -E 'AC powered|USB powered|Wireless powered|status:|level|temperature' | sed 's/^ */# /' | tr -d '\r' || true
  adb shell 'for c in 0 6; do echo "# cpu$c cap $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq) kHz"; done' | tr -d '\r' || true
  adb shell dumpsys display | grep -m1 mScreenState | sed 's/^ */# /' | tr -d '\r' || true
  adb shell 'head -3 /proc/meminfo; for z in /sys/class/thermal/thermal_zone*; do t=$(cat $z/type 2>/dev/null); case $t in cpu*) echo "# $t $(cat $z/temp 2>/dev/null)";; esac; done' | tr -d '\r' || true
}

phone() {
  adb push "$LLAMA_ANDROID/llama-bench" "$REMOTE/llama-bench" >/dev/null
  adb shell "mkdir -p $REMOTE/testdata/gguf"
  for m in "${PHONE_GGUF[@]}"; do
    adb push --sync "$m" "$REMOTE/testdata/gguf/" >/dev/null
  done
  wake
  ( while :; do sleep 30; adb shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; adb shell input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
  KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT
  for m in testdata/gguf/LFM3-1.7B-Q4_K_M.gguf testdata/gguf/LFM3-1.7B-QAD-Q4_0.gguf; do
    n=$(basename "$m" .gguf)
    wait_cool; wait_caps; wake
    { conditions; flock /tmp/aether-phone.lock adb shell "cd $REMOTE && taskset fc ./llama-bench -m testdata/gguf/$n.gguf -p 64,128,512 -n 64 -t 6 -r 3" 2>&1 | tr -d '\r'; } > "$OUT/phone-llamacpp-$n.txt"
    echo ">>> $n"; grep -E '^\|' "$OUT/phone-llamacpp-$n.txt" || tail -5 "$OUT/phone-llamacpp-$n.txt"
  done
}

case "$STAGE" in
  pc) pc ;;
  phone) phone ;;
  all) pc; phone ;;
  *) sed -n 2,10p "$0"; exit 1 ;;
esac
