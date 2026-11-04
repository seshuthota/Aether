#!/usr/bin/env bash
# A7 phone measurements for plans/lfm3.md: aether against llama.cpp in the
# same session, Q4_K_M and the QAD Q4_0 file, on battery, screen on, cooled.
#
#   scripts/lfm3-a7.sh [OUT]
#
# Round 2 reverses the order. Also runs a two-minute sustained decode with
# the monitor (energy/temperature) and leaves a simpleperf command in the
# output for the profile run.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-results/phase-lfm3/raw/a7}
mkdir -p "$OUT"
COOL_C=${COOL_C:-45}
ROUNDS=${ROUNDS:-2}
REMOTE=/data/local/tmp/aether
LLAMA_BENCH=${LLAMA_BENCH:-$HOME/llama.cpp/build-android/bin/llama-bench}
MODELS=(LFM3-1.7B-Q4_K_M LFM3-1.7B-QAD-Q4_0)

wait_cool() {
  while :; do
    local max
    max=$(adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type 2>/dev/null) in cpu*) cat $z/temp 2>/dev/null;; esac; done' | sort -n | tail -1 | tr -d '\r')
    [ "${max:-0}" -lt $((COOL_C * 1000)) ] && return
    sleep 15
  done
}
wake() {
  adb shell input keyevent KEYCODE_WAKEUP
  adb shell wm dismiss-keyguard
  adb shell input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1 || true
}
conditions() {
  echo "# $(date -Iseconds), commit $(git rev-parse --short HEAD)"
  adb shell dumpsys battery | grep -E 'AC powered|USB powered|Wireless powered|status:|level|temperature' | sed 's/^ */# /' | tr -d '\r' || true
  adb shell 'for c in 0 6; do echo "# cpu$c cap $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq) kHz"; done' | tr -d '\r' || true
  adb shell dumpsys display | grep -m1 mScreenState | sed 's/^ */# /' | tr -d '\r' || true
  adb shell 'head -3 /proc/meminfo' | tr -d '\r' || true
}
ours() { # name, model, extra...
  local name=$1 model=$2; shift 2
  wait_cool; wake
  { conditions; cargo run -q --release --target aarch64-linux-android --example bench_suite -- "testdata/gguf/$model.gguf" -c 2-7 --pp 64,128,512 --tg 64 --reps 3 --json --monitor "$@" 2>&1 | grep -v pushed; } > "$OUT/$name.txt"
  echo ">>> $name"
}
theirs() { # name, model, extra...
  local name=$1 model=$2; shift 2
  wait_cool; wake
  { conditions; flock /tmp/aether-phone.lock adb shell "cd $REMOTE && taskset fc ./llama-bench -m testdata/gguf/$model.gguf -p 64,128,512 -n 64 -t 6 -r 3 $*" 2>&1 | tr -d '\r'; } > "$OUT/$name.txt"
  echo ">>> $name"
}
order() { if [ $(($1 % 2)) = 1 ]; then shift; echo "$@"; else shift; echo "$@" | tr ' ' '\n' | tac | tr '\n' ' '; fi; }

adb push "$LLAMA_BENCH" "$REMOTE/llama-bench" >/dev/null
adb shell "mkdir -p $REMOTE/testdata/gguf"
for m in "${MODELS[@]}"; do adb push --sync "testdata/gguf/$m.gguf" "$REMOTE/testdata/gguf/" >/dev/null; done
adb shell settings put system screen_off_timeout 1800000
trap 'adb shell settings put system screen_off_timeout 30000' EXIT
wake
( while :; do sleep 30; adb shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; adb shell input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null; adb shell settings put system screen_off_timeout 30000' EXIT

for r in $(seq 1 "$ROUNDS"); do
  for m in "${MODELS[@]}"; do
    for who in $(order "$r" ours theirs); do
      if [ "$who" = ours ]; then ours "r$r-ours-$m" "$m"; else theirs "r$r-llama-$m" "$m"; fi
    done
  done
done

# Two minutes of sustained decode, with power, clocks and temperature.
for m in "${MODELS[@]}"; do
  ours "sustain-$m" "$m" --sustain 120
done

# Profile one decode with simpleperf when it is installed; the command is
# recorded for a manual run if not.
echo "# simpleperf: adb shell simpleperf record -g -o /data/local/tmp/aether/perf.data -- $REMOTE/aether ..." > "$OUT/simpleperf.txt"
