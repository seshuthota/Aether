#!/usr/bin/env bash
# Prefill A/B on the phone: this tree's bench_suite against a baseline build
# (another worktree's bench_suite binary) and llama.cpp's llama-bench, run
# alternately on the same GGUF files and cores, ROUNDS rounds, each run
# started below COOL_C on the hottest CPU zone.
#
#   scripts/prefill-ab.sh OUT_DIR BASELINE_BENCH_SUITE_BINARY
#
# NEW=BINARY tests another build instead of this tree's (e.g. an older
# commit's, to measure one change at a time). BASELINE "-" skips the
# baseline (this build against llama.cpp only). FORMATS lists the
# testdata/gguf/qwen2.5-0.5b-<format>.gguf files (default "q4_0 q8_0").
#
# The tests run are bench_suite's ARGS and llama-bench's LLAMA_ARGS; the
# defaults are prefill. For decode at depth, for example:
#
#   ARGS="-c 2-7 --pp 1 --tg 64 --depth 0,512" LLAMA_ARGS="-p 0 -n 64 -d 0,512" \
#     scripts/prefill-ab.sh OUT_DIR BASELINE
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir}; BASE=${2:?baseline bench_suite binary}; mkdir -p "$OUT"
COOL_C=${COOL_C:-48}; ROUNDS=${ROUNDS:-2}; CORES=${CORES:-2-7}; MASK=${MASK:-fc}; THREADS=${THREADS:-6}
LLAMA_BENCH=${LLAMA_BENCH:-$HOME/llama.cpp/build-android/bin/llama-bench}
REMOTE=/data/local/tmp/aether
read -r -a ARGS <<< "${ARGS:--c $CORES --pp 64,256,512 --tg 16 --depth 0}"
LLAMA_ARGS=${LLAMA_ARGS:--p 64,256,512 -n 0}

max_temp() {
  adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type 2>/dev/null) in cpu*) cat $z/temp 2>/dev/null;; esac; done' | sort -n | tail -1 | tr -d '\r'
}
wait_cool() {
  while [ "$(max_temp)" -ge $((COOL_C * 1000)) ]; do sleep 15; done
}
conditions() {
  echo "# $(date -Iseconds), commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src || echo '+uncommitted'), start $(max_temp) m°C"
  adb shell dumpsys battery | grep -E 'powered: true|status:|level' | sed 's/^ */# /' | tr -d '\r' || true
}
# The clock caps (scaling_max_freq, MHz) of the performance and prime
# cores, sampled every 0.5 s on the phone until stop_caps; caps_seen prints
# the lowest of each. Thermal limits lower them partway through some runs
# (prime 1.69 -> 1.02 GHz seen), which moves timings by 20 %.
CAPS=$REMOTE/caps.txt
start_caps() {
  local f=/sys/devices/system/cpu/cpu
  # setsid: a plain background job keeps `adb shell` from returning.
  adb shell "rm -f $CAPS $CAPS.stop; setsid sh -c 'while [ ! -e $CAPS.stop ]; do echo \$(cat ${f}2/cpufreq/scaling_max_freq ${f}6/cpufreq/scaling_max_freq) >> $CAPS; sleep 0.5; done' </dev/null >/dev/null 2>&1 &"
}
caps_seen() {
  adb shell "touch $CAPS.stop; sleep 0.6; sort -n -k1,1 $CAPS | head -1 | cut -d' ' -f1; sort -n -k2,2 $CAPS | head -1 | cut -d' ' -f2" | tr -d '\r' | tr '\n' ' ' | awk '{printf "# lowest clock caps: perf %d MHz, prime %d MHz\n", $1 / 1000, $2 / 1000}'
}
run() { # name, command...
  local name=$1; shift
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  start_caps
  { conditions; "$@" 2>&1 | grep -v pushed | tr -d '\r'; caps_seen; } > "$OUT/$name.txt"
  echo ">>> $name $(tail -1 "$OUT/$name.txt" | sed 's/# lowest clock caps: //')"; grep -E '^(pp|tg)|\| *(pp|tg)' "$OUT/$name.txt" || tail -3 "$OUT/$name.txt"
}

# NEW: the build under test (default: this tree's, built now).
if [ -z "${NEW:-}" ]; then
  cargo build -q --release --target aarch64-linux-android --example bench_suite
  NEW=target/aarch64-linux-android/release/examples/bench_suite
fi
adb push "$LLAMA_BENCH" $REMOTE/llama-bench >/dev/null
# Keep the screen on (screen-off lowers the clock caps); see phase1-runs.sh.
( while :; do sleep 60; adb shell taskset 1 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT

for r in $(seq 1 "$ROUNDS"); do
  for q in ${FORMATS:-q4_0 q8_0}; do
    M=gguf/qwen2.5-0.5b-$q.gguf
    run "new-$q-r$r" scripts/adb-run.sh "$NEW" testdata/$M "${ARGS[@]}"
    [ "$BASE" = - ] || run "base-$q-r$r" scripts/adb-run.sh "$BASE" testdata/$M "${ARGS[@]}"
    run "llamacpp-$q-r$r" flock /tmp/aether-phone.lock adb shell "cd $REMOTE && taskset $MASK ./llama-bench -m testdata/$M $LLAMA_ARGS -t $THREADS -r 3"
  done
done
