#!/usr/bin/env bash
# Prefill A/B on the phone: this tree's bench_suite against a baseline build
# (another worktree's bench_suite binary) and llama.cpp's llama-bench, run
# alternately on the same GGUF files and cores, ROUNDS rounds, each run
# started below COOL_C on the hottest CPU zone.
#
#   scripts/prefill-ab.sh OUT_DIR BASELINE_BENCH_SUITE_BINARY
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir}; BASE=${2:?baseline bench_suite binary}; mkdir -p "$OUT"
COOL_C=${COOL_C:-48}; ROUNDS=${ROUNDS:-2}; CORES=${CORES:-2-7}; MASK=${MASK:-fc}; THREADS=${THREADS:-6}
LLAMA_BENCH=${LLAMA_BENCH:-$HOME/llama.cpp/build-android/bin/llama-bench}
REMOTE=/data/local/tmp/aether
ARGS=(-c "$CORES" --pp 64,256,512 --tg 16 --depth 0)

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
run() { # name, command...
  local name=$1; shift
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  { conditions; "$@" 2>&1 | grep -v pushed | tr -d '\r'; } > "$OUT/$name.txt"
  echo ">>> $name"; grep -E '^pp|\| *pp' "$OUT/$name.txt" || tail -3 "$OUT/$name.txt"
}

cargo build -q --release --target aarch64-linux-android --example bench_suite
NEW=target/aarch64-linux-android/release/examples/bench_suite
adb push "$LLAMA_BENCH" $REMOTE/llama-bench >/dev/null
# Keep the screen on (screen-off lowers the clock caps); see phase1-runs.sh.
( while :; do sleep 60; adb shell taskset 1 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT

for r in $(seq 1 "$ROUNDS"); do
  for q in q4_0 q8_0; do
    M=gguf/qwen2.5-0.5b-$q.gguf
    run "new-$q-r$r" scripts/adb-run.sh "$NEW" testdata/$M "${ARGS[@]}"
    run "base-$q-r$r" scripts/adb-run.sh "$BASE" testdata/$M "${ARGS[@]}"
    run "llamacpp-$q-r$r" flock /tmp/aether-phone.lock adb shell "cd $REMOTE && taskset $MASK ./llama-bench -m testdata/$M -p 64,256,512 -n 0 -t $THREADS -r 3"
  done
done
