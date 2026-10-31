#!/usr/bin/env bash
# The plan's verification leftovers in one phone session, on battery (adb
# over Wi-Fi), screen kept on, every run cooled first and its conditions
# recorded (battery, clock caps, foreground app; `--monitor` adds the
# clocks and temperatures during the run):
#
#   energy  4.4: J/token per core set, Qwen2.5 Q4_0, SUSTAIN_S of decode each
#   bench   7.5: ours against llama-bench on the same GGUF files and cores,
#           Q4_0 / Q8_0 / Q4_K_M on cores 2-7 and 6-7, ROUNDS rounds, the
#           order reversed each round
#   memory  6.7: peak RSS per format against f32, SmolLM2 and Qwen2.5
#   sweep   4.4: Qwen2.5 Q4_0 decode and prefill per core set, in order then
#           in reverse
#
#   scripts/plan-checks.sh OUT [energy bench memory sweep]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir}; shift
STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(energy bench memory sweep)
mkdir -p "$OUT"
COOL_C=${COOL_C:-45}
ROUNDS=${ROUNDS:-2}
SUSTAIN_S=${SUSTAIN_S:-180}
REMOTE=/data/local/tmp/aether
QWEN=testdata/gguf/qwen2.5-0.5b

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
  adb shell 'for c in 0 6; do echo "# cpu$c cap $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq) kHz"; done' | tr -d '\r' || true
  adb shell dumpsys activity activities | grep -m1 -E 'topResumedActivity|mResumedActivity' | sed 's/^ */# foreground: /' | tr -d '\r' || true
  adb shell dumpsys power | grep -m1 'mWakefulness=' | sed 's/^ */# screen: /' | tr -d '\r' || true
}
ours() { # name, bench_suite args...
  local name=$1; shift
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  { conditions; cargo run -q --release --target aarch64-linux-android --example bench_suite -- "$@" --monitor 2>&1 | grep -v pushed || echo "FAILED"; } > "$OUT/$name.txt"
  echo ">>> $name: $(grep -E '^(pp|tg)[0-9]' "$OUT/$name.txt" | cut -d, -f1,2 | tr '\n' ' ')$(grep -E 'peak RSS|energy|J/token' "$OUT/$name.txt" | tr '\n' ' ')"
}
theirs() { # name, mask, threads, model
  local name=$1 mask=$2 t=$3 model=$4
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  { conditions; flock /tmp/aether-phone.lock adb shell "cd $REMOTE && taskset $mask ./llama-bench -m $model -p 64,256,512 -n 64 -d 0,256,512 -t $t -r 3" 2>&1 | tr -d '\r'; } > "$OUT/$name.txt"
  echo ">>> $name"; grep -E '\| *(pp|tg)' "$OUT/$name.txt" | awk -F'|' '{print $(NF-2), $(NF-1)}' || tail -3 "$OUT/$name.txt"
}

adb shell input keyevent KEYCODE_WAKEUP
( while :; do sleep 60; adb shell taskset 1 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT
cargo build -q --release --target aarch64-linux-android --example bench_suite

for stage in "${STAGES[@]}"; do case $stage in
energy)
  for c in 7 6-7 2-7 0-7; do
    ours "energy-qwen-q4_0-c$c" $QWEN-q4_0.gguf -c $c --pp 64 --depth 0 --tg 64 --reps 1 --sustain "$SUSTAIN_S"
  done ;;
bench)
  for r in $(seq 1 "$ROUNDS"); do
    for q in q4_0 q8_0 q4_k_m; do
      for set in "2-7 fc 6" "6-7 c0 2"; do
        read -r c mask t <<< "$set"
        if [ $((r % 2)) = 1 ]; then
          ours "aether-qwen-$q-c$c-r$r" $QWEN-$q.gguf -c $c --warmup 1
          theirs "llamacpp-qwen-$q-c$c-r$r" $mask $t $QWEN-$q.gguf
        else
          theirs "llamacpp-qwen-$q-c$c-r$r" $mask $t $QWEN-$q.gguf
          ours "aether-qwen-$q-c$c-r$r" $QWEN-$q.gguf -c $c --warmup 1
        fi
      done
    done
  done ;;
memory)
  for m in hostdata/smollm2-135m-f32.gguf testdata/gguf/smollm2-135m-{q8_0,q5_0,q4_k_m,q4_0}.gguf \
           hostdata/qwen2.5-0.5b-f32.gguf $QWEN-{q8_0,q4_k_m,q4_0}.gguf; do
    ours "memory-$(basename "$m" .gguf)" "$m" -c 2-7 --pp 64 --depth 0 --tg 64 --reps 3
  done ;;
sweep)
  SETS=(7 6-7 5-7 4-7 3-7 2-7 1-7 0-7 0 0-1 2-5 0-5)
  for pass in 1 2; do
    order=("${SETS[@]}")
    [ $pass = 2 ] && order=($(printf '%s\n' "${SETS[@]}" | tac))
    for c in "${order[@]}"; do
      ours "sweep-qwen-q4_0-c$c-p$pass" $QWEN-q4_0.gguf -c "$c" --pp 64 --depth 0 --tg 64 --reps 3
    done
  done ;;
esac; done
echo "# done $(date -Iseconds)"
