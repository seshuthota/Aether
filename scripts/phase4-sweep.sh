#!/usr/bin/env bash
# Phase 4 clean sweep: decode and prefill speed against thread count and core
# set, interrupts per CPU during each run, and energy per token for a few core
# sets over 5 minutes. Output: one file per run in OUT, plus interrupts.csv.
#
#   scripts/phase4-sweep.sh results/phase-4/raw
#
# Close every app and stop any video on the phone first, and leave it on
# battery. The script records the foreground app and any playing media
# session before every run, so a disturbed run can be spotted afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir}; mkdir -p "$OUT"
COOL_C=${COOL_C:-48}
SUSTAIN_S=${SUSTAIN_S:-300}
SMOL=testdata/models/SmolLM2-135M
QWEN=testdata/gguf/qwen2.5-0.5b-q4_0.gguf
# One prime core, then performance cores added one at a time from the top;
# then performance cores alone, and the sets that do or do not include 0-1.
SETS=(7 6-7 5-7 4-7 3-7 2-7 1-7 0-7 0 0-1 2-5 0-5)

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
  adb shell dumpsys media_session | grep -c 'state=PlaybackState {state=3' | sed 's/^/# media sessions playing: /' | tr -d '\r' || true
}
# Total interrupts per CPU (columns CPU0..CPU7 of /proc/interrupts), one line.
irq_totals() {
  { adb shell cat /proc/interrupts 2>/dev/null || true; } | tr -d '\r' | awk '
    NR == 1 { n = NF; next }
    { for (i = 2; i <= n + 1; i++) if ($i ~ /^[0-9]+$/) s[i - 2] += $i }
    END { for (i = 0; i < n; i++) printf "%s%d", (i ? " " : ""), s[i]; print "" }'
}
irq_delta() { # label, before, after -> interrupts.csv
  local label=$1; local -a a=($2) b=($3); local line="$label"
  for i in "${!a[@]}"; do line+=",$(( b[i] - a[i] ))"; done
  echo "$line" >> "$OUT/interrupts.csv"
}
run() { # name, bench_suite args...
  local name=$1; shift
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  local before after
  before=$(irq_totals)
  { conditions; cargo run -q --release --target aarch64-linux-android --example bench_suite -- "$@" 2>&1 | grep -v pushed || echo "FAILED"; } > "$OUT/$name.txt"
  after=$(irq_totals)
  irq_delta "$name" "$before" "$after"
  echo ">>> $name: $(grep -E '^(pp|tg)' "$OUT/$name.txt" | tr '\n' ' ')$(grep -E 'energy per token' "$OUT/$name.txt" || true)"
}

# Keep the screen on (screen-off lowers the clock caps); see phase1-runs.sh.
( while :; do sleep 60; adb shell taskset 1 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT

echo "run,cpu0,cpu1,cpu2,cpu3,cpu4,cpu5,cpu6,cpu7" > "$OUT/interrupts.csv"
cargo build -q --release --target aarch64-linux-android --example bench_suite --example pool_handoff

# Interrupts with nothing of ours running, as the baseline for the hypothesis
# that Android sends most of them to cores 0-1.
wait_cool
b=$(irq_totals); sleep 60; irq_delta "idle-60s" "$b" "$(irq_totals)"

# The cost of handing out one job, the floor under every parallel op.
{ conditions; for c in 6-7 4-7 2-7 0-7; do wait_cool
    cargo run -q --release --target aarch64-linux-android --example pool_handoff -- "$c" 2>&1 | grep -v pushed || echo "FAILED $c"
  done; } > "$OUT/pool-handoff.txt"
grep -v '^#' "$OUT/pool-handoff.txt"

# Speed: every set in order, then in reverse, so slow drift (heat, caps)
# falls on all sets alike.
for pass in 1 2; do
  order=("${SETS[@]}")
  [ $pass = 2 ] && order=($(printf '%s\n' "${SETS[@]}" | tac))
  for c in "${order[@]}"; do
    run "smol-q4_0-c$c-p$pass" "$SMOL" -q q4_0 -c "$c" --pp 64 --depth 0 --tg 64 --reps 3
  done
done
for pass in 1 2; do
  order=(6-7 2-7 0-7 0-5)
  [ $pass = 2 ] && order=(0-5 0-7 2-7 6-7)
  for c in "${order[@]}"; do
    run "qwen-q4_0-c$c-p$pass" "$QWEN" -c "$c" --pp 64 --depth 0,512 --tg 64 --reps 3
  done
done

# Energy per token: minutes of decoding per core set.
for c in 7 6-7 2-7 0-7; do
  run "smol-q4_0-c$c-sustained" "$SMOL" -q q4_0 -c "$c" --pp 64 --depth 0 --tg 64 --reps 1 --sustain "$SUSTAIN_S"
done
