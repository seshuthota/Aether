#!/usr/bin/env bash
# Runs every Phase 1 measurement on the phone and saves the raw output under
# results/phase-1/raw/. Each run goes through the cargo runner (and its lock).
#
#   scripts/phase1-runs.sh sweeps      # cache/DRAM sweeps, alignment, kernels
#   scripts/phase1-runs.sh sustained   # 5-minute all-core run (heats the phone)
#   scripts/phase1-runs.sh energy      # prime vs performance J/GB; UNPLUGGED,
#                                      # adb over Wi-Fi, screen off
#
# Before each run: phone idle and cool (the script waits for the hottest CPU
# zone to drop below COOL_C), screen state noted in results/phase-1.md.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/phase-1/raw
mkdir -p "$OUT"
COOL_C=${COOL_C:-40}
RUN=(cargo run -q --release --target aarch64-linux-android --example)

# Waits until every readable CPU thermal zone is below COOL_C degrees.
wait_cool() {
  while :; do
    local max
    max=$(adb shell 'for z in /sys/class/thermal/thermal_zone*; do
             case $(cat $z/type 2>/dev/null) in *cpu*) cat $z/temp 2>/dev/null;; esac
           done' | sort -n | tail -1 | tr -d '\r')
    [ -z "$max" ] && { echo "no CPU thermal zones readable; not waiting"; return; }
    [ "$max" -lt $((COOL_C * 1000)) ] && { echo "cool: $((max / 1000)) °C"; return; }
    echo "waiting to cool: $((max / 1000)) °C (target < $COOL_C)"; sleep 20
  done
}

# Records what the binary cannot see: screen, charging, the full battery dump.
conditions() {
  {
    echo "# host time $(date -Iseconds), commit $(git rev-parse --short HEAD)"
    adb shell dumpsys power | grep -m1 -E 'mWakefulness=' | sed 's/^ */# /'
    adb shell dumpsys battery | grep -E 'powered|status|level|temperature' | sed 's/^ */# /'
  } | tr -d '\r'
}

run() { # name, example, args...
  local name=$1 ex=$2; shift 2
  wait_cool
  echo ">>> $name"
  { conditions; "${RUN[@]}" "$ex" -- "$@"; } > "$OUT/$name.csv"
  tail -1 "$OUT/$name.csv"
}

case "${1:-}" in
  sweeps)
    run sweep-read-core7      bw_sweep --kernel read --cores 7
    run sweep-read-core0      bw_sweep --kernel read --cores 0
    run sweep-read-perf6-par  bw_sweep --kernel read --cores 0-5 --parallel --min-kb 64
    run sweep-read-prime2-par bw_sweep --kernel read --cores 6-7 --parallel --min-kb 64
    run sweep-read-all8-par   bw_sweep --kernel read --cores 0-7 --parallel --min-kb 64
    run sweep-read-core7-off1 bw_sweep --kernel read --cores 7 --offset 1
    for k in copy scale add; do
      run "sweep-$k-core7" bw_sweep --kernel "$k" --cores 7
    done
    ;;
  sustained)
    run sustained-all8 bw_sustained --cores 0-7 --parallel --seconds 300
    ;;
  energy)
    adb shell dumpsys battery | grep -q 'USB powered: true' && { echo "phone is on USB power: unplug and use adb over Wi-Fi"; exit 1; }
    run energy-core7 bw_sustained --cores 7 --seconds 30 --idle-s 30
    run energy-core0 bw_sustained --cores 0 --seconds 30 --idle-s 30
    ;;
  *) echo "usage: $0 sweeps|sustained|energy"; exit 2 ;;
esac
