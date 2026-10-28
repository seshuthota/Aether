#!/usr/bin/env bash
# Record a Perfetto system trace of a bench_suite run on the phone: which
# thread ran on which core and when (sched), CPU clocks, idle states, and
# thermal/power counters. Output: <OUT>.pftrace, open it at ui.perfetto.dev.
#
#   scripts/perfetto-trace.sh results/phase-N/raw/trace-q4-c2-7 \
#       testdata/gguf/qwen2.5-0.5b-q4_0.gguf -c 2-7 --warmup 1 --reps 3
#
# Everything after OUT is passed to bench_suite (plus --json --monitor).
# TRACE_S (default 60) is the longest the trace records; the script stops it
# when the benchmark ends.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output path without extension}; shift
TRACE_S=${TRACE_S:-60}
REMOTE_TRACE=/data/misc/perfetto-traces/aether.pftrace
mkdir -p "$(dirname "$OUT")"

adb shell rm -f "$REMOTE_TRACE"
# --background returns at once; the trace stops after duration_ms or on kill.
adb shell perfetto --background --txt -c - -o "$REMOTE_TRACE" <<EOF >/dev/null
duration_ms: $((TRACE_S * 1000))
buffers { size_kb: 65536 fill_policy: RING_BUFFER }
data_sources {
  config {
    name: "linux.ftrace"
    ftrace_config {
      ftrace_events: "sched/sched_switch"
      ftrace_events: "sched/sched_waking"
      ftrace_events: "sched/sched_process_exit"
      ftrace_events: "power/cpu_frequency"
      ftrace_events: "power/cpu_idle"
      ftrace_events: "power/suspend_resume"
      ftrace_events: "thermal/thermal_temperature"
    }
  }
}
data_sources { config { name: "linux.process_stats" process_stats_config { scan_all_processes_on_start: true proc_stats_poll_ms: 500 } } }
data_sources { config { name: "linux.sys_stats" sys_stats_config { cpufreq_period_ms: 100 stat_period_ms: 500 stat_counters: STAT_CPU_TIMES } } }
data_sources { config { name: "android.power" android_power_config { battery_poll_ms: 1000 battery_counters: BATTERY_COUNTER_CAPACITY_PERCENT battery_counters: BATTERY_COUNTER_CURRENT } } }
EOF

cargo run -q --release --target aarch64-linux-android --example bench_suite -- "$@" --json --monitor 2>&1 \
  | grep -v pushed | tee "$OUT.txt" || true

# Stop the trace (flushes the buffer to the file), then pull it.
adb shell 'pkill -TERM perfetto' || true
sleep 2
adb pull "$REMOTE_TRACE" "$OUT.pftrace"
adb shell rm -f "$REMOTE_TRACE"
echo "trace: $OUT.pftrace  (open at https://ui.perfetto.dev)"
