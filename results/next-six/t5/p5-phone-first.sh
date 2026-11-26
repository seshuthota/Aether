#!/usr/bin/env bash
# T5 P5: GPU decode A/B (new vs old paths) and CPU vs GPU decode, phone, battery, screen awake.
set -u
# Run from aether-infer (branch gpu-decode).
OUT=../Aether/results/next-six/t5/p5; mkdir -p "$OUT"
cond() { adb shell 'echo "zone0 $(cat /sys/class/thermal/thermal_zone0/temp) cpu0 $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) cpu6 $(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq) gpu $(cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null) wake $(dumpsys power 2>/dev/null | grep -m1 mWakefulness= | cut -d= -f2)"; dumpsys battery | grep -E "AC powered|level"'; }
cool() { while t=$(adb shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'); [ "${t:-99999}" -gt 42000 ]; do adb shell input keyevent KEYCODE_WAKEUP; sleep 20; done; }
ab() { # name model flags...
  local name=$1 model=$2; shift 2
  cool; { cond; cargo run -q --release --target aarch64-linux-android -p aether-bench --features vulkan --example vk_gemma_decode -- "testdata/gguf/$model.gguf" 0,512,2048 32 "$@"; cond; } > "$OUT/$name.log" 2>&1
  grep -E "^[0-9]+," "$OUT/$name.log" | sed "s/^/$name,/"
}
bench() { # name model dev
  local name=$1 model=$2 dev=$3
  cool; { cond; adb shell "cd /data/local/tmp/aether && ./aether-gd bench -m testdata/gguf/$model.gguf -p 0 -n 32 -d 0,512,2048 -r 2 --cores 2-7 -dev $dev -o csv"; cond; } > "$OUT/$name.log" 2>&1
  grep -vE "^(zone0|  )" "$OUT/$name.log" | sed "s/^/$name,/"
}
for round in 1 2; do
  ab gemma-new-$round gemma-4-E2B-it-Q4_0
  ab gemma-old-$round gemma-4-E2B-it-Q4_0 per-head unsplit
  ab lfm3-new-$round LFM3-1.7B-Q4_K_M
  ab lfm3-f32dec-$round LFM3-1.7B-Q4_K_M dec-f32
  ab lfm3-old-$round LFM3-1.7B-Q4_K_M per-head unsplit dec-f32
  for m in gemma-4-E2B-it-Q4_0 LFM3-1.7B-Q4_K_M qwen2.5-0.5b-q4_0; do
    bench $m-cpu-$round $m cpu
    bench $m-gpu-$round $m gpu
  done
done
echo "end: $(cond)"
