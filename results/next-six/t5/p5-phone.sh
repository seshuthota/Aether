#!/usr/bin/env bash
# T5 P5, the runs the Wi-Fi drops lost: each started detached on the phone
# (nohup, output to a file there), polled, pulled; adb reconnected over mDNS.
set -u
# Run from aether-infer (branch gpu-decode).
# ANDROID_SERIAL: the phone (USB).
OUT=../Aether/results/next-six/t5/p5; mkdir -p "$OUT"
D=/data/local/tmp/aether
ser() { echo "$ANDROID_SERIAL"; }
A() { ANDROID_SERIAL=$(ser) timeout 60 adb "$@"; }
reconnect() { until A shell true >/dev/null 2>&1; do sleep 10; done; }
cond() { reconnect; A shell 'echo "zone0 $(cat /sys/class/thermal/thermal_zone0/temp) cpu0 $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) cpu6 $(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq) gpu $(cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null) wake $(dumpsys power 2>/dev/null | grep -m1 mWakefulness= | cut -d= -f2) $(dumpsys battery | grep -E "AC powered|level" | tr -d "\n")"'; }
cool() { while reconnect; t=$(A shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'); [ "${t:-99999}" -gt 42000 ]; do A shell input keyevent KEYCODE_WAKEUP; sleep 20; done; }
run() { # name command (run in $D on the phone)
  local name=$1 cmd=$2
  cool
  { echo "# $cmd"; cond; } > "$OUT/$name.log"
  reconnect; A shell "cd $D && rm -f p5/$name.done && mkdir -p p5 && nohup sh -c '$cmd > p5/$name.out 2>&1; echo \$? > p5/$name.done' >/dev/null 2>&1 &"
  until reconnect; A shell "test -f $D/p5/$name.done" >/dev/null 2>&1; do sleep 20; done
  A shell "cat $D/p5/$name.out" >> "$OUT/$name.log"
  cond >> "$OUT/$name.log"
  echo "== $name: $(grep -cE '^[0-9]+,|^"' "$OUT/$name.log") rows, exit $(A shell cat $D/p5/$name.done | tr -d '\r')"
}
reconnect
A push target/aarch64-linux-android/release/examples/vk_gemma_decode $D/vk_gemma_decode-gd >/dev/null
G=testdata/gguf/gemma-4-E2B-it-Q4_0.gguf; L=testdata/gguf/LFM3-1.7B-Q4_K_M.gguf; Q=testdata/gguf/qwen2.5-0.5b-q4_0.gguf
ab() { echo "./vk_gemma_decode-gd $1 0,512,2048 32 ${2:-}"; }
bench() { echo "./aether-gd bench -m $1 -p 0 -n 32 -d 0,512,2048 -r 2 --cores 2-7 -dev $2 -o csv"; }
run gemma-old-1 "$(ab $G 'per-head unsplit')"
run gemma-new-2 "$(ab $G)"
run lfm3-new-1 "$(ab $L)"
run lfm3-old-1 "$(ab $L 'per-head unsplit dec-f32')"
run lfm3-f32dec-2 "$(ab $L dec-f32)"
for round in 1 2; do
  for m in G L; do
    f=${!m}; n=$(basename $f .gguf)
    run $n-cpu-$round "$(bench $f cpu)"
    run $n-gpu-$round "$(bench $f gpu)"
  done
done
run qwen2.5-0.5b-q4_0-cpu-2 "$(bench $Q cpu)"
run qwen2.5-0.5b-q4_0-gpu-2 "$(bench $Q gpu)"
echo "end: $(cond)"
