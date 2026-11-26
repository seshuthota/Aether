#!/usr/bin/env bash
# LFM3-VL image chat on the phone: CPU / DSP (HMX encoder) / each with the GPU encoder.
set -u
exec 9>/tmp/aether-phone.lock; flock 9
cond() { adb shell 'dumpsys power | grep -q mWakefulness=Awake || { input keyevent KEYCODE_WAKEUP; sleep 3; }; echo "[caps $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)/$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq) temp $(cat /sys/class/thermal/thermal_zone0/temp) wake $(dumpsys power | grep -m1 mWakefulness= | cut -d= -f2) ac $(dumpsys battery | grep -m1 "AC powered" | cut -d: -f2)]"'; }
R='cd /data/local/tmp/aether && AETHER_DSP_PROFILE=1 ./aether-m4'
VL='-m LFM3-VL-2B-Q4_K_M.gguf --mmproj mmproj-LFM3-VL-2B-Q8_0.gguf --image testdata/lfm3-vl/images/scene.png -p "Describe the image in one sentence." -n 16 --temp 0'
for round in 1 2; do
  for dev in cpu dsp "cpu --mmproj-gpu" "dsp --mmproj-gpu"; do
    echo "== VL $dev $(cond)"; adb shell "$R run $VL -dev $dev 2>&1 | grep -E 'prefill|vision encoder|eval|^ *[A-Z]' | tail -6"; sleep 10
  done
done
echo "end $(cond)"
