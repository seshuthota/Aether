#!/usr/bin/env bash
# DSP route: HMX encoder (aether-m4) against the CPU encoder (aether-m2), alternated.
set -u
exec 9>/tmp/aether-phone.lock; flock 9
cond() { adb shell 'dumpsys power | grep -q mWakefulness=Awake || { input keyevent KEYCODE_WAKEUP; sleep 3; }; echo "[caps $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)/$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_max_freq) temp $(cat /sys/class/thermal/thermal_zone0/temp) wake $(dumpsys power | grep -m1 mWakefulness= | cut -d= -f2)]"' 2>/dev/null; }
VL='-m LFM3-VL-2B-Q4_K_M.gguf --mmproj mmproj-LFM3-VL-2B-Q8_0.gguf --image testdata/lfm3-vl/images/scene.png -p "Describe the image in one sentence." -n 16 --temp 0 -dev dsp'
for round in 1 2 3; do
  for b in m4 m2; do
    echo "== $b $(cond)"; adb shell "cd /data/local/tmp/aether && AETHER_DSP_PROFILE=1 ./aether-$b run $VL 2>&1 | grep -E 'prefill|encode'"; sleep 15
  done
done
echo "end $(cond)"
