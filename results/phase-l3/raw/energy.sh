#!/bin/bash
# energy.sh: 120 s sustained decode, GPU (vk_decode sustain) then CPU
# (bench_suite --sustain, cores 2-7), each starting below 42 °C.
export ANDROID_SERIAL=adb-10BF6M0R0R004FR-zfjPc2._adb-tls-connect._tcp
S=$(dirname "$0"); D=/data/local/tmp/aether
exec 9>/tmp/aether-phone.lock; flock 9
adb push -q $S/vk_decode $D/vk_decode >/dev/null; adb push -q $S/bs-cur $D/bs-cur >/dev/null
cool() { while [ "$(adb shell cat /sys/class/thermal/thermal_zone0/temp)" -gt 42000 ]; do adb shell input keyevent KEYCODE_WAKEUP; sleep 10; done; }
(while true; do sleep 40; adb shell input keyevent KEYCODE_WAKEUP; done) & W=$!
cool; echo ">>> GPU start $(adb shell cat /sys/class/thermal/thermal_zone0/temp)"
adb shell "cd $D && ./vk_decode testdata/gguf/qwen2.5-0.5b-q4_0.gguf sustain 120"
cool; echo ">>> CPU start $(adb shell cat /sys/class/thermal/thermal_zone0/temp)"
adb shell "cd $D && ./bs-cur testdata/gguf/qwen2.5-0.5b-q4_0.gguf -c 2-7 --pp 1 --tg 64 --depth 0 --reps 1 --sustain 120" | grep -v "^pp\|^tg\|ttft\|peak RSS\|^test"
kill $W
