#!/system/bin/sh
# f16 KV after merging kv-f16 into supra (plans/supra.md, S6): decode tg128 at
# depths 0 and 256, the supra build (f32) against the merged build's f32 and
# f16 caches, two rounds in opposite orders, each run after zone0 < 40 °C,
# screen held awake. On the phone, from /data/local/tmp/aether:
#   sh kv.sh "Q4_0 Q8_0" > kv.log; echo DONE >> kv.log
FILES=$1
cool() { while [ $(cat /sys/class/thermal/thermal_zone0/temp) -ge 40000 ]; do sleep 5; input keyevent KEYCODE_WAKEUP; done; }
run() {
  cool; input keyevent KEYCODE_WAKEUP
  echo "== $1 $2 $q $(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq)/$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq) $(dumpsys power | grep -o 'mWakefulness=[A-Za-z]*' | head -1)"
  ./$1 bench -m Supra2.5-65M-fixed.$q.gguf -p 0 -n 128 -d 0,256 -r 3 -t 1,6 -C 0xfc $3 2>&1 | grep tg
}
for q in $FILES; do
  run aether-s3 f32 ""
  run aether-merged f32 ""
  run aether-merged f16 "-ctk f16 -ctv f16"
  run aether-merged f16 "-ctk f16 -ctv f16"
  run aether-merged f32 ""
  run aether-s3 f32 ""
done
