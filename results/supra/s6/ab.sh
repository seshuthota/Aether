#!/system/bin/sh
# S6 A/B on the phone (aether-infer plans/supra.md): decode tg128 at depth 0,
# two binaries alternating in two rounds of opposite order, each run after
# zone0 < 40 °C, screen held awake. Run on the phone from /data/local/tmp/aether:
#   sh ab.sh OLD NEW "Q4_0 Q8_0 Q4_K_M" "1,2,6"
OLD=$1; NEW=$2; FILES=$3; T=$4
cool() { while [ $(cat /sys/class/thermal/thermal_zone0/temp) -ge 40000 ]; do sleep 5; input keyevent KEYCODE_WAKEUP; done; }
for q in $FILES; do
  for order in "$OLD $NEW" "$NEW $OLD"; do
    for b in $order; do
      cool; input keyevent KEYCODE_WAKEUP
      echo "== $b $q $(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq) $(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq) $(dumpsys power | grep -o 'mWakefulness=[A-Za-z]*' | head -1) $(dumpsys battery | grep 'AC powered')"
      ./$b bench -m Supra2.5-65M-fixed.$q.gguf -p 0 -n 128 -r 3 -t $T -C 0xfc 2>&1 | grep tg
    done
  done
done
