#!/system/bin/sh
# LFM3-1.7B decode A/B (aether-infer plans/supra.md, S7): OLD/NEW/NEW/OLD,
# tg64 at depth 0, 6 threads, cooled below 40 °C before each run.
OLD=$1; NEW=$2
cool() { while [ $(cat /sys/class/thermal/thermal_zone0/temp) -ge 40000 ]; do sleep 5; input keyevent KEYCODE_WAKEUP; done; }
for b in $OLD $NEW $NEW $OLD; do
  cool; input keyevent KEYCODE_WAKEUP
  echo "== $b LFM3 $(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq)/$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)"
  ./$b bench -m LFM3-1.7B-Q4_K_M.gguf -p 0 -n 64 -r 3 -t 6 -C 0xfc 2>&1 | grep tg
done
echo DONE
