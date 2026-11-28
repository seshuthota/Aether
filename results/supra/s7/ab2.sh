#!/system/bin/sh
# Decode A/B on the phone (aether-infer plans/supra.md, S7): binaries
# alternated old/new/new/old, decode tg128 at depths 0 and 256, each run
# after zone0 < 40 °C, screen held awake. From /data/local/tmp/aether:
#   sh ab2.sh OLD NEW "Q4_0" "1,6" "-ctk f16 -ctv f16" > out.log
OLD=$1; NEW=$2; FILES=$3; T=$4; EXTRA=$5
cool() { while [ $(cat /sys/class/thermal/thermal_zone0/temp) -ge 40000 ]; do sleep 5; input keyevent KEYCODE_WAKEUP; done; }
for q in $FILES; do
  for b in $OLD $NEW $NEW $OLD; do
    cool; input keyevent KEYCODE_WAKEUP
    echo "== $b $q $(cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_max_freq)/$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)"
    ./$b bench -m Supra2.5-65M-fixed.$q.gguf -p 0 -n 128 -d 0,256 -r 3 -t $T -C 0xfc $EXTRA 2>&1 | grep tg
  done
done
echo DONE
