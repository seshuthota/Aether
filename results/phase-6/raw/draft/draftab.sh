#!/bin/bash
# draftab.sh: SmolLM2-360M target, 135M draft; plain vs --draft-k vs --lookup.
export ANDROID_SERIAL=adb-10BF6M0R0R004FR-zfjPc2._adb-tls-connect._tcp
S=$(dirname "$0"); D=/data/local/tmp/aether; M=$D/testdata/models
exec 9>/tmp/aether-phone.lock; flock 9
adb push -q $S/aether-draft $D/aether-draft >/dev/null; adb shell chmod +x $D/aether-draft
adb shell "printf 'The history of the printing press begins' > $D/p-prose.txt"
Q=${Q:-q4_0}
for r in 1; do
 for p in code prose; do
  pf=$D/prompt-$p.txt; [ $p = prose ] && pf=$D/p-prose.txt
  for mode in plain "--draft-k 2" "--draft-k 3" "--draft-k 4" "--lookup 3"; do
   extra=""; case "$mode" in --draft*) extra="--draft $M/SmolLM2-135M $mode";; --lookup*) extra="$mode";; esac
   adb shell input keyevent KEYCODE_WAKEUP
   t=$(adb shell cat /sys/class/thermal/thermal_zone0/temp)
   out=$(adb shell "cd $D && ./aether-draft $M/SmolLM2-360M -q $Q -c 2-7 -t 0 -n 256 -i \"\$(cat $pf)\" $extra 2>&1 >/data/local/tmp/aether/o.txt; md5sum o.txt | cut -c1-8")
   echo "r$r $p [$mode] ${t}: $(echo "$out" | grep -E 'speculative|decode|^[0-9a-f]{8}$' | tr '\n' ' ')"
  done
 done
done
