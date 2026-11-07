#!/bin/bash
# P3 leftover: bench/migrate-check.sh on the phone, main's aether (e5e0bbf) against llama.cpp d02b4c0d6.
S=$(cd "$(dirname "$0")" && pwd)  # the adb wrappers beside this script
cd /home/curious/Documents/hermes-projects/aether-infer
exec 9>/tmp/aether-phone.lock; flock 9
hottest(){ adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in cpu*) cat $z/temp;; esac; done' | sort -n | tail -1 | awk '{print int($1/1000)}'; }
t=$(hottest); while ((t >= 45)); do sleep 15; t=$(hottest); done
echo "date: $(date -u +%FT%TZ), start $t °C, aether main $(git rev-parse --short HEAD), llama.cpp d02b4c0d6, -t 6"
adb push target-phone/aarch64-linux-android/release/aether /data/local/tmp/aether/aether-main >/dev/null
M=/data/local/tmp/aether/testdata/gguf; W=/data/local/tmp/aether/testdata/wikitext/wikitext-2-raw/wiki.test.raw
for m in qwen2.5-0.5b-q4_0 qwen2.5-0.5b-q4_k_m; do
  echo "######## $m"
  LLAMA=$S/phone-llama AETHER=$S/phone-aether/aether THREADS=6 bench/migrate-check.sh $M/$m.gguf "" $W
done
adb shell rm /data/local/tmp/aether/aether-main
echo "end $(hottest) °C"
