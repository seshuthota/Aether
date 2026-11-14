#!/bin/bash
# T3 C0/C4 on the phone: f32 vs f16 cache alternated, llama-bench beside.
exec 9>/tmp/aether-phone.lock; flock 9
adb shell svc power stayon true
cool() { while :; do t=$(adb shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'); [ "$t" -lt 47000 ] && break; sleep 20; done; echo "# temp $t $(date +%T)"; }
ours() { # model kv
  cool
  echo "## aether $1 kv=$2"
  adb shell "cd /data/local/tmp/aether && toybox time -v ./aether-t3 bench -m $1 --cores 2-7 -p 512 -n 64 -d 0,512,2048,4096 -r 1 -ctk $2 -ctv $2 2>&1" | grep -E "^\| |Max RSS"
}
lcpp() { # model ctk
  cool
  echo "## llama-bench $1 ctk=$2"
  adb shell "cd /data/local/tmp/aether && LD_LIBRARY_PATH=. toybox time -v taskset fc ./llama-bench -m $1 -t 6 -fa 1 -ctk $2 -ctv $2 -p 512 -n 64 -d 0,512,2048,4096 -r 1 2>&1" | grep -E "^\| |Max RSS"
}
for m in qwen-instruct-q4_0.gguf LFM3-1.7B-Q4_K_M.gguf; do
  ours $m f32; ours $m f16
  ours $m f16; ours $m f32
  ours $m f32; ours $m f16
  lcpp $m f16; lcpp $m q8_0
done
adb shell svc power stayon false
echo "# done $(date +%T)"
