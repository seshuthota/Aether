#!/bin/bash
# Gemma 4 E2B prompt and decode speed on the CPU (cores 2-7, 6 threads): Aether, llama.cpp, LiteRT-LM.
exec 9>/tmp/aether-phone.lock; flock 9
adb shell svc power stayon true
cool() { while :; do t=$(adb shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'); [ "$t" -lt 45000 ] && break; sleep 20; done; echo "# temp $t $(date +%T) battery $(adb shell dumpsys battery | grep -m1 level | tr -d ' \r')"; }
G=/data/local/tmp/aether/gemma4
for q in Q4_0 Q4_K_M; do
  cool; echo "## aether bench $q"
  adb shell "cd /data/local/tmp/aether && toybox time -v ./aether-t1 bench -m $G/gemma-4-E2B-it-$q.gguf --cores 2-7 -p 128,512,1024,2048 -n 64 -r 2 2>&1" | grep -E "^\| |Max RSS"
  cool; echo "## llama-bench $q"
  adb shell "cd /data/local/tmp/aether && LD_LIBRARY_PATH=. toybox time -v taskset fc ./llama-bench -m $G/gemma-4-E2B-it-$q.gguf -t 6 -p 128,512,1024,2048 -n 64 -r 2 2>&1" | grep -E "^\| |Max RSS|error"
done
for n in 128 512 1024 2048; do
  cool; echo "## litert cpu p$n"
  adb shell "cd /data/local/tmp/litert && LD_LIBRARY_PATH=. toybox time -v taskset fc ./litert_lm_main --backend=cpu --model_path=model.litertlm --benchmark --input_prompt_file=p$n.txt --benchmark_decode_tokens=64 --max_num_tokens=4096 --async=false --num_cpu_threads=6 --cache_dir=/data/local/tmp/litert/cache 2>&1" | grep -E "Time to first|Processed|Speed|Init Total|Max RSS"
done
adb shell svc power stayon false
echo "# done $(date +%T)"
