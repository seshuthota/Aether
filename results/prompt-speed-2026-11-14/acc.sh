#!/bin/bash
# Gemma 4 E2B on the accelerators: Aether Vulkan / DSP, LiteRT GPU, llama.cpp OpenCL / Hexagon.
exec 9>/tmp/aether-phone.lock; flock 9
adb push -q /tmp/claude-1000/-home-curious-Documents-hermes-projects-aether-infer/db08fe39-a46e-4005-afd9-05b02a7c3a54/scratchpad/target-dev/aarch64-linux-android/release/aether /data/local/tmp/aether/aether-dev
adb shell svc power stayon true
cool() { while :; do t=$(adb shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'); [ "$t" -lt 45000 ] && break; sleep 20; done; echo "# temp $t $(date +%T) battery $(adb shell dumpsys battery | grep -m1 level | tr -d ' \r')"; }
G=/data/local/tmp/aether/gemma4/gemma-4-E2B-it-Q4_0.gguf
cool; echo "## aether gpu Q4_0"
adb shell "cd /data/local/tmp/aether && toybox time -v ./aether-dev bench -m $G -dev gpu --cores 2-7 -p 128,512,1024,2048 -n 64 -r 2 2>&1" | grep -E "^\| |Max RSS|rror|nsupported"
cool; echo "## aether dsp Q4_0"
adb shell "cd /data/local/tmp/aether && toybox time -v ./aether-dev bench -m $G -dev dsp --cores 2-7 -p 128,512,1024,2048 -n 64 -r 2 2>&1" | grep -E "^\| |Max RSS|rror|nsupported"
for n in 128 512 1024 2048; do
  cool; echo "## litert gpu p$n"
  adb shell "cd /data/local/tmp/litert && LD_LIBRARY_PATH=. toybox time -v ./litert_lm_main --backend=gpu --model_path=model.litertlm --benchmark --input_prompt_file=p$n.txt --benchmark_decode_tokens=64 --max_num_tokens=4096 --async=false --cache_dir=/data/local/tmp/litert/cache 2>&1" | grep -E "Time to first|Processed|Speed|Init Total|Max RSS|rror"
done
cool; echo "## llama.cpp opencl Q4_0"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=llama-ocl/lib timeout 600 toybox time -v llama-ocl/bin/llama-bench -m $G -ngl 99 -t 6 -p 128,512,1024,2048 -n 64 -r 2 2>&1" | grep -E "^\| |Max RSS|rror|ailed" | head -20
cool; echo "## llama.cpp hexagon Q4_0"
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=llama-hex/lib ADSP_LIBRARY_PATH=llama-hex/lib timeout 600 toybox time -v llama-hex/bin/llama-bench -m $G -ngl 99 -t 6 -p 128,512,1024,2048 -n 64 -r 2 2>&1" | grep -E "^\| |Max RSS|rror|ailed" | head -20
adb shell svc power stayon false
echo "# done $(date +%T)"
