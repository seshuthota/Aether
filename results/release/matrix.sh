#!/bin/bash
cd /home/curious/Documents/hermes-projects/aether-infer
export TEMP_CMD="adb shell 'cat /sys/class/thermal/thermal_zone*/type /dev/null' >/dev/null; adb shell 'for z in /sys/class/thermal/thermal_zone*; do case \$(cat \$z/type) in cpu*) cat \$z/temp;; esac; done'"
export AETHER="adb shell /data/local/tmp/aether/aether"
export COOL_C=45
export ARGS="-p 64,256,512 -n 128 -d 0,512 -r 5"
exec 9>/tmp/aether-phone.lock; flock 9
LLAMA_BENCH="adb shell /data/local/tmp/llama-android/llama-bench" LLAMA_ARGS="-t 6 --cpu-mask 0xfc" AETHER_ARGS="-t 6 -C 0xfc"   bench/compare.sh /home/curious/Documents/hermes-projects/Aether/results/release/compare-iqoo13-cpu /data/local/tmp/aether/testdata/gguf/qwen2.5-0.5b-q4_0.gguf /data/local/tmp/aether/testdata/gguf/qwen2.5-0.5b-q4_k_m.gguf /data/local/tmp/aether/testdata/gguf/LFM3-1.7B-Q4_K_M.gguf /data/local/tmp/aether/testdata/gguf/gemma-4-E2B-it-Q4_K_M.gguf
echo cpu-done rc=$?
LLAMA_BENCH="adb shell LD_LIBRARY_PATH=/data/local/tmp/llama-ocl/lib /data/local/tmp/llama-ocl/bin/llama-bench" LLAMA_ARGS="-ngl 99 --mmap 0" AETHER_ARGS="-dev gpu"   bench/compare.sh /home/curious/Documents/hermes-projects/Aether/results/release/compare-iqoo13-gpu /data/local/tmp/aether/testdata/gguf/qwen2.5-0.5b-q4_0.gguf /data/local/tmp/aether/testdata/gguf/qwen2.5-0.5b-q4_k_m.gguf
echo gpu-done rc=$?
