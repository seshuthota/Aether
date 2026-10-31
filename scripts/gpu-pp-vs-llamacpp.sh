#!/usr/bin/env bash
# GPU prompt speed, ours against llama.cpp's OpenCL backend, in one session:
# our vk_pp (int8 and f16 products) and llama-bench -p 64,128,256,512 -n 0
# -ngl 99, per model, in ROUNDS rounds with the engine order reversed each
# round, cooled below COOL_C before every run. Screen kept awake.
#
#   scripts/gpu-pp-vs-llamacpp.sh OUTDIR [MODEL ...]
#
# llama.cpp's OpenCL build is expected on the phone in $OCL (bin/ and lib/).
set -euo pipefail
OUT=$1; shift
if [ $# -gt 0 ]; then MODELS=("$@"); else MODELS=(qwen2.5-0.5b-q4_0 qwen2.5-0.5b-q4_k_m); fi
ROUNDS=${ROUNDS:-2}
COOL_C=${COOL_C:-40}
OCL=${OCL:-/data/local/tmp/llama-ocl}
REMOTE=/data/local/tmp/aether
mkdir -p "$OUT"

temp() { adb shell cat /sys/class/thermal/thermal_zone0/temp | tr -d '\r'; }
cool() { while [ "$(temp)" -ge $((COOL_C * 1000)) ]; do sleep 15; done; }
gpu_clk() { adb shell cat /sys/class/kgsl/kgsl-3d0/max_gpuclk 2>/dev/null | tr -d '\r'; }

adb shell input keyevent KEYCODE_WAKEUP
( while :; do sleep 60; adb shell taskset 8 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
trap 'kill $! 2>/dev/null' EXIT

ours() { cargo run -q --release --target aarch64-linux-android --features vulkan --example vk_pp -- "testdata/gguf/$1.gguf" ${2:-} 5 2>&1 | grep -v pushed; }
llama() { flock /tmp/aether-phone.lock adb shell "cd $REMOTE && LD_LIBRARY_PATH=$OCL/lib $OCL/bin/llama-bench -m testdata/gguf/$1.gguf -p 64,128,256,512 -n 0 -ngl 99 --mmap 0 -r 5" 2>&1 | grep -E '^\|' ; }

run() {
  local name=$1; shift
  cool
  { echo "# $(date -Iseconds) start $(temp) m°C, GPU max clock $(gpu_clk)"; "$@"; echo "# end $(temp) m°C"; } | tr -d '\r' > "$OUT/$name.txt"
  echo ">>> $name"; grep -E '^pp|pp[0-9]+ ' "$OUT/$name.txt" || tail -3 "$OUT/$name.txt"
}

for r in $(seq 1 "$ROUNDS"); do
  for m in "${MODELS[@]}"; do
    if [ $((r % 2)) = 1 ]; then
      run "$m-ours-int8-r$r" ours "$m"; run "$m-ours-f16-r$r" ours "$m" f16; run "$m-llamacpp-r$r" llama "$m"
    else
      run "$m-llamacpp-r$r" llama "$m"; run "$m-ours-f16-r$r" ours "$m" f16; run "$m-ours-int8-r$r" ours "$m"
    fi
  done
done
