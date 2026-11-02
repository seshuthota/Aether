#!/usr/bin/env bash
# F4 of plans/flash-attention.md in one phone session, on battery (adb over
# Wi-Fi), screen kept on, every run cooled first and its conditions
# recorded (battery, clock caps, foreground app):
#
#   tests  the whole suite on the phone with --features vulkan,hexagon
#          (flash is the default prompt path; c_abi_gpu_session included)
#   kld    the prompt pass at wikitext-2 20 x 512, flash (the default)
#          against the two passes it replaces (batch-chunked), SmolLM2-135M
#          Q4_0, Qwen2.5-0.5B Q4_0 and Q4_K_M, each against the same f32
#          file's logits for KL. Compare with results/phase-l3.md step 22's
#          raw/vk-kld-*-batch-tile.txt (the path it replaces) and the step
#          20 table's llama.cpp error bar.
#   deep   Qwen2.5-0.5B Q4_0 at context 2048 and 4096, flash and chunked,
#          against the f32 base: the first check of the GPU prompt path
#          beyond 512 tokens.
#   llamacpp  llama-bench (the OpenCL build on the phone, $OCL) at pp512
#          after 0/1024/2048/4096 cached tokens, for the same-session
#          comparison of step 26 — the chunked runs of `flash` are at the
#          same depths.
#
# The f32 base files are pushed to the phone once (they are not testdata).
#
#   scripts/flash-checks.sh OUT [tests kld deep llamacpp]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir}; shift
STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(tests kld deep llamacpp)
mkdir -p "$OUT"
COOL_C=${COOL_C:-45}
QWEN=testdata/gguf/qwen2.5-0.5b
SMOLLM=testdata/gguf/smollm2-135m
REMOTE=/data/local/tmp/aether
PHONE="--target aarch64-linux-android"
OCL=${OCL:-/data/local/tmp/llama-ocl}

wait_cool() {
  while :; do
    local max
    max=$(adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type 2>/dev/null) in cpu*) cat $z/temp 2>/dev/null;; esac; done' | sort -n | tail -1 | tr -d '\r')
    [ "${max:-0}" -lt $((COOL_C * 1000)) ] && return
    sleep 15
  done
}
conditions() {
  echo "# $(date -Iseconds), commit $(git rev-parse --short HEAD)"
  adb shell dumpsys battery | grep -E 'powered: true|status:|level' | sed 's/^ */# /' | tr -d '\r' || true
  adb shell 'for c in 0 6; do echo "# cpu$c cap $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq) kHz"; done' | tr -d '\r' || true
  adb shell dumpsys activity activities | grep -m1 -E 'topResumedActivity|mResumedActivity' | sed 's/^ */# foreground: /' | tr -d '\r' || true
  adb shell dumpsys power | grep -m1 'mWakefulness=' | sed 's/^ */# screen: /' | tr -d '\r' || true
}
run() { # name, cargo-run args after `--`
  local name=$1; shift
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  { conditions; cargo run -q --release $PHONE --features vulkan "$@" 2>&1 | grep -v pushed || echo "FAILED"; } > "$OUT/$name.txt"
  if grep -q FAILED "$OUT/$name.txt"; then
    echo ">>> $name: FAILED"
  else
    echo ">>> $name: $(grep -E 'Final estimate|Mean KLD|^test result' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-260)"
  fi
}
base() { # remote name, host file
  adb shell "test -f $REMOTE/$1" || { echo "# pushing $1"; adb push "$2" "$REMOTE/$1" >/dev/null; }
}

# Battery numbers only, as the other phone scripts: wait until unplugged.
if [ "${ON_BATTERY:-1}" = 1 ]; then
  until adb shell dumpsys battery | grep -q -E 'AC powered: false' && ! adb shell dumpsys battery | grep -q -E '(USB|Wireless) powered: true'; do
    echo "# $(date -Iseconds) waiting for the phone to be unplugged"; sleep 60
  done
fi
( while :; do sleep 60; adb shell taskset 1 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT
adb shell input keyevent KEYCODE_WAKEUP

for stage in "${STAGES[@]}"; do case $stage in
tests)
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  { conditions; cargo test -q --release $PHONE --features vulkan,hexagon 2>&1 | grep -v pushed || echo "FAILED"; } > "$OUT/vk-tests-flash-phone.txt"
  grep -E '^test result|FAILED' "$OUT/vk-tests-flash-phone.txt" | tail -3 ;;
kld)
  base f32-smollm2-135m.gguf hostdata/gguf/smollm2-135m-f32.gguf
  base f32-qwen2.5-0.5b.gguf hostdata/gguf/qwen2.5-0.5b-f32.gguf
  for m in "smollm2-135m-q4_0 $SMOLLM-q4_0.gguf f32-smollm2-135m.gguf" \
           "qwen2.5-0.5b-q4_0 $QWEN-q4_0.gguf f32-qwen2.5-0.5b.gguf" \
           "qwen2.5-0.5b-q4_k_m $QWEN-q4_k_m.gguf f32-qwen2.5-0.5b.gguf"; do
    read -r name model f32 <<< "$m"
    run "vk-kld-$name-flash" --example vk_ppl -- "$model" 20 512 batch "$f32"
    run "vk-kld-$name-flash-chunked" --example vk_ppl -- "$model" 20 512 batch-chunked "$f32"
  done ;;
deep)
  base f32-qwen2.5-0.5b.gguf hostdata/gguf/qwen2.5-0.5b-f32.gguf
  for ctx in 2048 4096; do for mode in batch batch-chunked; do
    run "vk-kld-qwen2.5-0.5b-q4_0-$mode-c$ctx" --example vk_ppl -- $QWEN-q4_0.gguf 1 $ctx $mode f32-qwen2.5-0.5b.gguf
  done; done ;;
llamacpp)
  adb push --sync testdata/. $REMOTE/testdata/ >/dev/null
  for q in q4_0 q4_k_m; do
    wait_cool; adb shell input keyevent KEYCODE_WAKEUP
    { conditions; flock /tmp/aether-phone.lock adb shell "cd $REMOTE && LD_LIBRARY_PATH=$OCL/lib $OCL/bin/llama-bench -m testdata/gguf/qwen2.5-0.5b-$q.gguf -p 512 -n 0 -d 0,1024,2048,4096 -ngl 99 --mmap 0 -r 3" 2>&1 | tr -d '\r' || echo FAILED; } > "$OUT/llamacpp-qwen2.5-0.5b-$q-pp512.txt"
    echo ">>> llamacpp-qwen2.5-0.5b-$q-pp512: $(grep -E '^\\| *pp512' "$OUT/llamacpp-qwen2.5-0.5b-$q-pp512.txt" | tr '\n' ' ' | cut -c1-240)"
  done ;;
esac; done
echo "# done $(date -Iseconds)"
