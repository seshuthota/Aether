#!/usr/bin/env bash
# Same-session A/B of the 2026-11-01 optimizations, on battery (adb over
# Wi-Fi), screen kept on, every run cooled first and its conditions
# recorded:
#
#   prompt  GPU prompt attention: attention_tile.comp against one workgroup
#           per token (`vk_pp` against `vk_pp attn1`), Qwen2.5 Q4_0 and Q4_K_M
#   decode  GPU decode: layer products with x in 8 bits (`vk_decode plain q8`)
#           against f32 x (`vk_decode plain`)
#   lookup  prompt lookup K = 4, adaptive (session::DraftLen) against fixed
#           (--lookup-fixed) against plain decoding, on the CPU (cores 2-7),
#           the GPU and the DSP, the phase-6 prompts (code copies the
#           prompt; prose and story mostly do not)
#
# Each pair or triple runs back to back, its order reversed in round 2.
#
#   scripts/opt-ab.sh OUT [prompt decode lookup]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?output dir}; shift
STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(prompt decode lookup)
mkdir -p "$OUT"
COOL_C=${COOL_C:-45}
ROUNDS=${ROUNDS:-2}
QWEN=testdata/gguf/qwen2.5-0.5b
PROMPTS=results/phase-6/raw/lookup
FEAT=vulkan,hexagon

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
  adb shell dumpsys power | grep -m1 'mWakefulness=' | sed 's/^ */# screen: /' | tr -d '\r' || true
}
run() { # name, cargo run args...
  local name=$1; shift
  wait_cool; adb shell input keyevent KEYCODE_WAKEUP
  { conditions; cargo run -q --release --target aarch64-linux-android --features $FEAT "$@" 2>&1 | grep -v pushed || echo "FAILED"; } > "$OUT/$name.txt"
  echo ">>> $name: $(grep -E '^pp[0-9]|tok/s|speculative' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-300)"
}
# Reverses the arguments on even rounds.
order() { if [ $(($1 % 2)) = 1 ]; then shift; echo "$@"; else shift; echo "$@" | tr ' ' '\n' | tac | tr '\n' ' '; fi; }

adb shell input keyevent KEYCODE_WAKEUP
( while :; do sleep 60; adb shell taskset 1 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPALIVE=$!; trap 'kill $KEEPALIVE 2>/dev/null' EXIT
cargo build -q --release --target aarch64-linux-android --features $FEAT --bin aether --example vk_pp --example vk_decode

for stage in "${STAGES[@]}"; do case $stage in
prompt)
  for r in $(seq 1 "$ROUNDS"); do for q in q4_0 q4_k_m; do for v in $(order $r tile attn1); do
    a=; [ $v = attn1 ] && a=attn1
    run "prompt-$q-$v-r$r" --example vk_pp -- $QWEN-$q.gguf $a 5
  done; done; done ;;
decode)
  for r in $(seq 1 "$ROUNDS"); do for q in q4_0 q4_k_m; do for v in $(order $r q8 f32); do
    a=; [ $v = q8 ] && a=q8
    run "decode-$q-$v-r$r" --example vk_decode -- $QWEN-$q.gguf plain $a
  done; done; done ;;
lookup)
  for r in $(seq 1 "$ROUNDS"); do for e in cpu gpu dsp; do for q in q4_0 q4_k_m; do for p in code prose story; do
    case $e in cpu) eng="-c 2-7";; gpu) eng="--gpu";; dsp) eng="--dsp";; esac
    for v in $(order $r adaptive fixed plain); do
      case $v in adaptive) l="--lookup 4";; fixed) l="--lookup 4 --lookup-fixed";; plain) l=;; esac
      run "lookup-$e-$q-$p-$v-r$r" --bin aether -- $QWEN-$q.gguf $eng --chat -t 0 -n 512 $l -i "$(cat $PROMPTS/prompt-$p.txt)"
    done
  done; done; done; done ;;
esac; done
echo "# done $(date -Iseconds)"
