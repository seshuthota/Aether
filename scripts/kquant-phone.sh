#!/usr/bin/env bash
# The phone half of plans/kquant-decode.md (results/phase-kquant.md, "Phone
# checks pending"), plus llama.cpp in the same session (K5) and Gemma 4 E2B
# in llama.cpp, in one run: on battery, screen on, cooled before every
# timed run, conditions recorded with each.
#
#   scripts/kquant-phone.sh [OUT]
#
# Each K step is measured on its own commit, built in a temporary worktree of
# the kquant-decode tree (IMPL, default ../Aether-impl), so the tree the
# implementation track works in is never checked out or built here.
# STAGES picks a subset, e.g. STAGES="lane tip k5" scripts/kquant-phone.sh.
#
# Stages, in the order they run:
#   k0    25bed76  kquant_bench, the bandwidth ceiling, tg64, the profile
#   k1    5116346  k_quant tests, PPL unchanged to the digit, bench, tg64
#   k2    79439bc  PPL spot checks (LFM3 20x512, ctx 2048, Qwen), bench, tg64
#   lane  9a7b59d / 3febb3a  kquant_bench --repack before/after the lane sdots
#   tip   $TIP     tests, PPL, kquant_bench --repack, tg64, peak RSS, profile
#   k4    $K4_REV  only if K4_REV is set: tip and K4 alternated; K4_FEATURES
#                  (cargo features) selects layout B if it sits behind one
#   k5    $TIP     ours against llama-bench, alternating, ROUNDS rounds
#   gemma          llama-bench on Gemma 4 E2B (Q4_0 and Q4_K_M), CPU, 6 threads
set -euo pipefail
cd "$(dirname "$0")/.."
HERE=$PWD
OUT=$(realpath -m "${1:-results/phase-kquant/raw/phone}")
mkdir -p "$OUT"
IMPL=$(realpath "${IMPL:-../Aether-impl}")
TIP=${TIP:-$(git -C "$IMPL" rev-parse --short kquant-decode)}
K4_REV=${K4_REV:-}
K4_FEATURES=${K4_FEATURES:-}
STAGES=${STAGES:-k0 k1 k2 lane tip k4 k5 gemma}
COOL_C=${COOL_C:-45}
ROUNDS=${ROUNDS:-2}
CORES=${CORES:-2-7}
REMOTE=/data/local/tmp/aether
LLAMA_BENCH=${LLAMA_BENCH:-$HOME/llama.cpp/build-android/bin/llama-bench}
GEMMA_DIR=${GEMMA_DIR:-$HERE/testdata/gguf}
GEMMA=(gemma-4-E2B-it-Q4_0 gemma-4-E2B-it-Q4_K_M)
WIKI=testdata/wikitext/wikitext-2-raw/wiki.test.raw
LFM=testdata/gguf/LFM3-1.7B-Q4_K_M.gguf
QWEN=testdata/gguf/qwen2.5-0.5b-q4_k_m.gguf
# One target directory for every commit: cargo rebuilds what changed.
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/.cache/aether-kquant-phone}

wait_cool() {
  while :; do
    local max
    max=$(adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type 2>/dev/null) in cpu*) cat $z/temp 2>/dev/null;; esac; done' | sort -n | tail -1 | tr -d '\r')
    [ "${max:-0}" -lt $((COOL_C * 1000)) ] && return
    sleep 15
  done
}
wake() {
  adb shell input keyevent KEYCODE_WAKEUP
  adb shell wm dismiss-keyguard
  adb shell input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1 || true
}
conditions() { # rev
  echo "# $(date -Iseconds), commit $1"
  adb shell dumpsys battery | grep -E 'AC powered|USB powered|Wireless powered|status:|level|temperature' | sed 's/^ */# /' | tr -d '\r' || true
  adb shell 'for c in 0 6; do echo "# cpu$c cap $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq) kHz"; done' | tr -d '\r' || true
  adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type 2>/dev/null) in cpu*) cat $z/temp;; esac; done' | sort -n | tail -1 | sed 's/^/# max cpu zone (mC) /' | tr -d '\r' || true
  adb shell dumpsys display | grep -m1 mScreenState | sed 's/^ */# /' | tr -d '\r' || true
}

# A detached worktree of IMPL at REV, with the gitignored pieces a build and
# the runner need: the cargo config (linker, runner) and testdata/ (models,
# goldens, wikitext), linked from IMPL.
tree() { # rev -> prints the path
  local dir=/tmp/kqp-$1
  if [ ! -d "$dir" ]; then
    git -C "$IMPL" worktree add --detach "$dir" "$1" >/dev/null 2>&1
    mkdir -p "$dir/.cargo"
    cp "$IMPL/.cargo/config.toml" "$dir/.cargo/"
    ln -s "$IMPL/testdata" "$dir/testdata"
  fi
  echo "$dir"
}
cleanup() {
  adb shell settings put system screen_off_timeout 30000 || true
  for d in /tmp/kqp-*; do [ -d "$d" ] && git -C "$IMPL" worktree remove --force "$d"; done
  kill "$KEEPER" 2>/dev/null || true
}

# cargo on the phone through the runner, in the tree of REV. The build comes
# first, so the cooling and the recorded conditions are those of the run.
# NAME.txt gets the conditions, then the output. A failed run is noted and the
# session goes on. FEATURES (cargo features) applies to both.
run() { # name rev cmd what args... (cmd: run|test, what: --example X|--bin X|--lib)
  local name=$1 rev=$2 cmd=$3 dir; shift 3
  local what=("$1"); [ "$1" = --lib ] || { what+=("$2"); shift; }; shift
  dir=$(tree "$rev")
  local feat=(); [ -n "$FEATURES" ] && feat=(--features "$FEATURES")
  local build=(build); [ "$cmd" = test ] && build=(test --no-run)
  (cd "$dir" && cargo "${build[@]}" -q --release --target aarch64-linux-android "${feat[@]}" "${what[@]}")
  wait_cool; wake
  { conditions "$rev $FEATURES"
    (cd "$dir" && cargo "$cmd" -q --release --target aarch64-linux-android "${feat[@]}" "${what[@]}" "$@" 2>&1 | { grep -v pushed || true; })
  } > "$OUT/$name.txt" || echo "!!! $name failed (see $OUT/$name.txt)"
  echo ">>> $name"
}
kbench() { local n=$1 r=$2; shift 2; run "$n" "$r" run --example kquant_bench -- "$LFM" "$@"; }
suite() { local n=$1 r=$2; shift 2; run "$n" "$r" run --example bench_suite -- "$LFM" -c "$CORES" --pp 64,128,512 --tg 64 --reps 3 --json --monitor "$@"; }
ppl() { # name rev model ctx chunks
  run "$1" "$2" run --bin aether -- "$3" --ppl "$WIKI" --ctx "$4" --chunks "$5" -c "$CORES"
}
ktest() { run "$1" "$2" test --lib k_quant; }

# 300 greedy tokens under simpleperf; perf.data and the binary are pulled
# for report_html.py (the instruction-level listing of the K-quant kernels).
profile() { # name rev
  local name=$1 rev=$2 dir; dir=$(tree "$rev")
  (cd "$dir" && cargo build -q --release --target aarch64-linux-android --bin aether) || { echo "!!! $name: build failed"; return; }
  adb push "$CARGO_TARGET_DIR/aarch64-linux-android/release/aether" "$REMOTE/aether" >/dev/null
  adb push --sync "$IMPL/testdata/gguf/LFM3-1.7B-Q4_K_M.gguf" "$REMOTE/testdata/gguf/" >/dev/null
  wait_cool; wake
  { conditions "$rev"
    flock /tmp/aether-phone.lock adb shell "cd $REMOTE && simpleperf record -e cpu-cycles:u -o perf.data -- ./aether $LFM -i 'The history of the printing press' -t 0 -n 300 -c $CORES" 2>&1 | tr -d '\r'
    adb shell "cd $REMOTE && simpleperf report -i perf.data --sort symbol" 2>&1 | tr -d '\r' | head -60
  } > "$OUT/$name.txt"
  mkdir -p "$OUT/$name"
  adb pull "$REMOTE/perf.data" "$OUT/$name/perf.data" >/dev/null
  cp "$CARGO_TARGET_DIR/aarch64-linux-android/release/aether" "$OUT/$name/aether"
  adb shell "rm -f $REMOTE/perf.data $REMOTE/aether"
  echo ">>> $name"
}
llama() { # name model-path-on-phone extra...
  local name=$1 model=$2; shift 2
  wait_cool; wake
  { conditions "llama.cpp $(git -C "$HOME/llama.cpp" rev-parse --short HEAD)"
    flock /tmp/aether-phone.lock adb shell "cd $REMOTE && taskset fc ./llama-bench -m $model -t 6 -r 3 $*" 2>&1 | tr -d '\r'; } > "$OUT/$name.txt"
  echo ">>> $name"
}
has() { [[ " $STAGES " == *" $1 "* ]]; }

adb get-state >/dev/null
adb shell settings put system screen_off_timeout 1800000
wake
( while :; do sleep 30; adb shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1; adb shell input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
KEEPER=$!
trap cleanup EXIT
FEATURES=
{ echo "# kquant-phone.sh, $(date -Iseconds), tip $TIP, stages: $STAGES"; conditions start; } > "$OUT/session.txt"

if has k0; then
  kbench k0-bench 25bed76
  run k0-ceiling-1 25bed76 run --example bw_sweep -- \
    --kernel read --cores 7 --min-kb 32 --max-mb 1024 --steps 4 --target-mb 512 --samples 7
  run k0-ceiling-6 25bed76 run --example bw_sweep -- \
    --kernel read --cores 2-7 --parallel --min-kb 32 --max-mb 1024 --steps 4 --target-mb 512 --samples 7
  run k0-sustained 25bed76 run --example bw_sustained -- \
    --cores 2-7 --parallel --seconds 30 --size-mb 1024
  suite k0-tg64 25bed76
  profile k0-profile 25bed76
fi
if has k1; then
  ktest k1-tests 5116346
  ppl k1-ppl-lfm3 5116346 "$LFM" 512 20        # 24.8822 before: same digits
  ppl k1-ppl-qwen 5116346 "$QWEN" 512 20
  kbench k1-bench 5116346
  suite k1-tg64 5116346
  profile k1-profile 5116346                    # are the 16 vaddlvq gone?
fi
if has k2; then
  ppl k2-ppl-lfm3 79439bc "$LFM" 512 20
  ppl k2-ppl-lfm3-ctx2048 79439bc "$LFM" 2048 1
  ppl k2-ppl-qwen 79439bc "$QWEN" 512 20
  kbench k2-bench 79439bc
  suite k2-tg64 79439bc
fi
if has lane; then                               # alternate, so heat is shared
  for r in $(seq "$ROUNDS"); do
    kbench "lane-before-r$r" 9a7b59d --repack
    kbench "lane-after-r$r" 3febb3a --repack
  done
fi
if has tip; then
  ktest tip-tests "$TIP"
  ppl tip-ppl-lfm3 "$TIP" "$LFM" 512 20
  ppl tip-ppl-qwen "$TIP" "$QWEN" 512 20
  kbench tip-bench "$TIP" --repack
  suite tip-tg64 "$TIP"                         # peak RSS is in its output
  profile tip-profile "$TIP"                    # dot4_q6_k_q8_k_x4: ql/qh loads
fi
if has k4 && [ -n "$K4_REV" ]; then
  for r in $(seq "$ROUNDS"); do
    suite "k4-a-r$r" "$TIP"
    FEATURES=$K4_FEATURES suite "k4-b-r$r" "$K4_REV"
  done
fi
if has k5; then
  adb push "$LLAMA_BENCH" "$REMOTE/llama-bench" >/dev/null
  adb push --sync "$IMPL/$LFM" "$REMOTE/testdata/gguf/" >/dev/null
  for r in $(seq "$ROUNDS"); do
    if [ $((r % 2)) = 1 ]; then
      suite "k5-ours-r$r" "$TIP"; llama "k5-llama-r$r" "$LFM" -p 64,128,512 -n 64
    else
      llama "k5-llama-r$r" "$LFM" -p 64,128,512 -n 64; suite "k5-ours-r$r" "$TIP"
    fi
  done
fi
if has gemma; then
  adb push "$LLAMA_BENCH" "$REMOTE/llama-bench" >/dev/null
  adb shell "mkdir -p $REMOTE/gemma4"
  for m in "${GEMMA[@]}"; do
    adb push --sync "$GEMMA_DIR/$m.gguf" "$REMOTE/gemma4/" >/dev/null
    # tg at depth 0 and 512 brackets the app's run (12 tokens in, 1227 out);
    # tg 256 is long enough to show the first minute's heat.
    llama "gemma-$m" "gemma4/$m.gguf" -p 64,512 -n 64,256 -d 0,512
  done
fi
echo "done: $OUT"
