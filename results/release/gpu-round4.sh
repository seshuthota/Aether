#!/bin/bash
# 2026-11-07: a fourth Adreno round (aether first, as even rounds are), so the
# GPU decode medians rest on three quiet rounds besides round 1 (low on both
# engines). Same llama.cpp binary and args as rounds 1-3; aether is today's
# main (a309a74, plus another thread's uncommitted chat-API additions that
# touch no compute path), pushed as aether-r4, so its table stays separate
# (table-round4.md) and checks that rounds 1-3 still describe the current code.
# Cool below 45 °C per cell.
cd /home/curious/Documents/hermes-projects/aether-infer
R=/home/curious/Documents/hermes-projects/Aether/results/release
M=/data/local/tmp/aether/testdata/gguf
ARGS="-p 64,256,512 -n 128 -d 0,512 -r 5"
hottest(){ adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in cpu*) cat $z/temp;; esac; done' | sort -n | tail -1 | awk '{print int($1/1000)}'; }
cool(){ t=$(hottest); while ((t >= 45)); do sleep 15; t=$(hottest); done; echo $t; }
exec 9>/tmp/aether-phone.lock; flock 9
adb push target-phone/aarch64-linux-android/release/aether /data/local/tmp/aether/aether-r4 >/dev/null
O=$R/compare-iqoo13-gpu
echo "round 4 added $(date -u +%FT%TZ) (aether md5 $(adb shell md5sum /data/local/tmp/aether/aether-r4 | cut -c1-8))" >> $O/conditions.txt
cell(){ r=$1 e=$2 m=$3; t=$(cool); echo "round $r, $e, $m (start $t °C)" >> $O/conditions.txt
  if [ $e = aether ]; then adb shell /data/local/tmp/aether/aether-r4 bench -m $M/$m.gguf $ARGS -dev gpu -o json > $O/round$r-$e-$m.json
  else adb shell LD_LIBRARY_PATH=/data/local/tmp/llama-ocl/lib /data/local/tmp/llama-ocl/bin/llama-bench -m $M/$m.gguf $ARGS -ngl 99 --mmap 0 -o json > $O/round$r-$e-$m.json; fi
  echo "  end $t -> $(hottest) °C" >> $O/conditions.txt; }
for m in qwen2.5-0.5b-q4_0 qwen2.5-0.5b-q4_k_m; do cell 4 aether $m; cell 4 llama.cpp $m; done
adb shell rm -f /data/local/tmp/aether/aether-r4
args=(); for f in $O/round4-*.json; do e=$(basename $f | sed -E 's/round[0-9]-(aether|llama\.cpp)-.*/\1/'); args+=("$e:$f"); done
python3 bench/compare.py "${args[@]}" > $O/table-round4.md
echo gpu-done
