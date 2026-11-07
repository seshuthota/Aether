#!/bin/bash
# 2026-11-07: the handoff's section 7 step 2.
# (1) LFM3-1.7B Q4_K_M on the CPU again, alone, two rounds in opposite order
#     (round 2 of 2026-11-06 swung 14.8-34.7 tok/s), into compare-iqoo13-cpu-lfm3/.
# (2) A third Adreno round (llama.cpp first, as odd rounds are) for the wide
#     GPU decode ranges, into compare-iqoo13-gpu/round3-*.json.
# Same binaries as 2026-11-06 (aether md5 e11a8e1e), same args, cool below 45 °C.
cd /home/curious/Documents/hermes-projects/aether-infer
R=/home/curious/Documents/hermes-projects/Aether/results/release
M=/data/local/tmp/aether/testdata/gguf
ARGS="-p 64,256,512 -n 128 -d 0,512 -r 5"
hottest(){ adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in cpu*) cat $z/temp;; esac; done' | sort -n | tail -1 | awk '{print int($1/1000)}'; }
cool(){ t=$(hottest); while ((t >= 45)); do sleep 15; t=$(hottest); done; echo $t; }
exec 9>/tmp/aether-phone.lock; flock 9

export TEMP_CMD="adb shell 'for z in /sys/class/thermal/thermal_zone*; do case \$(cat \$z/type) in cpu*) cat \$z/temp;; esac; done'"
export AETHER="adb shell /data/local/tmp/aether/aether" COOL_C=45 ARGS="$ARGS"
LLAMA_BENCH="adb shell /data/local/tmp/llama-android/llama-bench" LLAMA_ARGS="-t 6 --cpu-mask 0xfc" AETHER_ARGS="-t 6 -C 0xfc" \
  bench/compare.sh $R/compare-iqoo13-cpu-lfm3 $M/LFM3-1.7B-Q4_K_M.gguf
echo lfm3-done rc=$?

O=$R/compare-iqoo13-gpu
echo "round 3 added $(date -u +%FT%TZ) for the wide decode ranges (aether md5 e11a8e1e unchanged)" >> $O/conditions.txt
cell(){ r=$1 e=$2 m=$3; t=$(cool); echo "round $r, $e, $m (start $t °C)" >> $O/conditions.txt
  if [ $e = aether ]; then adb shell /data/local/tmp/aether/aether bench -m $M/$m.gguf $ARGS -dev gpu -o json > $O/round$r-$e-$m.json
  else adb shell LD_LIBRARY_PATH=/data/local/tmp/llama-ocl/lib /data/local/tmp/llama-ocl/bin/llama-bench -m $M/$m.gguf $ARGS -ngl 99 --mmap 0 -o json > $O/round$r-$e-$m.json; fi
  echo "  end $t -> $(hottest) °C" >> $O/conditions.txt; }
for m in qwen2.5-0.5b-q4_0 qwen2.5-0.5b-q4_k_m; do cell 3 llama.cpp $m; cell 3 aether $m; done
args=(); for f in $O/round*.json; do e=$(basename $f | sed -E 's/round[0-9]-(aether|llama\.cpp)-.*/\1/'); args+=("$e:$f"); done
python3 bench/compare.py "${args[@]}" > $O/table.md
echo gpu-done
