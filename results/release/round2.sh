#!/bin/bash
cd /home/curious/Documents/hermes-projects/aether-infer
O=/home/curious/Documents/hermes-projects/Aether/results/release/compare-iqoo13-cpu
hottest(){ adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in cpu*) cat $z/temp;; esac; done' | sort -n | tail -1 | awk '{print int($1/1000)}'; }
cool(){ t=$(hottest); while ((t >= 45)); do sleep 15; t=$(hottest); done; echo $t; }
exec 9>/tmp/aether-phone.lock; flock 9
echo "round 2 resumed $(date -u +%FT%TZ) after a phone break (llama-bench md5 1116ba37, aether md5 e11a8e1e, both unchanged)" >> $O/conditions.txt
ARGS="-p 64,256,512 -n 128 -d 0,512 -r 5"
for m in qwen2.5-0.5b-q4_0 qwen2.5-0.5b-q4_k_m LFM3-1.7B-Q4_K_M gemma-4-E2B-it-Q4_K_M; do
  for e in aether llama.cpp; do
    t=$(cool); echo "round 2, $e, $m (start $t °C)" >> $O/conditions.txt
    if [ $e = aether ]; then adb shell /data/local/tmp/aether/aether bench -m /data/local/tmp/aether/testdata/gguf/$m.gguf $ARGS -t 6 -C 0xfc -o json > $O/round2-$e-$m.json
    else adb shell /data/local/tmp/llama-android/llama-bench -m /data/local/tmp/aether/testdata/gguf/$m.gguf $ARGS -t 6 --cpu-mask 0xfc -o json > $O/round2-$e-$m.json; fi
    echo "  end $t -> $(hottest) °C" >> $O/conditions.txt
  done
done
args=(); for f in $O/round*.json; do e=$(basename $f | sed -E 's/round[0-9]-(aether|llama\.cpp)-.*/\1/'); args+=("$e:$f"); done
python3 bench/compare.py "${args[@]}" > $O/table.md
echo cpu-done
flock -u 9
export TEMP_CMD="adb shell 'for z in /sys/class/thermal/thermal_zone*; do case \$(cat \$z/type) in cpu*) cat \$z/temp;; esac; done'"
export AETHER="adb shell /data/local/tmp/aether/aether" COOL_C=45 ARGS="$ARGS"
exec 9>/tmp/aether-phone.lock; flock 9
LLAMA_BENCH="adb shell LD_LIBRARY_PATH=/data/local/tmp/llama-ocl/lib /data/local/tmp/llama-ocl/bin/llama-bench" LLAMA_ARGS="-ngl 99 --mmap 0" AETHER_ARGS="-dev gpu"   bench/compare.sh /home/curious/Documents/hermes-projects/Aether/results/release/compare-iqoo13-gpu /data/local/tmp/aether/testdata/gguf/qwen2.5-0.5b-q4_0.gguf /data/local/tmp/aether/testdata/gguf/qwen2.5-0.5b-q4_k_m.gguf
echo gpu-done
