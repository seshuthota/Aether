#!/bin/bash
cd /home/curious/Documents/hermes-projects/aether-infer
O=/home/curious/Documents/hermes-projects/Aether/results/release/compare-iqoo13-gpu
hottest(){ adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in cpu*) cat $z/temp;; esac; done' | sort -n | tail -1 | awk '{print int($1/1000)}'; }
cool(){ t=$(hottest); while ((t >= 45)); do sleep 15; t=$(hottest); done; echo $t; }
exec 9>/tmp/aether-phone.lock; flock 9
echo "resumed $(date -u +%FT%TZ): round 1 llama.cpp q4_k_m was cut off mid-run (partial JSON discarded) and is redone" >> $O/conditions.txt
ARGS="-p 64,256,512 -n 128 -d 0,512 -r 5"
cell(){ r=$1 e=$2 m=$3; t=$(cool); echo "round $r, $e, $m (start $t °C)" >> $O/conditions.txt
  if [ $e = aether ]; then adb shell /data/local/tmp/aether/aether bench -m /data/local/tmp/aether/testdata/gguf/$m.gguf $ARGS -dev gpu -o json > $O/round$r-$e-$m.json
  else adb shell LD_LIBRARY_PATH=/data/local/tmp/llama-ocl/lib /data/local/tmp/llama-ocl/bin/llama-bench -m /data/local/tmp/aether/testdata/gguf/$m.gguf $ARGS -ngl 99 --mmap 0 -o json > $O/round$r-$e-$m.json; fi
  echo "  end $t -> $(hottest) °C" >> $O/conditions.txt; }
cell 1 llama.cpp qwen2.5-0.5b-q4_k_m
cell 1 aether qwen2.5-0.5b-q4_k_m
for m in qwen2.5-0.5b-q4_0 qwen2.5-0.5b-q4_k_m; do cell 2 aether $m; cell 2 llama.cpp $m; done
args=(); for f in $O/round*.json; do e=$(basename $f | sed -E 's/round[0-9]-(aether|llama\.cpp)-.*/\1/'); args+=("$e:$f"); done
python3 bench/compare.py "${args[@]}" > $O/table.md
echo gpu-done
