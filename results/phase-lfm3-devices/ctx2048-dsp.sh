#!/bin/bash
# 2026-11-07: C3's owed check. LFM3-1.7B Q4_K_M, wikitext-2, 1 chunk of 2048,
# on the DSP through the HMX prompt pass (batch 192), then the one-token
# decode path on the same binary for comparison (C3 recorded 13.7980;
# PC: llama.cpp 13.9561, our CPU 14.0234). Cool below 45 °C before each.
# Build first: CARGO_TARGET_DIR=target-phone cargo build --release \
#   --target aarch64-linux-android --features hexagon -p aether-bench --example dsp_ppl
cd /home/curious/Documents/hermes-projects/aether-infer
O=/home/curious/Documents/hermes-projects/Aether/results/phase-lfm3-devices
B=target-phone/aarch64-linux-android/release/examples/dsp_ppl
M=testdata/gguf/LFM3-1.7B-Q4_K_M.gguf
hottest(){ adb shell 'for z in /sys/class/thermal/thermal_zone*; do case $(cat $z/type) in cpu*) cat $z/temp;; esac; done' | sort -n | tail -1 | awk '{print int($1/1000)}'; }
cool(){ t=$(hottest); while ((t >= 45)); do sleep 15; t=$(hottest); done; echo $t; }
echo "engine $(git rev-parse --short HEAD), dsp_ppl md5 $(md5sum $B | cut -c1-8)" > $O/ctx2048-conditions.txt
run(){ name=$1; shift; t=$(cool); echo "$name: start $t °C, $(date -u +%FT%TZ)" >> $O/ctx2048-conditions.txt
  scripts/adb-run.sh $B $M 1 2048 6 1f "$@" > $O/ctx2048-$name.txt 2>&1; echo "  rc=$? end $(hottest) °C" >> $O/ctx2048-conditions.txt; }
run hmx-batch192 192
run decode 1
echo done
