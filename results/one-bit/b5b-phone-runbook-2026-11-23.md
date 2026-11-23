# B5b phone runbook (owed; PC side complete)

Everything below is the only remaining work. The PC results are in
`b5b-pc-2026-11-23.md`; the engine is at `11d6d88` + the PC records.

## Files to build and push

```
cargo build --release --target aarch64-linux-android -p aether-cli
adb push target/aarch64-linux-android/release/aether /data/local/tmp/aether/aether-b5b
adb shell chmod 755 /data/local/tmp/aether/aether-b5b
```

Models (host `hostdata/bonsai/`):
- `Ternary-Bonsai-2-2B-Q2_0.gguf` (546,689,504 B)
- `Ternary-Bonsai-2-4B-Q2_0.gguf` (1,212,103,520 B)
- `Ternary-Bonsai-2-9B-Q2_0.gguf` (2,556,640,128 B, only if MemAvailable allows)
- reference `Ternary-Bonsai-1.7B-Q2_0_g64.gguf` and/or `Bonsai-1.7B-Q1_0.gguf`

Bases / text:
- `kl-base-llama-q2_0-t2-2b-wiki2.bin`
- `kl-base-llama-q2_0-t2-4b-wiki2.bin`
- `kl-base-llama-q2_0-t2-9b-wiki2.bin`
- `wiki.test.raw`

llama.cpp: `/data/local/tmp/llama/llama-bench` at `d02b4c0d6` already
supports the Hadamard runtime on the PC; verify on the phone before the
A/B. If it does not, stop and ask before replacing the phone build.

## Memory

```
adb shell 'cat /proc/meminfo | grep MemAvailable'
adb shell 'cd /data/local/tmp/aether && ./aether-b5b run -m Ternary-Bonsai-2-2B-Q2_0.gguf -p x -n 1 -c 512 -t 6 --cores 2-7 --no-display-prompt'
# repeat for 4B and 9B
```

Record file size, weights line and peak RSS at `-c 512`. Recurrent state
(calculated): 18.9 MB + 1.3 MB conv for 2B; 50.3 MB + 2.4 MB for 4B/9B.
KV per token: 24 KiB (2B), 64 KiB (4B/9B).

## Greedy and KLD

Greedy (16 tokens):

```
./aether-b5b run -m Ternary-Bonsai-2-2B-Q2_0.gguf -p "The capital of France is" -n 16 --temp 0 -t 6 --cores 2-7 --no-display-prompt
```

KLD, 2 chunks:

```
./aether-b5b perplexity -m Ternary-Bonsai-2-2B-Q2_0.gguf -f wiki.test.raw -c 512 --chunks 2 \
  --kl-divergence-base kl-base-llama-q2_0-t2-2b-wiki2.bin --kl-divergence -t 6 --cores 2-7
```

Repeat for 4B (and 9B if memory allows). PC references: 2B 0.000187 /
99.608 %, 4B 0.000191 / 99.216 %, 9B 0.000222 / 99.608 %.

## Speed

3 alternated cooled rounds, 6 threads on cores 2–7, one process per job,
`scripts/phone_ab.py`; include the 1.7B reference in the same session.
Use `llama-bench` on the same GGUF for each model:

```
./aether-b5b bench -m MODEL -dev cpu --cores 2-7 -t 6 -p 128 -n 0 -r 1 -o md
taskset fc /data/local/tmp/llama/llama-bench -m MODEL -t 6 -p 128 -n 0 -r 1
./aether-b5b bench -m MODEL -dev cpu --cores 2-7 -t 6 -p 0 -n 64 -r 1 -o md
taskset fc /data/local/tmp/llama/llama-bench -m MODEL -t 6 -p 0 -n 64 -r 1
```

Record caps and temperatures at the start and end of every job; only n=3
cells go into tables. Prefer the top step (board < 37 °C, caps
>= 2.75/2.65); otherwise say second step. Effective GB/s is `weights MB
× tok/s / 1000`.
