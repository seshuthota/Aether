# Supra2.5 S7: decode past 1000 tok/s with context

aether-infer `plans/supra.md`, S7. iQOO 13 (SM8750), on battery, wireless
adb, screen held awake; each run started under 40 °C (zone0). Supra2.5-65M
Q4_0, `bench -p 0 -n 128 -d 0,256 -r 3 -t 1,6 -C 0xfc -ctk f16 -ctv f16`,
binaries alternated OLD/NEW/NEW/OLD (`ab2.sh`); LFM3-1.7B Q4_K_M tg64,
6 threads (`ab3.sh`).

| Log | OLD → NEW | Caps (cpu2/cpu7, kHz) | Result |
| --- | --- | --- | --- |
| `a1.log` | merged `supra` → no-allocation decode, step 1 | 1785600-2400000 / 2438400 | even |
| `a2.log`, `a2-lfm3.log` | → no-allocation decode, final (`99620b9`) | 2400000 / 2438400 | even |
| `pf2.log` | `99620b9` → attention prefetch (`7187df7`) | 2918400 / 3072000 | d256, 6 threads: 999 / 879 → 1084 / 1099; 1 thread 278 / 281 → 318 / 318; d0 even |
| `pf2-lfm3.log` | same | 2400000 / 2438400 | 49.4 / 50.2 → 50.3 / 50.4 (even) |

Greedy check, 300 tokens, "Write a long story about a robot who learns to
paint.", `--temp 0 -t 6`: output md5 unchanged by the prefetch (f32 cache
`6d3ad9e7…`, f16 `aae4eac7…`).

Attention micro-benchmark (`attn_bench --cores 7 --n 320`, one layer, 10
heads of 64), ns per head and position, without → with prefetch: f32 warm
8.45 → 9.65, cold 11.04 → 11.67; f16 warm 8.64 → 9.66, cold 12.01 → 10.69.
The gain shows in the model (where each layer's rows come back cold from
DRAM, 128 bytes of each 1280-byte row), not in the benchmark.

Thread shares weighted by core speed were tried and dropped: under the
phone's caps the 3.53 and 4.32 GHz cores ran at nearly the same clock.
