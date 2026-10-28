# The memory hierarchy

## Overview
- **Topic**: Memory hierarchy, as taught on the Phase 1 memory page
- **Hook**: The core is fast, DRAM is far, and the weights live in DRAM
- **Target Audience**: Just finished Phase 0. Has never heard "cache hit" before
- **Length**: about 3 minutes, muted. Captions are held for a reading pace of ~3 words/s
- **Key Insight**: Caches only help with reuse. Decode reads every weight once per token and the weights are far bigger than the caches, so they are pushed out before the next token: every token reads them from DRAM. Bandwidth, not latency, is the number that matters. Nothing here is a measurement of this phone.

## Scene beats
1. **Title** — why decode waits on DRAM, every token.
2. **The wait** — one weight travels from DRAM; an idle-cycle counter climbs to ~400 (~100 ns at ~4 GHz, typical).
3. **The cache** — need w0: miss, a whole 64-byte line comes back and is kept. Need w1: hit. Locality named.
4. **The ladder** — registers → L1 → L2 → SLC → DRAM. A load checks each level (miss, miss, miss, found); data climbs back up leaving copies.
5. **Latency** — all five rungs on the human scale (1 ns = 1 s), log-scale bars labelled as such.
6. **Bandwidth** — two lanes, same trip time: one load at a time vs many in flight. Counters show delivered lines. One core may not fill the bus.
7. **Capacity** — 12 weight blocks stream through a 3-slot cache; oldest evicted; token 2 starts at block 1: miss. Then the 1-in-100 grid at real sizes.
8. **Levers** — fewer bytes (2 GB f32 → 0.5 GB int8), reuse each byte (prefill, batching), faster core crossed out.
9. **Close** — tokens/s ≤ bandwidth / bytes read per token. Phase 1 measures the top.

## Color
Dark paper, so the film reads as a video on either site theme. Teal is the fast rung, blue the middle, copper the slow rung and every assumption.
