# The roofline

Planned with the `3b1b-explainer` skill (`.claude/skills/3b1b-explainer/`).

## Overview
- **Topic**: The roofline, as taught on the Phase 1 roofline page
- **Premise**: This video shows a beginner how to tell whether code waits on memory or on math, by turning "work per byte" into two speed limits and drawing them as a roof.
- **Hook**: a core needs bytes and arithmetic. Which runs out first?
- **Target audience**: has seen the bandwidth page; has not heard "arithmetic intensity"
- **Length**: about 4 minutes, narrated (no on-screen captions). Each beat lasts as long as its voice clip
- **Key insight**: work per byte picks the limit. f32 decode sits low on the slope, so faster math does not help it and fewer bytes do. A point under its roof (the Phase 0 sum) means something else is slowing it.

## Beats (one new element per line; motion plays while its line is spoken)
1. **Title**.
2. **Two limits**: memory and core boxes; byte packets slide along a belt ("bandwidth, bytes per second"); + × symbols pop in the core ("compute, FLOPs per second"); "which runs out first?"
3. **Work per byte**: a 4-byte float block, an arrow, then FLOP chips: one + (sum, 0.25), × + (GEMV, 0.5), 64 chips in an 8×8 grid (prefill 32 tokens, 16). Each result fades into a table on the right. Only then the name: arithmetic intensity.
4. **Which limit wins**: made-up limits (30 GB/s, 28 GFLOP/s). Gold "core can do 28" bar; teal "memory can feed" bar grows to 15 at 0.5 FLOP/byte; dashed gap, "memory-bound". The work-per-byte value slides to 16, and the teal bar runs to 480 and off the scale: "compute-bound".
5. **The roof**: log-log axes. A gold dot walks along the best possible speed and traces the slope (bandwidth roof), then the flat line (compute roof). The ridge drops in with memory-bound and compute-bound sides; the tick numbers flash for "ten times per step".
6. **Kernels on it**: GEMV on the slope, prefill on the flat. The compute roof doubles: the flat part and the ridge move, but the GEMV dot does not. A 4-bit decode dot slides right, climbs the slope and stops at the flat roof ("we'll have to measure where").
7. **Below the roof**: the Phase 0 sum's roof (hollow copper ring), an arrow down to the measured 0.82 GFLOP/s (teal: measured), and a brace: "waiting on its chain of adds".
8. **Recap**: three lines.

## Color
Same dark paper as the other Phase 1 films. Teal = bytes, bandwidth, memory-bound, and measured values. Gold = arithmetic, compute, the core. Blue = kernels placed on the roof. Copper = the Phase 0 sum below its roof.

## Narration
Voice-over, no captions. `SCRIPT` at the top of `scene.py` is the source; each render writes `script.md` / `script.txt` (the lines to record) and `timeline.json`. Record one clip per line into `audio/<id>.wav`, then:

```
python3 videos/narrate.py check videos/phase-1-roofline
manimgx render videos/phase-1-roofline/scene.py Roofline -r 1280x720 --fps 30 -o silent.mp4
python3 videos/narrate.py mux videos/phase-1-roofline silent.mp4 site/phase-1/roofline.mp4
python3 videos/narrate.py vtt videos/phase-1-roofline site/phase-1/roofline.vtt
```
