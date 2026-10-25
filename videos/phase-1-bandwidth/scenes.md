# Bandwidth

Planned with the `3b1b-explainer` skill (`.claude/skills/3b1b-explainer/`).

## Overview
- **Topic**: Bandwidth, as taught on the Phase 1 bandwidth page
- **Premise**: This video shows a beginner how to measure the one number decode lives on, how fast a core can read memory, by showing why one running sum measures waiting, counting bytes honestly, and growing a buffer until its speed falls from the caches to DRAM.
- **Hook**: Phase 0's sum read at 3.3 GB/s (measured, teal). Is that the memory? No, it is the adds.
- **Target audience**: has seen the memory hierarchy page; has not written a benchmark kernel
- **Length**: about 4.5 minutes, narrated (no on-screen captions). Each beat lasts as long as its voice clip
- **Key insight**: the yardstick for decode is the pure read with several independent sums, at a DRAM-sized buffer, on a named core, with the bytes counted one fixed way.

## Beats (one new element per caption; motion plays while its caption is up)
1. **Title** (0–5 s).
2. **Puzzle**: weights fly from a memory block into the core (copies; memory keeps them). Then "speed = bytes read ÷ seconds", then the word bandwidth / GB/s. Then the Phase 0 number, 3.3 GB/s, which pulses copper on "how fast the adds were".
3. **One running sum is a chain**: a time arrow. One copper row `s` of eight "+" boxes appears one box at a time (each waits for the one before). Then four teal rows `s0–s3` fill side by side; a dashed line marks where they finish while the copper chain is still going.
4. **Four kernels**: copy, scale, add and read appear one row per caption, each with its traffic drawn as read (blue) and write (copper) blocks and a byte count. A teal frame goes round the read row (decode never writes weights); the copy row's write block pulses.
5. **Count bytes honestly**: one copy run (1 GB read + 1 GB written in 1 s, labelled illustration) becomes 2 GB/s or 1 GB/s depending on what you count.
6. **The sweep**: axes and a "where it fits" ladder (L1, L2, SLC, DRAM). The curve draws itself as the buffer grows from 4 KB to 256 MB, with a size readout on the dot. The lit rung follows the size, so it changes exactly at each cliff. The plateau names arrive only after the whole curve exists; the DRAM tail thickens copper ("the number decode is judged by"); L1 pulses ("not the tall left end").
7. **Conditions**: a six-item checklist in two groups, then a five-minute run line that starts high and settles lower once the phone heats up.
8. **Recap**: three lines appear one at a time as they are said.

## Color
Same dark paper as the other Phase 1 films. Teal = measured, and the good habit (independent sums, the read row). Copper = the trap (one chain), writes, and DRAM. Blue = reads, and L2/SLC. Gold = the moving buffer-size dot and the long-run line.

## Narration
Voice-over, no captions. `SCRIPT` at the top of `scene.py` is the source; each render writes `script.md` / `script.txt` (the lines to record) and `timeline.json` (when each line starts). Record one clip per line into `audio/<id>.wav`, re-render, then:

```
python3 videos/narrate.py check videos/phase-1-bandwidth           # Whisper on Groq: does each clip say its line?
manimgx render videos/phase-1-bandwidth/scene.py Bandwidth -r 1280x720 --fps 30 -o silent.mp4
python3 videos/narrate.py mux videos/phase-1-bandwidth silent.mp4 site/phase-1/bandwidth.mp4
python3 videos/narrate.py vtt videos/phase-1-bandwidth site/phase-1/bandwidth.vtt
```
