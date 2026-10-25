# Cores and pinning

Planned with the `3b1b-explainer` skill (`.claude/skills/3b1b-explainer/`).

## Overview
- **Topic**: Cores and pinning, as taught on the Phase 1 topology page
- **Premise**: This video shows a beginner why every benchmark must name its core, by watching the scheduler move an unpinned thread across this phone's eight cores and then fencing it into one.
- **Hook**: same code, same phone, three runs, three different answers (bars labelled "illustration, not a measurement")
- **Target audience**: just finished Phase 0; has never heard of the scheduler or affinity
- **Length**: about 3.6 minutes, muted. Captions held for `1.6 + 0.36 × words` s
- **Key insight**: an unpinned number is a blend of whatever cores the scheduler chose; pinning names the core, but not the clock, not the core's exclusivity, and not the memory.

## Beats (one new element per caption; motion plays while its caption is up)
1. **Title** (0–5 s).
2. **Hook** — three bars grow one by one; run 2 pulses copper on "Why?".
3. **The map** — eight empty boxes fade in left to right (cpu0–cpu7); 0–5 fill blue with "3.53 GHz max"; 6–7 fill gold with "4.32 GHz max"; circle around 6–7 on "not 0 and 1"; "same design, part 0x001" line; no efficiency core.
4. **Who picks the core?** — the copper dot "your benchmark" lands on cpu2; a run bar fills blue, the dot hops to cpu6 (bar turns gold), then cpu4. Run 2 fills with a different mix (cpu7, cpu0, cpu5). Ties back to the hook.
5. **Pinning** — teal fence around cpu6, others dim; the dot moves in; the run bar fills gold end to end, "cpu6 only". Then the name: `sched_setaffinity(thread, {6})`, "a set of one is a pin", and the copper warning to check the return value.
6. **What pinning does not do** — the cpu6 clock readout falls 4.32 → 2.80 GHz (labelled illustration); a grey "another app" dot slides onto cpu6 beside ours; a shared DRAM bar with wires from all eight cores.
7. **Which cores to measure** — the fence jumps between sets as each is named: {6}, {0}, {0–5}, {0–7}. Write the set down.
8. **Recap** — three lines appear one at a time as they are said (gold, teal, copper).

## Color
Same dark paper as the other Phase 1 films. Blue = performance core, gold = prime core, teal = pinned/allowed set, copper = our thread and every warning, grey = someone else.

## Narration
`narration.json` is written by the scene on every render: each caption's real start time, end time, on-screen caption and voice line. `narration.md` is the readable version. To add voice, generate one clip per line and place it at its `start`; if a clip runs long, add `extra=` to that `cap()`/`say()` call and re-render so the film waits for the voice.
