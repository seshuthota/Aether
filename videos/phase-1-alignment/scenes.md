# Alignment

## Overview
- **Topic**: Alignment and cache lines, as taught on the Phase 1 alignment page
- **Target Audience**: Complete beginner; has seen the memory hierarchy film
- **Length**: about 3 minutes, muted. One idea per beat; captions held ~1.2 s + 0.36 s per word
- **Key Insight**: Cache lines are 64 bytes with fixed edges. NEON reads 16 bytes at a time. An array that starts on a line edge never splits a load; one that starts 4 bytes later splits one load in four. Split loads are correct on AArch64, just slower (unmeasured here).

## Scene beats
1. **Title** — where your data starts in memory, and why it matters.
2. **Bytes** — a row of numbered bytes; four of them make one f32; an array is f32s side by side.
3. **Lines** — zoom out to 48 floats; three 64-byte lines; edges at 0, 64, 128, 192 are fixed.
4. **Loads** — one NEON load is 16 bytes = 4 floats; a loop walks in 16-byte steps.
5. **Case 1** — array starts at 0: eleven loads, 0 split.
6. **Case 2** — array starts at byte 4: 2 of 11 loads split; zoom on one (12 bytes in line 0, 4 in line 1). Correct, just extra work; Phase 1 measures how much.
7. **Definition** — aligned to N = multiple of N (`address % N == 0`); 1024, 1040, 1028 checked one row at a time.
8. **Rust** — `Vec<f32>` promises 4 (better only by luck); the aligned buffer guarantees 64.
9. **Summary** — three lines.

## Color
Dark paper. Neutral bands for cache lines, blue for the array's floats, teal for a load inside one line, copper for a split load.
