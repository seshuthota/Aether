---
description: "Worked examples of applying 3Blue1Brown style to concrete topics, including the alignment/cache-line case. Read when needing a model for a specific explanation."
connections: [principles, techniques]
---

# Examples

## Alignment / Cache Lines (from static slides to progressive)

**Static version**: Finished diagram of cache lines + colored loads, then voiceover explains.

**3b1b-style version**:
1. Memory boxes appear one by one as “memory is a row of numbered bytes.”
2. Four-byte groups highlight as f32s are introduced.
3. Zoom out; cache-line boundaries draw themselves while saying “the cache moves memory in 64-byte lines.”
4. A 16-byte load rectangle appears and walks along an aligned array — no color change.
5. Array start slides 4 bytes right; the same walk is repeated and every fourth load visibly sticks out and turns copper exactly when “every fourth load sticks out” is spoken.
6. Formal definition “address % N == 0” appears only after the two cases have been seen.
7. Recap: three points appear in sequence as spoken.

## Vectors (Essence of Linear Algebra)
- Three perspectives introduced with distinct visuals (arrows, lists of numbers, abstract).
- Coordinate system drawn live.
- Addition shown by actually moving the second arrow so its tail meets the tip of the first.

## Fourier Transform
- Sound waves drawn.
- Graph wraps around a circle in real time.
- Center-of-mass dot appears and snaps right at the matching frequency while the narrator says “whoop.”

## Neural Network
- Starts with a single handwritten digit.
- Input neurons light up according to pixel brightness.
- Hidden layers appear one after another.
- Forward pass is shown as activations propagating before any formula.
