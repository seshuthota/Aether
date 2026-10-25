# Alignment film: narration script

Timed against `site/phase-1/alignment.mp4` (193.8 s), as rendered from `scene.py`.
Times come from `manimgx inspect scene.py Alignment`; re-check them if the scene changes.

- **Start–end:** when the line may start, and when the next line takes over. Speak inside that window.
- **Fit:** estimated speaking time at 2.4 words/s (about 144 words per minute) against the room available.
- Numbers are written the way they should be spoken (“ten twenty-four”), so a voice engine does not read digits one by one.
- 406 words in total, about 2.8 minutes of speech in a 3.2-minute film, so there is breathing room between lines.
- The same lines are in `narration.json` for generating audio.

| # | Start–end | Scene | On screen | Narration | Fit |
|---|---|---|---|---|---|
| 1 | 0:00.0–0:04.6 | Title | Title card: “Alignment”, subtitle fades in | Alignment: where your data starts in memory, and why it matters. | 4.6 / 4.6 s  |
| 2 | 0:04.9–0:12.1 | Bytes | Heading, a row of 12 empty boxes, addresses 0–11 appear | Memory is a long row of boxes. Each holds one byte. Its number is its address. | 6.7 / 7.2 s  |
| 3 | 0:12.1–0:18.5 | Bytes | Boxes 0–3 fill teal, brace “x[0]”; then x[1], x[2] at 16.7 s | A floating-point number, an f32, takes four bytes. Then the next one, and the next. | 6.2 / 6.4 s  |
| 4 | 0:18.5–0:25.1 | Bytes | Three f32s side by side | An array of f32 is just these four-byte groups, side by side, no gaps. | 5.8 / 6.6 s  |
| 5 | 0:25.1–0:32.7 | Lines | Fade; heading “The cache works in 64-byte lines”; strip of 48 small boxes | Let’s zoom out. Each small box is now a whole float: four bytes. | 5.4 / 7.6 s  |
| 6 | 0:32.7–0:39.8 | Lines | Three cache-line bands appear with names | The cache doesn’t fetch single floats. It moves memory in sixty-four-byte lines: sixteen floats per line. | 6.7 / 7.1 s  |
| 7 | 0:39.8–0:48.3 | Lines | Edge ticks drawn at 0, 64, 128, 192 | Here’s the key point. The line edges are fixed by the hardware: zero, sixty-four, one twenty-eight. You can’t move them. | 8.3 / 8.5 s  |
| 8 | 0:48.3–0:55.0 | Loads | Heading changes; one 4-float box highlighted: “one load = 16 bytes” | NEON, the phone’s vector unit, reads sixteen bytes at once: four floats in one instruction. | 6.2 / 6.7 s  |
| 9 | 0:55.0–1:00.4 | Loads | Same picture | So a loop walks through an array in sixteen-byte steps. | 4.2 / 5.4 s  |
| 10 | 1:00.4–1:07.3 | Case 1 | “Case 1: the array starts at 0”; array fills blue; arrow at byte 0; eleven loads drop in, counter stays at 0 | Case one. The array starts at zero, right on a line edge. Watch the loads. | 6.2 / 6.9 s  |
| 11 | 1:07.3–1:13.4 | Case 1 | All eleven loads in place, all teal | Four loads fill each line exactly. Not a single load crosses an edge. | 5.4 / 6.1 s  |
| 12 | 1:13.4–1:20.5 | Case 2 | “Case 2: the array starts at byte 4”; strip clears, array refills shifted; arrow at byte 4 | Case two. Same array, same loop, but it starts just four bytes later. | 5.4 / 7.1 s  |
| 13 | 1:20.5–1:23.0 | Case 2 | Loads 1–4 drop in | Watch the fourth load. | 1.7 / 2.5 s  |
| 14 | 1:23.0–1:25.3 | Case 2 | Load 4 turns copper and flashes; both lines pulse | There. It crosses the edge. | 2.1 / 2.3 s  |
| 15 | 1:25.3–1:27.2 | Case 2 | Load 8 flashes copper | And there’s another. | 1.2 / 1.9 s  |
| 16 | 1:27.2–1:33.4 | Case 2 | Counter: “split loads: 2 of 11” | Every fourth load sticks out over an edge, so it touches two lines. | 5.4 / 6.2 s  |
| 17 | 1:33.4–1:41.7 | Case 2 | Braces over the first split load: “12 bytes in line 0”, “4 bytes in line 1” | Look closer. Twelve of its bytes come from line zero, and the last four come from line one. | 7.5 / 8.3 s  |
| 18 | 1:41.7–1:45.7 | Case 2 | Same picture | That’s called a split load. | 2.1 / 4.0 s  |
| 19 | 1:45.7–1:52.3 | Case 2 | Same picture | A split load still gives the right answer here. It just costs extra work. | 5.8 / 6.6 s  |
| 20 | 1:52.3–1:58.7 | Case 2 | Same picture | How much extra, on this core? That’s something Phase 1 will measure. | 5.0 / 6.4 s  |
| 21 | 1:59.0–2:09.2 | Definition | Heading; rule “aligned to N = multiple of N”; at 123.6 s the code line address % N == 0 | Now the word itself. An address is aligned to N when it’s a multiple of N. In code: address modulo N equals zero. | 9.6 / 10.2 s  |
| 22 | 2:09.2–2:14.6 | Definition | Table headers; row 1024: yes yes yes | Ten twenty-four is a multiple of sixty-four: right on a line edge. | 5.0 / 5.4 s  |
| 23 | 2:14.6–2:20.7 | Definition | Row 1040: yes yes no | Ten forty divides by sixteen, not sixty-four. No splits, but not a line edge. | 5.8 / 6.1 s  |
| 24 | 2:20.7–2:27.1 | Definition | Row 1028: yes no no | Ten twenty-eight is only a multiple of four. Loads will split. | 4.6 / 6.4 s  |
| 25 | 2:27.1–2:33.7 | Definition | Full table | The rule: start every array at a multiple of sixty-four, and no load ever splits. | 6.2 / 6.6 s  |
| 26 | 2:34.2–2:42.2 | Rust | Heading “What Rust promises today”; card Vec<f32> | What does Rust promise today? A Vec of f32 only promises what a float needs: a multiple of four. | 7.9 / 8.0 s  |
| 27 | 2:42.2–2:48.8 | Rust | Same card | In practice it’s often better aligned. But often is luck, not a promise. | 5.4 / 6.6 s  |
| 28 | 2:48.8–2:56.4 | Rust | Second card: aligned buffer, multiple of 64 | So in Phase 1 we build a small buffer whose start is always a multiple of sixty-four. | 7.1 / 7.6 s  |
| 29 | 2:56.4–3:01.6 | Rust | Both cards | Then every benchmark starts from the same layout, every single run. | 4.6 / 5.2 s  |
| 30 | 3:01.6–3:05.8 | Summary | Fade; “In three lines”; point 1 appears at 182.8 s | In short: cache lines are sixty-four bytes, fixed edges. | 3.8 / 4.2 s  |
| 31 | 3:05.8–3:08.8 | Summary | Point 2 appears | Our code reads sixteen bytes at once. | 2.9 / 3.0 s  |
| 32 | 3:08.8–3:13.8 | Summary | Point 3 appears; hold to end | Start arrays at a multiple of sixty-four: no split loads. | 4.2 / 5.0 s  |
