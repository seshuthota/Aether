# Narration script

Film: `videos/phase-1-bandwidth/`. Record one clip per line and save it as
`videos/phase-1-bandwidth/audio/<id>.wav` (or .mp3). The id is the file name.
Then re-render: every beat stretches or shrinks to fit the real voice.

| id | line |
|---|---|
| `title` | Bandwidth. How fast can one core read memory? |
| `puzzle-weights` | To write each new token, the model reads every one of its weights from memory. Every token, all of them. |
| `puzzle-formula` | So the number that matters is simple: how many bytes per second can a core read? |
| `puzzle-bandwidth` | That's called bandwidth. We write it in gigabytes per second: billions of bytes, every second. |
| `puzzle-phase0` | Back in Phase 0, we summed a big array on this phone, and it ran at three point three gigabytes per second. |
| `puzzle-question` | So, is that how fast memory is? No. That's how fast the additions were. Let's see why. |
| `chain-one` | Here's that sum: s plus-equals x, one element at a time. Each box is one addition, and time runs to the right. |
| `chain-wait` | Watch. Each addition needs the answer from the one before it. So it waits. And waits again. One long chain. |
| `chain-measured` | So what we timed was additions waiting on each other. Memory was barely trying. |
| `chain-four` | Now split the work into four separate sums, s zero to s three, and add them together at the very end. |
| `chain-finish` | Same eight additions. But the four chains run side by side, and they're finished here, while the single chain is still going. |
| `chain-sooner` | Faster additions ask memory for data sooner. Now memory can finally become the limit. |
| `chain-rule` | And that's what we want to measure. So the rule is: several independent sums, always. |
| `kernel-copy` | There are four classic ways to stream through memory. Copy reads one float and writes one: eight bytes per element. |
| `kernel-scale` | Scale moves the same bytes, plus a multiply. It asks: does a little math slow the stream down? |
| `kernel-add` | Add reads two floats and writes one. Twelve bytes. |
| `kernel-read` | And read only reads, into several sums. Four bytes, and nothing is written back. |
| `kernel-decode` | Decode is this last row. Weights are read, and never written back. |
| `kernel-copy-writes` | Copy also pays for writes, which decode never does. So judge decode against the read, not the copy. |
| `count-run` | One more trap. Take a single copy run: one gigabyte read, and one gigabyte written, in one second. Made-up numbers, just to count. |
| `count-both` | Count both, and you get two gigabytes per second. |
| `count-read` | Count only the read, and the very same run is one gigabyte per second. |
| `count-rule` | Same run, two answers. So pick one way of counting, and write it down. |
| `sweep-intro` | Now the main experiment. Time the read at many buffer sizes, from four kilobytes up to two hundred and fifty-six megabytes. On the right: where the buffer fits. |
| `sweep-l1` | A small buffer fits in L1, the cache closest to the core. The read is fast. |
| `sweep-cliff` | Grow it past L1's size, and the speed falls off a cliff. Now it lives in L2. |
| `sweep-l2` | A new, lower plateau. It stays flat as long as the buffer still fits in L2. |
| `sweep-slc` | Past L2, there may be a small step for the system-level cache. Or just a slope. |
| `sweep-dram-flat` | And from tens of megabytes on, every read goes all the way out to D-RAM. Flat again. |
| `sweep-name` | Only now do we name each flat stretch, once we've seen where the cliffs really are. |
| `sweep-dram` | This flat right end, measured on one pinned core, is the number decode will be judged by. |
| `sweep-not-l1` | Not the tall left end. That's a cache talking, not the phone's memory. |
| `cond-alone` | A bandwidth number on its own can't be compared with next week's. So write down which cores you used, and that the pin worked. Whether the screen was on. Whether it was charging. |
| `cond-more` | The starting temperature. The real clock during the run. And the exact build and commit. |
| `cond-long` | Then run the read for five whole minutes. The first few seconds look great. |
| `cond-hot` | But as the phone heats up, it slows down. This lower number is what a real chat actually gets. |
| `cond-both` | Keep both. They're different results. |
| `recap-1` | So: use several independent sums, or you're measuring waiting. |
| `recap-2` | Bandwidth is bytes divided by seconds. Count the bytes the same way every time. |
| `recap-3` | And sweep the size. The flat right end is D-RAM: decode's yardstick. |

40 lines, 654 words (about 4.2 min of speech).
