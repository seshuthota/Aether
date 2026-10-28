# Narration script

Film: `videos/phase-1-roofline/`. Record one clip per line and save it as
`videos/phase-1-roofline/audio/<id>.wav` (or .mp3). The id is the file name.
Then re-render: every beat stretches or shrinks to fit the real voice.

| id | line |
|---|---|
| `title` | The roofline. Is your code waiting on memory, or on math? |
| `two-bytes` | A core needs two things. First, bytes. They arrive from memory at some rate: that's bandwidth, bytes per second. |
| `two-math` | Second, arithmetic. The core adds and multiplies those numbers. Each add or multiply is one FLOP, and the core can do so many per second. |
| `two-question` | So which one runs out first? It comes down to a single number. |
| `wpb-sum` | Take the Phase 0 sum. One float is four bytes, and it gets one add. One FLOP for four bytes: a quarter of a FLOP per byte. |
| `wpb-gemv` | Decode's matrix times vector. Each four-byte weight gets a multiply and an add. Two FLOPs for four bytes: half a FLOP per byte. |
| `wpb-prefill` | Prefill with thirty-two tokens reuses that same weight thirty-two times. Sixty-four FLOPs from the same four bytes: sixteen per byte. |
| `wpb-name` | FLOPs per byte. It's called arithmetic intensity: how much work each byte brings with it. |
| `limit-setup` | Now some made-up limits. Memory delivers thirty gigabytes per second. The core can do twenty-eight billion FLOPs per second. |
| `limit-feed` | At half a FLOP per byte, thirty gigabytes a second can only feed fifteen billion FLOPs a second. |
| `limit-gemv` | The shorter bar is the limit. The core sits idle about half the time, waiting for bytes. That's memory-bound. |
| `limit-prefill` | Now slide up to sixteen FLOPs per byte. Memory could feed four hundred and eighty. |
| `limit-compute` | Way more than the core can use. Now the core is the limit. That's compute-bound. |
| `roof-axes` | Do that for every possible work per byte, and plot it. Work per byte along the bottom, the best possible speed up the side. |
| `roof-slope` | On the left, more work per byte means more speed, because every byte brings more math with it. That's the bandwidth roof. |
| `roof-flat` | Then it hits the core's peak, and goes flat. That's the compute roof. |
| `roof-ridge` | Where the two meet is the ridge. Left of it, memory-bound. Right of it, compute-bound. |
| `roof-log` | Each step along both axes is ten times bigger. That's what keeps the slope a straight line. |
| `pts-gemv` | Decode's matrix times vector lands here, low on the slope. |
| `pts-prefill` | Prefill lands up here, on the flat roof. |
| `pts-raise` | Watch what happens if we double the core's speed. The flat roof rises. The decode dot doesn't move at all. |
| `pts-lesson` | For decode, faster math doesn't help. Reading fewer bytes does. |
| `pts-q4` | Store each weight in four bits, and the same math needs about seven times fewer bytes. The dot slides right, and climbs the slope. |
| `pts-q4-flat` | Until it meets the flat roof. From there, the math is the limit again. Where exactly that happens, we'll have to measure. |
| `below-sum` | One last point. The Phase 0 sum, at a quarter of a FLOP per byte. Its roof is up here. |
| `below-real` | But we measured it way down here: point eight two billion FLOPs per second. Far below its roof. |
| `below-why` | A roof is a limit, not a promise. The sum wasn't waiting on memory. It was waiting on its own chain of adds. |
| `below-gap` | A point under the roof means something else is slowing it down. Finding out what, is the skill. |
| `recap-1` | So: work per byte decides which limit you hit. |
| `recap-2` | Left of the ridge, memory is the limit. Decode lives there. |
| `recap-3` | And the roof is only an upper bound. If you're below it, go find out why. |

31 lines, 539 words (about 3.5 min of speech).
