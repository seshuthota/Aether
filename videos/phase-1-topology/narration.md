# Cores and pinning — narration script

Timed against `site/phase-1/topology.mp4` (216.4 s). Times come from the scene itself (`narration.json` is rewritten on every render), so they cannot drift from the picture. Each line starts as its caption appears; the motion it describes plays within the next second.

| # | start–end (s) | room (s) | narration |
|---|---|---|---|
| 0 | 0.0–5.4 | 5.4 | Cores and pinning: why a benchmark has to say which core it ran on. *(title card, no caption)* |
| 1 | 5.4–10.2 | 4.8 | Same code. Same phone. Three runs of one benchmark. |
| 2 | 10.2–14.1 | 3.8 | Three different answers. Why? |
| 3 | 14.1–20.8 | 6.8 | To see it, we need a map of the phone's cores. |
| 4 | 20.8–25.0 | 4.1 | Linux gives them numbers: cpu0 to cpu7. |
| 5 | 25.0–30.2 | 5.2 | Cores 0 to 5 can run up to 3.53 GHz. |
| 6 | 30.2–35.4 | 5.2 | Cores 6 and 7 go faster: up to 4.32 GHz. |
| 7 | 35.4–41.8 | 6.4 | So the fast ones are 6 and 7, not 0 and 1. |
| 8 | 41.8–48.8 | 7.0 | The numbers are names, not a ranking. Phase 0 read this from the phone itself. |
| 9 | 48.8–54.7 | 5.9 | All eight cores are the same design. Only the top speed differs. |
| 10 | 54.7–60.4 | 5.7 | There is no slow 'efficiency' core on this phone. |
| 11 | 60.4–66.0 | 5.6 | When the benchmark starts, the scheduler picks a core. Say, cpu2. |
| 12 | 66.0–71.2 | 5.2 | This bar is one run, filling up as time passes. |
| 13 | 71.2–76.9 | 5.7 | Partway through, the scheduler moves it. Then moves it again. |
| 14 | 76.9–82.6 | 5.7 | The result is one number, made on three different cores. |
| 15 | 82.6–89.5 | 6.9 | Next run, the scheduler makes different choices. |
| 16 | 89.5–97.2 | 7.7 | A different mix of cores, so a different number. That was our puzzle. |
| 17 | 97.2–103.1 | 5.9 | The fix: tell the kernel this thread may run only on cpu6. |
| 18 | 103.1–109.0 | 5.9 | Now it can't be moved. The whole run happens on one core. |
| 19 | 109.0–114.7 | 5.6 | That is pinning. The number now names its core. |
| 20 | 114.7–120.6 | 5.9 | The call is sched_setaffinity. You give it a set of allowed cores. |
| 21 | 120.6–126.1 | 5.6 | A set with just one core in it is a pin. |
| 22 | 126.1–134.0 | 7.9 | Always check that the call succeeded. A failed pin you ignore is an unpinned run. |
| 23 | 134.0–141.9 | 7.9 | It does not set the speed. The phone still slows cpu6 down when it heats up. |
| 24 | 141.9–148.7 | 6.8 | So record the real clock during the run, not just the maximum. |
| 25 | 148.7–155.4 | 6.7 | It does not reserve the core. Other programs can still run on cpu6. |
| 26 | 155.4–162.0 | 6.6 | And it does not give the thread its own memory. All cores share DRAM. |
| 27 | 162.0–169.0 | 7.0 | Pinning fixes one thing only: which core runs the thread. |
| 28 | 169.0–173.4 | 4.4 | First, one prime core on its own. |
| 29 | 173.4–180.0 | 6.6 | Then one performance core. Same test. Does the faster clock change the answer? |
| 30 | 180.0–184.1 | 4.1 | Then all six performance cores together. |
| 31 | 184.1–190.7 | 6.6 | Then all eight. 'All cores' is not one thing when the cores differ. |
| 32 | 190.7–198.6 | 7.9 | Whatever you run, write the core set down next to the number. |
| 33 | 198.6–204.5 | 5.9 | Core numbers are names. Here, 6 and 7 are the fast ones. |
| 34 | 204.5–210.1 | 5.6 | An unpinned run blends cores. Pin it to name the core. |
| 35 | 210.1–216.4 | 6.3 | Pinning fixes the core, not the clock. Record both. |

392 words. At ~2.4 words/s every line fits its window with room to spare; the tightest is line 2 ("Three different answers. Why?"), which is a deliberate pause.
