---
description: "Concrete animation techniques, scripting patterns, and Manim-style moves for 3Blue1Brown explainers. Read when writing scripts or storyboarding specific scenes."
connections: [principles, examples]
---

# Techniques

## Scripting Patterns
- Prefer physical language: “the load sticks out over the edge,” “the center of mass snaps right.”
- Mark visual events inline: “[load rectangle walks right] Every fourth load crosses the line.”
- After a reveal, add a pause note: “[hold 0.8 s]”
- Open with motivation or puzzle before mechanism.

## Signature Animation Moves
| Technique | When to Use | Example |
|-----------|-------------|---------|
| Write / Draw | Introduce new lines, text, or boundaries | Cache-line edges draw themselves |
| GrowFromCenter / FadeIn (lagged) | Bring in groups of elements | Memory boxes appear one by one |
| Indicate / Flash / color shift | Emphasize the exact phrase | Split load turns copper as “split load” is said |
| Walking / sliding element | Show process over time | 16-byte load rectangle moves along array |
| Morph / Transform | Transition between cases | Array start slides by 4 bytes; loads re-evaluate |
| Persistent reference grid | Keep context while foreground changes | Faint cache-line grid stays while loads move |
| Animated recap | Close | Each of three summary points appears as spoken |

## Color as Memory
Assign colors early and keep them consistent:
- Neutral / teal for correct or aligned
- Warm / copper for problem or split
- Accent for the active element being discussed

## Timing Guidelines
- New element appears within 0.3 s of the word that introduces it.
- Hold 0.5–1.0 s after a key visual insight.
- Keep individual segments short; vary rhythm so attention does not habituate.

## Manim-Style Pseudocode Notes
```python
# Example: walking load that changes color on split
load = Rectangle(...).set_color(TEAL)
self.play(load.animate.shift(RIGHT * step), run_time=0.8)
if crosses_boundary:
    self.play(load.animate.set_color(ORANGE), Indicate(load), run_time=0.4)
```
Prefer code that makes the motion match the spoken phrase.
