---
name: 3b1b-explainer
description: "Create 3Blue1Brown-style explainer videos. Use when planning, scripting, or storyboarding mathematical, technical, or conceptual explanations with visual-first animation, tight audio-visual synchrony, intuition-before-formalism, and progressive revelation."
type: knowledge
lifecycle: active
---

# 3Blue1Brown-Style Explainer Videos

Produce explanations where the animation *is* the argument. Diagrams build live as ideas form; narration describes motion while it happens; intuition precedes formalism.

## When to Use
- User asks for a 3Blue1Brown-style, Manim-style, or visual-first explainer video.
- Need to turn a technical concept (math, systems, ML, physics, etc.) into a clear, memorable animated explanation.
- Existing slides or static diagrams need to be upgraded to progressive, synchronized animation.

## Core Principles (Non-Negotiable)

1. **Animation is the explanation** — never present a finished complex diagram and then describe it. Elements appear, move, transform, or highlight exactly when the words need them.
2. **Intuition → formalism** — concrete picture or physical metaphor first; name, formula, or definition only after the idea clicks.
3. **Progressive revelation** — one new element at a time. Use lagged starts. Keep prior context visible.
4. **Tight audio-visual sync** — narration describes the motion *while it occurs*. Pause after key reveals. Use silence deliberately.
5. **Calm, measured pacing** — short clear segments + breathing room. Accelerate only after the core idea is solid.
6. **Consistent visual language** — dark background, semantic colors, precise smooth motion.
7. **Early motivation** — open with why it matters, a puzzle, or a surprising consequence.

## Workflow

### 1. Define the One-Sentence Premise
Write: “This video shows [audience] how [outcome] by [mechanism].”
Example: “This video shows why starting an array on a 64-byte boundary eliminates split loads by walking 16-byte NEON loads across fixed cache lines.”

### 2. Outline the Beat Sheet (Intuition First)
- Hook / motivation (5–15 s)
- Concrete picture or metaphor (no formal name yet)
- Step-by-step build of the mechanism (one element at a time)
- Formalization (definition, formula, name) only after intuition is solid
- Concrete examples / edge cases
- Practical payoff or summary recap

### 3. Script with Timing Cues
Write the narration so each sentence maps to a visual event.
- Mark where new elements must appear or move.
- Note deliberate pauses.
- Keep language concrete and physical where possible (“the load sticks out over the edge”).

### 4. Storyboard the Motion
For each beat, specify:
- What is already on screen
- What appears, transforms, or highlights (and how)
- Exact moment it happens relative to the words
- Color / highlight changes that become semantic

### 5. Visual Techniques (Signature Moves)
- Draw / Write text and lines as they are introduced.
- GrowFromCenter, FadeIn with lag, TransformMatching for morphs.
- Indicate / Flash / color shift on the exact phrase being spoken.
- Sliding or walking elements (e.g., a load rectangle that moves along an array and changes color when it crosses a boundary).
- Grid or background that stays as reference while foreground changes.
- Smooth transitions between cases (e.g., array start slides by 4 bytes and loads re-evaluate live).

### 6. Audio–Visual Alignment Rules
- Voice describes the action while the action occurs.
- After a key reveal, hold the picture for a beat before continuing.
- Avoid describing a static image that has already finished animating.
- Prefer “watch what happens when…” over “this diagram shows…”.

### 7. Recap and Close
End with a short animated summary where each key point appears or highlights as it is spoken. Tie back to the original motivation.

## Output Formats
When asked to produce material for such a video, deliver one or more of:
- Timed script with visual cues in brackets.
- Scene-by-scene storyboard (what appears when).
- Manim-style pseudocode or concrete animation notes.
- Side-by-side comparison of static vs. progressive version.

## Common Issues
| Problem | Fix |
|---------|-----|
| Finished diagram explained after the fact | Rebuild so elements enter live with the words |
| Formal definition too early | Move name/formula after the concrete picture has clicked |
| Static slides with voiceover | Convert key moments into motion timed to narration |
| Too many elements at once | Split into lagged successive reveals |
| No pause after key insight | Add 0.5–1 s of hold after the “whoop” moment |

## References
- `references/principles.md` — expanded principles with video examples
- `references/techniques.md` — concrete animation and scripting patterns
- `references/examples.md` — worked examples (alignment, vectors, Fourier-style)
