"""Explainer for site/phase-1/roofline.html.

Planned with the 3b1b-explainer skill: two limits as a picture (bytes on a
belt, math in the core) before any chart; work per byte counted from three
of our own kernels; the limit picked by two bars; and only then the roof,
traced by a dot so the viewer sees where it comes from. The roofs are made
up (30 GB/s, 28 GFLOP/s) and labelled so; the one measured point is the
Phase 0 sum.

Narrated, no captions: SCRIPT below is the voice-over, and each beat lasts
as long as its recorded clip in audio/ (see videos/narrate.py).
"""

import math
import sys
from pathlib import Path

import manimgx as m

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from narrate import Narrated  # noqa: E402

# Written to be heard: numbers spelled the way they are said, and each line
# describes what is moving on screen while it is spoken.
SCRIPT = [
    ("title", "The roofline. Is your code waiting on memory, or on math?"),
    ("two-bytes", "A core needs two things. First, bytes. They arrive from memory at some "
                  "rate: that's bandwidth, bytes per second."),
    ("two-math", "Second, arithmetic. The core adds and multiplies those numbers. Each add "
                 "or multiply is one FLOP, and the core can do so many per second."),
    ("two-question", "So which one runs out first? It comes down to a single number."),
    ("wpb-sum", "Take the Phase 0 sum. One float is four bytes, and it gets one add. One "
                "FLOP for four bytes: a quarter of a FLOP per byte."),
    ("wpb-gemv", "Decode's matrix times vector. Each four-byte weight gets a multiply and an "
                 "add. Two FLOPs for four bytes: half a FLOP per byte."),
    ("wpb-prefill", "Prefill with thirty-two tokens reuses that same weight thirty-two times. "
                    "Sixty-four FLOPs from the same four bytes: sixteen per byte."),
    ("wpb-name", "FLOPs per byte. It's called arithmetic intensity: how much work each byte "
                 "brings with it."),
    ("limit-setup", "Now some made-up limits. Memory delivers thirty gigabytes per second. "
                    "The core can do twenty-eight billion FLOPs per second."),
    ("limit-feed", "At half a FLOP per byte, thirty gigabytes a second can only feed fifteen "
                   "billion FLOPs a second."),
    ("limit-gemv", "The shorter bar is the limit. The core sits idle about half the time, "
                   "waiting for bytes. That's memory-bound."),
    ("limit-prefill", "Now slide up to sixteen FLOPs per byte. Memory could feed four hundred "
                      "and eighty."),
    ("limit-compute", "Way more than the core can use. Now the core is the limit. That's "
                      "compute-bound."),
    ("roof-axes", "Do that for every possible work per byte, and plot it. Work per byte "
                  "along the bottom, the best possible speed up the side."),
    ("roof-slope", "On the left, more work per byte means more speed, because every byte "
                   "brings more math with it. That's the bandwidth roof."),
    ("roof-flat", "Then it hits the core's peak, and goes flat. That's the compute roof."),
    ("roof-ridge", "Where the two meet is the ridge. Left of it, memory-bound. Right of it, "
                   "compute-bound."),
    ("roof-log", "Each step along both axes is ten times bigger. That's what keeps the "
                 "slope a straight line."),
    ("pts-gemv", "Decode's matrix times vector lands here, low on the slope."),
    ("pts-prefill", "Prefill lands up here, on the flat roof."),
    ("pts-raise", "Watch what happens if we double the core's speed. The flat roof rises. "
                  "The decode dot doesn't move at all."),
    ("pts-lesson", "For decode, faster math doesn't help. Reading fewer bytes does."),
    ("pts-q4", "Store each weight in four bits, and the same math needs about seven times "
               "fewer bytes. The dot slides right, and climbs the slope."),
    ("pts-q4-flat", "Until it meets the flat roof. From there, the math is the limit again. "
                    "Where exactly that happens, we'll have to measure."),
    ("below-sum", "One last point. The Phase 0 sum, at a quarter of a FLOP per byte. Its "
                  "roof is up here."),
    ("below-real", "But we measured it way down here: point eight two billion FLOPs per "
                   "second. Far below its roof."),
    ("below-why", "A roof is a limit, not a promise. The sum wasn't waiting on memory. It "
                  "was waiting on its own chain of adds."),
    ("below-gap", "A point under the roof means something else is slowing it down. Finding "
                  "out what, is the skill."),
    ("recap-1", "So: work per byte decides which limit you hit."),
    ("recap-2", "Left of the ridge, memory is the limit. Decode lives there."),
    ("recap-3", "And the roof is only an upper bound. If you're below it, go find out why."),
]

BG = "#121816"
INK = "#f3efe6"
MUTED = "#b7b0a4"
FAINT = "#5b625e"
TEAL = "#7dcec4"
BLUE = "#9ec0e8"
GOLD = "#e8c46a"
COPPER = "#e0a07a"

m.config.background_color = BG

BW = 30.0  # made-up bandwidth roof, GB/s
PEAK = 28.0  # made-up compute roof, GFLOP/s

# Roofline chart: log10 ranges mapped onto the frame.
AX0, AX1 = -1.0, 1.6  # work per byte 0.1 .. 40
AY0, AY1 = math.log10(0.3), math.log10(200)  # GFLOP/s 0.3 .. 200
CX0, CX1 = -5.0, 5.6
CY0, CY1 = -2.7, 2.4


def T(text, size=30, color=INK):
    return m.Text(text, font="sans-serif", font_size=size, color=color)


def C(text, size=28, color=INK):
    return m.Text(text, font="monospace", font_size=size, color=color)


def box(width, height, color, opacity=1.0, stroke=None):
    return m.Rectangle(
        width=width,
        height=height,
        stroke_width=2,
        stroke_color=stroke or color,
        fill_color=color,
        fill_opacity=opacity,
        color=stroke or color,
    )


def cx(a):
    return CX0 + (math.log10(a) - AX0) / (AX1 - AX0) * (CX1 - CX0)


def cy(v):
    return CY0 + (math.log10(v) - AY0) / (AY1 - AY0) * (CY1 - CY0)


def pt(a, v):
    return [cx(a), cy(v), 0]


def roof(a, peak):
    return min(peak, a * BW)


class Roofline(Narrated, m.Scene):
    def construct(self):
        self.start_voice(__file__, SCRIPT)
        self.title_card()
        self.two_limits()
        self.work_per_byte()
        self.which_limit()
        self.the_roof()
        self.kernels()
        self.below()
        self.summary()
        self.finish_voice()

    def clear(self):
        mobs = list(self.mobjects)
        if mobs:
            self.play(*[m.FadeOut(mob) for mob in mobs], run_time=0.5)
        self.wait(0.3)

    def heading(self, text):
        h = T(text, 38, INK).to_edge(m.UP, buff=0.4)
        self.play(m.FadeIn(h), run_time=0.4)
        return h

    # ----------------------------------------------------------------- title
    def title_card(self):
        kicker = T("Phase 1", 24, TEAL)
        title = T("The roofline", 60, INK)
        sub = T("waiting on memory, or on math?", 30, MUTED)
        m.VGroup(kicker, title, sub).arrange(m.DOWN, buff=0.3)
        self.vo("title")
        self.play(m.FadeIn(kicker), m.Write(title), run_time=1.1)
        self.play(m.FadeIn(sub, shift=m.UP * 0.15))
        self.hold(0.6)
        self.clear()

    # ------------------------------------------------------------ two limits
    def two_limits(self):
        mem = m.VGroup(box(2.4, 2.0, INK, 0.06, stroke=MUTED), T("memory", 28, MUTED))
        mem[1].move_to(mem[0])
        mem.move_to([-4.6, 0.6, 0])
        core = m.VGroup(box(2.4, 2.0, BLUE, 0.15), T("core", 28, BLUE))
        core[1].move_to(core[0]).shift(m.UP * 0.55)
        core.move_to([4.6, 0.6, 0])
        belt = m.Line([-3.3, 0.6, 0], [3.3, 0.6, 0], color=FAINT, stroke_width=3)
        bw_lbl = m.VGroup(T("bandwidth", 28, TEAL), T("bytes per second", 22, MUTED))
        bw_lbl.arrange(m.DOWN, buff=0.1).next_to(belt, m.DOWN, buff=0.5)

        self.vo("two-bytes")
        self.play(m.FadeIn(mem), m.FadeIn(core), m.Create(belt), run_time=0.8)
        packets = [box(0.36, 0.36, TEAL, 0.8).move_to([-3.1, 0.6, 0]) for _ in range(8)]
        self.play(m.LaggedStart(*[m.Succession(m.FadeIn(p, run_time=0.1),
                                               p.animate(run_time=1.6, rate_func=m.linear)
                                               .move_to([3.2, 0.6, 0]),
                                               m.FadeOut(p, run_time=0.1))
                                  for p in packets], lag_ratio=0.25), run_time=3.4)
        self.play(m.FadeIn(bw_lbl, shift=m.UP * 0.1), run_time=0.5)
        self.hold()

        ops = m.VGroup(*[T(sym, 34, GOLD) for sym in "+×+×"]).arrange(m.RIGHT, buff=0.3)
        ops.move_to(core[0]).shift(m.DOWN * 0.3)
        fl_lbl = m.VGroup(T("compute", 28, GOLD), T("FLOPs per second", 22, MUTED))
        fl_lbl.arrange(m.DOWN, buff=0.1).next_to(core, m.DOWN, buff=0.4)
        self.vo("two-math")
        self.play(m.LaggedStart(*[m.FadeIn(o, scale=1.4) for o in ops], lag_ratio=0.35),
                  run_time=1.6)
        self.play(m.FadeIn(fl_lbl, shift=m.UP * 0.1), run_time=0.5)
        self.hold()

        q = T("which runs out first?", 34, INK).move_to([0, 2.7, 0])
        self.vo("two-question")
        self.play(m.FadeIn(q), m.Indicate(bw_lbl[0], color=TEAL), m.Indicate(fl_lbl[0], color=GOLD),
                  run_time=1.2)
        self.hold()
        self.clear()

    # --------------------------------------------------------- work per byte
    def float_block(self, color):
        cells = m.VGroup(*[box(0.42, 0.42, color, 0.75) for _ in range(4)]).arrange(m.RIGHT, buff=0.04)
        lbl = T("4 bytes", 22, MUTED).next_to(cells, m.DOWN, buff=0.12)
        return m.VGroup(cells, lbl)

    def chips(self, symbols, cols, size=0.34):
        out = m.VGroup()
        for sym in symbols:
            chip = m.VGroup(box(size, size, GOLD, 0.25), T(sym, int(size * 100), GOLD))
            chip[1].move_to(chip[0])
            out.add(chip)
        return out.arrange_in_grid(cols=cols, buff=0.06)

    def work_per_byte(self):
        h = self.heading("Work per byte")
        table = m.VGroup()
        rows = [("Phase 0 sum", "0.25"), ("decode GEMV", "0.5"), ("prefill, 32 tokens", "16")]
        for name, val in rows:
            table.add(m.VGroup(T(name, 26, MUTED), C(val, 28, INK)).arrange(m.RIGHT, buff=0.4))
        table.arrange(m.DOWN, buff=0.4, aligned_edge=m.RIGHT).move_to([4.6, 0.2, 0])

        def demo(key, sym, cols, result, color, row, chip_size=0.34):
            block = self.float_block(color).move_to([-4.6, 0.6, 0])
            work = self.chips(sym, cols, chip_size).move_to([-1.2, 0.6, 0])
            arrow = m.Arrow(block.get_right(), work.get_left(), color=FAINT, buff=0.25)
            eq = C(result, 30, INK).move_to([-2.6, -1.6, 0])
            self.vo(key)
            self.play(m.FadeIn(block, shift=m.RIGHT * 0.2), run_time=0.5)
            self.play(m.GrowArrow(arrow), run_time=0.4)
            self.play(m.LaggedStart(*[m.FadeIn(c, scale=1.3) for c in work],
                                    lag_ratio=min(0.3, 3.0 / len(work))),
                      run_time=min(2.4, 0.5 + 0.08 * len(work)))
            self.play(m.FadeIn(eq, shift=m.UP * 0.1), run_time=0.5)
            self.play(m.FadeIn(table[row], shift=m.LEFT * 0.3), run_time=0.6)
            self.hold()
            self.play(m.FadeOut(m.VGroup(block, work, arrow, eq)), run_time=0.4)

        demo("wpb-sum", "+", 1, "1 FLOP ÷ 4 bytes = 0.25", BLUE, 0)
        demo("wpb-gemv", "×+", 2, "2 FLOPs ÷ 4 bytes = 0.5", BLUE, 1)
        demo("wpb-prefill", "×+" * 32, 8, "64 FLOPs ÷ 4 bytes = 16", BLUE, 2, chip_size=0.3)

        name = m.VGroup(T("FLOPs per byte", 30, INK), T("= arithmetic intensity", 30, TEAL))
        name.arrange(m.DOWN, buff=0.15).next_to(table, m.UP, buff=0.5)
        self.vo("wpb-name")
        self.play(m.FadeIn(name[0]), run_time=0.5)
        self.play(m.Write(name[1]), run_time=0.9)
        self.hold()
        self.clear()

    # ----------------------------------------------------------- which limit
    def which_limit(self):
        h = self.heading("Which limit wins?")
        note = T("made-up limits, not this phone", 22, MUTED).next_to(h, m.DOWN, buff=0.15)
        scale = 4.0 / PEAK  # scene units per GFLOP/s
        x0 = -3.6
        cap = 8.6
        ai = m.ValueTracker(0.5)

        def feed_bar():
            v = ai.get_value() * BW
            w = min(cap, v * scale)
            bar = box(max(w, 0.02), 0.55, TEAL, 0.8).move_to([x0 + w / 2, 0.7, 0])
            return bar

        feed = m.always_redraw(feed_bar)
        feed_val = m.always_redraw(lambda: C(f"{ai.get_value() * BW:.0f}", 28, TEAL).next_to(
            [x0 + min(cap, ai.get_value() * BW * scale), 0.7, 0], m.RIGHT, buff=0.2))
        core_bar = box(PEAK * scale, 0.55, GOLD, 0.8).move_to([x0 + PEAK * scale / 2, -0.6, 0])
        core_val = C("28", 28, GOLD).next_to(core_bar, m.RIGHT, buff=0.2)
        feed_name = T("memory can feed", 26, TEAL).next_to([x0, 0.7, 0], m.LEFT, buff=0.3)
        core_name = T("core can do", 26, GOLD).next_to([x0, -0.6, 0], m.LEFT, buff=0.3)
        unit = T("billion FLOPs per second", 22, MUTED).move_to([0.6, -1.5, 0])
        ai_lbl = m.always_redraw(lambda: C(f"work per byte: {ai.get_value():.1f}", 28, INK)
                                 .move_to([0, 1.95, 0]))

        given = m.VGroup(T("memory: 30 GB/s", 26, TEAL), T("core: 28 GFLOP/s", 26, GOLD))
        given.arrange(m.RIGHT, buff=1.2).move_to([0, -2.4, 0])
        self.vo("limit-setup")
        self.play(m.FadeIn(note), m.FadeIn(given[0]), run_time=0.5)
        self.play(m.FadeIn(given[1]), run_time=0.5)
        self.play(m.FadeIn(core_name), m.GrowFromEdge(core_bar, m.LEFT), m.FadeIn(core_val),
                  m.FadeIn(unit), run_time=0.9)
        self.hold()

        self.vo("limit-feed")
        self.play(m.FadeIn(ai_lbl), m.FadeIn(feed_name), run_time=0.4)
        self.add(feed, feed_val)
        self.play(m.FadeIn(feed), run_time=0.6)
        self.hold()

        gap = m.DashedLine([x0 + 15 * scale, 0.35, 0], [x0 + 15 * scale, -0.95, 0], color=MUTED)
        verdict = T("memory-bound", 32, TEAL).move_to([4.6, 0.05, 0])
        self.vo("limit-gemv", after=1.0)
        self.play(m.Create(gap), run_time=0.5)
        self.play(m.Indicate(core_bar, color=FAINT, scale_factor=1.02), run_time=0.8)
        self.play(m.FadeIn(verdict, shift=m.LEFT * 0.2), run_time=0.5)
        self.hold()

        self.vo("limit-prefill")
        self.play(m.FadeOut(gap), m.FadeOut(verdict), run_time=0.3)
        self.play(ai.animate.set_value(16), run_time=2.4, rate_func=m.smooth)
        # Done moving: freeze the live bar and labels so later waits still draw them.
        for mob in (feed, feed_val, ai_lbl):
            mob.clear_updaters()
        self.hold()

        verdict2 = T("compute-bound", 32, GOLD).move_to([4.6, -1.05, 0])
        self.vo("limit-compute", after=1.0)
        self.play(m.Indicate(core_bar, color=GOLD, scale_factor=1.05), run_time=0.9)
        self.play(m.FadeIn(verdict2, shift=m.LEFT * 0.2), run_time=0.5)
        self.hold()
        self.remove(feed, feed_val, ai_lbl)
        self.clear()

    # --------------------------------------------------------------- the roof
    def axes(self):
        x_axis = m.Line(pt(0.1, 0.3), [CX1 + 0.3, CY0, 0], color=FAINT, stroke_width=2)
        y_axis = m.Line(pt(0.1, 0.3), [CX0, CY1 + 0.2, 0], color=FAINT, stroke_width=2)
        ticks = m.VGroup()
        for a in (0.1, 1, 10):
            ticks.add(C(f"{a:g}", 20, MUTED).next_to(pt(a, 0.3), m.DOWN, buff=0.12))
        for v in (1, 10, 100):
            ticks.add(C(f"{v:g}", 20, MUTED).next_to(pt(0.1, v), m.LEFT, buff=0.12))
        x_lbl = T("work per byte (FLOP/byte)", 22, MUTED).next_to(x_axis, m.DOWN, buff=0.5)
        y_lbl = T("best speed (GFLOP/s)", 22, MUTED).rotate(math.pi / 2).next_to(
            y_axis, m.LEFT, buff=0.6)
        return m.VGroup(m.VGroup(), x_axis, y_axis, ticks, x_lbl, y_lbl)

    def the_roof(self):
        h = self.heading("The roofline")
        note = T("made-up roofs: 30 GB/s, 28 GFLOP/s", 22, MUTED).next_to(h, m.DOWN, buff=0.12)
        axes = self.axes()
        self.vo("roof-axes")
        self.play(m.FadeIn(note), m.Create(axes[1]), m.Create(axes[2]), run_time=0.8)
        self.play(m.FadeIn(axes[3]), m.FadeIn(axes[4]), m.FadeIn(axes[5]), run_time=0.6)
        self.hold()

        peak = m.ValueTracker(PEAK)
        a = m.ValueTracker(0.1)
        self.peak = peak

        def roof_path(lo, hi):
            return m.ParametricFunction(lambda t: pt(10 ** t, roof(10 ** t, peak.get_value())),
                                        t_range=[lo, hi, 0.005], color=INK, stroke_width=5)

        traced = m.always_redraw(lambda: roof_path(AX0, max(AX0 + 0.01, math.log10(a.get_value()))))
        dot = m.always_redraw(lambda: m.Dot(pt(a.get_value(), roof(a.get_value(), peak.get_value())),
                                            radius=0.1, color=GOLD))
        self.add(traced, dot)
        ridge_a = PEAK / BW
        # On log-log axes the slope rises one decade per decade; tilt the label to match.
        tilt = math.atan(((CY1 - CY0) / (AY1 - AY0)) / ((CX1 - CX0) / (AX1 - AX0)))
        slope_lbl = T("bandwidth roof", 26, TEAL).rotate(tilt).move_to(pt(0.25, 0.25 * BW)).shift(
            m.DOWN * 0.5 + m.RIGHT * 0.3)
        self.vo("roof-slope")
        self.play(a.animate.set_value(ridge_a), run_time=3.0, rate_func=m.linear)
        self.play(m.FadeIn(slope_lbl), run_time=0.5)
        self.hold()

        flat_lbl = T("compute roof", 26, GOLD).move_to(pt(12, PEAK)).shift(m.UP * 0.35)
        self.vo("roof-flat")
        self.play(a.animate.set_value(40), run_time=2.2, rate_func=m.linear)
        self.play(m.FadeIn(flat_lbl), run_time=0.5)
        self.hold()
        self.remove(dot)

        ridge = m.always_redraw(lambda: m.DashedLine(
            pt(peak.get_value() / BW, 0.3), pt(peak.get_value() / BW, peak.get_value()),
            color=MUTED, stroke_width=2))
        ridge_lbl = T("ridge", 24, MUTED).next_to(pt(ridge_a, 0.3), m.UP, buff=0.15).shift(m.RIGHT * 0.4)
        left = T("memory-bound", 24, TEAL).move_to(pt(0.32, 0.7))
        right = T("compute-bound", 24, GOLD).move_to(pt(9, 2.0))
        self.vo("roof-ridge", after=0.8)
        self.play(m.Create(ridge), m.FadeIn(ridge_lbl), run_time=0.7)
        self.play(m.FadeIn(left, shift=m.LEFT * 0.2), run_time=0.5)
        self.play(m.FadeIn(right, shift=m.RIGHT * 0.2), run_time=0.5)
        self.hold()

        self.vo("roof-log")
        self.play(m.LaggedStart(*[m.Indicate(t, color=INK) for t in axes[3]], lag_ratio=0.15),
                  run_time=1.6)
        self.hold()
        self.chart = m.VGroup(h, note, axes, slope_lbl, flat_lbl, ridge_lbl, left, right)
        self.roof_mobs = (traced, ridge)
        self.a = a

    # --------------------------------------------------------------- kernels
    def kernels(self):
        peak = self.peak
        gemv = m.always_redraw(lambda: m.Dot(pt(0.5, roof(0.5, peak.get_value())), radius=0.11,
                                             color=BLUE))
        gemv_lbl = T("decode GEMV", 22, BLUE).next_to(pt(0.5, 15), m.UL, buff=0.12)
        self.vo("pts-gemv")
        self.play(m.FadeOut(self.chart[6]), m.FadeOut(self.chart[7]), run_time=0.4)
        self.add(gemv)
        self.play(m.FadeIn(gemv, scale=2), m.FadeIn(gemv_lbl), run_time=0.6)
        self.hold()

        pre = m.always_redraw(lambda: m.Dot(pt(16, roof(16, peak.get_value())), radius=0.11,
                                            color=BLUE))
        pre_lbl = m.always_redraw(lambda: T("prefill, 32 tokens", 22, BLUE).next_to(
            pt(16, roof(16, peak.get_value())), m.DOWN, buff=0.2))
        self.vo("pts-prefill")
        self.add(pre, pre_lbl)
        self.play(m.FadeIn(pre, scale=2), m.FadeIn(pre_lbl), run_time=0.6)
        self.hold()

        self.vo("pts-raise", after=0.8)
        flat_lbl = self.chart[4]
        self.play(peak.animate.set_value(2 * PEAK),
                  flat_lbl.animate.shift(m.UP * (cy(2 * PEAK) - cy(PEAK))),
                  self.chart[5].animate.shift(m.RIGHT * (cx(2 * PEAK / BW) - cx(PEAK / BW))),
                  run_time=2.4)
        self.play(m.Indicate(gemv_lbl, color=BLUE), run_time=0.9)
        self.hold()
        self.vo("pts-lesson")
        self.play(peak.animate.set_value(PEAK),
                  flat_lbl.animate.shift(m.DOWN * (cy(2 * PEAK) - cy(PEAK))),
                  self.chart[5].animate.shift(m.LEFT * (cx(2 * PEAK / BW) - cx(PEAK / BW))),
                  run_time=1.2)
        self.hold()

        q = m.ValueTracker(0.5)
        q4 = m.always_redraw(lambda: m.Dot(pt(q.get_value(), roof(q.get_value(), peak.get_value())),
                                           radius=0.11, color=TEAL))
        q4_lbl = m.always_redraw(lambda: T("4-bit decode", 22, TEAL).next_to(
            pt(q.get_value(), roof(q.get_value(), peak.get_value())), m.UL, buff=0.12))
        self.vo("pts-q4")
        self.add(q4)
        self.play(m.FadeIn(q4, scale=2), run_time=0.4)
        self.play(q.animate.set_value(PEAK / BW), run_time=2.6, rate_func=m.linear)
        self.add(q4_lbl)
        self.play(m.FadeIn(q4_lbl), run_time=0.4)
        self.hold()
        self.vo("pts-q4-flat")
        self.play(q.animate.set_value(2 / (18 / 32)), run_time=1.6, rate_func=m.linear)
        self.play(m.Indicate(self.chart[4], color=GOLD), run_time=0.9)
        self.hold()
        self.play(m.FadeOut(q4), m.FadeOut(q4_lbl), m.FadeOut(pre), m.FadeOut(pre_lbl), run_time=0.4)
        self.remove(q4, q4_lbl, pre, pre_lbl)
        self.gemv = (gemv, gemv_lbl)

    # ------------------------------------------------------------------ below
    def below(self):
        ghost = m.Circle(radius=0.12, color=COPPER, stroke_width=3).move_to(pt(0.25, 0.25 * BW))
        ghost_lbl = T("its roof", 22, COPPER).next_to(ghost, m.RIGHT, buff=0.2).shift(m.DOWN * 0.15)
        self.vo("below-sum")
        self.play(m.FadeOut(self.chart[3]), m.FadeOut(self.chart[5]), m.FadeOut(self.roof_mobs[1]),
                  run_time=0.3)
        self.remove(self.roof_mobs[1])
        self.play(m.Create(ghost), m.FadeIn(ghost_lbl), run_time=0.7)
        self.hold()

        real = m.Dot(pt(0.25, 0.82), radius=0.12, color=COPPER)
        real_lbl = m.VGroup(T("Phase 0 sum", 22, COPPER), T("measured: 0.82 GFLOP/s", 22, TEAL))
        real_lbl.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT).next_to(real, m.RIGHT, buff=0.25)
        drop = m.Arrow(ghost.get_bottom(), real.get_top(), color=COPPER, buff=0.1)
        self.vo("below-real", after=0.8)
        self.play(m.GrowArrow(drop), run_time=1.0)
        self.play(m.FadeIn(real, scale=2), m.FadeIn(real_lbl), run_time=0.6)
        self.hold()

        gap = m.Brace(m.Line(pt(0.25, 1.2), pt(0.25, 0.25 * BW * 0.8)), m.RIGHT, color=MUTED)
        gap.shift(m.RIGHT * 0.25)
        why = T("waiting on its chain of adds", 22, MUTED).next_to(gap, m.RIGHT, buff=0.15)
        self.vo("below-why")
        self.play(m.FadeOut(ghost_lbl), run_time=0.3)
        self.play(m.GrowFromCenter(gap), run_time=0.6)
        self.play(m.FadeIn(why), run_time=0.5)
        self.hold()
        self.vo("below-gap", after=1.0)
        self.play(m.Indicate(m.VGroup(gap, why), color=COPPER), run_time=1.0)
        self.hold()
        self.clear()

    # --------------------------------------------------------------- summary
    def summary(self):
        self.heading("In short")
        points = [
            ("Work per byte decides which limit you hit.", INK),
            ("Left of the ridge, memory is the limit. Decode lives there.", TEAL),
            ("The roof is an upper bound. Below it? Find out why.", COPPER),
        ]
        rows = m.VGroup(*[T(text, 32, color) for text, color in points])
        rows.arrange(m.DOWN, buff=0.6, aligned_edge=m.LEFT).move_to([0, 0.2, 0])
        for k, row in enumerate(rows):
            self.vo(f"recap-{k + 1}")
            self.play(m.FadeIn(row, shift=m.RIGHT * 0.2), run_time=0.6)
            self.hold()
        self.wait(1.2)
