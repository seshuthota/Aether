"""Explainer for site/phase-1/bandwidth.html.

Planned with the 3b1b-explainer skill: the Phase 0 number as the puzzle,
the add chain before the word "accumulator", the sweep drawn live before
any plateau gets a name. Only 3.3 GB/s is a measurement from this phone;
every other bar, curve and value is an illustration and says so on screen.

Narrated, no captions: SCRIPT below is the voice-over, and each beat lasts
as long as its recorded clip in audio/ (see videos/narrate.py).
"""

import math
import sys
from pathlib import Path

import manimgx as m

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from narrate import Narrated  # noqa: E402

# Written to be heard: numbers spelled the way they are said ("D-RAM", not
# "DRAM", which voices read as "drum"), and each line describes what is
# moving on screen while it is spoken.
SCRIPT = [
    ("title", "Bandwidth. How fast can one core read memory?"),
    ("puzzle-weights", "To write each new token, the model reads every one of its weights "
                       "from memory. Every token, all of them."),
    ("puzzle-formula", "So the number that matters is simple: how many bytes per second "
                       "can a core read?"),
    ("puzzle-bandwidth", "That's called bandwidth. We write it in gigabytes per second: "
                         "billions of bytes, every second."),
    ("puzzle-phase0", "Back in Phase 0, we summed a big array on this phone, and it ran "
                      "at three point three gigabytes per second."),
    ("puzzle-question", "So, is that how fast memory is? No. That's how fast the additions "
                        "were. Let's see why."),
    ("chain-one", "Here's that sum: s plus-equals x, one element at a time. Each box is "
                  "one addition, and time runs to the right."),
    ("chain-wait", "Watch. Each addition needs the answer from the one before it. So it "
                   "waits. And waits again. One long chain."),
    ("chain-measured", "So what we timed was additions waiting on each other. Memory was "
                       "barely trying."),
    ("chain-four", "Now split the work into four separate sums, s zero to s three, and "
                   "add them together at the very end."),
    ("chain-finish", "Same eight additions. But the four chains run side by side, and "
                     "they're finished here, while the single chain is still going."),
    ("chain-sooner", "Faster additions ask memory for data sooner. Now memory can finally "
                     "become the limit."),
    ("chain-rule", "And that's what we want to measure. So the rule is: several "
                   "independent sums, always."),
    ("kernel-copy", "There are four classic ways to stream through memory. Copy reads one "
                    "float and writes one: eight bytes per element."),
    ("kernel-scale", "Scale moves the same bytes, plus a multiply. It asks: does a little "
                     "math slow the stream down?"),
    ("kernel-add", "Add reads two floats and writes one. Twelve bytes."),
    ("kernel-read", "And read only reads, into several sums. Four bytes, and nothing is "
                    "written back."),
    ("kernel-decode", "Decode is this last row. Weights are read, and never written back."),
    ("kernel-copy-writes", "Copy also pays for writes, which decode never does. So judge "
                           "decode against the read, not the copy."),
    ("count-run", "One more trap. Take a single copy run: one gigabyte read, and one "
                  "gigabyte written, in one second. Made-up numbers, just to count."),
    ("count-both", "Count both, and you get two gigabytes per second."),
    ("count-read", "Count only the read, and the very same run is one gigabyte per second."),
    ("count-rule", "Same run, two answers. So pick one way of counting, and write it down."),
    ("sweep-intro", "Now the main experiment. Time the read at many buffer sizes, from four "
                    "kilobytes up to two hundred and fifty-six megabytes. On the right: "
                    "where the buffer fits."),
    ("sweep-l1", "A small buffer fits in L1, the cache closest to the core. The read is fast."),
    ("sweep-cliff", "Grow it past L1's size, and the speed falls off a cliff. Now it lives "
                    "in L2."),
    ("sweep-l2", "A new, lower plateau. It stays flat as long as the buffer still fits in L2."),
    ("sweep-slc", "Past L2, there may be a small step for the system-level cache. Or just "
                  "a slope."),
    ("sweep-dram-flat", "And from tens of megabytes on, every read goes all the way out to "
                        "D-RAM. Flat again."),
    ("sweep-name", "Only now do we name each flat stretch, once we've seen where the "
                   "cliffs really are."),
    ("sweep-dram", "This flat right end, measured on one pinned core, is the number decode "
                   "will be judged by."),
    ("sweep-not-l1", "Not the tall left end. That's a cache talking, not the phone's memory."),
    ("cond-alone", "A bandwidth number on its own can't be compared with next week's. So "
                   "write down which cores you used, and that the pin worked. Whether the "
                   "screen was on. Whether it was charging."),
    ("cond-more", "The starting temperature. The real clock during the run. And the exact "
                  "build and commit."),
    ("cond-long", "Then run the read for five whole minutes. The first few seconds look great."),
    ("cond-hot", "But as the phone heats up, it slows down. This lower number is what a "
                 "real chat actually gets."),
    ("cond-both", "Keep both. They're different results."),
    ("recap-1", "So: use several independent sums, or you're measuring waiting."),
    ("recap-2", "Bandwidth is bytes divided by seconds. Count the bytes the same way "
                "every time."),
    ("recap-3", "And sweep the size. The flat right end is D-RAM: decode's yardstick."),
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

# The working-set sweep: buffer size as log2(bytes), 4 KB (12) to 256 MB (28).
S0, S1 = 12, 28
PX0, PX1 = -6.0, 1.6
PY0, PY_SCALE = -2.0, 1.75


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


def sig(x):
    return 1 / (1 + math.exp(-x))


def speed(s):
    """Illustrative read speed (0..2) at buffer size 2**s: three cliffs."""
    l1, l2, slc, dram = 2.0, 1.15, 0.75, 0.4
    return (dram + (l1 - l2) * sig((15.6 - s) * 4) + (l2 - slc) * sig((22.8 - s) * 4)
            + (slc - dram) * sig((24.4 - s) * 4))


def px(s):
    return PX0 + (s - S0) / (S1 - S0) * (PX1 - PX0)


def py(v):
    return PY0 + v * PY_SCALE


def size_text(s):
    b = 2 ** s
    if b >= 2 ** 20:
        return f"{b / 2 ** 20:.0f} MB"
    return f"{b / 2 ** 10:.0f} KB"


class Bandwidth(Narrated, m.Scene):
    def construct(self):
        self.start_voice(__file__, SCRIPT)
        self.title_card()
        self.puzzle()
        self.chains()
        self.kernels()
        self.counting()
        self.sweep()
        self.conditions()
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
        title = T("Bandwidth", 60, INK)
        sub = T("how fast can one core read memory?", 30, MUTED)
        m.VGroup(kicker, title, sub).arrange(m.DOWN, buff=0.3)
        self.vo("title")
        self.play(m.FadeIn(kicker), m.Write(title), run_time=1.1)
        self.play(m.FadeIn(sub, shift=m.UP * 0.15))
        self.hold(0.6)
        self.clear()

    # ---------------------------------------------------------------- puzzle
    def puzzle(self):
        # A weight block streams from memory into the core.
        mem = m.VGroup(box(3.2, 2.6, INK, 0.06, stroke=MUTED), T("memory", 26, MUTED))
        mem[1].next_to(mem[0], m.UP, buff=0.15)
        core = m.VGroup(box(2.0, 1.4, BLUE, 0.2), T("core", 28, BLUE))
        core[1].move_to(core[0])
        mem.move_to([-4.0, 0.4, 0])
        core.move_to([4.0, 0.4, 0])
        cells = m.VGroup(*[box(0.42, 0.42, BLUE, 0.7) for _ in range(20)])
        cells.arrange_in_grid(rows=4, cols=5, buff=0.08).move_to(mem[0])

        self.vo("puzzle-weights")
        self.play(m.FadeIn(mem), m.FadeIn(core), m.FadeIn(cells), run_time=0.8)
        # Reading copies the weight in; memory keeps it for the next token.
        flying = m.VGroup(*[c.copy() for c in cells])
        self.play(m.LaggedStart(*[f.animate.move_to(core[0]).set_opacity(0) for f in flying],
                                lag_ratio=0.12), run_time=3.0)
        self.remove(flying)
        self.hold()

        formula = T("speed  =  bytes read  ÷  seconds", 34, INK).move_to([0, -1.6, 0])
        self.vo("puzzle-formula")
        self.play(m.Write(formula), run_time=1.0)
        self.hold()
        self.say("puzzle-bandwidth")

        self.play(m.FadeOut(mem), m.FadeOut(core), m.FadeOut(formula), run_time=0.5)
        old = m.VGroup(T("Phase 0 sum of an array", 30, MUTED), C("3.3 GB/s", 64, TEAL),
                       T("measured on this phone", 22, TEAL))
        old.arrange(m.DOWN, buff=0.25).move_to([0, 0.5, 0])
        self.vo("puzzle-phase0")
        self.play(m.FadeIn(old, shift=m.UP * 0.15), run_time=0.7)
        self.hold()
        self.vo("puzzle-question", after=1.2)
        self.play(m.Indicate(old[1], color=COPPER), run_time=1.0)
        self.hold()
        self.clear()

    # ---------------------------------------------------------------- chains
    def chain_row(self, n, y, width, color):
        boxes = m.VGroup(*[box(width - 0.08, 0.42, color, 0.75) for _ in range(n)])
        boxes.arrange(m.RIGHT, buff=0.08)
        boxes.move_to([-4.6 + boxes.width / 2, y, 0])
        for b in boxes:
            b.add(T("+", 26, BG).move_to(b))
        return boxes

    def chains(self):
        h = self.heading("One running sum is a chain")
        axis = m.Arrow([-4.7, 2.15, 0], [5.4, 2.15, 0], color=FAINT, stroke_width=2, buff=0)
        axis_lbl = T("time", 22, MUTED).next_to(axis, m.UP, buff=0.08).align_to(axis, m.RIGHT)
        one = self.chain_row(8, 1.3, 1.2, COPPER)
        one_lbl = C("s", 26, COPPER).next_to(one, m.LEFT, buff=0.3)
        code1 = C("s += x[i]", 26, COPPER).next_to(one, m.DOWN, buff=0.2).align_to(one, m.LEFT)

        self.vo("chain-one")
        self.play(m.FadeIn(axis), m.FadeIn(axis_lbl), m.FadeIn(one_lbl), m.FadeIn(code1),
                  run_time=0.5)
        self.play(m.FadeIn(one[0]), run_time=0.5)
        self.hold()
        self.vo("chain-wait")
        self.play(m.LaggedStart(*[m.FadeIn(b, shift=m.RIGHT * 0.3) for b in one[1:]],
                                lag_ratio=1.0), run_time=4.2, rate_func=m.linear)
        self.hold()
        self.say("chain-measured")

        rows = m.VGroup(*[self.chain_row(2, -0.1 - 0.52 * k, 1.2, TEAL) for k in range(4)])
        names = m.VGroup(*[C(f"s{k}", 24, TEAL).next_to(r, m.LEFT, buff=0.3)
                           for k, r in enumerate(rows)])
        self.vo("chain-four")
        self.play(m.FadeIn(names), run_time=0.5)
        self.play(*[m.LaggedStart(*[m.FadeIn(b, shift=m.RIGHT * 0.3) for b in r], lag_ratio=1.0)
                    for r in rows], run_time=1.05, rate_func=m.linear)
        self.hold()

        done = rows[0][-1].get_right()[0]
        mark = m.DashedLine([done + 0.05, 1.7, 0], [done + 0.05, -2.0, 0], color=TEAL)
        self.vo("chain-finish", after=1.2)
        self.play(m.Create(mark), run_time=0.6)
        self.play(m.Indicate(m.VGroup(*one[2:]), color=COPPER), run_time=1.0)
        self.hold()
        self.say("chain-sooner")
        self.say("chain-rule", extra=0.8)
        self.clear()

    # --------------------------------------------------------------- kernels
    def traffic(self, kinds):
        cells = m.VGroup()
        for kind in kinds:
            color = BLUE if kind == "R" else COPPER
            b = box(0.95, 0.46, color, 0.75)
            b.add(T("read" if kind == "R" else "write", 20, BG).move_to(b))
            cells.add(b)
        return cells.arrange(m.RIGHT, buff=0.06)

    def kernels(self):
        h = self.heading("Four ways to stream memory")
        specs = [
            ("copy", "a[i] = b[i]", "RW", "8 bytes"),
            ("scale", "a[i] = q * b[i]", "RW", "8 bytes"),
            ("add", "a[i] = b[i] + c[i]", "RRW", "12 bytes"),
            ("read", "s0 += b[i] ...", "R", "4 bytes"),
        ]
        voices = ["kernel-copy", "kernel-scale", "kernel-add", "kernel-read"]
        rows = m.VGroup()
        for k, ((name, code, kinds, total), line) in enumerate(zip(specs, voices)):
            row = m.VGroup(C(name, 28, INK), C(code, 26, MUTED), self.traffic(kinds),
                           T(total, 26, INK))
            y = 1.9 - k * 0.95
            row[0].move_to([-5.4, y, 0]).align_to([-6.2, 0, 0], m.LEFT)
            row[1].move_to([-1.9, y, 0])
            row[2].move_to([1.9, y, 0]).align_to([1.2, 0, 0], m.LEFT)
            row[3].move_to([5.6, y, 0])
            rows.add(row)
            self.vo(line)
            self.play(m.FadeIn(row[0]), m.FadeIn(row[1]), run_time=0.5)
            self.play(m.LaggedStart(*[m.GrowFromEdge(c, m.LEFT) for c in row[2]], lag_ratio=0.4),
                      m.FadeIn(row[3]), run_time=0.8)
            self.hold()

        read_row = rows[3]
        frame = m.SurroundingRectangle(read_row, color=TEAL, buff=0.15, stroke_width=3)
        self.vo("kernel-decode", after=1.2)
        self.play(m.Create(frame), *[r.animate.set_opacity(0.35) for r in rows[:3]],
                  run_time=0.8)
        self.hold()
        self.vo("kernel-copy-writes")
        self.play(rows[0].animate.set_opacity(1.0), run_time=0.4)
        self.play(m.Indicate(rows[0][2][1], color=COPPER), run_time=0.9)
        self.hold()
        self.clear()

    # -------------------------------------------------------------- counting
    def counting(self):
        h = self.heading("Count the bytes honestly")
        note = T("illustration, not a measurement", 22, MUTED).next_to(h, m.DOWN, buff=0.2)
        run = m.VGroup(T("one copy run:", 28, MUTED), C("1 GB read  +  1 GB written  in 1 s", 30))
        run.arrange(m.RIGHT, buff=0.3).move_to([0, 1.1, 0])
        self.vo("count-run")
        self.play(m.FadeIn(note), m.FadeIn(run), run_time=0.7)
        self.hold()

        left = m.VGroup(T("count read + write", 26, BLUE), C("2 GB/s", 48, BLUE))
        right = m.VGroup(T("count only the read", 26, COPPER), C("1 GB/s", 48, COPPER))
        for g in (left, right):
            g.arrange(m.DOWN, buff=0.25)
        left.move_to([-3.0, -0.6, 0])
        right.move_to([3.0, -0.6, 0])
        self.vo("count-both")
        self.play(m.FadeIn(left, shift=m.UP * 0.15), run_time=0.6)
        self.hold()
        self.vo("count-read")
        self.play(m.FadeIn(right, shift=m.UP * 0.15), run_time=0.6)
        self.hold()
        self.say("count-rule", extra=0.8)
        self.clear()

    # ----------------------------------------------------------------- sweep
    def sweep(self):
        h = self.heading("Grow the buffer, watch the speed")
        note = T("the shape to expect, not data from this phone", 22, MUTED)
        note.next_to(h, m.DOWN, buff=0.15)
        x_axis = m.Arrow([PX0 - 0.2, PY0, 0], [PX1 + 0.4, PY0, 0], color=FAINT,
                         stroke_width=2, buff=0)
        y_axis = m.Arrow([PX0 - 0.2, PY0, 0], [PX0 - 0.2, py(2.35), 0], color=FAINT,
                         stroke_width=2, buff=0)
        x_lbl = T("buffer size (each step doubles)", 22, MUTED).next_to(x_axis, m.DOWN, buff=0.15)
        y_lbl = T("GB/s", 22, MUTED).next_to(y_axis, m.UP, buff=0.1)

        # The cache ladder beside the plot; the level the buffer fits in lights up.
        names = ["L1", "L2", "SLC", "DRAM"]
        colors = [TEAL, BLUE, BLUE, COPPER]
        ladder = m.VGroup()
        for name, color in zip(names, colors):
            r = box(2.2, 0.62, color, 0.08, stroke=FAINT)
            r.add(T(name, 26, MUTED).move_to(r))
            ladder.add(r)
        ladder.arrange(m.DOWN, buff=0.18).move_to([4.6, -0.3, 0])
        ladder_lbl = T("where it fits", 22, MUTED).next_to(ladder, m.UP, buff=0.15)

        s = m.ValueTracker(S0 + 0.01)
        curve = m.always_redraw(lambda: m.ParametricFunction(
            lambda t: [px(t), py(speed(t)), 0], t_range=[S0, max(S0 + 0.02, s.get_value())],
            color=INK, stroke_width=4))
        dot = m.always_redraw(lambda: m.Dot([px(s.get_value()), py(speed(s.get_value())), 0],
                                            radius=0.09, color=GOLD))
        readout = m.always_redraw(lambda: C(size_text(s.get_value()), 26, GOLD).next_to(
            [px(s.get_value()), py(speed(s.get_value())), 0], m.UR, buff=0.2))

        self.vo("sweep-intro")
        self.play(m.FadeIn(note), m.Create(x_axis), m.Create(y_axis), m.FadeIn(x_lbl),
                  m.FadeIn(y_lbl), run_time=0.8)
        self.play(m.FadeIn(ladder), m.FadeIn(ladder_lbl), run_time=0.6)
        self.hold()

        # The lit rung follows the buffer size, so it changes exactly at each cliff.
        def level(v):
            return 0 if v < 15.6 else 1 if v < 22.8 else 2 if v < 24.4 else 3

        lit = m.always_redraw(lambda: box(2.2, 0.62, colors[level(s.get_value())], 0.35).move_to(
            ladder[level(s.get_value())]))
        self.add(lit, curve, dot, readout)

        steps = [
            (15.0, "sweep-l1"),
            (18.0, "sweep-cliff"),
            (22.4, "sweep-l2"),
            (24.0, "sweep-slc"),
            (28.0, "sweep-dram-flat"),
        ]
        for target, line in steps:
            self.vo(line)
            self.play(s.animate.set_value(target), run_time=2.6, rate_func=m.linear)
            self.hold()

        tags = m.VGroup(
            T("L1", 24, TEAL).move_to([px(13.6), py(speed(13.6)) + 0.32, 0]),
            T("L2", 24, BLUE).move_to([px(19.5), py(speed(19.5)) + 0.32, 0]),
            T("SLC?", 22, BLUE).move_to([px(23.9) + 0.45, py(speed(23.6)) + 0.4, 0]),
            T("DRAM", 24, COPPER).move_to([px(26.6), py(speed(26.6)) + 0.32, 0]),
        )
        self.play(m.FadeOut(readout), run_time=0.3)
        self.remove(readout)
        self.vo("sweep-name")
        self.play(m.LaggedStart(*[m.FadeIn(t, shift=m.DOWN * 0.1) for t in tags],
                                lag_ratio=0.4), run_time=1.4)
        self.hold()

        tail = m.Line([px(25.2), py(speed(25.2)), 0], [px(28), py(speed(28)), 0],
                      color=COPPER, stroke_width=9)
        self.vo("sweep-dram", after=1.2)
        self.play(m.Create(tail), run_time=0.8)
        self.play(m.Indicate(tags[3], color=COPPER), run_time=0.8)
        self.hold()
        self.vo("sweep-not-l1")
        self.play(m.Indicate(tags[0], color=TEAL), run_time=0.9)
        self.hold()
        self.remove(curve, dot, lit)
        self.clear()

    # ------------------------------------------------------------ conditions
    def conditions(self):
        h = self.heading("Write the conditions next to every number")
        items = [
            "which cores, and that the pin worked",
            "screen on or off",
            "charging or not",
            "starting temperature",
            "the real clock during the run",
            "release build and the commit",
        ]
        rows = m.VGroup(*[
            m.VGroup(m.Square(0.28, color=TEAL, stroke_width=2), T(text, 28, INK)).arrange(
                m.RIGHT, buff=0.25)
            for text in items
        ])
        rows.arrange(m.DOWN, buff=0.32, aligned_edge=m.LEFT).move_to([0, 0.3, 0])
        self.vo("cond-alone")
        self.play(m.LaggedStart(*[m.FadeIn(r, shift=m.RIGHT * 0.15) for r in rows[:3]],
                                lag_ratio=0.5), run_time=1.5)
        self.hold()
        self.vo("cond-more")
        self.play(m.LaggedStart(*[m.FadeIn(r, shift=m.RIGHT * 0.15) for r in rows[3:]],
                                lag_ratio=0.5), run_time=1.5)
        self.hold()

        # The long run: optimistic at first, lower once the phone is hot.
        base_y, x0, x1 = -2.3, -4.0, 4.0
        frame = m.Line([x0, base_y, 0], [x1, base_y, 0], color=FAINT, stroke_width=2)
        f_lbl = T("5 minutes →", 20, MUTED).next_to(frame, m.DOWN, buff=0.08).align_to(frame,
                                                                                       m.RIGHT)
        long_run = m.ParametricFunction(
            lambda t: [x0 + t * (x1 - x0), base_y + 0.35 + 0.75 * sig((0.18 - t) * 25), 0],
            t_range=[0, 1], color=GOLD, stroke_width=4)
        first = T("first seconds", 20, GOLD).move_to([x0 + 0.9, base_y + 1.35, 0])
        later = T("after the phone heats up", 20, GOLD).move_to([x0 + 5.6, base_y + 0.65, 0])
        self.play(m.FadeOut(rows), run_time=0.4)
        self.vo("cond-long")
        self.play(m.Create(frame), m.FadeIn(f_lbl), run_time=0.4)
        self.play(m.Create(long_run), run_time=2.4, rate_func=m.linear)
        self.play(m.FadeIn(first), run_time=0.4)
        self.hold()
        self.vo("cond-hot")
        self.play(m.FadeIn(later), run_time=0.4)
        self.hold()
        self.say("cond-both")
        self.clear()

    # --------------------------------------------------------------- summary
    def summary(self):
        self.heading("In short")
        points = [
            ("Use several independent sums, or you measure waiting.", TEAL),
            ("GB/s is bytes ÷ seconds. Count bytes the same way every time.", BLUE),
            ("Sweep the size. The flat right end is DRAM: decode's yardstick.", COPPER),
        ]
        rows = m.VGroup(*[T(text, 32, color) for text, color in points])
        rows.arrange(m.DOWN, buff=0.6, aligned_edge=m.LEFT).move_to([0, 0.3, 0])
        if rows.width > 12.8:
            rows.scale_to_fit_width(12.8)
        for k, row in enumerate(rows):
            self.vo(f"recap-{k + 1}")
            self.play(m.FadeIn(row, shift=m.RIGHT * 0.2), run_time=0.6)
            self.hold()
        self.wait(1.5)
