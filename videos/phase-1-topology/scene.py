"""Explainer for site/phase-1/topology.html.

Planned with the 3b1b-explainer skill: a puzzle first, the picture before
the name, one new element per caption, and motion that plays while its
caption is on screen. The eight cores stay put the whole film so the
viewer never has to re-orient. The core map is the Phase 0 measurement;
the bars and clock values are illustrations, and say so on screen.

Each caption's start time is logged to narration.json, so a voice track
can be laid on the exact same beats.
"""

import json
from pathlib import Path

import manimgx as m

BG = "#121816"
INK = "#f3efe6"
MUTED = "#b7b0a4"
FAINT = "#5b625e"
TEAL = "#7dcec4"
BLUE = "#9ec0e8"
GOLD = "#e8c46a"
COPPER = "#e0a07a"
ON = "#121816"

m.config.background_color = BG

CORE_Y = 0.9
THREAD_Y = 2.1
TL_Y = -1.3
TL_X0 = -5.0
TL_W = 10.0


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


def core_color(i):
    return GOLD if i >= 6 else BLUE


class Topology(m.Scene):
    def construct(self):
        self.caption = None
        self.log = []
        self.hook()
        self.core_map()
        self.scheduler()
        self.pinning()
        self.not_pinning()
        self.which_sets()
        self.summary()
        self.write_log()

    # Captions stay up long enough to read twice. cap() shows one and sets a
    # deadline; animations played before hold() happen while it is on screen.
    # text=None shows no caption (the words are already on screen) but still
    # logs a narration line.
    def cap(self, text, extra=0.0, voice=None):
        start = self.time
        voice = voice or text
        if text is None:
            if self.caption is not None:
                self.play(m.FadeOut(self.caption), run_time=0.4)
            self.caption = None
        else:
            new = T(text, 30, INK)
            if new.width > 12.8:
                new.scale_to_fit_width(12.8)
            new.to_edge(m.DOWN, buff=0.35)
            if self.caption is None:
                self.play(m.FadeIn(new), run_time=0.4)
            else:
                self.play(m.FadeOut(self.caption), m.FadeIn(new), run_time=0.4)
            self.caption = new
        self.log.append({"start": round(start, 2), "caption": text, "voice": voice})
        self.deadline = start + 0.4 + 1.2 + 0.36 * len(voice.split()) + extra

    def hold(self):
        self.wait(max(0.5, self.deadline - self.time))

    def say(self, text, extra=0.0, voice=None):
        self.cap(text, extra, voice)
        self.hold()

    def clear(self, keep=()):
        mobs = [mob for mob in self.mobjects if mob not in keep]
        if mobs:
            self.play(*[m.FadeOut(mob) for mob in mobs], run_time=0.5)
        self.caption = None
        self.wait(0.3)

    def heading(self, text, old=None):
        h = T(text, 38, INK).to_edge(m.UP, buff=0.4)
        if old is None:
            self.play(m.FadeIn(h), run_time=0.4)
        else:
            self.play(m.FadeOut(old), m.FadeIn(h), run_time=0.4)
        return h

    def focus(self, active):
        """Light the cores in `active`, dim the rest."""
        anims = []
        for i, (rect, name) in enumerate(self.cores):
            on = i in active
            col = core_color(i)
            anims.append(rect.animate.set_fill(col, opacity=0.25 if on else 0.05)
                         .set_stroke(col, opacity=1.0 if on else 0.3))
            anims.append(name.animate.set_opacity(1.0 if on else 0.3))
        return anims

    def write_log(self):
        end = round(self.time, 2)
        lines = []
        for k, item in enumerate(self.log):
            nxt = self.log[k + 1]["start"] if k + 1 < len(self.log) else end
            lines.append({"id": k + 1, "start": item["start"], "end": nxt, **item})
        out = Path(__file__).with_name("narration.json")
        out.write_text(json.dumps({"video": "site/phase-1/topology.mp4", "duration": end,
                                   "lines": lines}, indent=2, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ hook
    def hook(self):
        kicker = T("Phase 1", 24, TEAL)
        title = T("Cores and pinning", 60, INK)
        sub = T("why a benchmark has to say which core it ran on", 30, MUTED)
        m.VGroup(kicker, title, sub).arrange(m.DOWN, buff=0.3)
        self.play(m.FadeIn(kicker), m.Write(title), run_time=1.1)
        self.play(m.FadeIn(sub, shift=m.UP * 0.15))
        self.wait(2.0)
        self.clear()

        base = -1.6
        heights = [2.0, 3.0, 2.4]
        bars = m.VGroup()
        names = m.VGroup()
        for k, h in enumerate(heights):
            bar = box(1.3, h, TEAL, 0.75)
            bar.move_to([(k - 1) * 2.4, base + h / 2, 0])
            bars.add(bar)
            names.add(T(f"run {k + 1}", 26, MUTED).next_to(bar, m.DOWN, buff=0.2))
        floor = m.Line([-3.8, base, 0], [3.8, base, 0], color=FAINT, stroke_width=2)
        note = T("illustration, not a measurement", 22, MUTED).move_to([0, 2.6, 0])
        self.play(m.Create(floor), m.FadeIn(note), run_time=0.5)

        self.cap("Same code. Same phone. Three runs of one benchmark.")
        for bar, name in zip(bars, names):
            self.play(m.GrowFromEdge(bar, m.DOWN), m.FadeIn(name), run_time=0.7)
        self.hold()
        self.cap("Three different answers. Why?", extra=0.8)
        self.play(m.Indicate(bars[1], color=COPPER), run_time=0.9)
        self.hold()
        self.say("To see it, we need a map of the phone's cores.")
        self.clear()

    # ------------------------------------------------------------------ map
    def core_map(self):
        h = self.heading("This phone has eight cores")
        cores = m.VGroup()
        for i in range(8):
            w = 1.5 if i >= 6 else 1.15
            cores.add(m.VGroup(box(w, 1.0, INK, 0.0, stroke=MUTED), C(f"cpu{i}", 24, INK)))
        perf = m.VGroup(*cores[:6]).arrange(m.RIGHT, buff=0.15)
        prime = m.VGroup(*cores[6:]).arrange(m.RIGHT, buff=0.15)
        m.VGroup(perf, prime).arrange(m.RIGHT, buff=0.55).move_to([0, CORE_Y, 0])
        self.cores = cores

        self.cap("Linux gives them numbers: cpu0 to cpu7.")
        self.play(m.LaggedStart(*[m.FadeIn(c, shift=m.UP * 0.2) for c in cores], lag_ratio=0.12),
                  run_time=1.6)
        self.hold()

        perf_lbl = m.VGroup(T("performance", 26, BLUE), C("3.53 GHz max", 24, BLUE))
        perf_lbl.arrange(m.DOWN, buff=0.12).next_to(perf, m.DOWN, buff=0.35)
        prime_lbl = m.VGroup(T("prime", 26, GOLD), C("4.32 GHz max", 24, GOLD))
        prime_lbl.arrange(m.DOWN, buff=0.12).next_to(prime, m.DOWN, buff=0.35)

        self.cap("Cores 0 to 5 can run up to 3.53 GHz.")
        self.play(*[c[0].animate.set_fill(BLUE, opacity=0.25).set_stroke(BLUE) for c in perf],
                  m.FadeIn(perf_lbl), run_time=0.9)
        self.hold()
        self.cap("Cores 6 and 7 go faster: up to 4.32 GHz.")
        self.play(*[c[0].animate.set_fill(GOLD, opacity=0.25).set_stroke(GOLD) for c in prime],
                  m.FadeIn(prime_lbl), run_time=0.9)
        self.hold()

        self.cap("So the fast ones are 6 and 7, not 0 and 1.", extra=0.5)
        self.play(m.Circumscribe(prime, color=GOLD), run_time=1.2)
        self.hold()
        self.say("The numbers are names, not a ranking. Phase 0 read this from the phone itself.")

        same = T("all eight: the same core design, part 0x001", 26, MUTED)
        same.move_to([0, -1.5, 0])
        self.cap("All eight cores are the same design. Only the top speed differs.")
        self.play(m.FadeIn(same, shift=m.UP * 0.15), run_time=0.6)
        self.hold()
        self.say("There is no slow 'efficiency' core on this phone.")
        self.play(m.FadeOut(same), m.FadeOut(h), run_time=0.5)
        self.labels = m.VGroup(perf_lbl, prime_lbl)

    # ------------------------------------------------------------- scheduler
    def thread_token(self):
        dot = m.Circle(radius=0.2, color=COPPER, fill_color=COPPER, fill_opacity=1.0)
        lbl = T("your benchmark", 22, COPPER).next_to(dot, m.UP, buff=0.1)
        return m.VGroup(dot, lbl)

    def above(self, i):
        return [self.cores[i].get_center()[0], THREAD_Y, 0]

    def timeline(self, y, label):
        frame = box(TL_W, 0.45, BG, 0.0, stroke=FAINT).move_to([TL_X0 + TL_W / 2, y, 0])
        name = T(label, 24, MUTED).next_to(frame, m.LEFT, buff=0.25)
        return m.VGroup(frame, name)

    def run_on(self, thread, plan, y, run_time_per=1.4, start=0.0):
        """Move the thread through (core, fraction) steps, filling a timeline
        from `start` (a fraction of the bar)."""
        segs = m.VGroup()
        x = TL_X0 + TL_W * start
        for core, frac in plan:
            if abs(thread.get_center()[0] - self.above(core)[0]) > 0.01:
                self.play(thread.animate.move_to(self.above(core)), run_time=0.6)
            w = TL_W * frac
            seg = box(w, 0.45, core_color(core), 0.8).move_to([x + w / 2, y, 0])
            tag = C(f"cpu{core}", 20, core_color(core)).next_to(seg, m.DOWN, buff=0.1)
            self.play(m.GrowFromEdge(seg, m.LEFT), run_time=run_time_per * frac / 0.34,
                      rate_func=m.linear)
            self.play(m.FadeIn(tag), run_time=0.25)
            segs.add(m.VGroup(seg, tag))
            x += w
        return segs

    def scheduler(self):
        h = self.heading("Who picks the core?")
        thread = self.thread_token().move_to(self.above(2))
        self.thread = thread
        tl1 = self.timeline(TL_Y, "run 1")

        self.cap("When the benchmark starts, the scheduler picks a core. Say, cpu2.")
        self.play(m.FadeIn(thread, shift=m.DOWN * 0.2), run_time=0.6)
        self.play(m.Indicate(self.cores[2], color=BLUE), run_time=0.8)
        self.hold()

        self.cap("This bar is one run, filling up as time passes.")
        self.play(m.FadeIn(tl1), run_time=0.5)
        segs1 = self.run_on(thread, [(2, 0.34)], TL_Y)
        self.hold()
        self.cap("Partway through, the scheduler moves it. Then moves it again.", extra=0.5)
        segs1.add(*self.run_on(thread, [(6, 0.33), (4, 0.33)], TL_Y, start=0.34))
        self.hold()
        self.say("The result is one number, made on three different cores.", extra=0.5)

        tl2 = self.timeline(TL_Y - 1.05, "run 2")
        self.cap("Next run, the scheduler makes different choices.")
        self.play(m.FadeIn(tl2), run_time=0.5)
        segs2 = self.run_on(thread, [(7, 0.2), (0, 0.5), (5, 0.3)], TL_Y - 1.05, run_time_per=1.0)
        self.hold()
        self.say("A different mix of cores, so a different number. That was our puzzle.",
                 extra=0.5)
        self.play(m.FadeOut(m.VGroup(tl1, tl2, segs1, segs2)), m.FadeOut(h), run_time=0.5)

    # --------------------------------------------------------------- pinning
    def pinning(self):
        h = self.heading("Pinning: allow just one core")
        cores = self.cores
        fence = m.SurroundingRectangle(cores[6], color=TEAL, buff=0.12, stroke_width=4)

        self.cap("The fix: tell the kernel this thread may run only on cpu6.")
        self.play(m.Create(fence),
                  *self.focus([6]),
                  self.labels.animate.set_opacity(0.3), run_time=1.0)
        self.play(self.thread.animate.move_to(self.above(6)), run_time=0.7)
        self.hold()

        tl = self.timeline(TL_Y, "run")
        self.cap("Now it can't be moved. The whole run happens on one core.")
        self.play(m.FadeIn(tl), run_time=0.4)
        seg = box(TL_W, 0.45, GOLD, 0.8).move_to([TL_X0 + TL_W / 2, TL_Y, 0])
        self.play(m.GrowFromEdge(seg, m.LEFT), run_time=3.0, rate_func=m.linear)
        tag = C("cpu6 only", 22, GOLD).next_to(seg, m.DOWN, buff=0.1)
        self.play(m.FadeIn(tag), run_time=0.3)
        self.hold()
        self.say("That is pinning. The number now names its core.", extra=0.8)

        code = C("sched_setaffinity(thread, {6})", 30, TEAL).move_to([0, -2.4, 0])
        self.cap("The call is sched_setaffinity. You give it a set of allowed cores.")
        self.play(m.FadeIn(code, shift=m.UP * 0.15), run_time=0.6)
        self.hold()
        self.cap("A set with just one core in it is a pin.")
        self.play(m.Indicate(code, color=TEAL, scale_factor=1.08), run_time=0.9)
        self.hold()

        warn = T("check it worked: a failed pin is just an unpinned run", 26, COPPER)
        warn.move_to(code)
        self.cap("Always check that the call succeeded. A failed pin you ignore is an unpinned run.")
        self.play(m.FadeOut(code), m.FadeIn(warn), run_time=0.7)
        self.hold()
        self.play(m.FadeOut(m.VGroup(tl, seg, tag, warn)), m.FadeOut(h), run_time=0.5)
        self.fence = fence

    # ------------------------------------------------------- what it doesn't
    def not_pinning(self):
        h = self.heading("What pinning does not do")
        cores = self.cores

        # 1. The clock still moves.
        ghz = m.ValueTracker(4.32)
        dial = m.always_redraw(lambda: C(f"{ghz.get_value():.2f} GHz", 30, GOLD).move_to(
            [cores[6].get_center()[0], -0.9, 0]))
        caveat = T("clock values: illustration", 20, MUTED).next_to(dial, m.DOWN, buff=0.15)
        self.cap("It does not set the speed. The phone still slows cpu6 down when it heats up.",
                 extra=0.5)
        self.play(self.labels[1].animate.set_opacity(0.0), run_time=0.3)
        self.play(m.FadeIn(dial), m.FadeIn(caveat), run_time=0.4)
        self.play(ghz.animate.set_value(2.80), run_time=2.5)
        self.hold()
        self.say("So record the real clock during the run, not just the maximum.", extra=0.5)
        self.play(m.FadeOut(dial), m.FadeOut(caveat), run_time=0.4)
        self.remove(dial)

        # 2. Other programs can still land there.
        other = m.VGroup(m.Circle(radius=0.2, color=FAINT, fill_color=MUTED, fill_opacity=1.0),
                         T("another app", 22, MUTED))
        other[1].next_to(other[0], m.RIGHT, buff=0.12)
        cx = cores[6].get_center()[0]
        dot_y = self.thread[0].get_center()[1]
        other.shift([cx + 1.7, dot_y, 0] - other[0].get_center())
        # Both dots share cpu6; the labels point outward so they never touch.
        home = self.thread.copy()
        squeezed = self.thread.copy()
        squeezed[0].move_to([cx - 0.28, dot_y, 0])
        squeezed[1].next_to(squeezed[0], m.LEFT, buff=0.12)
        self.cap("It does not reserve the core. Other programs can still run on cpu6.")
        self.play(m.FadeIn(other), run_time=0.4)
        self.play(other.animate.shift([cx + 0.28 - other[0].get_center()[0], 0, 0]),
                  m.Transform(self.thread, squeezed), run_time=1.0)
        self.play(m.Indicate(other, color=COPPER), run_time=0.8)
        self.hold()
        self.play(m.FadeOut(other), m.Transform(self.thread, home), run_time=0.4)

        # 3. Memory is shared.
        dram = m.VGroup(box(11.0, 0.7, INK, 0.12, stroke=MUTED),
                        T("DRAM: one memory, shared by every core", 26, INK)).move_to([0, -1.6, 0])
        wires = m.VGroup(*[
            m.Line(cores[i].get_bottom() + m.DOWN * 0.05, [cores[i].get_center()[0], -1.25, 0],
                   color=FAINT, stroke_width=2)
            for i in range(8)
        ])
        self.cap("And it does not give the thread its own memory. All cores share DRAM.")
        self.play(self.labels[0].animate.set_opacity(0.0), run_time=0.3)
        self.play(m.FadeIn(dram), m.LaggedStart(*[m.Create(w) for w in wires], lag_ratio=0.1),
                  run_time=1.2)
        self.hold()
        self.say("Pinning fixes one thing only: which core runs the thread.", extra=0.8)
        self.play(m.FadeOut(m.VGroup(dram, wires, self.fence, self.thread)), m.FadeOut(h),
                  *self.focus(range(8)), run_time=0.6)

    # ---------------------------------------------------------- which sets
    def which_sets(self):
        h = self.heading("Which cores to measure")
        cores = self.cores
        rows = [
            ([6], "{6}", "one prime core"),
            ([0], "{0}", "one performance core"),
            (list(range(6)), "{0, 1, 2, 3, 4, 5}", "all six performance cores"),
            (list(range(8)), "{0, ..., 7}", "all eight cores"),
        ]
        voices = [
            "First, one prime core on its own.",
            "Then one performance core. Same test. Does the faster clock change the answer?",
            "Then all six performance cores together.",
            "Then all eight. 'All cores' is not one thing when the cores differ.",
        ]
        fence = None
        label = None
        for (members, set_txt, words), line in zip(rows, voices):
            group = m.VGroup(*[cores[i] for i in members])
            new_fence = m.SurroundingRectangle(group, color=TEAL, buff=0.12, stroke_width=4)
            new_label = m.VGroup(C(set_txt, 30, TEAL), T(words, 28, INK))
            new_label.arrange(m.DOWN, buff=0.2).move_to([0, -1.3, 0])
            self.cap(line, extra=0.3)
            anims = [m.FadeIn(new_label)]
            anims += self.focus(members)
            if fence is None:
                anims.append(m.Create(new_fence))
            else:
                anims += [m.ReplacementTransform(fence, new_fence), m.FadeOut(label)]
            self.play(*anims, run_time=0.8)
            fence, label = new_fence, new_label
            self.hold()
        self.say("Whatever you run, write the core set down next to the number.", extra=0.8)
        self.clear()

    # --------------------------------------------------------------- summary
    def summary(self):
        self.heading("In short")
        points = [
            ("Core numbers are names. Here, 6 and 7 are the fast ones.", GOLD),
            ("An unpinned run blends cores. Pin it to name the core.", TEAL),
            ("Pinning fixes the core, not the clock. Record both.", COPPER),
        ]
        rows = m.VGroup(*[T(text, 32, color) for text, color in points])
        rows.arrange(m.DOWN, buff=0.6, aligned_edge=m.LEFT).move_to([0, 0.3, 0])
        for row, (text, _) in zip(rows, points):
            self.cap(None, voice=text)
            self.play(m.FadeIn(row, shift=m.RIGHT * 0.2), run_time=0.6)
            self.hold()
        self.wait(1.5)
