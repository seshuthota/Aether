"""Explainer for site/phase-1/alignment.html.

Muted on purpose: the notes page has to work without sound. One idea per
beat, and every caption is held long enough to read twice. The middle
scenes reuse one memory strip so the viewer never has to re-orient.
Nothing on screen is a measurement from this phone.
"""

import manimgx as m

BG = "#121816"
INK = "#f3efe6"
MUTED = "#b7b0a4"
FAINT = "#5b625e"
TEAL = "#7dcec4"
BLUE = "#9ec0e8"
COPPER = "#e0a07a"
ON = "#121816"

m.config.background_color = BG

SLOTS = 48  # three 64-byte lines of 4-byte floats
SLOT_W = 0.26
X0 = -SLOTS * SLOT_W / 2
STRIP_Y = -0.35


def T(text, size=30, color=INK):
    return m.Text(text, font="sans-serif", font_size=size, color=color)


def C(text, size=28, color=INK):
    return m.Text(text, font="monospace", font_size=size, color=color)


def box(width, height, color, opacity=1.0, stroke=None):
    return m.Rectangle(
        width=width,
        height=height,
        stroke_width=1.5,
        stroke_color=stroke or color,
        fill_color=color,
        fill_opacity=opacity,
        color=stroke or color,
    )


def slot_x(byte):
    """x position of a byte offset along the three-line strip."""
    return X0 + byte / 4 * SLOT_W


class Alignment(m.Scene):
    def construct(self):
        self.cap = None
        self.title_card()
        self.bytes()
        self.strip_story()
        self.definition()
        self.rust()
        self.summary()

    # Captions stay up long enough to read twice: beginners need the time.
    def say(self, text, extra=0.0):
        new = T(text, 30, INK)
        if new.width > 12.8:
            new.scale_to_fit_width(12.8)
        new.to_edge(m.DOWN, buff=0.35)
        if self.cap is None:
            self.play(m.FadeIn(new), run_time=0.4)
        else:
            self.play(m.FadeOut(self.cap), m.FadeIn(new), run_time=0.4)
        self.cap = new
        self.wait(1.2 + 0.36 * len(text.split()) + extra)

    def clear(self):
        mobs = list(self.mobjects)
        if mobs:
            self.play(*[m.FadeOut(mob) for mob in mobs], run_time=0.5)
        self.cap = None
        self.wait(0.3)

    def heading(self, text):
        h = T(text, 38, INK).to_edge(m.UP, buff=0.4)
        self.play(m.FadeIn(h), run_time=0.4)
        return h

    # ------------------------------------------------------------------ 1
    def title_card(self):
        kicker = T("Phase 1", 24, TEAL)
        title = T("Alignment", 60, INK)
        sub = T("where your data starts in memory, and why it matters", 30, MUTED)
        m.VGroup(kicker, title, sub).arrange(m.DOWN, buff=0.3)
        self.play(m.FadeIn(kicker), m.Write(title), run_time=1.1)
        self.play(m.FadeIn(sub, shift=m.UP * 0.15))
        self.wait(2.0)
        self.clear()

    # ------------------------------------------------------------------ 2
    def bytes(self):
        self.heading("Memory is a row of numbered bytes")
        n = 12
        w = 0.95
        cells = m.VGroup(*[box(w, 0.8, BG, 1.0, stroke=MUTED) for _ in range(n)])
        cells.arrange(m.RIGHT, buff=0).move_to([0, 0.3, 0])
        addrs = m.VGroup(*[
            C(str(i), 22, MUTED).next_to(c, m.UP, buff=0.18) for i, c in enumerate(cells)
        ])
        addr_title = T("address", 24, MUTED).next_to(addrs, m.UP, buff=0.25)

        self.play(m.LaggedStart(*[m.FadeIn(c) for c in cells], lag_ratio=0.06), run_time=1.0)
        self.play(m.FadeIn(addrs), m.FadeIn(addr_title), run_time=0.6)
        self.say("Each box holds one byte. Its number is its address.")

        colors = [TEAL, BLUE, TEAL]
        braces = m.VGroup()
        for k in range(3):
            group = m.VGroup(*cells[4 * k: 4 * k + 4])
            self.play(*[c.animate.set_fill(colors[k], opacity=0.85) for c in group], run_time=0.5)
            br = m.Brace(group, m.DOWN, color=colors[k])
            label = C(f"x[{k}]", 26, colors[k]).next_to(br, m.DOWN, buff=0.12)
            braces.add(m.VGroup(br, label))
            self.play(m.FadeIn(braces[-1]), run_time=0.4)
            if k == 0:
                self.say("One f32 takes 4 bytes.", extra=0.3)
        self.say("An array of f32 is just these 4-byte groups, side by side, no gaps.")
        self.clear()

    # ------------------------------------------------------------------ 3–6
    def strip_story(self):
        heading = self.heading("The cache works in 64-byte lines")

        # Three cache lines as background bands, then 48 float slots on top.
        lines = m.VGroup()
        for k in range(3):
            band = box(16 * SLOT_W, 1.3, INK, [0.07, 0.14, 0.07][k], stroke=MUTED)
            band.move_to([slot_x(64 * k + 32), STRIP_Y, 0])
            lines.add(band)
        line_names = m.VGroup(*[
            T(f"cache line {k}", 24, MUTED).next_to(lines[k], m.UP, buff=0.14)
            for k in range(3)
        ])
        slots = m.VGroup()
        for i in range(SLOTS):
            s = box(SLOT_W - 0.05, 0.7, BG, 1.0, stroke=MUTED)
            s.move_to([slot_x(4 * i + 2), STRIP_Y, 0])
            slots.add(s)
        edges = m.VGroup()
        for k in range(4):
            x = slot_x(64 * k)
            tick = m.Line([x, STRIP_Y - 0.8, 0], [x, STRIP_Y + 1.35, 0], color=INK, stroke_width=3)
            num = C(str(64 * k), 24, INK).next_to(tick, m.DOWN, buff=0.1)
            edges.add(m.VGroup(tick, num))

        self.play(m.FadeIn(slots), run_time=0.8)
        self.say("Zoom out. Each small box is now one f32: 4 bytes.")
        self.play(m.FadeIn(lines), m.FadeIn(line_names), run_time=0.8)
        self.say("The cache moves memory in 64-byte lines. That is 16 floats per line.")
        self.play(m.LaggedStart(*[m.Create(e) for e in edges], lag_ratio=0.25), run_time=1.2)
        self.say("The line edges are fixed by the hardware: 0, 64, 128, 192. You cannot move them.")

        # One NEON load.
        new_head = T("Our code reads 16 bytes at a time", 38, INK).move_to(heading)
        self.play(m.ReplacementTransform(heading, new_head), run_time=0.5)
        heading = new_head
        demo = box(4 * SLOT_W, 0.5, TEAL, 0.0, stroke=TEAL).set_stroke(width=4)
        demo.move_to([slot_x(8), STRIP_Y + 1.05, 0])
        demo_label = T("one load = 16 bytes = 4 floats", 24, TEAL).next_to(demo, m.UP, buff=0.12)
        demo_label.align_to(demo, m.LEFT)
        self.play(m.FadeIn(demo), m.FadeIn(demo_label), m.FadeOut(line_names), run_time=0.6)
        self.say("NEON, the phone's vector unit, loads four floats in one instruction.")
        self.say("So a loop walks an array in 16-byte steps.", extra=0.2)
        self.play(m.FadeOut(demo), m.FadeOut(demo_label), run_time=0.4)

        # Case A: the array starts on a line edge.
        new_head = T("Case 1: the array starts at 0", 38, TEAL).move_to(heading)
        self.play(m.ReplacementTransform(heading, new_head), run_time=0.5)
        heading = new_head
        loads_a, splits_a = self.run_loads(slots, lines, offset=0)
        self.say("Four loads fill each line exactly. No load crosses an edge.")
        self.play(m.FadeOut(loads_a), m.FadeOut(splits_a), *[s.animate.set_fill(BG) for s in slots], run_time=0.5)

        # Case B: four bytes later.
        new_head = T("Case 2: the array starts at byte 4", 38, COPPER).move_to(heading)
        self.play(m.ReplacementTransform(heading, new_head), run_time=0.5)
        heading = new_head
        self.say("Now start the very same array just 4 bytes later.", extra=-0.4)
        loads_b, splits_b = self.run_loads(slots, lines, offset=4)
        self.say("Every fourth load sticks out over a line edge. It touches two lines.")
        first_split = loads_b[2 + 3]
        y = first_split.get_top()[1] + 0.12
        left = m.Line([slot_x(52), y, 0], [slot_x(64), y, 0])
        right = m.Line([slot_x(64), y, 0], [slot_x(68), y, 0])
        b_left = m.Brace(left, m.UP, color=COPPER, buff=0.04)
        b_right = m.Brace(right, m.UP, color=COPPER, buff=0.04)
        t_left = T("12 bytes in line 0", 24, COPPER).next_to(b_left, m.UP, buff=0.08)
        t_left.align_to(b_left, m.RIGHT).shift(m.LEFT * 0.15)
        t_right = T("4 bytes in line 1", 24, COPPER).next_to(b_right, m.UP, buff=0.08)
        t_right.align_to(b_right, m.LEFT).shift(m.RIGHT * 0.15)
        self.play(m.FadeOut(splits_b), run_time=0.3)
        self.play(m.FadeIn(b_left), m.FadeIn(t_left), run_time=0.6)
        self.play(m.FadeIn(b_right), m.FadeIn(t_right), run_time=0.6)
        self.say("Look closer: 12 bytes come from one line and 4 from the next.", extra=0.5)
        self.say("That is called a split load.", extra=0.2)
        self.say("It still gives the right answer on this phone. It just costs extra work.")
        self.say("How much extra on this core is something Phase 1 will measure.")
        self.clear()

    def run_loads(self, slots, lines, offset):
        """Animate eleven 16-byte loads over the strip, starting at `offset` bytes."""
        first = offset // 4
        self.play(*[slots[i].animate.set_fill(BLUE, opacity=0.55) for i in range(first, first + 44)], run_time=0.5)
        start_mark = m.Arrow(
            [slot_x(offset), STRIP_Y - 2.0, 0], [slot_x(offset), STRIP_Y - 1.2, 0],
            color=INK, buff=0, stroke_width=4,
        )
        start_txt = T(f"array starts at byte {offset}", 24, INK).next_to(start_mark, m.DOWN, buff=0.1)
        start_txt.shift(m.RIGHT * (start_txt.width / 2 - 0.2))
        self.play(m.GrowArrow(start_mark), m.FadeIn(start_txt), run_time=0.5)
        self.wait(0.8)

        loads = m.VGroup(start_mark, start_txt)
        count = 0
        counter = T("split loads: 0 of 0", 26, INK).move_to([0, 1.75, 0])
        self.play(m.FadeIn(counter), run_time=0.3)
        for i in range(11):
            start = offset + 16 * i
            split = start // 64 != (start + 15) // 64
            color = COPPER if split else TEAL
            ld = box(4 * SLOT_W - 0.04, 0.42, color, 0.25, stroke=color).set_stroke(width=3.5)
            ld.move_to([slot_x(start + 8), STRIP_Y + 0.95, 0])
            loads.add(ld)
            count += split
            new_counter = T(f"split loads: {count} of {i + 1}", 26, COPPER if count else INK).move_to(counter)
            anims = [m.FadeIn(ld, shift=m.DOWN * 0.15), m.Transform(counter, new_counter)]
            slow = i < 4 or split
            self.play(*anims, run_time=0.55 if slow else 0.3)
            if split:
                k = start // 64
                self.play(
                    m.Indicate(lines[k], color=COPPER, scale_factor=1.0),
                    m.Indicate(lines[k + 1], color=COPPER, scale_factor=1.0),
                    m.Flash(ld, color=COPPER, line_length=0.18),
                    run_time=0.9,
                )
        return loads, counter

    # ------------------------------------------------------------------ 7
    def definition(self):
        self.heading("What “aligned” means")
        rule = T("An address is aligned to N when it is a multiple of N.", 32, INK).move_to([0, 2.15, 0])
        code = C("address % N == 0", 30, TEAL).next_to(rule, m.DOWN, buff=0.3)
        self.play(m.FadeIn(rule), run_time=0.5)
        self.say("Now the word itself.", extra=0.6)
        self.play(m.FadeIn(code), run_time=0.4)
        self.say("In code: the remainder after dividing by N is zero.")

        heads = m.VGroup(*[T(t, 28, MUTED) for t in ["address", "4-aligned", "16-aligned", "64-aligned"]])
        xs = [-4.2, -1.2, 1.4, 4.0]
        for h, x in zip(heads, xs):
            h.move_to([x, 0.35, 0])
        self.play(m.FadeIn(heads), run_time=0.4)
        rows = [
            (1024, "a line edge: perfect"),
            (1040, "no splits, but not a line edge"),
            (1028, "fine for one f32, but loads will split"),
        ]
        for r, (addr, verdict) in enumerate(rows):
            y = -0.4 - r * 0.8
            items = [C(str(addr), 28, INK).move_to([xs[0], y, 0])]
            for n, x in zip([4, 16, 64], xs[1:]):
                ok = addr % n == 0
                items.append(T("yes" if ok else "no", 28, TEAL if ok else COPPER).move_to([x, y, 0]))
            self.play(m.LaggedStart(*[m.FadeIn(it) for it in items], lag_ratio=0.3), run_time=1.0)
            self.say(f"{addr}: {verdict}.", extra=0.6)
        self.say("Start every array at a multiple of 64, and no 16-byte load ever splits.")
        self.clear()

    # ------------------------------------------------------------------ 8
    def rust(self):
        self.heading("What Rust promises today")

        def card(title, promise, note, color):
            frame = m.Rectangle(width=6.0, height=2.9, stroke_color=color, stroke_width=2.5,
                                fill_color=color, fill_opacity=0.1, color=color)
            t = C(title, 36, color)
            p = T(promise, 32, INK)
            n = T(note, 26, MUTED)
            words = m.VGroup(t, p, n).arrange(m.DOWN, buff=0.3)
            words.move_to(frame)
            return m.VGroup(frame, words)

        vec = card("Vec<f32>", "start: multiple of 4", "often better, but only by luck", COPPER)
        ours = card("aligned buffer", "start: multiple of 64", "guaranteed, every time", TEAL)
        m.VGroup(vec, ours).arrange(m.RIGHT, buff=0.5).move_to([0, 0.35, 0])

        self.play(m.FadeIn(vec), run_time=0.6)
        self.say("A Vec of f32 only promises what one f32 needs: a multiple of 4.")
        self.say("It is often better aligned in practice. But often is luck, not a promise.")
        self.play(m.FadeIn(ours), run_time=0.6)
        self.say("So Phase 1 builds a small buffer whose start is always a multiple of 64.")
        self.say("Then every benchmark starts from the same layout, every run.")
        self.clear()

    # ------------------------------------------------------------------ 9
    def summary(self):
        self.heading("In three lines")
        points = [
            ("1", "The cache works in 64-byte lines with fixed edges.", BLUE),
            ("2", "Our code reads 16 bytes at a time.", TEAL),
            ("3", "Start arrays at a multiple of 64: no split loads.", INK),
        ]
        rows = m.VGroup()
        for num, text, color in points:
            dot = m.Circle(radius=0.28, color=color, fill_color=color, fill_opacity=1)
            n = T(num, 24, ON).move_to(dot)
            rows.add(m.VGroup(m.VGroup(dot, n), T(text, 32, INK)).arrange(m.RIGHT, buff=0.35))
        rows.arrange(m.DOWN, buff=0.55, aligned_edge=m.LEFT).move_to([0, 0.1, 0])
        for row in rows:
            self.play(m.FadeIn(row, shift=m.RIGHT * 0.2), run_time=0.6)
            self.wait(2.4)
        self.wait(2.0)
