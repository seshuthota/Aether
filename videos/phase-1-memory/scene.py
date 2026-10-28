"""Explainer for site/phase-1/memory.html.

Muted on purpose: the notes page has to work without sound, so every idea
is carried by a caption held long enough to read.
Every size and latency on screen is an order of magnitude or a published
claim, never a measurement from this phone.
"""

import manimgx as m

BG = "#121816"
INK = "#f3efe6"
MUTED = "#b7b0a4"
TEAL = "#7dcec4"
BLUE = "#9ec0e8"
COPPER = "#e0a07a"
ON = "#121816"

m.config.background_color = BG


def T(text, size=32, color=INK):
    return m.Text(text, font="sans-serif", font_size=size, color=color)


def chip(width, height, color, opacity=1.0):
    return m.Rectangle(
        width=width,
        height=height,
        stroke_width=1.5,
        stroke_color=color,
        fill_color=color,
        fill_opacity=opacity,
        color=color,
    )


def labelled(width, height, color, text, size=26):
    box = chip(width, height, color)
    return m.VGroup(box, T(text, size, ON).move_to(box))


class MemoryHierarchy(m.Scene):
    def construct(self):
        self.cap = None
        self.title_card()
        self.the_wait()
        self.the_cache()
        self.the_ladder()
        self.latency()
        self.bandwidth()
        self.capacity()
        self.levers()
        self.close()

    # A caption stays up for a reading pace of roughly three words a second.
    def say(self, text, extra=0.0):
        new = T(text, 28, INK)
        if new.width > 12.8:
            new.scale_to_fit_width(12.8)
        new.to_edge(m.DOWN, buff=0.32)
        if self.cap is None:
            self.play(m.FadeIn(new), run_time=0.3)
        else:
            self.play(m.FadeOut(self.cap), m.FadeIn(new), run_time=0.3)
        self.cap = new
        self.wait(0.7 + 0.28 * len(text.split()) + extra)

    def clear(self):
        mobs = list(self.mobjects)
        if mobs:
            self.play(*[m.FadeOut(mob) for mob in mobs], run_time=0.45)
        self.cap = None

    def heading(self, text):
        h = T(text, 36, INK).to_edge(m.UP, buff=0.34)
        self.play(m.FadeIn(h), run_time=0.3)
        return h

    # ------------------------------------------------------------------ 1
    def title_card(self):
        kicker = T("Phase 1", 24, TEAL)
        title = T("The memory hierarchy", 56, INK)
        sub = T("why decode waits on DRAM, every token", 30, MUTED)
        m.VGroup(kicker, title, sub).arrange(m.DOWN, buff=0.3)
        self.play(m.FadeIn(kicker), m.Write(title), run_time=1.1)
        self.play(m.FadeIn(sub, shift=m.UP * 0.15))
        self.wait(1.4)
        self.clear()

    # ------------------------------------------------------------------ 2
    def the_wait(self):
        self.heading("The core is fast. DRAM is far.")
        core = labelled(2.1, 1.45, TEAL, "core", 30).move_to([-4.0, 0.5, 0])
        dram = labelled(3.3, 1.7, COPPER, "DRAM", 30).move_to([3.9, 0.5, 0])
        dram_note = T("weights live here", 22, COPPER).next_to(dram, m.DOWN, buff=0.2)
        arrow = m.Arrow(
            dram.get_left(), core.get_right(), color=MUTED, buff=0.15, stroke_width=3
        )
        self.play(m.FadeIn(core), m.FadeIn(dram), m.FadeIn(dram_note), m.Create(arrow), run_time=0.8)
        self.say("Model weights live in DRAM, the phone's main memory.")

        weight = chip(0.34, 0.34, COPPER).move_to(dram[0].get_left() + m.RIGHT * 0.4)
        cycles = m.ValueTracker(0)
        counter = m.always_redraw(
            lambda: T(f"idle cycles: {int(cycles.get_value())}", 24, MUTED).next_to(
                core, m.DOWN, buff=0.3
            )
        )
        self.play(m.FadeIn(weight), m.FadeIn(counter), run_time=0.3)
        self.say("Ask DRAM for one number, and watch the core wait.", extra=-0.6)
        self.play(
            weight.animate.move_to(core[0].get_right() + m.RIGHT * 0.36),
            cycles.animate.set_value(400),
            run_time=3.0,
            rate_func=m.linear,
        )
        typical = T("~100 ns at ~4 GHz  ·  typical, not measured", 20, COPPER)
        typical.next_to(counter, m.DOWN, buff=0.14)
        self.play(m.FadeIn(typical), run_time=0.3)
        self.say("One trip takes about 100 ns: hundreds of cycles doing nothing.")
        self.say("The hardware's answer: keep copies of data close to the core.")
        self.clear()

    # ------------------------------------------------------------------ 3
    def the_cache(self):
        self.heading("A cache keeps copies close by")
        core = labelled(1.7, 1.3, TEAL, "core", 28).move_to([-5.2, 0.6, 0])

        cache_box = m.Rectangle(width=3.2, height=1.3, stroke_color=TEAL, stroke_width=2.5)
        cache_box.move_to([-1.3, 0.6, 0])
        cache_name = T("cache", 24, TEAL).next_to(cache_box, m.UP, buff=0.14)
        cache_size = T("small, fast", 20, MUTED).next_to(cache_box, m.DOWN, buff=0.14)
        slots = [cache_box.get_left() + m.RIGHT * (0.5 + 0.73 * i) for i in range(4)]

        dram_box = m.Rectangle(width=3.5, height=2.2, stroke_color=COPPER, stroke_width=2.5)
        dram_box.move_to([4.2, 0.6, 0])
        dram_name = T("DRAM", 24, COPPER).next_to(dram_box, m.UP, buff=0.14)
        dram_size = T("huge, slow", 20, MUTED).next_to(dram_box, m.DOWN, buff=0.14)
        cells = m.VGroup()
        for r in range(2):
            for c in range(4):
                cell = labelled(0.62, 0.62, COPPER, f"w{r * 4 + c}", 20)
                cell.move_to(dram_box.get_center() + [(c - 1.5) * 0.78, (0.5 - r) * 0.82, 0])
                cells.add(cell)
        line = m.SurroundingRectangle(m.VGroup(*cells[:4]), color=INK, buff=0.07, stroke_width=2)
        line_label = T("one cache line = 64 bytes", 20, INK).next_to(dram_size, m.DOWN, buff=0.16)

        self.play(
            m.FadeIn(core),
            m.FadeIn(cache_box), m.FadeIn(cache_name), m.FadeIn(cache_size),
            m.FadeIn(dram_box), m.FadeIn(dram_name), m.FadeIn(dram_size), m.FadeIn(cells),
            run_time=0.8,
        )

        # First request: a miss.
        req = T("need w0", 24, INK).next_to(core, m.DOWN, buff=0.3)
        self.play(m.FadeIn(req), run_time=0.3)
        probe = m.Arrow(core.get_right(), cache_box.get_left(), color=INK, buff=0.1, stroke_width=3)
        self.play(m.GrowArrow(probe), run_time=0.5)
        miss = T("miss", 30, COPPER).move_to(cache_box)
        self.play(m.FadeIn(miss), run_time=0.3)
        self.say("Ask for w0. The cache does not have it: a miss.", extra=-0.3)

        self.play(m.Create(line), m.FadeIn(line_label), run_time=0.5)
        copies = m.VGroup(*[cell.copy() for cell in cells[:4]])
        self.play(m.FadeOut(miss), run_time=0.2)
        self.play(
            *[c.animate.move_to(s) for c, s in zip(copies, slots)],
            run_time=2.2,
            rate_func=m.smooth,
        )
        self.say("DRAM sends a whole 64-byte line: w0 and its neighbours. The cache keeps a copy.")

        # Second request: a hit.
        req2 = T("need w1", 24, INK).move_to(req)
        self.play(m.ReplacementTransform(req, req2), run_time=0.3)
        hit = T("hit", 30, TEAL).next_to(cache_box, m.UP, buff=0.55)
        self.play(m.Indicate(copies[1][0], color=TEAL), m.FadeIn(hit), run_time=0.6)
        self.say("Ask for w1. It is already there: a hit, about a hundred times sooner.")
        self.say("Caches bet you reuse data soon, and use its neighbours. That bet is called locality.")
        self.clear()

    # ------------------------------------------------------------------ 4
    def the_ladder(self):
        self.heading("Real chips stack several caches")
        specs = [
            (2.0, "registers", "under 1 KB", TEAL),
            (3.1, "L1 cache", "tens of KB", TEAL),
            (4.2, "L2 cache", "megabytes", BLUE),
            (5.2, "SLC", "megabytes", BLUE),
            (6.3, "DRAM", "gigabytes", COPPER),
        ]
        rows = m.VGroup()
        for width, name, _size, color in specs:
            rows.add(labelled(width, 0.66, color, name, 24))
        rows.arrange(m.DOWN, buff=0.13, aligned_edge=m.LEFT)
        rows.move_to([-0.2, 0.05, 0])
        sizes = m.VGroup(*[
            T(spec[2], 22, MUTED).next_to(row, m.RIGHT, buff=0.22) for row, spec in zip(rows, specs)
        ])
        inside = T("inside the core", 20, TEAL).next_to(rows[0], m.LEFT, buff=0.3)
        near = T("closer, faster", 22, TEAL).next_to(rows[1], m.LEFT, buff=0.3)
        far = T("bigger, slower", 22, COPPER).next_to(rows[-1], m.LEFT, buff=0.3)

        self.play(
            m.LaggedStart(*[m.FadeIn(r, shift=m.UP * 0.08) for r in rows], lag_ratio=0.18),
            run_time=1.4,
        )
        self.play(m.FadeIn(sizes), m.FadeIn(inside), m.FadeIn(near), m.FadeIn(far), run_time=0.5)
        self.say("Each step down is bigger, and slower. Wider means bigger, not faster.")

        # A load checks each level in turn, top to bottom. Marks sit in a
        # column right of the size labels so they never cover a rung's name.
        col = sizes.get_right()[0] + 0.9
        marks = []
        for row in rows[1:-1]:
            mk = T("miss", 22, COPPER).move_to([col, row.get_y(), 0])
            self.play(m.Circumscribe(row[0], color=INK), m.FadeIn(mk), run_time=0.55)
            marks.append(mk)
        found = T("found", 22, COPPER).move_to([col, rows[-1].get_y(), 0])
        self.play(m.Circumscribe(rows[-1][0], color=INK), m.FadeIn(found), run_time=0.55)
        self.say("A load checks each level in turn. Missing every cache means a trip to DRAM.")

        # The data climbs back up, leaving copies on the way.
        data = chip(0.3, 0.3, INK).move_to([col + 0.9, rows[-1].get_y(), 0])
        self.play(m.FadeIn(data), run_time=0.2)
        for row, mk in zip(reversed(rows[1:-1]), reversed(marks)):
            copy = chip(0.24, 0.24, INK).move_to([col, row.get_y(), 0])
            self.play(data.animate.move_to([col + 0.9, row.get_y(), 0]), m.FadeOut(mk), run_time=0.45)
            self.play(m.FadeIn(copy), run_time=0.15)
        self.play(data.animate.move_to([col, rows[0].get_y(), 0]), run_time=0.45)
        self.play(m.Circumscribe(rows[0][0], color=TEAL), run_time=0.5)
        self.say("The data climbs back up, and each cache keeps a copy on the way.")
        self.say("These sizes are orders of magnitude, not measurements of this phone.", extra=-0.4)
        self.clear()

    # ------------------------------------------------------------------ 5
    def latency(self):
        self.heading("Latency: how long one load waits")
        note = T("1 ns drawn as 1 second  ·  typical phone CPU, not measured  ·  log scale", 20, COPPER)
        note.next_to(self.mobjects[-1], m.DOWN, buff=0.16)
        self.play(m.FadeIn(note), run_time=0.3)

        import math

        levels = [
            ("register", 0.3, "0.3 s", TEAL),
            ("L1 hit", 1, "1 second", TEAL),
            ("L2 hit", 5, "5 seconds", BLUE),
            ("SLC hit", 20, "20 seconds", BLUE),
            ("DRAM", 120, "2 minutes", COPPER),
        ]
        left = -2.6
        bars = []
        for i, (name, ns, human, color) in enumerate(levels):
            y = 1.35 - i * 0.78
            width = 0.3 + 6.4 * math.log10(ns / 0.3) / math.log10(120 / 0.3)
            label = T(name, 24, color).move_to([left - 0.3, y, 0], aligned_edge=m.RIGHT)
            bar = chip(width, 0.42, color).move_to([left + width / 2, y, 0])
            time = T(human, 24, INK).next_to(bar, m.RIGHT, buff=0.22)
            bars.append((label, bar, time))
        for label, bar, time in bars:
            self.play(m.FadeIn(label), m.GrowFromEdge(bar, m.LEFT), m.FadeIn(time), run_time=0.55)
        self.say("Stretch time so a nanosecond lasts a second. An L1 hit takes one second.")
        self.say("A trip to DRAM takes two minutes. Waiting out each one would waste almost everything.")
        self.clear()

    # ------------------------------------------------------------------ 6
    def bandwidth(self):
        self.heading("Bandwidth: how many bytes arrive per second")
        travel = 1.4
        sim_time = 8.0
        x0, x1 = -4.6, 3.2
        lanes = [(1.25, "one load at a time", travel, MUTED), (-1.15, "many loads in flight", 0.2, TEAL)]
        clock = m.ValueTracker(0)
        dots = []
        for y, title, gap, color in lanes:
            track = m.Line([x0, y, 0], [x1, y, 0], color=MUTED, stroke_width=3)
            src = T("DRAM", 22, COPPER).next_to(track.get_start(), m.LEFT, buff=0.2)
            dst = T("core", 22, TEAL).next_to(track.get_end(), m.RIGHT, buff=0.2)
            name = T(title, 24, color).next_to(track, m.UP, buff=0.32).align_to(track, m.LEFT)
            self.add(track, src, dst, name)
            launches = [k * gap for k in range(int(sim_time / gap) + 1)]
            for launch in launches:
                dot = chip(0.26, 0.26, COPPER)

                def place(mob, launch=launch, y=y):
                    t = clock.get_value() - launch
                    if 0 <= t <= travel:
                        mob.move_to([x0 + (x1 - x0) * t / travel, y, 0]).set_opacity(1)
                    else:
                        mob.move_to([x0, y, 0]).set_opacity(0)

                place(dot)
                dot.add_updater(place)
                dots.append(dot)

            def delivered(launches=launches, y=y):
                n = sum(1 for s in launches if clock.get_value() - s > travel)
                return T(f"{n} lines delivered", 24, INK).move_to([x1 + 2.1, y - 0.55, 0])

            self.add(m.always_redraw(delivered))
        self.add(*dots)
        self.say("Both lanes have the same two-minute trip. Only the number of loads in flight differs.", extra=-1.5)
        self.play(clock.animate.set_value(sim_time), run_time=sim_time, rate_func=m.linear)
        for d in dots:
            d.clear_updaters()
        self.say("Overlapping loads moves far more bytes per second. That rate is bandwidth.")
        self.say("Decode streams gigabytes, so decode cares about bandwidth, not the single wait.")
        self.say("A core can only track so many loads at once, so one core may not fill the bus.")
        self.clear()

    # ------------------------------------------------------------------ 7
    def capacity(self):
        self.heading("Why the cache cannot keep the weights")
        n = 12
        blocks = m.VGroup(*[labelled(0.62, 0.62, COPPER, str(i + 1), 22) for i in range(n)])
        blocks.arrange(m.RIGHT, buff=0.12).move_to([0, 1.7, 0])
        order = T("weight blocks, in the order one token reads them", 22, MUTED)
        order.next_to(blocks, m.UP, buff=0.18)

        cache_box = m.Rectangle(width=2.75, height=1.0, stroke_color=TEAL, stroke_width=2.5)
        cache_box.move_to([-1.0, -0.6, 0])
        cache_name = T("cache: room for 3", 22, TEAL).next_to(cache_box, m.DOWN, buff=0.16)
        slots = [cache_box.get_left() + m.RIGHT * (0.5 + 0.87 * i) for i in range(3)]
        core = labelled(1.5, 1.0, TEAL, "core", 26).move_to([3.4, -0.6, 0])
        to_core = m.Arrow(cache_box.get_right(), core.get_left(), color=MUTED, buff=0.12, stroke_width=3)
        token = T("token 1", 26, INK).move_to([-5.3, -0.6, 0])

        self.play(
            m.FadeIn(blocks), m.FadeIn(order), m.FadeIn(cache_box), m.FadeIn(cache_name),
            m.FadeIn(core), m.Create(to_core), m.FadeIn(token),
            run_time=0.8,
        )
        self.say("Decode reads every weight once per token, in order, start to finish.", extra=-0.5)

        held = []
        for i, block in enumerate(blocks):
            speed = 0.6 if i < 4 else 0.28
            copy = block.copy()
            anims = []
            if len(held) == 3:
                old = held.pop(0)
                anims.append(old.animate.shift(m.DOWN * 0.9).set_opacity(0))
                anims += [h.animate.move_to(s) for h, s in zip(held, slots)]
            anims.append(copy.animate.move_to(slots[len(held)]))
            self.play(*anims, run_time=speed)
            held.append(copy)
            self.play(m.Indicate(core[0], color=BLUE, scale_factor=1.08), run_time=speed * 0.6)
            if i == 3:
                self.say("Once the cache is full, each new block pushes out the oldest.", extra=-0.8)
        self.say("Token 1 is done. The cache holds only blocks 10, 11 and 12.", extra=-0.3)

        token2 = T("token 2", 26, INK).move_to(token)
        self.play(m.ReplacementTransform(token, token2), m.Indicate(blocks[0][0], color=INK), run_time=0.6)
        miss = T("block 1? miss", 26, COPPER).next_to(cache_box, m.UP, buff=0.3)
        self.play(m.FadeIn(miss), m.Indicate(cache_box, color=COPPER), run_time=0.6)
        self.say("Token 2 starts again at block 1. It was pushed out long ago: another miss.")
        self.say("Every block misses again, on every token. The cache never gets a second use.")
        self.clear()

        # The real ratio.
        self.heading("At real sizes")
        cells = [chip(0.36, 0.36, TEAL if i == 0 else COPPER) for i in range(100)]
        grid = m.VGroup(*cells)
        for r in range(10):
            for c in range(10):
                cells[r * 10 + c].move_to([c * 0.42, -r * 0.42, 0])
        grid.move_to([-1.8, 0.0, 0])
        fit = m.VGroup(chip(0.32, 0.32, TEAL), T("could fit in all the caches", 22, INK))
        fit[1].next_to(fit[0], m.RIGHT, buff=0.16)
        out = m.VGroup(chip(0.32, 0.32, COPPER), T("comes from DRAM, every token", 22, INK))
        out[1].next_to(out[0], m.RIGHT, buff=0.16)
        legend = m.VGroup(fit, out).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
        legend.next_to(grid, m.RIGHT, buff=0.55)
        sizes = m.VGroup(
            T("0.5B model in f32: 2 GB", 20, COPPER),
            T("all caches: tens of MB (assumption)", 20, COPPER),
        ).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        sizes.next_to(legend, m.DOWN, buff=0.4).align_to(legend, m.LEFT)
        self.play(m.FadeIn(grid), m.FadeIn(legend), m.FadeIn(sizes), run_time=0.7)
        self.play(m.Circumscribe(cells[0], color=TEAL))
        self.say("2 GB of weights against tens of megabytes of cache: about 1 in 100 fits.")
        self.say("This is a capacity miss. No clever ordering puts the model in the cache.")
        self.clear()

    # ------------------------------------------------------------------ 8
    def levers(self):
        self.heading("So what can make decode faster?")

        def card(title, color):
            box = m.Rectangle(
                width=3.9, height=3.0, stroke_color=color, stroke_width=2.5,
                fill_color=color, fill_opacity=0.12, color=color,
            )
            head = T(title, 28, color).move_to(box.get_top() + m.DOWN * 0.42)
            return m.VGroup(box, head)

        a = card("Read fewer bytes", TEAL)
        b = card("Reuse each byte", BLUE)
        c = card("A faster core", COPPER)
        m.VGroup(a, b, c).arrange(m.RIGHT, buff=0.3).move_to([0, 0.2, 0])

        self.play(m.FadeIn(a), run_time=0.4)
        big = chip(3.0, 0.45, COPPER).move_to(a[0].get_center() + m.UP * 0.1)
        big_l = T("f32: 2 GB", 20, INK).next_to(big, m.DOWN, buff=0.14)
        self.play(m.FadeIn(big), m.FadeIn(big_l), run_time=0.4)
        small_l = T("int8: 0.5 GB", 20, INK).move_to(big_l)
        self.play(
            big.animate.stretch_to_fit_width(0.75).align_to(big, m.LEFT).set_color(TEAL),
            m.ReplacementTransform(big_l, small_l),
            run_time=1.0,
        )
        self.say("Quantization stores each weight in fewer bits, so each token reads fewer bytes.")

        self.play(m.FadeIn(b), run_time=0.4)
        w = chip(0.5, 0.5, BLUE).move_to(b[0].get_center() + m.LEFT * 1.0 + m.DOWN * 0.15)
        toks = m.VGroup(*[chip(0.3, 0.3, INK) for _ in range(4)]).arrange(m.DOWN, buff=0.12)
        toks.move_to(b[0].get_center() + m.RIGHT * 1.0 + m.DOWN * 0.15)
        arrows = m.VGroup(*[
            m.Arrow(w.get_right(), t.get_left(), buff=0.08, color=MUTED, stroke_width=2) for t in toks
        ])
        tl = T("one read, many tokens", 20, INK).next_to(m.VGroup(w, toks), m.DOWN, buff=0.2)
        self.play(m.FadeIn(w), m.FadeIn(toks), m.LaggedStart(*[m.GrowArrow(x) for x in arrows]), m.FadeIn(tl), run_time=1.0)
        self.say("Or use each weight for many tokens per read: prefill, batching.")

        self.play(m.FadeIn(c), run_time=0.4)
        same = T("still waits on 2 GB", 22, COPPER).next_to(c, m.DOWN, buff=0.2)
        cross = m.Cross(c[0], stroke_color=COPPER, stroke_width=4)
        self.play(m.FadeIn(same), run_time=0.3)
        self.play(m.Create(cross), c.animate.set_opacity(0.45), run_time=0.6)
        self.say("A faster core does not help. It just waits sooner. The model is still 2 GB.")
        self.clear()

    # ------------------------------------------------------------------ 9
    def close(self):
        formula = m.MathTex(
            r"\text{tokens/s} \;\le\; \frac{\text{bandwidth (bytes/s)}}{\text{bytes read per token}}",
            font_size=46,
            color=INK,
        )
        formula.move_to([0, 0.75, 0])
        if formula.width > 12.2:
            formula.scale_to_fit_width(12.2)
        tag = T("the ceiling, counting weights only", 22, MUTED).next_to(formula, m.DOWN, buff=0.4)
        self.play(m.Write(formula), run_time=1.4)
        self.play(m.FadeIn(tag))
        self.say("Phase 1 measures the top of this fraction on this phone. Until then, it is a guess.")
        self.wait(1.0)
