"""NFC test cases from Python's unicodedata (the reference).

    tools/.venv/bin/python -I tools/gen_nfc_tests.py testdata/unicode/nfc.json

A JSON array of [input, NFC(input)] pairs, written with \\u escapes (so the
engine's JSON parser, surrogate pairs included, is exercised too):
every code point whose NFC differs from itself, every character with a
canonical decomposition, and random sequences of base letters, combining
marks (in scrambled order) and Hangul jamo.
"""

import json
import random
import sys
import unicodedata
from pathlib import Path


def main():
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    marks = [cp for cp in range(0x300, 0x370)] + [0x1DC0, 0x1DC1, 0x20D0, 0x0591, 0x05B0, 0x0E31, 0x302A, 0x3099, 0x309A]
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if unicodedata.normalize("NFC", c) != c or (unicodedata.decomposition(c) and not unicodedata.decomposition(c).startswith("<")):
            cases.append([c, unicodedata.normalize("NFC", c)])
    rng = random.Random(7)
    bases = "aeiouAEIOUcnsyzkhgwKHGW" + "αεηιοωΑΕ" + "дийДИ" + "かきくカキク"
    for _ in range(4000):
        s = ""
        for _ in range(rng.randint(1, 4)):
            r = rng.random()
            if r < 0.15:  # Hangul jamo: L V (T)
                s += chr(0x1100 + rng.randrange(19)) + chr(0x1161 + rng.randrange(21))
                if rng.random() < 0.5:
                    s += chr(0x11A8 + rng.randrange(27))
            else:
                s += rng.choice(bases)
            s += "".join(chr(rng.choice(marks)) for _ in range(rng.randint(0, 3)))
        cases.append([s, unicodedata.normalize("NFC", s)])
    out.write_text(json.dumps(cases, ensure_ascii=True))
    print(f"{len(cases)} cases (Unicode {unicodedata.unidata_version})")


if __name__ == "__main__":
    main()
