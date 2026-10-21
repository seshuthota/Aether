"""Writes a tokenizer test corpus: strings separated by 0x1E (record separator).

    tools/.venv/bin/python -I tools/gen_tokenizer_corpus.py testdata/stories15M/corpus.txt

Hand-picked edge cases first (empty, whitespace runs, punctuation, digits,
accents, CJK, emoji, bytes that need byte fallback), then random mixes drawn
from those pieces with a fixed seed, so the file is the same every run.
"""

import random
import sys

CASES = [
    "",
    " ",
    "  ",
    "Once upon a time",
    "Once upon a time, there was a little girl named Lily.",
    "Hello world",
    " Hello world",
    "hello  world   !",
    "\t\ttabs\tand\nnewlines\n\n",
    "It's 3:45pm, and 12345 + 678 = 13023.",
    "\"Quoted,\" she said -- then left...",
    "café naïve résumé façade",
    "日本語のテキスト",
    "中文测试",
    "emoji 🎉🚀 and 👩‍👩‍👧",
    "Ünïcödé ßtraße",
    "a" * 300,
    "The quick brown fox jumps over the lazy dog. " * 4,
    " non-breaking space",
    "zero​width",
    "math: ∑ x² ≤ ∞",
    "\x01\x02 control bytes",
    "<s> </s> <unk> look like special tokens",
]

PIECES = [
    "the", " the", "Tom", " Lily", " dog", "!", "?", ",", ".", " ", "  ", "\n",
    "1", "42", "2024", "é", "ñ", "中", "😀", "'s", "n't", "-", "...", "\t", "A", "ZZ",
]


def main():
    rng = random.Random(15)
    cases = list(CASES)
    for _ in range(300):
        cases.append("".join(rng.choice(PIECES) for _ in range(rng.randint(1, 20))))
    for s in cases:
        assert "\x1e" not in s
    with open(sys.argv[1], "w", encoding="utf-8", newline="") as f:
        f.write("\x1e".join(cases))
    print(f"wrote {len(cases)} strings to {sys.argv[1]}")


if __name__ == "__main__":
    main()
