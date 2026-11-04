"""Golden strings for LFM3's tokenizer, from HF `tokenizers`.

    tools/.venv/bin/python -I tools/gen_lfm3_tokenizer.py \
        testdata/models/LFM3-1.7B/tokenizer.json testdata/lfm3

Writes corpus.txt (0x1E-separated), encoded.txt (one id list per line) and
decoded.txt (0x1E-separated), the format `src/tokenizer/bytelevel.rs`
checks.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gen_tokenizer_corpus import CASES, PIECES  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

EXTRA = [
    "The capital of France is",
    "1234567",
    "١٢٣٤٥٦٧",
    "１２３４５６７",
    "0 12 345 6789 3.14159 1,000,000",
    "Ελληνικά, русский, עברית, العربية, हिन्दी, ไทย, 한국어",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    "<think>reasoning</think>answer",
    "<|endoftext|>Hello<|endoftext|>",
    "def f(x):\n    return x**2  # square\n\n\tif x  >= 10:",
    "URL https://example.com/a?b=c&d=e#f and email a.b@c.de",
    "MiXeD CaSe, ALLCAPS, contractions: I'm, you're, they've, we'll, he'd, it's, don't",
]


def main() -> None:
    tokenizer, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(15)
    cases = list(CASES) + EXTRA
    for _ in range(400):
        cases.append("".join(rng.choice(PIECES + [" ", "\n", "x", "Ω", "1", "<|im_end|>"]) for _ in range(rng.randint(1, 25))))
    tok = Tokenizer.from_file(str(tokenizer))
    enc = [tok.encode(s).ids for s in cases]
    (out / "corpus.txt").write_text("\x1e".join(cases), encoding="utf-8")
    (out / "encoded.txt").write_text("\n".join(" ".join(map(str, e)) for e in enc) + "\n")
    (out / "decoded.txt").write_text("\x1e".join(tok.decode(e, skip_special_tokens=False) for e in enc), encoding="utf-8")
    print(f"wrote {len(cases)} strings to {out}")


if __name__ == "__main__":
    main()
