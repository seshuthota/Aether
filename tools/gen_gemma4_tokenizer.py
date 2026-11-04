"""Golden strings for Gemma 4's tokenizer, from HF `tokenizers`.

    tools/.venv/bin/python -I tools/gen_gemma4_tokenizer.py \
        hostdata/gemma-4-E2B-it/tokenizer.json testdata/gemma4

Writes corpus.txt (0x1E-separated), encoded.txt (one id list per line,
without the BOS the model adds) and decoded.txt (0x1E-separated), the format
`src/tokenizer/spm_bpe.rs` checks, as `gen_lfm3_tokenizer.py` does for LFM3.
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
    "0 12 345 6789 3.14159 1,000,000",
    "Ελληνικά, русский, עברית, العربية, हिन्दी, ไทย, 한국어",
    "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n",
    "<|turn>system\n<|think|>You are a helpful assistant.<turn|>\n",
    "<|channel>thought\nreasoning<channel|>answer",
    "<bos>Hello<eos>",
    "def f(x):\n    return x**2  # square\n\n\tif x  >= 10:",
    "URL https://example.com/a?b=c&d=e#f and email a.b@c.de",
    "\n",
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
    "a\n\nb\n\n\nc",
    "   leading and trailing   ",
    "▁literal metaspace▁",
    "\x00\x01\x7f control bytes",
    "\U0001f9ec\U0001fae0 rare emoji \U00010348",
]


def main() -> None:
    tokenizer, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(16)
    cases = list(CASES) + EXTRA
    for _ in range(400):
        cases.append("".join(rng.choice(PIECES + [" ", "\n", "\n\n", "x", "Ω", "1", "<turn|>", "<|turn>"]) for _ in range(rng.randint(1, 25))))
    tok = Tokenizer.from_file(str(tokenizer))
    enc = [tok.encode(s, add_special_tokens=False).ids for s in cases]
    (out / "corpus.txt").write_text("\x1e".join(cases), encoding="utf-8")
    (out / "encoded.txt").write_text("\n".join(" ".join(map(str, e)) for e in enc) + "\n")
    (out / "decoded.txt").write_text("\x1e".join(tok.decode(e, skip_special_tokens=False) for e in enc), encoding="utf-8")
    print(f"wrote {len(cases)} strings to {out}")


if __name__ == "__main__":
    main()
