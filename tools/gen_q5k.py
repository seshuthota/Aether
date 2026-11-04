"""Q5_K goldens: random blocks and gguf's dequantized rows.

Gemma 4's Q4_K_M file stores its per-layer embedding table as Q5_K, and the
engine only ever reads rows of it (a lookup, no matrix product), so the
reference is gguf's dequantizer. As for Q4_K / Q6_K in `gen_quant.py`, the
`gguf` package cannot encode Q5_K, so the blocks are random bytes with
realistic f16 scales (every bit pattern is then a valid block):

    tools/.venv/bin/python -I tools/gen_q5k.py testdata/quant_q5k

Per case directory `q5k_<d>x<n>/`:

  w_q5_k.bin      [d] rows of raw 176-byte blocks
  deq_q5_k.npy    [d, n] gguf's dequantized weights
"""

import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import quants


def main():
    root = Path(sys.argv[1])
    rng = np.random.default_rng(609)
    cases = [(1, 256), (3, 512), (4, 2560), (5, 8960)]
    for d, n in cases:
        b = rng.integers(0, 256, size=(d * n // 256, 176), dtype=np.uint8)
        # d and dmin come first, f16, in 1e-4..1e-2.
        b[:, 0:4] = rng.uniform(1e-4, 1e-2, size=(b.shape[0], 2)).astype(np.float16).view(np.uint8)
        w = b.reshape(d, -1)
        out = root / f"q5k_{d}x{n}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "w_q5_k.bin").write_bytes(w.tobytes())
        np.save(out / "deq_q5_k.npy", quants.dequantize(w, T.Q5_K).astype(np.float32))
    print(f"wrote {len(cases)} cases to {root}")


if __name__ == "__main__":
    main()
