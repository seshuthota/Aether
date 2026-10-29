"""Golden data for our own quantizers: llama.cpp's `quantize_q4_0`,
`quantize_q5_0`, `quantize_q4_K` and `quantize_q6_K`, with and without an
importance matrix, called straight from ggml-base (the C library
`llama-quantize` uses), so the reference is the exact code, not a port.

    tools/.venv/bin/python -I tools/gen_quant_imatrix.py \\
        ~/llama.cpp/build-nodl/bin/libggml-base.so testdata/quant_imatrix

ggml-base is compiled without -march (no FMA contraction, software f16
rounding), so its float arithmetic is plain IEEE f32 in source order, which
Rust reproduces bit for bit.

Per case directory qi_<d>x<n>/:

  w.npy   [d, n] f32           qw.npy  [n] f32  importance per column
  w_<type>.bin                 rows quantized without an importance matrix
  w_<type>_im.bin              rows quantized with qw

for <type> in q4_0, q5_0 and, when n is a multiple of 256, q4_k and q6_k.
The weights mix Gaussian rows, heavy-tailed rows (Student t, 3 degrees of
freedom), an outlier per row, an all-zero block and an all-zero row; qw has
the spread of a real importance matrix (log-normal) and a few zeros.
"""

import ctypes
import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(1729)

# ggml_type values (ggml.h).
TYPES = {"q4_0": (2, 32, 18), "q5_0": (6, 32, 22), "q4_k": (12, 256, 144), "q6_k": (14, 256, 210)}


def main():
    lib = ctypes.CDLL(sys.argv[1])
    root = Path(sys.argv[2])
    fns = {}
    for name in TYPES:
        f = getattr(lib, "quantize_" + name.replace("_k", "_K"))
        f.restype = ctypes.c_size_t
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        fns[name] = f

    shapes = [(1, 32), (3, 96), (4, 576), (5, 896), (1, 256), (3, 512), (7, 1536), (2, 2048), (16, 4864)]
    for d, n in shapes:
        w = (rng.standard_normal((d, n)) * 0.05).astype(np.float32)
        if d > 1:
            w[1] = (rng.standard_t(3, n) * 0.02).astype(np.float32)
        for r in range(d):
            w[r, rng.integers(n)] *= 12.0
        w[0, :32] = 0.0
        if d > 2:
            w[2] = 0.0
        qw = np.exp(rng.standard_normal(n) * 1.5).astype(np.float32)
        qw[rng.integers(n, size=max(1, n // 64))] = 0.0

        out = root / f"qi_{d}x{n}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "w.npy", w)
        np.save(out / "qw.npy", qw)
        src = np.ascontiguousarray(w)
        for name, (_, qk, bsize) in TYPES.items():
            if n % qk:
                continue
            for suffix, imatrix in (("", None), ("_im", qw)):
                dst = np.zeros(d * (n // qk) * bsize, np.uint8)
                size = fns[name](src.ctypes.data, dst.ctypes.data, d, n, None if imatrix is None else imatrix.ctypes.data)
                assert size == dst.size, (name, size, dst.size)
                (out / f"w_{name}{suffix}.bin").write_bytes(dst.tobytes())
        print(out)


if __name__ == "__main__":
    main()
