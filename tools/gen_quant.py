"""Golden data for Q8_0 / Q4_0 using llama.cpp's own reference quantizers
(the `gguf` Python package from the llama.cpp repository, bit-exact with
ggml-quants.c).

    tools/.venv/bin/python -I tools/gen_quant.py testdata/quant

Per case directory quant_<d>x<n>/ (n is a multiple of 32, the block size):

  w.npy   [d, n] f32         x.npy  [n] f32
  w_q8_0.bin, w_q4_0.bin     the weight rows quantized, raw block bytes
  x_q8_0.bin                 x quantized to Q8_0 (what a kernel using integer
                             dot products does to the activation)
  out_q8_0_a.npy             dequant(W_q8_0) @ x            (float64 → f32)
  out_q8_0_b.npy             dequant(W_q8_0) @ dequant(x_q8_0)
  out_q4_0_a.npy, out_q4_0_b.npy   the same for Q4_0

"a" is the reference for kernels that dequantize weights and multiply by the
f32 activation; "b" for kernels that quantize the activation and use int8
dot products (sdot / smmla).
"""

import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import quants

rng = np.random.default_rng(606)


def main():
    root = Path(sys.argv[1])
    shapes = [(1, 32), (3, 64), (4, 96), (5, 160), (8, 288), (17, 576), (64, 576), (288, 768), (576, 1536), (1536, 576), (896, 4864), (130, 2048)]
    for d, n in shapes:
        w = (rng.standard_normal((d, n)) * 0.05).astype(np.float32)
        # A few exact zeros and an all-zero block exercise d = 0.
        w[0, :32] = 0.0
        x = rng.standard_normal(n).astype(np.float32)
        out = root / f"quant_{d}x{n}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "w.npy", w)
        np.save(out / "x.npy", x)
        xq = quants.quantize(x.reshape(1, n), T.Q8_0)
        (out / "x_q8_0.bin").write_bytes(xq.tobytes())
        xd = quants.dequantize(xq, T.Q8_0).reshape(n).astype(np.float64)
        for t, name in [(T.Q8_0, "q8_0"), (T.Q4_0, "q4_0")]:
            wq = quants.quantize(w, t)
            (out / f"w_{name}.bin").write_bytes(wq.tobytes())
            wd = quants.dequantize(wq, t).astype(np.float64)
            np.save(out / f"out_{name}_a.npy", (wd @ x.astype(np.float64)).astype(np.float32))
            np.save(out / f"out_{name}_b.npy", (wd @ xd).astype(np.float32))
    print(f"wrote {len(shapes)} cases to {root}")


if __name__ == "__main__":
    main()
