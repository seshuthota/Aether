"""Q8_K goldens for the K-quant cases in `testdata/quant_k`.

`tools/gen_quant.py` writes each case's weights, x, Q8_0 activation and the
"b" references. This script adds what K2 needs, using llama.cpp's own
`quantize_row_q8_K_ref` from ggml-base (the same C function llama.cpp runs,
called through ctypes as `gen_quant_imatrix.py` calls the quantizers):

    tools/.venv/bin/python -I tools/gen_q8k.py \\
        ~/llama.cpp/build/bin/libggml-base.so testdata/quant_k

Per case directory:

  x_q8_k.bin                 x quantized to Q8_K, row by row (292-byte
                             blocks: f32 d, 256 int8 qs, 16 int16 bsums)
  out_q4_k_k.npy, out_q6_k_k.npy   [t, d]: dequant(X_q8_K) @ dequant(W)ᵀ,
                             in float64 then cast to f32

The output buffer is zeroed before the call, so the bsums of an all-zero
block (which the reference leaves untouched) are zero, as our quantizer
writes them.
"""

import ctypes
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import quants

Q8K = np.dtype([("d", "<f4"), ("qs", "i1", (256,)), ("bsums", "<i2", (16,))], align=False)


def main():
    lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "llama.cpp/build/bin/libggml-base.so"))
    root = Path(sys.argv[2] if len(sys.argv) > 2 else "testdata/quant_k")
    f = lib.quantize_row_q8_K_ref
    f.restype = None
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

    assert Q8K.itemsize == 292, Q8K.itemsize
    for case in sorted(root.glob("quant_k_*")):
        x = np.ascontiguousarray(np.load(case / "x.npy"), dtype=np.float32)
        t, n = x.shape
        assert n % 256 == 0, case
        blocks = np.zeros(t * n // 256 * Q8K.itemsize, np.uint8)
        f(x.ctypes.data, blocks.ctypes.data, t * n)
        (case / "x_q8_k.bin").write_bytes(blocks.tobytes())

        q = blocks.view(Q8K)
        xd = q["d"][:, None].astype(np.float64) * q["qs"].astype(np.float64)
        xd = xd.reshape(t, n)
        d = np.load(case / "deq_q4_k.npy").shape[0]
        for typ, name in [(T.Q4_K, "q4_k"), (T.Q6_K, "q6_k")]:
            wq = np.fromfile(case / f"w_{name}.bin", np.uint8).reshape(d, -1)
            wd = quants.dequantize(wq, typ).astype(np.float64)
            np.save(case / f"out_{name}_k.npy", (xd @ wd.T).astype(np.float32))
        print(case)


if __name__ == "__main__":
    main()
