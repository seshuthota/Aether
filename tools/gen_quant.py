"""Golden data for Q8_0 / Q4_0 / Q5_0 using llama.cpp's own reference quantizers
(the `gguf` Python package from the llama.cpp repository, bit-exact with
ggml-quants.c).

    tools/.venv/bin/python -I tools/gen_quant.py testdata/quant testdata/quant_batch testdata/quant_k

Per case directory quant_<d>x<n>/ (n is a multiple of 32, the block size):

  w.npy   [d, n] f32         x.npy  [n] f32
  w_q8_0.bin, w_q4_0.bin, w_q5_0.bin   the weight rows quantized, raw bytes
  x_q8_0.bin                 x quantized to Q8_0 (what a kernel using integer
                             dot products does to the activation)
  out_q8_0_a.npy             dequant(W_q8_0) @ x            (float64 → f32)
  out_q8_0_b.npy             dequant(W_q8_0) @ dequant(x_q8_0)
  out_q4_0_a.npy, out_q4_0_b.npy   the same for Q4_0 (and _q5_0_ for Q5_0)

"a" is the reference for kernels that dequantize weights and multiply by the
f32 activation; "b" for kernels that quantize the activation and use int8
dot products (sdot / smmla).

Batched cases (prefill: t tokens at once) go to the second directory, one
quant_<d>x<n>_t<t>/ each, with odd row and token counts for the tails of
the tiles (full tiles of 4 token pairs, then 1, 2 or 3 pairs):

  w.npy  [d, n] f32          x.npy  [t, n] f32 (each row quantized on its own)
  out_q8_0_b.npy, out_q4_0_b.npy, out_q5_0_b.npy
                             [t, d]: dequant(X_q8_0) @ dequant(W)ᵀ
  w_q8_0.bin, w_q4_0.bin, w_q5_0.bin   the weight rows quantized, raw bytes
  out_q8_0_a.npy, …          [t, d]: X @ dequant(W)ᵀ (f32 activations, as
                             the GPU kernels use)

K-quant cases go to the third directory, one quant_k_<d>x<n>_t<t>/ each (n
a multiple of 256, the super-block size). The `gguf` package can decode
Q4_K and Q6_K but not encode them, so the weights are random block bytes
(every bit pattern is a valid block once the f16 scales are finite), which
exercises every bit of the layout; the reference is gguf's dequantizer:

  w_q4_k.bin, w_q6_k.bin     [d] rows of raw blocks
  x.npy [t, n] f32           x_q8_0.bin  x quantized to Q8_0, row by row
  deq_q4_k.npy, deq_q6_k.npy [d, n] gguf's dequantized weights
  out_q4_k_b.npy, out_q6_k_b.npy   [t, d]: dequant(X_q8_0) @ dequant(W)ᵀ
  out_q4_k_a.npy, out_q6_k_a.npy   [t, d]: X @ dequant(W)ᵀ (f32 activations)
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
        for t, name in [(T.Q8_0, "q8_0"), (T.Q4_0, "q4_0"), (T.Q5_0, "q5_0")]:
            wq = quants.quantize(w, t)
            (out / f"w_{name}.bin").write_bytes(wq.tobytes())
            wd = quants.dequantize(wq, t).astype(np.float64)
            np.save(out / f"out_{name}_a.npy", (wd @ x.astype(np.float64)).astype(np.float32))
            np.save(out / f"out_{name}_b.npy", (wd @ xd).astype(np.float32))
    print(f"wrote {len(shapes)} cases to {root}")
    if len(sys.argv) > 2:
        batch(Path(sys.argv[2]))
    if len(sys.argv) > 3:
        kquants(Path(sys.argv[3]))


def random_blocks(rng, typ, d, n):
    """d rows of n/256 random blocks of `typ` with scales of a realistic
    size: f16 d (and dmin) in 1e-4..1e-2, every other byte random."""
    size = {T.Q4_K: 144, T.Q6_K: 210}[typ]
    b = rng.integers(0, 256, size=(d * n // 256, size), dtype=np.uint8)
    f16 = lambda k: (rng.uniform(1e-4, 1e-2, size=(b.shape[0], k)).astype(np.float16).view(np.uint8))
    if typ == T.Q4_K:
        b[:, 0:4] = f16(2)  # d, dmin
    else:
        b[:, 208:210] = f16(1)  # d (last)
    return b.reshape(d, -1)


def kquants(root):
    rng = np.random.default_rng(608)
    cases = [(1, 256, 1), (3, 512, 2), (4, 256, 5), (5, 768, 4), (17, 1536, 3), (18, 2560, 8), (64, 1536, 7), (6, 512, 11), (5, 256, 13)]
    for d, n, t in cases:
        out = root / f"quant_k_{d}x{n}_t{t}"
        out.mkdir(parents=True, exist_ok=True)
        x = rng.standard_normal((t, n)).astype(np.float32)
        np.save(out / "x.npy", x)
        xq = quants.quantize(x, T.Q8_0)
        (out / "x_q8_0.bin").write_bytes(xq.tobytes())
        xd = quants.dequantize(xq, T.Q8_0).astype(np.float64)
        for typ, name in [(T.Q4_K, "q4_k"), (T.Q6_K, "q6_k")]:
            wq = random_blocks(rng, typ, d, n)
            (out / f"w_{name}.bin").write_bytes(wq.tobytes())
            wd = quants.dequantize(wq, typ)
            np.save(out / f"deq_{name}.npy", wd.astype(np.float32))
            np.save(out / f"out_{name}_b.npy", (xd @ wd.astype(np.float64).T).astype(np.float32))
            np.save(out / f"out_{name}_a.npy", (x.astype(np.float64) @ wd.astype(np.float64).T).astype(np.float32))
    print(f"wrote {len(cases)} cases to {root}")


def batch(root):
    # Own generator, so the single-vector cases above stay byte-identical.
    rng = np.random.default_rng(607)
    cases = [(1, 32, 2), (3, 64, 3), (5, 160, 4), (6, 96, 5), (17, 576, 8), (18, 2048, 3), (130, 896, 17), (64, 576, 64), (896, 896, 7), (12, 640, 11), (9, 384, 13)]
    for d, n, t in cases:
        w = (rng.standard_normal((d, n)) * 0.05).astype(np.float32)
        w[0, :32] = 0.0
        x = rng.standard_normal((t, n)).astype(np.float32)
        out = root / f"quant_{d}x{n}_t{t}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "w.npy", w)
        np.save(out / "x.npy", x)
        xd = quants.dequantize(quants.quantize(x, T.Q8_0), T.Q8_0).astype(np.float64)
        for typ, name in [(T.Q8_0, "q8_0"), (T.Q4_0, "q4_0"), (T.Q5_0, "q5_0")]:
            wq = quants.quantize(w, typ)
            wd = quants.dequantize(wq, typ).astype(np.float64)
            np.save(out / f"out_{name}_b.npy", (xd @ wd.T).astype(np.float32))
            (out / f"w_{name}.bin").write_bytes(wq.tobytes())
            np.save(out / f"out_{name}_a.npy", (x.astype(np.float64) @ wd.T).astype(np.float32))
    print(f"wrote {len(cases)} cases to {root}")


if __name__ == "__main__":
    main()
