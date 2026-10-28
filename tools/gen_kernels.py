"""Golden inputs and outputs for every kernel, computed with numpy.

    tools/.venv/bin/python -I tools/gen_kernels.py testdata/kernels

One directory per case, each array a .npy file (float32 inputs, outputs
computed in float64 from those exact float32 inputs, then stored as float32):

  matmul_<d>x<n>/        w [d, n], x [n], out [d]               out = W @ x
  gemm_<t>x<d>x<n>/      w [d, n], x [t, n], out [t, d]         out = X @ W.T
  rmsnorm_<n>/           x, weight, out                         eps = 1e-5
  softmax_<n>/           x, out
  swiglu_<n>/            gate, up, out                          silu(gate) * up
  rope_<heads>x<hs>_p<pos>/  x, out (adjacent pairs, theta 1e4)
  rope_halves_<heads>x<hs>_p<pos>/  x, out (rotate_half layout, theta 1e4)
  attention_<n>x<hs>/    q [hs], k [n, hs], v [n, hs], out [hs]
                         out = softmax(k @ q / sqrt(hs)) @ v   (one head, n positions)

Sizes include 1, values just around multiples of 4/8/16/32 (NEON width and
unroll factors: tail bugs live there), and the real stories15M / SmolLM2 /
Qwen2.5-0.5B layer shapes.
"""

import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(2026)


def f32(*shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def save(root, name, **arrays):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(v, dtype=np.float32))


def rope_ref(x, pos, hs, theta, halves):
    x = x.astype(np.float64).reshape(-1, hs)
    out = np.empty_like(x)
    i = np.arange(0, hs, 2)
    freq = 1.0 / theta ** (i / hs)
    ang = pos * freq
    c, s = np.cos(ang), np.sin(ang)
    if halves:
        a, b = x[:, : hs // 2], x[:, hs // 2 :]
        out[:, : hs // 2] = a * c - b * s
        out[:, hs // 2 :] = a * s + b * c
    else:
        a, b = x[:, 0::2], x[:, 1::2]
        out[:, 0::2] = a * c - b * s
        out[:, 1::2] = a * s + b * c
    return out.reshape(-1)


def main():
    root = Path(sys.argv[1])
    odd = [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 129]
    # (d, n): output rows x input length. Odd sizes, then real layer shapes:
    # stories15M (288, 768), SmolLM2-135M (576, 1536), Qwen2.5-0.5B (896, 4864).
    shapes = [(d, n) for d in (1, 3, 8, 17) for n in odd] + [
        (64, 64), (288, 288), (768, 288), (288, 768), (576, 576), (1536, 576),
        (576, 1536), (896, 896), (128, 896), (4864, 896), (896, 4864), (1000, 17), (17, 1000),
    ]
    for d, n in shapes:
        w, x = f32(d, n), f32(n)
        save(root, f"matmul_{d}x{n}", w=w, x=x, out=w.astype(np.float64) @ x.astype(np.float64))
    for t, d, n in [(1, 8, 8), (2, 3, 5), (4, 16, 16), (5, 17, 33), (7, 64, 100), (16, 288, 288),
                    (13, 768, 288), (33, 288, 768), (64, 576, 576), (8, 4864, 896), (3, 896, 4864)]:
        w, x = f32(d, n), f32(t, n)
        save(root, f"gemm_{t}x{d}x{n}", w=w, x=x, out=x.astype(np.float64) @ w.astype(np.float64).T)
    for n in odd + [288, 576, 896, 4864]:
        x, wt = f32(n, scale=3.0), f32(n)
        x64 = x.astype(np.float64)
        ref = wt * (x64 / np.sqrt(np.mean(x64 * x64) + 1e-5))
        save(root, f"rmsnorm_{n}", x=x, weight=wt, out=ref)
        z = f32(n, scale=4.0)
        e = np.exp(z.astype(np.float64) - z.max())
        save(root, f"softmax_{n}", x=z, out=e / e.sum())
        g, u = f32(n, scale=3.0), f32(n)
        g64 = g.astype(np.float64)
        save(root, f"swiglu_{n}", gate=g, up=u, out=g64 / (1.0 + np.exp(-g64)) * u)
    for heads, hs in [(1, 2), (2, 8), (6, 48), (9, 64), (14, 64)]:
        for pos in (0, 1, 7, 255, 4000):
            x = f32(heads * hs)
            save(root, f"rope_{heads}x{hs}_p{pos}", x=x, out=rope_ref(x, pos, hs, 10000.0, False))
            save(root, f"rope_halves_{heads}x{hs}_p{pos}", x=x, out=rope_ref(x, pos, hs, 10000.0, True))
    for hs in (2, 5, 8, 17, 64, 128, 48):  # 48: stories15M (appended last, so earlier cases keep their data)
        for n in (1, 2, 7, 64, 257, 1024):
            q, k, v = f32(hs), f32(n, hs), f32(n, hs)
            sc = k.astype(np.float64) @ q.astype(np.float64) / np.sqrt(hs)
            e = np.exp(sc - sc.max())
            save(root, f"attention_{n}x{hs}", q=q, k=k, v=v, out=(e / e.sum()) @ v.astype(np.float64))
    print(f"wrote {len(list(root.iterdir()))} cases to {root}")


if __name__ == "__main__":
    main()
