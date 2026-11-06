"""Golden inputs and outputs for every kernel, computed with numpy.

    tools/.venv/bin/python -I tools/gen_kernels.py testdata/kernels

One directory per case, each array a .npy file (float32 inputs, outputs
computed in float64 from those exact float32 inputs, then stored as float32):

  matmul_<d>x<n>/        w [d, n], x [n], out [d]               out = W @ x
  gemm_<t>x<d>x<n>/      w [d, n], x [t, n], out [t, d]         out = X @ W.T
  rmsnorm_<n>/           x, weight, out                         eps = 1e-5
  gelu_<n>/              gate, up, out                          gelu_tanh(gate) * up (Gemma)
  softcap_<n>/           x, out                                 30 tanh(x / 30)
  plainnorm_<h>x<hs>/    x, out                                 RMSNorm per head, no gain, eps 1e-6
  softmax_<n>/           x, out
  swiglu_<n>/            gate, up, out                          silu(gate) * up
  rope_<heads>x<hs>_p<pos>/  x, out (adjacent pairs, theta 1e4)
  rope_halves_<heads>x<hs>_p<pos>/  x, out (rotate_half layout, theta 1e4)
  attention_<n>x<hs>/    q [hs], k [n, hs], v [n, hs], out [hs]
                         out = softmax(k @ q / sqrt(hs)) @ v   (one head, n positions)
  attention_causal_<nh>x<nkv>x<hs>_p<pos0>_t<nt>/
                         q [nt, nh, hs], k [pos0 + nt, nkv, hs], v [same], out [nt, nh, hs]
                         token i is at position pos0 + i and sees positions 0..=pos0 + i;
                         query head h reads the key/value head h // (nh // nkv)

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


def attention_causal_ref(q, k, v, pos0):
    """Causal grouped-query attention over a batch, in float64 from the
    float32 inputs: token i (at position pos0 + i) sees positions
    0 ..= pos0 + i, and query head h reads key/value head h // (nh // nkv)."""
    nt, nh, hs = q.shape
    g = nh // k.shape[1]
    q64, k64, v64 = (a.astype(np.float64) for a in (q, k, v))
    out = np.empty((nt, nh, hs), np.float64)
    for i in range(nt):
        for h in range(nh):
            kvh = h // g
            s = k64[: pos0 + i + 1, kvh] @ q64[i, h] / np.sqrt(hs)
            e = np.exp(s - s.max())
            out[i, h] = (e / e.sum()) @ v64[: pos0 + i + 1, kvh]
    return out


def rope_partial_ref(x, pos, hs, theta, pairs):
    """LFM3's zero-tail partial RoPE: the first `pairs` pairs use the
    frequencies of a full `hs`-wide head; the rest do not rotate."""
    x = x.astype(np.float64).reshape(-1, hs)
    out = x.copy()
    half = hs // 2
    i = np.arange(0, 2 * pairs, 2)
    ang = pos / theta ** (i / hs)
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[:, :pairs], x[:, half : half + pairs]
    out[:, :pairs] = a * c - b * s
    out[:, half : half + pairs] = a * s + b * c
    return out.reshape(-1)


def short_conv_ref(u, weight, state, width):
    u = u.astype(np.float64)
    w = weight.astype(np.float64)
    st = state.astype(np.float64).copy()
    out = np.empty_like(u)
    for t in range(u.shape[0]):
        for ch in range(u.shape[1]):
            acc = 0.0
            for k in range(width - 1):
                acc += st[k, ch] * w[ch, k]
            acc += u[t, ch] * w[ch, width - 1]
            out[t, ch] = acc
        st[:-1] = st[1:]
        st[-1] = u[t]
    return out, st


def rmsnorm_heads_ref(x, weight, heads, hs):
    x = x.astype(np.float64).reshape(heads, hs)
    w = weight.astype(np.float64)
    out = w * (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5))
    return out.reshape(-1)


def gate_ref(x, gates, hs):
    x = x.astype(np.float64).reshape(-1, hs)
    g = 1.0 / (1.0 + np.exp(-gates.astype(np.float64)))
    return (x * g[:, None]).reshape(-1)


def attention_window_ref(q, k, v, pos, window):
    q64, k64, v64 = (a.astype(np.float64) for a in (q, k, v))
    start = max(pos + 1 - window, 0)
    s = k64[start : pos + 1] @ q64 / np.sqrt(q.shape[-1])
    e = np.exp(s - s.max())
    return (e / e.sum()) @ v64[start : pos + 1]


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
    # Causal, grouped-query attention over a batch (the GPU prompt shaders).
    # pos0 crosses the 16-position sub-tiles, the 32-token workgroups and
    # the 64-position chunks; the deep positions run one size each, since
    # the loop is the same once its tiles are full.
    for nh, nkv, hs in [(14, 2, 64), (9, 3, 64), (4, 4, 17)]:
        for pos0, nts in [(0, (1, 31, 32, 33, 128)), (1, (1, 31, 32, 33, 128)),
                          (63, (1, 31, 32, 33, 128)), (64, (1, 31, 32, 33, 128)),
                          (1000, (33,)), (4000, (33,))]:
            for nt in nts:
                q, k, v = f32(nt, nh, hs), f32(pos0 + nt, nkv, hs), f32(pos0 + nt, nkv, hs)
                save(root, f"attention_causal_{nh}x{nkv}x{hs}_p{pos0}_t{nt}", q=q, k=k, v=v, out=attention_causal_ref(q, k, v, pos0))
    # LFM3 kernels: short conv, per-head RMSNorm, per-head gate, zero-tail
    # RoPE and sliding-window attention.
    for dim in (7, 17, 64, 128, 2048):
        width = 7
        for t in (1, 3, 7, 16):
            u, w, st = f32(t, dim), f32(dim, width), f32(6, dim)
            out, end = short_conv_ref(u, w, st, width)
            save(root, f"shortconv_d{dim}_w{width}_t{t}", u=u, weight=w, state=st, out=out, state_out=end)
    for heads, hs in ((2, 7), (3, 64), (4, 128), (16, 128)):
        x, wt = f32(heads * hs, scale=3.0), f32(hs)
        save(root, f"headnorm_{heads}x{hs}", x=x, weight=wt, out=rmsnorm_heads_ref(x, wt, heads, hs))
        g = f32(heads, scale=2.0)
        save(root, f"gate_{heads}x{hs}", x=x, gates=g, out=gate_ref(x, g, hs))
    for heads, hs, pairs in ((2, 128, 32), (4, 128, 32), (2, 64, 32), (3, 128, 48)):
        for pos in (0, 1, 255, 4000):
            x = f32(heads * hs)
            save(root, f"partial_rope_{heads}x{hs}_p{pos}_r{pairs}", x=x, out=rope_partial_ref(x, pos, hs, 1e6, pairs))
    for hs in (64, 128):
        for n, window, pos in ((300, 256, 254), (300, 256, 255), (300, 256, 256),
                               (300, 256, 257), (1024, 256, 1023), (1024, 1024, 1023)):
            q, k, v = f32(hs), f32(n, hs), f32(n, hs)
            save(root, f"window_attention_{hs}_n{n}_p{pos}_w{window}", q=q, k=k, v=v, window=np.array(window, np.float32),
                 out=attention_window_ref(q, k, v, pos, window))
    for nh, nkv, hs in ((4, 4, 128), (4, 2, 128)):
        for pos0, nts in ((0, (1, 33, 128)), (64, (1, 33, 128)), (1000, (33,))):
            for nt in nts:
                q, k, v = f32(nt, nh, hs), f32(pos0 + nt, nkv, hs), f32(pos0 + nt, nkv, hs)
                save(root, f"attention_causal_{nh}x{nkv}x{hs}_p{pos0}_t{nt}", q=q, k=k, v=v, out=attention_causal_ref(q, k, v, pos0))
    # Gemma 4 (appended last, so earlier cases keep their data): GELU (tanh
    # form) × up, RMSNorm without a gain (eps 1e-6), the logit soft-cap, and
    # attention with head sizes 256 / 512 and one K/V head for eight queries.
    for n in odd + [256, 6144]:
        g, u = f32(n, scale=3.0), f32(n)
        g64 = g.astype(np.float64)
        gelu = 0.5 * g64 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (g64 + 0.044715 * g64**3)))
        save(root, f"gelu_{n}", gate=g, up=u, out=gelu * u)
        x = f32(n, scale=40.0)
        save(root, f"softcap_{n}", x=x, out=30.0 * np.tanh(x.astype(np.float64) / 30.0))
    for heads, hs in ((3, 17), (1, 256), (1, 512)):
        x = f32(heads * hs, scale=3.0).astype(np.float64).reshape(heads, hs)
        out = x / np.sqrt((x * x).mean(axis=1, keepdims=True) + 1e-6)
        save(root, f"plainnorm_{heads}x{hs}", x=x.reshape(-1), out=out.reshape(-1))
    for hs in (256, 512):
        for n in (1, 7, 600):
            q, k, v = f32(hs), f32(n, hs), f32(n, hs)
            s_ = k.astype(np.float64) @ q.astype(np.float64) / np.sqrt(hs)
            e = np.exp(s_ - s_.max())
            save(root, f"attention_{n}x{hs}", q=q, k=k, v=v, out=(e / e.sum()) @ v.astype(np.float64))
        for pos0, nt in ((0, 1), (0, 33), (100, 7)):
            q, k, v = f32(nt, 8, hs), f32(pos0 + nt, 1, hs), f32(pos0 + nt, 1, hs)
            save(root, f"attention_causal_8x1x{hs}_p{pos0}_t{nt}", q=q, k=k, v=v, out=attention_causal_ref(q, k, v, pos0))
    for n, window, pos in ((600, 512, 510), (600, 512, 511), (600, 512, 512), (600, 512, 599)):
        q, k, v = f32(256), f32(n, 256), f32(n, 256)
        save(root, f"window_attention_256_n{n}_p{pos}_w{window}", q=q, k=k, v=v, window=np.array(window, np.float32),
             out=attention_window_ref(q, k, v, pos, window))
    # Gemma 4 on the GPU (appended last): the per-head step after q/k/v
    # (RMSNorm with gain for q and k, without for v; RoPE on the first
    # `pairs` pairs, element i with i + hs/2; q × √hs), for one token of
    # 8 query heads and 1 K/V head, the (sin, cos) table from numpy; and the
    # post-norm-and-add, x = (x + w ⊙ rmsnorm(t)) · scale, for 3 tokens.
    for hs, pairs, theta, pos in ((256, 128, 1e4, 37), (512, 64, 1e6, 600)):
        nq = 8
        q, k, v = f32(nq, hs, scale=2.0), f32(1, hs, scale=2.0), f32(1, hs, scale=2.0)
        qn, kn = f32(hs, scale=0.5) + 1.0, f32(hs, scale=0.5) + 1.0
        ang = pos * theta ** (-2.0 * np.arange(hs // 2) / hs)
        rot = np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)

        def norm(x, g):
            x = x.astype(np.float64)
            y = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6)
            return y if g is None else y * g.astype(np.float64)

        def rope(x):
            x = x.copy()
            s_, c_ = rot[:pairs, 0].astype(np.float64), rot[:pairs, 1].astype(np.float64)
            a, b = x[..., :pairs].copy(), x[..., hs // 2:hs // 2 + pairs].copy()
            x[..., :pairs] = a * c_ - b * s_
            x[..., hs // 2:hs // 2 + pairs] = a * s_ + b * c_
            return x

        save(root, f"qknorm_{hs}_{pairs}", q=q, k=k, v=v, qn=qn, kn=kn, rot=rot,
             q_out=rope(norm(q, qn)) * np.sqrt(hs), k_out=rope(norm(k, kn)), v_out=norm(v, None))
    for n in (1536, 300):
        x, t, w = f32(3, n), f32(3, n, scale=4.0), f32(n, scale=0.5) + 1.0
        t64 = t.astype(np.float64)
        y = t64 / np.sqrt((t64 * t64).mean(axis=1, keepdims=True) + 1e-6) * w.astype(np.float64)
        save(root, f"normadd_{n}", x=x, t=t, w=w, scale=np.array([0.7], np.float32), out=(x.astype(np.float64) + y) * 0.7)
    print(f"wrote {len(list(root.iterdir()))} cases to {root}")


if __name__ == "__main__":
    main()
