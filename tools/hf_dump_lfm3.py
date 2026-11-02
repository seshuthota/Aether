"""HF LFM3 reference logits and layer taps for A4.

    tools/.venv/bin/python -I tools/hf_dump_lfm3.py \
        hostdata/LFM3-1.7B hostdata/lfm3/ref.safetensors

Runs the pinned repository code in float32 on the CPU (Triton falls back to
the PyTorch conv). Saves a short prompt's full logits and every layer's
residual output, a long prompt's logits at positions around the sliding
window edge, and a greedy continuation.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

NEW = 16


def main() -> None:
    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, trust_remote_code=True, local_files_only=True, attn_implementation="eager"
    ).eval()

    short_text = "The capital of France is Paris. The capital of Japan is Tokyo. Once upon a time, in a small village by the sea, there lived"
    short_ids = tok.encode(short_text, add_special_tokens=False)
    long_text = Path("testdata/wikitext/wikitext-2-raw/wiki.test.raw").read_text(encoding="utf-8")
    long_ids = tok.encode(long_text, add_special_tokens=False)[:320]
    assert len(long_ids) > 300, len(long_ids)

    acts = {}
    taps = {}

    def hook(l):
        def f(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            taps[f"L{l}.resid_ffn"] = t[0].detach().float().numpy().copy()
        return f

    handles = [layer.register_forward_hook(hook(l)) for l, layer in enumerate(model.model.layers)]

    with torch.no_grad():
        out_short = model(torch.tensor([short_ids]), use_cache=False)
        short_logits = out_short.logits[0].float().numpy()
    acts.update(taps)
    for h in handles:
        h.remove()

    with torch.no_grad():
        out_long = model(torch.tensor([long_ids]), use_cache=False)
        positions = np.array([0, 1, 254, 255, 256, 257, 300, 318, 319], dtype=np.int64)
        long_logits = out_long.logits[0, positions].float().numpy()
        greedy = model.generate(
            torch.tensor([short_ids]), max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False,
            repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
        )[0].tolist()

    acts["short_ids"] = np.array(short_ids, dtype=np.float32)
    acts["short_logits"] = short_logits
    acts["long_ids"] = np.array(long_ids, dtype=np.float32)
    acts["long_positions"] = positions.astype(np.float32)
    acts["long_logits"] = long_logits
    acts["greedy_ids"] = np.array(greedy, dtype=np.float32)
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in acts.items()}, str(out))
    print(f"short {len(short_ids)} tokens, long {len(long_ids)}, greedy {NEW}: {tok.decode(greedy[len(short_ids):])!r}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
