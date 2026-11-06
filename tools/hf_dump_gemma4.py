"""HF Gemma 4 E2B reference logits and layer taps (plans/gemma4.md G0/G4).

    tools/.venv/bin/python -I tools/hf_dump_gemma4.py \
        hostdata/gemma-4-E2B-it hostdata/gemma4/ref.safetensors

Runs transformers' own `Gemma4TextModel` in float32 on the CPU (eager
attention). The checkpoint is bf16, so widening it to f32 is exact. To fit
the PC's memory (the whole model in f32 is ~20 GB), the 262144 × 8960
per-layer table is not loaded: the model takes `inputs_embeds` and the raw
`per_layer_inputs` (the table rows × √256, as `get_per_layer_inputs`
computes them), and only the rows of the prompts' tokens are read. The
embedding is scaled by √1536 in f32, which is what the f32 model's own
`embed_tokens` does (the bf16 rounding of the scale in the modeling file's
comment applies only to a bf16 model). Logits are `30 · tanh(h·Eᵀ / 30)`
from the tied embedding, as `Gemma4ForCausalLM` computes them.

Saves, for a short prompt: every layer's output (`L{l}.layer_out`), the
projected per-layer inputs (`ple`) and all logits; for a 600-token wikitext
prompt (past the 512 sliding window): logits at positions around the window
edge and at the end; and a 16-token greedy continuation of the short prompt
(full forward passes, no cache).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
from tokenizers import Tokenizer

NEW = 16
BOS = 2
PRE = "model.language_model."


def main() -> None:
    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    short_text = "The capital of France is Paris. The capital of Japan is Tokyo. Once upon a time, in a small village by the sea, there lived"
    short_ids = [BOS] + tok.encode(short_text, add_special_tokens=False).ids
    long_text = Path("testdata/wikitext/wikitext-2-raw/wiki.test.raw").read_text(encoding="utf-8")
    long_ids = [BOS] + tok.encode(long_text[:20000], add_special_tokens=False).ids[:599]
    assert len(long_ids) == 600, len(long_ids)

    cfg = AutoConfig.from_pretrained(model_dir).text_config
    cfg._attn_implementation = "eager"
    n_layers, pd = cfg.num_hidden_layers, cfg.hidden_size_per_layer_input
    big = cfg.vocab_size_per_layer_input
    cfg.vocab_size_per_layer_input = 1  # the table is not used (see above)
    model = Gemma4TextModel(cfg).float().eval()

    # One tensor at a time into the model's own (f32) parameters, so the
    # weights are never held twice.
    f = safe_open(str(model_dir / "model.safetensors"), "pt")
    own = model.state_dict(keep_vars=True)
    loaded = set()
    for k in f.keys():
        name = k[len(PRE):]
        if k.startswith(PRE) and name in own and "embed_tokens_per_layer" not in name:
            own[name].data.copy_(f.get_tensor(k))
            loaded.add(name)
    missing = sorted(set(own) - loaded)
    assert missing == ["embed_tokens_per_layer.weight"], missing
    emb = model.embed_tokens.weight  # [vocab, 1536] f32
    table = f.get_slice(PRE + "embed_tokens_per_layer.weight")
    assert table.get_shape() == [big, n_layers * pd]
    rows = {}

    def per_layer(ids):
        for i in set(ids) - rows.keys():
            rows[i] = table[i : i + 1].float()[0]
        return (torch.stack([rows[i] for i in ids]) * pd**0.5).reshape(1, len(ids), n_layers, pd)

    taps = {}
    real_project = model.project_per_layer_inputs

    def project(e, p):
        r = real_project(e, p)
        taps["ple"] = r[0].reshape(len(e[0]), -1).numpy().copy()
        return r

    model.project_per_layer_inputs = project
    for l, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda _m, _i, o, l=l: taps.__setitem__(f"L{l}.layer_out", o[0].numpy().copy()))

    def logits(ids):
        e = emb[torch.tensor(ids)].unsqueeze(0) * torch.tensor(cfg.hidden_size**0.5, dtype=torch.float32)
        h = model(inputs_embeds=e, per_layer_inputs=per_layer(ids), use_cache=False).last_hidden_state[0]
        cap = cfg.final_logit_softcapping
        return torch.tanh(h @ emb.T / cap) * cap

    acts = {}
    short_logits = logits(short_ids).numpy()
    acts.update(taps)
    acts["short_ids"] = np.array(short_ids)
    acts["short_logits"] = short_logits

    positions = np.array([0, 1, 510, 511, 512, 513, 514, 598, 599])
    acts["long_ids"] = np.array(long_ids)
    acts["long_positions"] = positions
    acts["long_logits"] = logits(long_ids)[positions].numpy()

    ids = list(short_ids)
    for _ in range(NEW):
        ids.append(int(logits(ids)[-1].argmax()))
    acts["greedy_ids"] = np.array(ids)

    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in acts.items()}, str(out))
    print(f"short {len(short_ids)} tokens, long {len(long_ids)}, greedy {NEW}: {tok.decode(ids[len(short_ids):])!r}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
