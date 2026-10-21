"""Golden data from Hugging Face transformers / tokenizers for a Llama-style
checkpoint (SmolLM2), read back by the engine's own safetensors parser.

    tools/.venv/bin/python -I tools/hf_dump.py testdata/models/SmolLM2-135M testdata/smollm2-135m

Writes to the output directory:

  acts.safetensors   float32 activations for every prompt position, recorded
                     with forward hooks, named like the engine's observer:
                       embed [T, dim], final_norm [T, dim], logits [T, vocab]
                       L{l}.attn_norm, L{l}.v, L{l}.attn_out, L{l}.resid_attn,
                       L{l}.ffn_norm, L{l}.ffn_out, L{l}.resid_ffn  [T, width]
                     prompt_ids [T], greedy_ids [T + NEW] (as float32, exact)
  corpus.txt         tokenizer test strings, 0x1E separated
  encoded.txt        HF `tokenizers` ids for each string, one line each
  decoded.txt        HF decode of those ids, 0x1E separated

The model runs in float32 (bf16 weights widened exactly, as the engine does)
with eager attention, so the only differences left are summation order.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

PROMPT = "Once upon a time, in a small village by the sea, there lived"
NEW = 48

sys.path.insert(0, str(Path(__file__).parent))
from gen_tokenizer_corpus import CASES, PIECES  # noqa: E402


def main():
    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    ids = tok.encode(PROMPT).ids
    acts = {}
    # Generate before attaching hooks: otherwise every generation step would
    # overwrite the prompt's activations.
    with torch.no_grad():
        greedy = model.generate(torch.tensor([ids]), max_new_tokens=NEW, do_sample=False, min_new_tokens=NEW)[0].tolist()

    def save(name):
        def hook(_mod, inputs, output):
            if name.endswith("resid_attn"):  # input of post_attention_layernorm
                t = inputs[0]
            else:
                t = output[0] if isinstance(output, tuple) else output
            acts[name] = t[0].detach().float().numpy().copy()

        return hook

    m = model.model
    m.embed_tokens.register_forward_hook(save("embed"))
    for l, layer in enumerate(m.layers):
        layer.input_layernorm.register_forward_hook(save(f"L{l}.attn_norm"))
        layer.self_attn.v_proj.register_forward_hook(save(f"L{l}.v"))
        layer.self_attn.register_forward_hook(save(f"L{l}.attn_out"))
        layer.post_attention_layernorm.register_forward_hook(save(f"L{l}.resid_attn"))
        layer.post_attention_layernorm.register_forward_hook(save(f"L{l}.ffn_norm"))
        layer.mlp.register_forward_hook(save(f"L{l}.ffn_out"))
        layer.register_forward_hook(save(f"L{l}.resid_ffn"))
    m.norm.register_forward_hook(save("final_norm"))

    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].float().numpy()
    acts["logits"] = logits
    acts["prompt_ids"] = np.array(ids, dtype=np.float32)
    acts["greedy_ids"] = np.array(greedy, dtype=np.float32)
    save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in acts.items()}, str(out / "acts.safetensors"))
    print(f"prompt {len(ids)} tokens, greedy continuation: {tok.decode(greedy[len(ids):])!r}")

    # Tokenizer corpus: the stories15M cases, code, chat markup with special
    # tokens, and random mixes.
    rng = np.random.default_rng(5)
    extra = [
        "def f(x):\n    return x**2  # square\n\n\tif x  >= 10:",
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
        "<|endoftext|>Hello<|endoftext|>",
        "   leading and trailing spaces   ",
        "numbers 0 12 345 6789 3.14159 1,000,000 ½ ² Ⅻ ٣",
        "MiXeD CaSe, ALLCAPS, and contractions: I'm, you're, they've, we'll, he'd, it's, don't",
        "URL https://example.com/a?b=c&d=e#f and email a.b@c.de",
        "Tabs\tand\r\nWindows newlines line sep",
        "Ελληνικά, русский, עברית, العربية, हिन्दी, ไทย, 한국어",
        "combining: é ä ñ; zwj 👨‍💻; flags 🇮🇳🇯🇵",
    ]
    cases = list(CASES) + extra
    for _ in range(400):
        cases.append("".join(rng.choice(PIECES + [" ", "\n", "x", "Ω", "1", "<|im_end|>"]) for _ in range(rng.integers(1, 25))))
    enc = [tok.encode(s).ids for s in cases]
    (out / "corpus.txt").write_text("\x1e".join(cases), encoding="utf-8")
    (out / "encoded.txt").write_text("\n".join(" ".join(map(str, e)) for e in enc) + "\n")
    (out / "decoded.txt").write_text("\x1e".join(tok.decode(e, skip_special_tokens=False) for e in enc), encoding="utf-8")
    print(f"{len(cases)} tokenizer strings")


if __name__ == "__main__":
    main()
