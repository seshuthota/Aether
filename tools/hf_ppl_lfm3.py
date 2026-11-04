"""HF LFM3 perplexity of one long wikitext chunk, as llama-perplexity scores it.

    tools/.venv/bin/python -I tools/hf_ppl_lfm3.py hostdata/LFM3-1.7B 2048

Runs the pinned repository code in float32 on the CPU over the first CTX
tokens of wikitext-2's test file (no BOS, as LFM3's tokenizer adds none)
and scores the second half: the logits at positions CTX/2 .. CTX-2 predict
tokens CTX/2+1 .. CTX-1, which is llama-perplexity's `-c CTX --chunks 1`.
Past 512 positions the sliding layers and the full layers see different
prefixes, so this is the reference for the context-2048 runs.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_dir, ctx = Path(sys.argv[1]), int(sys.argv[2])
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, trust_remote_code=True, local_files_only=True, attn_implementation="eager"
    ).eval()
    text = Path("testdata/wikitext/wikitext-2-raw/wiki.test.raw").read_text(encoding="utf-8")
    ids = tok.encode(text, add_special_tokens=False)[:ctx]
    with torch.no_grad():
        logits = model(torch.tensor([ids]), use_cache=False).logits[0].double()
    first = ctx // 2
    logp = torch.log_softmax(logits[first : ctx - 1], dim=-1)
    nll = -logp[torch.arange(ctx - 1 - first), torch.tensor(ids[first + 1 : ctx])]
    ppl = float(torch.exp(nll.mean()))
    print(f"HF f32, ctx {ctx}: PPL {ppl:.4f} over {len(nll)} tokens")


if __name__ == "__main__":
    main()
