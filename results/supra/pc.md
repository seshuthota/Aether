# Supra2.5-Pro-Instruct on the PC: PPL and KL against llama.cpp

Plan: `aether-infer/plans/supra.md` (S2), branch `supra`. Script: `pc.sh`
(run from aether-infer); logs: `pc/` (`conditions.txt`: both commits, date).

- wikitext-2 test, `-c 512 --chunks 40`; KL against llama.cpp's logits on
  the fixed F16 (`Supra2.5-65M-fixed.F16.gguf`, `tools/supra_gguf.py`).
- PC: x86-64, llama.cpp CPU build `d02b4c0d6`, 8 threads; aether release.

| File | PPL aether | PPL llama.cpp | KLD aether | KLD llama.cpp | same top aether | same top llama.cpp |
| --- | --- | --- | --- | --- | --- | --- |
| F16 | 26.2073 | 26.2076 | 0.000001 | (base) | 99.90 % | |
| Q8_0 | 26.2376 | 26.2409 | 0.001605 | 0.001637 | 97.67 % | 97.28 % |
| Q4_K_M (Q5_0 + Q8_0) | 26.2849 | 26.2675 | 0.026444 | 0.026366 | 89.85 % | 89.75 % |
| Q4_0, Q8_0 embeddings | 28.7110 | 28.6888 | 0.125915 | 0.125649 | 79.06 % | 79.07 % |
| Q4_0 pure | 30.0196 | 30.0173 | 0.155501 | 0.155651 | 76.55 % | 76.36 % |

- The two engines agree within llama.cpp's own error bars on every file.
- The published F16 (`tokenizer.ggml.pre = default`) gives 22.12 in
  llama.cpp (`llama-published-F16.log`): not comparable, the same text cut
  into different (wrong) tokens.
- Q4_0 is a poor trade for this 65M model: +10-15 % PPL, a fifth to a
  quarter of the top tokens change. The Q5_0 mix (54 MB) is the smallest
  file close to Q8_0.
