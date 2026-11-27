# Supra2.5 on the phone: end to end (not a benchmark)

Plan: `aether-infer/plans/supra.md` (S4). 2026-11-27, iQOO 13, wireless adb,
**charging** (AC powered, 38 %), zone0 48.0-48.8 °C, clock caps cpu0 2.00 /
cpu7 2.25 GHz, screen awake. aether `100b525` (branch `supra`). Files: the
fixed GGUFs (`tools/supra_gguf.py`), sha256 checked on the phone.

`aether run -m F -cnv -p 'What is the capital of France?' -n 80 --temp 0 --single-turn --cores 2-7`:

| File | Reply | Prompt | Decode | Peak RSS |
| --- | --- | --- | --- | --- |
| Q8_0 (71.3 MB of weights) | "The capital of France is **Paris**. It is the country's largest city and its political, cultural, and economic center, located in the north-central part of France along the Seine River." (as on the PC) | 16 tok, 11.3 ms (1421 tok/s) | 45 tok, 96.5 ms (466 tok/s) | 121 MB |
| Q4_K_M = Q5_0 mix (53.3 MB) | the same, a straight apostrophe | 16 tok, 11.4 ms (1404 tok/s) | 44 tok, 84.9 ms (518 tok/s) | 103 MB |
| Q4_0 (37.8 MB) | the first sentence and "political, cultural, and economic center." | 16 tok, 17.0 ms (941 tok/s) | 29 tok, 46.0 ms (630 tok/s) | 88 MB |

One run each, under 0.1 s of decode on a hot phone: the speeds are a sign
of scale, not a measurement (that is S5, cooled, against llama-bench).

AAR: `dev.aether.SupraChatTest#chatsTwoTurns` (Q4_K_M, cores 2-7, context
512) passes: `INSTRUMENTATION_STATUS_CODE: 0`, 0.52 s for the test.
