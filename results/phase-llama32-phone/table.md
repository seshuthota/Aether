| model | backend | threads | test | llama.cpp t/s | aether t/s | aether / llama.cpp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct-Q8_0.gguf | CPU | 6 | pp64 | 353.3 (240.0–392.6) | 343.7 (283.2–350.5) | 0.97× |
| Llama-3.2-1B-Instruct-Q8_0.gguf | CPU | 6 | pp256 | 285.8 (229.0–342.6) | 328.2 (323.4–330.0) | 1.15× |
| Llama-3.2-1B-Instruct-Q8_0.gguf | CPU | 6 | pp512 | 309.3 (211.8–358.7) | 285.1 (214.2–291.1) | 0.92× |
| Llama-3.2-1B-Instruct-Q8_0.gguf | CPU | 6 | tg128 | 45.6 (42.3–46.8) | 44.6 (41.8–46.5) | 0.98× |
| Llama-3.2-1B-Instruct-Q8_0.gguf | CPU | 6 | pp64 @ d512 | 212.6 (198.8–226.1) | 233.2 (215.0–235.3) | 1.10× |
| Llama-3.2-1B-Instruct-Q8_0.gguf | CPU | 6 | pp256 @ d512 | 215.6 (198.1–240.4) | 222.6 (216.8–227.2) | 1.03× |
| Llama-3.2-1B-Instruct-Q8_0.gguf | CPU | 6 | pp512 @ d512 | 202.7 (162.0–213.2) | 209.5 (202.3–211.5) | 1.03× |
| Llama-3.2-1B-Instruct-Q8_0.gguf | CPU | 6 | tg128 @ d512 | 37.7 (32.4–38.0) | 42.4 (40.5–42.9) | 1.12× |

Median tokens per second (min–max over every repetition of every round).
