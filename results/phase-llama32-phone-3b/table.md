| model | backend | threads | test | llama.cpp t/s | aether t/s | aether / llama.cpp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-3B-Instruct-Q8_0.gguf | CPU | 6 | pp64 | 131.5 (131.0–132.3) | 130.7 (129.9–131.5) | 0.99× |
| Llama-3.2-3B-Instruct-Q8_0.gguf | CPU | 6 | pp256 | 112.3 (105.5–123.2) | 108.3 (85.0–113.2) | 0.96× |
| Llama-3.2-3B-Instruct-Q8_0.gguf | CPU | 6 | pp512 | 101.2 (99.1–102.7) | 88.2 (86.7–94.1) | 0.87× |
| Llama-3.2-3B-Instruct-Q8_0.gguf | CPU | 6 | tg128 | 17.7 (17.2–18.0) | 17.8 (17.2–18.0) | 1.00× |
| Llama-3.2-3B-Instruct-Q8_0.gguf | CPU | 6 | pp64 @ d512 | 68.1 (61.1–81.7) | 65.1 (62.4–70.9) | 0.96× |
| Llama-3.2-3B-Instruct-Q8_0.gguf | CPU | 6 | pp256 @ d512 | 70.5 (65.9–76.3) | 59.8 (56.0–60.7) | 0.85× |
| Llama-3.2-3B-Instruct-Q8_0.gguf | CPU | 6 | pp512 @ d512 | 69.8 (65.3–73.0) | 57.2 (56.3–57.9) | 0.82× |
| Llama-3.2-3B-Instruct-Q8_0.gguf | CPU | 6 | tg128 @ d512 | 14.5 (13.5–14.7) | 15.8 (15.1–15.9) | 1.09× |

Median tokens per second (min–max over every repetition of every round).
