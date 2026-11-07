| model | backend | threads | test | llama.cpp t/s | aether t/s | aether / llama.cpp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LFM3-1.7B-Q4_K_M.gguf | CPU | 6 | pp64 | 154.7 (128.3–179.9) | 120.1 (118.5–121.2) | 0.78× |
| LFM3-1.7B-Q4_K_M.gguf | CPU | 6 | pp256 | 147.1 (128.0–174.0) | 134.1 (133.4–136.6) | 0.91× |
| LFM3-1.7B-Q4_K_M.gguf | CPU | 6 | pp512 | 139.1 (126.9–152.9) | 132.6 (131.4–134.7) | 0.95× |
| LFM3-1.7B-Q4_K_M.gguf | CPU | 6 | tg128 | 41.6 (40.0–44.1) | 45.7 (45.0–47.5) | 1.10× |
| LFM3-1.7B-Q4_K_M.gguf | CPU | 6 | pp64 @ d512 | 108.4 (94.8–118.0) | 104.0 (100.3–105.2) | 0.96× |
| LFM3-1.7B-Q4_K_M.gguf | CPU | 6 | pp256 @ d512 | 118.3 (114.3–122.5) | 110.6 (108.1–113.1) | 0.93× |
| LFM3-1.7B-Q4_K_M.gguf | CPU | 6 | pp512 @ d512 | 120.7 (107.5–124.2) | 109.9 (108.7–110.8) | 0.91× |
| LFM3-1.7B-Q4_K_M.gguf | CPU | 6 | tg128 @ d512 | 36.9 (30.3–39.6) | 37.1 (34.9–37.9) | 1.01× |

Median tokens per second (min–max over every repetition of every round).
