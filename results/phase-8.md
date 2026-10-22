# Phase 8 — C ABI and Android app

## What exists

- `src/ffi.rs` + `include/aether.h`: `aether_open` / `aether_generate`
  (streaming callback) / `aether_close` / `aether_last_error`; every entry
  point under `catch_unwind`, errors as codes plus a message.
  `ffi::tests::c_abi_round_trip` drives it exactly as C would and reproduces
  llama2.c's greedy stories15M text; `examples/c/smoke.c` is the same from C.
- `android/`: a one-screen Kotlin app (no AndroidX) over a small C JNI shim
  (`android/app/src/main/cpp/aether_jni.c`); `scripts/android-native.sh`
  builds `libaether.so` + `libaether_jni.so` into `jniLibs/arm64-v8a`.
  Build and model setup: `android/README.md`.

## Verification (2026-10-22, iQOO 13)

Debug APK installed with `./gradlew installDebug`; Qwen2.5-0.5B-Instruct
Q4_0 GGUF copied into the app's internal storage (`run-as`); prompt "Why is
the sky blue?" with the chat box ticked, cores `2-7`, temperature 0.7,
top-p 0.9. The reply streamed in token by token and was a coherent ChatML
answer ending at the model's end-of-turn token.

Speeds as the app reports them, against the CLI right after with the same
prompt, cores and sampling (`aether MODEL --chat -i "Why is the sky blue?"
-c 2-7 -t 0.7 -n 512`), same session:

| Run | Prefill (35 tokens) | Decode |
| --- | --- | --- |
| app, 1st generate after load | 262 ms (133.5 tok/s) | 243 tokens at 94.6 tok/s |
| app, 2nd | 205 ms (170.9 tok/s) | 231 tokens at 100.4 tok/s |
| app, 3rd | 200 ms (175.2 tok/s) | 230 tokens at 93.6 tok/s |
| CLI, seed 1 | 216 ms (162.4 tok/s) | 372 tokens at 98.0 tok/s |
| CLI, seed 2 | 215 ms (162.9 tok/s) | 135 tokens at 103.9 tok/s |

Once warm the app matches the CLI within run-to-run spread. The first
generate is slower because the model is memory-mapped: its pages are read
from storage the first time each is touched, which happens during that
first prefill.

Conditions: on battery (54 %), and another app was playing a
picture-in-picture video during these runs, so absolute speeds are below
`results/phase-7.md`'s; the app and the CLI ran under the same load.
