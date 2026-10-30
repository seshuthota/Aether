# Aether Chat (Android demo)

A one-screen app that runs the engine on the phone: pick a `.gguf` model,
type a prompt, watch the reply stream in with prefill and decode speeds.

```
Kotlin (MainActivity, Aether.kt)
  │  external fun nativeGenerate(...)          JNI
  ▼
libaether_jni.so   app/src/main/cpp/aether_jni.c   byte[] ⇄ char*, callbacks
  │  aether_generate(...)                      C ABI, include/aether.h
  ▼
libaether.so       the Rust engine (src/ffi.rs)
```

## Build and install

From the repository root (needs the NDK linker set in `.cargo/config.toml`
and `sdk.dir` in `android/local.properties`):

```
scripts/android-native.sh                    # engine + JNI shim → jniLibs/
(cd android && ./gradlew installDebug)       # build the APK, install over adb
```

## Put a model on the phone

The app lists `.gguf` files in its own folders. The simplest is shared
storage, which `adb` can write directly (start the app once first so the
folder exists):

```
adb push qwen2.5-0.5b-instruct-q4_0.gguf /sdcard/Android/data/dev.aether.chat/files/
```

Reading through shared storage goes through Android's FUSE layer. To keep
the model in the app's internal storage instead (the debug build allows
`run-as`):

```
adb push model.gguf /data/local/tmp/model.gguf
adb shell "cat /data/local/tmp/model.gguf | run-as dev.aether.chat sh -c 'cat > files/model.gguf'"
```

## Settings on screen

- **cores**: CPU list the engine spreads over, one thread each. `2-7` was
  fastest for Qwen2.5-0.5B Q4_0 decode on the iQOO 13 (`results/phase-7.md`).
- **chat**: wrap the prompt in the ChatML template (instruct models).
- **CPU / GPU / DSP**: where the model runs. GPU decodes on the GPU
  (Vulkan) and runs the prompt on the CPU cores; DSP runs everything on the
  Hexagon DSP (a `--pure` Q4_0 model only; its code is written to the app's
  files directory and loaded from there). Changing it reloads the model.

Sampling is temperature 0.7, top-p 0.9; replies stop at 512 tokens or the
model's end token; **Stop** ends generation early.
