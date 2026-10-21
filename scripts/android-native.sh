#!/usr/bin/env bash
# Builds the engine as a shared library for the phone and the JNI shim that
# exposes it to Kotlin, into android/app/src/main/jniLibs/arm64-v8a/.
# Uses the NDK clang named as the linker in .cargo/config.toml.
set -euo pipefail
cd "$(dirname "$0")/.."

cc=$(sed -n 's/^linker *= *"\(.*\)"/\1/p' .cargo/config.toml | head -1)
[ -x "$cc" ] || { echo "NDK clang not found (linker in .cargo/config.toml): $cc" >&2; exit 1; }
strip="$(dirname "$cc")/llvm-strip"
out=android/app/src/main/jniLibs/arm64-v8a
mkdir -p "$out"

cargo build --release --target aarch64-linux-android --lib
lib=target/aarch64-linux-android/release
# The release profile keeps debug symbols for simpleperf; the APK doesn't need them.
"$strip" --strip-debug -o "$out/libaether.so" "$lib/libaether.so"
"$cc" -shared -fPIC -O2 -Wall -Wextra -Werror -Iinclude -Wl,-z,max-page-size=16384 \
    android/app/src/main/cpp/aether_jni.c -L"$out" -laether -o "$out/libaether_jni.so"
ls -l "$out"
