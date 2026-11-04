#!/bin/sh
# Cargo runner for aarch64-unknown-linux-musl: runs the binary under QEMU
# user mode with dotprod and i8mm (and no SVE, like the phone's Oryon).
#
#   QEMU_AARCH64=/path/to/qemu-aarch64-static cargo test \
#       --release --target aarch64-unknown-linux-musl
#
# QEMU proves correctness, not speed: never record a time measured here.
exec "${QEMU_AARCH64:-$HOME/.local/opt/qemu/usr/bin/qemu-aarch64-static}" -cpu max,sve=off "$@"
