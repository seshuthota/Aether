#!/usr/bin/env bash
# Cargo "runner" for the aarch64-linux-android target.
# Cargo calls:  scripts/adb-run.sh <binary> [args...]
# We copy the binary to the phone, run it there, and return its exit code.
set -euo pipefail

BIN="$1"      # path of the binary Cargo just built
shift         # what's left in "$@" are the arguments meant for that binary

# /data/local/tmp is the one place the adb shell user can both write and execute.
REMOTE_DIR=/data/local/tmp/aether
REMOTE_BIN="$REMOTE_DIR/$(basename "$BIN")"

# 1. Copy the binary to the phone. mkdir -p creates the folder if it's missing.
adb shell "mkdir -p $REMOTE_DIR"
adb push "$BIN" "$REMOTE_BIN" >/dev/null

# 2. Run it. adb shell glues its arguments into one string for the phone's shell,
#    so we quote each argument first (printf %q) to keep spaces and special
#    characters intact. set +e stops the script from exiting immediately if the
#    run fails, because we still want to clean up first.
set +e
adb shell "chmod +x $REMOTE_BIN && $REMOTE_BIN $(printf '%q ' "$@")"
STATUS=$?
set -e

# 3. Delete the copy (each build has a different hash name, so they'd pile up),
#    then exit with the phone's exit code so Cargo sees pass/fail correctly.
adb shell "rm -f $REMOTE_BIN"
exit $STATUS
