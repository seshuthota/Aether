#!/usr/bin/env bash
# Cargo "runner" for the aarch64-linux-android target.
# Cargo calls:  scripts/adb-run.sh <binary> [args...]
# We copy the binary to the phone, run it there, and return its exit code.
set -euo pipefail

# One phone, possibly several cargo processes (two worktrees, two Claude
# sessions). Take an exclusive lock for the whole run so their binaries never
# run at the same time: concurrent runs would distort each other's timings.
# The lock is released automatically when this script exits.
exec 9>/tmp/aether-phone.lock   # fixed path: every session must see the same file
flock 9

BIN="$1"      # path of the binary Cargo just built
shift         # what's left in "$@" are the arguments meant for that binary

# /data/local/tmp is the one place the adb shell user can both write and execute.
REMOTE_DIR=/data/local/tmp/aether
REMOTE_BIN="$REMOTE_DIR/$(basename "$BIN")"

# 1. Copy the binary to the phone. mkdir -p creates the folder if it's missing.
adb shell "mkdir -p $REMOTE_DIR"
adb push "$BIN" "$REMOTE_BIN" >/dev/null

# Golden files and models for on-device tests. --sync copies only files that
# changed since the last push, so after the first run this costs almost nothing.
TESTDATA="$(dirname "$0")/../testdata"
if [ -d "$TESTDATA" ]; then
  adb push --sync "$TESTDATA/." "$REMOTE_DIR/testdata/" >/dev/null
fi

# Keep the screen on for the run: with it off the vendor lowers the clock caps
# (1.0/1.8 GHz instead of 2.4 GHz on battery, results/phase-1.md), and every
# timing taken through this runner would be on the lowered clocks. Wake it,
# then press left Shift (does nothing) every minute so the screen timeout
# restarts; the key event runs on core 3. No phone setting is changed.
# AETHER_SCREEN_OFF=1 skips this, for runs that measure the screen-off state.
if [ -z "${AETHER_SCREEN_OFF:-}" ]; then
  adb shell input keyevent KEYCODE_WAKEUP
  ( while :; do sleep 60; adb shell taskset 8 input keyevent KEYCODE_SHIFT_LEFT >/dev/null 2>&1; done ) &
  KEEPALIVE=$!
  trap 'kill $KEEPALIVE 2>/dev/null' EXIT
fi

# 2. Run it. adb shell glues its arguments into one string for the phone's shell,
#    so we quote each argument first (printf %q) to keep spaces and special
#    characters intact. set +e stops the script from exiting immediately if the
#    run fails, because we still want to clean up first.
set +e
adb shell "chmod +x $REMOTE_BIN && cd $REMOTE_DIR && AETHER_TESTDATA=$REMOTE_DIR/testdata $REMOTE_BIN $(printf '%q ' "$@")"
STATUS=$?
set -e

# 3. Delete the copy (each build has a different hash name, so they'd pile up),
#    then exit with the phone's exit code so Cargo sees pass/fail correctly.
adb shell "rm -f $REMOTE_BIN"
exit $STATUS
