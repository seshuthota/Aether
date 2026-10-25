#!/usr/bin/env bash
# Serve the notes site (site/) on this PC and share it through a Cloudflare
# Quick Tunnel, so it opens from anywhere at a https://….trycloudflare.com link.
# Usage: scripts/serve-site.sh        (Ctrl+C stops everything it started)
#
# Quick Tunnels need no Cloudflare account. The link is random, changes on
# every run, stops working when this script stops, and has no uptime guarantee.
# Anyone with the link can read the site, but only the site/ folder is served.
set -euo pipefail

PORT=8765
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # the repo root, wherever this is run from

command -v cloudflared >/dev/null || { echo "cloudflared is not installed" >&2; exit 1; }

# 1. Stop any earlier tunnel to this port, so only one link is live.
pkill -f "cloudflared tunnel.*localhost:$PORT" 2>/dev/null || true

# 2. Start the local web server, unless something already answers on the port.
#    Only a server this script started is stopped again on exit.
SERVER_PID=""
if curl -s -o /dev/null "http://localhost:$PORT/"; then
  echo "Reusing the server already running on port $PORT."
else
  python3 -m http.server "$PORT" --directory "$ROOT/site" >/dev/null 2>&1 &
  SERVER_PID=$!
  sleep 1
  echo "Started the local server: http://localhost:$PORT"
fi

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  echo "Stopped."
}
trap cleanup EXIT

# 3. Open the tunnel. cloudflared prints a lot; pick out the public link and
#    show it plainly, but keep the full log in case something goes wrong.
LOG="$(mktemp -t aether-tunnel.XXXXXX.log)"
cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" >"$LOG" 2>&1 &
TUNNEL_PID=$!
trap 'kill $TUNNEL_PID 2>/dev/null; cleanup' EXIT

URL=""
for _ in $(seq 1 30); do
  URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1 || true)"
  [ -n "$URL" ] && break
  kill -0 "$TUNNEL_PID" 2>/dev/null || break   # cloudflared exited early
  sleep 1
done

if [ -z "$URL" ]; then
  echo "No public link after 30 s. cloudflared said:" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

echo
echo "Public link: $URL"
echo "(it can take up to a minute before the link opens)"
echo "Full tunnel log: $LOG"
echo "Press Ctrl+C to stop."
wait "$TUNNEL_PID"
