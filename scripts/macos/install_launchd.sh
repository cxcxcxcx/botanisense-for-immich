#!/bin/bash
# Run BotaniSense natively on macOS as a per-user launchd agent: starts at
# login, restarts if it dies, logs to data/server.log.
#
#   scripts/macos/install_launchd.sh              # install / reinstall + start
#   PORT=8780 scripts/macos/install_launchd.sh    # another port
#   scripts/macos/install_launchd.sh --uninstall  # stop + remove
#
# Run from anywhere; it uses the checkout it lives in. Needs the venv from
# `uv sync --frozen --extra mac` and a built frontend (see README).
set -euo pipefail

LABEL=app.botanisense
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${PORT:-8765}"
# gui/ is the login session; over ssh (no GUI login on this session)
# launchctl can only reach the user/ domain. The plist in LaunchAgents
# loads into gui/ at the next login either way.
DOMAINS=("gui/$(id -u)" "user/$(id -u)")

for d in "${DOMAINS[@]}"; do launchctl bootout "$d/$LABEL" 2>/dev/null || true; done
if [[ "${1:-}" == "--uninstall" ]]; then
  rm -f "$PLIST"
  echo "removed $LABEL"
  exit 0
fi

PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || { echo "no $PY — run: uv sync --frozen --extra mac" >&2; exit 1; }
"$PY" -c "import coremltools" 2>/dev/null \
  || { echo "coremltools missing — run: uv sync --frozen --extra mac" >&2; exit 1; }
[[ -f "$REPO/frontend/dist/index.html" ]] \
  || { echo "no frontend build — run: (cd frontend && npm ci && npm run build)" >&2; exit 1; }
mkdir -p "$REPO/data" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PY</string><string>-m</string><string>uvicorn</string><string>app:app</string>
    <string>--host</string><string>0.0.0.0</string><string>--port</string><string>$PORT</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>ENGINE</key><string>auto</string>
    <key>HF_HOME</key><string>$REPO/data/huggingface</string>
    <key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$REPO/data/server.log</string>
  <key>StandardErrorPath</key><string>$REPO/data/server.log</string>
</dict>
</plist>
EOF

for d in "${DOMAINS[@]}"; do
  if launchctl bootstrap "$d" "$PLIST" 2>/dev/null; then
    echo "started $LABEL in $d on port $PORT — log: $REPO/data/server.log"
    exit 0
  fi
done
echo "launchctl bootstrap failed in ${DOMAINS[*]} (over ssh there is no GUI session)." >&2
echo "The agent starts at the next login, or re-run this in Terminal on the Mac." >&2
exit 1
