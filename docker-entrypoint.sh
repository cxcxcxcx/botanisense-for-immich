#!/bin/sh
# BotaniSense for Immich image entrypoint.
#
# Runs as root for one purpose: Compose auto-creates a missing ./data
# bind mount as root:root, which the unprivileged app user cannot write
# (first boot then dies in storage.init_db → sqlite3.connect, and
# restart: unless-stopped turns that into a crash loop). Fix the
# mount's ownership, then drop to uid 1000 — the same pattern the
# official postgres image uses. Never touch anything outside /app/data.
set -e

if [ "$(id -u)" = "0" ]; then
    if [ "$(stat -c %u /app/data 2>/dev/null)" != "1000" ]; then
        chown -R botanisense:botanisense /app/data
    fi
    exec setpriv --reuid botanisense --regid botanisense --clear-groups "$@"
fi

exec "$@"
