#!/usr/bin/env python3
"""Minimal fake Immich API for the Playwright e2e suite.

Only the endpoints the app touches during onboarding + health are
implemented:

  GET /api/server/ping   reachability + version (no auth required)
  GET /api/albums        API-key scope check (401 without the key)

Everything else 404s on purpose: an unexpected call shows up in the
output instead of silently passing. No real photos, no database — this
server exists so the connect flow can be tested without an Immich
instance.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "e2e-1.0.0"
API_KEY = "e2e-api-key"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path == "/api/server/ping":
            self._json({"res": "pong", "version": VERSION})
        elif path == "/api/albums":
            if self.headers.get("x-api-key") != API_KEY:
                self._json({"message": "unauthorized"}, status=401)
            else:
                self._json([])
        else:
            self._json(
                {"message": f"fake Immich: {path} not implemented"},
                status=404,
            )

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[fake-immich] {fmt % args}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2284)
    args = ap.parse_args()
    print(f"[fake-immich] listening on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
