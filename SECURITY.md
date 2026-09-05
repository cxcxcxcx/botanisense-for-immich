# Security

## Reporting a vulnerability

**Please do not file a public GitHub issue for suspected security bugs.**

Email the maintainer at **cxcxcxcx@gmail.com** with a description of the
issue. Optional: include a minimal reproduction. You should receive a
response within 72 hours; if you don't, feel free to ping the issue
tracker once you're confident the disclosure was received.

For lower-severity reports or anything you're unsure about, you can also
open a [GitHub Security Advisory][advisory] privately — it routes to the
same maintainer without exposing the report publicly.

[advisory]: https://github.com/cxcxcxcx/botanisense-for-immich/security/advisories/new

## Scope

In scope:

- Anything that lets an unauthenticated remote actor read or modify the
  Immich instance the app is connected to (e.g. API key leakage, SSRF in
  asset URLs, server-side request forgery via search query).
- Anything that lets a remote actor read or modify files on the host
  where the app runs (path traversal in thumbnail serving, unsafe file
  writes from the API surface).
- Authentication bypass — though the app currently has no auth (see
  README → "Known limits"), please flag if you find a way to access it
  across users in a multi-tenant setup.

Out of scope:

- The BioCLIP-2 model itself (upstream issue; report to
  [imageomics/bioclip-2](https://huggingface.co/imageomics/bioclip-2)).
- The Immich server (report to [immich-app/immich](https://github.com/immich-app/immich)).

## Supported versions

| Version | Supported |
|---|---|
| `main`  | ✅ |
| < `main` | ❌ |

Only the latest commit on `main` receives security fixes. Until v1.0,
there are no backports.

## Deployment posture (read before exposing this app)

- **There is no authentication.** Anyone who can reach the port can
  trigger pipeline runs, KNN builds, and Immich album writes/deletes.
  This is designed for a trusted LAN; put auth in front (reverse-proxy
  basic-auth at minimum) before any wider exposure. See README →
  "Known limits".
- **YOLO-World checkpoint loading disables torch's `weights_only`
  safe-load** (`yolo.py` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`),
  because the ultralytics checkpoint needs full unpickling. This is an
  accepted risk: it widens what a tampered checkpoint file could do.
  The weights come from the official ultralytics release over HTTPS,
  and `yolov8s-world.pt` is never committed to this repo.
- **Two diagnostic endpoints are unauthenticated by design**:
  `/api/mem` (tracemalloc heap snapshots) and `/api/perf` (per-route
  latency stats). They expose details about the host's memory layout
  and usage patterns to anyone who can reach the port. Acceptable on a
  single-user LAN; consider a proxy rule to block them if you expose
  the app more widely.

## Configuration

The Immich URL and API key are configured either from the web UI
(stored **in plaintext** in the SQLite DB under `data/state.db` — the
same posture as a `.env` file on disk) or via the `IMMICH_BASE_URL` /
`IMMICH_API_KEY` environment variables, which take precedence. Note
that `POST /api/settings/immich` is unauthenticated: anyone who can
reach the port can repoint the app at a different Immich server. This
is consistent with the no-auth posture above and acceptable on a
single-user LAN; don't expose this app more widely. **Never commit**
`.env` or `data/state.db`. The `.env.example` template documents the
supported environment variables.