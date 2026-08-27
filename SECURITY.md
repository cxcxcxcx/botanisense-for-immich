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

## Configuration

The app loads `config.toml` (gitignored) for non-secret settings and
`.env` for the Immich API key. **Never commit these files.** The
templates `config.example.toml` and `.env.example` show what each key does.