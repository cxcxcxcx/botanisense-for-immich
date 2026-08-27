## What

<!-- one-paragraph summary of the change -->

## Why

<!-- motivation, link to issue if any -->

## How to verify

<!-- commands to run, URLs to open, screenshots for UI -->

## Checklist

- [ ] Tests pass (`uv run pytest tests/ -x` and `npm run typecheck`)
- [ ] Frontend builds (`cd frontend && npm run build`)
- [ ] No new secrets in code or config files
- [ ] No hardcoded personal defaults (LAN IPs, filesystem paths)
- [ ] New env vars documented in `.env.example` and the README
      "Configuration" table
- [ ] Touched backend? `pkill -TERM` and restart uvicorn — it doesn't
      reload from `--reload`-exclude patterns (see DEVELOPMENT.md)