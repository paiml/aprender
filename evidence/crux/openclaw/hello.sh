#!/usr/bin/env bash
# Canonical OpenCLAW bootstrap transcript (CRUX-J-01 + J-05).
# Source: openclaw.ai landing page, 2026-04-18.
# This is the golden reference; aprender parity is measured against it.
set -euo pipefail

# Step 1 — install (one-liner). npm or curl, either is documented.
curl -fsSL https://openclaw.ai/install.sh | bash
# alt: npm i -g openclaw

# Step 2 — first-run onboard. Idempotent; safe to re-run.
openclaw onboard

# Step 3 — optional daemon install (CRUX-J-06).
openclaw onboard --install-daemon

# Step 4 — open the loopback Control UI (CRUX-J-05).
openclaw dashboard
# opens http://127.0.0.1:18789 in the default browser.
# bind is loopback-only by default; no LAN exposure without an explicit flag.

# Step 5 — smoke test (optional). Writes a memory key (CRUX-J-10).
openclaw memory put hello world --ttl 3600
openclaw memory get hello      # → "world"
