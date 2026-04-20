# Canonical OpenCLAW CLI verbs (extracted from openclaw.ai, 2026-04-18)
# Source: https://openclaw.ai/ and docs.openclaw.ai
# Used as the competitor verb list for CRUX category J.

# Install (one-liner bootstrap)
curl -fsSL https://openclaw.ai/install.sh | bash
npm i -g openclaw

# First-run onboard (idempotent)
openclaw onboard
openclaw onboard --install-daemon
openclaw onboard --safe      # stricter destructive-op consent (CRUX-J-14)

# Runtime surface
openclaw dashboard           # opens http://127.0.0.1:18789 (CRUX-J-05)
openclaw update              # pull latest release + restart (CRUX-J-15)
openclaw skills list         # show registered skill/plugin catalog (CRUX-J-11)
openclaw skills add <pkg>    # install a skill
openclaw memory get <key>    # persistent memory (CRUX-J-10)
openclaw memory put <key> <val> [--ttl]

# Config path
~/.openclaw/openclaw.json    # JSON5 user config (CRUX-J-02)

# Audit / observability
~/.openclaw/audit.log        # NDJSON append-only audit trail (CRUX-J-16)

# Transport adapters (CRUX-J-12)
#   whatsapp | telegram | discord | slack | signal | imessage
# Each adapter consumes messages and produces the normalized envelope
# `{transport, channel_id, sender_id, body, ts, attachments}`.
