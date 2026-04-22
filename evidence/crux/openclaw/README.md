# OpenCLAW evidence (category J, CRUX-J-01..J-20)

Competitor: [openclaw.ai](https://openclaw.ai/) — local-first personal AI
assistant / agent orchestration layer. **NOT** OpenCLIP (vision-language
contrastive models). The phonetic similarity is a trap; the projects are
unrelated.

Resolution date: 2026-04-18.

## Files

- `readme-verbs.txt` — canonical CLI surface extracted from openclaw.ai
  and docs.openclaw.ai. Authoritative list of OpenCLAW verbs aprender
  must map into apr-space.
- `config-schema.json5` — shape of `~/.openclaw/openclaw.json` (JSON5
  with comments). Used by CRUX-J-02 (config round-trip) and CRUX-J-03
  (per-channel allowFrom).
- `hello.sh` — minimal end-to-end bootstrap (install + onboard +
  dashboard). Used by CRUX-J-01 and CRUX-J-05 as a golden transcript.
- `capability-matrix.yaml` — 19 capabilities tagged `resolution:
  documented` or `resolution: gap` to drive contract prioritisation.
- `gaps.md` — 8 documented upstream gaps that keep the 20 J-contracts
  at `status: draft` until evidence firms up.

## Apr-side mappings (short)

- Skills + tool catalog → MCP tools (PMAT-CODE-MCP-CLIENT-001, closed).
- Chat-app transports → transport-agnostic Message envelope
  (PMAT-CLAUDE-PROXY-001 Claude Messages API proxy is the closest analogue).
- System-control shell.exec → SSC classifier gate (ssc-canary-eval-v1).
- Persistent memory → `apr memory put/get` + `$HOME`-rooted store.
- Onboard + dashboard + daemon → apr install path + `apr serve`
  loopback bind + `apr serve --install-daemon`.
