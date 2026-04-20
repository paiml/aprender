# OpenCLAW — documented upstream gaps

These are product claims openclaw.ai makes that are not backed by
pinned technical documentation. Each gap is referenced from the
corresponding CRUX-J contract's `description` and blocks promotion
from `draft` → `active` until closed with captured evidence.

Captured: 2026-04-18.

## 1. Daemon uninstall flow (CRUX-J-06)

`onboard --install-daemon` is documented. The inverse
`--uninstall-daemon` is implied but no transcript or unit-file
cleanup spec is published. Aprender parity contract asserts
`uninstall ∘ install = id`; we verify against our own daemon path,
not upstream's.

## 2. Per-sender session isolation (CRUX-J-07)

Product page claims "per-sender sessions". No formal description
of the isolation boundary (KV cache? tool history? memory keys?).
Aprender must enforce the full triple.

## 3. Safety-prompt catalogue (CRUX-J-14)

`onboard --safe` exists. The exact prompt list and which
capabilities default deny is not enumerated. Aprender's deny-list
is pinned explicitly in CRUX-J-14.

## 4. Persistent memory backing store (CRUX-J-10)

`openclaw memory put/get` verbs are visible. Whether the backing
store is SQLite, a vector DB, or a plain file is not documented.
Aprender picks SQLite for the apr-side parity contract because it
matches the dogfooded rusqlite dependency already in aprender-train.

## 5. Audit log format (CRUX-J-16)

Path `~/.openclaw/audit.log` is visible. Field schema is not
pinned (NDJSON assumed). Aprender parity pins NDJSON with four
event kinds: tool_call / msg_in / msg_out / daemon_state.

## 6. MCP tool envelope byte-exactness (CRUX-J-20)

OpenCLAW is MCP-shaped but we have no evidence it is byte-identical
to Claude-Code's tool-use envelope. Aprender parity pins against
the Anthropic MCP schema, not OpenCLAW's specific encoding.

## 7. Offline local fallback trigger (CRUX-J-19)

"Local-first" is advertised; the exact trigger (latency threshold?
explicit opt-in? network probe?) is not documented. Aprender
parity pins an explicit `provider=local` config path as the
first-class entry point.

## 8. Credential backend selection matrix (CRUX-J-18)

Keychain integration is claimed on all three OSes (macOS / Linux /
Windows). The exact fallback order when a backend is unavailable
is not documented. Aprender parity pins: keychain > opt-in
plaintext > refuse.
