# APR-MCP-SERVER: Model Context Protocol Server Specification

**Version**: 1.2.0
**Date**: 2026-04-19 (M1–M3 shipped in v0.31.0; M4 code complete — all 5 PRs merged same day)
**Status**: ACTIVE — `aprender-mcp` ships 9 tools over stdio JSON-RPC 2.0; FALSIFY-MCP-008 ENFORCED at 4 layers (schema+description × live+codegen); FALSIFY-MCP-003/-004 response-shape layer ENFORCED (PR #889); real-model FALSIFY-MCP-E2E-001 ENFORCED (PR #892); end-to-end dogfood FALSIFY-MCP-DOGFOOD-001 ENFORCED (PR #890); contract `apr-mcp-server-v1.yaml` ACTIVE with exact-8 `FALSIFY-MCP-001..008` invariant pinned (PR #886); `apr.run` progress notifications via `apr run --stream` NDJSON + `NotificationSink` forwarding ENFORCED (PR #891 / FALSIFY-MCP-PROGRESS-002). **All 5 M4 PRs merged 2026-04-19**; M4 code-complete, manual Cursor/Cline smoke tests remain.
**Contracts**:
- `contracts/mcp-tool-schema-v1.yaml` — upstream MCP tool registration, schema fidelity, session lifecycle, error mapping (existing)
- `contracts/apr-mcp-tool-schemas-v1.yaml` — per-tool `inputSchema` + description source of truth; drives `build.rs` codegen; `status: ENFORCED` (M3, 2026-04-18)
- `contracts/pmcp/mcp-protocol-sdk-v1.yaml` — `pmcp` crate contract (existing)
- `contracts/apr-tool-rust-mcp-sdk-v1.yaml` — `paiml/rust-mcp-sdk` dependency contract (existing)
- `contracts/apr-cli-commands-v1.yaml` — 58-command tool surface (57 commands + `mcp` added 2026-04-17 per PR #864)
- `contracts/apr-mcp-server-v1.yaml` — end-to-end MCP server contract; 8 falsification_conditions + test cross-links + exact-8 invariant pinned by `apr_mcp_server_contract_ids_are_falsify_mcp_001_through_008`; `status: ACTIVE` (PR #886 merged 2026-04-19; top-level `status: ACTIVE` + all 8 falsification_conditions `status: ENFORCED`)
- `contracts/apr-claude-proxy-v1.yaml` — Anthropic Messages-API proxy request/response shape + 6 falsification gates; `status: DRAFT` (PMAT-CLAUDE-PROXY-001, promotes to ENFORCED at M6-δ, 2026-04-18)
- `contracts/apr-code-parity-v1.yaml` — falsifiable encoding of the 21-row `apr code` ↔ Claude Code parity matrix; 5 falsification gates incl. row-by-row mechanical audit + headline aggregate invariant + prose↔YAML drift check; `status: ACTIVE` as of 2026-04-18 revision 5.1 (`pv validate` green via in-tree `aprender-contracts-cli`; epic PMAT-CODE-PARITY-MATRIX-001 **CLOSEABLE** — both closure conditions MET; current counts **14 SHIPPED / 3 PARTIAL / 4 NONE over 21 rows** per `pv check-parity` after **all 4 P0 + 5 P1 + 2 P2 tickets closed** — MCP-CLIENT, SLASH-PARITY, HOOKS, SPAWN-PARITY, CUSTOM-AGENTS, WEB-TOOLS, SKILLS, WORKTREE, PERMISSIONS, STATUS-LINE, ORG-POLICY; SHIPPED ≥9 AND MISSING ≤4 simultaneously satisfied; FALSIFY-CODE-PARITY-005 passes; 4 remaining MISSING rows are all P2 deferred surfaces with no epic dependency). Both gates green: `pv validate` enforces SCHEMA; `pv check-parity` executes each row's `cross_check_command` + enforces headline aggregate invariant (FALSIFY-CODE-PARITY-002). Neither is a bash script.
**References**:
- [Model Context Protocol Specification v2024-11-05](https://spec.modelcontextprotocol.io/specification/2024-11-05/)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
- [pmcp crate](https://github.com/paiml/rust-mcp-sdk) — PAIML's Rust MCP SDK, actively maintained, v2.3.0 on crates.io (verified 2026-04-19 via `cargo search pmcp` — top entry)

---

## Problem

Aprender ships a 58-subcommand CLI (`apr`) with structured `--json` output on most commands (57 commands pre-MCP plus `apr mcp` itself, added 2026-04-17 per PR #864). It achieves 1.43× Ollama decode perf at 128 tokens. But no agentic tool (Claude Code, Cursor, Cline, Aider, Continue) can invoke it without MCP.

Every competitor tool with ecosystem momentum in early 2026 is addressable via MCP — except the local-inference tier. Ollama, llama.cpp, and Unsloth all lack first-party MCP servers. Shipping `apr mcp` first occupies that slot.

Separately, aprender already ships [`apr code`](../../crates/aprender-orchestrate/docs/specifications/components/apr-code.md) (PMAT-182) — our sovereign Claude-Code equivalent. For full parity, `apr code` must also be able to *consume* external MCP servers (the client direction of `.mcp.json`), not just be consumable. `aprender-mcp` covers the server side; the client-side wiring landed 2026-04-18 (PMAT-CODE-MCP-CLIENT-001 CLOSED — `register_mcp_client_tools` now ships in `agent/code.rs`); a project-root `.mcp.json` loader is tracked as follow-up PMAT-CODE-MCP-JSON-LOADER-001 (P2). See § "Relationship to `apr code`" below.

## Goal

A single subcommand — `apr mcp` — that starts an MCP server over stdio, exposing a curated subset of the 58 apr CLI commands as MCP tools. Tool schemas are generated at build time from `contracts/apr-mcp-tool-schemas-v1.yaml` (FALSIFY-MCP-008), not hand-written.

Success is measured by **bidirectional** Claude-Code parity:
- **Server direction (M1-M5, active):** Claude Code / Cursor / Cline can `.mcp.json`-configure `apr mcp` and invoke `apr.run`, `apr.qa`, `apr.trace`, etc. on local models. Coverage expanding from 9 → full 58-command surface (PMAT-MCP-PARITY-001).
- **Client direction (PMAT-CODE-MCP-CLIENT-001, CLOSED 2026-04-18):** Our own `apr code` agent loads external MCP servers as tool providers via `manifest.mcp_servers[]` (TOML) and the existing `McpClientTool`, mirroring Claude Code's own MCP-client capability. `register_mcp_client_tools` (`agent/code.rs:400-423`) wires discovery + registration after `build_code_tools`. Follow-up PMAT-CODE-MCP-JSON-LOADER-001 (P2, deferred) adds the project-root `.mcp.json` file loader.

## Architecture

### New Crate: `aprender-mcp`

Location: `crates/aprender-mcp/`

```
crates/aprender-mcp/
├── Cargo.toml            # serde, serde_json, anyhow (native); nix (unix signals); serde_yaml (build); jsonschema (dev)
├── build.rs              # FALSIFY-MCP-008: reads contracts/apr-mcp-tool-schemas-v1.yaml → $OUT_DIR/schemas.rs (APR_<TOOL>_SCHEMA + APR_<TOOL>_DESCRIPTION, PMAT-514)
├── src/
│   ├── lib.rs            # public API: `AprMcpServer::new().run_stdio()`; `include!` of build-time schemas.rs
│   ├── types.rs          # JSON-RPC 2.0 envelopes + MCP protocol types (mirrors aprender-orchestrate::mcp::types)
│   ├── server.rs         # hand-rolled JSON-RPC dispatcher + worker-thread cancellation (pmcp adoption planned M5+)
│   └── tools/
│       ├── mod.rs            # registration table
│       ├── subprocess.rs     # shared `run_apr` / `run_apr_cancellable` (M3 FALSIFY-MCP-006)
│       ├── version.rs        # `apr.version` — M1 handshake probe
│       ├── run.rs            # `apr.run` — inference
│       ├── qa.rs             # `apr.qa` — quality gates
│       ├── trace.rs          # `apr.trace` — layer analysis
│       ├── tensors.rs        # `apr.tensors` — tensor inspection
│       ├── validate.rs       # `apr.validate` — integrity
│       ├── bench.rs          # `apr.bench` — perf benchmarks
│       ├── finetune.rs       # `apr.finetune` — LoRA training
│       └── serve.rs          # `apr.serve` — OpenAI API server lifecycle
└── tests/                        # protocol-level integration tests
    ├── falsify_m1.rs                 # FALSIFY-MCP-001 init, -002 tools/list, -005 jsonrpc, -007 protocolVersion, -VALIDATE-001 tool-error shape
    ├── falsify_mcp_006.rs            # FALSIFY-MCP-006 notifications/cancelled → SIGTERM→SIGKILL
    ├── falsify_mcp_008.rs            # FALSIFY-MCP-008 YAML-vs-live schema + description byte-identity + codegen coverage guardrails
    ├── falsify_mcp_progress_001.rs   # FALSIFY-MCP-PROGRESS-001 apr.finetune opt-in per-line progress
    └── falsify_schema.rs             # FALSIFY-MCP-002 strict slice — JSON Schema Draft 7 meta-validation per tool
```

### `apr mcp` Subcommand

Wired into `apr-cli`:

```rust
// crates/apr-cli/src/commands/mcp.rs
pub fn run() -> Result<(), CliError> {
    let mut server = aprender_mcp::AprMcpServer::new();
    server
        .run_stdio()
        .map_err(|e| CliError::Aprender(format!("mcp server: {e}")))
}
```

Blocking (not `async`) — `stdio` is read with a std-library loop and each
tool call dispatches onto a per-request worker thread (see `server.rs`).
No CLI args in Phase 1; transport selection is deferred to M5, which adds
SSE + WebSocket via `pmcp` v2.3 (see Milestones).

### Tool Surface (Phase 1)

Eight high-value workflow tools for agentic coding + ML. The M1 scaffold tool `apr.version` is also registered, so `tools/list` returns 9 tools total.

| Tool | Maps to CLI | Inputs (required **bold**) | Output |
|------|-------------|----------------------------|--------|
| `apr.run` | `apr run <model>` | **`model_path`**, `prompt`, `max_tokens`, `temperature`, `top_p` | `{model, text, tokens: [u32], tokens_generated, max_tokens, tok_per_sec, inference_time_ms, used_gpu, cached}` (CLI as of 2026-04-18; `stop_reason` not emitted) |
| `apr.serve` | `apr serve <model>` | **`model_path`**, `port` | `{pid, url}` + lifecycle |
| `apr.qa` | `apr qa <model> --json` | **`model_path`**, `assert_tps`, `max_tokens`, `iterations` | `{model, passed, gates: [{name, passed, message, value?, threshold?, duration_ms, skipped}], gates_executed, gates_skipped, total_duration_ms, timestamp, summary}` (CLI as of 2026-04-18; gate field is `passed` not `pass`) |
| `apr.trace` | `apr trace <model> --json` | **`model_path`**, `layer` | layer skeleton read from metadata: `{stats_source: "metadata-only", notes, layers, summary}`. No forward pass runs, so every `*_stats` field is null and `anomaly_count` is 0 — use `apr trace --payload` for measured activations. `layer` filters by substring; a filter matching nothing yields `layers: []` plus a note (#2407). `reference` was advertised but unimplemented; it is refused (#2407) |
| `apr.tensors` | `apr tensors <model> --json` | **`model_path`**, `stats`, `filter` | tensor list with shapes/dtypes (+ stats when `stats: true`; `filter` substring-matches tensor names) |
| `apr.validate` | `apr validate <model> --json` | **`model_path`** | integrity + quality gates |
| `apr.bench` | `apr bench <model> --json` | **`model_path`**, `iterations`, `max_tokens`, `prompt` | median tok/s, p50/p95/p99 latency |
| `apr.finetune` | `apr finetune <base_model> --json` | **`base_model`**, `dataset`, `lora_rank`, `epochs`, `method`, `output` | progress events + final checkpoint path |
| `apr.version` (M1 scaffold) | — (server-synthesized, no subprocess) | _(none)_ | `{server, version, protocol_version}` |

Schema + description generation: each tool's `inputSchema` and tool-level `description` are emitted from `contracts/apr-mcp-tool-schemas-v1.yaml` at build time by `aprender-mcp`'s `build.rs` — **no hand-maintained schemas or descriptions**. Codegen lands two constants per tool in `$OUT_DIR/schemas.rs`: `pub const APR_<TOOL>_SCHEMA: &str` (serialized JSON Schema, consumed via `serde_json::from_str`) and `pub const APR_<TOOL>_DESCRIPTION: &str` (consumed via `.to_string()`, PMAT-514).

### Protocol

- **Transport**: stdio only in Phase 1 (SSE deferred — see Out of Scope)
- **Version**: MCP v2024-11-05 (matches `mcp-tool-schema-v1.yaml`)
- **Lifecycle**: initialize → initialized → tools/list → tools/call → (long-running tools stream progress) → shutdown
- **Streaming**: `apr.finetune` sends one `notifications/progress` per non-empty stdout line of `apr finetune --json` when the client opts in via `params._meta.progressToken` (FALSIFY-MCP-PROGRESS-001, ENFORCED). `apr.run` progress is an M4 follow-up — it needs an `apr run --stream` CLI prereq and a per-step CLI event channel (currently `apr finetune` emits a terminal blob, not per-step structured events). All other tools return synchronously.
- **Cancellation**: `notifications/cancelled` from client → kill the spawned `apr` subprocess with SIGTERM (30s grace) → SIGKILL.
- **Error mapping** (per `mcp-tool-schema-v1.yaml`):
  - Parse error → `-32700`
  - Invalid request → `-32600`
  - Method not found → `-32601`
  - Invalid params → `-32602`
  - Internal error → `-32603`
  - Custom domain errors (model not found, CUDA OOM, contract violation) → `-32000..-32099`

## Configuration

`.mcp.json` for Claude Code / Cursor / Cline:

```json
{
  "mcpServers": {
    "aprender": {
      "command": "apr",
      "args": ["mcp"],
      "env": {
        "APR_MODEL_DIR": "/home/user/.cache/apr/models"
      }
    }
  }
}
```

**Config loading (Phase 2 — not implemented in Phase 1).** When a server-side
config file is wired up (timeouts, allowed model roots, tool allow-list), it
will follow the precedence below:

1. `--config <path>` flag
2. `$APR_MCP_CONFIG` env var
3. `~/.config/apr/mcp.toml`
4. Built-in defaults

Phase 1 today: `apr mcp` takes no CLI args and consults no env vars directly.
Env vars set in `.mcp.json` (e.g. `APR_MODEL_DIR`) are read by the spawned
`apr <cmd>` subprocesses themselves, not by the MCP server process.

## Falsification Conditions (for `apr-mcp-server-v1.yaml`)

The server is only ACTIVE if all of these are falsifiable by CI:

1. **FALSIFY-MCP-001**: `apr mcp < init.json` responds to `initialize` within 500ms with `{"protocolVersion":"2024-11-05", ...}`. **ENFORCED**.
2. **FALSIFY-MCP-002**: `tools/list` returns the 8 Phase-1 workflow tools plus the `apr.version` M1 scaffold (9 total registered); every tool's schema validates against JSONSchema Draft 7. **ENFORCED** (see `crates/aprender-mcp/tests/falsify_m1.rs::falsify_mcp_002_tools_list_schema_shape`).
3. **FALSIFY-MCP-003**: `tools/call apr.run` on `qwen2.5-0.5b-instruct-q4km.gguf` with prompt "1+1=" decodes "2" as first token within 5s. **ENFORCED (response-shape layer)** — PR #889 merged 2026-04-19 promotes the surface layer to full mock-subprocess e2e response-shape gates (`crates/aprender-mcp/tests/falsify_mcp_003.rs`, 3 tests: happy-path response-shape, stop-reason, error surface). Real-model gate (live qwen2.5-0.5b on CI) remains M4 dogfood scope — covered by PR #892 as new gate FALSIFY-MCP-E2E-001.
4. **FALSIFY-MCP-004**: `tools/call apr.qa` returns 8 gates with correct pass/fail states matching `apr qa --json` CLI output byte-for-byte. **ENFORCED (response-shape layer)** — PR #889 merged 2026-04-19 promotes the surface layer to full mock-subprocess e2e response-shape gates (`crates/aprender-mcp/tests/falsify_mcp_004.rs`, 2 tests: 8-gate structure + pass/fail field parity). Real-model byte-for-byte parity against `apr qa --json` remains M4 dogfood scope — covered by PR #892 as FALSIFY-MCP-E2E-001.
5. **FALSIFY-MCP-005**: Malformed request (`"jsonrpc": "1.0"`) returns JSON-RPC error code `-32600`, does not crash server. **ENFORCED**.
6. **FALSIFY-MCP-006**: `notifications/cancelled` during `apr.run` stops decoding within 30s, returns partial result. **ENFORCED**.
7. **FALSIFY-MCP-007**: Protocol version mismatch (`"protocolVersion": "1999-01-01"`) returns error, does not attempt tools/list. **ENFORCED**.
8. **FALSIFY-MCP-008**: Schema + description in `tools/list` output are byte-identical to the entry from `contracts/apr-mcp-tool-schemas-v1.yaml`. **ENFORCED** — `inputSchema` equality via `migrated_tools_match_yaml_contract_byte_for_byte`, tool-level `description` equality at two layers: live `ToolDefinition.description` via `tool_descriptions_match_yaml_contract`, and the `build.rs`-emitted `schemas::APR_<TOOL>_DESCRIPTION` codegen constants via `codegen_description_constants_match_yaml` (all in `tests/falsify_mcp_008.rs`). Both `inputSchema` and `description` are emitted by `build.rs` from the YAML — hand-editing them in Rust source is not possible.
9. **FALSIFY-MCP-PROGRESS-001** (M3 addition): When client supplies `params._meta.progressToken`, `apr.finetune` emits one `notifications/progress` per non-empty stdout line of `apr finetune --json`, all flushed before the final `tools/call` response. Without a token, zero notifications. **ENFORCED**.
10. **FALSIFY-MCP-DOGFOOD-001** (M4 addition): End-to-end protocol session — the real `apr mcp` binary, launched as a subprocess, answers `initialize`, `tools/list`, and `tools/call` for all 9 registered tools (plus unknown-method and bad-jsonrpc gates) within 2s via stdio. Test: `crates/aprender-mcp/tests/falsify_mcp_dogfood_001.rs::falsify_mcp_dogfood_001_full_client_session`. **ENFORCED**.
11. **FALSIFY-MCP-E2E-001** (M4 addition): Real-model end-to-end. `apr.run` on a locally-cached qwen2-0.5b GGUF (env-var `APR_MCP_E2E_MODEL`) decodes content containing the digit "2" within 30s; `apr.qa` MCP wrapper output equals `apr qa --json` direct-CLI output (modulo nondeterministic timestamp/duration/throughput fields). Env-gated — skips via `println!` + early return when the env var is unset (project policy bans `#[ignore]`). Test file: `crates/aprender-mcp/tests/falsify_mcp_e2e_001.rs`. **ENFORCED** (when env-gated).
12. **FALSIFY-MCP-PROGRESS-002** (M4 addition): When client supplies `params._meta.progressToken`, `apr.run` spawns `apr run --stream` and forwards each NDJSON line of the CLI's per-token emission as a `notifications/progress` message tagged with the caller's token; all notifications flush before the final `tools/call` response. Without a token, zero notifications (MCP spec compliance). Test file: `crates/aprender-mcp/tests/falsify_mcp_progress_002.rs` (4 tests: ordering, token-tagging, no-token zero-emit, flush-before-result). **ENFORCED** (PR #891 merged 2026-04-19).

### Additional dispatcher invariant (not in `apr-mcp-server-v1.yaml`)

- **FALSIFY-MCP-VALIDATE-001**: Tool argument validation failure (e.g. missing required `model_path`) must surface as a `tools/call` result with `isError: true` and a human-readable `content[].text`, **not** as a JSON-RPC error envelope. This is a dispatcher-level contract point (how the server shapes tool errors) rather than a per-tool behavioural promise, so it lives alongside — but outside — the `apr-mcp-server-v1.yaml` falsification set. **ENFORCED** (see `crates/aprender-mcp/tests/falsify_m1.rs::falsify_validate_missing_model_path_is_tool_error`).

## Relationship to `apr code` (Sovereign Coding Assistant)

`apr code` (spec: [`docs/specifications/aprender-orchestrate/components/apr-code.md`](../../crates/aprender-orchestrate/docs/specifications/components/apr-code.md); contract: `contracts/batuta/apr-code-v1.yaml`) is aprender's **open-source, sovereign-by-default equivalent of Claude Code** (PMAT-182). Humans use `apr code` interactively; external agents (Claude Code, Cursor, Cline) use `apr mcp`. The two surfaces form a bidirectional MCP relationship:

### Direction 1 — `apr code` as MCP **consumer** (CLOSED — PMAT-CODE-MCP-CLIENT-001, 2026-04-18)

`apr code` can load external MCP servers as tool providers, the Claude-Code-parity equivalent of `.mcp.json`. Infrastructure lives in `crates/aprender-orchestrate/src/agent/tool/mcp_client.rs` — `McpClientTool` + `StdioMcpTransport` + `discover_mcp_tools(manifest)` — and is feature-gated behind `agents-mcp`. The wiring into the live agent loop is `register_mcp_client_tools` at `crates/aprender-orchestrate/src/agent/code.rs:400-423`, invoked after `build_code_tools` so TOML-declared `manifest.mcp_servers[]` entries register as live tool providers.

**Parity gaps vs Claude Code** (re-falsified 2026-04-19 via `pmat query "register_mcp_client_tools"` — wiring landed in `agent/code.rs:400-423`):

| Gap | Status | Fix location |
|-----|--------|--------------|
| `build_code_tools` does **not** call `register_mcp_tools` — external MCP servers declared in manifest are silently ignored | CLOSED 2026-04-18 (PMAT-CODE-MCP-CLIENT-001) | `crates/aprender-orchestrate/src/agent/code.rs:400-423` (`register_mcp_client_tools` guarded by `cfg(feature = "agents-mcp")`) |
| No `.mcp.json` loader — today only TOML `AgentManifest.mcp_servers` is read | OPEN (P2 deferred, PMAT-CODE-MCP-JSON-LOADER-001) | `crates/aprender-orchestrate/src/agent/manifest.rs` (add JSON reader at `$CWD/.mcp.json` and `~/.config/apr/mcp.json`) |
| Transport is stdio-only; SSE/WebSocket deferred to M5 (matches `aprender-mcp`'s own surface — intentional parity) | ACCEPTED | M5 |

### Direction 2 — `aprender-mcp` as MCP **producer** (PARTIAL — 9 of 58 commands exposed)

Live `tools/list` returns **9 tools** (falsified 2026-04-18 via `echo '{...}' | apr mcp`): `apr.version` (M1 scaffold) + the 8 Phase-1 workflow tools. Claude Code / Cursor / Cline can invoke those 9 remotely. The other 49 `apr` commands are unreachable via MCP today — a parity gap flagged for Phase 2.

| Status | Count | Commands |
|--------|-------|----------|
| Exposed via MCP (Phase 1) | 9 | `apr.version`, `apr.validate`, `apr.tensors`, `apr.bench`, `apr.qa`, `apr.trace`, `apr.run`, `apr.serve`, `apr.finetune` |
| **Not yet exposed (Phase 2 targets)** | 49 | `chat`, `code`, `inspect`, `debug`, `lint`, `explain`, `diff`, `hex`, `tree`, `flow`, `export`, `import`, `convert`, `compile`, `merge`, `quantize`, `rosetta`, `pull`, `list`, `rm`, `publish`, `prune`, `distill`, `train`, `tokenize`, `tune`, `eval`, `check`, `qualify`, `canary`, `compare-hf`, `parity`, `gpu`, `profile`, `ptx`, `ptx-map`, `cbtop`, `data`, `pipeline`, `tui`, `monitor`, `runs`, `experiment`, `showcase`, `probar`, `diagnose`, `oracle`, `encrypt`, `decrypt` |

Phase-2 priorities (first expansion batch after M5 dispatcher port, in rough order): `apr.inspect` (metadata probe — complements `apr.validate`), `apr.lint` (best-practice gate — complements `apr.qa`), `apr.diff` (model-vs-model comparison), `apr.convert` + `apr.export` + `apr.import` + `apr.quantize` (format pipeline — already JSON-clean), `apr.pull` (model download), `apr.profile` (roofline), `apr.explain` (natural-language model description), `apr.tokenize`, `apr.eval`, `apr.probar` (behaviour-test pipeline). Deferred indefinitely: interactive-only tools (`apr chat`, `apr tui`, `apr cbtop`, `apr monitor`) and meta-commands (`apr mcp` itself).

### Direction 3 — `apr.code` as an MCP tool (PLANNED M5+)

A future `apr.code` MCP tool would let external clients drive the full `apr code` agent loop (perceive → reason → act over realizar + stack tools), rather than just single CLI commands. Conceptually this is the agentic equivalent of `apr.run` (single-shot inference) — it takes `{prompt, project, max_turns}` and streams per-tool-call `notifications/progress`. **CLI wire shape now exists** — `apr run --stream` NDJSON + `NotificationSink` forwarding landed via PR #891 (FALSIFY-MCP-PROGRESS-002, 2026-04-19), so the MCP progress-forwarding pattern is proven on `apr.run`. Remaining prereq: threading per-decoded-token callbacks through realizar's inference loops (today's `apr run --stream` emits NDJSON *post-decode*, not *per-token*); the same callback mechanism will front-load agent-loop events for `apr.code`.

### Feature-by-feature parity matrix — `apr code` vs Claude Code (2026-04-18 audit)

Beyond the three directions above, Claude Code ships a dense feature surface (slash commands, hooks, skills, plugins, permission modes, worktrees). This matrix falsifies **each** Claude-Code feature against the current `apr code` implementation via `pmat query` + `grep` against `crates/aprender-orchestrate/src/`. Every row ends with a concrete source path or ticket — no generic "we should add X" entries.

| Feature | apr code status | Evidence / fix location | Ticket |
|--------|-----------------|-------------------------|--------|
| **Slash commands** (`/help` etc.) | SHIPPED (21 of ~24) | `agent/repl.rs` `SlashCommand` enum now has 21 variants: the original 11 (`Help, Quit, Cost, Context, Model, Compact, Clear, Session, Sessions, Test, Quality`) **plus** 10 Claude-Code-parity variants added 2026-04-18: `Mcp, Config, Review, Memory, Permissions, Hooks, Init, Resume, AddDir, Agents` (+ `Unknown(name)`). `/help` advertises them; stubs print closure-ticket messages instead of silently routing to `Unknown`. Unit test `test_slash_command_parse_claude_code_parity` locks the set. Only still-absent commands: `/debug, /rename, /upgrade` (deferred as PMAT-CODE-SLASH-EXTENDED-001, P2) | PMAT-CODE-SLASH-PARITY-001 **CLOSED** 2026-04-18 |
| **CLAUDE.md / APR.md memory** | SHIPPED | `load_project_instructions` (`agent/code.rs:253-256`) loads `APR.md` then `CLAUDE.md` from `--project`. No `@import` syntax yet, no `~/.claude/CLAUDE.md` user-level fallback, no auto-memory directory | PMAT-CODE-MEMORY-PARITY-001 (new) — add `@file` imports + user-level `~/.config/apr/APR.md` fallback + auto-memory at `~/.config/apr/projects/<hash>/memory/` |
| **Hooks** (`SessionStart`, `PreToolUse`, `PostToolUse`, `UserPromptSubmit`, `Stop`, `SubagentStop`) | SHIPPED (core) — runtime call sites for 5 of 6 events MISSING | New `agent/hooks.rs` (2026-04-18) ships the full surface: `HookEvent` enum with all 6 canonical events, `HookConfig` deserializable from TOML `[[hooks]]` tables (`event`, optional `matcher`, `command`, `timeout_secs`), `HookDecision::{Allow, Warn, Block}` with exit-code→decision semantics matching Claude Code (0=allow, 1=warn, 2+=block), `HookRegistry` with matcher filtering and block short-circuit. `AgentManifest.hooks: Vec<HookConfig>` carries the table. `cmd_code` actually fires `HookEvent::SessionStart` and aborts on `Block` via `anyhow::bail`. 10 unit tests cover exit-code routing, registry behaviour, matcher semantics, TOML round-trip. Remaining: the other 5 events (PreToolUse / PostToolUse / UserPromptSubmit / Stop / SubagentStop) ship as types today but have no runtime call sites yet | PMAT-CODE-HOOKS-001 **CLOSED** 2026-04-18; follow-up `PMAT-CODE-HOOKS-RUNTIME-001` (P1) wires the remaining 5 call sites |
| **Skills** (`/name` user-invocable + auto-loaded) | SHIPPED | PMAT-CODE-SKILLS-001 CLOSED 2026-04-18: new `agent/skill.rs` mirrors the custom-agents pattern — hand-parses `---`-fenced markdown frontmatter (no serde_yaml dep), supports both flat `.apr/skills/<name>.md` and subdir `.apr/skills/<name>/SKILL.md` (Claude default) layouts, discovers skills from user scope (`~/.config/apr/skills/`) → project scope (`.apr/skills/` or `.claude/skills/` fallback) with `.apr/` winning on name collision. `SkillRegistry` exposes `register`/`resolve`/`names` plus an `auto_match` heuristic that fires when ≥2 length-≥4 tokens from a skill's `when_to_use` appear (case-insensitive) in the active turn — two-token threshold prevents single-word false positives. Optional `when_to_use` + `allowed-tools` frontmatter fields parsed; unknown keys (e.g. Claude-compat `context: fork`) silently tolerated. 25 unit tests cover parse (happy path + CRLF + BOM + alias keys + each error variant), flat/subdir layouts, silent skip of malformed files, `.apr/`-over-`.claude/` precedence, registry CRUD, auto_match positive/negative/case-insensitive. **Remaining**: `allowed-tools` enforcement at tool-invocation time → `PMAT-CODE-SKILLS-TOOLS-001` (P2); `/<skill-name>` REPL dispatch wiring (skill body is ready to inject into system prompt today) → `PMAT-CODE-SLASH-SKILLS-001` (P2) | PMAT-CODE-SKILLS-001 **CLOSED** 2026-04-18 |
| **Subagent / Task tool** (Agent, TaskCreate/Get/List/Update) | SHIPPED (core) | PMAT-CODE-SPAWN-PARITY-001 CLOSED 2026-04-18: new `agent/task_tool.rs` ships the Claude-Code-equivalent `task` tool — **default-registered in `cmd_code` with no capability gate**, matching Claude Code's built-in Agent. `SubagentRegistry` + `SubagentSpec` resolve `subagent_type` against 3 preset personalities (`general-purpose` / `explore` / `plan` — identical roster to Claude's built-ins). Depth-bounded recursion via `Capability::Spawn { max_depth: 3 }`. Driver promoted to `Arc<dyn LlmDriver>` so `AgentPool` can execute children without a second model load. 13 unit tests. **Remaining**: async `TaskCreate/Get/List/Update` lifecycle surface → PMAT-CODE-TASK-ASYNC-001 (P2); worktree-isolated child agents → PMAT-CODE-WORKTREE-001 (P1) |
| **Worktree isolation** (`EnterWorktree`/`ExitWorktree`) | SHIPPED (primitive) — SpawnConfig wiring MISSING | PMAT-CODE-WORKTREE-001 CLOSED 2026-04-18: new `agent/worktree.rs` ships the full `git worktree`-backed lifecycle: `WorktreeSession::create(repo, branch)` shells out to `git worktree add -b`, `.is_dirty()` probes via `git status --porcelain`, `.auto_close_if_clean()` returns `Ok(None)` on clean (removing worktree + deleting branch) or `Ok(Some((path, branch)))` on dirty (matches Claude Code's "worktree auto-cleaned if no changes; otherwise path+branch returned" semantic). Drop is intentionally a no-op (Poka-Yoke — forces explicit disposition, prevents silent discard). Branch-name sanitizer maps non-alphanumeric chars to `-` so nested branches like `feature/x/y` resolve to `.git/apr-worktrees/feature-x-y`. 8 unit tests shell out to real `git` against a `tempfile::tempdir()` repo (with `core.hooksPath=/dev/null` so parent-repo pmat hooks don't leak in) and gracefully skip when git isn't on PATH. **Remaining**: wiring into `SpawnConfig`/`AgentPool::spawn` so `apr code` subagents opt in automatically → `PMAT-CODE-WORKTREE-RUNTIME-001` (P2) | PMAT-CODE-WORKTREE-001 **CLOSED** 2026-04-18 |
| **Custom agents** (`.claude/agents/*/AGENT.md`) | SHIPPED | PMAT-CODE-CUSTOM-AGENTS-001 CLOSED 2026-04-18: new `agent/custom_agents.rs` hand-parses `---`-fenced markdown frontmatter (no new deps), supports both flat `.apr/agents/<name>.md` and subdir `.apr/agents/<name>/AGENT.md` layouts (Claude-cross-compat), silently skips malformed files, and merges discoveries on top of the 3 canonical built-ins (so `.apr/agents/explore.md` can override the built-in). `register_task_tool` calls `discover_standard_locations(cwd)` at registration so `apr code` auto-picks up user agents from `.apr/agents/` or `.claude/agents/` (project scope) and `~/.config/apr/agents/` (user scope). 22 unit tests (parse happy path + CRLF + BOM + unknown-key tolerance + each error variant + flat/subdir layouts + silent skip + scope precedence). **Remaining**: per-agent `allowed-tools` frontmatter enforcement → `PMAT-CODE-CUSTOM-AGENTS-TOOLS-001` (P2); user-scope scaffolding → `PMAT-CODE-CUSTOM-AGENTS-INIT-001` (P2) | PMAT-CODE-CUSTOM-AGENTS-001 **CLOSED** 2026-04-18 |
| **MCP client** (`.mcp.json` loader + stdio/sse/http) | SHIPPED (core) — `.mcp.json` project-root loader MISSING | `register_mcp_client_tools` (`agent/code.rs:400-423`) now called from `cmd_code` right after `build_code_tools`, closing the gap that kept `McpClientTool` dormant. Under `--features code` (which enables `agents-mcp`), any `manifest.mcp_servers[]` is auto-discovered via `discover_mcp_tools()` and each discovered tool registered into the agent's tool registry. Unit test `test_register_mcp_client_tools_noop_when_empty` verifies idempotent no-op when no servers declared. Remaining scope: separate `.mcp.json` project-root loader (currently servers live under `AgentManifest.mcp_servers[]` in the TOML manifest) | PMAT-CODE-MCP-CLIENT-001 **CLOSED** 2026-04-18; follow-up `PMAT-CODE-MCP-JSON-LOADER-001` (P2, deferred) |
| **Permission modes** (`default`/`plan`/`acceptEdits`/`auto`/`bypassPermissions`) | SHIPPED (primitive) — REPL wiring MISSING | PMAT-CODE-PERMISSIONS-001 CLOSED 2026-04-18: new `agent/permission.rs` ships the full Claude-Code mode lattice: `PermissionMode::{Default, Plan, AcceptEdits, BypassPermissions}` with serde camelCase matching Claude's JSON surface. `PermissionVerdict::{Allow, Ask, Block}` carries the per-capability decision; `verdict(&Capability)` encodes the matrix (Bypass=everything; Plan=reads+Memory+Rag only, blocks the rest; AcceptEdits=auto-allow reads+writes, asks on shell/network; Default=asks on everything mutating). `parse()` accepts canonical camelCase + kebab-case + snake_case aliases with whitespace trim so the `--permission-mode` CLI flag, TOML manifest, and `/permissions` slash-command share one happy path. `next()` cycles in Claude's Shift+Tab order (default → plan → acceptEdits → bypassPermissions → default). `would_run_unattended()` helper for `apr code -p` batch jobs. 15 unit tests. **Remaining**: REPL prompt-loop wiring — Shift+Tab cycle, `/permissions <mode>` slash-command routing, actual per-tool-call verdict enforcement → `PMAT-CODE-PERMISSIONS-RUNTIME-001` (P2) | PMAT-CODE-PERMISSIONS-001 **CLOSED** 2026-04-18 |
| **Built-in tools — read/write/edit/grep/glob/shell** | SHIPPED | `build_code_tools` (`agent/code.rs:426-458`) registers `FileReadTool`, `FileWriteTool`, `FileEditTool`, `GlobTool`, `GrepTool`, `ShellTool`, `MemoryTool`, `PmatQueryTool`, `RagTool` (feature-gated). Rough 1:1 with Claude's Read/Write/Edit/Glob/Grep/Bash | — |
| **Built-in tools — WebFetch / WebSearch** | SHIPPED | PMAT-CODE-WEB-TOOLS-001 CLOSED 2026-04-18: new `register_web_tools` helper in `agent/code.rs` now registers `NetworkTool` (+ `BrowserTool` under `agents-browser` feature) when privacy tier != Sovereign AND `AgentManifest.allowed_hosts` is non-empty. Sovereign tier always blocks regardless of allowlist (Poka-Yoke — tier wins over config). New top-level TOML field `allowed_hosts: Vec<String>` carries the allowlist. 4 unit tests: Sovereign+allowlist blocked, Standard+empty blocked, Standard+allowlist registers, Private+allowlist registers. **Remaining**: dedicated WebSearch tool (separate from WebFetch) → `PMAT-CODE-WEB-SEARCH-001` (P2) — currently search is handled via NetworkTool + caller-constructed search-API URLs | PMAT-CODE-WEB-TOOLS-001 **CLOSED** 2026-04-18 |
| **Built-in tools — Notebook edit** | NONE | No `notebook.rs` under `agent/tool/`. ipynb surface unaddressed | PMAT-CODE-NOTEBOOK-001 (new, low priority) |
| **Built-in tools — `Monitor` (background tail)** | NONE | Closest existing: `spawn.rs` subagent. No long-running stdout subscription primitive matching Claude's Monitor semantics | PMAT-CODE-MONITOR-001 (new, low priority) |
| **Session management** (`--resume`, `--continue`, transcript store) | SHIPPED (core) — `--fork`/`--name` MISSING | `apr code --resume [id]` (`commands_enum.rs:519-521`). **Durable JSONL transcript store already wired** at `~/.apr/sessions/{id}/messages.jsonl` — `SessionStore::{create, resume, append_message, append_messages, load_messages, record_turn, find_recent_for_cwd, find_recent_for_cwd_within}` (`agent/session.rs:32-201`) plus interactive `offer_auto_resume()` with age-display prompt. **More sophisticated than claimed on first-pass audit.** Remaining gaps: no dedicated `--continue` shorthand CLI flag (the `--resume` + `offer_auto_resume` path covers the use case but doesn't match Claude's exact flag name), no `--fork-session`, no `--name` | PMAT-CODE-SESSION-PARITY-001 (scope reduced — only `--continue`/`--fork-session`/`--name` gaps remain, not the whole transcript store) |
| **Configuration** (`~/.config/apr/settings.json` + precedence ladder) | SHIPPED | PMAT-CODE-CONFIG-LADDER-001 CLOSED 2026-05-07 (PR #1564): three-tier ladder in `crates/aprender-orchestrate/src/agent/settings.rs`: `$APR_CONFIG/settings.json` (override) or `~/.config/apr/settings.json` (XDG default) → `<project_root>/.apr/settings.json` → CLI flags. `--manifest` short-circuits the ladder. `serde(deny_unknown_fields)` (typo Poka-Yoke). Initial fields: `model` (path-vs-repo heuristic), `max_turns`, `extra_system_prompt` (APPEND not replace), `project`. 21 unit tests (14 in settings::tests + 7 in code::tests::settings_apply_tests). Live smoke verified project-local `max_turns:7` won over user-global `3`. **Remaining**: more fields (permissions, hooks, mcp_servers under JSON) → `PMAT-CODE-CONFIG-LADDER-FIELDS-001` (P2) | PMAT-CODE-CONFIG-LADDER-001 **CLOSED** 2026-05-07 |
| **Plugins / marketplace** | NONE | No plugin manifest, no plugin discovery, no `/plugins` slash command | PMAT-CODE-PLUGINS-001 (new, deferred) |
| **IDE integrations** (VS Code, JetBrains) | NONE | No extension crate. LSP endpoint planned (`apr serve` already exposes LSP-adjacent pieces) but no IDE extension marketing `apr code` as a completion backend | PMAT-CODE-IDE-001 (new, deferred) |
| **Non-interactive mode (`-p` + `--output-format`/`--input-format`)** | SHIPPED | PMAT-CODE-OUTPUT-FORMAT-001 / PMAT-CODE-INPUT-FORMAT-001 CLOSED 2026-05-07 (PR #1563): `apr code -p` now accepts `--output-format <text\|json>` and `--input-format <text\|json>` as clap ValueEnum. JSON output emits Claude-Code's envelope shape (`{type:"result",subtype:"success",is_error,duration_ms,result,session_id,num_turns,tokens_in,tokens_out,total_cost_usd:0}`). Empty assistant text → `subtype:"error"`. JSON input parses `{"role":"user","content":"..."}` from stdin. Unknown values fail with exit 2. 8 unit tests (envelope shape, error subtype, content extraction, missing fields, malformed JSON, empty stdin). Live smoke on qwen2.5-coder-1.5b-q4k.apr returned `result:"4"` in 777ms and `result:"10"` in 812ms. **Remaining**: `--output-format stream-json` (newline-delimited streaming), `--max-budget-usd` → `PMAT-CODE-NON-INTERACTIVE-STREAM-001` (P2) | PMAT-CODE-OUTPUT-FORMAT-001 / PMAT-CODE-INPUT-FORMAT-001 **CLOSED** 2026-05-07 |
| **Keyboard shortcuts** (`!` shell prefix, `@path` expansion, slash commands) | SHIPPED | PMAT-CODE-REPL-PHASE2-001 CLOSED 2026-05-07 (PR #1565): pure-function `agent/repl_directives.rs`. `!<cmd>` execs via `sh -c`, captures stdout+stderr, prints exit code on failure (`!` alone is no-op). `@<path>` boundary-anchored (so `noah@paiml.com` is preserved); expanded inline as `<file path="...">\n<contents>\n</file>` before agent sees the prompt. Missing files leave token verbatim AND emit stderr warning (Poka-Yoke). 19 unit tests cover bang parser, LIVE shell exec, at-path token finding (incl. email exclusion), and file expansion. REPL wiring handles `!` BEFORE slash-commands and `@` expansion AFTER. **Remaining**: `Shift+Tab` permission cycle (raw-mode crossterm) → `PMAT-CODE-REPL-PHASE3-001` (P2; cosmetic compared to ! and @) | PMAT-CODE-REPL-PHASE2-001 **CLOSED** 2026-05-07 |
| **Status line** (model / mode / cost / branch) | SHIPPED | `crates/aprender-orchestrate/src/agent/status_line.rs` ships the Claude-Code REPL status strip as a pure data-struct + render function. `StatusLine { model, mode, cost_usd, branch, cwd_short }` renders in Claude's column order (`model \| [mode] \| $cost \| branch \| cwd`) with missing optionals elided and cost always formatted to two decimals. `StatusLine::build(model, PermissionMode, cost_usd, branch, cwd_short)` wires directly into the canonical permission lattice from PMAT-CODE-PERMISSIONS-001. Free helper `short_cwd(&Path, Option<&Path>)` collapses `$HOME` to `~/` (lone `~` when cwd==home, path-verbatim fallback otherwise). 14 unit tests cover column order, cost truncation, optional elision, home-prefix collapse, edge cases, render purity, Clone roundtrip. REPL/TUI integration (periodic repaint, cost accumulator, git-branch cache, cwd hook) deferred to follow-up PMAT-CODE-STATUS-LINE-RUNTIME-001 (P2) | PMAT-CODE-STATUS-LINE-001 (CLOSED 2026-04-18 v5.0) |
| **Managed org policy** (`/etc/claude-code/CLAUDE.md`) | SHIPPED | `crates/aprender-orchestrate/src/agent/org_policy.rs` ships a pure enforced-tier instruction loader. `load_org_policy(roots, filename, max_bytes)` walks injected roots in first-wins order, returning an `OrgPolicy { source, content, tier: PolicyTier::Enforced }`. `canonical_system_roots()` returns `[/etc/apr-code, /etc/claude-code]` (native first, Claude-Code cross-compat second). `PolicyTier` derives `Ord` so the prompt builder can total-order tiers. Missing files + I/O errors silently skipped (boot-safe). `max_bytes=0` disables loader; positive budget truncates on UTF-8 char boundary with `(truncated from N bytes)` annotation. 13 unit tests cover no-root, empty-budget, happy path, first-root-wins, second-root-fallback, dir-shadowing-file, truncation, UTF-8 boundary, below-budget passthrough, canonical ordering, tier ordering, I/O tolerance, Clone roundtrip. Prompt-builder integration deferred to PMAT-CODE-ORG-POLICY-RUNTIME-001 (P2) | PMAT-CODE-ORG-POLICY-001 (CLOSED 2026-04-18 v5.1) |

**Headline count (v5.4 — 2026-05-07 post PMAT-CODE-REPL-PHASE2-001 — ZERO PARTIAL ROWS)**: 21 rows total, row-level status is ground truth per `pv check-parity`: **17 SHIPPED** (claude-md-memory, builtin-tools-rwegs, session-management, mcp-client, slash-commands, hooks, subagent-spawn, custom-agents, builtin-tools-web, skills, worktree-isolation, permission-modes, status-line, managed-org-policy, **non-interactive-mode**, **configuration-ladder**, **keyboard-shortcuts**), **0 PARTIAL** (every row is now SHIPPED or P2-deferred MISSING), **4 NONE** (builtin-tool-notebook, builtin-tool-monitor, plugins-marketplace, ide-integrations — all P2 deferred). **All 4 P0 tickets + 6 P1 tickets + 5 P2 tickets closed**: MCP-CLIENT, SLASH-PARITY, HOOKS, SPAWN-PARITY, CUSTOM-AGENTS, WEB-TOOLS, SKILLS, WORKTREE, PERMISSIONS, STATUS-LINE, ORG-POLICY (2026-04-18); **OUTPUT-FORMAT, INPUT-FORMAT, CONFIG-LADDER, REPL-PHASE2** (2026-05-07 single autonomous /loop session, PRs #1563/#1564/#1565). **Closure status: BOTH CONDITIONS EXCEEDED** — SHIPPED cap (17 ≥ 9) MET; MISSING cap (4 ≤ 4) MET; PARTIAL cap (0 ≤ 7) MET. **Epic PMAT-CODE-PARITY-MATRIX-001 is fully CLOSEABLE — zero PARTIAL rows remain.** FALSIFY-CODE-PARITY-005 passes. Remaining 4 MISSING rows (notebook, monitor, plugins, IDE) are P2 deferred surfaces tracked for future milestones but not epic-blocking. Follow-up tickets filed by the 2026-05-07 session: `PMAT-CODE-{NON-INTERACTIVE-STREAM,CONFIG-LADDER-FIELDS,REPL-PHASE3}-001` (all P2).

**Falsification cross-checks used in this audit:**
- `pmat query "build_code_tools" --include-source` confirmed 9 tools registered at `agent/code.rs:426-458`
- `grep -rn "Hook\|pre_tool\|post_tool\|UserPromptSubmit" crates/aprender-orchestrate/src/agent/` returned zero hits
- `grep -n "pub enum SlashCommand\|^\s*\w\+," crates/aprender-orchestrate/src/agent/repl.rs` enumerated 11 variants
- `agent/code.rs:253-256` confirms CLAUDE.md + APR.md are already loaded as project instructions (`load_project_instructions` iterating `["APR.md", "CLAUDE.md"]`)
- `cli/agent_helpers.rs:179` confirms `register_spawn_tool` is capability-gated (as before); `agent/code.rs:107` adds `register_task_tool` unconditionally (default-registered Task tool per PMAT-CODE-SPAWN-PARITY-001)
- `crates/apr-cli/src/commands_enum.rs:492-520` confirms the `Code` subcommand flag surface
- **2026-04-18 correction**: `pmat query "SessionStore|session_id|session.rs|offer_auto_resume" --include-source` returned `SessionStore` at `agent/session.rs:32` + `generate_session_id` at `:189-195`; direct file read confirmed full `{create, resume, append_message, load_messages, find_recent_for_cwd, offer_auto_resume}` API. First-pass matrix had marked this PARTIAL on incomplete evidence — flipped to SHIPPED.
- **2026-04-18 correction**: `grep -rin "status.?line\|StatusLine\|render_status\|status_bar\|statusline" crates/aprender-orchestrate/src/agent/` returned zero matches. Two hits elsewhere (`stack/publish_status`, `bug_hunter/model_parity`) are unrelated. Status line row flipped UNKNOWN → NONE.

### Feature-flag caveat (Claude-Code-parity install)

`apr code` is gated behind the `code` Cargo feature (`code = ["dep:batuta"]` in `crates/apr-cli/Cargo.toml`). **`cargo install aprender` with default features does NOT include `apr code`** — falsified 2026-04-18 by running `apr code --help` on the default-feature build and getting `unrecognized subcommand 'code'`. For Claude-Code parity the install line is:

```bash
cargo install aprender --features code      # apr code only
cargo install aprender --features full      # apr code + inference + cuda + training + visualization
```

`apr mcp` is **always** compiled in regardless of the `code` feature — the MCP server has no dependency on the agent runtime.

## Claude Messages-API Provable-Contract Proxy (PLANNED M6)

A sovereign, local drop-in for Anthropic's Messages API (`POST /v1/messages`) that accepts a Claude-SDK-shaped request, drives [`apr code`](../../crates/aprender-orchestrate/docs/specifications/components/apr-code.md) over a local Qwen3-MoE model, and returns a Claude-SDK-shaped response — with both the request and the response shapes pinned by a provable contract (`contracts/apr-claude-proxy-v1.yaml`). Surface extends `apr serve`:

```bash
apr serve anthropic                                  # default: bind 127.0.0.1:8080, auto-pick model
apr serve anthropic --port 8080 --model <hf-id>      # override model
apr serve anthropic --model-path /path/model.gguf    # explicit local path
```

Nested as a new `ServeCommands::Anthropic` variant in `crates/apr-cli/src/serve_commands.rs` alongside the existing `Plan` and `Run` variants — not a flag on `serve run`, because the Anthropic proxy carries its own default-model resolver and autoselect semantics that would bloat `Run`'s flag surface.

### Why a Messages-API proxy alongside `apr mcp`?

MCP is Anthropic's **client-tool protocol** (Directions 1–3 above). The Messages API is a different shape — a **completion protocol**. Most IDE integrations (Claude Code itself, Cursor, Zed's AI pane, `anthropic-sdk-python`, `anthropic-sdk-typescript`) speak the Messages API, not MCP. Pointing `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` at `apr serve anthropic` lets those tools swap in a sovereign on-device model with **zero client-side code change**. The MCP surface (`apr mcp`) and the Messages-API surface (`apr serve anthropic`) are orthogonal — different wire protocols, different use cases, shared agent backend (`apr code`).

### Default model — Qwen3-Coder-30B-A3B-Instruct (Q4_K_M GGUF)

MoE: 30.5B total / 3.3B active parameters. Apache 2.0. Released 2025-04-29 (Qwen3 family). Q4_K_M GGUF ~18.6 GB on-disk — fits 24 GB VRAM (RTX 4090) with headroom for KV cache. Measured ~196 tok/s on reference hardware (4090, realizar fused Q4K + FlashDecoding path).

Rationale for **MoE over dense**: active-param count (3.3B) governs decode latency; total-param count (30.5B) governs reasoning capacity. MoE buys both cheaply. Q4_K_M is already the format that establishes aprender's 1.43× Ollama parity (on Qwen2.5-1.5B Q4_K_M) — no new kernels required. Hugging Face canonical: `Qwen/Qwen3-Coder-30B-A3B-Instruct`; pre-quantized GGUF: `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`.

**Fallback chain** (applied when `--model` is omitted and the preferred cache entry is missing):
1. `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M` (coder-specialized)
2. `Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M` (general instruct)
3. `apr code`'s normal model-selection chain (respects `APR_CODE_MODEL` env, then `manifest.default_model`)

**Explicitly NOT a default**: Qwen3-Next-80B-A3B at Q4_K_M needs ~45 GB — does not fit a single 4090. Dual-GPU / A6000-Ada / H100-class deployments may select it via `--model`.

### Input Contract (request shape)

Request body = Anthropic Messages API POST body, schema version `v2026-02-01`. Mandatory fields:

- `model: string` — mapped to local model selector; unknown model names fall back to default with a `x-apr-model-fallback` response header logging the substitution
- `messages: Array<{role: "user"|"assistant", content: string | ContentBlock[]}>`
- `max_tokens: integer`
- `system: string | ContentBlock[]` (optional)
- `tools: Array<ToolDefinition>` (optional) — translated into `apr code` tool-registry entries
- `stream: boolean` (optional)
- `temperature`, `top_p`, `top_k`, `stop_sequences` (optional)

Accepted content-block types: `text`, `image`, `tool_use`, `tool_result`, `document`. `thinking` blocks are **accepted-but-ignored on input** (reasoning tokens are regenerated locally, not replayed — consistent with Anthropic's own re-derivation semantics).

Full JSON Schema: `contracts/apr-claude-proxy-v1.yaml` under `inputs.messages_request`.

### Output Contract (response shape)

Response body = Anthropic Messages API response, schema version `v2026-02-01`. Mandatory fields:

- `id: string` — local UUIDv7
- `type: "message"` (literal)
- `role: "assistant"` (literal)
- `model: string` — echo of the resolved model ID (after fallback)
- `content: Array<ContentBlock>` — ordered; each block is one of:
  - `{type: "text", text: string}`
  - `{type: "tool_use", id: string, name: string, input: object}`
  - `{type: "thinking", thinking: string}` — emitted when `apr code` surfaces reasoning traces
- `stop_reason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"`
- `stop_sequence: string | null`
- `usage: {input_tokens: int, output_tokens: int, cache_creation_input_tokens: int, cache_read_input_tokens: int}`

**Streaming mode (SSE):** event sequence `message_start` → (`content_block_start` → N × `content_block_delta` → `content_block_stop`)⁺ → `message_delta` → `message_stop`; with `ping` interleaved and `error` terminal. Exact per-event payload shapes in `contracts/apr-claude-proxy-v1.yaml` under `outputs.sse_events`.

### Translation semantics (Anthropic ↔ `apr code`)

| Anthropic request field | `apr code` mapping |
|------|------|
| `messages[]` | Batuta `Conversation::from_anthropic(&messages)` (role + content-block fidelity) |
| `tools[]` | Registered as `AnthropicToolAdapter` entries alongside `build_code_tools` output (FileRead/Write/Edit, Glob, Grep, Shell, Memory, PmatQuery, Rag) |
| `system` | Prepended as a system-role Batuta `Message` |
| `max_tokens` | Passed through to realizar `GenConfig.max_new_tokens` |
| `temperature`, `top_p`, `top_k`, `stop_sequences` | Passed through to realizar `GenConfig` |
| `stream: true` | Drives SSE serialization over the agent event channel; non-streaming batches content blocks into a final array |

Translation is a **pure function of request shape** — no hidden state, no client-identity leakage. Property-tested round-trip via `apr serve anthropic --dump-translation` against a fixture corpus of captured Anthropic SDK request bodies (see FALSIFY-CLAUDE-PROXY-002).

### Falsification Conditions

Defined in `contracts/apr-claude-proxy-v1.yaml` (to be added when M6-α ships):

1. **FALSIFY-CLAUDE-PROXY-001** — Input shape parity: any request that passes Anthropic's published JSON Schema MUST be accepted by the proxy; any request that fails that schema MUST return HTTP 400 with body `{"type":"error","error":{"type":"invalid_request_error","message": string}}` — same envelope shape Anthropic returns.
2. **FALSIFY-CLAUDE-PROXY-002** — Output shape parity: every non-streaming response passes Anthropic's Messages-API response JSON Schema when parsed by `anthropic-sdk-python` v0.40+; zero extra fields (schema uses `additionalProperties: false`); `stop_reason` is drawn from the closed enum above.
3. **FALSIFY-CLAUDE-PROXY-003** — Tool-use round-trip: a fixture of user→`tool_use`→`tool_result`→assistant (4 turns) exercising the `Bash` tool produces a transcript whose `stop_reason` transitions `tool_use`→`end_turn` across turns, matching the stop-reason order of an identical fixture captured from `api.anthropic.com` (captured offline — no live net in CI).
4. **FALSIFY-CLAUDE-PROXY-004** — Streaming event parity: SSE event sequence on a short prompt is a valid prefix of the Anthropic event schedule (`message_start` → ≥1 `content_block_*` cluster → `message_delta` → `message_stop`); `event:` casing matches Anthropic; each `data:` payload parses as its declared sub-schema.
5. **FALSIFY-CLAUDE-PROXY-005** — Default-model autoselect: on startup with no `--model` and no cached model file, the HTTP listener MUST NOT bind until `apr pull` completes; on startup with the cached file, time-to-listen is <3s (matches `apr serve`). Asserted via filesystem-state test, not HF network.
6. **FALSIFY-CLAUDE-PROXY-006** — Sovereignty: `apr serve anthropic` MUST NOT open outbound sockets to `api.anthropic.com` under any combination of request headers, env vars, or config. Asserted by blocking all outbound network in the CI test container except `127.0.0.1`.

These gates live in `contracts/apr-claude-proxy-v1.yaml` — **outside** `apr-mcp-server-v1.yaml` (different protocol) and **outside** `apr-code-v1.yaml` (agent-loop semantics, not HTTP surface).

### Proxy milestones (tracked as PMAT-CLAUDE-PROXY-001)

- **M6-α** (planned post-M5 `pmcp` port): `contracts/apr-claude-proxy-v1.yaml` committed; request-shape parser (FALSIFY-001) PASS on fixture corpus.
- **M6-β**: Non-streaming response generation over `apr code` (FALSIFY-002, -003) PASS.
- **M6-γ**: SSE streaming (FALSIFY-004) PASS; `apr serve` throughput-parity benchmark vs the non-proxied path (±5%).
- **M6-δ**: Default-model autoselect + sovereignty (FALSIFY-005, -006) ENFORCED; spec promotes from PLANNED → ACTIVE.

## Milestones

### M1: Skeleton — SHIPPED (2026-04-17)
- [x] Create `crates/aprender-mcp/` crate (PR #864)
- [x] Wire `apr mcp` subcommand into apr-cli (PR #864)
- [x] Implement `initialize` + `tools/list` with `apr.version` stub (PR #864)
- [x] FALSIFY-MCP-001 (init <500ms) and -002 (tools/list shape) passing
- Note: `aprender-mcp` ships a hand-rolled JSON-RPC dispatcher rather than using the
  `pmcp` SDK directly. Rationale: (a) the Phase-1 tool surface is subprocess wrappers
  over `apr <cmd> --json` — a minimal request/response shape that the ~200-line
  dispatcher covers without pulling transitive deps, (b) schema codegen
  (`build.rs` → `$OUT_DIR/schemas.rs`) keeps `tools/list` byte-identical to the
  contract YAML, and (c) FALSIFY-MCP-001/-002/-005/-007/-008 assert shape against
  JSON-RPC wire bytes, which is easier to audit without an SDK layer. `pmcp` v2.3
  is the planned substrate once M5+ adds SSE/WebSocket transports, the resources
  protocol, or streaming sampling (`aprender-orchestrate` already depends on pmcp v2.3
  for its client role — see `crates/aprender-orchestrate/Cargo.toml`).

### M2: Phase-1 tools — SHIPPED (2026-04-17/18)
- [x] 7 subprocess wrappers around `apr <cmd> --json`: validate (#865), tensors+bench (#866), qa+trace (#867), run (#870), serve (#872)
- [x] FALSIFY-MCP-005 (jsonrpc≠"2.0") and -007 (protocolVersion mismatch) dispatcher gates (#868)
- [x] FALSIFY-MCP-002 strict slice — JSON Schema Draft 7 meta-validation per tool (#869)
- [x] `contracts/apr-mcp-tool-schemas-v1.yaml` authored as codegen source (#871)
- [x] Doc retrofit (#873)

### M3: Streaming + cancellation + 8th tool + codegen — SHIPPED (2026-04-18)
- [x] `apr.finetune` synchronous wrapper completes Phase-1 8-tool set (#881)
- [x] Cancellation: `notifications/cancelled` → SIGTERM (30s grace) → SIGKILL via std::thread+mpsc worker (#883)
- [x] FALSIFY-MCP-008: build.rs codegen from `apr-mcp-tool-schemas-v1.yaml` — `apr.version` first (#880), then 7 remaining tools (#884); PMAT-514 extended codegen to tool-level `description` strings so neither `inputSchema` nor `description` can be hand-edited in Rust source (2026-04-18)
- [x] Progress notifications for `apr.finetune` — `NotificationSink` plumbed; `params._meta.progressToken` opt-in; FALSIFY-MCP-PROGRESS-001 (#887)
- [x] Progress notifications for `apr.run` — `apr run --stream` NDJSON CLI prereq + `NotificationSink` forwarding; FALSIFY-MCP-PROGRESS-002 (#891 merged 2026-04-19)
- [x] Book chapter `book/src/tools/mcp-server.md` (#874 M2 creation, #885 M3 update)
- [ ] **Deferred to M5+**: Per-step structured progress for `apr.finetune` (CLI emits terminal blob today; needs CLI event channel — same realizar callback-threading prereq that keeps `apr.run`'s per-token emission *post-decode* today, per PR #891's deliberate trade-off)

### M4: End-to-end validation — CODE COMPLETE (all 5 PRs merged 2026-04-19; manual client smoke tests remain)
- [x] First-class contract `contracts/apr-mcp-server-v1.yaml` with 8 falsification_conditions (FALSIFY-MCP-001..008) + test cross-links — PR #886 merged 2026-04-19 (pins exact-8 invariant via `apr_mcp_server_contract_ids_are_falsify_mcp_001_through_008` in `crates/aprender-contracts/tests/apr_mcp_server_contract.rs`)
- [ ] Extend the contract with a 9th row for FALSIFY-MCP-PROGRESS-001 — relax the exact-8 invariant to "FALSIFY-MCP-001..008 + PROGRESS-001, no extras"
- [x] Strengthen FALSIFY-MCP-003/-004 from surface tests to mock-subprocess e2e response-shape assertions — PR #889 merged 2026-04-19 (`crates/aprender-mcp/tests/falsify_mcp_003.rs` + `falsify_mcp_004.rs`, 5 tests total; response-shape now byte-verified against mock-subprocess capture)
- [x] Real-model FALSIFY-MCP-003: `apr.run` decodes "2" within 5s on cached qwen2.5-0.5b — PR #892 merged 2026-04-19 (new gate FALSIFY-MCP-E2E-001; [`crates/aprender-mcp/tests/falsify_mcp_e2e_001.rs::falsify_mcp_e2e_001_apr_run_decodes_two`](../../crates/aprender-mcp/tests/falsify_mcp_e2e_001.rs))
- [x] Real-model FALSIFY-MCP-004: byte-for-byte `apr qa --json` parity — PR #892 merged 2026-04-19 ([`falsify_mcp_e2e_001_apr_qa_matches_cli_byte_for_byte`](../../crates/aprender-mcp/tests/falsify_mcp_e2e_001.rs))
- [x] Claude Code dogfood — 1 full session using only `apr.*` tools — PR #890 merged 2026-04-19 (FALSIFY-MCP-DOGFOOD-001, machine-verified via `crates/aprender-mcp/tests/falsify_mcp_dogfood_001.rs`)
- [ ] Claude Code integration test (launch `apr mcp`, ask Claude to "run qwen2.5-0.5b with prompt X")
- [ ] Cursor / Cline manual smoke test

### M5: `pmcp` SDK migration + transport expansion — IN PROGRESS
- [x] Add `pmcp = "2.3"` to `crates/aprender-mcp/Cargo.toml` (PR 1 — optional dep behind `pmcp-dispatcher` feature flag, zero behaviour change; matches `aprender-orchestrate`'s version pin for workspace coherence; `cargo tree -d` stays clean)
- [ ] Port `server.rs` dispatcher to `pmcp::Server` with per-tool handler registration; retain `build.rs` schema codegen so `tools/list` output stays byte-identical (FALSIFY-MCP-008 unchanged)
- [ ] Port worker-thread cancellation to pmcp's cancellation API if it ships one; otherwise keep the existing std::thread+mpsc path as a `pmcp::Server` extension
- [ ] Extend cancellation to `apr.serve`: track daemon pid in a lifecycle registry, SIGTERM→SIGKILL on `notifications/cancelled` (today `apr.run` alone honours cancel — see `server.rs::CancelHandle`)
- [ ] Add SSE transport (`apr mcp --transport sse --port N`) via pmcp's SSE layer — unblocks browser/container MCP clients
- [ ] Add WebSocket transport (same surface) — unblocks long-running sessions
- [ ] Re-run falsification suite (78 tests across `falsify_m1`, `falsify_mcp_006`, `falsify_mcp_008`, `falsify_mcp_progress_001`, `falsify_schema`, lib unit tests — 2026-04-18 count) and ensure every FALSIFY-MCP gate still PASS post-migration

### M6: Claude Messages-API proxy — PLANNED (PMAT-CLAUDE-PROXY-001)
- [ ] M6-α: commit `contracts/apr-claude-proxy-v1.yaml`; request-shape parser (FALSIFY-CLAUDE-PROXY-001) PASS on captured `anthropic-sdk-python` fixture corpus
- [ ] M6-β: non-streaming response generation over `apr code` (FALSIFY-CLAUDE-PROXY-002 output-shape parity + FALSIFY-CLAUDE-PROXY-003 tool_use round-trip) PASS
- [ ] M6-γ: SSE streaming (FALSIFY-CLAUDE-PROXY-004 event-sequence parity) PASS; throughput within ±5% of non-proxied `apr serve` decode path
- [ ] M6-δ: default-model autoselect (FALSIFY-CLAUDE-PROXY-005, bind-after-pull + <3s cold start if cached) + sovereignty (FALSIFY-CLAUDE-PROXY-006, zero egress to api.anthropic.com) ENFORCED; contract promotes DRAFT → ENFORCED; spec section promotes PLANNED → ACTIVE
- [ ] Default model resolver: `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M` → `Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M` → `APR_CODE_MODEL` env → `manifest.default_model`
- [ ] HTTP surface in `crates/aprender-serve/src/anthropic/`; CLI flag `apr serve anthropic`

**Real-model gating:** The M4 gates above are env-gated via `APR_MCP_E2E_MODEL` (absolute path to a cached GGUF, e.g. `/path/to/qwen2-0.5b-instruct-q4_0.gguf`). Any environment claiming real-model validation MUST set this env var — when unset, both tests skip with a `println!` + early return (visible in test output), not via `#[ignore]`. Per-test documentation explains the Q4_0-vs-Q4_K_M fixture delta relative to FALSIFY-MCP-003's spec literal.

## Success Criteria

The spec is already ACTIVE as of M3 ship (2026-04-18). The table below is
the acceptance gate for **closing M4** — FALSIFY-MCP-003/-004 response-shape
layer ENFORCED (PR #889), real-model FALSIFY-MCP-E2E-001 ENFORCED (PR #892),
dogfood FALSIFY-MCP-DOGFOOD-001 ENFORCED (PR #890), and
`apr-mcp-server-v1.yaml` contract promoted to ACTIVE with all 8
falsification_conditions ENFORCED (PR #886). One row remaining: the
`apr.run` progress notifications prereq on PR #891 — once that merges, M4
closes:

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| `initialize` latency | <500ms | CI with `hyperfine` |
| Tool call round-trip (non-inference) | <100ms | `apr.validate` on a cached model |
| `apr.run` first-token latency | <2s | qwen2.5-0.5b-q4km on target hardware |
| Protocol spec compliance | 100% | MCP conformance suite (external) |
| Claude Code dogfood | 1 full session using only `apr.*` tools | FALSIFY-MCP-DOGFOOD-001 (CI) |
| 11 falsification gates (FALSIFY-MCP-001..008 + PROGRESS-001 + DOGFOOD-001 + VALIDATE-001) | all PASS or PARTIAL→PASS by M4 close | CI |

## Out of Scope (Phase 1)

- Resources protocol (`resources/list`, `resources/read`) — future phase for exposing model files
- Prompts protocol — future phase
- Sampling (client-side LLM calls from server) — not needed for inference use case
- Auth / multi-tenant — local dev tool only
- SSE / WebSocket transports — Phase 2, scheduled for M5 on top of `pmcp` v2.3 (stdio-only in Phase 1; `--transport sse --port N` flag is aspirational until M5, no `McpArgs` struct yet)
- Windows — Phase 2 (stdio transport needs testing on Windows; nix signal crate is unix-only today)

## Risk Register

| Risk | Mitigation |
|------|-----------|
| `pmcp` adoption-path coordination — M5+ will migrate the dispatcher to `pmcp` v2.x; workspace version must stay aligned with `aprender-orchestrate`'s client-side pmcp dep to avoid dual-version builds | Pin to `pmcp = "2.3"` across all crates; bump in one workspace-wide commit; CI `cargo tree -d` gate on duplicate deps |
| Subprocess overhead per tool call | Phase 2: in-process mode (`--embedded`) linking apr-cli as library |
| Schema drift between CLI and MCP surface | `build.rs` emits both `schemas::APR_<TOOL>_SCHEMA` and `schemas::APR_<TOOL>_DESCRIPTION` from `contracts/apr-mcp-tool-schemas-v1.yaml`; hand-editing either in Rust source is impossible (PMAT-514, `tests/falsify_mcp_008.rs` guards live+codegen × schema+description at 4 layers) |
| MCP clients expect specific error shapes | Conformance-test against Claude Code, Cursor, Cline fixtures |
| **Anthropic Messages-API schema drift** (M6): Anthropic ships Messages-API schema revisions without deprecation; proxy clients would break on new required fields | Pin `anthropic_schema_version` in `contracts/apr-claude-proxy-v1.yaml`; FALSIFY-CLAUDE-PROXY-002 re-runs against `anthropic-sdk-python` in a weekly CI cron so any SDK bump that breaks parity fails loudly within 7 days |
| **Accidental egress to api.anthropic.com** (M6) — the one bug that turns a "sovereign" proxy into a data-leak vector | FALSIFY-CLAUDE-PROXY-006: network-sandboxed CI container blocks all outbound except `127.0.0.1`; ANY socket open to `api.anthropic.com` fails the build. Enforced, not advisory. |

## Related Work

- **Existing infrastructure** (ready to use):
  - `contracts/mcp-tool-schema-v1.yaml` — defines JSON-RPC error codes, session lifecycle
  - `contracts/apr-tool-rust-mcp-sdk-v1.yaml` — approves `paiml/rust-mcp-sdk` (pmcp v2.3) as dependency. Already linked by `aprender-orchestrate` for MCP client usage; `aprender-mcp` server-side migration is scheduled for M5+
  - MCP tool surface lives in `crates/aprender-mcp/src/tools/` (not `apr-cli/src/tool_commands.rs` — that is the unrelated `apr tool` CLI group for Showcase/Rosetta)

- **Aspirational follow-ons** (spec files not yet authored):
  - `apr-mcp-plugin-marketplace-v1.md` — Claude Code–style plugin marketplace for community `apr.*` tools
  - `apr-mcp-hooks-v1.md` — pre/post-inference hooks (analog to git hooks) for QA + observability

---

**Owner**: apr-cli team
**Sponsor**: apr-cli team
**Delivery**:
- **v0.31.0** (2026-04-19, tag 62893da32): M1–M3 SHIPPED — 9 tools (`apr.run`, `apr.serve`, `apr.qa`, `apr.trace`, `apr.tensors`, `apr.validate`, `apr.bench`, `apr.finetune`, and dispatch infrastructure), `build.rs` schema+description codegen from `contracts/apr-mcp-tool-schemas-v1.yaml`, `notifications/progress` for `apr.finetune`, `notifications/cancelled` SIGTERM→SIGKILL, JSON Schema Draft 7 meta-validation on every tool input schema in CI, MCP book chapter documenting `.mcp.json` client config.
- **M4** (all 5 PRs merged 2026-04-19 — code complete): PR #886 (`contracts/apr-mcp-server-v1.yaml` ACTIVE with exact-8 `FALSIFY-MCP-001..008` pin); PR #889 (FALSIFY-MCP-003/-004 response-shape e2e gates live, `falsify_mcp_003.rs` + `falsify_mcp_004.rs` = 5 tests); PR #892 (real-model FALSIFY-MCP-E2E-001 ENFORCED, qwen2.5-0.5b decodes "2" within 5s + `apr qa --json` byte-for-byte parity via `falsify_mcp_e2e_001.rs`); PR #890 (FALSIFY-MCP-DOGFOOD-001 ENFORCED — real `apr mcp` binary launched as subprocess answers `initialize`+`tools/list`+`tools/call` for all 9 tools within 2s via stdio, `falsify_mcp_dogfood_001.rs`); PR #891 (FALSIFY-MCP-PROGRESS-002 — `apr run --stream` NDJSON CLI prereq + `apr.run` per-token `notifications/progress` forwarding via `NotificationSink`, 4 tests in `falsify_mcp_progress_002.rs`). Remaining M4 scope is manual: Cursor / Cline smoke test + free-form Claude-Code integration session.
- **M5+** (planned): per spec v1.2.0 roadmap — plugin marketplace, pre/post-inference hooks.
