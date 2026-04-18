# APR-MCP-SERVER: Model Context Protocol Server Specification

**Version**: 1.2.0
**Date**: 2026-04-18
**Status**: ACTIVE (M1–M3 shipped; M4 dogfood pending; M5 pmcp migration planned)
**Contracts**:
- `contracts/mcp-tool-schema-v1.yaml` — MCP tool registration, schema fidelity, session lifecycle, error mapping (existing)
- `contracts/pmcp/mcp-protocol-sdk-v1.yaml` — `pmcp` crate contract (existing)
- `contracts/apr-tool-rust-mcp-sdk-v1.yaml` — `paiml/rust-mcp-sdk` dependency contract (existing)
- `contracts/apr-cli-commands-v1.yaml` — 57-command tool surface (existing)
- **Pending (PR #886)**: `contracts/apr-mcp-server-v1.yaml` — end-to-end MCP server contract (not yet in-tree; promotes to M4)
**References**:
- [Model Context Protocol Specification v2024-11-05](https://spec.modelcontextprotocol.io/specification/2024-11-05/)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
- [pmcp crate](https://github.com/paiml/rust-mcp-sdk) — PAIML's Rust MCP SDK, actively maintained, v2.3.1 on crates.io (2026-04-16)

---

## Problem

Aprender ships a 57-subcommand CLI (`apr`) with structured `--json` output on most commands. It achieves 1.43× Ollama decode perf at 128 tokens. But no agentic tool (Claude Code, Cursor, Cline, Aider, Continue) can invoke it, because aprender does not speak MCP.

Every competitor tool with ecosystem momentum in early 2026 is addressable via MCP — except the local-inference tier. Ollama, llama.cpp, and Unsloth all lack first-party MCP servers. Shipping `apr mcp` first occupies that slot.

## Goal

A single subcommand — `apr mcp` — that starts an MCP server over stdio, exposing a curated subset of the 57 apr CLI commands as MCP tools. Tool schemas are generated at build time from `contracts/apr-mcp-tool-schemas-v1.yaml` (FALSIFY-MCP-008), not hand-written.

Success is measured by Claude Code / Cursor / Cline being able to `.mcp.json`-configure `apr mcp` and have the LLM invoke `apr.run`, `apr.qa`, `apr.trace`, etc. on local models.

## Architecture

### New Crate: `aprender-mcp`

Location: `crates/aprender-mcp/`

```
crates/aprender-mcp/
├── Cargo.toml            # serde, serde_json, anyhow (native); nix (unix signals); serde_yaml (build); jsonschema (dev)
├── build.rs              # FALSIFY-MCP-008: reads contracts/apr-mcp-tool-schemas-v1.yaml → $OUT_DIR/schemas.rs
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
└── tests/                # protocol-level integration tests
    ├── falsify_m1.rs         # FALSIFY-MCP-001 init + -002 tools/list
    ├── falsify_mcp_006.rs    # FALSIFY-MCP-006 notifications/cancelled
    ├── falsify_mcp_008.rs    # FALSIFY-MCP-008 codegen byte-identity
    └── falsify_schema.rs     # JSON Schema Draft 7 meta-validation
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

| Tool | Maps to CLI | Inputs | Output |
|------|-------------|--------|--------|
| `apr.run` | `apr run <model>` | `model_path`, `prompt`, `max_tokens`, `temperature`, `top_p` | `{model, text, tokens: [u32], tokens_generated, max_tokens, tok_per_sec, inference_time_ms, used_gpu, cached}` (CLI as of 2026-04-18; `stop_reason` not emitted) |
| `apr.serve` | `apr serve <model>` | `model_path`, `port` | `{pid, url}` + lifecycle |
| `apr.qa` | `apr qa <model> --json` | `model_path`, `assert_tps?` | `{model, passed, gates: [{name, passed, message, value?, threshold?, duration_ms, skipped}], gates_executed, gates_skipped, total_duration_ms, timestamp, summary}` (CLI as of 2026-04-18; gate field is `passed` not `pass`) |
| `apr.trace` | `apr trace <model> --prompt X` | `model_path`, `prompt`, `steps?` | per-layer tensor stats |
| `apr.tensors` | `apr tensors <model> --json` | `model_path` | tensor list with shapes/dtypes/stats |
| `apr.validate` | `apr validate <model> --quality` | `model_path` | integrity + quality gates |
| `apr.bench` | `apr bench <model>` | `model_path`, `runs`, `tokens` | median tok/s, p50/p95/p99 latency |
| `apr.finetune` | `apr finetune` | `base_model`, `dataset`, `lora_rank`, `epochs` | progress events + final checkpoint path |

Schema generation: each tool's JSONSchema is derived from the entry in `contracts/apr-mcp-tool-schemas-v1.yaml` at build time by `aprender-mcp`'s `build.rs` — **no hand-maintained schemas**. Codegen lands `pub const APR_<TOOL>_SCHEMA: &str` per tool in `$OUT_DIR/schemas.rs`.

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
3. **FALSIFY-MCP-003**: `tools/call apr.run` on `qwen2.5-0.5b-instruct-q4km.gguf` with prompt "1+1=" decodes "2" as first token within 5s. **PARTIAL** — surface tests ENFORCED; mock-subprocess e2e in flight; real-model gate deferred to M4 dogfood.
4. **FALSIFY-MCP-004**: `tools/call apr.qa` returns 8 gates with correct pass/fail states matching `apr qa --json` CLI output byte-for-byte. **PARTIAL** — surface tests ENFORCED; byte-for-byte parity deferred to M4 dogfood.
5. **FALSIFY-MCP-005**: Malformed request (`"jsonrpc": "1.0"`) returns JSON-RPC error code `-32600`, does not crash server. **ENFORCED**.
6. **FALSIFY-MCP-006**: `notifications/cancelled` during `apr.run` stops decoding within 30s, returns partial result. **ENFORCED**.
7. **FALSIFY-MCP-007**: Protocol version mismatch (`"protocolVersion": "1999-01-01"`) returns error, does not attempt tools/list. **ENFORCED**.
8. **FALSIFY-MCP-008**: Schema in `tools/list` output is byte-identical to generated schema from `contracts/apr-mcp-tool-schemas-v1.yaml`. **ENFORCED**.
9. **FALSIFY-MCP-PROGRESS-001** (M3 addition): When client supplies `params._meta.progressToken`, `apr.finetune` emits one `notifications/progress` per non-empty stdout line of `apr finetune --json`, all flushed before the final `tools/call` response. Without a token, zero notifications. **ENFORCED**.

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
- [x] FALSIFY-MCP-008: build.rs codegen from `apr-mcp-tool-schemas-v1.yaml` — `apr.version` first (#880), then 7 remaining tools (#884)
- [x] Progress notifications for `apr.finetune` — `NotificationSink` plumbed; `params._meta.progressToken` opt-in; FALSIFY-MCP-PROGRESS-001 (#887)
- [x] Book chapter `book/src/tools/mcp-server.md` (#874 M2 creation, #885 M3 update)
- [ ] **Deferred to M4**: Per-step structured progress for `apr.finetune` (CLI emits terminal blob today; needs CLI event channel)
- [ ] **Deferred to M4**: Progress notifications for `apr.run` (needs `apr run --stream` CLI flag prereq)

### M4: End-to-end validation — IN PROGRESS
- [ ] First-class contract `contracts/apr-mcp-server-v1.yaml` with 8 falsification_conditions (FALSIFY-MCP-001..008) + test cross-links (PR #886 open — pins exact-8 invariant via `apr_mcp_server_contract_ids_are_falsify_mcp_001_through_008`)
- [ ] Extend the contract with a 9th row for FALSIFY-MCP-PROGRESS-001 after PR #886 merges — relax the exact-8 invariant to "FALSIFY-MCP-001..008 + PROGRESS-001, no extras"
- [ ] Strengthen FALSIFY-MCP-003/-004 from surface tests to mock-subprocess e2e response-shape assertions (in flight)
- [ ] Real-model FALSIFY-MCP-003: `apr.run` decodes "2" within 5s on cached qwen2.5-0.5b
- [ ] Real-model FALSIFY-MCP-004: byte-for-byte `apr qa --json` parity
- [ ] Claude Code dogfood — 1 full session using only `apr.*` tools
- [ ] Cursor / Cline manual smoke test

### M5: `pmcp` SDK migration + transport expansion — PLANNED
- [ ] Add `pmcp = "2.3"` to `crates/aprender-mcp/Cargo.toml` (already in `aprender-orchestrate`, keep versions aligned)
- [ ] Port `server.rs` dispatcher to `pmcp::Server` with per-tool handler registration; retain `build.rs` schema codegen so `tools/list` output stays byte-identical (FALSIFY-MCP-008 unchanged)
- [ ] Port worker-thread cancellation to pmcp's cancellation API if it ships one; otherwise keep the existing std::thread+mpsc path as a `pmcp::Server` extension
- [ ] Extend cancellation to `apr.serve`: track daemon pid in a lifecycle registry, SIGTERM→SIGKILL on `notifications/cancelled` (today `apr.run` alone honours cancel — see `server.rs::CancelHandle`)
- [ ] Add SSE transport (`apr mcp --transport sse --port N`) via pmcp's SSE layer — unblocks browser/container MCP clients
- [ ] Add WebSocket transport (same surface) — unblocks long-running sessions
- [ ] Re-run falsification suite (75 tests across `falsify_m1`, `falsify_mcp_006`, `falsify_mcp_008`, `falsify_mcp_progress_001`, `falsify_schema`, lib unit tests — 2026-04-18 count) and ensure every FALSIFY-MCP gate still PASS post-migration

## Success Criteria

Acceptance gate for promoting to ACTIVE:

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| `initialize` latency | <500ms | CI with `hyperfine` |
| Tool call round-trip (non-inference) | <100ms | `apr.validate` on a cached model |
| `apr.run` first-token latency | <2s | qwen2.5-0.5b-q4km on target hardware |
| Protocol spec compliance | 100% | MCP conformance suite (external) |
| Claude Code dogfood | 1 full session using only `apr.*` tools | Manual |
| 9 falsification gates (FALSIFY-MCP-001..008 + PROGRESS-001) | all PASS or PARTIAL→PASS by M4 close | CI |

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
| Schema drift between CLI and MCP surface | `build.rs` fails build if contract YAML differs from generated Rust structs |
| MCP clients expect specific error shapes | Conformance-test against Claude Code, Cursor, Cline fixtures |

## Related Work

- **Existing infrastructure** (ready to use):
  - `contracts/mcp-tool-schema-v1.yaml` — defines JSON-RPC error codes, session lifecycle
  - `contracts/apr-tool-rust-mcp-sdk-v1.yaml` — approves `paiml/rust-mcp-sdk` (pmcp v2.3) as dependency. Already linked by `aprender-orchestrate` for MCP client usage; `aprender-mcp` server-side migration is scheduled for M5+
  - MCP tool surface lives in `crates/aprender-mcp/src/tools/` (not `apr-cli/src/tool_commands.rs` — that is the unrelated `apr tool` CLI group for Showcase/Rosetta)

- **Aspirational follow-ons** (spec files not yet authored):
  - `apr-mcp-plugin-marketplace-v1.md` — Claude Code–style plugin marketplace for community `apr.*` tools
  - `apr-mcp-hooks-v1.md` — pre/post-inference hooks (analog to git hooks) for QA + observability

---

**Owner**: TBD
**Sponsor**: apr-cli team
**Target**: v0.32.0 (M1–M2), v0.33.0 (M3–M4). Latest released tag at spec write time is `v0.30.0`; M1–M3 are merged on `main` but unreleased, so target tags are the intended publication points, not current reality.
