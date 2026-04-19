# APR-MCP-SERVER: Model Context Protocol Server Specification

**Version**: 1.2.0
**Date**: 2026-04-19 (M1–M3 shipped in v0.31.0; M4 in flight)
**Status**: ACTIVE — `aprender-mcp` ships 9 tools over stdio JSON-RPC 2.0; FALSIFY-MCP-008 ENFORCED at 4 layers (schema+description × live+codegen). M4 open (PRs #886/#889/#890/#891/#892).
**Contracts**:
- `contracts/mcp-tool-schema-v1.yaml` — MCP tool registration, schema fidelity, session lifecycle, error mapping (existing)
- `contracts/pmcp/mcp-protocol-sdk-v1.yaml` — `pmcp` crate contract (existing)
- `contracts/apr-tool-rust-mcp-sdk-v1.yaml` — `paiml/rust-mcp-sdk` dependency contract (existing)
- `contracts/apr-cli-commands-v1.yaml` — 57-command tool surface (existing)
- **New**: `contracts/apr-mcp-server-v1.yaml` — end-to-end MCP server contract
**References**:
- [Model Context Protocol Specification v2024-11-05](https://spec.modelcontextprotocol.io/specification/2024-11-05/)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
- [pmcp crate](https://github.com/paiml/rust-mcp-sdk)

---

## Problem

Aprender ships a 57-subcommand CLI (`apr`) with structured `--json` output on most commands. It achieves 1.43× Ollama decode perf at 128 tokens. But no agentic tool (Claude Code, Cursor, Cline, Aider, Continue) can invoke it, because aprender does not speak MCP.

Every competitor tool with ecosystem momentum in early 2026 is addressable via MCP — except the local-inference tier. Ollama, llama.cpp, and Unsloth all lack first-party MCP servers. Shipping `apr mcp` first occupies that slot.

## Goal

A single subcommand — `apr mcp` — that starts an MCP server over stdio, exposing a curated subset of the 57 apr CLI commands as MCP tools. Schema is generated from `contracts/apr-cli-commands-v1.yaml`, not hand-written.

Success is measured by Claude Code / Cursor / Cline being able to `.mcp.json`-configure `apr mcp` and have the LLM invoke `apr.run`, `apr.qa`, `apr.trace`, etc. on local models.

## Architecture

### New Crate: `aprender-mcp`

Location: `crates/aprender-mcp/`

```
crates/aprender-mcp/
├── Cargo.toml            # depends on pmcp, tokio, serde, clap, apr-cli
├── src/
│   ├── lib.rs            # public API: `Server::from_config().serve_stdio()`
│   ├── server.rs         # pmcp::Server wiring
│   ├── tools/            # one module per tool
│   │   ├── mod.rs        # registration table
│   │   ├── run.rs        # `apr.run` — inference
│   │   ├── qa.rs         # `apr.qa` — quality gates
│   │   ├── trace.rs      # `apr.trace` — layer analysis
│   │   ├── tensors.rs    # `apr.tensors` — tensor inspection
│   │   ├── validate.rs   # `apr.validate` — integrity
│   │   ├── bench.rs      # `apr.bench` — perf benchmarks
│   │   ├── finetune.rs   # `apr.finetune` — LoRA training
│   │   └── serve.rs      # `apr.serve` — OpenAI API server lifecycle
│   └── schema.rs         # parses contracts/apr-cli-commands-v1.yaml → MCP tool schemas
└── tests/                # protocol-level integration tests
```

### `apr mcp` Subcommand

Wired into `apr-cli`:

```rust
// crates/apr-cli/src/commands/mcp.rs
pub async fn mcp_command(args: McpArgs) -> Result<()> {
    let server = aprender_mcp::Server::from_default_config()?;
    match args.transport {
        Transport::Stdio => server.serve_stdio().await,
        Transport::Sse { port } => server.serve_sse(port).await,
    }
}
```

### Tool Surface (Phase 1)

Eight high-value tools for agentic coding + ML workflows:

| Tool | Maps to CLI | Inputs | Output |
|------|-------------|--------|--------|
| `apr.run` | `apr run <model>` | `model_path`, `prompt`, `max_tokens`, `temperature`, `top_p` | `{tokens: [...], tok_per_sec, stop_reason}` |
| `apr.serve` | `apr serve <model>` | `model_path`, `port` | `{pid, url}` + lifecycle |
| `apr.qa` | `apr qa <model> --json` | `model_path`, `assert_tps?` | 8 gates × `{pass, value, threshold}` |
| `apr.trace` | `apr trace <model> --prompt X` | `model_path`, `prompt`, `steps?` | per-layer tensor stats |
| `apr.tensors` | `apr tensors <model> --json` | `model_path` | tensor list with shapes/dtypes/stats |
| `apr.validate` | `apr validate <model> --quality` | `model_path` | integrity + quality gates |
| `apr.bench` | `apr bench <model>` | `model_path`, `runs`, `tokens` | median tok/s, p50/p95/p99 latency |
| `apr.finetune` | `apr finetune` | `base_model`, `dataset`, `lora_rank`, `epochs` | progress events + final checkpoint path |

Schema generation: each tool's JSONSchema is derived from the entry in `contracts/apr-cli-commands-v1.yaml` at build time by `aprender-mcp`'s `build.rs` — **no hand-maintained schemas**.

### Protocol

- **Transport**: stdio (primary), SSE (optional, gated behind `--transport sse --port N`)
- **Version**: MCP v2024-11-05 (matches `mcp-tool-schema-v1.yaml`)
- **Lifecycle**: initialize → initialized → tools/list → tools/call → (long-running tools stream progress) → shutdown
- **Streaming**: `apr.run` and `apr.finetune` send `notifications/progress` for each decoded token / training step. Other tools return synchronously.
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

Config precedence (highest first):
1. `--config <path>` flag
2. `$APR_MCP_CONFIG` env var
3. `~/.config/apr/mcp.toml`
4. Built-in defaults

## Falsification Conditions (for `apr-mcp-server-v1.yaml`)

The server is only ACTIVE if all of these are falsifiable by CI:

1. **FALSIFY-MCP-001**: `apr mcp < init.json` responds to `initialize` within 500ms with `{"protocolVersion":"2024-11-05", ...}`.
2. **FALSIFY-MCP-002**: `tools/list` returns exactly the 8 Phase-1 tools; schema for each validates against JSONSchema Draft 7.
3. **FALSIFY-MCP-003**: `tools/call apr.run` on `qwen2.5-0.5b-instruct-q4km.gguf` with prompt "1+1=" decodes "2" as first token within 5s.
4. **FALSIFY-MCP-004**: `tools/call apr.qa` returns 8 gates with correct pass/fail states matching `apr qa --json` CLI output byte-for-byte.
5. **FALSIFY-MCP-005**: Malformed request (`"jsonrpc": "1.0"`) returns JSON-RPC error code `-32600`, does not crash server.
6. **FALSIFY-MCP-006**: `notifications/cancelled` during `apr.run` stops decoding within 30s, returns partial result.
7. **FALSIFY-MCP-007**: Protocol version mismatch (`"protocolVersion": "1999-01-01"`) returns error, does not attempt tools/list.
8. **FALSIFY-MCP-008**: Schema in `tools/list` output is byte-identical to generated schema from `contracts/apr-cli-commands-v1.yaml`.

## Milestones

### M1: Skeleton (Week 1)
- [ ] Create `crates/aprender-mcp/` crate
- [ ] Add `pmcp` dependency (use `paiml/rust-mcp-sdk` per existing contract)
- [ ] Wire `apr mcp` subcommand into apr-cli
- [ ] Implement `initialize` + `tools/list` with 1 stub tool (`apr.version`)
- [ ] FALSIFY-MCP-001, -002 passing

### M2: Phase-1 tools (Week 2)
- [ ] Implement 8 tools as subprocess wrappers around `apr <cmd> --json`
- [ ] Schema generation from `contracts/apr-cli-commands-v1.yaml` (build.rs)
- [ ] FALSIFY-MCP-003, -004, -008 passing

### M3: Streaming + cancellation (Week 3)
- [ ] Progress notifications for `apr.run` / `apr.finetune`
- [ ] Cancellation handling (SIGTERM→SIGKILL chain)
- [ ] FALSIFY-MCP-005, -006 passing

### M4: End-to-end validation (Week 4)
- [ ] Claude Code integration test (launch `apr mcp`, ask Claude to "run qwen2.5-0.5b with prompt X")
- [ ] Cursor / Cline manual smoke test
- [ ] Contract `apr-mcp-server-v1.yaml` promoted from DRAFT → ENFORCED
- [ ] Docs: `book/mcp-server.md` with `.mcp.json` examples for each client

## Success Criteria

Acceptance gate for promoting to ACTIVE:

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| `initialize` latency | <500ms | CI with `hyperfine` |
| Tool call round-trip (non-inference) | <100ms | `apr.validate` on a cached model |
| `apr.run` first-token latency | <2s | qwen2.5-0.5b-q4km on target hardware |
| Protocol spec compliance | 100% | MCP conformance suite (external) |
| Claude Code dogfood | 1 full session using only `apr.*` tools | Manual |
| 8 falsification gates | all PASS | CI |

## Out of Scope (Phase 1)

- Resources protocol (`resources/list`, `resources/read`) — future phase for exposing model files
- Prompts protocol — future phase
- Sampling (client-side LLM calls from server) — not needed for inference use case
- Auth / multi-tenant — local dev tool only
- Windows — Phase 2 (stdio transport needs testing on Windows)

## Risk Register

| Risk | Mitigation |
|------|-----------|
| `pmcp` crate API instability | Pin exact version; contribute upstream if breaking changes needed |
| Subprocess overhead per tool call | Phase 2: in-process mode (`--embedded`) linking apr-cli as library |
| Schema drift between CLI and MCP surface | `build.rs` fails build if contract YAML differs from generated Rust structs |
| MCP clients expect specific error shapes | Conformance-test against Claude Code, Cursor, Cline fixtures |

## Related Work

- **Existing infrastructure** (ready to use):
  - `contracts/mcp-tool-schema-v1.yaml` — defines JSON-RPC error codes, session lifecycle
  - `contracts/apr-tool-rust-mcp-sdk-v1.yaml` — approves `paiml/rust-mcp-sdk` as dependency
  - `crates/apr-cli/src/tool_commands.rs` — planned MCP tool surface (referenced but unimplemented)

- **Aspirational follow-ons**:
  - `apr-mcp-plugin-marketplace-v1.md` — Claude Code–style plugin marketplace for community `apr.*` tools
  - `apr-mcp-hooks-v1.md` — pre/post-inference hooks (analog to git hooks) for QA + observability

---

**Owner**: apr-cli team
**Sponsor**: apr-cli team
**Delivery**:
- **v0.31.0** (2026-04-19, tag 62893da32): M1–M3 SHIPPED — 9 tools (`apr.run`, `apr.serve`, `apr.qa`, `apr.trace`, `apr.tensors`, `apr.validate`, `apr.bench`, `apr.finetune`, and dispatch infrastructure), `build.rs` schema+description codegen from `contracts/apr-mcp-tool-schemas-v1.yaml`, `notifications/progress` for `apr.finetune`, `notifications/cancelled` SIGTERM→SIGKILL, JSON Schema Draft 7 meta-validation on every tool input schema in CI, MCP book chapter documenting `.mcp.json` client config.
- **M4** (in flight): PRs #886/#889/#890/#891/#892 — additional tool coverage + conformance hardening.
- **M5+** (planned): per spec v1.2.0 roadmap — plugin marketplace, pre/post-inference hooks.
