# aprender-mcp

Model Context Protocol (MCP) server for [aprender](https://github.com/paiml/aprender).
Exposes the `apr` CLI as MCP tools for Claude Code, Cursor, Cline, and other
MCP clients over JSON-RPC 2.0 stdio transport.

- Spec: [`docs/specifications/apr-mcp-server-spec.md`](../../docs/specifications/apr-mcp-server-spec.md)
- Protocol: [MCP v2024-11-05](https://spec.modelcontextprotocol.io/specification/2024-11-05/)

## Usage

### As a library

```rust
let mut server = aprender_mcp::AprMcpServer::new();
server.run_stdio()?;
```

### As `apr mcp` subcommand

```bash
apr mcp
```

### `.mcp.json` for Claude Code / Cursor / Cline

```json
{
  "mcpServers": {
    "aprender": {
      "command": "apr",
      "args": ["mcp"]
    }
  }
}
```

## Milestones

- **M1** (shipped): skeleton with `initialize` + `tools/list` + `apr.version`
- **M2** (shipped): 7 Phase-1 tools as subprocess wrappers over
  `apr <cmd> --json`: `apr.validate`, `apr.tensors`, `apr.bench`,
  `apr.qa`, `apr.trace`, `apr.run`, `apr.serve` + dispatcher hardening
  (jsonrpc/protocolVersion gates).
- **M3** (shipped 2026-04-18): `apr.finetune` synchronous wrapper
  (8th Phase-1 workflow tool — 9th registered, counting the M1
  `apr.version` scaffold), `notifications/cancelled` → SIGTERM→SIGKILL
  for `apr.run`, build.rs schema + description codegen from
  `contracts/apr-mcp-tool-schemas-v1.yaml` (all 9 registered tools;
  `APR_<TOOL>_SCHEMA` and `APR_<TOOL>_DESCRIPTION` constants — PMAT-514),
  and opt-in per-line `notifications/progress` for `apr.finetune` via
  `params._meta.progressToken`.
- **M4** (in progress): Claude Code dogfood + contract promotion to
  ENFORCED + real-model FALSIFY-MCP-003/-004.
- **M5** (planned): port dispatcher to `pmcp = "2.3"`, add SSE/WebSocket
  transports, extend cancel to `apr.serve`.

`apr.serve` is still fire-and-forget — M3's cancellation path covers
`apr.run` only; the serve-daemon lifecycle registry is queued for M5.

## Falsification gates

Spec reference: [`docs/specifications/apr-mcp-server-spec.md`](../../docs/specifications/apr-mcp-server-spec.md#falsification-conditions-for-apr-mcp-server-v1yaml).

| Gate | Status | What it asserts |
|------|--------|-----------------|
| FALSIFY-MCP-001 | ENFORCED | `initialize` round-trip under 500ms (CI threshold 50ms) |
| FALSIFY-MCP-002 | ENFORCED | every registered tool exposes a valid object-typed JSON Schema Draft 7 |
| FALSIFY-MCP-003 | PARTIAL → M4 | `tools/call apr.run` decodes `"2"` from `"1+1="` within 5s on qwen2.5-0.5b |
| FALSIFY-MCP-004 | PARTIAL → M4 | `tools/call apr.qa` byte-identical to `apr qa --json` |
| FALSIFY-MCP-005 | ENFORCED | `jsonrpc != "2.0"` is rejected with `-32600 Invalid Request` |
| FALSIFY-MCP-006 | ENFORCED | `notifications/cancelled` during `apr.run` stops decoding within 30s grace |
| FALSIFY-MCP-007 | ENFORCED | `initialize` **negotiates**: an unsupported `params.protocolVersion` still completes the handshake, answering with the version the server does speak (0.63.0 returned `-32602`, locking out every client that negotiates 2025-03-26 / 2025-06-18) |
| FALSIFY-MCP-008 | ENFORCED | each tool's live `inputSchema` **and** `description` in `tools/list` are byte-identical to `contracts/apr-mcp-tool-schemas-v1.yaml` (`tests/falsify_mcp_008.rs`); both fields are emitted from the YAML by `build.rs` (`schemas::APR_<TOOL>_SCHEMA` + `APR_<TOOL>_DESCRIPTION`), so hand-editing the Rust source fails CI |
| FALSIFY-MCP-010 | ENFORCED | `ping` is base protocol — answered with an empty result, not `-32601` |
| FALSIFY-MCP-011 | ENFORCED | a malformed line costs one line, not the session: an invalid UTF-8 byte yields `-32700` and the server keeps serving; a valid object missing `jsonrpc`/`method` is `-32600` with its id echoed; a batch array is declined by name |
| FALSIFY-MCP-012 | ENFORCED | a `tools/call` answer is delivered even when stdin hits EOF immediately after the request (`printf ... \| apr mcp`) |
| FALSIFY-MCP-013 | ENFORCED | a required argument of the wrong JSON type is reported as a type error, not as "Missing required argument" |
| FALSIFY-MCP-PROGRESS-001 | ENFORCED | opt-in `notifications/progress` for `apr.finetune` flushed before final response |
| FALSIFY-MCP-VALIDATE-001 | ENFORCED | tool argument validation surfaces as `isError:true`, not as a JSON-RPC error |
