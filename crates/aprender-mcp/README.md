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
- **M2** (in progress): 8 Phase-1 tools as subprocess wrappers over
  `apr <cmd> --json`. Shipped: `apr.validate`, `apr.tensors`, `apr.bench`,
  `apr.qa`, `apr.trace`, `apr.run`, `apr.serve` (fire-and-forget; full
  lifecycle in M3) + dispatcher hardening (jsonrpc/protocolVersion gates).
  Remaining: `apr.finetune` (streaming candidate, likely M3).
- **M3**: streaming progress notifications + cancellation
- **M4**: Claude Code dogfood + contract promotion to ENFORCED

## Falsification gates

Currently enforced (see [`docs/specifications/apr-mcp-server-spec.md`](../../docs/specifications/apr-mcp-server-spec.md#falsification-conditions-for-apr-mcp-server-v1yaml)):

| Gate | What it asserts |
|------|-----------------|
| FALSIFY-MCP-001 | `initialize` round-trip under 50ms (spec budget 500ms) |
| FALSIFY-MCP-002 | every registered tool exposes a valid object-typed schema |
| FALSIFY-MCP-005 | `jsonrpc != "2.0"` is rejected with `-32600 Invalid Request` |
| FALSIFY-MCP-007 | `initialize.params.protocolVersion` mismatch returns `-32602 Invalid Params` (positive case verifies happy path) |
| FALSIFY-MCP-VALIDATE-001 | tool argument validation surfaces as `isError:true`, not as a JSON-RPC error |
