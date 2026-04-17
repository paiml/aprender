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
  `apr <cmd> --json`. Shipped so far: `apr.validate`. Remaining: `apr.run`,
  `apr.serve`, `apr.qa`, `apr.trace`, `apr.tensors`, `apr.bench`, `apr.finetune`
- **M3**: streaming progress notifications + cancellation
- **M4**: Claude Code dogfood + contract promotion to ENFORCED

## Falsification gates

See [`contracts/apr-mcp-server-v1.yaml`](../../contracts/apr-mcp-server-v1.yaml)
for the full set. Currently enforced: FALSIFY-MCP-001 (initialize latency),
FALSIFY-MCP-002 (progressive — every registered tool has a valid schema),
and FALSIFY-MCP-VALIDATE-001 (argument validation surfaces as tool error).
