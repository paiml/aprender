# `batuta mcp`

Run Batuta as an MCP (Model Context Protocol) server for AI tool integration.

## Synopsis

```bash
batuta mcp [TRANSPORT]
```

## Description

The MCP server exposes Batuta's HuggingFace integration as tools that AI assistants (Claude, etc.) can invoke via JSON-RPC 2.0 over stdio. This enables AI-assisted model discovery and management.

## Transport Modes

| Transport | Description |
|-----------|-------------|
| `stdio` (default) | JSON-RPC 2.0 over stdin/stdout |

## Available Tools

| Tool | Description |
|------|-------------|
| `hf_search` | Search HuggingFace Hub for models, datasets, or spaces |
| `hf_info` | Get metadata about a specific repository |
| `hf_pull` | Download a model or dataset from HuggingFace |
| `hf_push` | Upload artifacts to HuggingFace Hub |

## Examples

### Start MCP Server

```bash
$ batuta mcp

# Server listens on stdin for JSON-RPC 2.0 messages
```

### JSON-RPC Initialize

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}
```

### List Available Tools

```json
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
```

### Claude Desktop Integration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "batuta": {
      "command": "batuta",
      "args": ["mcp"]
    }
  }
}
```

## See Also

- [MCP Tooling](../part3/mcp-tooling.md)
- [`batuta hf`](./cli-hf.md) - CLI HuggingFace commands

---

**Previous:** [`batuta bug-hunter`](./cli-bug-hunter.md)
**Next:** [`batuta serve`](./cli-serve.md)
