# MCP Tooling

The **Model Context Protocol (MCP)** is an open standard for connecting AI assistants to external tools and data sources. The PAIML stack provides first-class MCP support through two complementary crates:

| Crate | Version | Purpose |
|-------|---------|---------|
| **pmcp** | v1.8.6 | Low-level Rust SDK for building MCP servers and clients |
| **pforge** | v0.1.4 | High-level declarative framework for MCP servers |

## Why MCP?

MCP enables AI assistants (like Claude) to:
- Execute tools and functions
- Access external data sources
- Integrate with APIs and services
- Maintain stateful sessions

```
┌─────────────────┐     MCP Protocol     ┌─────────────────┐
│   AI Assistant  │ ◄─────────────────► │   MCP Server    │
│   (Claude)      │                      │   (Your Tools)  │
└─────────────────┘                      └─────────────────┘
```

## Stack Integration

MCP tooling integrates with the broader PAIML ecosystem:

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Server (pforge)                  │
├─────────────────────────────────────────────────────────┤
│  Tool: train_model    │  Tool: query_data               │
│  → Entrenar           │  → Trueno-DB                    │
├───────────────────────┼─────────────────────────────────┤
│  Tool: run_inference  │  Tool: visualize                │
│  → Realizar           │  → Trueno-Viz                   │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Option 1: pforge (Recommended)

For most use cases, pforge provides the fastest path to a working MCP server:

```bash
# Install pforge CLI
cargo install pforge-cli

# Create new server
pforge new my-ml-server
cd my-ml-server

# Run server
pforge serve
```

### Option 2: pmcp (Low-Level)

For custom implementations or advanced use cases:

```rust
use pmcp::{Server, Tool, ToolHandler};

#[tokio::main]
async fn main() {
    let server = Server::new("my-server")
        .with_tool(MyTool::new())
        .build();

    server.serve_stdio().await.unwrap();
}
```

## Use Cases

| Use Case | Recommended Approach |
|----------|---------------------|
| Simple tool server | pforge with YAML config |
| Complex business logic | pforge with native handlers |
| Custom protocol needs | pmcp directly |
| Embedded in larger app | pmcp as library |

## Next Steps

- [pmcp: Rust MCP SDK](./pmcp.md) - Deep dive into the SDK
- [pforge: Declarative Framework](./pforge.md) - YAML-based server development
