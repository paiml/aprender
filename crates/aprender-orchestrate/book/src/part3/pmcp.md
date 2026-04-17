# pmcp: Rust MCP SDK

**pmcp** (v1.8.6) is a high-quality Rust SDK for the Model Context Protocol with full TypeScript SDK compatibility.

## Installation

```toml
[dependencies]
pmcp = "1.8"
```

## Features

| Feature | Description |
|---------|-------------|
| Full MCP compliance | Compatible with TypeScript SDK |
| Async-first | Built on Tokio for high performance |
| Type-safe | Rust's type system prevents runtime errors |
| Transport agnostic | stdio, HTTP, WebSocket support |
| Schema generation | Automatic JSON Schema via schemars |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      pmcp SDK                           │
├─────────────────────────────────────────────────────────┤
│  Server          │  Client          │  Transport       │
│  - Tool registry │  - Tool calling  │  - Stdio         │
│  - Resource mgmt │  - Resource read │  - HTTP/SSE      │
│  - Prompt system │  - Prompt list   │  - WebSocket     │
└─────────────────────────────────────────────────────────┘
```

## Basic Server

```rust
use pmcp::{Server, ServerBuilder};
use pmcp::tool::{Tool, ToolBuilder, ToolHandler};
use async_trait::async_trait;

struct GreetTool;

#[async_trait]
impl ToolHandler for GreetTool {
    async fn call(&self, args: serde_json::Value) -> pmcp::Result<serde_json::Value> {
        let name = args["name"].as_str().unwrap_or("World");
        Ok(serde_json::json!({
            "greeting": format!("Hello, {}!", name)
        }))
    }
}

#[tokio::main]
async fn main() -> pmcp::Result<()> {
    let server = ServerBuilder::new("greeting-server")
        .version("1.0.0")
        .tool(
            ToolBuilder::new("greet")
                .description("Greet someone by name")
                .param("name", "string", "Name to greet", true)
                .handler(GreetTool)
                .build()
        )
        .build();

    server.serve_stdio().await
}
```

## Tool Definition

Tools are the primary way to expose functionality:

```rust
use pmcp::tool::{ToolBuilder, ToolSchema};

let tool = ToolBuilder::new("analyze_code")
    .description("Analyze source code for issues")
    .param("code", "string", "Source code to analyze", true)
    .param("language", "string", "Programming language", false)
    .param("strict", "boolean", "Enable strict mode", false)
    .handler(AnalyzeHandler)
    .build();
```

## Resources

Resources provide read-only data access:

```rust
use pmcp::resource::{Resource, ResourceBuilder};

let resource = ResourceBuilder::new("file://config.yaml")
    .name("Configuration")
    .description("Application configuration")
    .mime_type("application/yaml")
    .handler(ConfigResourceHandler)
    .build();
```

## Prompts

Prompts are reusable message templates:

```rust
use pmcp::prompt::{Prompt, PromptBuilder};

let prompt = PromptBuilder::new("code_review")
    .description("Review code for best practices")
    .argument("code", "Code to review", true)
    .argument("focus", "Area to focus on", false)
    .build();
```

## Transport Options

### Stdio (Default)

```rust
server.serve_stdio().await?;
```

### HTTP with SSE

```rust
server.serve_http("127.0.0.1:8080").await?;
```

### WebSocket

```rust
server.serve_websocket("127.0.0.1:8081").await?;
```

## Integration with PAIML Stack

### Entrenar Integration

```rust
use pmcp::tool::ToolHandler;
use entrenar::train::Trainer;

struct TrainModelTool {
    trainer: Trainer,
}

#[async_trait]
impl ToolHandler for TrainModelTool {
    async fn call(&self, args: serde_json::Value) -> pmcp::Result<serde_json::Value> {
        let config_path = args["config"].as_str().unwrap();
        // Load YAML config and train
        let metrics = self.trainer.train_from_yaml(config_path)?;
        Ok(serde_json::to_value(metrics)?)
    }
}
```

### Realizar Integration

```rust
use realizar::inference::InferenceEngine;

struct InferenceTool {
    engine: InferenceEngine,
}

#[async_trait]
impl ToolHandler for InferenceTool {
    async fn call(&self, args: serde_json::Value) -> pmcp::Result<serde_json::Value> {
        let prompt = args["prompt"].as_str().unwrap();
        let response = self.engine.generate(prompt).await?;
        Ok(serde_json::json!({ "response": response }))
    }
}
```

## Error Handling

```rust
use pmcp::{Error, ErrorCode};

// Return structured errors
Err(Error::new(
    ErrorCode::InvalidParams,
    "Missing required parameter: name"
))
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use pmcp::testing::MockClient;

    #[tokio::test]
    async fn test_greet_tool() {
        let client = MockClient::new(server);
        let result = client.call_tool("greet", json!({"name": "Alice"})).await;
        assert_eq!(result["greeting"], "Hello, Alice!");
    }
}
```

## Best Practices

1. **Use descriptive tool names** - `analyze_python_code` not `analyze`
2. **Document all parameters** - Include description and required flag
3. **Return structured JSON** - Not raw strings
4. **Handle errors gracefully** - Use proper error codes
5. **Keep tools focused** - One tool, one purpose

## Agent Integration: MCP Client

The [Agent Runtime](./agent-runtime.md) uses pmcp via `McpClientTool` to
discover and call external MCP servers. The agent manifest declares MCP
servers; at startup, tools are wrapped as `McpClientTool` instances:

```toml
# Agent manifest — connect to external MCP server
[[mcp_servers]]
name = "code-search"
transport = "stdio"
command = ["node", "server.js"]
capabilities = ["*"]
```

**Privacy enforcement:** Sovereign tier restricts to `stdio` transport only.
`sse` and `websocket` are blocked (both at validation and runtime).

See [Agent Runtime: MCP Client Tool](./agent-runtime.md#mcp-client-tool) for details.

## See Also

- [pforge](./pforge.md) - Declarative framework built on pmcp
- [Agent Runtime](./agent-runtime.md) - McpClientTool integration
- [MCP Specification](https://modelcontextprotocol.io/) - Official protocol docs
