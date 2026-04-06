# pforge: Declarative MCP Framework

**pforge** (v0.1.4) is a zero-boilerplate framework for building MCP servers using YAML configuration.

## Installation

```bash
cargo install pforge-cli
```

## Quick Start

```bash
# Create new project
pforge new my-server
cd my-server

# Project structure:
# my-server/
# ├── pforge.yaml      # Server configuration
# ├── src/
# │   └── handlers/    # Native Rust handlers
# └── Cargo.toml

# Run the server
pforge serve
```

## Configuration (pforge.yaml)

```yaml
forge:
  name: ml-tools-server
  version: 0.1.0
  transport: stdio
  description: "ML tools for model training and inference"

tools:
  # Native Rust handler
  - type: native
    name: train_model
    description: "Train a model using YAML configuration"
    handler:
      path: handlers::train_model
    params:
      config_path:
        type: string
        required: true
        description: "Path to training YAML config"
      epochs:
        type: integer
        required: false
        description: "Override number of epochs"

  # CLI handler - execute shell commands
  - type: cli
    name: list_models
    description: "List available models"
    command: "ls -la models/"

  # HTTP proxy handler
  - type: http
    name: huggingface_search
    description: "Search HuggingFace Hub"
    endpoint: "https://huggingface.co/api/models"
    method: GET
    params:
      search:
        type: string
        required: true

  # Pipeline handler - chain tools
  - type: pipeline
    name: train_and_export
    description: "Train model and export to GGUF"
    steps:
      - tool: train_model
        params:
          config_path: "{{config}}"
      - tool: export_gguf
        params:
          model_path: "{{previous.model_path}}"
```

## Handler Types

### Native Handlers

Full Rust implementation with type safety:

```rust
// src/handlers/mod.rs
use pforge_runtime::prelude::*;

pub async fn train_model(args: ToolArgs) -> ToolResult {
    let config_path = args.get_string("config_path")?;
    let epochs = args.get_optional_int("epochs");

    // Your training logic here
    let metrics = run_training(config_path, epochs).await?;

    Ok(json!({
        "status": "completed",
        "metrics": metrics
    }))
}
```

### CLI Handlers

Execute shell commands:

```yaml
tools:
  - type: cli
    name: run_benchmark
    description: "Run performance benchmark"
    command: "cargo bench --bench inference"
    timeout_ms: 60000
    working_dir: "./benchmarks"
```

### HTTP Handlers

Proxy external APIs:

```yaml
tools:
  - type: http
    name: fetch_model_info
    description: "Get model info from registry"
    endpoint: "https://api.example.com/models/{{model_id}}"
    method: GET
    headers:
      Authorization: "Bearer {{env.API_TOKEN}}"
```

### Pipeline Handlers

Chain multiple tools:

```yaml
tools:
  - type: pipeline
    name: full_workflow
    description: "Complete ML workflow"
    steps:
      - tool: validate_data
        params:
          path: "{{data_path}}"
      - tool: train_model
        params:
          data: "{{previous.validated_path}}"
      - tool: evaluate_model
        params:
          model: "{{previous.model_path}}"
```

## Resources

Define read-only data sources:

```yaml
resources:
  - uri: "file://config/default.yaml"
    name: "Default Configuration"
    description: "Default training configuration"
    mime_type: "application/yaml"

  - uri: "db://experiments"
    name: "Experiment History"
    description: "Past experiment results"
    handler:
      path: handlers::get_experiments
```

## Prompts

Reusable prompt templates:

```yaml
prompts:
  - name: code_review
    description: "Review code for ML best practices"
    arguments:
      - name: code
        description: "Code to review"
        required: true
      - name: focus
        description: "Specific area to focus on"
        required: false
    template: |
      Review this ML code for best practices:

      ```{{language}}
      {{code}}
      ```

      {{#if focus}}Focus on: {{focus}}{{/if}}
```

## Environment Variables

Reference environment variables:

```yaml
forge:
  name: secure-server

tools:
  - type: http
    name: api_call
    endpoint: "{{env.API_ENDPOINT}}"
    headers:
      Authorization: "Bearer {{env.API_KEY}}"
```

## CLI Commands

```bash
# Create new project
pforge new <name>

# Serve MCP server
pforge serve [--port 8080] [--transport stdio|http|ws]

# Validate configuration
pforge validate

# Generate Rust code (without running)
pforge codegen

# List defined tools
pforge list tools

# Test a specific tool
pforge test <tool_name> --args '{"param": "value"}'
```

## Integration Examples

### Entrenar Training Server

```yaml
forge:
  name: entrenar-mcp
  version: 0.1.0

tools:
  - type: native
    name: train
    description: "Train model from YAML config"
    handler:
      path: handlers::entrenar_train
    params:
      config: { type: string, required: true }

  - type: native
    name: quantize
    description: "Quantize model to 4-bit"
    handler:
      path: handlers::entrenar_quantize
    params:
      model_path: { type: string, required: true }
      bits: { type: integer, required: false, default: 4 }
```

### Realizar Inference Server

```yaml
forge:
  name: realizar-mcp
  version: 0.1.0

tools:
  - type: native
    name: generate
    description: "Generate text with LLM"
    handler:
      path: handlers::realizar_generate
    params:
      prompt: { type: string, required: true }
      max_tokens: { type: integer, required: false, default: 256 }
      temperature: { type: number, required: false, default: 0.7 }
```

### Trueno-DB Query Server

```yaml
forge:
  name: trueno-db-mcp
  version: 0.1.0

tools:
  - type: native
    name: query
    description: "Execute SQL query"
    handler:
      path: handlers::trueno_query
    params:
      sql: { type: string, required: true }

  - type: native
    name: vector_search
    description: "Semantic vector search"
    handler:
      path: handlers::trueno_vector_search
    params:
      query: { type: string, required: true }
      top_k: { type: integer, required: false, default: 10 }
```

## MCP Registry

pforge servers can be published to the MCP Registry:

```bash
# Publish to registry
pforge publish

# Registry entry
# Name: io.github.paiml/my-server
# Install: cargo install my-server-mcp
```

## Best Practices

1. **Keep tools atomic** - One tool, one responsibility
2. **Use pipelines for workflows** - Chain atomic tools
3. **Validate inputs** - Use JSON Schema constraints
4. **Document thoroughly** - Good descriptions help AI assistants
5. **Use native handlers for complex logic** - CLI/HTTP for simple cases
6. **Test with `pforge test`** - Validate before deployment

## Agent Integration: MCP Server

The [Agent Runtime](./agent-runtime.md) exposes agent tools as MCP server
endpoints via the `HandlerRegistry`, which is forward-compatible with
pforge's `Handler` trait:

| Handler | Actions | Description |
|---------|---------|-------------|
| `MemoryHandler` | `store`, `recall` | Agent memory fragments |
| `RagHandler` | `search` | BM25+vector document retrieval |
| `ComputeHandler` | `run`, `parallel` | Sandboxed command execution |

External LLM clients (Claude Code, other agents) can query the agent's
knowledge base and memory directly over MCP.

See [Agent Runtime: MCP Server](./agent-runtime.md#mcp-server-handler-registry) for details.

## See Also

- [pmcp](./pmcp.md) - Low-level SDK that pforge builds on
- [Agent Runtime](./agent-runtime.md) - HandlerRegistry integration
- [pforge GitHub](https://github.com/paiml/pforge) - Source and examples
- [MCP Registry](https://registry.modelcontextprotocol.io/) - Published servers
