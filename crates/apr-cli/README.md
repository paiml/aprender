# apr-cli

CLI tool for APR model inspection, debugging, and operations.

## Installation

```bash
cargo install apr-cli
```

This installs the `apr` binary.

## Features

- **Model Inspection**: View APR model structure, metadata, and weights
- **Debugging**: Hex dumps, tree visualization, flow analysis
- **Operations**: List, compare, and validate APR models
- **TUI Mode**: Interactive terminal interface for model exploration

## Usage

```bash
# Show help
apr --help

# Inspect a model
apr inspect model.apr

# List models in directory
apr list ./models/

# Interactive TUI mode
apr tui model.apr

# Compare two models
apr diff model1.apr model2.apr
```

## Chat Interface

Interactive chat with language models (supports APR, GGUF, SafeTensors):

```bash
# Chat with a GGUF model (GPU acceleration by default)
apr chat model.gguf

# Force CPU inference
apr chat model.gguf --no-gpu

# Explicitly request GPU acceleration
apr chat model.gguf --gpu

# Adjust generation parameters
apr chat model.gguf --temperature 0.7 --top-p 0.9 --max-tokens 512
```

## Optional Features

### Inference Server

Enable the `inference` feature to serve models via HTTP:

```bash
cargo install apr-cli --features inference

apr serve model.gguf --port 8080
```

The server provides an OpenAI-compatible API:

```bash
# Health check
curl http://localhost:8080/health

# Chat completions
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'

# Streaming
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}],"stream":true,"max_tokens":50}'
```

### Debugging with Tracing

Use the `X-Trace-Level` header to enable inference tracing for debugging:

```bash
# Brick-level tracing (token operations)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Trace-Level: brick" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}'

# Step-level tracing (forward pass steps)
curl -H "X-Trace-Level: step" ...

# Layer-level tracing (per-layer timing)
curl -H "X-Trace-Level: layer" ...
```

Trace levels:
- `brick`: Token-by-token operation timing
- `step`: Forward pass steps (embed, attention, mlp, lm_head)
- `layer`: Per-layer timing breakdown (24+ layers)

### CUDA GPU Acceleration

Enable CUDA support for NVIDIA GPUs:

```bash
cargo install apr-cli --features inference,cuda
```

## Examples

```bash
# Run the tracing example
cargo run --example serve_with_tracing --features inference
```

## License

MIT
