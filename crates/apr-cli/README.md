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

#### GPU-Accelerated Server

Start the server with GPU acceleration for maximum throughput:

```bash
# Single-request GPU mode (~83 tok/s on RTX 4090)
apr serve model.gguf --port 8080 --gpu

# Batched GPU mode - 2.9x faster than Ollama (~850 tok/s)
apr serve model.gguf --port 8080 --gpu --batch
```

#### Performance Comparison

| Mode | Throughput | vs Ollama | Memory |
|------|------------|-----------|--------|
| CPU (baseline) | ~15 tok/s | 0.05x | 1.1 GB |
| GPU (single) | ~83 tok/s | 0.25x | 1.5 GB |
| GPU (batched) | ~850 tok/s | 2.9x | 1.9 GB |
| Ollama | ~333 tok/s | 1.0x | - |

#### GPU Server Output

```
=== APR Serve ===

Model: qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
Binding: 127.0.0.1:8080

Detected format: GGUF
Loading GGUF model (mmap)...
GGUF loaded: 339 tensors, 26 metadata entries
Building quantized inference model...
Model ready: 28 layers, vocab_size=151936, hidden_dim=1536
Enabling optimized CUDA acceleration (PAR-111)...
  Initializing GPU on device 0...
  Pre-uploaded 934 MB weights to GPU
CUDA optimized model ready

Performance: 755+ tok/s (2.6x Ollama)
```

#### Example GPU Request

```bash
# Chat completion with GPU acceleration
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Write a Rust function to add two numbers"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Examples

```bash
# Run the tracing example
cargo run --example serve_with_tracing --features inference

# Run the GPU chat inference example (requires CUDA)
cargo run --example gpu_chat_inference --features inference,cuda
```

## Performance Testing

Test GPU inference performance:

```bash
# Start GPU server
apr serve /path/to/model.gguf --port 8096 --gpu --batch

# Run benchmark (separate terminal)
for i in {1..10}; do
  time curl -s -X POST http://localhost:8096/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}' > /dev/null
done
```

## QA and Testing

The apr CLI includes comprehensive QA commands for model validation:

```bash
# Run falsifiable QA checklist
apr qa model.gguf

# With custom throughput threshold
apr qa model.gguf --assert-tps 100

# Compare against Ollama
apr qa model.gguf --assert-speedup 2.0

# JSON output for CI integration
apr qa model.gguf --skip-ollama --json
```

For automated QA testing, use the example runners:

```bash
# Full 21-cell QA matrix
cargo run --example qa_run -- --full-matrix

# Popperian falsification tests
cargo run --example qa_falsify
```

## License

MIT
