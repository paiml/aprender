# CLI Inspect Example

This example demonstrates model inspection for architecture and parameter analysis.

## Running the Example

```bash
cargo run --example cli_inspect
```

## Code

```rust
{{#include ../../../examples/cli_inspect.rs}}
```

## CLI Usage

```bash
# Basic inspection
entrenar inspect model.safetensors

# Verbose output with layer details
entrenar inspect model.safetensors -v

# JSON output for programmatic use
entrenar inspect model.safetensors --json
```

## Output

```
Model: model.safetensors
Architecture: transformer

Layers:
  embed_tokens: [32000, 4096] - 131M params
  layers.0.self_attn.q_proj: [4096, 4096] - 16.8M params
  layers.0.self_attn.k_proj: [4096, 4096] - 16.8M params
  ...

Summary:
  Total parameters: 6.7B
  Total size: 13.4 GB (FP16)
  Quantization: None
```
