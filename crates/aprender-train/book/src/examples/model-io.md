# Model I/O

This example demonstrates saving and loading models in JSON and YAML formats.

## Running the Example

```bash
cargo run --example model_io
```

## Code

```rust
{{#include ../../../examples/model_io.rs}}
```

## Expected Output

```
=== Model I/O Example ===

Creating model...
  Model name: example-model
  Architecture: simple-mlp
  Parameters: 4

Saving model to JSON...
  ✓ Saved to example_model.json

Saving model to YAML...
  ✓ Saved to example_model.yaml

Loading model from JSON...
  ✓ Loaded model: example-model
  ✓ Parameters: 4
  ✓ Data integrity check: true

Loading model from YAML...
  ✓ Loaded model: example-model
  ✓ Parameters: 4
```

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| JSON | `.json` | API interchange, debugging |
| YAML | `.yaml` | Human-readable configs |
| SafeTensors | `.safetensors` | Production deployment |
| GGUF | `.gguf` | llama.cpp compatibility |

## Usage

```rust
use entrenar::io::{save_model, load_model, SaveConfig, ModelFormat};

// Save to JSON
let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
save_model(&model, "model.json", &config)?;

// Load (auto-detects format)
let loaded = load_model("model.json")?;
```
