# Model I/O Overview

**Model I/O** provides save/load functionality for neural network
models with support for multiple serialization formats.

## The Problem

After training a model, you need to:
- **Save model weights** for deployment
- **Load trained models** for inference or continued training
- **Share models** with collaborators
- **Version control** model checkpoints
- **Metadata tracking** (hyperparameters, training config, etc.)

## The Solution

Entrenar's Model I/O system (from `src/io/`) provides:

```rust
use entrenar::io::{save_model, load_model, Model, ModelMetadata, SaveConfig, ModelFormat};

// Create model with metadata
let metadata = ModelMetadata::new("my-model", "transformer")
    .with_version("0.1.0")
    .with_custom("learning_rate", 0.001);

let model = Model::new(metadata, parameters);

// Save to JSON
let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
save_model(&model, "model.json", &config)?;

// Load (format auto-detected from extension)
let loaded = load_model("model.json")?;
```

## Supported Formats

| Format | Extension | Use Case | Status |
|--------|-----------|----------|--------|
| **SafeTensors** | `.safetensors` | Production, HuggingFace Hub | ✅ Recommended |
| **JSON** | `.json` | Human-readable, debugging | ✅ Implemented |
| **YAML** | `.yaml`, `.yml` | Configuration-friendly | ✅ Implemented |
| **GGUF** | `.gguf` | LLaMA-compatible format | ⚠️ Placeholder (future Realizar integration) |

### SafeTensors Format (Recommended)

The recommended format for production use. Provides security (no arbitrary code
execution), efficiency (zero-copy loading), and HuggingFace Hub compatibility:

```rust
// Save as SafeTensors
let config = SaveConfig::new(ModelFormat::SafeTensors);
save_model(&model, "model.safetensors", &config)?;

// Load (format auto-detected)
let model = load_model("model.safetensors")?;
```

### JSON Format

**Compact** (single-line):
```rust
let config = SaveConfig::new(ModelFormat::Json).with_pretty(false);
save_model(&model, "model.json", &config)?;
```

**Pretty** (indented):
```rust
let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
save_model(&model, "model.json", &config)?;
```

### YAML Format

Human-friendly for configuration:
```rust
let config = SaveConfig::new(ModelFormat::Yaml);
save_model(&model, "model.yaml", &config)?;
```

### GGUF Format

**Placeholder** for future integration with Realizar:
```rust
// Will be supported in v0.2.0+
let config = SaveConfig::new(ModelFormat::Gguf);
save_model(&model, "model.gguf", &config)?;  // Currently returns error
```

## Model Structure

### Model

Contains parameters and metadata:

```rust
pub struct Model {
    pub metadata: ModelMetadata,
    pub parameters: Vec<(String, Tensor)>,
}
```

### ModelMetadata

Tracks model information:

```rust
pub struct ModelMetadata {
    pub name: String,
    pub architecture: String,
    pub version: String,
    pub training_config: Option<HashMap<String, Value>>,
    pub custom: HashMap<String, Value>,  // Flexible key-value pairs
}
```

**Example**:
```rust
let metadata = ModelMetadata::new("llama-7b-lora", "transformer")
    .with_version("0.1.0")
    .with_custom("lora_rank", 64)
    .with_custom("lora_alpha", 128)
    .with_custom("base_model", "meta-llama/Llama-2-7b");
```

## Round-Trip Integrity

All save/load operations maintain **round-trip integrity**:

```rust
// Original model
let original = create_model();

// Save and load
save_model(&original, "temp.json", &config)?;
let loaded = load_model("temp.json")?;

// Verify parameters match
assert_eq!(original.parameters.len(), loaded.parameters.len());
for (orig, load) in original.parameters.iter().zip(loaded.parameters.iter()) {
    assert_eq!(orig.0, load.0);  // Parameter names
    assert_tensors_equal(&orig.1, &load.1);  // Tensor values
}
```

**Validation**: 54 I/O tests ensure round-trip correctness

## Auto-Format Detection

Format automatically detected from file extension:

```rust
// Detects SafeTensors from .safetensors extension
let model = load_model("model.safetensors")?;

// Detects JSON from .json extension
let model = load_model("model.json")?;

// Detects YAML from .yaml extension
let model = load_model("config.yaml")?;
```

## Example Workflow

From `examples/model_io.rs`:

```rust
use entrenar::io::{Model, ModelMetadata, save_model, load_model, SaveConfig, ModelFormat};
use entrenar::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create model
    let params = vec![
        ("layer1.weight".to_string(), Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], true)),
        ("layer1.bias".to_string(), Tensor::from_vec(vec![0.01, 0.02], true)),
        ("layer2.weight".to_string(), Tensor::from_vec(vec![0.5, 0.6], true)),
        ("layer2.bias".to_string(), Tensor::from_vec(vec![0.1], true)),
    ];

    let metadata = ModelMetadata::new("example-model", "simple-mlp")
        .with_version("0.1.0")
        .with_custom("input_dim", 4)
        .with_custom("hidden_dim", 2)
        .with_custom("output_dim", 1);

    let model = Model::new(metadata, params);

    // Save as JSON
    let json_config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
    save_model(&model, "example_model.json", &json_config)?;

    // Save as YAML
    let yaml_config = SaveConfig::new(ModelFormat::Yaml);
    save_model(&model, "example_model.yaml", &yaml_config)?;

    // Load and verify
    let loaded = load_model("example_model.json")?;
    println!("✅ Loaded model: {}", loaded.metadata.name);

    Ok(())
}
```

## Next Steps

- [Save Models](./save-models.md) - Detailed save functionality
- [Load Models](./load-models.md) - Loading and deserialization
- [Model Metadata](./metadata.md) - Metadata management
- [Supported Formats](./formats.md) - Format details
- [SafeTensors Format](./safetensors-format.md) - HuggingFace compatible format

## Implementation

All Model I/O code is in `src/io/`:
- `mod.rs` - Public API exports
- `model.rs` - Model and ModelMetadata structs
- `format.rs` - ModelFormat enum and SaveConfig
- `save.rs` - save_model() function (incl. SafeTensors serialization)
- `load.rs` - load_model() function (incl. SafeTensors deserialization)
- `tests.rs` - Integration tests (54 total across module)
