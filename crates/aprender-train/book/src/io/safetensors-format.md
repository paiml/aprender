# SafeTensors Format

SafeTensors is the recommended format for production models in entrenar. It provides
security, efficiency, and full HuggingFace Hub compatibility.

## Why SafeTensors?

SafeTensors was developed by HuggingFace to address security concerns with Python's
pickle format. Key benefits:

- **Security**: No arbitrary code execution (pickle files can run malicious code)
- **Zero-copy loading**: Memory-mapped tensor access without full deserialization
- **Cross-platform**: Works consistently across Python, Rust, JavaScript
- **HuggingFace compatible**: Direct upload/download from HuggingFace Hub

## File Structure

SafeTensors files have a simple binary structure:

```
┌─────────────────────────────────────────┐
│ Header Length (8 bytes, little-endian)  │
├─────────────────────────────────────────┤
│ JSON Header (variable length)           │
│ - Tensor metadata (names, shapes, types)│
│ - Custom metadata (__metadata__ key)    │
├─────────────────────────────────────────┤
│ Tensor Data (contiguous binary)         │
│ - Aligned to 8-byte boundaries          │
│ - Ordered by dtype then name            │
└─────────────────────────────────────────┘
```

## Saving to SafeTensors

```rust
use entrenar::io::{Model, ModelMetadata, save_model, SaveConfig, ModelFormat};
use entrenar::Tensor;

// Create model with parameters
let params = vec![
    ("model.embed_tokens.weight".to_string(),
     Tensor::from_vec(vec![0.1; 4096 * 768], false)),
    ("model.layers.0.self_attn.q_proj.weight".to_string(),
     Tensor::from_vec(vec![0.01; 768 * 768], false)),
];

let metadata = ModelMetadata::new("my-llm", "llama");
let model = Model::new(metadata, params);

// Save as SafeTensors
let config = SaveConfig::new(ModelFormat::SafeTensors);
save_model(&model, "model.safetensors", &config)?;
```

## Loading from SafeTensors

Format is auto-detected from file extension:

```rust
use entrenar::io::load_model;

let model = load_model("model.safetensors")?;

// Access metadata
println!("Model: {}", model.metadata.name);
println!("Architecture: {}", model.metadata.architecture);

// Access tensors
for (name, tensor) in &model.parameters {
    println!("{}: {} elements", name, tensor.len());
}
```

## Custom Metadata

SafeTensors supports custom metadata stored in the `__metadata__` header field:

```rust
// When saving, metadata is automatically included:
// - name: model name
// - architecture: model architecture
// - version: model version

// For merge operations, additional metadata is added:
// - merge_method: TIES, DARE, SLERP, or Average
// - tensor_count: number of tensors
```

## CLI Usage

### Merge to SafeTensors

```bash
# Output format is detected from extension
entrenar merge model1.safetensors model2.safetensors \
    --method ties \
    --output merged.safetensors
```

### Inspect SafeTensors

```bash
# Use entrenar-inspect crate
entrenar-inspect model.safetensors
```

## Memory-Mapped Loading

For large models, use memory mapping to avoid loading entire file into RAM:

```rust
use memmap2::MmapOptions;
use safetensors::SafeTensors;

let file = std::fs::File::open("large_model.safetensors")?;
let mmap = unsafe { MmapOptions::new().map(&file)? };
let tensors = SafeTensors::deserialize(&mmap)?;

// Tensors are loaded on-demand from mmap
for name in tensors.names() {
    let tensor = tensors.tensor(name)?;
    // Process tensor...
}
```

## HuggingFace Hub Integration

Models saved in SafeTensors format can be directly uploaded:

```bash
# Using HuggingFace CLI
huggingface-cli upload my-org/my-model ./model.safetensors

# Or programmatically
huggingface-cli repo create my-org/my-model
huggingface-cli upload my-org/my-model ./model.safetensors model.safetensors
```

And downloaded models load directly:

```rust
use entrenar::hf_pipeline::HfModelFetcher;
use entrenar::io::load_model;

let fetcher = HfModelFetcher::new()?;
let artifact = fetcher.download_model(
    "microsoft/codebert-base",
    Default::default()
)?;

let model = load_model(&artifact.path)?;
```

## Performance

| Model Size | Save Time | Load Time | File Size |
|------------|-----------|-----------|-----------|
| 100MB      | ~100ms    | ~50ms     | 100MB     |
| 1GB        | ~1s       | ~500ms    | 1GB       |
| 7GB        | ~7s       | ~3s       | 7GB       |

Compare to JSON format:

| Model Size | JSON Save | JSON Load | JSON Size |
|------------|-----------|-----------|-----------|
| 100MB      | ~3s       | ~2.5s     | ~300MB    |
| 1GB        | ~30s      | ~25s      | ~3GB      |

## Error Handling

```rust
use entrenar::io::load_model;
use entrenar::Error;

match load_model("model.safetensors") {
    Ok(model) => {
        println!("Loaded {} tensors", model.parameters.len());
    }
    Err(Error::Serialization(msg)) => {
        // Invalid SafeTensors format
        eprintln!("Parse error: {}", msg);
    }
    Err(Error::Io(e)) => {
        // File not found, permission denied, etc.
        eprintln!("IO error: {}", e);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## See Also

- [JSON Format](json-format.md) - Human-readable alternative
- [YAML Format](yaml-format.md) - Configuration-friendly format
- [GGUF Format](gguf-format.md) - Quantized model format
- [Knowledge Distillation](../distillation/what-is-distillation.md) - Using with distillation
