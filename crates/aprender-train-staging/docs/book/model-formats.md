# Model Formats & I/O

This chapter covers entrenar's model serialization and deserialization capabilities,
including SafeTensors support for HuggingFace compatibility.

## Supported Formats

Entrenar supports multiple model formats for different use cases:

| Format | Extension | Use Case | HF Compatible |
|--------|-----------|----------|---------------|
| SafeTensors | `.safetensors` | Production models, HuggingFace Hub | Yes |
| JSON | `.json` | Debugging, human-readable | No |
| YAML | `.yaml`, `.yml` | Configuration, small models | No |

## SafeTensors Format

SafeTensors is the recommended format for production use. It provides:

- **Security**: No arbitrary code execution (unlike pickle)
- **Efficiency**: Zero-copy tensor loading with memory mapping
- **Compatibility**: Direct upload/download from HuggingFace Hub
- **Metadata**: Custom metadata storage (model name, architecture, etc.)

### Saving Models

```rust
use entrenar::io::{Model, ModelMetadata, save_model, SaveConfig, ModelFormat};
use entrenar::Tensor;

// Create a model
let params = vec![
    ("encoder.weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0], false)),
    ("encoder.bias".to_string(), Tensor::from_vec(vec![0.1], false)),
];

let metadata = ModelMetadata::new("my-model", "transformer")
    .with_custom("version", serde_json::json!("1.0.0"))
    .with_custom("hidden_size", serde_json::json!(768));

let model = Model::new(metadata, params);

// Save as SafeTensors
let config = SaveConfig::new(ModelFormat::SafeTensors);
save_model(&model, "model.safetensors", &config)?;

// Or save as JSON for debugging
let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
save_model(&model, "model.json", &config)?;
```

### Loading Models

```rust
use entrenar::io::load_model;

// Format is auto-detected from extension
let model = load_model("model.safetensors")?;

println!("Model: {}", model.metadata.name);
println!("Architecture: {}", model.metadata.architecture);
println!("Parameters: {}", model.parameters.len());

// Access individual tensors
if let Some(tensor) = model.get_parameter("encoder.weight") {
    println!("encoder.weight shape: {}", tensor.len());
}
```

## CLI Commands

### Model Merging with SafeTensors Output

The `entrenar merge` command automatically detects output format from the file extension:

```bash
# Output as SafeTensors (HuggingFace compatible)
entrenar merge model1.safetensors model2.safetensors \
    --method ties \
    --output merged.safetensors

# Output as JSON (for debugging)
entrenar merge model1.safetensors model2.safetensors \
    --method slerp \
    --output merged.json
```

Merge methods available:
- `ties` - Task Inference via Elimination and Sign voting
- `dare` - Drop And REscale for stochastic merging
- `slerp` - Spherical Linear intERPolation
- `average` - Simple weighted average

### Model Quantization

```bash
# Quantize to 4-bit
entrenar quantize model.safetensors \
    --bits 4 \
    --method symmetric \
    --output model_q4.json

# Quantize to 8-bit with per-channel
entrenar quantize model.safetensors \
    --bits 8 \
    --method asymmetric \
    --per-channel \
    --output model_q8.json
```

Note: Quantized models are saved as JSON because they use custom block formats
(Q4_0, Q8_0) that are not directly compatible with SafeTensors.

## API Reference

### ModelFormat

```rust
pub enum ModelFormat {
    /// JSON format (human-readable, larger file size)
    Json,

    /// YAML format (human-readable, good for configs)
    Yaml,

    /// SafeTensors format (HuggingFace compatible, efficient binary)
    SafeTensors,
}
```

### SaveConfig

```rust
let config = SaveConfig::new(ModelFormat::SafeTensors)
    .with_pretty(true)    // For text formats
    .with_compress(true); // Future: compression support
```

### ModelMetadata

```rust
let metadata = ModelMetadata::new("model-name", "architecture")
    .with_custom("key", serde_json::json!("value"));
```

## SafeTensors Metadata

When saving to SafeTensors format, entrenar automatically includes metadata:

```json
{
  "name": "model-name",
  "architecture": "transformer",
  "version": "0.1.0"
}
```

This metadata is preserved during save/load roundtrips and can be accessed via:

```rust
let model = load_model("model.safetensors")?;
println!("Name: {}", model.metadata.name);
println!("Architecture: {}", model.metadata.architecture);
```

## Integration with HuggingFace Hub

Models saved in SafeTensors format can be directly uploaded to HuggingFace Hub:

```bash
# After saving with entrenar
huggingface-cli upload my-org/my-model ./merged.safetensors

# Or use the HuggingFace Python API
from huggingface_hub import upload_file
upload_file("merged.safetensors", "my-org/my-model", "model.safetensors")
```

Models downloaded from HuggingFace Hub can be loaded directly:

```rust
use entrenar::hf_pipeline::HfModelFetcher;

let fetcher = HfModelFetcher::new()?;
let artifact = fetcher.download_model("microsoft/codebert-base", Default::default())?;

// The artifact.path points to the SafeTensors file
let model = load_model(&artifact.path)?;
```

## Performance Considerations

### Memory Mapping

SafeTensors supports memory-mapped loading for large models:

```rust
use memmap2::MmapOptions;
use safetensors::SafeTensors;

let file = std::fs::File::open("large_model.safetensors")?;
let mmap = unsafe { MmapOptions::new().map(&file)? };
let tensors = SafeTensors::deserialize(&mmap)?;
```

This allows loading models larger than available RAM by lazily loading tensors.

### Format Comparison

| Operation | SafeTensors | JSON | YAML |
|-----------|-------------|------|------|
| Save 1GB model | ~1s | ~30s | ~45s |
| Load 1GB model | ~0.5s | ~25s | ~40s |
| File size | 1x | ~3x | ~3.5x |
| Memory-map | Yes | No | No |

## Error Handling

```rust
use entrenar::io::load_model;
use entrenar::Error;

match load_model("model.safetensors") {
    Ok(model) => println!("Loaded: {}", model.metadata.name),
    Err(Error::Serialization(msg)) => eprintln!("Parse error: {}", msg),
    Err(Error::Io(e)) => eprintln!("IO error: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## See Also

- [HuggingFace Distillation](huggingface-distillation.md) - Using SafeTensors with distillation
- [Real-time Monitoring](real-time-monitoring.md) - Training visualization
- [CITL Patterns](citl.md) - Continuous integration for training
