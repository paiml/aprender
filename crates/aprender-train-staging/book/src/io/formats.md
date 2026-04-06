# Supported Formats

Entrenar supports multiple model serialization formats for different use cases.

## Format Comparison

| Format | Extension | Binary | HF Compatible | Use Case |
|--------|-----------|--------|---------------|----------|
| SafeTensors | `.safetensors` | Yes | Yes | Production, sharing |
| JSON | `.json` | No | No | Debugging, inspection |
| YAML | `.yaml` | No | No | Configuration |
| GGUF | `.gguf` | Yes | No | Quantized models |

## Choosing a Format

### SafeTensors (Recommended)

Use SafeTensors for:
- Production deployments
- Uploading to HuggingFace Hub
- Large models (supports memory mapping)
- Security-sensitive applications

```rust
let config = SaveConfig::new(ModelFormat::SafeTensors);
save_model(&model, "model.safetensors", &config)?;
```

### JSON

Use JSON for:
- Debugging and inspection
- Small models
- Human-readable output
- Version control diffs

```rust
let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
save_model(&model, "model.json", &config)?;
```

### YAML

Use YAML for:
- Configuration files
- Human-friendly syntax
- Small models with metadata focus

```rust
let config = SaveConfig::new(ModelFormat::Yaml);
save_model(&model, "model.yaml", &config)?;
```

### GGUF (Future)

GGUF format will be supported for:
- Quantized model export
- LLaMA.cpp compatibility
- Integration with Realizar crate

## Format Detection

Format is auto-detected from file extension:

```rust
// Auto-detect based on extension
let model = load_model("model.safetensors")?;  // SafeTensors
let model = load_model("model.json")?;         // JSON
let model = load_model("model.yaml")?;         // YAML
let model = load_model("config.yml")?;         // YAML (alternate extension)
```

## Performance Characteristics

| Format | 100MB Save | 100MB Load | Compression |
|--------|------------|------------|-------------|
| SafeTensors | ~100ms | ~50ms | 1x |
| JSON | ~3s | ~2.5s | ~0.33x |
| YAML | ~4s | ~3.5s | ~0.29x |

## See Also

- [SafeTensors Format](safetensors-format.md) - Detailed SafeTensors documentation
- [JSON Format](json-format.md) - JSON specifics
- [YAML Format](yaml-format.md) - YAML specifics
- [GGUF Format](gguf-format.md) - Future GGUF support
