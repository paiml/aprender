# Realizar GGUF Export

The ecosystem module provides GGUF export functionality for model
quantization and distribution via integration with Realizar.

## Quantization Types

```rust
use entrenar::ecosystem::QuantizationType;

// Available quantization types
let types = [
    QuantizationType::Q2K,   // 2.5 bits, extreme compression
    QuantizationType::Q3KM,  // 3.5 bits, aggressive compression
    QuantizationType::Q4KM,  // 4.5 bits, recommended balance
    QuantizationType::Q5KM,  // 5.5 bits, higher quality
    QuantizationType::Q6K,   // 6.5 bits, high quality
    QuantizationType::Q80,   // 8 bits, highest quantized quality
    QuantizationType::F16,   // 16 bits, no quantization
    QuantizationType::F32,   // 32 bits, full precision
];
```

### Quantization Properties

| Type | Bits/Weight | Quality Score | Size Ratio |
|------|-------------|---------------|------------|
| Q2_K | 2.5 | 50 | 0.078x |
| Q3_K_M | 3.5 | 65 | 0.109x |
| Q4_K_M | 4.5 | 78 | 0.141x |
| Q5_K_M | 5.5 | 85 | 0.172x |
| Q6_K | 6.5 | 92 | 0.203x |
| Q8_0 | 8.0 | 97 | 0.250x |
| F16 | 16.0 | 100 | 0.500x |
| F32 | 32.0 | 100 | 1.000x |

### Type Methods

```rust
let quant = QuantizationType::Q4KM;

println!("Type: {}", quant.as_str());           // "Q4_K_M"
println!("Bits: {}", quant.bits_per_weight());  // 4.5
println!("Quality: {}", quant.quality_score()); // 78

// Estimate output size
let original_size = 14_000_000_000u64; // 14GB model
let estimated = quant.estimate_size(original_size);
println!("Estimated size: {:.2} GB", estimated as f64 / 1e9);
```

### Parsing

```rust
// Parse from string (case-insensitive)
assert_eq!(QuantizationType::parse("Q4_K_M"), Some(QuantizationType::Q4KM));
assert_eq!(QuantizationType::parse("q4km"), Some(QuantizationType::Q4KM));
assert_eq!(QuantizationType::parse("F16"), Some(QuantizationType::F16));
assert_eq!(QuantizationType::parse("fp16"), Some(QuantizationType::F16));
```

## GgufExporter

The main export interface:

```rust
use entrenar::ecosystem::{GgufExporter, QuantizationType};

let exporter = GgufExporter::new(QuantizationType::Q4KM)
    .with_threads(8)
    .without_validation();
```

### Adding Metadata

```rust
use entrenar::ecosystem::{GgufExporter, GeneralMetadata, ExperimentProvenance};

let exporter = GgufExporter::new(QuantizationType::Q5KM)
    .with_general(GeneralMetadata::new("llama", "my-finetuned-model")
        .with_author("PAIML")
        .with_description("LoRA fine-tuned LLaMA model")
        .with_license("MIT"))
    .with_provenance(ExperimentProvenance::new("exp-001", "run-123")
        .with_config_hash("abc123def456")
        .with_dataset("custom-dataset")
        .with_base_model("llama-2-7b")
        .with_metric("loss", 0.125)
        .with_metric("accuracy", 0.92)
        .with_git_commit("deadbeef")
        .with_custom("framework", "entrenar"));
```

## Experiment Provenance

Track model lineage and training metadata:

```rust
use entrenar::ecosystem::ExperimentProvenance;

let provenance = ExperimentProvenance::new("experiment-id", "run-id")
    .with_config_hash("sha256-of-config")
    .with_dataset("imagenet-1k")
    .with_base_model("llama-7b")
    .with_metric("final_loss", 0.123)
    .with_metric("perplexity", 4.56)
    .with_git_commit("abc123")
    .with_custom("trainer", "entrenar")
    .with_custom("epochs", "10");

// Convert to GGUF metadata pairs
let pairs = provenance.to_metadata_pairs();
for (key, value) in &pairs {
    println!("{}: {}", key, value);
}
// entrenar.experiment_id: experiment-id
// entrenar.run_id: run-id
// entrenar.timestamp: 2024-01-15T10:30:00Z
// entrenar.metric.final_loss: 0.123
// entrenar.custom.trainer: entrenar
```

## General Metadata

Standard GGUF metadata fields:

```rust
use entrenar::ecosystem::GeneralMetadata;

let general = GeneralMetadata::new("mistral", "my-model")
    .with_author("Your Name")
    .with_description("Fine-tuned Mistral model for code generation")
    .with_license("Apache-2.0");
```

## Exporting Models

```rust
let exporter = GgufExporter::new(QuantizationType::Q4KM)
    .with_general(general)
    .with_provenance(provenance);

// Export model
let result = exporter.export("input_model.safetensors", "output_model.gguf")?;

println!("Output: {:?}", result.output_path);
println!("Quantization: {}", result.quantization);
println!("Metadata keys: {}", result.metadata_keys);
println!("Estimated size: {} bytes", result.estimated_size_bytes);
```

### Collecting Metadata

Get all metadata as key-value pairs:

```rust
let pairs = exporter.collect_metadata();

for (key, value) in &pairs {
    println!("{} = {}", key, value);
}
```

## Error Handling

```rust
use entrenar::ecosystem::GgufExportError;

match exporter.export("model.safetensors", "model.gguf") {
    Ok(result) => println!("Exported to {:?}", result.output_path),
    Err(GgufExportError::InvalidQuantization(msg)) => {
        eprintln!("Quantization error: {}", msg);
    }
    Err(GgufExportError::IoError(msg)) => {
        eprintln!("I/O error: {}", msg);
    }
    Err(e) => eprintln!("Export failed: {}", e),
}
```

## Integration with Research Artifacts

Combine with research module for full provenance:

```rust
use entrenar::research::ResearchArtifact;
use entrenar::ecosystem::{GgufExporter, ExperimentProvenance};

// Create research artifact
let artifact = ResearchArtifact::new(
    "model-artifact",
    "Fine-tuned LLaMA for Code",
    ArtifactType::Model,
    License::Mit,
);

// Create provenance from artifact
let provenance = ExperimentProvenance::new(&artifact.id, "run-001")
    .with_custom("artifact_version", &artifact.version);

// Export with full provenance
let exporter = GgufExporter::new(QuantizationType::Q4KM)
    .with_provenance(provenance);
```

## See Also

- [Ecosystem Overview](./overview.md)
- [Model I/O](../io/overview.md) - Other model formats
- [GGUF Format](../io/gguf-format.md) - GGUF specification
