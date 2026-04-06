# Registering Models

This example demonstrates semantic versioning for ML models.

## Running the Example

```bash
cargo run --example model_versioning
```

## Semantic Versioning for ML

Pacha uses semantic versioning adapted for machine learning:

| Version | When to Bump | Example |
|---------|--------------|---------|
| **MAJOR** | Architecture change (incompatible inputs/outputs) | Logistic → Transformer |
| **MINOR** | Retraining with new data (backward compatible) | Q2 → Q3 data |
| **PATCH** | Bug fixes, quantization, optimization | INT8 quantization |

## Version Examples

### Initial Release (1.0.0)

```rust
let card = ModelCard::builder()
    .description("Initial fraud detector - logistic regression")
    .metrics([("auc", 0.85)])
    .build();

registry.register_model(
    "fraud-detector",
    &ModelVersion::new(1, 0, 0),
    model_weights,
    card,
)?;
```

### Patch Version (1.0.1) - Quantization

```rust
registry.register_model(
    "fraud-detector",
    &ModelVersion::new(1, 0, 1),
    quantized_weights,
    card,
)?;
```

### Minor Version (1.1.0) - Retrained

```rust
registry.register_model(
    "fraud-detector",
    &ModelVersion::new(1, 1, 0),
    retrained_weights,
    card,
)?;
```

### Major Version (2.0.0) - New Architecture

```rust
registry.register_model(
    "fraud-detector",
    &ModelVersion::new(2, 0, 0),
    transformer_weights,
    card,
)?;
```

## Version Comparison

```rust
let v1 = ModelVersion::new(1, 0, 0);
let v1_1 = ModelVersion::new(1, 1, 0);
let v2 = ModelVersion::new(2, 0, 0);

assert!(v1 < v1_1);   // Minor version is higher
assert!(v1_1 < v2);   // Major version is higher
```

## Pre-release Versions

```rust
let beta = ModelVersion::new(2, 1, 0).with_prerelease("beta.1");
println!("{}", beta);             // "2.1.0-beta.1"
println!("{}", beta.is_stable()); // false
```
