# Case Study: Model Serialization (.apr Format)

Save and load ML models with built-in quality: checksums, signatures, encryption.

## Quick Start

```rust
use aprender::format::{save, load, ModelType, SaveOptions};
use aprender::linear_model::LinearRegression;

// Train model
let mut model = LinearRegression::new();
model.fit(&x, &y)?;

// Save
save(&model, ModelType::LinearRegression, "model.apr", SaveOptions::default())?;

// Load
let loaded: LinearRegression = load("model.apr", ModelType::LinearRegression)?;
```

## Format Structure

```text
┌─────────────────────────────────────────┐
│ Header (32 bytes, fixed)                │ ← Magic, version, type, sizes
├─────────────────────────────────────────┤
│ Metadata (variable, MessagePack)        │ ← Hyperparameters, metrics
├─────────────────────────────────────────┤
│ Payload (variable, compressed)          │ ← Model weights (bincode)
├─────────────────────────────────────────┤
│ Checksum (4 bytes, CRC32)               │ ← Integrity verification
└─────────────────────────────────────────┘
```

## Built-in Quality (Jidoka)

### CRC32 Checksum

Every `.apr` file has a CRC32 checksum. Corruption is detected immediately:

```rust
// Automatic verification on load
let model: LinearRegression = load("model.apr", ModelType::LinearRegression)?;
// If checksum fails: AprenderError::ChecksumMismatch { expected, actual }
```

### Type Safety

Model type is encoded in header. Loading wrong type fails fast:

```rust
// Saved as LinearRegression
save(&lr_model, ModelType::LinearRegression, "lr.apr", opts)?;

// Attempt to load as KMeans - fails immediately
let result: Result<KMeans> = load("lr.apr", ModelType::KMeans);
// Error: "Model type mismatch: file contains LinearRegression, expected KMeans"
```

## Metadata

Store hyperparameters, metrics, and custom data:

```rust
let options = SaveOptions::default()
    .with_name("house-price-predictor")
    .with_description("Trained on Boston Housing dataset");

// Add hyperparameters
options.metadata.hyperparameters.insert(
    "learning_rate".to_string(),
    serde_json::json!(0.01)
);

// Add metrics
options.metadata.metrics.insert(
    "r2_score".to_string(),
    serde_json::json!(0.95)
);

save(&model, ModelType::LinearRegression, "model.apr", options)?;
```

## Inspection Without Loading

Check model info without deserializing weights:

```rust
use aprender::format::inspect;

let info = inspect("model.apr")?;
println!("Model type: {:?}", info.model_type);
println!("Format version: {}.{}", info.format_version.0, info.format_version.1);
println!("Payload size: {} bytes", info.payload_size);
println!("Created: {}", info.metadata.created_at);
println!("Encrypted: {}", info.encrypted);
println!("Signed: {}", info.signed);
```

## Model Types

| Value | Type | Use Case |
|-------|------|----------|
| 0x0001 | LinearRegression | Regression |
| 0x0002 | LogisticRegression | Binary classification |
| 0x0003 | DecisionTree | Interpretable classification |
| 0x0004 | RandomForest | Ensemble classification |
| 0x0005 | GradientBoosting | High-performance ensemble |
| 0x0006 | KMeans | Clustering |
| 0x0007 | Pca | Dimensionality reduction |
| 0x0010 | NgramLm | Language modeling |
| 0x0020 | NeuralSequential | Deep learning |
| 0x0030 | ContentRecommender | Recommendations |
| 0x00FF | Custom | User-defined |

## Encryption (Feature: `format-encryption`)

### Password-Based

```rust
use aprender::format::{save_encrypted, load_encrypted};

// Save with password
save_encrypted(&model, ModelType::LinearRegression, "secure.apr",
    SaveOptions::default(), "my-password")?;

// Load with password
let model: LinearRegression = load_encrypted("secure.apr",
    ModelType::LinearRegression, "my-password")?;
```

Uses Argon2id key derivation + AES-256-GCM.

### Recipient-Based (Asymmetric)

```rust
use aprender::format::{save_for_recipient, load_with_key};

// Seller encrypts for buyer's public key
save_for_recipient(&model, ModelType::LinearRegression, "commercial.apr",
    SaveOptions::default(), &buyer_public_key)?;

// Only buyer's private key can decrypt
let model: LinearRegression = load_with_key("commercial.apr",
    ModelType::LinearRegression, &buyer_private_key)?;
```

Uses X25519 key agreement + AES-256-GCM. No password sharing needed.

## Digital Signatures (Feature: `format-signing`)

Verify model provenance:

```rust
use aprender::format::{save_signed, load_verified};

// Sign with private key
save_signed(&model, ModelType::LinearRegression, "signed.apr",
    SaveOptions::default(), &signing_key)?;

// Verify with public key before loading
let model: LinearRegression = load_verified("signed.apr",
    ModelType::LinearRegression, &[trusted_public_key])?;
```

Uses Ed25519 signatures.

## Compression (Feature: `format-compression`)

```rust
use aprender::format::{Compression, SaveOptions};

let options = SaveOptions::default()
    .with_compression(Compression::ZstdDefault);  // Level 3, good balance

// Or maximum compression for archival
let archival = SaveOptions::default()
    .with_compression(Compression::ZstdMax);  // Level 19
```

| Algorithm | Ratio | Speed | Use Case |
|-----------|-------|-------|----------|
| None | 1:1 | Instant | Debugging |
| ZstdDefault | ~3:1 | Fast | Distribution |
| ZstdMax | ~4:1 | Slow | Archival |
| LZ4 | ~2:1 | Very fast | Streaming |

## Private Inference (HIPAA/GDPR)

For sensitive data, use bidirectional encryption:

```rust
// Model publishes public key in metadata
let info = inspect("medical-model.apr")?;
let model_pub_key = info.metadata.custom.get("inference_pub_key");

// User encrypts input with model's public key
let encrypted_input = encrypt_for_model(&patient_data, model_pub_key)?;

// Send encrypted_input to model owner
// Model owner decrypts, runs inference, encrypts response with user's public key
// Only user can decrypt the prediction
```

No intermediary sees the data. Zero-trust ML.

## Toyota Way Principles

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** | CRC32 checksum stops on corruption |
| **Jidoka** | Type verification stops on mismatch |
| **Jidoka** | Signature verification stops on tampering |
| **Genchi Genbutsu** | `inspect()` to see actual file contents |
| **Kaizen** | Semantic versioning for format evolution |

## Error Handling

```rust
use aprender::error::AprenderError;

match load::<LinearRegression>("model.apr", ModelType::LinearRegression) {
    Ok(model) => { /* use model */ },
    Err(AprenderError::ChecksumMismatch { expected, actual }) => {
        eprintln!("File corrupted: expected {:08X}, got {:08X}", expected, actual);
    },
    Err(AprenderError::UnsupportedVersion { found, supported }) => {
        eprintln!("Version {}.{} not supported (max {}.{})",
            found.0, found.1, supported.0, supported.1);
    },
    Err(AprenderError::FormatError { message }) => {
        eprintln!("Invalid format: {}", message);
    },
    Err(e) => eprintln!("Error: {}", e),
}
```

## Specification

Full specification: [docs/specifications/model-format-spec.md](https://github.com/paiml/aprender/blob/main/docs/specifications/model-format-spec.md)

**Key properties:**
- Pure Rust (Sovereign AI, zero C/C++ dependencies)
- 32-byte fixed header for fast scanning
- MessagePack metadata (compact, fast)
- bincode payload (zero-copy potential)
- CRC32 integrity, Ed25519 signatures, AES-256-GCM encryption
- trueno-native mode for zero-copy SIMD inference
