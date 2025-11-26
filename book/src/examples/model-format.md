# Case Study: Model Serialization (.apr Format)

Save and load ML models with built-in quality: checksums, signatures, encryption, WASM compatibility.

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

## WASM Compatibility (Hard Requirement)

The `.apr` format is designed for **universal deployment**. Every feature works in:

- Native (Linux, macOS, Windows)
- WASM (browsers, Cloudflare Workers, Vercel Edge)
- Embedded (no_std with alloc)

```rust
// Same model works everywhere
#[cfg(target_arch = "wasm32")]
async fn load_in_browser() -> Result<LinearRegression> {
    let bytes = fetch("https://models.example.com/house-prices.apr").await?;
    load_from_bytes(&bytes, ModelType::LinearRegression)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_native() -> Result<LinearRegression> {
    load("house-prices.apr", ModelType::LinearRegression)
}
```

**Why this matters:**
- Train once, deploy anywhere
- Browser-based ML demos
- Edge inference (low latency)
- Serverless functions

## Format Structure

```text
┌─────────────────────────────────────────┐
│ Header (32 bytes, fixed)                │ ← Magic, version, type, sizes
├─────────────────────────────────────────┤
│ Metadata (variable, MessagePack)        │ ← Hyperparameters, metrics
├─────────────────────────────────────────┤
│ Salt + Nonce (if ENCRYPTED)             │ ← Security parameters
├─────────────────────────────────────────┤
│ Payload (variable, compressed)          │ ← Model weights (bincode)
├─────────────────────────────────────────┤
│ Signature (if SIGNED)                   │ ← Ed25519 signature
├─────────────────────────────────────────┤
│ License (if LICENSED)                   │ ← Commercial protection
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
| 0x0008 | NaiveBayes | Probabilistic classification |
| 0x0009 | Knn | Distance-based classification |
| 0x000A | Svm | Support vector machine |
| 0x0010 | NgramLm | Language modeling |
| 0x0011 | TfIdf | Text vectorization |
| 0x0012 | CountVectorizer | Bag of words |
| 0x0020 | NeuralSequential | Deep learning |
| 0x0021 | NeuralCustom | Custom architectures |
| 0x0030 | ContentRecommender | Recommendations |
| 0x00FF | Custom | User-defined |

## Encryption (Feature: `format-encryption`)

### Password-Based (Personal/Team)

```rust
use aprender::format::{save_encrypted, load_encrypted};

// Save with password (Argon2id + AES-256-GCM)
save_encrypted(&model, ModelType::LinearRegression, "secure.apr",
    SaveOptions::default(), "my-strong-password")?;

// Load with password
let model: LinearRegression = load_encrypted("secure.apr",
    ModelType::LinearRegression, "my-strong-password")?;
```

**Security properties:**
- Argon2id: Memory-hard, GPU-resistant key derivation
- AES-256-GCM: Authenticated encryption (detects tampering)
- Random salt: Same password produces different ciphertexts

### Recipient-Based (Commercial Distribution)

```rust
use aprender::format::{save_for_recipient, load_as_recipient};
use x25519_dalek::{PublicKey, StaticSecret};

// Generate buyer's keypair (done once by buyer)
let buyer_secret = StaticSecret::random_from_rng(&mut rng);
let buyer_public = PublicKey::from(&buyer_secret);

// Seller encrypts for buyer's public key (no password sharing!)
save_for_recipient(&model, ModelType::LinearRegression, "commercial.apr",
    SaveOptions::default(), &buyer_public)?;

// Only buyer's secret key can decrypt
let model: LinearRegression = load_as_recipient("commercial.apr",
    ModelType::LinearRegression, &buyer_secret)?;
```

**Benefits:**
- No password sharing required
- Cryptographically bound to buyer (non-transferable)
- Forward secrecy via ephemeral sender keys
- Perfect for model marketplaces

## Digital Signatures (Feature: `format-signing`)

Verify model provenance:

```rust
use aprender::format::{save_signed, load_verified};
use ed25519_dalek::{SigningKey, VerifyingKey};

// Generate seller's keypair (done once)
let signing_key = SigningKey::generate(&mut rng);
let verifying_key = VerifyingKey::from(&signing_key);

// Sign model with private key
save_signed(&model, ModelType::LinearRegression, "signed.apr",
    SaveOptions::default(), &signing_key)?;

// Verify signature before loading (reject tampering)
let model: LinearRegression = load_verified("signed.apr",
    ModelType::LinearRegression, Some(&verifying_key))?;
```

**Use cases:**
- Model marketplaces (verify seller identity)
- Compliance (audit trail)
- Supply chain security

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

## WASM Loading Patterns

### Browser (Fetch API)

```rust
#[cfg(target_arch = "wasm32")]
pub async fn load_from_url<M: DeserializeOwned>(
    url: &str,
    model_type: ModelType,
) -> Result<M> {
    let response = fetch(url).await?;
    let bytes = response.bytes().await?;
    load_from_bytes(&bytes, model_type)
}

// Usage
let model = load_from_url::<LinearRegression>(
    "https://models.example.com/house-prices.apr",
    ModelType::LinearRegression
).await?;
```

### IndexedDB Cache

```rust
#[cfg(target_arch = "wasm32")]
pub async fn load_cached<M: DeserializeOwned>(
    cache_key: &str,
    url: &str,
    model_type: ModelType,
) -> Result<M> {
    // Try cache first
    if let Some(bytes) = idb_get(cache_key).await? {
        return load_from_bytes(&bytes, model_type);
    }

    // Fetch and cache
    let bytes = fetch(url).await?.bytes().await?;
    idb_set(cache_key, &bytes).await?;
    load_from_bytes(&bytes, model_type)
}
```

### Graceful Degradation

Some features are native-only (STREAMING, TRUENO_NATIVE). In WASM, they're silently ignored:

```rust
// This works in both native and WASM
let options = SaveOptions::default()
    .with_compression(Compression::ZstdDefault)  // Works everywhere
    .with_streaming(true);  // Ignored in WASM, no error

// WASM: loads via in-memory path
// Native: uses mmap for large models
let model: LinearRegression = load("model.apr", ModelType::LinearRegression)?;
```

## Ecosystem Integration

The `.apr` format coordinates with alimentar's `.ald` dataset format:

```text
Training Pipeline (Native):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ dataset.ald │ → │  aprender   │ → │  model.apr  │
│ (alimentar) │    │  training   │    │  (aprender) │
└─────────────┘    └─────────────┘    └─────────────┘

Inference Pipeline (WASM):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Fetch .apr  │ → │   aprender  │ → │ Prediction  │
│ from CDN    │    │  inference  │    │ in browser  │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Shared properties:**
- Same crypto stack (aes-gcm, ed25519-dalek, x25519-dalek)
- Same WASM compatibility requirements
- Same Toyota Way principles (Jidoka, checksums, signatures)

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

**Use cases:**
- HIPAA-compliant medical inference
- GDPR-compliant EU data processing
- Financial data analysis
- Zero-trust ML APIs

## Toyota Way Principles

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** | CRC32 checksum stops on corruption |
| **Jidoka** | Type verification stops on mismatch |
| **Jidoka** | Signature verification stops on tampering |
| **Jidoka** | Decryption fails on wrong key (authenticated) |
| **Genchi Genbutsu** | `inspect()` to see actual file contents |
| **Kaizen** | Semantic versioning for format evolution |
| **Heijunka** | Graceful degradation (WASM ignores native-only flags) |

## Error Handling

```rust
use aprender::error::AprenderError;

match load::<LinearRegression>("model.apr", ModelType::LinearRegression) {
    Ok(model) => { /* use model */ },
    Err(AprenderError::ChecksumMismatch { expected, actual }) => {
        eprintln!("File corrupted: expected {:08X}, got {:08X}", expected, actual);
    },
    Err(AprenderError::ModelTypeMismatch { expected, found }) => {
        eprintln!("Wrong model type: expected {:?}, found {:?}", expected, found);
    },
    Err(AprenderError::SignatureInvalid) => {
        eprintln!("Signature verification failed - model may be tampered");
    },
    Err(AprenderError::DecryptionFailed) => {
        eprintln!("Decryption failed - wrong password or key");
    },
    Err(AprenderError::UnsupportedVersion { found, supported }) => {
        eprintln!("Version {}.{} not supported (max {}.{})",
            found.0, found.1, supported.0, supported.1);
    },
    Err(e) => eprintln!("Error: {}", e),
}
```

## Feature Flags

| Feature | Crates Added | Binary Size | WASM |
|---------|--------------|-------------|------|
| (core) | bincode, rmp-serde | ~60KB | ✓ |
| `format-compression` | zstd | +250KB | ✓ |
| `format-signing` | ed25519-dalek | +150KB | ✓ |
| `format-encryption` | aes-gcm, argon2, x25519-dalek, hkdf, sha2 | +180KB | ✓ |

```toml
# Cargo.toml
[dependencies]
aprender = { version = "0.9", features = ["format-encryption", "format-signing"] }
```

## Specification

Full specification: [docs/specifications/model-format-spec.md](https://github.com/paiml/aprender/blob/main/docs/specifications/model-format-spec.md)

**Key properties:**
- Pure Rust (Sovereign AI, zero C/C++ dependencies)
- WASM compatibility (hard requirement, spec §1.0)
- 32-byte fixed header for fast scanning
- MessagePack metadata (compact, fast)
- bincode payload (zero-copy potential)
- CRC32 integrity, Ed25519 signatures, AES-256-GCM encryption
- trueno-native mode for zero-copy SIMD inference (native only)
