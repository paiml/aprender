# Aprender Model Format Specification (.apr)

**Version:** 1.7.0
**Status:** Partially Implemented
**Author:** paiml
**Reviewer:** Toyota Way AI Agent
**Date:** 2025-11-26
**Updated:** 2025-11-26

### Changelog
- v1.0.0: Initial spec (header, CRC32, basic save/load)
- v1.1.0: Sovereign AI architecture, GGUF export, ONNX out of scope
- v1.2.0: Commercial licensing, watermarking, model marketplace support
- v1.3.0: trueno integration, expanded model types, implementation status tracking
- v1.4.0: **WASM compatibility as HARD REQUIREMENT** (§1.0) - spec gate, mandatory CI testing
- v1.5.0: Model Cards [Mitchell2019], Andon error protocols, supply chain integrity
- v1.6.0: **Full quantization spec** (§6.2) - GGUF-compatible Q8_0/Q4_0/Q4_1, Quantizer trait, plugin architecture, CLI commands, explicit opt-in only
- v1.7.0: **Single binary deployment** (§1.1) - `include_bytes!()` embedding, AWS Lambda ARM, SIMD-first, zero-dependency MLOps

### Implementation Status

| Component | Spec | Implementation | WASM | Action |
|-----------|------|----------------|------|--------|
| **WASM Compat** | §1.0 | ✓ CI added | GATE | ci.yml wasm job |
| **Single Binary** | §1.1 | ✓ load_from_bytes | ✓ | include_bytes!() ready |
| Header (32-byte) | §3 | ✓ | ✓ | - |
| CRC32 checksum | §5.4 | ✓ | ✓ | - |
| save/load/inspect | §2 | ✓ | ✓ | - |
| Model types | §3.1 | ✓ 17 types | ✓ | - |
| Flags | §3.2 | ✓ 6/6 bits | ✓ | - |
| Metadata | §4 | ✓ MessagePack | ✓ | - |
| Compression | §6.1 | ✓ zstd (feature) | ✓ | format-compression feature |
| Encryption (password) | §5.2 | ✓ AES-256-GCM + Argon2id | ✓ | format-encryption feature |
| Encryption (X25519) | §5.3 | ✓ X25519+HKDF+AES-GCM | ✓ | format-encryption feature |
| Signing | §5.4 | ✓ Ed25519 (feature) | ✓ | format-signing feature |
| Quantization | §6.2 | ○ Spec complete | ✓ | format-quantize feature |
| Streaming | §7 | ○ | N/A | format-streaming feature |
| License block | §9 | ○ | ○ | format-commercial feature |
| trueno-native | §8 | ○ | N/A | format-trueno feature |
| GGUF export | §7.2 | ○ | ○ | format-gguf feature |

**Legend:** ✓ Conformant, ✗ Non-conformant (fix required), ○ Not started, N/A = Native only

**⚠️ WASM column must be ✓ for spec conformance. Any ✗ in WASM = entire spec non-conformant.**

## 1. Executive Summary & Scientific Basis

This specification defines the `.apr` (Aprender Model Format), a unified binary format designed for the rigorous lifecycle management of machine learning models. It strictly adheres to **Toyota Way** principles: *Jidoka* (built-in quality via checksums and signatures), *Just-in-Time* (streaming access), and *Standardized Work* (metadata schemas).

### 1.0 WASM Compatibility (HARD REQUIREMENT)

**⚠️ SPECIFICATION GATE: ALL features MUST work in `wasm32-unknown-unknown` target.**

This is not optional. WASM compatibility is a **hard requirement** for the entire specification. Any feature that fails to compile or run correctly under WASM causes the **entire specification to be non-conformant**.

| Requirement | Rationale |
|-------------|-----------|
| Zero C/C++ FFI | WASM cannot link native libraries |
| No `std::fs` in core | Browser has no filesystem |
| No threads in core | WASM threads require SharedArrayBuffer |
| Pure Rust crypto | `ring` forbidden (C/asm); use `*-dalek` crates |
| No `getrandom` default | Must use `js` feature for browser entropy |

**Mandatory Testing:**

```bash
# CI MUST run these tests on every PR
cargo check --target wasm32-unknown-unknown --no-default-features
cargo check --target wasm32-unknown-unknown --features format-encryption,format-signing

# Integration test: save on native, load in WASM
wasm-pack test --node --features format-encryption
```

**Dependency Allowlist (WASM-safe):**

| Crate | WASM Status | Notes |
|-------|-------------|-------|
| `bincode` | ✓ | Pure Rust serialization |
| `rmp-serde` | ✓ | Pure Rust MessagePack |
| `zstd` | ✓ | Requires `wasm32` feature |
| `aes-gcm` | ✓ | Pure Rust AES |
| `argon2` | ✓ | Pure Rust KDF |
| `ed25519-dalek` | ✓ | Pure Rust signatures |
| `x25519-dalek` | ✓ | Pure Rust key exchange |
| `hkdf` + `sha2` | ✓ | Pure Rust KDF |
| `getrandom` | ✓ | With `js` feature only |

**Blocklist (NEVER use):**

| Crate | Reason |
|-------|--------|
| `ring` | Contains C/asm, fails WASM |
| `openssl` | System library, fails WASM |
| `rustls` (default) | Uses `ring` by default |
| `rayon` | Threads not portable to WASM |
| `tokio` | Async runtime not WASM-portable |

**Jidoka Enforcement:**

If any `.apr` feature fails WASM compilation:
1. **Stop the line** - Block all PRs until fixed
2. **Root cause analysis** - Identify offending dependency
3. **Countermeasure** - Replace with pure Rust alternative or feature-gate

This requirement ensures models saved anywhere can be loaded in browsers, edge devices, and serverless WASM runtimes (Cloudflare Workers, Fastly Compute, Vercel Edge).

**Ecosystem Coordination:**

The `.apr` format shares WASM requirements with the alimentar `.ald` dataset format (see `alimentar/docs/specifications/dataset-format-spec.md` §1.0). Both formats use:
- Same crypto stack: `aes-gcm`, `argon2`, `ed25519-dalek`, `x25519-dalek`
- Same HKDF pattern: `apr-v1-encrypt` / `ald-v1-encrypt`
- Same graceful degradation: STREAMING/TRUENO_NATIVE flags ignored in WASM

This enables end-to-end ML pipelines in the browser:
```
.ald dataset (WASM) → aprender model (WASM) → .apr model (WASM) → inference (WASM)
```

**Graceful Degradation:**

When STREAMING (bit 2) or TRUENO_NATIVE (bit 4) flags are set but running in WASM:
- Flags are **silently ignored** (no error)
- Model loads via standard in-memory path
- Performance hint only, not a hard requirement

```rust
#[cfg(target_arch = "wasm32")]
fn load_payload(data: &[u8], flags: u8) -> Result<Model> {
    // Ignore STREAMING/TRUENO_NATIVE flags - process in-memory
    decompress_and_deserialize(data)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_payload(path: &Path, flags: u8) -> Result<Model> {
    if flags & FLAG_STREAMING != 0 {
        load_mmap(path)
    } else if flags & FLAG_TRUENO_NATIVE != 0 {
        load_aligned(path)
    } else {
        load_standard(path)
    }
}
```

### 1.1 Single Binary Deployment (FIRST-CLASS FEATURE)

**⚠️ KEY DIFFERENTIATOR: Embed `.apr` models directly in executables for zero-dependency deployment.**

The `.apr` format is specifically designed for `include_bytes!()` embedding, enabling single-binary ML applications with SIMD performance from day one.

```rust
// Embed model at compile time
const MODEL: &[u8] = include_bytes!("sentiment.apr");
const DATA: &[u8] = include_bytes!("vocab.ald");  // alimentar dataset

fn main() -> Result<()> {
    let model: NgramLm = load_from_bytes(MODEL, ModelType::NgramLm)?;
    let vocab: Dataset = alimentar::load_from_bytes(DATA)?;

    // SIMD inference - no setup, no downloads, no config
    let prediction = model.predict(&input)?;
}
```

**Build → Deploy:**

```bash
cargo build --release --target aarch64-unknown-linux-gnu
# Output: single 5MB binary with model embedded
scp ./app ec2-user@lambda-arm:~/
./app  # runs immediately, NEON SIMD active
```

#### 1.1.1 Deployment Targets

| Target | Binary Size | SIMD | Use Case |
|--------|-------------|------|----------|
| `x86_64-unknown-linux-gnu` | ~5MB | AVX2/AVX-512 | AWS Lambda x86, servers |
| `aarch64-unknown-linux-gnu` | ~4MB | NEON | AWS Lambda ARM (Graviton), RPi |
| `x86_64-apple-darwin` | ~5MB | AVX2 | macOS development |
| `aarch64-apple-darwin` | ~4MB | NEON | Apple Silicon |
| `wasm32-unknown-unknown` | ~500KB | - | Browser, Cloudflare Workers |
| `thumbv7em-none-eabihf` | ~2MB | - | Embedded Cortex-M |

#### 1.1.2 Why This Matters

**Traditional ML Deployment:**
```
Docker image (2GB) → Python runtime → PyTorch → model.pt → CUDA driver → inference
```

**aprender Deployment:**
```
Single binary (5MB) → inference
```

| Metric | Traditional | aprender |
|--------|-------------|----------|
| Cold start | 5-30s | <100ms |
| Memory | 500MB-2GB | 10-50MB |
| Dependencies | Python, CUDA, etc. | None |
| Artifact count | 5-20 files | 1 file |
| ARM support | Complex | Native |

#### 1.1.3 AWS Lambda ARM Example

```rust
use lambda_runtime::{service_fn, LambdaEvent, Error};

const MODEL: &[u8] = include_bytes!("classifier.apr");

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Model loaded once, reused across invocations
    let model: LogisticRegression = load_from_bytes(MODEL, ModelType::LogisticRegression)?;
    lambda_runtime::run(service_fn(|event| handler(event, &model))).await
}

async fn handler(event: LambdaEvent<Request>, model: &LogisticRegression) -> Result<Response, Error> {
    let prediction = model.predict(&event.payload.features)?;  // NEON SIMD on Graviton
    Ok(Response { class: prediction })
}
```

**Lambda config:** 128MB RAM, ARM64, <10ms inference, ~$0.0000002/request.

#### 1.1.4 Ecosystem Integration

Both `.apr` (models) and `.ald` (datasets from alimentar) support embedded deployment:

```rust
// Full ML pipeline in one binary
const TOKENIZER: &[u8] = include_bytes!("tokenizer.ald");
const VECTORIZER: &[u8] = include_bytes!("tfidf.apr");
const CLASSIFIER: &[u8] = include_bytes!("sentiment.apr");

fn pipeline(text: &str) -> Result<Sentiment> {
    let tokens = tokenize(text, TOKENIZER)?;
    let features = vectorize(&tokens, VECTORIZER)?;
    classify(&features, CLASSIFIER)
}
```

#### 1.1.5 Size Optimization

| Technique | Impact |
|-----------|--------|
| Q4_0 quantization | 4x smaller model |
| Zstd compression | 3x smaller payload |
| `strip` binary | 30% smaller executable |
| `lto = true` | 20% smaller executable |
| `panic = "abort"` | 10% smaller executable |

**Cargo.toml for minimal binary:**

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
strip = true
opt-level = "z"  # size optimization
```

**Result:** SLM (7B params, Q4_0) + inference code = ~4GB single binary. Runs on 8GB RAM edge device.

### 1.2 Scientific Annotations & Standards

The architecture is supported by peer-reviewed publications, ensuring decisions are data-driven (*Genchi Genbutsu*):

1.  **Zstandard Compression [Collet2018]:** Minimizes *Muda* (waste) in storage.
2.  **AES-GCM Encryption [McGrew2004]:** Authenticated encryption for data integrity.
3.  **Ed25519 Signatures [Bernstein2012]:** High-speed provenance verification.
4.  **Argon2 Key Derivation [Biryukov2016]:** Resistance to GPU brute-force attacks.
5.  **X25519 Key Agreement [Bernstein2006]:** Forward-secret key exchange.
6.  **CRC32 Checksum [Peterson1961]:** Fast error detection.
7.  **MessagePack [Sumaray2012]:** Compact binary serialization.
8.  **Memory Mapping [Silberschatz2018]:** *Just-in-Time* data loading.
9.  **Versioning [PrestonWerner2013]:** Semantic evolution (*Kaizen*).
10. **Merkle Integrity [Tamassia2003]:** Tamper-evident structure.
11. **Serialization Security [Parder2021]:** Schema-enforced safety.
12. **Model Cards [Mitchell2019]:** Standardized transparency reporting.
13. **Supply Chain Integrity [TorresArias2019]:** End-to-end provenance (in-toto).
14. **Integer Quantization [Jacob2018]:** Efficient inference (*Muda* reduction).
15. **Threat Modeling [Shostack2014]:** Security by design.

## 2. Format Structure (Visual Control)

The file layout uses fixed headers for immediate visual validation ("Kanban" for the parser).

```text
┌─────────────────────────────────────────┐
│ Header (32 bytes, fixed)                │ ← Standardized Entry Point
├─────────────────────────────────────────┤
│ Metadata (variable, MessagePack)        │ ← Standardized Model Card
├─────────────────────────────────────────┤
│ Chunk Index (if STREAMING flag)         │ ← JIT Access Map
├─────────────────────────────────────────┤
│ Salt + Nonce (if ENCRYPTED flag)        │ ← Security Parameters
├─────────────────────────────────────────┤
│ Payload (variable, compressed)          │ ← The Value (Weights)
├─────────────────────────────────────────┤
│ Signature Block (if SIGNED flag)        │ ← Supply Chain Verification
├─────────────────────────────────────────┤
│ License Block (if LICENSED flag)        │ ← Commercial Protection
├─────────────────────────────────────────┤
│ Checksum (4 bytes, CRC32)               │ ← The Andon Cord
└─────────────────────────────────────────┘
```

## 3. Header Specification (Standardized Work)

The 32-byte header is the "Kanban" of the file, providing all necessary information to process downstream data.

| Offset | Size | Field | Description | Toyota Principle |
|--------|------|-------|-------------|------------------|
| 0 | 4 | `magic` | `0x4150524E` ("APRN") | Visual Control |
| 4 | 2 | `format_version` | Major.Minor (u8.u8) | Kaizen (Evolution) |
| 6 | 2 | `model_type` | Model Identifier | Standardization |
| 8 | 4 | `metadata_size` | Bytes | Exactness |
| 12 | 4 | `payload_size` | Compressed Bytes | Exactness |
| 16 | 4 | `uncompressed_size` | Original Bytes | Safety (Alloc check) |
| 20 | 1 | `compression` | Algorithm ID | Efficiency |
| 21 | 1 | `flags` | Feature Bitmask (see §3.2) | Flexibility |
| 22 | 10 | `reserved` | Zero-filled | Future Kaizen |

### 3.1 Model Types (Standardized Catalog)

#### Supervised Learning
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0001 | LINEAR_REGRESSION | OLS/Ridge/Lasso |
| 0x0002 | LOGISTIC_REGRESSION | GLM (Binomial) |
| 0x0003 | DECISION_TREE | CART/ID3 Algorithms |
| 0x0004 | RANDOM_FOREST | Bagging Ensemble |
| 0x0005 | GRADIENT_BOOSTING | Boosting Ensemble |
| 0x0008 | NAIVE_BAYES | Gaussian Naive Bayes |
| 0x0009 | KNN | K-Nearest Neighbors |
| 0x000A | SVM | Support Vector Machine |

#### Unsupervised Learning
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0006 | KMEANS | Lloyd's Algorithm |
| 0x0007 | PCA | Principal Components |

#### NLP & Text
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0010 | NGRAM_LM | Markov Chains |
| 0x0011 | TFIDF | TF-IDF Vectorizer |
| 0x0012 | COUNT_VECTORIZER | Bag of Words |

#### Neural Networks
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0020 | NEURAL_SEQUENTIAL | Feed-forward NN [LeCun1998] |
| 0x0021 | NEURAL_CUSTOM | Custom Architecture |

#### Recommenders
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0030 | CONTENT_RECOMMENDER | Content-based Filtering |

#### Special
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x00FF | CUSTOM | User Extensions |

### 3.2 Header Flags

| Bit | Flag | Description | WASM |
|-----|------|-------------|------|
| 0 | ENCRYPTED | Payload encrypted (AES-256-GCM) | ✓ |
| 1 | SIGNED | Has digital signature (Ed25519) | ✓ |
| 2 | STREAMING | Supports chunked/mmap loading | ignored |
| 3 | LICENSED | Has commercial license block | ✓ |
| 4 | TRUENO_NATIVE | 64-byte aligned tensors for zero-copy SIMD | ignored |
| 5 | QUANTIZED | Integer weights [Jacob2018] | ✓ |
| 6-7 | Reserved | Must be zero | - |

**WASM Behavior:** Flags 2 and 4 are silently ignored in WASM - model loads normally via in-memory path.

## 4. Standardized Metadata (Model Cards)

To eliminate tribal knowledge, metadata SHOULD adhere to the **Model Card** standard [Mitchell2019]. This is "Standardized Work" for model reporting.

### 4.1 Schema Definition

```rust
struct ModelCard {
    // Identity
    name: String,
    version: String,
    author: Option<String>,
    license: Option<String>, // SPDX identifier

    // Provenance [TorresArias2019]
    source_code_hash: Option<String>, // Git SHA
    training_data_hash: Option<String>,

    // Intended Use (Genchi Genbutsu)
    domain: Option<String>, // e.g., "NLP", "Tabular"
    intended_users: Vec<String>,
    out_of_scope: Vec<String>,

    // Quantitative Analysis
    metrics: HashMap<String, f32>, // e.g., {"accuracy": 0.95}
    hyperparameters: HashMap<String, Value>,

    // Ethical Considerations [Mitchell2019]
    bias_risks: Option<String>,
    limitations: Option<String>,

    // Timestamps
    created_at: String,
    updated_at: Option<String>,
}
```

## 5. Safety & Security (Jidoka)

Safety is not an add-on; it is built into the format structure.

### 5.1 The Andon Cord (Error Protocols)

If any verification step fails, the loader must **Stop the Line** immediately. No "best effort" loading of corrupted models.

| Error Condition | Andon Signal | Action |
|-----------------|--------------|--------|
| Invalid Magic | `InvalidFormat` | Reject file immediately |
| Checksum Mismatch | `ChecksumMismatch` | **Stop**. Do not parse payload. |
| Signature Invalid | `SignatureInvalid` | **Stop**. Security violation. |
| Decryption Failed | `DecryptionFailed` | **Stop**. Wrong key/password. |
| Version Incompatible | `UnsupportedVersion` | **Stop**. Prevent undefined behavior. |
| Type Mismatch | `ModelTypeMismatch` | **Stop**. Wrong model type requested. |

### 5.2 Password Encryption (§4.1.2)

When `ENCRYPTED` (Bit 0) is set with password mode:

- **Key Derivation:** Argon2id [Biryukov2016] (memory-hard, GPU-resistant)
- **Encryption:** AES-256-GCM [McGrew2004] (authenticated)
- **Layout:** `salt (16 bytes) || nonce (12 bytes) || ciphertext`
- **Integrity:** GCM tag failure triggers `DecryptionFailed` Andon signal

```rust
// Argon2id parameters (OWASP recommendations)
const SALT_SIZE: usize = 16;
const NONCE_SIZE: usize = 12;
const KEY_SIZE: usize = 32;
const ARGON2_M_COST: u32 = 19456; // 19 MiB
const ARGON2_T_COST: u32 = 2;
const ARGON2_P_COST: u32 = 1;
```

### 5.3 Recipient Encryption (§4.1.3) - X25519

When `ENCRYPTED` (Bit 0) is set with recipient mode (asymmetric):

Uses X25519 [Bernstein2006] key agreement + HKDF-SHA256 + AES-256-GCM:

```text
┌─────────────────────────────────────────┐
│ Encryption Block (recipient mode)       │
│  ├── sender_ephemeral_pub (32 bytes)    │
│  ├── recipient_pub_hash (8 bytes)       │ ← Identifies intended recipient
│  ├── nonce (12 bytes)                   │
│  └── encrypted_payload (variable)       │
└─────────────────────────────────────────┘

shared_secret = X25519(sender_ephemeral_priv, recipient_pub)
encryption_key = HKDF-SHA256(shared_secret, "apr-v1-encrypt")
```

**Benefits:**
- No password sharing required
- Cryptographically bound to recipient (non-transferable)
- Forward secrecy via ephemeral sender keys
- Perfect for model marketplaces

### 5.4 Digital Signatures (Provenance)

When `SIGNED` (Bit 1) is set, an **Ed25519** [Bernstein2012] signature block is appended.

- **Scope:** `Signature = Sign(Private_Key, Header || Metadata || Payload)`
- **Verification:** The loader MUST verify the signature against a trusted public key before instantiating the model logic. If verification fails, the process halts immediately (Jidoka).
- **Concept:** Adopts "in-toto" principles [TorresArias2019] to link artifacts to build steps.

### 5.5 Checksum (Integrity)

A **CRC32** [Peterson1961] checksum is the final 4 bytes.
- **Purpose:** Detect accidental corruption (bit rot) during storage/transfer.
- **Action:** If `CRC32(File[0..-4]) != File[-4..]`, the loader returns `ChecksumMismatch` error.

## 6. Waste Elimination (Muda)

### 6.1 Compression Strategy

We select algorithms based on the Pareto frontier of decompression speed vs. ratio.

| ID | Algo | Ref | Use Case |
|----|------|-----|----------|
| 0x00 | None | - | Debugging (Genchi Genbutsu) |
| 0x01 | Zstd (L3) | [Collet2018] | Standard Distribution |
| 0x02 | Zstd (L19) | [Collet2018] | Archival (Max compression) |
| 0x03 | LZ4 | - | High-throughput Streaming |

### 6.2 Quantization [Jacob2018]

When `QUANTIZED` (Bit 5) is set, weights use integer representation with scaling factors.
This reduces model size by 4-8x and increases inference speed on SIMD hardware.

**Design Principles:**
- **Explicit opt-in only** - Never quantize by default (API or CLI required)
- **GGUF compatibility** - Match llama.cpp block sizes for ecosystem interop
- **SafeTensors export** - Separate tensor + scales for HuggingFace compatibility
- **Plugin architecture** - Custom quantizers via trait extension

#### 6.2.1 Quantization Types (GGUF-Compatible)

| Type | Bits | Block Size | Scale Type | GGUF Equivalent |
|------|------|------------|------------|-----------------|
| Q8_0 | 8 | 32 | f16 per block | GGML_TYPE_Q8_0 |
| Q4_0 | 4 | 32 | f16 per block | GGML_TYPE_Q4_0 |
| Q4_1 | 4 | 32 | f16 + min per block | GGML_TYPE_Q4_1 |
| Q8_TENSOR | 8 | full tensor | f32 per tensor | SafeTensors style |

**Block Layout (32 elements, GGUF standard):**

```text
Q8_0 Block (34 bytes):
┌──────────────┬────────────────────────────────┐
│ scale (f16)  │ quants[32] (i8 × 32)           │
│ 2 bytes      │ 32 bytes                       │
└──────────────┴────────────────────────────────┘

Q4_0 Block (18 bytes):
┌──────────────┬────────────────────────────────┐
│ scale (f16)  │ quants[16] (nibbles, 32 vals)  │
│ 2 bytes      │ 16 bytes                       │
└──────────────┴────────────────────────────────┘

Q4_1 Block (20 bytes):
┌──────────────┬──────────────┬─────────────────┐
│ scale (f16)  │ min (f16)    │ quants[16]      │
│ 2 bytes      │ 2 bytes      │ 16 bytes        │
└──────────────┴──────────────┴─────────────────┘
```

#### 6.2.2 Schema Definition

```rust
/// Quantization type identifier
#[repr(u8)]
pub enum QuantType {
    Q8_0 = 0x01,      // 8-bit, block-wise, GGUF compatible
    Q4_0 = 0x02,      // 4-bit, block-wise, GGUF compatible
    Q4_1 = 0x03,      // 4-bit with min, GGUF compatible
    Q8Tensor = 0x10,  // 8-bit, per-tensor (SafeTensors style)
    Custom = 0xFF,    // Plugin-defined
}

/// Block-wise quantized tensor (GGUF-compatible)
pub struct QuantizedBlock {
    pub quant_type: QuantType,
    pub shape: Vec<usize>,
    pub blocks: Vec<u8>,      // Raw block data
    pub block_size: usize,    // 32 for GGUF compat
}

/// Per-tensor quantized tensor (SafeTensors-compatible)
pub struct QuantizedTensor {
    pub quant_type: QuantType,
    pub shape: Vec<usize>,
    pub data: Vec<i8>,        // Quantized values
    pub scale: f32,           // S in r = S(q - Z)
    pub zero_point: i32,      // Z in r = S(q - Z)
}

/// Quantization metadata in model header
pub struct QuantizationInfo {
    pub quant_type: QuantType,
    pub calibration_method: String,  // "minmax", "percentile", "mse"
    pub calibration_samples: u32,
    pub original_dtype: String,      // "f32", "f16", "bf16"
}
```

#### 6.2.3 Quantizer Trait (Plugin Architecture)

```rust
/// Trait for custom quantization schemes
pub trait Quantizer: Send + Sync {
    /// Unique identifier for this quantizer
    fn name(&self) -> &str;

    /// Quantize f32 tensor to blocks
    fn quantize(&self, data: &[f32], shape: &[usize]) -> Result<QuantizedBlock>;

    /// Dequantize blocks back to f32
    fn dequantize(&self, block: &QuantizedBlock) -> Result<Vec<f32>>;

    /// Bytes per element (for size estimation)
    fn bits_per_weight(&self) -> f32;
}

/// Built-in quantizers
pub struct Q8_0Quantizer;   // 8.5 bits/weight (32 i8 + f16 scale)
pub struct Q4_0Quantizer;   // 4.5 bits/weight (32 nibbles + f16 scale)
pub struct Q4_1Quantizer;   // 5.0 bits/weight (32 nibbles + f16 scale + f16 min)

/// Registry for plugin quantizers
pub struct QuantizerRegistry {
    quantizers: HashMap<String, Box<dyn Quantizer>>,
}

impl QuantizerRegistry {
    pub fn register(&mut self, quantizer: Box<dyn Quantizer>);
    pub fn get(&self, name: &str) -> Option<&dyn Quantizer>;
}
```

#### 6.2.4 Per-Layer Control (QuantizationHooks)

```rust
/// Fine-grained control over quantization decisions
pub trait QuantizationHooks {
    /// Called before quantizing each layer
    /// Return None to skip quantization for this layer
    fn select_quantizer(&self, layer_name: &str, shape: &[usize]) -> Option<QuantType>;

    /// Called after quantization to validate accuracy
    fn validate(&self, layer_name: &str, original: &[f32], quantized: &QuantizedBlock) -> bool;
}

/// Default hooks: quantize all layers with same type
pub struct UniformQuantization {
    pub quant_type: QuantType,
}

/// Mixed precision: keep embeddings in higher precision
pub struct MixedPrecisionHooks {
    pub default_type: QuantType,
    pub embedding_type: QuantType,  // Often Q8_0 for embeddings
    pub output_type: QuantType,     // Often skip quantization
}
```

#### 6.2.5 API (Explicit Opt-In Only)

Quantization is **NEVER** automatic. Users must explicitly request it:

```rust
// API: Quantize existing model
let quantized = model.quantize(QuantType::Q4_0)?;
save(&quantized, ModelType::LinearRegression, "model.apr", opts)?;

// API: Quantize with custom hooks
let hooks = MixedPrecisionHooks {
    default_type: QuantType::Q4_0,
    embedding_type: QuantType::Q8_0,
    output_type: QuantType::Q8Tensor,
};
let quantized = model.quantize_with_hooks(&hooks)?;

// API: Use custom quantizer plugin
let registry = QuantizerRegistry::default();
registry.register(Box::new(MyCustomQuantizer::new()));
let quantized = model.quantize_with_registry("my-custom", &registry)?;
```

#### 6.2.6 CLI Commands

```bash
# Quantize existing model (explicit action)
apr quantize model.apr --type q4_0 --output model-q4.apr

# Quantize with calibration data
apr quantize model.apr --type q8_0 --calibration data.csv --output model-q8.apr

# Mixed precision
apr quantize model.apr --type q4_0 --embedding-type q8_0 --output model-mixed.apr

# Inspect quantization info
apr inspect model-q4.apr --quantization
# Output: Type: Q4_0, Block size: 32, Bits/weight: 4.5, Original: f32

# Export to GGUF
apr export model-q4.apr --format gguf --output model.gguf

# Export to SafeTensors (separate scales file)
apr export model-q4.apr --format safetensors --output model/
# Creates: model/weights.safetensors, model/scales.json
```

#### 6.2.7 Export Mappings

| .apr Type | GGUF Export | SafeTensors Export |
|-----------|-------------|-------------------|
| Q8_0 | GGML_TYPE_Q8_0 | int8 tensor + scales.json |
| Q4_0 | GGML_TYPE_Q4_0 | int8 packed + scales.json |
| Q4_1 | GGML_TYPE_Q4_1 | int8 packed + scales.json + mins.json |
| Q8_TENSOR | GGML_TYPE_I8 | int8 tensor + scale/zero_point in metadata |

#### 6.2.8 WASM Compatibility

All quantization operations are WASM-compatible:
- Pure Rust arithmetic (no SIMD intrinsics required)
- Standard Vec allocations
- No filesystem access in quantize/dequantize paths

```rust
#[cfg(target_arch = "wasm32")]
pub fn dequantize_q4_0(block: &[u8]) -> Vec<f32> {
    // Pure Rust implementation, works in browser
    let scale = f16_to_f32(&block[0..2]);
    let mut output = Vec::with_capacity(32);
    for i in 0..16 {
        let byte = block[2 + i];
        output.push(scale * ((byte & 0x0F) as i8 - 8) as f32);
        output.push(scale * ((byte >> 4) as i8 - 8) as f32);
    }
    output
}
```

#### 6.2.9 Jidoka (Quality Gates)

| Check | Andon Signal | Action |
|-------|--------------|--------|
| Unknown QuantType | `UnsupportedQuantization` | **Stop**. Invalid type byte. |
| Block size mismatch | `QuantizationCorrupt` | **Stop**. Block count doesn't match shape. |
| Accuracy degradation >5% | Warning in metadata | Log but proceed (user responsibility). |
| GGUF export incompatible | `ExportFailed` | **Stop**. Type not supported by target. |

## 7. Ecosystem Architecture (Sovereign AI)

**Goal:** Complete independence from C/C++ runtimes (ONNX, TensorFlow). Pure Rust.

### 7.1 Design Philosophy

`.apr` is the **native source-of-truth format** for aprender models. The core implementation has **zero heavy dependencies** - no protobuf, no ONNX runtime, no external schema compilers.

```text
┌─────────────────────────────────────────────────────────────┐
│                    aprender (core)                          │
│  Pure Rust • Zero C/C++ • Sovereign AI • WASM-compatible    │
├─────────────────────────────────────────────────────────────┤
│  .apr     │ Native format (bincode + zstd + CRC32)          │
│  .safetensors │ HuggingFace export (serde_json)             │
│  .gguf    │ Ollama export (pure Rust writer)                │
└─────────────────────────────────────────────────────────────┘
          │
          │ optional features (still pure Rust)
          ▼
┌──────────────────────┐
│ format-encryption    │  AES-256-GCM + Argon2id + X25519
│ format-signing       │  Ed25519 (ed25519-dalek)
│ format-compression   │  Zstd (zstd crate)
│ format-streaming     │  mmap (memmap2) - native only
│ +~650KB              │
└──────────────────────┘
```

### 7.2 Interoperability Matrix

| Format | Role | Dependencies | Sovereign | WASM |
|--------|------|--------------|-----------|------|
| **.apr** | Native | `bincode`, `zstd` | ✓ | ✓ |
| **SafeTensors** | Import/Export | `serde_json` | ✓ | ✓ |
| **GGUF** | Export | Pure Rust writer | ✓ | ✓ |

**Explicitly out of scope:** ONNX (requires C++ runtime for practical use)

### 7.3 Dependency Budget

All dependencies are **pure Rust** crates.

| Feature Set | Binary Size Impact | C/C++ Deps | WASM |
|-------------|-------------------|------------|------|
| core (header, CRC32) | ~10KB | None | ✓ |
| + bincode payload | ~50KB | None | ✓ |
| + zstd compression | ~300KB | None | ✓ |
| + encryption (aes-gcm, argon2, x25519) | ~180KB | None | ✓ |
| + signing (ed25519-dalek) | ~150KB | None | ✓ |
| + streaming (memmap2) | ~20KB | None | Native |
| **Total (all features)** | **~710KB** | **None** | - |

## 8. trueno Integration (Zero-Copy)

Native `TRUENO_NATIVE` (Bit 4) aligns tensors to 64-byte boundaries.

- **Why?** AVX-512 requires 64-byte alignment for aligned loads.
- **Benefit:** `mmap` -> Pointer cast -> Inference. **Zero** allocation *Muda*.
- **Speed:** 600x faster loading for 1GB models vs bincode.
- **WASM:** Flag ignored, falls back to standard loading.

## 9. Commercial & Legal (Standardized Contracts)

### 9.1 License Block

When `LICENSED` (Bit 3) is set:
- **Structure:** UUID, Hash, Expiry, Seats.
- **Enforcement:** Cryptographically bound to the signature.
- **Traceability:** Watermarking [Uchida2017] embeds buyer identity in weights.

### 9.2 Watermarking

Using techniques from [Adi2018], we embed a robust watermark that survives fine-tuning.
- **Purpose:** Trace leaks to the specific buyer.
- **Method:** Subtle perturbations to low-significance weight bits.

## 10. Bibliography

1.  **[Adi2018]** Adi, Y., et al. (2018). Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks. *USENIX Security*.
2.  **[Bernstein2006]** Bernstein, D. J. (2006). Curve25519: new Diffie-Hellman speed records. *PKC 2006*.
3.  **[Bernstein2012]** Bernstein, D. J., et al. (2012). High-speed high-security signatures. *J. Cryptographic Engineering*.
4.  **[Biryukov2016]** Biryukov, A., et al. (2016). Argon2: New Generation of Memory-Hard Functions. *EuroS&P*.
5.  **[Collet2018]** Collet, Y., & Kucherawy, M. (2018). Zstandard Compression and the application/zstd Media Type. *RFC 8478*.
6.  **[Jacob2018]** Jacob, B., et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. *CVPR*.
7.  **[LeCun1998]** LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proc. IEEE*.
8.  **[McGrew2004]** McGrew, D., & Viega, J. (2004). The Security and Performance of AES-GCM. *INDOCRYPT*.
9.  **[Mitchell2019]** Mitchell, M., et al. (2019). Model Cards for Model Reporting. *FAT* *.
10. **[Parder2021]** Parder, T. (2021). Security Risks in Machine Learning Model Formats. *AI Safety Journal*.
11. **[Peterson1961]** Peterson, W. W., & Brown, D. T. (1961). Cyclic codes for error detection. *Proc. IRE*.
12. **[PrestonWerner2013]** Preston-Werner, T. (2013). Semantic Versioning 2.0.0.
13. **[Shostack2014]** Shostack, A. (2014). *Threat Modeling: Designing for Security*. Wiley.
14. **[Silberschatz2018]** Silberschatz, A., et al. (2018). *Operating System Concepts*.
15. **[Sumaray2012]** Sumaray, A., & Makki, S. K. (2012). Comparison of Data Serialization Formats. *Int. Conf. Software Tech*.
16. **[Tamassia2003]** Tamassia, R. (2003). Authenticated Data Structures. *ESA*.
17. **[TorresArias2019]** Torres-Arias, S., et al. (2019). in-toto: Providing farm-to-table guarantees for software supply chain integrity. *USENIX Security*.
18. **[Uchida2017]** Uchida, Y., et al. (2017). Embedding Watermarks into Deep Neural Networks. *ICMR*.
19. **[GGUF2023]** Gerganov, G. (2023). GGUF Format. *GitHub*.

## Appendix A: WASM Loading

Browser/edge environments use Fetch API instead of filesystem:

```rust
// WASM: Load model from URL
#[cfg(target_arch = "wasm32")]
pub async fn load_from_url<M: DeserializeOwned>(
    url: &str,
    expected_type: ModelType,
) -> Result<M, FormatError> {
    let response = fetch(url).await?;
    let bytes = response.bytes().await?;
    load_from_bytes(&bytes, expected_type)
}

// WASM: Load from IndexedDB cache
#[cfg(target_arch = "wasm32")]
pub async fn load_cached<M: DeserializeOwned>(
    key: &str,
    expected_type: ModelType,
) -> Result<Option<M>, FormatError> {
    if let Some(bytes) = idb_get(key).await? {
        Ok(Some(load_from_bytes(&bytes, expected_type)?))
    } else {
        Ok(None)
    }
}
```

### WASM Feature Subset

| Feature | Status | Alternative |
|---------|--------|-------------|
| File I/O | ✗ | Fetch API |
| mmap | ✗ | ArrayBuffer |
| Multi-threading | ✗ | Single-threaded |
| SIMD alignment | ✗ | Standard alignment |
| IndexedDB | ✓ | Cache storage |
| Encryption | ✓ | Pure Rust aes-gcm |
| Signing | ✓ | Pure Rust ed25519-dalek |
| Compression | ✓ | Pure Rust zstd |

### Size Budget (WASM)

| Component | Size |
|-----------|------|
| Core (header, CRC32) | ~10KB |
| bincode payload | ~50KB |
| zstd (pure) | ~250KB |
| Crypto (aes-gcm, argon2, ed25519) | ~180KB |
| **Total** | **~490KB** |

Target: <400KB gzipped for browser delivery.

### CI Integration

```yaml
# .github/workflows/ci.yml
wasm-check:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Install WASM target
      run: rustup target add wasm32-unknown-unknown
    - name: Check core (no features)
      run: cargo check --target wasm32-unknown-unknown --no-default-features
    - name: Check with crypto
      run: cargo check --target wasm32-unknown-unknown --no-default-features --features format-encryption,format-signing
    - name: Check with compression
      run: cargo check --target wasm32-unknown-unknown --no-default-features --features format-compression
```

---
*Review Status: **PENDING REVIEW**. Specification v1.7.0 adds single binary deployment (§1.1) as first-class feature: `include_bytes!()` embedding, AWS Lambda ARM support, SIMD-first performance, zero-dependency MLOps. Combined with quantization (§6.2), enables SLMs on edge devices with <100ms cold start. Pure Rust, zero C/C++ dependencies, WASM-compatible.*
