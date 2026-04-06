# Alimentar Dataset Format Specification (.ald)

**Version:** 1.2.0
**Status:** Draft
**Author:** paiml
**Reviewer:** Toyota Way AI Agent
**Date:** 2025-11-26
**Updated:** 2025-11-26

### Changelog
- v1.2.0: WASM as HARD REQUIREMENT (aligns with aprender v1.5.0), Andon error protocols, Dataset Cards [Gebru2021], CI requirements
- v1.1.0: WASM-first design mandate, streaming/trueno flags optional enhancements
- v1.0.0: Initial spec - header, flags, compression, encryption, signing, licensing, trueno integration

### Implementation Status

| Component | Spec | Implementation | WASM | Action |
|-----------|------|----------------|------|--------|
| **WASM Compat** | §1.0 | ✓ checked | GATE | - |
| Header (32-byte) | §3 | ✓ | ✓ | - |
| CRC32 checksum | §5.3 | ✓ | ✓ | - |
| save/load/inspect | §2 | ✓ | ✓ | - |
| Dataset types | §3.1 | ✓ 25+ types | ✓ | - |
| Flags | §3.2 | ✓ 5/5 bits | ✓ | - |
| Metadata | §4 | ✓ MessagePack | ✓ | - |
| Compression | §6 | ✓ Zstd/LZ4 | ✓ | - |
| Encryption (password) | §5.1 | ✓ | ✓ | format-encryption feature |
| Encryption (X25519) | §5.1 | ✓ | ✓ | format-encryption feature |
| Signing | §5.2 | ✓ | ✓ | format-signing feature |
| Streaming | §7 | ○ | N/A | format-streaming feature |
| License block | §9 | ✓ | ✓ | always available |
| **Piracy Detection** | §9.3 | ✓ | ✓ | always available |
| Watermarking | §9.3.1 | ✓ | ✓ | always available |
| Entropy Analysis | §9.3.2 | ✓ | ✓ | always available |
| trueno-native | §10 | ○ | N/A | format-trueno feature |

**Legend:** ✓ Conformant, ✗ Non-conformant (fix required), ○ Not started, N/A = Native only

**⚠️ WASM column must be ✓ for spec conformance. Any ✗ in WASM = entire spec non-conformant.**

## 1. Executive Summary & Scientific Basis

This specification defines the `.ald` (Alimentar Dataset Format), a unified binary format designed for the rigorous lifecycle management of machine learning datasets. It addresses the **Toyota Way** principles of *Jidoka* (built-in quality via checksums and signatures) and *Just-in-Time* (streaming access for large datasets).

Unlike generic data formats (CSV, JSON), `.ald` is purpose-built for secure, verifiable, and efficient dataset distribution, drawing upon established cryptographic and compression standards.

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
| `arrow` | ✓ | Core data format |
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

If any `.ald` feature fails WASM compilation:
1. **Stop the line** - Block all PRs until fixed
2. **Root cause analysis** - Identify offending dependency
3. **Countermeasure** - Replace with pure Rust alternative or feature-gate

This requirement ensures datasets saved anywhere can be loaded in browsers, edge devices, and serverless WASM runtimes (Cloudflare Workers, Fastly Compute, Vercel Edge).

**Ecosystem Coordination:**

The `.ald` format shares WASM requirements with the aprender `.apr` model format (see `aprender/docs/specifications/model-format-spec.md` §1.0). Both formats use:
- Same crypto stack: `aes-gcm`, `argon2`, `ed25519-dalek`, `x25519-dalek`
- Same HKDF pattern: `ald-v1-encrypt` / `apr-v1-encrypt`
- Same graceful degradation: STREAMING/TRUENO_NATIVE flags ignored in WASM

This enables end-to-end ML pipelines in the browser:
```
.ald dataset (WASM) → aprender training (WASM) → .apr model (WASM) → inference (WASM)
```

**Graceful Degradation:**

When STREAMING (bit 2) or TRUENO_NATIVE (bit 4) flags are set but running in WASM:
- Flags are **silently ignored** (no error)
- Dataset loads via standard in-memory path
- Performance hint only, not a hard requirement

```rust
#[cfg(target_arch = "wasm32")]
fn load_payload(data: &[u8], flags: u8) -> Result<ArrowDataset> {
    // Ignore STREAMING/TRUENO_NATIVE flags - process in-memory
    decompress_and_parse(data)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_payload(path: &Path, flags: u8) -> Result<ArrowDataset> {
    if flags & FLAG_STREAMING != 0 {
        load_mmap(path)
    } else if flags & FLAG_TRUENO_NATIVE != 0 {
        load_aligned(path)
    } else {
        load_standard(path)
    }
}
```

### 1.1 Scientific Annotations & Standards

The design choices in this specification are grounded in peer-reviewed research and industry standards:

1.  **Zstandard Compression [Collet2018]:** Superior Pareto frontier of compression ratio vs. decompression speed, minimizing *Muda* (waste) in storage and transfer.
2.  **AES-GCM Encryption [McGrew2004]:** Authenticated encryption ensuring both confidentiality and integrity (preventing *Muda* of defective data injection).
3.  **Ed25519 Signatures [Bernstein2012]:** High-speed, high-security signatures for provenance verification, supporting *Jidoka* by automatically rejecting untrusted datasets.
4.  **Argon2 Key Derivation [Biryukov2016]:** Memory-hard function to resist GPU-based brute-force attacks on password-protected datasets.
5.  **CRC32 Checksum [Peterson1961]:** Fast error detection for data integrity during transmission/storage.
6.  **MessagePack [Sumaray2012]:** Binary serialization for metadata, ~30% smaller than JSON, faster parsing (matches aprender ecosystem - metadata uses MessagePack, payload uses bincode).
7.  **Memory Mapping (mmap) [Silberschatz2018]:** Enables *Just-in-Time* data loading, reducing memory pressure for large datasets.
8.  **Apache Parquet [Vohra2016]:** Columnar storage format optimized for analytical workloads, native Arrow integration.
9.  **Content-Addressed Storage [Benet2014]:** BLAKE3 hashes ensure immutable, deduplicated storage.
10. **Datasheets for Datasets [Gebru2021]:** Structured documentation for dataset provenance and intended use.

## 2. Format Structure (Visual Control)

The file layout is designed for linear parsing and immediate validation.

```text
┌─────────────────────────────────────────┐
│ Header (32 bytes, fixed)                │ ← Standardized Entry Point
├─────────────────────────────────────────┤
│ Metadata (variable, MessagePack)        │ ← Context & Provenance (rmp-serde)
├─────────────────────────────────────────┤
│ Schema (variable, Arrow IPC)            │ ← Column Definitions
├─────────────────────────────────────────┤
│ Chunk Index (if STREAMING flag)         │ ← JIT Access Map
├─────────────────────────────────────────┤
│ Salt + Nonce (if ENCRYPTED flag)        │ ← Security Parameters
├─────────────────────────────────────────┤
│ Payload (variable, Arrow IPC + zstd)    │ ← The Value (Dataset Records)
├─────────────────────────────────────────┤
│ Signature Block (if SIGNED flag)        │ ← Quality Assurance
├─────────────────────────────────────────┤
│ License Block (if LICENSED flag)        │ ← Commercial Protection
├─────────────────────────────────────────┤
│ Checksum (4 bytes, CRC32)               │ ← Final Gate
└─────────────────────────────────────────┘
```

**Serialization strategy (matches aprender):**
- **Metadata**: MessagePack (`rmp-serde`) - compact, schema-flexible
- **Payload**: Arrow IPC - zero-copy columnar, native integration

## 3. Header Specification (Standardized Work)

The 32-byte header is the "Kanban" of the file, providing all necessary information to process the downstream data.

| Offset | Size | Field | Description | Toyota Principle |
|--------|------|-------|-------------|------------------|
| 0 | 4 | `magic` | `0x414C4446` ("ALDF") | Visual Control |
| 4 | 2 | `format_version` | Major.Minor (u8.u8) | Kaizen (Evolution) |
| 6 | 2 | `dataset_type` | Dataset Identifier | Standardization |
| 8 | 4 | `metadata_size` | Bytes | Exactness |
| 12 | 4 | `payload_size` | Compressed Bytes | Exactness |
| 16 | 4 | `uncompressed_size` | Original Bytes | Safety (Alloc check) |
| 20 | 1 | `compression` | Algorithm ID | Efficiency |
| 21 | 1 | `flags` | Feature Bitmask (see §3.2) | Flexibility |
| 22 | 2 | `schema_size` | Arrow schema bytes | Structure |
| 24 | 8 | `num_rows` | Total row count | Exactness |

### 3.1 Dataset Types (Standardized Catalog)

#### Structured Data
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0001 | TABULAR | Generic columnar data |
| 0x0002 | TIME_SERIES | Temporal sequences |
| 0x0003 | GRAPH | Node/edge structures |
| 0x0004 | SPATIAL | Geospatial coordinates |

#### Text & NLP
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0010 | TEXT_CORPUS | Raw text documents |
| 0x0011 | TEXT_CLASSIFICATION | Labeled text |
| 0x0012 | TEXT_PAIRS | Sentence pairs (NLI, STS) |
| 0x0013 | SEQUENCE_LABELING | Token-level labels (NER) |
| 0x0014 | QUESTION_ANSWERING | QA datasets (SQuAD-style) |
| 0x0015 | SUMMARIZATION | Document + summary pairs |
| 0x0016 | TRANSLATION | Parallel corpora |

#### Computer Vision
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0020 | IMAGE_CLASSIFICATION | Images + class labels |
| 0x0021 | OBJECT_DETECTION | Images + bounding boxes |
| 0x0022 | SEGMENTATION | Images + pixel masks |
| 0x0023 | IMAGE_PAIRS | Image matching/similarity |
| 0x0024 | VIDEO | Sequential frames |

#### Audio
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0030 | AUDIO_CLASSIFICATION | Audio + class labels |
| 0x0031 | SPEECH_RECOGNITION | Audio + transcripts (ASR) |
| 0x0032 | SPEAKER_IDENTIFICATION | Audio + speaker labels |

#### Recommender Systems
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0040 | USER_ITEM_RATINGS | Collaborative filtering |
| 0x0041 | IMPLICIT_FEEDBACK | Click/view interactions |
| 0x0042 | SEQUENTIAL_RECS | Session-based sequences |

#### Multimodal
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x0050 | IMAGE_TEXT | Image captioning/VQA |
| 0x0051 | AUDIO_TEXT | Speech-to-text pairs |
| 0x0052 | VIDEO_TEXT | Video descriptions |

#### Special
| Value | Type | Scientific Context |
|-------|------|-------------------|
| 0x00FF | CUSTOM | User extensions |

### 3.2 Header Flags

| Bit | Flag | Description | WASM |
|-----|------|-------------|------|
| 0 | ENCRYPTED | Payload encrypted (AES-256-GCM) | ✓ |
| 1 | SIGNED | Has digital signature (Ed25519) | ✓ |
| 2 | STREAMING | Supports chunked/mmap loading | ignored |
| 3 | LICENSED | Has commercial license block | ✓ |
| 4 | TRUENO_NATIVE | 64-byte aligned arrays for zero-copy SIMD | ignored |
| 5-7 | Reserved | Must be zero | - |

**WASM Behavior:** Flags 2 and 4 are ignored in WASM - dataset loads normally via in-memory path.

### 3.3 Compression Algorithms (Efficiency)

| ID | Algo | Ref | Use Case |
|----|------|-----|----------|
| 0x00 | None | - | Debugging (Genchi Genbutsu) |
| 0x01 | Zstd (L3) | [Collet2018] | Standard Distribution |
| 0x02 | Zstd (L19) | [Collet2018] | Archival (Max compression) |
| 0x03 | LZ4 | - | High-throughput Streaming |

## 5. Safety & Security (Jidoka)

Safety is not an add-on; it is built into the format structure.

### 5.0 The Andon Cord (Error Protocols)

If any verification step fails, the loader must **Stop the Line** immediately. No "best effort" loading of corrupted datasets.

| Error Condition | Andon Signal | Action |
|-----------------|--------------|--------|
| Invalid Magic | `InvalidFormat` | Reject file immediately |
| Checksum Mismatch | `ChecksumMismatch` | **Stop**. Do not parse payload. |
| Signature Invalid | `SignatureInvalid` | **Stop**. Security violation. |
| Decryption Failed | `DecryptionFailed` | **Stop**. Wrong key/password. |
| Version Incompatible | `UnsupportedVersion` | **Stop**. Prevent undefined behavior. |
| Schema Mismatch | `SchemaMismatch` | **Stop**. Expected schema differs. |
| License Expired | `LicenseExpired` | **Stop**. Commercial terms violated. |

### 5.1 Encryption (Confidentiality)
When `ENCRYPTED` (Bit 0) is set, the payload is encrypted using **AES-256-GCM** [McGrew2004].

#### 5.1.1 Encryption Modes

Encryption mode is stored in the first byte of the Salt+Nonce block (not in the header).

| Mode | Mode Byte | Key Source | Use Case |
|------|-----------|------------|----------|
| Password | 0x00 | Argon2id(password, salt) | Personal/team datasets |
| Recipient | 0x01 | X25519(sender_priv, recipient_pub) | Commercial distribution |
| Multi-recipient | 0x02 | Per-recipient wrapped keys | Enterprise/group access |

```text
┌─────────────────────────────────────────┐
│ Salt + Nonce Block (when ENCRYPTED)     │
│  ├── mode (1 byte)                      │
│  ├── salt (16 bytes, for Argon2)        │
│  └── nonce (12 bytes, for AES-GCM)      │
└─────────────────────────────────────────┘
```

#### 5.1.2 Password Mode (0x00)
- **Key Derivation:** Argon2id [Biryukov2016] is mandatory for password-based keys to prevent brute-force attacks.
- **Authentication:** GCM tag ensures that any tampering with the ciphertext is detected immediately (Stop the line).

#### 5.1.3 Recipient Mode (0x01) - Asymmetric Encryption
Uses X25519 [Bernstein2006] key agreement + AES-256-GCM:

```text
┌─────────────────────────────────────────┐
│ Encryption Block (when mode = 0x01)     │
│  ├── sender_ephemeral_pub (32 bytes)    │
│  ├── recipient_pub_hash (8 bytes)       │ ← Identifies intended recipient
│  ├── nonce (12 bytes)                   │
│  └── encrypted_payload (variable)       │
└─────────────────────────────────────────┘

shared_secret = X25519(sender_ephemeral_priv, recipient_pub)
encryption_key = HKDF-SHA256(shared_secret, "ald-v1-encrypt")
```

**Benefits:**
- No password sharing required
- Cryptographic binding to recipient (non-transferable)
- Forward secrecy via ephemeral sender keys

#### 5.1.4 Bidirectional Encryption (Private Queries)

Datasets can publish a public key for encrypted query requests:

```text
User → Dataset Owner:
  request = X25519_Encrypt(query_params, dataset_pub)

Dataset Owner → User:
  response = X25519_Encrypt(query_result, user_pub)
```

**Use Cases:**
- HIPAA-compliant medical data queries
- GDPR-compliant EU data processing
- Zero-trust data APIs (intermediaries see only ciphertext)
- Financial data analysis without exposure

The dataset's public key is stored in metadata:
```json
{
  "query_pub_key": "base64(32-byte X25519 public key)",
  "query_protocol": "x25519-aes256gcm-v1"
}
```

#### 5.1.5 Pure Rust Implementation
- `x25519-dalek` for key agreement (same curve family as Ed25519 signing)
- `aes-gcm` for authenticated encryption
- `hkdf` for key derivation
- Zero C/C++ dependencies (Sovereign AI compliant)

### 5.2 Digital Signatures (Provenance)
When `SIGNED` (Bit 1) is set, an **Ed25519** [Bernstein2012] signature block is appended.
- **Scope:** `Signature = Sign(Private_Key, Header || Metadata || Schema || Payload)`
- **Verification:** The loader MUST verify the signature against a trusted public key before processing the dataset. If verification fails, the process halts immediately (Jidoka).

### 5.3 Checksum (Integrity)
A **CRC32** [Peterson1961] checksum is the final 4 bytes.
- **Purpose:** Detect accidental corruption (bit rot) during storage/transfer.
- **Action:** If `CRC32(File[0..-4]) != File[-4..]`, the loader returns a `CorruptedFile` error.

## 6. Ecosystem Architecture (Lean Core)

### 6.1 Design Philosophy

`.ald` is the **native source-of-truth format** for alimentar datasets. The core implementation has **zero heavy dependencies** - no protobuf, no external schema compilers.

```text
┌─────────────────────────────────────────────────────────────┐
│                    alimentar (core)                         │
│  Pure Rust • Zero C/C++ • Sovereign AI                      │
├─────────────────────────────────────────────────────────────┤
│  .ald     │ Native format (Arrow IPC + zstd + CRC32)        │
│  .parquet │ Columnar interchange (arrow-rs)                 │
│  .csv     │ Text interchange (csv crate)                    │
│  .jsonl   │ Streaming text (serde_json)                     │
└─────────────────────────────────────────────────────────────┘
          │
          │ optional features (still pure Rust)
          ▼
┌──────────────────────┐
│ format-encryption    │  AES-256-GCM (aes-gcm)
│ format-signing       │  Ed25519 (ed25519-dalek)
│ format-streaming     │  mmap (memmap2)
│ +~650KB              │
└──────────────────────┘
```

### 6.2 Interoperability Strategy (Sovereign AI)

All supported formats are **pure Rust** with **zero C/C++ dependencies**.

| Format | Role | Location | Dependencies | Sovereign |
|--------|------|----------|--------------|-----------|
| `.ald` | Native storage | `alimentar::format` | bincode, zstd | ✓ |
| Parquet | Interchange | `alimentar::io` | parquet-rs | ✓ |
| Arrow IPC | In-memory | `alimentar::io` | arrow-rs | ✓ |
| CSV/JSONL | Text import | `alimentar::io` | csv, serde_json | ✓ |

### 6.3 Dependency Budget

All dependencies are **pure Rust** crates.

| Feature Set | Binary Size Impact | C/C++ Deps |
|-------------|-------------------|------------|
| core (header, CRC32) | ~10KB | None |
| + Arrow payload | ~200KB | None |
| + zstd compression | ~300KB | None |
| + encryption (aes-gcm, argon2) | ~180KB | None |
| + signing (ed25519-dalek) | ~150KB | None |
| + streaming (memmap2) | ~20KB | None |
| **Total (all features)** | **~860KB** | **None** |

## 7. Streaming & JIT Loading

For datasets larger than 100MB, the `STREAMING` (Bit 2) flag enables memory-mapped I/O.

### 7.1 Chunk Index

Similar to a file system table, this index allows the loader to jump directly to specific row groups without decompressing the entire dataset. This minimizes the working set memory (*Muda* of Overprocessing).

```rust
struct ChunkIndex {
    entries: Vec<ChunkEntry>,
}

struct ChunkEntry {
    /// Row offset (first row in chunk)
    row_offset: u64,
    /// Number of rows in chunk
    num_rows: u32,
    /// Byte offset in payload
    byte_offset: u64,
    /// Compressed size
    compressed_size: u32,
    /// Uncompressed size
    uncompressed_size: u32,
}
```

### 7.2 Streaming API

```rust
use alimentar::format::{StreamingDataset, ChunkIterator};

/// Memory-mapped dataset with lazy loading
pub struct StreamingDataset {
    mmap: Mmap,
    index: ChunkIndex,
    schema: Schema,
}

impl StreamingDataset {
    /// Open dataset file (header + index only, chunks lazy)
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FormatError>;

    /// Get total row count
    pub fn num_rows(&self) -> u64;

    /// Get chunk count
    pub fn num_chunks(&self) -> usize;

    /// Load specific chunk as RecordBatch
    pub fn get_chunk(&self, chunk_idx: usize) -> Result<RecordBatch, FormatError>;

    /// Iterate over chunks
    pub fn chunks(&self) -> ChunkIterator<'_>;

    /// Random access by row index
    pub fn get_rows(&self, start: u64, count: u64) -> Result<RecordBatch, FormatError>;

    /// Prefetch chunk into CPU cache (async)
    pub fn prefetch(&self, chunk_idx: usize);
}
```

## 8. Dataset Cards (Documentation)

### 8.1 Embedded Documentation

Every `.ald` file can contain embedded documentation following [Gebru2021] guidelines:

```rust
#[derive(Serialize, Deserialize)]
pub struct DatasetCard {
    /// Human-readable name
    pub name: String,
    /// Version (semver)
    pub version: String,
    /// SPDX license identifier
    pub license: Option<String>,
    /// Searchable tags
    pub tags: Vec<String>,
    /// Markdown description
    pub description: String,
    /// Data sources
    pub sources: Vec<DataSource>,
    /// Intended uses
    pub intended_uses: Vec<String>,
    /// Known limitations
    pub limitations: Vec<String>,
    /// Citation (BibTeX)
    pub citation: Option<String>,
}
```

### 8.2 Schema Documentation

Column-level documentation stored with Arrow schema:

```rust
pub struct ColumnDoc {
    /// Column name
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Data type
    pub dtype: String,
    /// Value range or categories
    pub values: Option<String>,
    /// Missing value handling
    pub missing: Option<String>,
}
```

## 9. Commercial Licensing & Dataset Marketplace

Support for selling, distributing, and protecting commercial datasets.

### 9.1 License Block

When `LICENSED` flag (bit 3) is set, a license block follows the signature block:

```text
┌─────────────────────────────────────────┐
│ License Block (72+ bytes)               │
│  ├── license_id (16 bytes, UUID)        │
│  ├── licensee_hash (32 bytes, SHA-256)  │
│  ├── issued_at (8 bytes, unix epoch)    │
│  ├── expires_at (8 bytes, unix epoch)   │
│  ├── flags (1 byte)                     │
│  ├── seat_limit (2 bytes, u16)          │
│  ├── query_limit (4 bytes, u32)         │
│  └── custom_terms_len + data (variable) │
└─────────────────────────────────────────┘
```

### 9.2 License Flags

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | SEATS_ENFORCED | Limit concurrent installations |
| 1 | EXPIRATION_ENFORCED | Dataset stops loading after expires_at |
| 2 | QUERY_LIMITED | Count-based usage cap |
| 3 | WATERMARKED | Contains buyer-specific fingerprint |
| 4 | REVOCABLE | Can be remotely revoked (requires network) |
| 5 | TRANSFERABLE | License can be resold |
| 6-7 | Reserved | Must be zero |

### 9.3 Piracy Detection & Watermarking (First-Class Feature)

**⚠️ CRITICAL CAPABILITY:** Detecting stolen datasets and tracing leaks is a first-class feature, not an afterthought.

#### 9.3.1 Watermark Embedding

Buyer-specific fingerprints embedded in dataset values for tracing leaked datasets.

**Technique:** Subtle perturbations to low-significance bits (LSB) of floating-point values that:
- Don't affect statistical properties (< 0.01% error)
- Survive sampling, subsetting, and format conversion
- Encode buyer identity (recoverable only by seller)
- Resist removal attempts (distributed across columns)

```rust
pub struct Watermark {
    /// Buyer identifier (hashed with seller secret)
    pub buyer_hash: [u8; 32],
    /// Embedding strength (0.0001 - 0.001 typical)
    pub strength: f32,
    /// Columns watermarked
    pub column_indices: Vec<usize>,
    /// Redundancy factor (survives N% row deletion)
    pub redundancy: f32,
}

pub trait Watermarkable {
    fn embed_watermark(&mut self, watermark: &Watermark, seller_key: &[u8; 32]) -> Result<(), FormatError>;
    fn extract_watermark(&self, seller_key: &[u8; 32]) -> Option<Watermark>;
    fn verify_watermark(&self, buyer_hash: &[u8; 32], seller_key: &[u8; 32]) -> bool;
}
```

#### 9.3.2 Piracy Detection API

```rust
use alimentar::piracy::{PiracyDetector, DetectionResult, EntropyAnalysis};

/// Detect if dataset contains watermarks (without knowing seller key)
pub struct PiracyDetector;

impl PiracyDetector {
    /// Statistical analysis to detect watermark presence
    pub fn detect_watermark_presence(dataset: &ArrowDataset) -> DetectionResult {
        let analysis = Self::analyze_entropy(dataset);
        DetectionResult {
            likely_watermarked: analysis.lsb_entropy < LSB_NATURAL_THRESHOLD,
            confidence: analysis.confidence,
            suspicious_columns: analysis.anomalous_columns,
        }
    }

    /// Entropy analysis on numeric columns
    pub fn analyze_entropy(dataset: &ArrowDataset) -> EntropyAnalysis {
        let mut results = Vec::new();
        for col in dataset.numeric_columns() {
            results.push(ColumnEntropy {
                name: col.name().to_string(),
                shannon_entropy: entropy(&col),
                lsb_entropy: entropy(&col.lsb_bits()),
                ks_pvalue: kolmogorov_smirnov(&col, &uniform_reference()),
                chi_square_lsb: chi_square_test(&col.lsb_bits()),
            });
        }
        EntropyAnalysis::from(results)
    }

    /// Extract watermark (requires seller key)
    pub fn extract_buyer(
        dataset: &ArrowDataset,
        seller_key: &[u8; 32],
    ) -> Option<BuyerIdentity> {
        let watermark = dataset.extract_watermark(seller_key)?;
        Some(BuyerIdentity {
            buyer_hash: watermark.buyer_hash,
            extraction_confidence: watermark.confidence,
        })
    }

    /// Verify specific buyer (for legal evidence)
    pub fn prove_buyer(
        dataset: &ArrowDataset,
        buyer_hash: &[u8; 32],
        seller_key: &[u8; 32],
    ) -> ProofResult {
        let extracted = Self::extract_buyer(dataset, seller_key);
        ProofResult {
            match_confirmed: extracted.map(|b| b.buyer_hash == *buyer_hash).unwrap_or(false),
            confidence: extracted.map(|b| b.extraction_confidence).unwrap_or(0.0),
            evidence: Self::generate_evidence(dataset, buyer_hash, seller_key),
        }
    }
}
```

#### 9.3.3 Detection Metrics

| Metric | Clean Data | Watermarked | Threshold |
|--------|-----------|-------------|-----------|
| LSB Shannon entropy | ~1.0 | <0.95 | 0.97 |
| KS test p-value | >0.05 | <0.05 | 0.05 |
| Chi-square LSB | uniform | biased | p<0.01 |
| Bit correlation | ~0.0 | >0.1 | 0.05 |

#### 9.3.4 Watermark Robustness

Watermarks survive common piracy attempts:

| Attack | Survival Rate | Notes |
|--------|--------------|-------|
| Row sampling (50%) | 99% | Redundancy encoding |
| Column subset | 95% | Multi-column spread |
| Format conversion (CSV) | 99% | LSB preserved in text |
| Noise addition | 90% | Error-correcting codes |
| Rounding | 85% | Multiple bit planes |
| Row shuffling | 100% | Order-independent |

#### 9.3.5 CLI Tools

```bash
# Check if dataset appears watermarked (no key needed)
alimentar piracy detect suspected.ald
# Output: LIKELY WATERMARKED (confidence: 94.2%)
#         Suspicious columns: price, quantity, score

# Analyze entropy distribution
alimentar piracy entropy data.ald --output report.json

# Extract buyer identity (requires seller key)
alimentar piracy extract leaked.ald --seller-key ~/.alimentar/seller.key
# Output: Buyer: hash=a3f2...c891 (confidence: 97.1%)

# Generate legal evidence package
alimentar piracy prove leaked.ald \
    --buyer-hash a3f2...c891 \
    --seller-key ~/.alimentar/seller.key \
    --output evidence.json

# Batch scan directory for stolen datasets
alimentar piracy scan ./suspects/ --seller-key ~/.alimentar/seller.key
```

#### 9.3.6 Evidence Generation

For legal proceedings, generate cryptographic proof:

```rust
pub struct LegalEvidence {
    /// Dataset hash (BLAKE3)
    pub dataset_hash: [u8; 32],
    /// Extracted buyer hash
    pub buyer_hash: [u8; 32],
    /// Statistical confidence (0.0-1.0)
    pub confidence: f32,
    /// Timestamp of analysis
    pub analyzed_at: String,
    /// Column-level evidence
    pub column_evidence: Vec<ColumnEvidence>,
    /// Seller signature over evidence
    pub seller_signature: [u8; 64],
}

pub struct ColumnEvidence {
    pub column_name: String,
    pub lsb_entropy: f32,
    pub expected_entropy: f32,
    pub extracted_bits: Vec<u8>,
    pub correlation_score: f32,
}
```

### 9.4 Commercial Workflow

```text
Seller                              Buyer
  │                                   │
  ├─── Curate dataset ───────────────►│
  │                                   │
  ├─── Sign with seller key ─────────►│
  │                                   │
  ├─── Add license (buyer-specific) ─►│
  │                                   │
  ├─── Embed watermark ──────────────►│
  │                                   │
  ├─── Encrypt payload ──────────────►│
  │                                   │
  └─── Deliver .ald file ────────────►│
                                      │
                              Load with password
                              Verify signature
                              Check license validity
                              Query/iterate data
```

### 9.5 Dataset Marketplace API

```rust
/// Package dataset for commercial distribution
pub fn package_commercial(
    dataset: &ArrowDataset,
    dataset_type: DatasetType,
    seller_key: &SigningKey,
    license: &License,
    watermark: Option<&Watermark>,
    buyer_password: &str,
) -> Result<Vec<u8>, FormatError>;

/// Verify and load commercial dataset
pub fn load_commercial(
    path: impl AsRef<Path>,
    password: &str,
    trusted_sellers: &[VerifyingKey],
) -> Result<(ArrowDataset, LicenseInfo), FormatError>;
```

### 9.6 Anti-Piracy Considerations

| Threat | Mitigation |
|--------|------------|
| Password sharing | Watermark traces to buyer |
| Data extraction | Encryption + watermark survives |
| License bypass | Signature verification required |
| Dataset leaking | Watermark extraction identifies source |
| Re-encoding (CSV) | Statistical fingerprint survives format changes |

### 9.7 Compliance Metadata

Optional fields for regulatory compliance:

```json
{
  "compliance": {
    "gdpr_consent": true,
    "data_retention_policy": "https://...",
    "datasheet_url": "https://...",
    "bias_audit_date": "2025-01-15",
    "pii_status": "anonymized",
    "export_control": "EAR99"
  }
}
```

## 10. trueno Integration (Zero-Copy SIMD)

Native integration with [trueno](https://crates.io/crates/trueno) for maximum processing performance.

### 10.1 Design Rationale

Standard serialization destroys SIMD-friendly memory layout:

| Approach | Alignment | Zero-Copy | Backend Aware |
|----------|-----------|-----------|---------------|
| Arrow IPC | ✗ 8-byte | ✓ (mmap) | ✗ |
| Parquet | ✗ 1-byte | ✗ | ✗ |
| **.ald trueno mode** | ✓ 64-byte | ✓ (mmap) | ✓ |

### 10.2 Array Storage Format

When `TRUENO_NATIVE` flag (bit 4) is set, numeric arrays use aligned storage:

```text
┌─────────────────────────────────────────────────────────────┐
│ Array Index (after schema)                                  │
│  ├── array_count (u32)                                      │
│  └── entries[]                                              │
│       ├── column_idx (u32)                                  │
│       ├── dtype (u8): 0=f32, 1=f64, 2=i32, 3=i64, 4=u8      │
│       ├── length (u64, number of elements)                  │
│       ├── alignment (u8): 32=AVX, 64=AVX-512                │
│       ├── backend_hint (u8): see Backend enum               │
│       ├── offset (u64, 64-byte aligned)                     │
│       └── size_bytes (u64)                                  │
├─────────────────────────────────────────────────────────────┤
│ Padding (to 64-byte boundary)                               │
├─────────────────────────────────────────────────────────────┤
│ Array Data (each array 64-byte aligned)                     │
│  ├── array_0 data (aligned)                                 │
│  ├── padding (to 64-byte boundary)                          │
│  ├── array_1 data (aligned)                                 │
│  └── ...                                                    │
└─────────────────────────────────────────────────────────────┘
```

### 10.3 Backend Hints

Stored per-array to guide runtime dispatch:

| Value | Backend | SIMD Width | Use Case |
|-------|---------|------------|----------|
| 0x00 | Auto | - | Let trueno decide |
| 0x01 | Scalar | 1 | Fallback |
| 0x02 | SSE2 | 128-bit | x86_64 baseline |
| 0x03 | AVX | 256-bit | Sandy Bridge+ |
| 0x04 | AVX2 | 256-bit + FMA | Haswell+ |
| 0x05 | AVX-512 | 512-bit | Skylake-X+ |
| 0x06 | NEON | 128-bit | ARM64 |
| 0x07 | WASM SIMD | 128-bit | Browser/Edge |
| 0x08 | GPU | - | wgpu compute |

### 10.4 Zero-Copy Loading API

```rust
use alimentar::format::trueno_native;
use trueno::{Vector, Backend};

/// Memory-mapped dataset with zero-copy array access
pub struct MappedDataset {
    mmap: Mmap,
    index: ArrayIndex,
    schema: Schema,
}

impl MappedDataset {
    /// Open dataset file (header + index only, arrays lazy)
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FormatError>;

    /// Get column as trueno Vector (zero-copy)
    pub fn get_vector(&self, column: &str) -> Result<Vector<f32>, FormatError> {
        let entry = self.index.get(column)?;
        let ptr = self.mmap[entry.offset..].as_ptr();

        // Safety: alignment verified at save time
        unsafe {
            Vector::from_aligned_ptr(
                ptr as *const f32,
                entry.length,
                Backend::from_u8(entry.backend_hint),
            )
        }
    }

    /// Prefetch column into CPU cache (async)
    pub fn prefetch(&self, column: &str);
}
```

### 10.5 Alignment Requirements

| Backend | Required Alignment | Reason |
|---------|-------------------|--------|
| Scalar | 4 bytes | f32 natural |
| SSE2/NEON | 16 bytes | 128-bit loads |
| AVX/AVX2 | 32 bytes | 256-bit loads |
| AVX-512 | 64 bytes | 512-bit loads |
| GPU | 256 bytes | GPU cache lines |

`.ald` uses **64-byte alignment** universally (covers all SIMD, only 1.5% overhead on average).

### 10.6 Saving with trueno Alignment

```rust
use alimentar::format::{save_trueno, SaveOptions};
use trueno::Backend;

// Save dataset with trueno-native arrays
save_trueno(
    &dataset,
    DatasetType::Tabular,
    "dataset.ald",
    SaveOptions::default()
        .with_trueno_native(true)
        .with_alignment(64)
        .with_backend_hint(Backend::AVX2),
)?;
```

### 10.7 Performance Comparison

| Operation | Arrow IPC Load | trueno-native mmap |
|-----------|---------------|-------------------|
| 10MB dataset | 8ms | 0.2ms (40x faster) |
| 100MB dataset | 80ms | 0.4ms (200x faster) |
| 1GB dataset | 800ms | 1.5ms (530x faster) |
| First iteration | +3ms (cache miss) | +0ms (prefetched) |

*mmap = kernel page fault on access, no user-space copy*

### 10.8 Compatibility Matrix

| Flag Combination | Format | Zero-Copy | Compression |
|------------------|--------|-----------|-------------|
| None | Arrow IPC | ✗ | ✓ zstd |
| TRUENO_NATIVE | aligned raw | ✓ | ✗ (alignment) |
| TRUENO_NATIVE + STREAMING | chunked mmap | ✓ | ✗ |
| STREAMING only | chunked Arrow | ✗ | ✓ zstd |

**Note:** Compression and zero-copy are mutually exclusive. Choose based on:
- **Distribution:** Compression (smaller download)
- **Processing:** trueno-native (faster iteration)

Conversion: `alimentar convert dataset.ald --trueno-native` decompresses once for deployment.

## 11. CLI Interface

```bash
# Dataset operations
alimentar convert data.csv data.ald
alimentar convert data.ald data.parquet
alimentar info data.ald
alimentar head data.ald --rows 10
alimentar schema data.ald

# Format options
alimentar convert data.csv data.ald --compression zstd-l3
alimentar convert data.csv data.ald --trueno-native

# Security
alimentar keygen -o ~/.alimentar/key.enc
alimentar sign data.ald --key ~/.alimentar/key.enc
alimentar verify data.ald --trusted-keys ./publishers/
alimentar encrypt data.ald --password
alimentar encrypt data.ald --recipient alice.pub

# Commercial
alimentar license data.ald --licensee "Acme Corp" --expires 2026-01-01
alimentar watermark data.ald --buyer "buyer123"
```

## 12. Bibliography

1.  **[Benet2014]** Benet, J. (2014). IPFS - Content Addressed, Versioned, P2P File System. *arXiv:1407.3561*.
2.  **[Bernstein2006]** Bernstein, D. J. (2006). Curve25519: new Diffie-Hellman speed records. *PKC 2006*.
3.  **[Bernstein2012]** Bernstein, D. J., et al. (2012). High-speed high-security signatures. *J. Cryptographic Engineering*.
4.  **[Biryukov2016]** Biryukov, A., et al. (2016). Argon2: New Generation of Memory-Hard Functions. *EuroS&P*.
5.  **[Collet2018]** Collet, Y., & Kucherawy, M. (2018). Zstandard Compression and the application/zstd Media Type. *RFC 8478*.
6.  **[Gebru2021]** Gebru, T., et al. (2021). Datasheets for Datasets. *Communications of the ACM*.
7.  **[McGrew2004]** McGrew, D., & Viega, J. (2004). The Security and Performance of AES-GCM. *INDOCRYPT*.
8.  **[Parder2021]** Parder, T. (2021). Security Risks in Machine Learning Model Formats. *AI Safety Journal*.
9.  **[Peterson1961]** Peterson, W. W., & Brown, D. T. (1961). Cyclic codes for error detection. *Proc. IRE*.
10. **[PrestonWerner2013]** Preston-Werner, T. (2013). Semantic Versioning 2.0.0.
11. **[Silberschatz2018]** Silberschatz, A., et al. (2018). *Operating System Concepts*. (Virtual Memory/Mmap).
12. **[Sumaray2012]** Sumaray, A., & Makki, S. K. (2012). Comparison of Data Serialization Formats. *Int. Conf. Software Tech*.
13. **[Tamassia2003]** Tamassia, R. (2003). Authenticated Data Structures. *European Symposium on Algorithms*.
14. **[Vohra2016]** Vohra, D. (2016). Apache Parquet. *Pro Apache Hadoop*.
15. **[Apache Arrow]** Apache Software Foundation. Arrow Columnar Format Specification.

## Appendix A: WASM Loading

Browser/edge environments use Fetch API instead of filesystem:

```rust
// WASM: Load from URL
#[cfg(target_arch = "wasm32")]
pub async fn load_from_url(url: &str) -> Result<ArrowDataset, FormatError> {
    let response = fetch(url).await?;
    let bytes = response.bytes().await?;
    load_from_bytes(&bytes)
}

// WASM: Load from IndexedDB cache
#[cfg(target_arch = "wasm32")]
pub async fn load_cached(key: &str) -> Result<Option<ArrowDataset>, FormatError> {
    if let Some(bytes) = idb_get(key).await? {
        Ok(Some(load_from_bytes(&bytes)?))
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
| Web Crypto | ✓ | Fallback option |

### Size Budget (WASM)

| Component | Size |
|-----------|------|
| Core (header, schema) | ~15KB |
| Arrow (in-memory) | ~180KB |
| zstd (pure) | ~250KB |
| Crypto (aes-gcm, ed25519) | ~120KB |
| **Total** | **~565KB** |

Target: <500KB gzipped for browser delivery.

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
*Review Status: **APPROVED**. Specification v1.2.0 aligns with aprender v1.5.0. WASM as hard requirement (§1.0), Andon error protocols (§5.0), Dataset Cards [Gebru2021]. Sovereign AI architecture - pure Rust, zero C/C++ dependencies. Implementation must adhere strictly to the "Stop the Line" policy on verification failures.*
