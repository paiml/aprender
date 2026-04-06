# Dataset Format (.ald)

The Alimentar Dataset Format (`.ald`) is a binary format designed for secure, verifiable, and efficient dataset distribution.

## WASM-First Design

All core functionality works in WebAssembly (browser/edge). Native-only features degrade gracefully.

| Capability | WASM | Native |
|------------|------|--------|
| Header/schema parsing | ✓ | ✓ |
| Decompression (zstd) | ✓ | ✓ |
| Encryption/decryption | ✓ | ✓ |
| Signature verification | ✓ | ✓ |
| License validation | ✓ | ✓ |
| STREAMING (mmap) | - | ✓ |
| TRUENO_NATIVE (SIMD) | - | ✓ |

## Overview

| Feature | Description |
|---------|-------------|
| Magic | `0x414C4446` ("ALDF") |
| Header | 32 bytes, fixed |
| Payload | Arrow IPC + zstd compression |
| Checksum | CRC32 (final 4 bytes) |

## Header Flags

| Bit | Flag | Description | WASM |
|-----|------|-------------|------|
| 0 | ENCRYPTED | AES-256-GCM payload encryption | ✓ |
| 1 | SIGNED | Ed25519 digital signature | ✓ |
| 2 | STREAMING | Supports chunked/mmap loading | ignored |
| 3 | LICENSED | Commercial license block | ✓ |
| 4 | TRUENO_NATIVE | 64-byte aligned for zero-copy SIMD | ignored |

## Basic Usage

```rust
use alimentar::format::{save, load, SaveOptions};

// Save dataset
let options = SaveOptions::default()
    .with_compression(Compression::Zstd(3));
save(&dataset, "data.ald", options)?;

// Load dataset
let dataset = load("data.ald")?;
```

## Compression Options

```rust
use alimentar::format::{Compression, SaveOptions};

// No compression (debugging)
SaveOptions::default().with_compression(Compression::None);

// Standard (default)
SaveOptions::default().with_compression(Compression::Zstd(3));

// Maximum compression (archival)
SaveOptions::default().with_compression(Compression::Zstd(19));

// High-throughput streaming
SaveOptions::default().with_compression(Compression::Lz4);
```

## Signing

```rust
use alimentar::format::{save_signed, load_verified, KeyPair};

// Generate keypair
let keypair = KeyPair::generate();
keypair.save("~/.alimentar/key.enc", "passphrase")?;

// Sign on save
save_signed(&dataset, "data.ald", &keypair)?;

// Verify on load
let trusted = vec![keypair.verifying_key()];
let dataset = load_verified("data.ald", &trusted)?;
```

## Encryption

```rust
use alimentar::format::{save_encrypted, load_encrypted};

// Password encryption
save_encrypted(&dataset, "data.ald", "password123")?;
let dataset = load_encrypted("data.ald", "password123")?;

// Recipient encryption (asymmetric)
let recipient_pub = load_public_key("recipient.pub")?;
save_encrypted_for(&dataset, "data.ald", &recipient_pub)?;
```

## Streaming (Large Datasets)

```rust
use alimentar::format::StreamingDataset;

// Memory-mapped access
let stream = StreamingDataset::open("large.ald")?;

println!("Total rows: {}", stream.num_rows());
println!("Chunks: {}", stream.num_chunks());

// Iterate chunks (lazy loading)
for chunk in stream.chunks() {
    let batch = chunk?;
    process(batch);
}

// Random access
let rows = stream.get_rows(1000, 100)?;
```

## trueno Integration

```rust
use alimentar::format::{save_trueno, MappedDataset};
use trueno::Backend;

// Save with SIMD alignment
save_trueno(
    &dataset,
    "data.ald",
    SaveOptions::default()
        .with_trueno_native(true)
        .with_backend_hint(Backend::AVX2),
)?;

// Zero-copy vector access
let mapped = MappedDataset::open("data.ald")?;
let features: Vector<f32> = mapped.get_vector("features")?;
```

## Dataset Types

| Type | Value | Use Case |
|------|-------|----------|
| TABULAR | 0x0001 | Generic columnar |
| IMAGE_CLASSIFICATION | 0x0020 | Images + labels |
| TEXT_CLASSIFICATION | 0x0011 | Text + labels |
| QUESTION_ANSWERING | 0x0014 | QA pairs |
| USER_ITEM_RATINGS | 0x0040 | Recommendations |

## CLI

```bash
# Convert formats
alimentar convert data.csv data.ald
alimentar convert data.ald data.parquet

# Inspect
alimentar info data.ald
alimentar schema data.ald

# Security
alimentar sign data.ald --key ~/.alimentar/key.enc
alimentar verify data.ald
alimentar encrypt data.ald --password
```

## Comparison with Other Formats

| Feature | .ald | Parquet | CSV |
|---------|------|---------|-----|
| Binary | Yes | Yes | No |
| Schema | Yes | Yes | No |
| Compression | Yes | Yes | No |
| Encryption | Yes | No | No |
| Signing | Yes | No | No |
| Streaming | Yes | Yes | Limited |
| SIMD alignment | Yes | No | No |
| Licensing | Yes | No | No |

## WASM Usage

```rust
use alimentar::format::{load_from_bytes, load_from_url};

// Load from Fetch API response
#[cfg(target_arch = "wasm32")]
let dataset = load_from_url("https://example.com/data.ald").await?;

// Load from ArrayBuffer (e.g., file input)
#[cfg(target_arch = "wasm32")]
let dataset = load_from_bytes(&array_buffer)?;

// Cache in IndexedDB
#[cfg(target_arch = "wasm32")]
dataset.cache("my-dataset-v1").await?;
```

### Graceful Degradation

STREAMING and TRUENO_NATIVE flags are silently ignored in WASM:

```rust
// Same .ald file works everywhere
let dataset = load("data.ald")?;  // Native: uses mmap if STREAMING
let dataset = load_from_bytes(&bytes)?;  // WASM: in-memory, ignores flags
```

### Size Budget

| Component | Size |
|-----------|------|
| Core | ~15KB |
| Arrow | ~180KB |
| zstd | ~250KB |
| Crypto | ~120KB |
| **Total** | **~565KB** |

Target: <500KB gzipped.

See the [full specification](https://github.com/paiml/alimentar/blob/main/docs/specifications/dataset-format-spec.md) for complete details.
