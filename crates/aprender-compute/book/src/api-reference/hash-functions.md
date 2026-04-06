# Hash Functions

Trueno provides SIMD-optimized hash functions designed for high-performance key-value store operations. The hash module uses the FxHash algorithm with automatic backend selection for optimal performance.

## Overview

The hash module is designed for:
- Fast key hashing in KV stores
- Consistent hashing for distributed systems
- Shard/partition key assignment
- Cache key generation

## API Reference

### `hash_key`

Hash a string key to a 64-bit value.

```rust
use trueno::hash_key;

let hash = hash_key("user:1001");
println!("Hash: 0x{:016x}", hash);
```

**Signature:**
```rust
pub fn hash_key(key: &str) -> u64
```

**Properties:**
- Deterministic: Same input always produces same output
- Fast: Optimized for short keys typical in KV stores
- Non-cryptographic: Not suitable for security purposes

### `hash_bytes`

Hash raw bytes to a 64-bit value.

```rust
use trueno::hash_bytes;

let data = b"binary data";
let hash = hash_bytes(data);
```

**Signature:**
```rust
pub fn hash_bytes(bytes: &[u8]) -> u64
```

### `hash_keys_batch`

Hash multiple keys using SIMD acceleration. Automatically selects the best backend for the current CPU.

```rust
use trueno::hash_keys_batch;

let keys = ["user:1", "user:2", "user:3", "user:4"];
let hashes = hash_keys_batch(&keys);

for (key, hash) in keys.iter().zip(hashes.iter()) {
    println!("{} -> 0x{:016x}", key, hash);
}
```

**Signature:**
```rust
pub fn hash_keys_batch(keys: &[&str]) -> Vec<u64>
```

**Performance:** Batch hashing is significantly faster than individual calls when processing multiple keys. The speedup depends on the SIMD backend:
- AVX-512: Up to 8x speedup
- AVX2: Up to 4x speedup
- SSE2: Up to 2x speedup
- Scalar: Baseline (no vectorization)

### `hash_keys_batch_with_backend`

Hash multiple keys with explicit backend selection.

```rust
use trueno::{hash_keys_batch_with_backend, Backend};

let keys = ["a", "b", "c", "d"];

// Force scalar backend (useful for testing)
let scalar_hashes = hash_keys_batch_with_backend(&keys, Backend::Scalar);

// Use automatic selection (recommended)
let auto_hashes = hash_keys_batch_with_backend(&keys, Backend::Auto);

// Results are identical regardless of backend
assert_eq!(scalar_hashes, auto_hashes);
```

**Signature:**
```rust
pub fn hash_keys_batch_with_backend(keys: &[&str], backend: Backend) -> Vec<u64>
```

## Use Cases

### Partition/Shard Assignment

```rust
use trueno::hash_keys_batch;

let keys = ["order:1001", "order:1002", "order:1003", "order:1004"];
let hashes = hash_keys_batch(&keys);

let num_partitions = 4;
for (key, hash) in keys.iter().zip(hashes.iter()) {
    let partition = hash % num_partitions;
    println!("{} -> partition {}", key, partition);
}
```

### Consistent Key Distribution

The FxHash algorithm provides good distribution for typical key patterns:

```rust
use trueno::hash_key;

// Sequential keys still distribute well
for i in 0..10 {
    let key = format!("item:{}", i);
    let hash = hash_key(&key);
    println!("{}: 0x{:016x}", key, hash);
}
```

### Integration with `trueno-db`

The hash functions are re-exported by `trueno-db` for use with its KV store:

```rust
use trueno_db::kv::{hash_key, hash_keys_batch, KvStore, MemoryKvStore};

// Hash-based key lookup
let store = MemoryKvStore::new();
let key = "session:abc123";
let hash = hash_key(key);
println!("Key '{}' has hash 0x{:016x}", key, hash);
```

## Algorithm Details

Trueno uses the **FxHash** algorithm, which is:
- Extremely fast for small inputs (typical KV keys)
- Non-cryptographic (not suitable for security)
- Deterministic across platforms
- Well-suited for hash tables and bloom filters

**Constants:**
```rust
const FX_HASH_K: u64 = 0x517cc1b727220a95;
```

The algorithm processes input in 8-byte chunks using multiply-rotate operations, with special handling for the tail bytes.

## Backend Selection

The `Backend` enum controls SIMD acceleration:

| Backend | Description |
|---------|-------------|
| `Auto` | Automatically select best available (recommended) |
| `Scalar` | Force scalar implementation |
| `Sse2` | Force SSE2 (x86_64) |
| `Avx2` | Force AVX2 (x86_64) |
| `Avx512` | Force AVX-512 (x86_64) |
| `Neon` | Force NEON (ARM64) |
| `WasmSimd128` | Force WASM SIMD128 |

Runtime detection ensures the correct backend is used even when `Auto` is specified.

## Performance Benchmarks

Typical performance on modern x86_64 hardware (10,000 keys):

| Method | Time | Throughput |
|--------|------|------------|
| Sequential `hash_key` | ~1.5ms | ~6.7M keys/s |
| Batch `hash_keys_batch` | ~0.4ms | ~25M keys/s |

The exact speedup depends on:
- Key length (shorter keys benefit more from batching)
- CPU SIMD capabilities
- Memory access patterns

## Example: Complete Demo

```rust
use trueno::{hash_key, hash_keys_batch, hash_keys_batch_with_backend, Backend};

fn main() {
    // Single key hashing
    let key = "hello";
    let hash = hash_key(key);
    println!("hash_key({:?}) = 0x{:016x}", key, hash);

    // Batch hashing
    let keys = ["user:1", "user:2", "user:3", "user:4"];
    let hashes = hash_keys_batch(&keys);
    for (k, h) in keys.iter().zip(hashes.iter()) {
        println!("{} -> 0x{:016x}", k, h);
    }

    // Backend comparison
    let scalar = hash_keys_batch_with_backend(&keys, Backend::Scalar);
    let auto = hash_keys_batch_with_backend(&keys, Backend::Auto);
    assert_eq!(scalar, auto, "All backends produce identical results");
}
```

Run the example:
```bash
cargo run --example hash_demo
```
