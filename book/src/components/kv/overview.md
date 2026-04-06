# Key-Value Store Overview

Phase 6 of trueno-db introduces a high-performance key-value store module designed for PAIML stack integration, specifically targeting pforge state management.

## Design Goals

1. **SIMD-Optimized**: Leverage trueno's hash module for fast key hashing
2. **Async-First**: Full tokio compatibility for non-blocking operations
3. **Thread-Safe**: Lock-free concurrent access via DashMap
4. **Platform Agnostic**: Support WASM, GPU, and native backends
5. **Zero-Copy Friendly**: Designed for efficient memory usage

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    pforge                            │
│              (StateManager trait)                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  trueno-db                           │
│                   KvStore                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │MemoryKvStore│  │PersistKvStore│  │ Future...   │  │
│  │  (DashMap)  │  │  (Phase 6b)  │  │             │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                    trueno                            │
│           hash_key() / hash_keys_batch()            │
│        (FxHash with SIMD acceleration)              │
└─────────────────────────────────────────────────────┘
```

## KvStore Trait

The core abstraction for all KV store implementations:

```rust
pub trait KvStore: Send + Sync {
    /// Get a value by key
    fn get(&self, key: &str) -> impl Future<Output = Result<Option<Vec<u8>>>> + Send;

    /// Set a value for a key
    fn set(&self, key: &str, value: Vec<u8>) -> impl Future<Output = Result<()>> + Send;

    /// Delete a key
    fn delete(&self, key: &str) -> impl Future<Output = Result<()>> + Send;

    /// Check if a key exists
    fn exists(&self, key: &str) -> impl Future<Output = Result<bool>> + Send;

    /// Batch get (SIMD-optimized)
    fn batch_get(&self, keys: &[&str]) -> impl Future<Output = Result<Vec<Option<Vec<u8>>>>> + Send;

    /// Batch set (SIMD-optimized)
    fn batch_set(&self, pairs: Vec<(&str, Vec<u8>)>) -> impl Future<Output = Result<()>> + Send;
}
```

## MemoryKvStore

The primary in-memory implementation using DashMap for lock-free concurrent access:

```rust
use trueno_db::kv::{KvStore, MemoryKvStore};

#[tokio::main]
async fn main() -> trueno_db::Result<()> {
    let store = MemoryKvStore::new();

    // Basic CRUD
    store.set("key", b"value".to_vec()).await?;
    let value = store.get("key").await?;
    assert_eq!(value, Some(b"value".to_vec()));

    store.delete("key").await?;
    assert!(!store.exists("key").await?);

    Ok(())
}
```

## Batch Operations

Batch operations leverage SIMD for optimal performance:

```rust
use trueno_db::kv::{KvStore, MemoryKvStore};

let store = MemoryKvStore::new();

// Batch set
let pairs = vec![
    ("config:timeout", b"30000".to_vec()),
    ("config:retries", b"3".to_vec()),
    ("config:debug", b"false".to_vec()),
];
store.batch_set(pairs).await?;

// Batch get
let keys = ["config:timeout", "config:retries", "config:missing"];
let values = store.batch_get(&keys).await?;
// values[0] = Some(b"30000")
// values[1] = Some(b"3")
// values[2] = None
```

## Thread Safety

`MemoryKvStore` is fully thread-safe and can be shared across tokio tasks:

```rust
use std::sync::Arc;
use trueno_db::kv::{KvStore, MemoryKvStore};

let store = Arc::new(MemoryKvStore::new());
let mut handles = vec![];

// Spawn concurrent writers
for i in 0..100 {
    let store = Arc::clone(&store);
    handles.push(tokio::spawn(async move {
        let key = format!("key:{}", i);
        store.set(&key, vec![i as u8]).await.unwrap();
    }));
}

// Wait for all writes
for handle in handles {
    handle.await.unwrap();
}
```

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| `get` | O(1) average | DashMap lookup |
| `set` | O(1) average | DashMap insert |
| `delete` | O(1) average | DashMap remove |
| `exists` | O(1) average | DashMap contains |
| `batch_get` | O(n) | Sequential, SIMD hash |
| `batch_set` | O(n) | Sequential, SIMD hash |

## Example

Run the KV store example:

```bash
cargo run --example kv_store
```

Output:
```
=== Trueno-DB KV Store Demo ===

1. Basic CRUD Operations
   ---------------------
   SET user:1001 = Alice
   SET user:1002 = Bob
   GET user:1001 = Some("Alice")
   UPDATE user:1001 = Some("Alice Smith")
   EXISTS user:1001 = true
   EXISTS user:9999 = false
   DELETE user:1002
   EXISTS user:1002 = false

2. Batch Operations
   -----------------
   BATCH SET 3 config keys
   BATCH GET results:
     config:timeout = Some("30000")
     config:retries = Some("3")
     config:missing = None

3. SIMD Hash Functions (via trueno)
   --------------------------------
   hash_key("session:abc123") = 0x...
   Batch hashes for sharding:
     shard:0 -> partition 0
     ...
```

## Future Work (Phase 6b+)

- **Persistent Backend**: Disk-backed storage using memory-mapped files
- **TTL Support**: Time-to-live for cache use cases
- **Pub/Sub**: Real-time key change notifications
- **Compression**: LZ4/Snappy for large values
- **GPU Acceleration**: GPU-backed storage for massive datasets
