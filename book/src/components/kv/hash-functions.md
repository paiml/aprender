# Hash Functions

The KV store module integrates trueno's SIMD-optimized hash functions for fast key hashing, partition assignment, and consistent hashing.

## Re-exports

The KV module re-exports trueno's hash functions for convenience:

```rust
use trueno_db::kv::{hash_key, hash_bytes, hash_keys_batch};
```

## Single Key Hashing

Hash a string key to a 64-bit value:

```rust
use trueno_db::kv::hash_key;

let key = "session:abc123";
let hash = hash_key(key);
println!("Hash: 0x{:016x}", hash);
```

## Batch Hashing

Hash multiple keys using SIMD acceleration:

```rust
use trueno_db::kv::hash_keys_batch;

let keys = ["user:1", "user:2", "user:3", "user:4"];
let hashes = hash_keys_batch(&keys);

for (key, hash) in keys.iter().zip(hashes.iter()) {
    println!("{} -> 0x{:016x}", key, hash);
}
```

## Partition Assignment

Use hashes for shard/partition routing:

```rust
use trueno_db::kv::hash_keys_batch;

let keys = ["order:1001", "order:1002", "order:1003", "order:1004"];
let hashes = hash_keys_batch(&keys);

let num_partitions = 4;
for (key, hash) in keys.iter().zip(hashes.iter()) {
    let partition = hash % num_partitions;
    println!("{} -> partition {}", key, partition);
}
```

## Algorithm

The hash module uses **FxHash**, a fast non-cryptographic hash algorithm:

- **Deterministic**: Same input always produces same output
- **Fast**: Optimized for short keys typical in KV stores
- **Well-distributed**: Good distribution for sequential keys
- **Non-cryptographic**: Not suitable for security purposes

## SIMD Acceleration

Batch operations automatically use the best available SIMD backend:

| Backend | Platform | Speedup |
|---------|----------|---------|
| AVX-512 | x86_64 | ~8x |
| AVX2 | x86_64 | ~4x |
| SSE2 | x86_64 | ~2x |
| NEON | ARM64 | ~4x |
| WASM SIMD128 | WebAssembly | ~2x |
| Scalar | All | 1x (baseline) |

## Performance

Typical throughput on modern x86_64 hardware:

| Method | Keys | Time | Throughput |
|--------|------|------|------------|
| `hash_key` (sequential) | 10,000 | ~1.5ms | 6.7M/s |
| `hash_keys_batch` (SIMD) | 10,000 | ~0.4ms | 25M/s |

## Integration with KvStore

The hash functions are used internally by `MemoryKvStore` for:

1. **Key indexing**: DashMap uses FxHash internally
2. **Partition routing**: For future distributed scenarios
3. **Consistent hashing**: For cache-friendly key distribution

```rust
use trueno_db::kv::{hash_key, KvStore, MemoryKvStore};

let store = MemoryKvStore::new();
let key = "my-key";

// Hash is computed internally during set/get
store.set(key, b"value".to_vec()).await?;

// But you can also compute it explicitly for routing
let hash = hash_key(key);
let shard = hash % 4;
println!("Key '{}' routes to shard {}", key, shard);
```

## See Also

- [trueno Hash Functions](../../../api-reference/hash-functions.md) - Full API reference
- [KV Store Overview](./overview.md) - KV store architecture
- [pforge Integration](./pforge-integration.md) - State management integration
