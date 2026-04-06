# pforge Integration

The trueno-db KV store module is designed specifically for integration with pforge's `StateManager` trait, enabling SIMD-accelerated state management for MCP servers.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  pforge MCP Server                   │
│  ┌─────────────────────────────────────────────┐    │
│  │              Tool Handlers                   │    │
│  │   read_state() / write_state() / list_*()   │    │
│  └──────────────────────┬──────────────────────┘    │
│                         │                            │
│  ┌──────────────────────▼──────────────────────┐    │
│  │            StateManager Trait                │    │
│  │  get() / set() / delete() / list_keys()     │    │
│  └──────────────────────┬──────────────────────┘    │
└─────────────────────────┼───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│                    trueno-db                         │
│  ┌──────────────────────────────────────────────┐   │
│  │               KvStore Trait                   │   │
│  │  get() / set() / delete() / exists()         │   │
│  │  batch_get() / batch_set()                   │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                            │
│  ┌──────────────────────▼───────────────────────┐   │
│  │            MemoryKvStore                      │   │
│  │         (DashMap + trueno hash)              │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## StateManager → KvStore Mapping

| pforge StateManager | trueno-db KvStore | Notes |
|---------------------|-------------------|-------|
| `get(key)` | `get(key)` | Direct mapping |
| `set(key, value)` | `set(key, value)` | Direct mapping |
| `delete(key)` | `delete(key)` | Direct mapping |
| `list_keys(prefix)` | Custom impl | Prefix scan |
| `exists(key)` | `exists(key)` | Direct mapping |

## Implementation

### pforge Side (future)

```rust
// crates/pforge-runtime/src/state.rs
use trueno_db::kv::{KvStore, MemoryKvStore};

pub struct TruenoStateManager {
    store: MemoryKvStore,
}

impl TruenoStateManager {
    pub fn new() -> Self {
        Self {
            store: MemoryKvStore::new(),
        }
    }
}

#[async_trait]
impl StateManager for TruenoStateManager {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.store.get(key).await.map_err(|e| Error::State(e.to_string()))
    }

    async fn set(&self, key: &str, value: Vec<u8>) -> Result<()> {
        self.store.set(key, value).await.map_err(|e| Error::State(e.to_string()))
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.store.delete(key).await.map_err(|e| Error::State(e.to_string()))
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        self.store.exists(key).await.map_err(|e| Error::State(e.to_string()))
    }
}
```

### Feature Flags

When trueno-db is published to crates.io, pforge will add it as an optional dependency:

```toml
# pforge/Cargo.toml
[dependencies]
trueno-db = { version = "0.3", optional = true }

[features]
default = ["memory-state"]
memory-state = []
trueno-state = ["trueno-db"]
```

## Benefits

### 1. SIMD-Accelerated Hashing

Batch state operations benefit from trueno's SIMD hash functions:

```rust
// Batch read multiple keys efficiently
let keys = ["session:1", "session:2", "session:3"];
let values = store.batch_get(&keys).await?;

// Hash computation uses AVX2/AVX-512 when available
```

### 2. Lock-Free Concurrency

DashMap provides excellent concurrent read/write performance:

```rust
// Multiple MCP tool handlers can access state concurrently
// without lock contention
```

### 3. Platform Agnostic

Same API works across:
- Native (x86_64, ARM64)
- WebAssembly (browser, Node.js)
- GPU environments (future)

### 4. Stack Consistency

Using trueno-db for state management aligns with the PAIML stack:
- trueno: SIMD compute primitives
- trueno-db: Database with KV store
- pforge: MCP server framework
- pmat: Quality gates

## Migration Path

### From MemoryStateManager

```rust
// Before (pforge built-in)
let state = MemoryStateManager::new();

// After (trueno-db)
let state = TruenoStateManager::new();
```

The API is intentionally compatible for easy migration.

### From sled (deprecated)

If using the deprecated sled integration:

```rust
// Before (sled - deprecated, unmaintained)
let state = SledStateManager::open("./data")?;

// After (trueno-db)
let state = TruenoStateManager::new();  // Memory only for now
// Persistence coming in Phase 6b
```

## Example: Full MCP Server

```yaml
# forge.yaml
forge:
  name: my-server
  version: 0.1.0
  transport: stdio

state:
  backend: trueno  # Uses trueno-db KvStore

tools:
  - name: save_note
    handler:
      type: native
      function: save_note
    params:
      - name: key
        type: string
      - name: content
        type: string

  - name: get_note
    handler:
      type: native
      function: get_note
    params:
      - name: key
        type: string
```

## Performance

State operations inherit trueno-db's performance characteristics:

| Operation | Latency | Notes |
|-----------|---------|-------|
| `get` | ~100ns | DashMap lookup |
| `set` | ~150ns | DashMap insert |
| `batch_get` (100 keys) | ~5μs | SIMD hash |
| Concurrent get (8 threads) | ~200ns | Lock-free |

## See Also

- [KV Store Overview](./overview.md) - Full KV store documentation
- [Hash Functions](./hash-functions.md) - SIMD hash API
- [pforge State Management](https://pforge.dev/docs/state) - pforge docs
