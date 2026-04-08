<!-- PCU: examples-model-bundling-paging | contract: contracts/apr-page-examples-model-bundling-paging-v1.yaml -->
<!-- Example: cargo run -p aprender-core --example none -->
<!-- Status: enforced -->

# Case Study: Model Bundling and Memory Paging

Deploy large ML models on resource-constrained devices using aprender's bundle module with LRU-based memory paging.

## Quick Start

```rust
use aprender::bundle::{ModelBundle, BundleBuilder, PagedBundle, PagingConfig};

// Create a bundle with multiple models
let bundle = BundleBuilder::new("models.apbundle")
    .add_model("encoder", encoder_weights)
    .add_model("decoder", decoder_weights)
    .add_model("classifier", classifier_weights)
    .build()?;

// Load with memory paging (10MB limit)
let mut paged = PagedBundle::open("models.apbundle",
    PagingConfig::new().with_max_memory(10_000_000))?;

// Access models on-demand - only loads what's needed
let weights = paged.get_model("encoder")?;
```

## Motivation

Modern ML models can exceed available RAM, especially on:
- Edge devices (IoT, embedded systems)
- Mobile applications
- Multi-model deployments
- Development machines running multiple services

The bundle module solves this with:
- **Model Bundling**: Package multiple models atomically
- **Memory Paging**: LRU-based on-demand loading
- **Pre-fetching**: Proactive loading based on access patterns

## The .apbundle Format

```
┌─────────────────────────────────────────────────┐
│ Magic: "APBUNDLE" (8 bytes)                      │
├─────────────────────────────────────────────────┤
│ Version: 1 (4 bytes)                             │
├─────────────────────────────────────────────────┤
│ Manifest Length (4 bytes)                        │
├─────────────────────────────────────────────────┤
│ Manifest (JSON)                                  │
│   - model_count                                  │
│   - models: [{name, offset, size, checksum}]     │
├─────────────────────────────────────────────────┤
│ Model Data                                       │
│   - encoder weights (aligned)                    │
│   - decoder weights (aligned)                    │
│   - classifier weights (aligned)                 │
└─────────────────────────────────────────────────┘
```

## Memory Paging Strategies

### LRU (Least Recently Used)

```rust
let config = PagingConfig::new()
    .with_max_memory(10_000_000)  // 10MB limit
    .with_eviction(EvictionStrategy::LRU);
```

Evicts models not accessed recently. Best for sequential workloads.

### LFU (Least Frequently Used)

```rust
let config = PagingConfig::new()
    .with_max_memory(10_000_000)
    .with_eviction(EvictionStrategy::LFU);
```

Evicts models with fewest accesses. Best for workloads with hot/cold patterns.

## Pre-fetching

Enable proactive loading based on access patterns:

```rust
let config = PagingConfig::new()
    .with_prefetch(true)
    .with_prefetch_count(2);  // Pre-fetch next 2 likely models

let mut bundle = PagedBundle::open("models.apbundle", config)?;

// Manual hint
bundle.prefetch_hint("classifier")?;
```

## Paging Statistics

Monitor cache performance:

```rust
let stats = bundle.stats();
println!("Hits: {}", stats.hits);
println!("Misses: {}", stats.misses);
println!("Evictions: {}", stats.evictions);
println!("Hit Rate: {:.1}%", stats.hit_rate() * 100.0);
println!("Memory Used: {} bytes", stats.memory_used);
```

## Shell Completion Example

aprender-shell uses paging for large histories:

```bash
# Train with 10MB memory limit
aprender-shell train --memory-limit 10

# Suggestions load n-gram segments on-demand
aprender-shell suggest "git " --memory-limit 10

# View paging statistics
aprender-shell stats --memory-limit 10
```

Output:
```
📊 Paged Model Statistics:
   N-gram size:     3
   Total commands:  50000
   Vocabulary size: 15000
   Total segments:  25
   Loaded segments: 3
   Memory limit:    10.0 MB
   Loaded bytes:    2.5 KB

📈 Paging Statistics:
   Page hits:       47
   Page misses:     3
   Evictions:       0
   Hit rate:        94.0%
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      PagedBundle                              │
├──────────────────────────────────────────────────────────────┤
│  BundleReader     │  LRU Cache      │  PageTable              │
│  ─────────────    │  ──────────     │  ─────────              │
│  read_manifest()  │  HashMap<K,V>   │  track access           │
│  read_model()     │  LRU ordering   │  find LRU/LFU           │
│                   │  eviction       │  timestamps             │
├──────────────────────────────────────────────────────────────┤
│                    PagingConfig                               │
│  max_memory: 10MB  │  eviction: LRU  │  prefetch: true        │
└──────────────────────────────────────────────────────────────┘
```

## API Reference

### BundleBuilder

```rust
let bundle = BundleBuilder::new("path.apbundle")
    .add_model("name", data)
    .with_config(BundleConfig::new()
        .with_compression(false)
        .with_max_memory(10_000_000))
    .build()?;
```

### ModelBundle

```rust
// Create empty bundle
let mut bundle = ModelBundle::new();
bundle.add_model("model1", weights);
bundle.save("path.apbundle")?;

// Load bundle
let bundle = ModelBundle::load("path.apbundle")?;
let weights = bundle.get_model("model1");
```

### PagedBundle

```rust
// Open with paging
let mut bundle = PagedBundle::open("path.apbundle",
    PagingConfig::new().with_max_memory(10_000_000))?;

// Get model (loads on-demand)
let data = bundle.get_model("model1")?;

// Check cache state
assert!(bundle.is_cached("model1"));

// Manually evict
bundle.evict("model1");

// Clear all cached data
bundle.clear_cache();
```

### PagingConfig

```rust
let config = PagingConfig::new()
    .with_max_memory(10_000_000)   // 10MB limit
    .with_page_size(4096)          // 4KB pages
    .with_prefetch(true)           // Enable pre-fetching
    .with_prefetch_count(2)        // Pre-fetch 2 models
    .with_eviction(EvictionStrategy::LRU);
```

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Bundle creation | O(n) | n = total model bytes |
| Bundle load (metadata) | O(m) | m = manifest size |
| Model access (cached) | O(1) | Hash lookup |
| Model access (uncached) | O(k) | k = model size, disk I/O |
| Eviction | O(1) | LRU: deque pop; LFU: heap |
| Pre-fetch | O(k) | Background loading |

## Best Practices

1. **Size models appropriately**: Split large models into logical components
2. **Choose eviction wisely**: LRU for sequential, LFU for hot/cold
3. **Monitor hit rates**: Target >80% for good performance
4. **Use pre-fetching**: Reduce latency for predictable access patterns
5. **Test memory limits**: Profile actual usage before deployment

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Low hit rate | Increase memory limit or reduce model sizes |
| High eviction count | Models too large for memory limit |
| Slow first access | Use pre-fetch hints for critical models |
| OOM errors | Reduce max_memory, ensure eviction works |

## Implementation Details

The bundle module is implemented in pure Rust with:
- 42 tests covering all components
- Zero unsafe code
- No external dependencies beyond std
- Cross-platform (Unix mmap simulation via std I/O)

See `src/bundle/` for implementation:
- `mod.rs`: ModelBundle, BundleBuilder, BundleConfig
- `format.rs`: Binary format reader/writer
- `manifest.rs`: JSON manifest handling
- `mmap.rs`: Memory-mapped file abstraction
- `paging.rs`: PagedBundle, PagingConfig, eviction strategies
