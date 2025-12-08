# Case Study: APR Model Cache

This example demonstrates the hierarchical caching system implementing Toyota Way Just-In-Time principles for model management.

## Overview

The caching module provides a multi-tier cache for model storage:
- **L1 (Hot)**: In-memory, lowest latency
- **L2 (Warm)**: Memory-mapped files
- **L3 (Cold)**: Persistent storage

## Toyota Way Principles

| Principle | Application |
|-----------|-------------|
| Right Amount | Cache only what's needed for current inference |
| Right Time | Prefetch before access, evict after use |
| Right Place | L1 = hot, L2 = warm, L3 = cold storage |

## Running the Example

```bash
cargo run --example apr_cache
```

## Eviction Policies

| Policy | Description | Best For |
|--------|-------------|----------|
| LRU | Least Recently Used | General workloads |
| LFU | Least Frequently Used | Repeated inference |
| ARC | Adaptive Replacement Cache | Mixed workloads |
| Clock | Clock algorithm (FIFO variant) | High throughput |
| Fixed | No eviction | Embedded systems |

```rust
let policies = [
    EvictionPolicy::LRU,
    EvictionPolicy::LFU,
    EvictionPolicy::ARC,
    EvictionPolicy::Clock,
    EvictionPolicy::Fixed,
];

for policy in &policies {
    println!("{:?}: {}", policy, policy.description());
    println!("  Supports eviction: {}", policy.supports_eviction());
    println!("  Recommended for: {}", policy.recommended_use_case());
}
```

## Memory Budget

Control cache memory with watermarks:

```rust
// Default watermarks (90% high, 70% low)
let budget = MemoryBudget::new(100);

// Check eviction decisions
println!("90 pages: needs_eviction={}", budget.needs_eviction(90));  // true
println!("70 pages: can_stop={}", budget.can_stop_eviction(70));     // true

// Custom watermarks
let custom = MemoryBudget::with_watermarks(1000, 0.95, 0.80);

// Reserved pages (won't be evicted)
budget.reserve_page(1);
budget.reserve_page(2);
println!("Page 1 can_evict: {}", budget.can_evict(1));  // false
```

## Access Statistics

Track cache performance:

```rust
let mut stats = AccessStats::new();

// Record cache accesses
for i in 0..80 {
    stats.record_hit(100 + (i % 50), i);
}
for i in 80..100 {
    stats.record_miss(i);
}

// Prefetch tracking
for _ in 0..30 {
    stats.record_prefetch_hit();
}

println!("Hit Rate: {:.1}%", stats.hit_rate() * 100.0);
println!("Avg Access Time: {:.1} ns", stats.avg_access_time_ns());
println!("Prefetch Effectiveness: {:.1}%", stats.prefetch_effectiveness() * 100.0);
```

## Cache Configuration

### Default Configuration

```rust
let default = CacheConfig::default();
println!("L1 Max: {} MB", default.l1_max_bytes / (1024 * 1024));
println!("L2 Max: {} MB", default.l2_max_bytes / (1024 * 1024));
println!("Eviction: {:?}", default.eviction_policy);
println!("Prefetch: {}", default.prefetch_enabled);
```

### Embedded Configuration

```rust
let embedded = CacheConfig::embedded(1024 * 1024);  // 1MB
// L2 disabled, no eviction (Fixed policy)
```

### Custom Configuration

```rust
let custom = CacheConfig::new()
    .with_l1_size(128 * 1024 * 1024)
    .with_l2_size(2 * 1024 * 1024 * 1024)
    .with_eviction_policy(EvictionPolicy::ARC)
    .with_ttl(Duration::from_secs(3600))
    .with_prefetch(true);
```

## Model Registry

Manage cached models:

```rust
let config = CacheConfig::new()
    .with_l1_size(10 * 1024)
    .with_eviction_policy(EvictionPolicy::LRU);

let mut registry = ModelRegistry::new(config);

// Insert models
for i in 0..5 {
    let data = vec![0u8; 2048];
    let entry = CacheEntry::new(
        [i as u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(data),
    );
    registry.insert_l1(format!("model_{}", i), entry);
}

// Access models
let _ = registry.get("model_0");
let _ = registry.get("model_2");

// Get statistics
let stats = registry.stats();
println!("L1 Entries: {}", stats.l1_entries);
println!("L1 Bytes: {} KB", stats.l1_bytes / 1024);
println!("Hit Rate: {:.1}%", stats.hit_rate() * 100.0);
```

## Cache Tiers

| Tier | Name | Typical Latency |
|------|------|-----------------|
| L1Hot | Hot Cache | ~1 microsecond |
| L2Warm | Warm Cache | ~100 microseconds |
| L3Cold | Cold Storage | ~10 milliseconds |

## Cache Data Variants

```rust
// In-memory (decompressed)
let decompressed = CacheData::Decompressed(vec![0u8; 1000]);

// In-memory (compressed)
let compressed = CacheData::Compressed(vec![0u8; 500]);

// Memory-mapped file
let mapped = CacheData::Mapped {
    path: "/tmp/model.cache".into(),
    offset: 0,
    length: 2000,
};

println!("Decompressed size: {}", decompressed.size());
println!("Compressed: {}", compressed.is_compressed());
println!("Mapped: {}", mapped.is_mapped());
```

## Source Code

- Example: `examples/apr_cache.rs`
- Module: `src/cache/mod.rs`
