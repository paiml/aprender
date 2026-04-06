# Memory Pinning (mlock)

Per Dean & Barroso (2013) "The Tail at Scale", page faults from swapped memory cause tail latency spikes.

## The Problem

Large models may be swapped to disk under memory pressure:
- Page fault latency: **10-100ms** (vs microseconds for RAM)
- Causes p99 latency spikes
- Violates Genchi Genbutsu (go and see the real problem)

## Solution: mlock for Hot Experts

```rust
use realizar::memory::{MlockConfig, PinnedRegion, ExpertTier};

let config = MlockConfig {
    enabled: true,
    max_locked_bytes: 1024 * 1024 * 1024,  // 1GB limit
};

// Pin hot expert memory
let data: Vec<u8> = load_model_weights();
let (region, result) = unsafe {
    PinnedRegion::new(data.as_ptr(), data.len(), &config)
};

if region.is_locked() {
    println!("Model pinned in RAM - no page faults possible");
}
```

## Expert Tiers

```rust
use realizar::memory::ExpertTier;

let tier = ExpertTier::from_access_count(
    access_count,
    hot_threshold: 100,   // >100 accesses = Hot
    warm_threshold: 10,   // >10 accesses = Warm
);

match tier {
    ExpertTier::Hot => {
        // Pin in memory with mlock
    }
    ExpertTier::Warm => {
        // Keep in memory, but allow swap
    }
    ExpertTier::Cold => {
        // Load on demand from disk
    }
}
```

## Graceful Degradation

mlock may fail due to:
- Insufficient privileges (`CAP_IPC_LOCK`)
- Resource limits (`RLIMIT_MEMLOCK`)
- Platform not supported (Windows)

```rust
match result {
    MlockResult::Locked => { /* success */ }
    MlockResult::InsufficientPrivileges => {
        // Fall back to unpinned memory
        eprintln!("Warning: mlock failed, latency spikes possible");
    }
    MlockResult::ResourceLimit => { /* reduce locked size */ }
    MlockResult::Unsupported => { /* platform doesn't support */ }
}
```
