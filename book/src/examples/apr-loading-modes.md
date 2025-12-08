# Case Study: APR Loading Modes

This example demonstrates the loading subsystem for `.apr` model files with different deployment targets following Toyota Way principles.

## Overview

The loading module provides flexible model loading strategies optimized for different deployment scenarios:
- **Embedded systems** with strict memory constraints
- **Server deployments** with maximum throughput
- **WASM** for browser-based inference

## Toyota Way Principles

| Principle | Application |
|-----------|-------------|
| Heijunka | Level resource demands during model initialization |
| Jidoka | Quality built-in with verification at each layer |
| Poka-yoke | Error-proofing via type-safe APIs |

## Loading Modes

### Eager Loading
Load entire model into memory upfront. Best for latency-critical inference.

### MappedDemand
Memory-map model and load sections on demand. Best for large models with partial access patterns.

### Streaming
Process model in chunks without loading entirely. Best for memory-constrained environments.

### LazySection
Load only metadata initially, defer weight loading. Best for model inspection/browsing.

## Verification Levels

| Level | Checksum | Signature | Use Case |
|-------|----------|-----------|----------|
| UnsafeSkip | No | No | Development only |
| ChecksumOnly | Yes | No | General use |
| Standard | Yes | Yes | Production |
| Paranoid | Yes | Yes + ASIL-D | Safety-critical |

## Running the Example

```bash
cargo run --example apr_loading_modes
```

## Key Code Patterns

### Deployment-Specific Configuration

```rust
// Embedded (automotive ECU)
let embedded = LoadConfig::embedded(1024 * 1024);  // 1MB budget

// Server (high throughput)
let server = LoadConfig::server();

// WASM (browser)
let wasm = LoadConfig::wasm();
```

### Custom Configuration

```rust
let custom = LoadConfig::new()
    .with_mode(LoadingMode::Streaming)
    .with_max_memory(512 * 1024)
    .with_verification(VerificationLevel::Paranoid)
    .with_backend(Backend::CpuSimd)
    .with_time_budget(Duration::from_millis(50))
    .with_streaming(128 * 1024);
```

### Buffer Pools for Deterministic Allocation

```rust
let pool = BufferPool::new(4, 64 * 1024);  // 4 buffers, 64KB each
let config = LoadConfig::new()
    .with_buffer_pool(Arc::new(pool))
    .with_mode(LoadingMode::Streaming);
```

## WCET (Worst-Case Execution Time)

The module provides WCET estimates for safety-critical systems:

| Platform | Read Speed | Decompress | Ed25519 Verify |
|----------|------------|------------|----------------|
| Automotive S32G | High | High | Fast |
| Aerospace RAD750 | Moderate | Moderate | Slow |
| Edge (RPi 4) | Variable | Moderate | Fast |

## Source Code

- Example: `examples/apr_loading_modes.rs`
- Module: `src/loading/mod.rs`
