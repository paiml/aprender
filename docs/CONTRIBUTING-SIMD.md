# Contributing to SIMD Collectors

This guide explains how to contribute to the SIMD-accelerated metric collectors in trueno-viz.

## Architecture Overview

The SIMD system consists of several layers:

```
src/monitor/simd/
├── mod.rs              # Backend detection, SimdStats, constants
├── kernels.rs          # Core SIMD operations (parse, delta, percentage)
├── ring_buffer.rs      # SimdRingBuffer with O(1) statistics
├── soa.rs              # Structure-of-Arrays layouts
├── compressed.rs       # Delta encoding for historical data
├── correlation.rs      # Pearson correlation calculations
└── integration_tests.rs

src/monitor/collectors/
├── cpu_simd.rs         # CPU metrics collector
├── memory_simd.rs      # Memory metrics collector
├── network_simd.rs     # Network metrics collector
├── disk_simd.rs        # Disk I/O metrics collector
├── process_simd.rs     # Process enumeration collector
├── gpu_simd.rs         # GPU metrics aggregator
└── battery_sensors_simd.rs  # Battery and sensor collector
```

## Adding a New SIMD Kernel

### Step 1: Define the Kernel in `kernels.rs`

```rust
/// SIMD-accelerated operation description.
///
/// # Performance Target
/// - Throughput: ≥Nx vs scalar
/// - Latency: < Yμs for Z elements
///
/// # Example
/// ```
/// let result = simd_my_operation(&input);
/// ```
#[must_use]
pub fn simd_my_operation(input: &[u64]) -> Vec<u64> {
    // Implementation using trueno or std::simd
}
```

### Step 2: Add Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_my_operation_basic() {
        let input = vec![1, 2, 3, 4];
        let result = simd_my_operation(&input);
        assert_eq!(result, vec![/* expected */]);
    }

    #[test]
    fn test_simd_my_operation_empty() {
        let result = simd_my_operation(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_simd_my_operation_large() {
        let input: Vec<u64> = (0..10000).collect();
        let start = std::time::Instant::now();
        let _ = simd_my_operation(&input);
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 10);
    }
}
```

### Step 3: Add Integration Test

Add a test to `integration_tests.rs`:

```rust
#[test]
fn test_my_operation_integration() {
    // Test that the operation works with real collectors
}
```

## Adding a New Collector

### Step 1: Create the SoA Layout

In `soa.rs`:

```rust
/// SoA layout for MyMetrics.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct MyMetricsSoA {
    /// Field 1 values across all items.
    pub field1: Vec<u64>,
    /// Field 2 values across all items.
    pub field2: Vec<f64>,
    /// Number of active items.
    pub count: usize,
}

impl MyMetricsSoA {
    pub fn new(capacity: usize) -> Self {
        let aligned = ((capacity + 7) / 8) * 8;
        Self {
            field1: vec![0; aligned],
            field2: vec![0.0; aligned],
            count: 0,
        }
    }
}
```

### Step 2: Create the Collector

In `collectors/my_simd.rs`:

```rust
use crate::monitor::error::Result;
use crate::monitor::simd::{kernels, SimdRingBuffer};
use crate::monitor::types::{Collector, Metrics};

#[derive(Debug)]
pub struct SimdMyCollector {
    // SoA storage
    metrics: MyMetricsSoA,
    // History buffers
    history: SimdRingBuffer,
}

impl Collector for SimdMyCollector {
    fn id(&self) -> &'static str {
        "my_collector_simd"
    }

    fn collect(&mut self) -> Result<Metrics> {
        // 1. Read raw data
        // 2. Parse with SIMD kernels
        // 3. Store in SoA layout
        // 4. Update history
        // 5. Return metrics
    }

    fn display_name(&self) -> &'static str {
        "My Collector (SIMD)"
    }
}
```

### Step 3: Register in `collectors/mod.rs`

```rust
pub mod my_simd;
pub use my_simd::SimdMyCollector;
```

## Performance Guidelines

### Memory Alignment

All SoA structures must be 64-byte aligned for AVX-512 compatibility:

```rust
#[repr(C, align(64))]
pub struct MyStruct { ... }
```

### Zero-Allocation Hot Paths

Avoid allocations in `collect()`:

```rust
// BAD: Allocates on every call
fn collect(&mut self) -> Result<Metrics> {
    let buffer = Vec::new();  // Allocation!
    ...
}

// GOOD: Reuse pre-allocated buffer
fn collect(&mut self) -> Result<Metrics> {
    self.buffer.clear();  // No allocation
    ...
}
```

### Chunk Processing

Process data in SIMD-friendly chunks:

```rust
const CHUNK_SIZE: usize = 8;  // AVX2 = 4 f64, process 2 at a time

for chunk in data.chunks(CHUNK_SIZE) {
    // SIMD process chunk
}
```

## Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench --features monitor

# Run specific benchmark
cargo bench --features monitor simd_parse

# Generate HTML report
cargo bench --features monitor -- --noplot
```

### Writing Benchmarks

In `benches/simd_bench.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_simd_operation(c: &mut Criterion) {
    let data: Vec<u64> = (0..1000).collect();

    c.bench_function("simd_my_operation", |b| {
        b.iter(|| simd_my_operation(black_box(&data)))
    });
}

criterion_group!(benches, bench_simd_operation);
criterion_main!(benches);
```

### Performance Targets

From the specification, key targets are:

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Integer parsing | ≥8x vs scalar | H₁ |
| Ring buffer push | ≥5x vs VecDeque | H₉ |
| Correlation (1K samples) | ≥8x vs scalar | H₁₂ |
| End-to-end collection | p99 < 500μs | H₅ |

## Testing Checklist

Before submitting a PR:

- [ ] All existing tests pass: `cargo test --features monitor`
- [ ] New tests added for new functionality
- [ ] Performance tests validate targets
- [ ] No new clippy warnings: `cargo clippy --features monitor`
- [ ] Documentation updated if API changed
- [ ] Integration tests added to `integration_tests.rs`

## Code Style

### Documentation

Every public function needs:

```rust
/// Brief description of function.
///
/// # Performance Target
/// - Specific measurable target
///
/// # Arguments
/// * `arg1` - Description
///
/// # Returns
/// Description of return value
///
/// # Example
/// ```
/// let result = function(input);
/// ```
#[must_use]
pub fn function(arg1: Type) -> ReturnType { ... }
```

### Error Handling

Use the `Result` type from `monitor::error`:

```rust
use crate::monitor::error::{MonitorError, Result};

fn might_fail() -> Result<Value> {
    // On error:
    return Err(MonitorError::CollectionFailed {
        collector: "my_collector".to_string(),
        details: "specific error".to_string(),
    });
}
```

## Questions?

- Check the specification: `docs/specifications/full-SIMD-collectors-queue.md`
- Review existing collectors for patterns
- Open an issue for design discussions
