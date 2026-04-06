# Sprint 36: Performance Optimization Specification

## Overview

**Goal**: Optimize Renacer's performance for production workloads by reducing overhead, improving memory efficiency, and implementing intelligent batching strategies.

**Current State** (v0.5.0):
- OTLP export with async Tokio runtime
- Span-per-syscall model with immediate export
- Dynamic allocations for span data
- No formal benchmark suite
- Performance characteristics unknown under load

**Target State** (Sprint 36):
- <5% overhead for basic tracing (vs. current unknown baseline)
- <10% overhead for full observability stack (OTLP + compute + distributed)
- Memory-efficient span handling with pooling
- Batch OTLP export to reduce network calls
- Zero-copy optimizations where possible
- Lazy span creation to defer work
- Automated benchmark suite for regression detection

## Performance Goals

### Overhead Targets
1. **Basic Syscall Tracing**: <3% overhead vs. native execution
2. **OTLP Export Only**: <5% overhead
3. **Full Stack** (OTLP + compute + distributed): <10% overhead
4. **Memory Footprint**: <50MB for typical workloads (10K syscalls)

### Throughput Targets
1. **Syscalls/sec**: >100K syscalls/sec tracing throughput
2. **OTLP Export**: >10K spans/sec export rate
3. **Batch Size**: Configurable batching (default: 512 spans)
4. **Flush Latency**: <100ms p99 flush latency

## Architecture Changes

### 1. Memory Pool Allocations

**Problem**: Current implementation allocates span data on-demand, causing allocator pressure and potential fragmentation.

**Solution**: Implement object pooling for frequently allocated structures.

```rust
// src/span_pool.rs
pub struct SpanPool {
    pool: Vec<Box<SpanData>>,
    capacity: usize,
    allocated: AtomicUsize,
}

impl SpanPool {
    pub fn new(capacity: usize) -> Self {
        // Pre-allocate pool
    }

    pub fn acquire(&mut self) -> SpanHandle {
        // Reuse or allocate
    }

    pub fn release(&mut self, handle: SpanHandle) {
        // Return to pool
    }
}
```

**Key Features**:
- Pre-allocated pool of span objects (default: 1024)
- O(1) acquire/release operations
- Automatic growth if pool exhausted
- Configurable via `--span-pool-size N`
- Zero-cost when disabled

**Expected Impact**: 20-30% reduction in allocation overhead

### 2. Batch OTLP Export

**Problem**: Current implementation exports spans individually or in small groups, causing excessive network calls.

**Solution**: Buffer spans and export in configurable batches.

```rust
// src/otlp_exporter.rs modifications
pub struct BatchConfig {
    max_batch_size: usize,      // Default: 512
    max_batch_delay_ms: u64,    // Default: 1000ms
    max_queue_size: usize,      // Default: 2048
}

pub struct BatchedExporter {
    buffer: VecDeque<SpanData>,
    config: BatchConfig,
    last_flush: Instant,
}
```

**Key Features**:
- Batch spans up to `--otlp-batch-size N` (default: 512)
- Auto-flush after `--otlp-batch-delay MS` (default: 1000ms)
- Bounded queue with backpressure handling
- Immediate flush on program exit
- Per-batch compression (gzip)

**Expected Impact**: 40-60% reduction in network overhead

### 3. Zero-Copy Optimizations

**Problem**: Multiple data copies when building OTLP protobuf messages.

**Solution**: Use references and in-place building where possible.

**Optimization Areas**:
1. **String Handling**: Use `Cow<'static, str>` for known strings
2. **Attribute Values**: Avoid intermediate allocations
3. **Span References**: Use borrows instead of clones
4. **Buffer Reuse**: Reuse protobuf encoding buffers

```rust
// Before
let span = SpanData {
    name: syscall_name.to_string(),  // Allocation
    attributes: attrs.clone(),        // Clone
    ...
};

// After
let span = SpanData {
    name: Cow::Borrowed(syscall_name),  // Zero-copy
    attributes: attrs,                   // Move
    ...
};
```

**Expected Impact**: 10-15% reduction in memory allocations

### 4. Lazy Span Creation

**Problem**: Spans are created even if they won't be exported (e.g., no OTLP endpoint configured).

**Solution**: Defer span creation until export is confirmed.

```rust
// src/tracer.rs modifications
pub enum SpanMode {
    Disabled,           // No overhead
    Lazy(SpanBuilder),  // Deferred creation
    Immediate(Span),    // Full span
}

impl Tracer {
    fn record_syscall(&mut self, syscall: &Syscall) {
        if self.otlp_enabled {
            // Only create span if needed
            let span = self.build_span(syscall);
            self.export_span(span);
        }
    }
}
```

**Expected Impact**: 5-10% overhead reduction when features disabled

### 5. Benchmark Suite

**Problem**: No automated way to detect performance regressions.

**Solution**: Comprehensive benchmark suite using `criterion`.

**Benchmark Categories**:

1. **Syscall Tracing Overhead**
   - `bench_native_baseline` - Raw program execution
   - `bench_basic_tracing` - Tracing without OTLP
   - `bench_otlp_export` - With OTLP export
   - `bench_full_stack` - All features enabled

2. **OTLP Export Performance**
   - `bench_span_creation` - Span building
   - `bench_batch_export` - Batch export throughput
   - `bench_protobuf_encoding` - Serialization speed
   - `bench_network_overhead` - Network I/O

3. **Memory Operations**
   - `bench_span_pool_acquire` - Pool allocation speed
   - `bench_span_pool_release` - Pool release speed
   - `bench_memory_footprint` - Memory usage over time

4. **Concurrency**
   - `bench_multi_thread_tracing` - Multi-process scenarios
   - `bench_concurrent_export` - Parallel export

**Directory Structure**:
```
benches/
├── syscall_overhead.rs      # End-to-end tracing overhead
├── otlp_export.rs           # OTLP-specific benchmarks
├── memory_pool.rs           # Pool allocation benchmarks
├── fixtures/
│   ├── syscall_heavy.rs    # Many syscalls
│   ├── compute_heavy.rs    # Compute-bound
│   └── io_heavy.rs         # I/O-bound
└── README.md               # Benchmark documentation
```

**Running Benchmarks**:
```bash
# All benchmarks
cargo bench

# Specific category
cargo bench --bench syscall_overhead

# With HTML report
cargo bench -- --save-baseline main
```

## Implementation Plan

### Phase 1: Benchmark Infrastructure (Week 1)
1. Add `criterion` dependency
2. Create benchmark directory structure
3. Implement baseline benchmarks (native execution)
4. Create test fixtures (syscall-heavy, compute-heavy, I/O-heavy)
5. Document benchmark methodology

**Deliverables**:
- `benches/syscall_overhead.rs` - Basic overhead benchmarks
- `benches/fixtures/` - Test programs
- `docs/BENCHMARKS.md` - Methodology documentation

### Phase 2: Memory Pool (Week 1)
1. Create `src/span_pool.rs` module
2. Implement object pool with pre-allocation
3. Add acquire/release operations
4. Integrate with span creation in `tracer.rs`
5. Add unit tests and benchmarks

**Deliverables**:
- `src/span_pool.rs` (300 lines, 15 tests)
- Integration in `src/tracer.rs`
- Benchmark showing 20-30% allocation reduction

### Phase 3: Batch OTLP Export (Week 2)
1. Add batch configuration to `OtlpConfig`
2. Implement buffering in `OtlpExporter`
3. Add time-based auto-flush
4. Handle graceful shutdown with flush
5. Add backpressure handling for queue overflow

**Deliverables**:
- Modified `src/otlp_exporter.rs` (+200 lines)
- CLI flags: `--otlp-batch-size`, `--otlp-batch-delay`
- Integration tests verifying batch behavior
- Benchmark showing 40-60% network reduction

### Phase 4: Zero-Copy Optimizations (Week 2)
1. Audit allocation hot paths
2. Replace `String` with `Cow<'static, str>` where possible
3. Use borrows instead of clones for span references
4. Reuse protobuf encoding buffers
5. Measure allocation reduction

**Deliverables**:
- Modified span building code
- Memory allocation benchmarks
- 10-15% reduction in allocations

### Phase 5: Lazy Span Creation (Week 3)
1. Add `SpanMode` enum
2. Implement lazy span builder
3. Defer creation until export confirmation
4. Add fast-path for disabled features
5. Verify zero overhead when disabled

**Deliverables**:
- Modified `src/tracer.rs`
- Benchmark showing <1% overhead when disabled

### Phase 6: Integration & Documentation (Week 3)
1. Run full benchmark suite
2. Compare before/after metrics
3. Update README with performance section
4. Update CHANGELOG
5. Create performance tuning guide

**Deliverables**:
- `docs/PERFORMANCE.md` - Tuning guide
- Updated README with benchmark results
- CHANGELOG entry for Sprint 36

## Testing Strategy

### Benchmark Tests
1. **Overhead Benchmarks**: Measure execution time with/without tracing
2. **Throughput Benchmarks**: Measure syscalls/sec and spans/sec
3. **Memory Benchmarks**: Measure memory footprint over time
4. **Latency Benchmarks**: Measure p50/p95/p99 flush latency

### Integration Tests
1. Verify batch export delivers all spans
2. Verify memory pool doesn't leak
3. Verify lazy spans work with all features
4. Verify graceful shutdown flushes batches

### Property-Based Tests
1. Pool acquire/release always balanced
2. Batch export preserves span ordering
3. Zero-copy doesn't corrupt data

## Configuration Options

### New CLI Flags

```bash
# Memory pool configuration
--span-pool-size N          # Pool capacity (default: 1024)
--span-pool-disable         # Disable pooling (for debugging)

# Batch export configuration
--otlp-batch-size N         # Max spans per batch (default: 512)
--otlp-batch-delay MS       # Max batch delay in ms (default: 1000)
--otlp-queue-size N         # Max queued spans (default: 2048)
--otlp-compression gzip     # Enable compression (default: off)

# Performance tuning
--perf-mode aggressive      # Maximize throughput (larger batches, more pooling)
--perf-mode balanced        # Default settings
--perf-mode low-latency     # Minimize latency (smaller batches, immediate flush)
```

### Performance Presets

**Balanced** (default):
- Pool size: 1024
- Batch size: 512
- Batch delay: 1000ms
- Queue size: 2048

**Aggressive** (max throughput):
- Pool size: 4096
- Batch size: 2048
- Batch delay: 5000ms
- Queue size: 8192

**Low-Latency** (min delay):
- Pool size: 256
- Batch size: 128
- Batch delay: 100ms
- Queue size: 512

## Success Criteria

### Performance Metrics
- [ ] <5% overhead for basic tracing
- [ ] <10% overhead for full stack
- [ ] >100K syscalls/sec throughput
- [ ] >10K spans/sec export rate
- [ ] <50MB memory footprint (10K syscalls)

### Code Quality
- [ ] Zero clippy warnings
- [ ] All tests passing (277+ existing + new)
- [ ] Benchmark suite integrated
- [ ] Documentation complete

### User Experience
- [ ] Simple performance presets (`--perf-mode`)
- [ ] Clear tuning guidance in docs
- [ ] Backward compatible (existing flags work)
- [ ] Zero-config good defaults

## Risks & Mitigations

### Risk 1: Batching Increases Latency
**Mitigation**: Configurable batch delay, auto-flush on exit, low-latency preset

### Risk 2: Pool Exhaustion Under Load
**Mitigation**: Automatic growth, configurable capacity, overflow handling

### Risk 3: Complexity Increase
**Mitigation**: Feature flags to disable optimizations, clear documentation

### Risk 4: Memory Leaks in Pool
**Mitigation**: Comprehensive testing, leak detection in CI, proper Drop implementation

## Future Enhancements (Post-Sprint 36)

1. **Lock-Free Data Structures**: Replace mutexes with atomic operations
2. **SIMD for Protobuf**: SIMD-accelerated protobuf encoding
3. **eBPF Backend**: Kernel-space tracing for minimal overhead
4. **Custom Allocator**: jemalloc or mimalloc for better performance
5. **Persistent Buffer**: Disk-backed buffer for extreme loads

## References

- OpenTelemetry SDK Performance Best Practices
- Criterion.rs Benchmarking Guide
- Rust Performance Book: https://nnethercote.github.io/perf-book/
- Object Pool Pattern: https://en.wikipedia.org/wiki/Object_pool_pattern
- Zero-Copy Networking in Rust
