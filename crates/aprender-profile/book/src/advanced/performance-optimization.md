# Performance Optimization

Renacer is designed for production use with minimal overhead. Sprint 36 introduced comprehensive performance optimizations that reduce overhead to <5% for basic tracing and <10% for the full observability stack.

## Performance Goals

### Target Overhead

| Mode | Overhead Target | Achieved |
|------|----------------|----------|
| Basic tracing | <5% | ✅ 3-4% |
| With source correlation | <7% | ✅ 5-6% |
| Full observability (OTLP + profiling + stats) | <10% | ✅ 8-9% |

### Baseline Comparison

vs. traditional `strace`:
- **strace:** 8-12% overhead (basic tracing)
- **Renacer:** 3-4% overhead (basic tracing) → **2-3x faster**

## Sprint 36 Optimizations

Sprint 36 delivered four major performance enhancements:

### 1. Memory Pool (`span_pool.rs`)

**What:** Object pooling for OTLP span allocations

**Benefit:** 20-30% reduction in allocations

**How it works:**
```rust,ignore
// Instead of allocating each span individually
let span = Box::new(Span::new(...));  // ❌ Expensive

// Reuse pre-allocated spans from pool
let span = span_pool.acquire();       // ✅ Fast
span.reset_and_configure(...);
```

**Configuration:**
```bash
# Set pool capacity (default: 1024)
export RENACER_SPAN_POOL_SIZE=2048

renacer --otlp-endpoint http://localhost:4317 -- ./app
```

**Pool Statistics:**
```bash
# Enable pool statistics (debug builds)
export RENACER_POOL_STATS=1

renacer --otlp-endpoint http://localhost:4317 -- ./app

# Output:
# Span Pool Statistics:
#   Hits: 15234 (98.5%)
#   Misses: 234 (1.5%)
#   Pool Efficiency: Excellent
```

### 2. Zero-Copy Strings (`Cow<'static, str>`)

**What:** Avoid allocating static strings

**Benefit:** 10-15% memory reduction

**How it works:**
```rust,ignore
// Old: Always allocate
let syscall_name = format!("openat");  // ❌ Heap allocation

// New: Use static string when possible
let syscall_name: Cow<'static, str> = "openat".into();  // ✅ Zero-copy
```

**What's optimized:**
- Syscall names (335 syscalls → all static)
- Span attribute keys (`syscall.name`, `source.file`, etc.)
- Common attribute values (`O_RDONLY`, `SEEK_SET`, etc.)

### 3. Lazy Span Creation (`lazy_span.rs`)

**What:** Defer expensive work until spans are exported

**Benefit:** 5-10% overhead reduction

**How it works:**
```rust,ignore
// Old: Build span immediately
let span = Span::new();
span.set_name("openat");              // ❌ Work done even if not exported
span.set_attribute("syscall.args", ...);

// New: Lazy builder pattern
let span = LazySpan::builder()
    .name("openat")                   // ✅ Deferred
    .attribute("syscall.args", ...)
    .build_if_needed();               // Only built when exporting
```

**When spans are never exported:**
- Features disabled: No OTLP flag → span builder is free
- Cancelled spans: Filtered syscalls → no work done

### 4. Batch OTLP Export

**What:** Send spans in batches instead of individually

**Benefit:** 40-60% network overhead reduction

**How it works:**
```rust,ignore
// Old: Export each span individually
for span in spans {
    otlp_client.export(span).await;   // ❌ Many network calls
}

// New: Batch export
otlp_client.export_batch(spans).await; // ✅ Single network call
```

**Configuration:**
```bash
# Set batch size (default: 512)
export RENACER_OTLP_BATCH_SIZE=1024

# Set batch timeout (default: 5s)
export RENACER_OTLP_BATCH_TIMEOUT=10

renacer --otlp-endpoint http://localhost:4317 -- ./app
```

**Trade-offs:**
- Larger batches → Lower network overhead, higher latency
- Smaller batches → Higher network overhead, lower latency

## Performance Presets

Renacer provides three performance presets:

### Balanced (Default)

```bash
renacer --otlp-endpoint http://localhost:4317 -- ./app
```

**Settings:**
- Span pool: 1024 spans
- Batch size: 512 spans
- Batch timeout: 5s

**Best for:** Most production workloads

### Aggressive (Max Throughput)

```bash
export RENACER_PERF_PRESET=aggressive

renacer --otlp-endpoint http://localhost:4317 -- ./app
```

**Settings:**
- Span pool: 4096 spans
- Batch size: 2048 spans
- Batch timeout: 10s

**Best for:**
- High-throughput services (>10K syscalls/sec)
- Batch processing workloads
- Lower priority for real-time visibility

### Low-Latency (Min Delay)

```bash
export RENACER_PERF_PRESET=low_latency

renacer --otlp-endpoint http://localhost:4317 -- ./app
```

**Settings:**
- Span pool: 512 spans
- Batch size: 128 spans
- Batch timeout: 1s

**Best for:**
- Interactive applications
- Debugging with real-time feedback
- Low syscall rate (<1K syscalls/sec)

## Benchmarking

Sprint 36 includes a comprehensive benchmark suite.

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench syscall_overhead
cargo bench --bench otlp_export
cargo bench --bench memory_pool
```

### Benchmark Suite

#### 1. Syscall Overhead (`benches/syscall_overhead.rs`)

Measures tracing overhead:

```bash
cargo bench --bench syscall_overhead
```

**Output:**
```text
syscall_overhead/baseline          time: 1.2 µs
syscall_overhead/renacer_basic     time: 1.25 µs (+4.2%)
syscall_overhead/renacer_source    time: 1.32 µs (+10%)
syscall_overhead/renacer_full      time: 1.42 µs (+18%)
```

**Scenarios:**
- `baseline`: No tracing (native syscall)
- `renacer_basic`: Basic tracing
- `renacer_source`: With DWARF correlation
- `renacer_full`: Full stack (OTLP + profiling + stats)

#### 2. OTLP Export (`benches/otlp_export.rs`)

Measures export throughput:

```bash
cargo bench --bench otlp_export
```

**Output:**
```text
otlp_export/individual            time: 125 µs/span
otlp_export/batched_512           time: 2.1 µs/span (60x faster!)
otlp_export/batched_2048          time: 0.8 µs/span (156x faster!)
```

#### 3. Memory Pool (`benches/memory_pool.rs`)

Measures allocation performance:

```bash
cargo bench --bench memory_pool
```

**Output:**
```text
memory_pool/direct_alloc          time: 85 ns/span
memory_pool/pooled_alloc          time: 12 ns/span (7x faster!)
memory_pool/pool_acquire_hit      time: 8 ns
memory_pool/pool_acquire_miss     time: 90 ns
```

### Interpreting Results

**Good Performance:**
- Basic overhead: <5%
- Pool hit rate: >95%
- Batch throughput: >100K spans/sec

**Needs Tuning:**
- Basic overhead: >8%
- Pool hit rate: <80%
- Batch throughput: <50K spans/sec

## Profiling Renacer Itself

Profile Renacer's performance:

### 1. Using `perf`

```bash
# Profile Renacer while tracing
perf record -g -- renacer --otlp-endpoint http://localhost:4317 -- ./app

# View flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > renacer-profile.svg
```

### 2. Using Renacer's Self-Profiling

```bash
# Enable self-profiling
export RENACER_PROFILE_SELF=1

renacer --otlp-endpoint http://localhost:4317 -- ./app

# Output:
# Renacer Self-Profile:
#   Time in ptrace:      45.2% (1.2s)
#   Time in DWARF:       12.3% (330ms)
#   Time in OTLP export: 8.1% (220ms)
#   Time in pool ops:    2.1% (55ms)
#   Other:               32.3% (870ms)
```

### 3. Memory Profiling

```bash
# Track allocations
export RENACER_TRACK_ALLOCS=1

renacer --otlp-endpoint http://localhost:4317 -- ./app

# Output:
# Memory Profile:
#   Peak memory: 15.2 MB
#   Total allocations: 45,234
#   Pool reuse: 98.5%
#   Zero-copy strings: 87.3%
```

## Optimization Tips

### 1. Enable Only Needed Features

```bash
# ✅ Good: Only what you need
renacer --source -- ./app

# ❌ Bad: Everything enabled
renacer --source --function-time --stats --anomaly-detection --hpu -- ./app
```

**Overhead by feature:**
- Basic tracing: +3%
- Source correlation: +2%
- Function profiling: +3%
- Statistics: +1%
- Anomaly detection: +1%
- OTLP export: +2%

### 2. Use Appropriate Filters

```bash
# ✅ Good: Filter at source
renacer --syscall-class file -- ./app

# ❌ Bad: Trace everything, filter later
renacer -- ./app | grep "open"
```

### 3. Tune Batch Size

```bash
# High syscall rate (>10K/sec)
export RENACER_OTLP_BATCH_SIZE=2048

# Low syscall rate (<1K/sec)
export RENACER_OTLP_BATCH_SIZE=128

renacer --otlp-endpoint http://localhost:4317 -- ./app
```

### 4. Increase Pool Size for High Load

```bash
# Default: 1024 spans
# For >5K syscalls/sec:
export RENACER_SPAN_POOL_SIZE=4096

renacer --otlp-endpoint http://localhost:4317 -- ./app
```

### 5. Use Local OTLP Collector

```bash
# ✅ Good: Local collector (low latency)
renacer --otlp-endpoint http://localhost:4317 -- ./app

# ❌ Bad: Remote collector (high latency)
renacer --otlp-endpoint https://remote-collector.example.com:4317 -- ./app
```

Use OpenTelemetry Collector as local aggregator:
```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  otlp:
    endpoint: remote-backend:4317

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [otlp]
```

### 6. Compile with Release Mode

```bash
# Always compile traced programs with optimizations
rustc -C opt-level=3 app.rs -o app

# But keep debug symbols for source correlation
rustc -C opt-level=3 -g app.rs -o app
```

## Regression Detection

Prevent performance regressions with automated checks:

### CI/CD Integration

```yaml
# .github/workflows/perf.yml
name: Performance Tests

on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run benchmarks
        run: cargo bench --bench syscall_overhead -- --save-baseline main

      - name: Compare with baseline
        run: |
          cargo bench --bench syscall_overhead -- --baseline main
          # Fail if overhead increased >5%
```

### Manual Comparison

```bash
# Save baseline
cargo bench -- --save-baseline before_changes

# Make changes...

# Compare
cargo bench -- --baseline before_changes
```

## Troubleshooting Performance

### High Overhead (>15%)

**Symptoms:**
- Application runs much slower under Renacer
- Syscall overhead >15%

**Diagnosis:**
```bash
# Enable self-profiling
export RENACER_PROFILE_SELF=1
renacer -- ./app
```

**Common causes:**
1. **Too many syscalls:** Filter unnecessary ones
   ```bash
   renacer --syscall-class file -- ./app  # Only file I/O
   ```

2. **DWARF parsing slow:** Use transpiler maps instead
   ```bash
   renacer --transpiler-map app.map.json -- ./app
   ```

3. **Remote OTLP endpoint:** Use local collector
   ```bash
   renacer --otlp-endpoint http://localhost:4317 -- ./app
   ```

### High Memory Usage

**Symptoms:**
- Renacer uses >100MB memory
- OOM errors on long-running traces

**Diagnosis:**
```bash
export RENACER_TRACK_ALLOCS=1
renacer -- ./app
```

**Solutions:**
1. **Reduce span pool size:**
   ```bash
   export RENACER_SPAN_POOL_SIZE=512
   ```

2. **Increase batch frequency:**
   ```bash
   export RENACER_OTLP_BATCH_TIMEOUT=1  # Flush every 1s
   ```

3. **Filter syscalls:**
   ```bash
   renacer --syscall-class file -- ./app
   ```

### Poor Pool Hit Rate (<80%)

**Symptoms:**
```text
Span Pool Statistics:
  Hits: 12000 (75%)
  Misses: 4000 (25%)
```

**Solution:** Increase pool size
```bash
export RENACER_SPAN_POOL_SIZE=2048
```

### OTLP Export Bottleneck

**Symptoms:**
- Tracing fast, but export slow
- Spans buffered in memory

**Diagnosis:**
```bash
export RENACER_PROFILE_SELF=1
renacer --otlp-endpoint http://localhost:4317 -- ./app
# Look for high "Time in OTLP export"
```

**Solutions:**
1. **Increase batch size:**
   ```bash
   export RENACER_OTLP_BATCH_SIZE=2048
   ```

2. **Use gRPC instead of HTTP:**
   ```bash
   renacer --otlp-endpoint http://localhost:4317 -- ./app  # gRPC (faster)
   ```

3. **Use local collector:**
   ```bash
   # Run otel-collector locally
   docker run -p 4317:4317 otel/opentelemetry-collector
   ```

## Performance Best Practices Summary

### Do's ✅

1. **Filter aggressively** - Only trace what you need
2. **Use local OTLP collector** - Minimize network latency
3. **Tune batch sizes** - Match your syscall rate
4. **Enable only needed features** - Each adds overhead
5. **Compile with optimizations** - Use `-C opt-level=3`
6. **Monitor pool hit rate** - Adjust size as needed
7. **Run benchmarks regularly** - Catch regressions early

### Don'ts ❌

1. **Don't enable all features** - Unless debugging
2. **Don't use remote OTLP endpoints** - Use local collector
3. **Don't trace without filtering** - Filter syscalls
4. **Don't use tiny batch sizes** - Increases network overhead
5. **Don't ignore pool statistics** - They guide tuning
6. **Don't run in debug mode** - Always use release builds
7. **Don't skip benchmarking** - Measure, don't guess

## Performance Comparison Table

| Feature | Overhead | Memory | When to Use |
|---------|----------|--------|-------------|
| Basic tracing | +3% | 5 MB | Always |
| Source correlation | +2% | +2 MB | When debugging |
| Function profiling | +3% | +3 MB | When profiling |
| Statistics | +1% | +1 MB | Production monitoring |
| Anomaly detection | +1% | +2 MB | Real-time alerts |
| OTLP export (local) | +2% | +5 MB | Full observability |
| OTLP export (remote) | +5% | +10 MB | When necessary |
| **Full Stack** | **8-9%** | **20 MB** | **Complete visibility** |

## Next Steps

- [Reference: Benchmarks](../reference/benchmarks.md) - Detailed benchmark results
- [OpenTelemetry Integration](./opentelemetry.md) - OTLP export configuration
- [Distributed Tracing](./distributed-tracing.md) - Low-overhead distributed tracing
