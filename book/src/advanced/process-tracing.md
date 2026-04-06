# Process-Level Tracing

The `process_tracer` module provides deep process-level syscall tracing for integration
with ptop (process top) and presentar visualization tools. This enables real-time
anomaly detection using statistical process control (Shewhart z-score methodology).

## Overview

Process tracing allows you to:

- Attach to running processes and trace syscalls in real-time
- Categorize syscalls into buckets (mmap, futex, ioctl, read, write, other, compute)
- Detect anomalies using z-score statistical analysis
- Export traces to OTLP format for Jaeger/Tempo integration
- Stream syscalls for real-time visualization

## Quick Start

```rust
use renacer::process_tracer::{
    attach, detach, collect, ProcessTraceConfig,
};
use std::time::Duration;

// Configure the tracer
let config = ProcessTraceConfig::default()
    .with_timeout(Duration::from_secs(5))
    .with_anomaly_threshold(3.0)  // 3 sigma threshold
    .with_source_correlation(true);

// Attach to a running process
let mut trace = attach(target_pid, config)?;

// Collect syscalls
let result = collect(&mut trace)?;

// Analyze results
println!("Collected {} syscalls", result.events.len());
println!("Max z-score: {:.2}", result.max_zscore);

for anomaly in &result.anomalies {
    println!("Anomaly: {} at {}us (z={:.2})",
        anomaly.syscall, anomaly.duration_us, anomaly.zscore);
}

// Clean detach
detach(trace)?;
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `timeout` | 5s | Maximum trace duration |
| `max_syscalls` | 10000 | Maximum events to collect |
| `anomaly_threshold` | 3.0 | Z-score threshold for anomalies |
| `source_correlation` | false | Enable DWARF source lookup |
| `rate_limit` | 100 | Max traces per second |

## Syscall Categories

The `SyscallBreakdown` categorizes syscalls into seven buckets:

```rust
pub struct SyscallBreakdown {
    pub mmap_us: u64,     // mmap, munmap, mprotect, brk, mremap
    pub futex_us: u64,    // futex (thread synchronization)
    pub ioctl_us: u64,    // ioctl (device control)
    pub read_us: u64,     // read, pread64, readv, preadv*
    pub write_us: u64,    // write, pwrite64, writev, pwritev*
    pub other_us: u64,    // All other syscalls
    pub compute_us: u64,  // Time between syscalls (CPU work)
}
```

### Efficiency Metric

The `efficiency()` method returns the ratio of compute time to total time:

```rust
let breakdown = SyscallBreakdown::from_events(&events, total_us);
let efficiency = breakdown.efficiency();  // 0.0 to 1.0
println!("Compute efficiency: {:.1}%", efficiency * 100.0);
```

A high efficiency (>80%) indicates the process is compute-bound.
A low efficiency (<50%) indicates I/O or synchronization bottlenecks.

## Z-Score Anomaly Detection

Anomalies are detected using Shewhart statistical process control:

```rust
use renacer::process_tracer::{compute_baseline, zscore};

// Build baseline from historical data
let baseline = compute_baseline(&historical_events);

// Check if an event is anomalous
let z = zscore(&event, &baseline);
if z.abs() > 3.0 {
    println!("Anomaly detected: z-score = {:.2}", z);
}
```

### Z-Score Thresholds

| Threshold | False Positive Rate | Use Case |
|-----------|---------------------|----------|
| 2.0σ | 4.55% | Aggressive detection |
| 3.0σ | 0.27% | Standard (recommended) |
| 4.0σ | 0.006% | Conservative |

## Streaming API

For real-time visualization, use the streaming iterator:

```rust
use renacer::process_tracer::stream_syscalls;

let stream = stream_syscalls(pid, config)?;

for event in stream {
    // Process each syscall in real-time
    println!("{}: {}us", event.syscall, event.duration.as_micros());

    // Check for anomalies
    let z = zscore(&event, &baseline);
    if z.abs() > threshold {
        trigger_alert(&event, z);
    }
}
```

## OTLP Export

Export traces to OpenTelemetry-compatible backends:

```rust
let result = collect(&mut trace)?;
let span = result.to_otlp_span();

// Span includes:
// - trace_id, span_id
// - pid, duration
// - breakdown attributes (mmap_us, futex_us, etc.)
// - anomaly events
// - max_zscore
```

### Span Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `process.pid` | int | Process ID |
| `syscall.count` | int | Total syscall count |
| `syscall.mmap_us` | int | mmap category time |
| `syscall.futex_us` | int | futex category time |
| `syscall.read_us` | int | read category time |
| `syscall.write_us` | int | write category time |
| `syscall.compute_us` | int | Compute time |
| `anomaly.max_zscore` | float | Maximum z-score |
| `anomaly.count` | int | Number of anomalies |

## Error Handling

```rust
use renacer::process_tracer::TracerError;

match attach(pid, config) {
    Ok(trace) => { /* success */ }
    Err(TracerError::PermissionDenied(msg)) => {
        eprintln!("Need CAP_SYS_PTRACE: {}", msg);
    }
    Err(TracerError::ProcessNotFound { pid }) => {
        eprintln!("Process {} not found", pid);
    }
    Err(TracerError::AlreadyTraced { pid }) => {
        eprintln!("Process {} already being traced", pid);
    }
    Err(TracerError::RateLimitExceeded { limit }) => {
        eprintln!("Rate limit {} traces/sec exceeded", limit);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Running the Example

```bash
# Run the demo with synthetic data
cargo run --example process_tracer_demo

# Trace a real process (requires root)
sudo cargo run --example process_tracer_demo -- <PID>
```

## ptop Integration

This module is designed for integration with ptop (process top) in the
presentar visualization suite. See the specification at:
`docs/specifications/ptop-presentar-tracing-support.md`

### Integration Points

1. **Escalation**: ptop calls `attach()` when user selects a process
2. **Collection**: `collect()` gathers syscall breakdown
3. **Visualization**: `SyscallBreakdown` renders in Trace Panel
4. **Anomalies**: Z-score anomalies trigger alerts

## Scientific Foundation

The anomaly detection is based on:

- **Shewhart Control Charts** (1931) - Statistical process control
- **Dapper** (Sigelman et al., 2010) - Adaptive sampling and rate limiting
- **Pivot Tracing** (Mace et al., 2015) - Dynamic instrumentation

## See Also

- [Anomaly Detection](./anomaly-detection.md)
- [OpenTelemetry Integration](./opentelemetry.md)
- [ComputeBrick Tracing](./brick-tracing.md)
