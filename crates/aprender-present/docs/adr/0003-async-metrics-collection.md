# ADR-0003: Async Metrics Collection

**Status:** Accepted
**Date:** 2026-01-12
**Decision Makers:** Engineering Team

## Context

System metrics collection (CPU, memory, disk, network) can be slow due to:
- Reading from `/proc` filesystem
- Parsing text files
- Network latency for container APIs (Docker/Podman)

Blocking the render loop for metrics collection causes visible stuttering.

## Decision

We implement **async metrics collection** with:

1. Dedicated collector thread running at configurable interval
2. Lock-free channel for snapshot delivery
3. Render loop reads latest snapshot without blocking
4. Graceful degradation (shows stale data if collector is slow)

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│ Collector Thread│     │  Render Thread  │
│                 │     │                 │
│  loop {         │     │  loop {         │
│    collect()    │────▶│    recv_snap()  │
│    send(snap)   │     │    render()     │
│    sleep(100ms) │     │    present()    │
│  }              │     │  }              │
└─────────────────┘     └─────────────────┘
```

### Interface Contract (Falsifiable)

```rust
/// CLAIM: Snapshot delivery latency < 10ms (P99)
/// FALSIFICATION: Measure 1000 deliveries, check P99
#[test]
fn falsify_snapshot_latency() {
    let (tx, rx) = channel();
    let latencies: Vec<Duration> = (0..1000)
        .map(|_| measure_delivery(&tx, &rx))
        .collect();
    let p99 = percentile(&latencies, 99);
    assert!(p99 < Duration::from_millis(10),
        "FALSIFIED: P99 latency {} > 10ms", p99.as_millis());
}
```

## Consequences

### Positive
- Smooth 60fps rendering even during heavy I/O
- Predictable frame times
- Can adjust collection frequency per-metric

### Negative
- More complex architecture
- Potential for stale data display
- Thread synchronization overhead

## References

- `crates/presentar-terminal/src/ptop/app.rs` - AsyncCollector
- `tests/cpu_exploded_async.rs` - Interface tests
