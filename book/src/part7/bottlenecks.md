# Bottleneck Identification

Identifying the true bottleneck before optimizing saves weeks of wasted effort. This chapter covers CPU profiling, syscall analysis, and memory allocation tracking.

## CPU Profiling with Flamegraph

```bash
cargo install flamegraph
cargo flamegraph --root --bin batuta -- analyze /path/to/project
```

### Reading the Flamegraph

| Pattern | Meaning | Action |
|---------|---------|--------|
| Wide plateau at top | Single function dominates | Optimize or parallelize |
| Many thin towers | Overhead spread evenly | Algorithmic improvement |
| Deep call stack | Excessive abstraction | Consider inlining |
| `alloc::` frames | Allocation overhead | Pre-allocate or stack buffers |

## Syscall Analysis with renacer

```bash
renacer trace -- batuta transpile --source ./src
```

| Symptom | Syscall Pattern | Fix |
|---------|-----------------|-----|
| Slow file I/O | Many small `read()` calls | `BufReader` |
| Slow startup | Many `open()` on configs | Lazy load or `include_str!` |
| Memory pressure | Frequent `mmap`/`munmap` | Pre-allocate, reuse buffers |
| Lock contention | `futex()` spinning | Reduce critical section |

## Memory Allocation Tracking

```rust
// Reuse buffers instead of allocating
let mut buffer = Vec::with_capacity(max_item_size);
for item in items {
    buffer.clear();
    buffer.extend_from_slice(item);
    process(&buffer);
}
```

## The Bottleneck Decision Tree

```
CPU-bound? (check with perf stat)
├── Yes -> Flamegraph -> Find hot function -> Optimize or SIMD
└── No
    ├── I/O-bound? (renacer trace)
    │   ├── Disk -> Buffered I/O, mmap, async
    │   └── Network -> Connection pooling, batching
    └── Memory-bound? (perf stat bandwidth)
        ├── Allocation-heavy -> DHAT, pre-allocate
        └── Cache-miss-heavy -> Improve data layout
```

The 2.05x throughput improvement in [Profiling](./profiling.md) was discovered by this process: `perf stat` showed low IPC, flamegraph showed rayon sync overhead, reducing threads from 48 to 16 eliminated cache line bouncing.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
