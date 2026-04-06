# Performance Optimization

Performance is a first-class concern in the Sovereign AI Stack. Rust provides the foundation -- zero-cost abstractions, no garbage collector, predictable memory layout -- but realizing peak performance requires systematic measurement and targeted optimization.

## Performance Philosophy

The Toyota Production System principle of Muda (waste elimination) applies directly to performance work:

- **Overprocessing waste**: Optimizing code that is not on the hot path
- **Waiting waste**: Unnecessary synchronization or allocation
- **Transport waste**: Data copies between layers that could be avoided

## The Optimization Workflow

```
┌───────────┐     ┌──────────────┐     ┌────────┐     ┌───────────┐
│  Measure  │────>│ Hypothesize  │────>│ Change │────>│  Measure  │
│           │     │              │     │        │     │           │
│ Flamegraph│     │ "Allocation  │     │ Use    │     │ Confirm   │
│ Criterion │     │  is the      │     │ stack  │     │ improved  │
│ perf stat │     │  bottleneck" │     │ buffer │     │ or revert │
└───────────┘     └──────────────┘     └────────┘     └───────────┘
```

## Performance Tiers in the Stack

| Tier | Backend | When to Use | Throughput |
|------|---------|-------------|------------|
| Scalar | CPU, no SIMD | Baseline, correctness reference | 1x |
| SIMD | AVX2/AVX-512/NEON via trueno | Data-parallel operations | 4-16x |
| GPU | wgpu via repartir | Large matrix ops, training | 50-200x |
| Distributed | repartir remote | Multi-node workloads | Nx nodes |

Batuta's backend selector automatically chooses the right tier based on workload size and the 5x PCIe rule (GPU overhead must be recouped by at least 5x compute advantage).

## Key Tools

| Tool | Purpose | Command |
|------|---------|---------|
| Criterion | Micro-benchmarks with statistical rigor | `cargo bench` |
| Flamegraph | CPU profiling visualization | `cargo flamegraph` |
| renacer | Syscall-level tracing | `renacer trace ./target/release/app` |
| PMAT | Complexity and quality analysis | `pmat analyze complexity .` |
| perf stat | Hardware counter analysis | `perf stat ./target/release/app` |

## Rules of Thumb

1. **Measure before optimizing.** Intuition about bottlenecks is wrong more often than not.
2. **Optimize the algorithm first, then the implementation.** An O(n log n) sort in Python beats an O(n^2) sort in hand-tuned assembly.
3. **Allocation is the silent killer.** Track `Vec::new()` in hot loops with DHAT or custom allocators.
4. **SIMD requires data alignment.** Unaligned loads on AVX-512 cost 2-3x more than aligned loads.

See [Profiling](./profiling.md) for detailed profiling techniques, [Bottleneck Identification](./bottlenecks.md) for systematic root cause analysis, and [Optimization Iteration](./optimization-iteration.md) for the benchmark-driven development cycle.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
