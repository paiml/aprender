# Heijunka: Level Scheduling

**Heijunka** (平準化) means "leveling" - the practice of smoothing workload to prevent resource spikes and idle periods.

## Core Principle

> Level the load. Bursty demand causes waste; steady flow maximizes throughput.

In Batuta, Heijunka governs how compute workloads are distributed across CPU, GPU, and SIMD backends to prevent any single resource from becoming a bottleneck.

## Heijunka in Batuta

### MoE Backend Selection

The Mixture-of-Experts backend selector levels load across compute targets:

```rust
// Heijunka: select backend based on current load, not just capability
let backend = BackendSelector::new()
    .with_cost_model(CostModel::Gregg5x)  // 5x PCIe transfer rule
    .with_load_balancing(true)              // Level across backends
    .select(&operation);

// Small matrix multiply → SIMD (avoid GPU transfer overhead)
// Large batch inference → GPU (amortize PCIe cost)
// Mixed workload → distribute across both
```

### The 5x PCIe Rule

Backend selection follows Gregg & Hazelwood (2011): GPU dispatch is only worthwhile when compute savings exceed 5x the PCIe transfer cost.

| Operation Size | Transfer Cost | Compute Savings | Backend |
|---------------|---------------|-----------------|---------|
| < 1K elements | Low | < 2x | Scalar |
| 1K - 100K | Medium | 2-5x | SIMD (AVX2/AVX-512) |
| > 100K | High | > 5x | GPU (wgpu) |

### Spillover Routing

The serve module implements Heijunka for inference requests:

```rust
// Heijunka: spillover prevents overloading primary backend
pub fn route_request(req: &InferenceRequest, state: &ServerState) -> Backend {
    let primary = state.primary_backend();

    if primary.queue_depth() < primary.capacity() {
        primary  // Primary has headroom
    } else {
        state.spillover_backend()  // Level to secondary
    }
}
```

### Circuit Breakers

Cost circuit breakers prevent runaway GPU usage — a Heijunka safety valve:

```bash
# Circuit breaker configuration
# batuta.toml
[serve.circuit_breaker]
gpu_cost_limit = 100.0      # Max GPU-seconds per minute
queue_depth_limit = 64       # Max queued requests
fallback = "cpu"             # Degrade gracefully to CPU
```

When the GPU budget is exhausted, requests spill over to CPU/SIMD backends rather than queuing unboundedly. Load stays level.

### Stack Release Leveling

Releases across the Sovereign AI Stack are leveled to avoid dependency cascades:

```
Week 1: trueno 0.16.1          (foundation)
Week 2: aprender 0.27.2        (depends on trueno)
Week 3: realizar 0.8.0         (depends on both)
Week 4: batuta 0.7.2           (orchestration)
```

Sequential, leveled releases prevent the "big bang" integration problem.

## Benefits

1. **No resource spikes** - GPU and CPU utilization stays predictable
2. **Cost control** - Circuit breakers enforce budget limits
3. **Graceful degradation** - Spillover routing prevents failures under load
4. **Predictable latency** - Level scheduling avoids queuing delays

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Next: Kanban](./kanban.md)
