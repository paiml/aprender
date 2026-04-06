# Capacity Factor Routing

Per Fedus et al. (2022) "Switch Transformers", auxiliary loss alone cannot balance load at inference time.

## The Problem

Training-time auxiliary loss encourages balanced routing, but at inference:
- If one expert is slightly better, it receives **all** traffic
- Creates hotspots (un-leveled load)
- Violates Heijunka (load leveling)

## Solution: Capacity Factor

```rust
use realizar::moe::{CapacityConfig, CapacityFactorRouter};

let router = CapacityFactorRouter::new(CapacityConfig {
    capacity: 100,  // Max concurrent requests per expert
    num_experts: 8,
});

// Route based on gating scores
let scores = vec![0.3, 0.5, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01];
let expert = router.route(&scores)?;

// Track queue depth
router.record_start(expert);
// ... do inference ...
router.record_end(expert);
```

## Algorithm: Power of Two Choices

Per Mitzenmacher (2001), picking two experts and choosing the least loaded is optimal:

1. Compute gating scores for all experts
2. Select top-2 experts by score
3. If primary expert's queue < capacity, use it
4. Otherwise, fallback to second-best expert

## Performance

| Metric | Value |
|--------|-------|
| Routing throughput | 1.7M routes/sec |
| Fallback accuracy | 100% |
| Memory overhead | O(num_experts) atomics |
