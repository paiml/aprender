# Optimization Iteration

Optimization is a scientific process: measure, hypothesize, change, measure again.

## The Iteration Cycle

1. **Measure**: Establish a baseline with Criterion
2. **Hypothesize**: Form a testable prediction ("removing this allocation will improve throughput by 15%")
3. **Change**: Make exactly one change
4. **Measure**: Compare with statistical rigor

```bash
cargo bench -- --save-baseline before
# Make the change
cargo bench -- --baseline before
```

## Avoiding Premature Optimization

| Question | If Yes | If No |
|----------|--------|-------|
| On the hot path? | Optimize | Skip |
| Profiling shows > 5% of time? | Optimize | Skip |
| Users notice the improvement? | Optimize | Skip |
| Code already simple? | Consider optimizing | Simplify first |

## Common Patterns

### Replace Allocation with Buffer Reuse

```rust
// Before: heap allocation per call
fn format_key(prefix: &str, id: u64) -> String {
    format!("{}_{}", prefix, id)
}

// After: reusable buffer
fn format_key(prefix: &str, id: u64, buf: &mut String) {
    buf.clear();
    buf.push_str(prefix);
    buf.push('_');
    buf.push_str(&id.to_string());
}
```

### Enable SIMD via trueno

```rust
use trueno::Vector;
let v = Vector::from_slice(data);
let sum = v.sum();  // Automatic AVX2/AVX-512/NEON
```

## Tracking Optimization History

| Date | Target | Hypothesis | Result | Kept? |
|------|--------|------------|--------|-------|
| 2025-03 | matmul | SIMD 4x throughput | 3.8x | Yes |
| 2025-04 | parser | Preallocate AST nodes | 2% | No |
| 2025-05 | inference | Reduce threads 48->16 | 2.05x | Yes |

Failed optimizations are valuable data. Recording them prevents repeating experiments.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
