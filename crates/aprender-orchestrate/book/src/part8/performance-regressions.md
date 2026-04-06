# Performance Regressions

Transpiled Rust code should be faster than the original, but regressions happen. This chapter covers the three most common causes.

## 1. Allocation Hotspots

The most frequent cause is excessive heap allocation from naive type translations:

```rust
// BAD: allocates every iteration
for line in lines {
    let tokens: Vec<&str> = line.split(',').collect();
    process(&tokens);
}

// GOOD: reuse the vector
let mut tokens: Vec<&str> = Vec::with_capacity(64);
for line in lines {
    tokens.clear();
    tokens.extend(line.split(','));
    process(&tokens);
}
```

Diagnose with `perf stat -e page-faults ./target/release/app`.

## 2. SIMD Not Engaging

Rust compiles for a conservative baseline CPU by default. AVX2/AVX-512 requires explicit opt-in:

```toml
# .cargo/config.toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

Or use trueno for automatic runtime SIMD dispatch:

```rust
use trueno::Vector;
let result = Vector::from_slice(&data).sum();
```

## 3. GPU Overhead Exceeding Benefit

The 5x PCIe rule: GPU compute must be 5x faster than CPU to overcome transfer overhead.

| Workload Size | CPU Time | GPU Total | Use GPU? |
|--------------|----------|-----------|----------|
| 1K elements | 0.1 ms | 0.52 ms | No |
| 100K elements | 10 ms | 1.0 ms | Yes |
| 10M elements | 1000 ms | 7 ms | Yes |

Batuta's backend selector applies this rule automatically.

## Regression Detection in CI

```bash
# Save baseline on main branch
cargo bench -- --save-baseline main

# On PR branch, compare
cargo bench -- --baseline main
```

Criterion reports statistical significance. A regression greater than 5% should block the merge.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
