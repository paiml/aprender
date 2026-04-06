# ADR 0005: Random Seed Management Strategy

## Status

Accepted

## Context

Reproducibility is a core requirement for scientific software. Random number generation
introduces non-determinism that can make tests flaky and benchmarks unreliable. We need
a consistent strategy for managing random seeds across:

1. Unit tests
2. Property-based tests (proptest)
3. Benchmarks (criterion)
4. Python utilities
5. ML model training (if applicable)

## Decision

We will implement a hierarchical random seed management system:

### Seed Hierarchy

```
RANDOM_SEED (global default: 42)
├── PRESENTAR_TEST_SEED (test-specific, inherits RANDOM_SEED)
├── PRESENTAR_BENCH_SEED (benchmark-specific: 12345)
├── PROPTEST_SEED (proptest: 0xdeadbeef)
└── CRITERION_SEED (criterion: 42)
```

### Implementation

1. **Rust Module** (`crates/presentar-terminal/src/random_seed.rs`):
   - `DEFAULT_SEED = 42`
   - `set_global_seed(u64)` - sets global seed
   - `get_seed() -> u64` - gets current seed
   - `init_from_env()` - reads RANDOM_SEED from environment
   - `with_seed(u64, FnOnce)` - scoped seed override
   - `SeededRng` - deterministic xorshift64 PRNG

2. **Python Module** (`conftest.py`):
   - `set_seed(int)` - sets random, numpy, torch, tensorflow seeds
   - Session-scoped fixture for automatic seeding

3. **Environment Configuration** (`.env.example`):
   - All seed variables documented
   - Default values provided

### Usage Patterns

```rust
// Rust: Use seeded RNG
use presentar_terminal::random_seed::{SeededRng, get_seed};

let mut rng = SeededRng::new(get_seed());
let value = rng.next_u64();

// Scoped seed override
with_seed(12345, || {
    // deterministic operations
});
```

```python
# Python: Seeds set automatically via conftest.py
def test_something(reproducible_seed):
    assert reproducible_seed == 42
```

## Consequences

### Positive

- All tests are deterministic and reproducible
- Flaky tests can be debugged by reproducing exact seed
- Benchmark results are comparable across runs
- Satisfies F1 Popperian criterion for ML reproducibility

### Negative

- Slight performance overhead from seed management
- Requires discipline to use SeededRng instead of rand::thread_rng()

### Neutral

- Need to document seed values in all benchmark reports
- CI must use fixed seeds via environment variables

## References

- [Reproducible ML](https://reproducible.ml/)
- [NumPy Reproducibility](https://numpy.org/doc/stable/reference/random/generator.html#reproducibility)
- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
