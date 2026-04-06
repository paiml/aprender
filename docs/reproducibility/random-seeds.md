# Random Seed Management

**Status:** Active
**Last Updated:** 2026-01-12

## Purpose

Ensure all stochastic operations in presentar are reproducible by controlling random seeds.

## Seed Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRESENTAR_TEST_SEED` | `42` | Seed for unit test randomness |
| `PROPTEST_SEED` | `0xdeadbeef` | Seed for property-based tests |
| `PRESENTAR_BENCH_SEED` | `12345` | Seed for benchmark data generation |

### Code Usage

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Create deterministic RNG from environment or default
pub fn deterministic_rng() -> ChaCha8Rng {
    let seed: u64 = std::env::var("PRESENTAR_TEST_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    ChaCha8Rng::seed_from_u64(seed)
}
```

### Deterministic Mode

The `--deterministic` flag enables fully reproducible output:

```bash
# Run ptop with deterministic data (for testing/screenshots)
ptop --deterministic

# Run tests with fixed seed
PRESENTAR_TEST_SEED=42 cargo test
```

### Sources of Non-Determinism

| Source | Mitigation |
|--------|------------|
| System time | Use monotonic clock or mock |
| Process list | Sort by PID before display |
| HashMap iteration | Use BTreeMap or sorted iteration |
| Thread scheduling | Use single-threaded test mode |
| Floating point | Use integer math where possible |

## Property-Based Testing Seeds

PropTest configuration in `proptest.toml`:

```toml
[proptest]
# Fixed seed for CI reproducibility
# Override with PROPTEST_SEED env var
seed = 0xdeadbeef

# Cases per test
cases = 256

# Max shrink iterations
max_shrink_iters = 10000
```

## Verification

```bash
# Verify deterministic mode produces identical output
for i in {1..10}; do
  ptop --deterministic --once > /tmp/out$i.txt
done
md5sum /tmp/out*.txt  # All should match
```

## References

- [Reproducible Builds](https://reproducible-builds.org/)
- [PropTest Book](https://altsysrq.github.io/proptest-book/)
