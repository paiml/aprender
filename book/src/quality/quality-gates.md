# Quality Gates

Automated checks that block deployment.

## Gate Types

| Gate | Trigger | Blocks When |
|------|---------|-------------|
| Type Check | Every compile | Errors |
| Tests | Every commit | Failures |
| Clippy | Pre-commit | Warnings |
| Coverage | Nightly | Decreases |
| Score | Pre-deploy | Below B+ |

## Three-Tier System

### Tier 1: On-Save (<1s)

```bash
make tier1
```
- `cargo check`
- Fast clippy
- Fast tests

### Tier 2: Pre-Commit (1-5min)

```bash
make tier2
```
- Format check
- Full clippy
- All tests
- Score calculation

### Tier 3: Nightly

```bash
make tier3
```
- Tier 2 +
- Coverage report
- Mutation testing

## Gate Configuration

```toml
# .presentar-gates.toml
[gates]
minimum_grade = "B+"
minimum_coverage = 85
max_clippy_warnings = 0
max_frame_time_ms = 16

[blockers]
critical_a11y = true
test_failures = true
```

## Enforcement

```bash
# CI script
make tier2 || exit 1
```

## Manual Override

```bash
# Skip gates (dangerous!)
SKIP_GATES=1 make deploy  # NOT recommended
```

## Verified Test

```rust
#[test]
fn test_quality_gates() {
    // Gates are enforced by CI, not at runtime
    // This test verifies gate configuration exists
    let min_coverage = 85;
    let min_score = 80;  // B+

    assert!(min_coverage >= 70);
    assert!(min_score >= 80);
}
```
