# Regression Prevention

Regressions are defects that were previously fixed but reappear. During migration, they can be introduced by transpilation errors, optimization passes, or incorrect type mappings.

## Snapshot Testing

Capture known-good output and compare on every test run:

```rust
use insta::assert_snapshot;

#[test]
fn pipeline_report_format() {
    let report = generate_analysis_report("./fixtures/sample_project");
    assert_snapshot!(report);
}
```

Review and accept intentional changes with `cargo insta review`.

| Use Case | Snapshot Type |
|----------|--------------|
| CLI output format | String snapshot |
| JSON/TOML generation | String snapshot |
| Numeric results | Rounded string snapshot |
| Error messages | String snapshot |

## Benchmark Regression Detection

Use Criterion to detect performance regressions:

```bash
# Save baseline before migration
cargo bench -- --save-baseline before

# Compare after migration
cargo bench -- --baseline before
```

Criterion reports statistical significance: `+2.3% (p = 0.04)` means a real regression.

## CI Quality Gates

```bash
batuta stack gate
```

| Check | Threshold | Action on Failure |
|-------|-----------|-------------------|
| Test coverage | >= 90% | Block merge |
| Clippy warnings | 0 | Block merge |
| Cyclomatic complexity | <= 30 | Block merge |
| Cognitive complexity | <= 25 | Block merge |
| Mutation score | >= 80% | Warn |

## Regression Test Workflow

When a bug is found:

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Tag the test with the issue number

```rust
#[test]
fn regression_cb042_negative_stride() {
    // CB-042: Negative stride caused index overflow
    let result = transpose_with_stride(&data, -1);
    assert!(result.is_ok());
}
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)
