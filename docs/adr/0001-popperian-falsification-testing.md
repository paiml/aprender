# ADR-0001: Popperian Falsification Testing

**Status:** Accepted
**Date:** 2026-01-11
**Decision Makers:** Engineering Team

## Context

Traditional software testing follows a confirmation bias pattern: write code, then write tests that confirm the code works. This approach fails to detect bugs that exist in the blind spots of the developer's mental model.

The presentar project experienced this failure mode directly. Tests reported "100% pass" while obvious visual bugs existed in the CPU panel (wrong temperatures, missing data, garbled text).

## Decision

We adopt Karl Popper's falsificationism as the epistemological foundation for our testing strategy.

### Key Principles

1. **Tests must TRY TO BREAK the code**, not confirm it works
2. **Bold conjectures**: Make specific, falsifiable claims that risk being wrong
3. **Severity levels**: Only tests that could possibly fail (S3+) are acceptable
4. **Corroboration over confirmation**: "We tried to break it and couldn't" vs "It works"

### Severity Scale

| Level | Name | Description |
|-------|------|-------------|
| S0 | Coconut Radio | Test cannot fail (tautology, mocked to pass) |
| S1 | Rubber Stamp | Test almost never fails |
| S2 | Soft | Weak falsification attempt |
| S3 | Rigorous | Strong falsification attempt |
| S4 | Ruthless | Maximum falsification potential |

Tests at S0-S2 are prohibited. Only S3+ tests are accepted.

### Implementation

Falsification tests are located in `tests/falsification_tests.rs` and follow this pattern:

```rust
/// FALSIFIABLE CLAIM: Load average matches /proc/loadavg
/// SEVERITY: S4 (compares against kernel source of truth)
#[test]
fn falsify_load_avg_vs_proc() {
    let app = App::new_with_deterministic(false);
    let proc_load: f64 = read_proc_loadavg();
    let app_load = app.load_avg.one;

    assert!(
        (app_load - proc_load).abs() < 0.5,
        "FALSIFIED: App shows {app_load:.2} but /proc/loadavg says {proc_load:.2}"
    );
}
```

## Consequences

### Positive
- Found 2 bugs immediately (k10temp parsing, load_avg initialization)
- Tests now serve as executable specifications
- Failure messages are diagnostic (show actual vs expected)
- Higher confidence in correctness

### Negative
- Requires more thought per test
- Some legacy tests need rewriting
- Learning curve for team

## References

- Popper, K. (1959). The Logic of Scientific Discovery
- `docs/specifications/pixel-by-pixel-demo-ptop-ttop.md` - Unified specification (Part 0: Epistemological Foundation)
- `tests/falsification_tests.rs` - Implementation
