# builder-pattern-v1

**Version:** 1.0.0

Generic builder-pattern contract — common Rust API pattern

## References

- Rust API Guidelines: https://rust-lang.github.io/api-guidelines/

## Equations

### builder_pattern

```
Builder::new().field(v).build() -> Result<T, BuilderError>
```

**Domain:** $Typestate builder with required and optional fields$

**Invariants:**

- $build() returns Err if required fields are unset$
- $Builder is consumed on build() — no reuse after build$
- $Partial builder is valid — only build() checks completeness$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | builder-pattern correctness |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-GEN-001 | builder-pattern contract | Function follows builder-pattern pattern | Implementation deviates from pattern |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-GEN-001 | builder-pattern correctness | 8 | bounded_int |

## QA Gate

**builder-pattern-v1 Contract** (F-BPV-001)

Quality gate for Generic builder-pattern contract — common Rust API pattern

**Checks:** validation, falsification

