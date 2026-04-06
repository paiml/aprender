# iterator-v1

**Version:** 1.0.0

Generic iterator contract — common Rust API pattern

## References

- Rust API Guidelines: https://rust-lang.github.io/api-guidelines/

## Equations

### iterator

$$
iterator follows standard Rust conventions
$$

**Invariants:**

- $Type safety preserved$
- $No panics on valid input$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | iterator correctness |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-GEN-001 | iterator contract | Function follows iterator pattern | Implementation deviates from pattern |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-GEN-001 | iterator correctness | 8 | bounded_int |

## QA Gate

**iterator-v1 Contract** (F-IV-001)

Quality gate for Generic iterator contract — common Rust API pattern

**Checks:** validation, falsification

