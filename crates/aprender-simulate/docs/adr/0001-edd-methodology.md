# ADR-001: Equation-Driven Development Methodology

## Status

Accepted

## Context

Simulation software requires mathematical correctness that traditional TDD cannot guarantee. Standard unit tests verify behavior but not mathematical validity. We need a methodology that:

1. Documents the governing equations explicitly
2. Ensures implementations match equations
3. Provides falsifiable criteria for validation
4. Supports reproducible verification

## Decision

We adopt Equation-Driven Development (EDD), a 5-phase methodology:

### Phase 1: Equation
Document the governing equation with explicit mathematical notation.

```rust
/// Governing Equation: E = ½mv² + mgh
/// Conservation: dE/dt = 0 (closed system)
```

### Phase 2: Failing Test
Write a test that fails without the correct implementation.

```rust
#[test]
fn phase2_energy_not_conserved_without_symplectic() {
    // Euler method loses energy
    assert!(energy_drift > THRESHOLD);
}
```

### Phase 3: Implementation
Implement the algorithm to pass the test.

### Phase 4: Verification
Verify against known solutions or analytical results.

```rust
#[test]
fn phase4_verify_against_analytical() {
    let computed = simulate(params);
    let analytical = exact_solution(params);
    assert!((computed - analytical).abs() < TOLERANCE);
}
```

### Phase 5: Falsification
Document conditions under which the model fails.

```rust
#[test]
fn phase5_falsify_with_extreme_params() {
    // Model breaks down at relativistic speeds
    assert!(!model_valid_at(0.9 * SPEED_OF_LIGHT));
}
```

## Consequences

### Positive
- Mathematical correctness is documented and tested
- Equations are traceable to code
- Falsifiability enables scientific rigor
- Reproducibility is built into the process

### Negative
- More verbose test structure
- Requires domain expertise to write equations
- Initial development time increases

### Neutral
- Teams must learn the 5-phase structure
- Documentation becomes part of the code
