# Equation-Driven Development (EDD)

## The Four Pillars of Scientific Simulation

Equation-Driven Development (EDD) is a methodology for building falsifiable, reproducible simulations grounded in peer-reviewed mathematics. Unlike traditional simulation development, EDD inverts the standard workflow: **equations first, code second**.

## The Problem with Traditional Simulation Development

Traditional simulation development often follows this pattern:

1. Write code that "seems right"
2. Tune parameters until output "looks reasonable"
3. Hope it matches reality

This approach leads to:
- **Black-box models** that can't be verified against theory
- **Overfitting** to historical data without theoretical justification
- **Non-reproducible results** due to implicit random states
- **Unfalsifiable claims** that resist scientific scrutiny

## The Four Pillars of EDD

EDD addresses these problems through four mandatory requirements:

### Pillar 1: Prove It (Governing Equation)

Every simulation must begin with a mathematically verified governing equation from peer-reviewed literature. No coding begins until the equation is documented.

```rust
// Little's Law: L = λW
let law = LittlesLaw::new();
assert_eq!(law.latex(), "L = \\lambda W");
assert!(!law.citation().authors.is_empty());
```

### Pillar 2: Fail It (TDD from Equations)

Tests are derived from analytical solutions **before** implementation. The equation provides the expected values; code must match them.

```rust
// Test from analytical solution: λ=5, W=2 → L=10
let result = law.evaluate(5.0, 2.0);
assert!((result - 10.0).abs() < 1e-10);
```

### Pillar 3: Seed It (Deterministic Reproducibility)

Every stochastic simulation requires an explicit random seed. No implicit `time(NULL)` or untracked random state.

```rust
// Explicit seed required - no exceptions
let spec = ExperimentSpec::builder()
    .name("Queueing Study")
    .seed(42)  // Mandatory
    .build()?;
```

### Pillar 4: Falsify It (Popperian Refutation)

Every model must specify falsification criteria - conditions under which the model would be rejected. We actively search for these conditions.

```rust
// Define what would disprove the model
sim.add_falsification_criterion(
    "energy_conservation",
    "|E(t) - E(0)| / E(0) < 1e-6",
    FalsificationAction::RejectModel,
);

// Actively search for violations
let result = sim.seek_falsification(&param_space);
```

## Why EDD Matters

EDD provides:

1. **Scientific Rigor**: Every simulation is grounded in verified mathematics
2. **Reproducibility**: Explicit seeds ensure identical results across runs
3. **Falsifiability**: Models can be scientifically tested and rejected
4. **Documentation**: The Equation Model Card ensures knowledge transfer
5. **Quality Assurance**: TDD from equations catches implementation bugs early

## Getting Started

The EDD module in simular provides:

- **Governing Equations**: Little's Law, Kingman's Formula, Square Root Law, Bullwhip Effect
- **Equation Model Cards**: Structured documentation for every equation
- **Experiment Specifications**: YAML-driven declarative experiments
- **Falsification Framework**: Active search for model-breaking conditions
- **TPS Validation**: Ten canonical test cases from Toyota Production System

Continue to the next chapter to learn about Operations Science equations.
