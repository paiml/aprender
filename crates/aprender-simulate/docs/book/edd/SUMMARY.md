# EDD Book Summary

## Equation-Driven Development Guide

This book covers the complete EDD (Equation-Driven Development) methodology for building falsifiable, reproducible simulations.

### Chapters

1. [Introduction](00-introduction.md) - The Four Pillars of Scientific Simulation
2. [Operations Science Equations](01-operations-equations.md) - Little's Law, Kingman's Formula, Square Root Law, Bullwhip Effect
3. [Equation Model Cards](02-equation-model-cards.md) - Mandatory Documentation for Every Simulation
4. [Experiment Specification](03-experiment-specification.md) - YAML-Driven Declarative Experiments
5. [Popperian Falsification](04-falsification.md) - Actively Searching for Model-Breaking Conditions
6. [TPS Validation](05-tps-validation.md) - Ten Canonical Test Cases for Operations Science

### Quick Reference

#### The Four Pillars

| Pillar | Principle | Key Artifact |
|--------|-----------|--------------|
| 1. Prove It | Every simulation begins with a governing equation | Equation Model Card |
| 2. Fail It | Tests derived from analytical solutions | Verification Tests |
| 3. Seed It | Explicit random seeds for reproducibility | Experiment Seed |
| 4. Falsify It | Active search for model-breaking conditions | Falsification Criteria |

#### Key Types

| Type | Purpose | Example |
|------|---------|---------|
| `LittlesLaw` | L = λW equation | `LittlesLaw::new().evaluate(5.0, 2.0)` |
| `EquationModelCard` | Documentation bridge | `EmcBuilder::new().name("...").build()` |
| `ExperimentSpec` | Declarative experiment | `ExperimentSpec::builder().seed(42).build()` |
| `FalsificationCriterion` | What would disprove model | `FalsificationCriterion::new("name", "cond", action)` |

#### Examples

```bash
# Operations science equations
cargo run --example edd_operations

# TPS test case validation
cargo run --example edd_tps_validation

# Equation Model Cards
cargo run --example edd_model_card

# Popperian falsification
cargo run --example edd_falsification

# YAML loading
cargo run --example edd_yaml_loader
```

### EDD Violation Codes

| Code | Description | Severity |
|------|-------------|----------|
| EDD-01 | Missing Equation Model Card | Critical |
| EDD-02 | Missing governing equation | Critical |
| EDD-03 | Missing analytical derivation | Major |
| EDD-04 | Missing verification tests | Major |
| EDD-05 | Missing explicit seed | Critical |
| EDD-06 | Missing falsification criteria | Major |
| EDD-07 | Verification test failed | Major |
| EDD-08 | Conservation law violated | Critical |
| EDD-09 | Cross-platform reproducibility failed | Critical |
| EDD-10 | TDD compliance not met | Major |
| **EDD-11** | **Z3 equation proof fails** | **Critical** |
| **EDD-12** | **Z3Provable trait not implemented** | **Critical** |
| **EDD-13** | **YAML-only configuration violated** | **Critical** |
| **EDD-14** | **Probar TUI verification fails** | **Critical** |
| **EDD-15** | **Probar WASM verification fails** | **Critical** |
| **EDD-16** | **Missing audit log for step** | **Critical** |
| **EDD-17** | **Equation evaluation not logged** | **Critical** |
| **EDD-18** | **Test case generation fails** | **Critical** |

### The Three Pillars of Provable Simulation

| Pillar | Requirements | Command |
|--------|--------------|---------|
| **Z3 Proofs** | EDD-11, EDD-12 | `cargo test --features z3-proofs` |
| **YAML Config** | EDD-05, EDD-13 | `simular validate *.yaml` |
| **Probar UX** | EDD-14, EDD-15 | `cargo test --features probar` |

### Reference Documentation

- [EDD Specification](../../specifications/EDD-equation-driven-development-spec.md)
- [EMC Schema (JSON)](../../schemas/emc-v1.0.json)
- [Experiment Schema (JSON)](../../schemas/experiment-v1.0.json)
- [Operations EMCs](../../emc/operations/)
