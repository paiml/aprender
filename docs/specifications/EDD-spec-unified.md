# Equation-Driven Development (EDD) Unified Specification

## simular: YAML-Only Falsifiable Simulation Framework

**Version:** 2.0.0
**Status:** Active
**Authors:** PAIML Engineering
**Date:** 2025-12-12

---

## Abstract

This specification defines **Equation-Driven Development (EDD) v2**, a simplified, rigorous methodology for simulation development. EDD v2 enforces:

1. **YAML-Only Simulations**: Zero JavaScript/HTML/user-defined custom code - all simulations defined declaratively.
2. **probar-First Testing**: Unified TUI/WASM testing with replayable, shareable simulations.
3. **95% Mutation Coverage**: Hard requirement, no exceptions.
4. **Popperian Falsification**: Every simulation must be falsifiable.
5. **Equation Model Cards**: Mandatory mathematical documentation.

> **Core Principle**: If it can't be expressed in YAML and proven with probar, it doesn't ship.

---

## Table of Contents

1. [The YAML-Only Mandate](#1-the-yaml-only-mandate)
2. [The probar Testing Pyramid](#2-the-probar-testing-pyramid)
3. [Equation Model Cards (EMC)](#3-equation-model-cards-emc)
4. [Popperian Falsification](#4-popperian-falsification)
5. [Quality Gates](#5-quality-gates)
6. [Implementation Guide](#6-implementation-guide)

---

## 1. The YAML-Only Mandate

### 1.1 Zero Custom Code Policy

**HARD REQUIREMENT**: All simulations MUST be defined entirely in YAML. No exceptions.

| Allowed | Prohibited |
|---------|------------|
| YAML experiment specs | JavaScript |
| YAML EMC definitions | HTML |
| YAML configuration | Custom rendering code |
| Core simular engine | User-defined code |

```yaml
# ALLOWED: Pure YAML simulation definition
experiment:
  id: "TSP-GRASP-001"
  seed: 42
  emc_ref: "optimization/tsp_grasp"

  simulation:
    type: "tsp_grasp"
    parameters:
      n_cities: 25
      rcl_size: 5
      max_iterations: 100

  falsification:
    criteria:
      - id: "optimality_gap"
        threshold: 0.25
        condition: "gap < threshold"
```

```javascript
// PROHIBITED: Any JavaScript
const simulation = new TspSimulation(); // ❌ NO
```

```html
<!-- PROHIBITED: Any HTML -->
<canvas id="sim"></canvas> <!-- ❌ NO -->
```

### 1.2 The simular Engine Contract

All simulation logic lives in the core `simular` crate. If functionality is needed:

1. **Request via GitHub Issue** with YAML interface proposal
2. **Core team implements** in Rust
3. **Expose via YAML schema** only
4. **Test via probar** before release

```
User Need → YAML Proposal → Core Implementation → probar Verification → Release
```

### 1.3 YAML Schema Enforcement

Every YAML file MUST validate against the authoritative simular schema defined in `schemas/experiment.schema.json`.

```bash
# Validation is MANDATORY before any simulation runs
simular validate experiment.yaml
```

---

## 2. The probar Testing Pyramid

### 2.1 probar-First Development

**probar is the foundation of all testing.** The testing pyramid is:

```
                    ┌─────────────┐
                    │   WASM/TUI  │  ← Output (shareable)
                    │   Replay    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Equations  │  ← Verification
                    │   (EMC)     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    YAML     │  ← Configuration
                    │   Specs     │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │         probar          │  ← FOUNDATION
              │  (Unified TUI/WASM)     │
              └─────────────────────────┘
```

### 2.2 probar Capabilities

probar provides unified testing for TUI and WASM with replayable simulations:

| Feature | Description |
|---------|-------------|
| **Unified Testing** | Same tests run on TUI and WASM |
| **Replayable Simulations** | Capture and replay any simulation |
| **Shareable Output** | Export as WASM bundle, TUI session, or .mp4 |
| **Visual Verification** | Assert on rendered output |
| **Equation Display** | Verify equation cards render correctly |
| **Headless Execution** | Run simulations without UI for CI/CD |

### 2.3 probar Test Structure

```rust
// All TUI/WASM tests use probar
#[probar::test] async fn test_tsp_simulation_renders_correctly() {
    // Load from YAML (the ONLY input source)
    let sim = probar::load_yaml("experiments/tsp_grasp.yaml");

    // Run simulation
    let session = probar::run(sim).await;

    // Assert equation card is displayed
    probar::assert_contains!(session.frame(), "L(π) = Σ d(πᵢ, πᵢ₊₁)");

    // Assert falsification criteria shown
    probar::assert_contains!(session.frame(), "Gap < 25%");

    // Export as shareable replay
    session.export_wasm("tsp_demo.wasm");
    session.export_mp4("tsp_demo.mp4");
}
```

### 2.4 Replayable Simulation Format

Every simulation produces a replay file:

```yaml
# .replay.yaml - Shareable simulation state
replay:
  version: "1.0"
  seed: 42
  experiment_ref: "experiments/tsp_grasp.yaml"

  timeline:
    - step: 0
      state: { tour_length: 1234.5, best: 1234.5 }
      equations:
        - id: "tour_length"
          value: 1234.5

    - step: 1
      state: { tour_length: 1198.2, best: 1198.2 }
      equations:
        - id: "two_opt_delta"
          value: 36.3

  outputs:
    wasm: "tsp_demo.wasm"
    mp4: "tsp_demo.mp4"
    tui_session: "tsp_demo.session"
```

### 2.5 Shareable Formats

| Format | Use Case | Command |
|--------|----------|---------|
| `.wasm` | Interactive web demo | `probar export --wasm` |
| `.mp4` | Video documentation | `probar export --mp4` |
| `.session` | TUI replay | `probar export --tui` |
| `.replay.yaml` | Full state capture | `probar export --replay` |

---

## 3. Equation Model Cards (EMC)

### 3.1 EMC Purpose

Every simulation MUST have an Equation Model Card documenting:

1. **Governing Equations** - Mathematical foundation
2. **Domain of Validity** - Where equations apply
3. **Analytical Solutions** - Test cases with known answers
4. **Falsification Criteria** - How to prove the model wrong

### 3.2 EMC Schema (YAML)

```yaml
# docs/emc/tsp_grasp.emc.yaml
emc_version: "1.0"
emc_id: "optimization/tsp_grasp"

identity:
  name: "TSP GRASP with 2-Opt"
  version: "1.0.0"

governing_equation:
  latex: "L(\pi) = \sum_{i=1}^{n} d(\pi_i, \pi_{i+1})"
  plain_text: "L(π) = Σ d(πᵢ, πᵢ₊₁)"
  description: "Tour length is sum of edge distances"

analytical_solution:
  test_cases:
    - name: "Square 4-city"
      input:
        cities: [[0,0], [1,0], [1,1], [0,1]]
      expected: 4.0  # Optimal tour length

domain_of_validity:
  parameters:
    n_cities:
      min: 3
      max: 1000
  assumptions:
    - "Euclidean distance metric"
    - "Complete graph"

falsification:
  criteria:
    - id: "optimality_gap"
      condition: "gap <= 0.25"
      description: "Solution within 25% of lower bound"
```

### 3.3 EMC Display in Simulations

Every simulation MUST display its EMC equation card:

```
┌─────────────────────────────────────────────────┐
│ EMC: optimization/tsp_grasp v1.0.0              │
├─────────────────────────────────────────────────┤
│ Equation: L(π) = Σ d(πᵢ, πᵢ₊₁)                 │
│                                                 │
│ Current: L = 1198.2                             │
│ Best:    L = 1145.7                             │
│ Gap:     18.3% (< 25% ✓)                        │
└─────────────────────────────────────────────────┘
```

---

## 4. Popperian Falsification

### 4.1 Falsifiability Requirement

**Every simulation MUST be falsifiable.** This means:

1. **Explicit Criteria**: Define what would prove the model wrong
2. **Measurable Thresholds**: Quantitative bounds
3. **Active Testing**: Seek conditions that break the model

```yaml
# Falsification is MANDATORY in every experiment
falsification:
  criteria:
    - id: "energy_conservation"
      metric: "energy_drift"
      threshold: 1e-9
      condition: "drift < threshold"
      severity: "critical"

    - id: "convergence_rate"
      metric: "error_slope"
      threshold: -0.5
      tolerance: 0.1
      condition: "abs(slope - (-0.5)) < tolerance"
      severity: "major"
```

### 4.2 Falsification in probar

```rust
#[probar::test] async fn test_falsification_criteria() {
    let sim = probar::load_yaml("experiments/orbit.yaml");
    let result = probar::run_to_completion(sim).await;

    // probar automatically checks falsification criteria
    assert!(result.falsification_passed());

    // Individual criteria accessible
    assert!(result.criterion("energy_conservation").passed);
    assert!(result.criterion("convergence_rate").passed);
}
```

### 4.3 Null Hypothesis Testing

Every experiment specifies a null hypothesis to reject:

```yaml
hypothesis:
  null_hypothesis: |
    H₀: The GRASP algorithm produces solutions no better than
    random tour construction.
  expected_outcome: "reject"  # We expect to FALSIFY H₀

  statistical_test:
    type: "paired_t_test"
    alpha: 0.05
    power: 0.80
```

---

## 5. Quality Gates

### 5.1 Hard Requirements

| Requirement | Threshold | Enforcement |
|-------------|-----------|-------------|
| **Mutation Coverage** | 95% | CI blocking |
| **Test Coverage** | 95% | CI blocking |
| **YAML-Only** | 100% | Compile-time |
| **probar Tests** | All pass | CI blocking |
| **EMC Present** | Required | Schema validation |
| **Falsification Defined** | Required | Schema validation |
| **Max Complexity** | 15 | CI blocking |
| **Max Nesting** | 4 | CI blocking |
| **Zero SATD** | No TODO/FIXME (Self-Admitted Technical Debt) | CI blocking |

### 5.2 The 95% Mutation Requirement

**HARD REQUIREMENT**: 95% mutation coverage with cargo-mutants.

```bash
# This MUST pass before any merge
cargo mutants --timeout 300

# Minimum surviving mutants
Mutants tested: 1000
Killed: 950 (95%)
Survived: 50 (5%)  # Maximum allowed
```

### 5.3 CI Pipeline

```yaml
# .github/workflows/edd.yaml
name: EDD v2 Compliance

on: [push, pull_request]

jobs:
  yaml-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate all YAML experiments
        run: simular validate examples/experiments/*.yaml
      - name: Validate all EMCs
        run: simular emc-check docs/emc/*.emc.yaml

  probar-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run probar TUI tests
        run: cargo test --features probar
      - name: Run probar WASM tests
        run: wasm-pack test --headless --chrome

  mutation-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install cargo-mutants
        run: cargo install cargo-mutants
      - name: Run mutation testing (95% required)
        run: |
          cargo mutants --timeout 300 --json > mutants.json
          MUTANTS_KILLED=$(jq '.killed' mutants.json)
          MUTANTS_TOTAL=$(jq '.total' mutants.json)
          if [ "$MUTANTS_TOTAL" -eq 0 ]; then
            echo "No mutants generated. Check configuration."
            exit 1
          fi
          RATIO=$(echo "scale=2; $MUTANTS_KILLED / $MUTANTS_TOTAL" | bc)
          echo "Mutation Coverage: $RATIO (Killed: $MUTANTS_KILLED, Total: $MUTANTS_TOTAL)"
          
          if (( $(echo "$RATIO < 0.95" | bc -l) )); then
            echo "FAILED: Mutation coverage $RATIO < 0.95"
            exit 1
          fi

  falsification:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run all experiments with falsification checks
        run: simular run examples/experiments/*.yaml --verify-falsification
```

### 5.4 Quality Grades

| Grade | Score | Decision |
|-------|-------|----------|
| **A+ (Toyota Standard)** | 95-100% | Release OK |
| **A (Kaizen Target)** | 90-94% | Release with minor improvements |
| **B+ (Acceptable)** | 85-89% | Beta only |
| **STOP THE LINE** | <85% or any hard requirement fail | Block release |

---

## 6. Implementation Guide

### 6.1 Creating a New Simulation

1. **Directory Structure Compliance**
   - Experiments: `examples/experiments/*.yaml`
   - EMCs: `docs/emc/*.emc.yaml`
   - Tests: `tests/probar_*.rs`

2. **Write YAML Experiment Spec**
```yaml
# examples/experiments/my_simulation.yaml
experiment:
  id: "MY-SIM-001"
  seed: 42
  emc_ref: "domain/my_equation"
  # ... full spec
```

3. **Write EMC**
```yaml
# docs/emc/my_equation.emc.yaml
emc_version: "1.0"
emc_id: "domain/my_equation"
governing_equation:
  latex: "..."
  # ... full EMC
```

4. **Write probar Tests**
```rust
#[probar::test] async fn test_my_simulation() {
    let sim = probar::load_yaml("examples/experiments/my_simulation.yaml");
    // ... assertions
}
```

5. **If Core Changes Needed**
   - Open GitHub Issue with YAML interface proposal
   - Core team reviews and implements
   - NO custom code in simulations

### 6.2 Running Simulations

```bash
# Validate YAML
simular validate examples/experiments/my_simulation.yaml

# Run simulation
simular run examples/experiments/my_simulation.yaml

# Export replay
simular run examples/experiments/my_simulation.yaml --export-replay

# Generate shareable outputs
probar export --wasm --mp4 examples/experiments/my_simulation.yaml
```

### 6.3 Verification Checklist

Before any simulation ships:

- [ ] YAML-only (no custom code)
- [ ] EMC exists and validates
- [ ] probar tests pass (TUI + WASM)
- [ ] Falsification criteria defined and tested
- [ ] 95% mutation coverage
- [ ] 95% test coverage
- [ ] Replay exports work (WASM, TUI, MP4)
- [ ] Equation card displays correctly
- [ ] Zero SATD (No TODOs/FIXMEs)

---

## Appendix A: Migration from EDD v1

### Removed Features

- JavaScript/HTML simulation code
- Custom rendering
- Non-YAML configuration
- Z3 formal proofs (moved to optional)

### New Requirements

- probar-first testing (was optional)
- 95% mutation coverage (was 80%)
- Shareable replays (new)
- YAML-only mandate (stricter)

### Migration Steps

1. Convert any custom code to YAML + core engine requests
2. Add probar tests for all simulations
3. Export existing simulations as replay files
4. Verify 95% mutation coverage

---

## References

1. Popper, K. R. (2002). *The Logic of Scientific Discovery*. Routledge.
2. Ohno, T. (1988). *Toyota Production System*. Productivity Press.
3. Hopp, W. J. & Spearman, M. L. (2004). *Factory Physics*. Waveland Press.
4. probar documentation: `../probar/README.md`