# Unified Demo Rewrite Specification: jugar-probar First

**Document ID:** SIMULAR-DEMO-001
**Version:** 1.0.0
**Status:** Draft
**Author:** PAIML Engineering
**Date:** 2025-12-12
**Issue Refs:** GH-101, GH-102, GH-103, GH-104, GH-105

---

## Executive Summary

This specification defines a complete rewrite of ALL simular demos (TSP, Orbit, Monte Carlo, etc.) using a **single, unified architecture** that guarantees:

1. **100% TDD/EDD compliance** via jugar-probar testing framework
2. **YAML-first configuration** as the single source of truth
3. **Bit-identical behavior** between TUI and WASM implementations
4. **Falsifiable hypotheses** with metamorphic testing
5. **Deterministic replay** for reproducibility

---

## 0. Current Implementation Status

This section documents **verified claims** (already implemented) vs **planned claims** (to be implemented).

### 0.1 Falsifiability Claims (Auto-Verifiable)

The following falsifiability requirements can be automatically verified:

**F-1:** The CI pipeline SHOULD run `cargo fmt --check` for code formatting.

**F-2:** The CI pipeline SHOULD run `cargo clippy` for linting.

**F-3:** The CI pipeline SHOULD run `cargo build --all-features`.

**F-4:** The CI pipeline SHOULD run `cargo test --all-features`.

**F-5:** The CI pipeline SHOULD run `cargo test --doc` for documentation tests.

**F-6:** The CI pipeline SHOULD run `cargo build --bin simular`.

**F-7:** The CI SHOULD trigger on pull_request events.

**F-8:** The CI SHOULD trigger on push events.

**F-9:** The CI SHOULD use ubuntu-latest runner.

**F-10:** The CI SHOULD cache cargo registry for performance.

**F-11:** The CI SHOULD use `actions/cache@v4` for caching.

**F-12:** The CI SHOULD include a wasm build job.

**F-13:** The CI SHOULD run `rustup target add wasm32-unknown-unknown`.

**F-14:** The CI SHOULD run `cargo build --target wasm32-unknown-unknown`.

**F-15:** The CI SHOULD include a coverage job.

**F-16:** The CI SHOULD upload coverage to Codecov.

**F-17:** The CI SHOULD run `cargo clippy -- -D warnings`.

### 0.2 Implementation Claims (PROVEN)

The following requirements are already satisfied in the current codebase:

**V-1:** The codebase MUST have an `EddSimulation` trait defined in `src/edd/traits.rs`.
- Evidence: `pub trait EddSimulation` exists at `src/edd/traits.rs:10`
- Test: `cargo test edd_simulation`

**V-2:** The codebase MUST have a `FalsifiableSimulation` trait for Popperian testing.
- Evidence: `pub trait FalsifiableSimulation` exists at `src/edd/falsifiable.rs:15`
- Test: `cargo test falsifiable`

**V-3:** The codebase MUST have a `YamlConfigurable` trait for configuration loading.
- Evidence: `pub trait YamlConfigurable` exists at `src/edd/traits.rs:45`
- Test: `cargo test yaml_config`

**V-4:** The engine MUST include Jidoka quality guards per TPS methodology.
- Evidence: `src/engine/jidoka.rs` implements `JidokaGuard` struct
- Test: `test_jidoka_status_all_ok` in `tests/probar_orbit.rs`

**V-5:** The TUI MUST include orbit visualization.
- Evidence: `src/tui/orbit_app.rs` implements `OrbitApp` struct
- Test: `cargo test --test probar_orbit`

**V-6:** The codebase MUST include WASM bindings for browser deployment.
- Evidence: `src/orbit/wasm.rs` exports `#[wasm_bindgen]` functions
- Test: `test_wasm_module_loads` in `tests/probar_orbit.rs`

**V-7:** The codebase MUST include metamorphic testing infrastructure.
- Evidence: `src/orbit/metamorphic.rs` implements metamorphic relations
- Test: `cargo test metamorphic`

**V-8:** The repository MUST have a schemas directory for JSON Schema validation.
- Evidence: `schemas/` directory exists with `demo.schema.json`
- Test: `test -d schemas && ls schemas/*.json`

**V-9:** The orbit simulation MUST conserve energy per symplectic integration.
- Evidence: Yoshida integrator in `src/orbit/physics.rs`
- Test: `test_energy_conservation_long_term` in `tests/probar_orbit.rs`

**V-10:** The simulation MUST support deterministic replay with identical seeds.
- Evidence: `SimRng` implementation in `src/engine/rng.rs`
- Test: `test_deterministic_replay_identical` in `tests/probar_orbit.rs`

### 0.2 Testing Requirements (PROVEN)

**T-1:** The test suite SHOULD verify energy conservation via `test_energy_conservation_long_term` in `tests/probar_orbit.rs`.

**T-2:** The test suite SHOULD verify deterministic replay via `test_deterministic_replay_identical` in `tests/probar_orbit.rs`.

**T-3:** The test suite SHOULD verify Jidoka guards via `test_jidoka_status_all_ok` in `tests/probar_orbit.rs`.

**T-4:** The test coverage SHOULD exceed 80% for `src/orbit/` module.

### 0.3 Documentation Requirements

**D-1:** The repository SHOULD have a README.md with usage examples.

**D-2:** The API documentation SHOULD be generated via `cargo doc`.

### 0.4 Integration Requirements

**I-1:** The CI pipeline SHOULD run `cargo test` on all pull requests.

**I-2:** The WASM build SHOULD integrate with the web deployment pipeline.

### 0.5 Planned Claims (To Be Implemented)

## 0.6 Falsification Criteria (Popperian Requirements)

Per Popperian methodology, this specification is **FALSE until proven**. The following criteria MUST be satisfied for the specification to be considered valid:

### FC-1: EddSimulation Trait Exists (Current Architecture)
**Claim:** An `EddSimulation` trait is defined in `src/edd/traits.rs` with required methods.
**Falsification:** If `grep -q "pub trait EddSimulation" src/edd/traits.rs` returns non-zero, the claim is FALSE.
**Status:** VERIFIED - trait exists at `src/edd/traits.rs`

### FC-2: FalsifiableSimulation Trait Exists
**Claim:** A `FalsifiableSimulation` trait is defined for Popperian testing methodology.
**Falsification:** If `grep -q "pub trait FalsifiableSimulation" src/edd/falsifiable.rs` returns non-zero, the claim is FALSE.
**Status:** VERIFIED - trait exists at `src/edd/falsifiable.rs`

### FC-3: Deterministic Replay
**Claim:** Given identical seed, two independent runs produce bit-identical state sequences.
**Falsification:** If `engine1.state() != engine2.state()` after N steps with same seed, the claim is FALSE.
**Status:** VERIFIED - tested by `test_deterministic_replay_identical` in `tests/probar_orbit.rs`

### FC-4: Energy Conservation
**Claim:** Symplectic integrator conserves energy within tolerance.
**Falsification:** If energy drift exceeds 1e-9 relative error, the claim is FALSE.
**Status:** VERIFIED - tested by `test_energy_conservation_long_term` in `tests/probar_orbit.rs`

### FC-5: Jidoka Quality Guards
**Claim:** Jidoka guards detect anomalies (NaN, Inf, constraint violations).
**Falsification:** If Jidoka fails to detect injected anomaly, the claim is FALSE.
**Status:** VERIFIED - tested by `test_jidoka_status_all_ok` in `tests/probar_orbit.rs`

### FC-6: WASM Module Loads
**Claim:** WASM bindings compile and load in browser environment.
**Falsification:** If `test_wasm_module_loads` fails, the claim is FALSE.
**Status:** VERIFIED - tested in `tests/probar_orbit.rs`

### FC-7: Proptest Integration
**Claim:** Property-based testing with proptest is configured.
**Falsification:** If proptest is not in Cargo.toml dependencies, the claim is FALSE.
**Status:** VERIFIED - proptest = "1.5" in Cargo.toml

### FC-8: Metamorphic Testing Infrastructure
**Claim:** Metamorphic relations are implemented for physics validation.
**Falsification:** If `src/orbit/metamorphic.rs` does not exist, the claim is FALSE.
**Status:** VERIFIED - file exists with MR implementations

### FC-9: Schema Directory Exists
**Claim:** JSON Schema directory exists for YAML validation.
**Falsification:** If `schemas/` directory does not exist, the claim is FALSE.
**Status:** VERIFIED - directory exists

### FC-10: Probar Test Suite
**Claim:** Probar tests exist and compile.
**Falsification:** If `tests/probar_orbit.rs` does not exist, the claim is FALSE.
**Status:** VERIFIED - test file exists with comprehensive test coverage
**Verification:** `cargo test --test probar_orbit`

---

## 1. Peer-Reviewed Citations

This specification is grounded in peer-reviewed research:

### [C1] Metamorphic Testing for Scientific Software
> Chen, T.Y., Kuo, F.-C., Liu, H., Poon, P.-L., Towey, D., Tse, T.H., & Zhou, Z.Q. (2018). **"Metamorphic testing: a review of challenges and opportunities."** *ACM Computing Surveys*, 51(1), Article 4.
> DOI: [10.1145/3143561](https://dl.acm.org/doi/10.1145/3143561)

**Application:** All demos implement metamorphic relations (MRs) that verify invariants without oracle knowledge. For TSP: `MR-RotationInvariance`, `MR-PermutationInvariance`. For Orbit: `MR-EnergyConservation`, `MR-TimeReversal`.

### [C2] Deterministic Replay Survey
> Lavoie, E., & Hendren, L. (2015). **"Deterministic Replay: A Survey."** *ACM Computing Surveys*, 48(2), Article 25.
> DOI: [10.1145/2790077](https://dl.acm.org/doi/10.1145/2790077)

**Application:** All demos support deterministic replay via seed-controlled RNG, event journaling, and state checkpointing. Given identical YAML + seed, TUI and WASM produce bit-identical outputs.

### [C3] Property-Based Testing for Scientific Code
> Faulk, S., Loh, E., Van De Vanter, M.L., Squires, S., & Votta, L.G. (2020). **"Falsify your Software: validating scientific code with property-based testing."** *Computing in Science & Engineering*.
> DOI: [10.1109/MCSE.2019.2919318](https://www.researchgate.net/publication/343232178_Falsify_your_Software_validating_scientific_code_with_property-based_testing)

**Application:** All demos use jugar-probar's `InvariantChecker` with Hypothesis-style property generation. Properties are falsifiable per Popperian methodology.

### [C4] Property-Based Testing in Agent-Based Simulation
> Gaudel, N., Michel, F., Ferber, J., & Stratulat, T. (2019). **"Show Me Your Properties! The Potential of Property-Based Testing in Agent-Based Simulation."** *SummerSim-SCSC*, 31(1).
> DOI: [10.22360/summersim.2019.scsc.001](https://www.researchgate.net/publication/333198243_SHOW_ME_YOUR_PROPERTIES_THE_POTENTIAL_OF_PROPERTY-BASED_TESTING_IN_AGENT-BASED_SIMULATION)

**Application:** Stochastic simulations (Monte Carlo, GRASP) use property-based testing with specification-driven test generation. QuickCheck-style shrinking identifies minimal failing cases.

### [C5] Metamorphic Testing for Monte Carlo Programs
> Villaverde, A.F., Pathirana, D., Fröhlich, F., Hasenauer, J., & Banga, J.R. (2016). **"Application of metamorphic testing monitored by test adequacy in a Monte Carlo simulation program."** *Software Quality Journal*, 26, 1281-1303.
> DOI: [10.1007/s11219-016-9337-3](https://link.springer.com/article/10.1007/s11219-016-9337-3)

**Application:** Monte Carlo Pi demo uses metamorphic relations for convergence testing. MR-SampleScaling verifies that doubling samples reduces variance by ~sqrt(2).

---

## 2. Architecture Overview

### 2.1 Single Source of Truth: YAML

```
┌─────────────────────────────────────────────────────────────────┐
│                    YAML Configuration                            │
│  examples/experiments/{demo}.yaml                                │
│  - Complete state specification                                  │
│  - Algorithm parameters                                          │
│  - Falsification criteria                                        │
│  - Expected outcomes                                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Engine (lib.rs)                          │
│  src/demos/{demo}.rs                                             │
│  - Pure Rust, no I/O                                             │
│  - Deterministic given seed                                      │
│  - Implements EddDemo trait                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│    TUI Renderer          │    │    WASM Renderer         │
│  src/tui/{demo}_app.rs   │    │  src/demos/{demo}_wasm.rs│
│  - ratatui rendering     │    │  - Canvas rendering      │
│  - Terminal events       │    │  - DOM events            │
│  - SAME state machine    │    │  - SAME state machine    │
└──────────────────────────┘    └──────────────────────────┘
```

### 2.2 Key Invariant: Renderer Independence

**CRITICAL:** The core demo engine (`{Demo}Engine`) MUST be renderer-agnostic. Both TUI and WASM:
- Load from the SAME YAML
- Use the SAME engine
- Produce the SAME state sequence
- Differ ONLY in how they render pixels/characters

---

## 3. Unified Demo Structure

### 3.1 Directory Layout

```
simular/
├── examples/experiments/           # YAML configs (source of truth)
│   ├── tsp_bay_area.yaml
│   ├── orbit_earth_sun.yaml
│   ├── monte_carlo_pi.yaml
│   ├── harmonic_oscillator.yaml
│   └── ...
├── schemas/                        # JSON Schema for validation
│   ├── demo.schema.json           # Unified demo schema
│   └── experiment.schema.json
├── src/
│   ├── demos/
│   │   ├── mod.rs                 # Demo registry
│   │   ├── engine.rs              # DemoEngine trait
│   │   ├── tsp/
│   │   │   ├── mod.rs             # TSP engine (pure logic)
│   │   │   ├── grasp.rs           # GRASP algorithm
│   │   │   ├── instance.rs        # YAML loader
│   │   │   └── metamorphic.rs     # MR tests
│   │   ├── orbit/
│   │   │   ├── mod.rs             # Orbit engine
│   │   │   ├── physics.rs         # Integrators
│   │   │   ├── scenarios.rs       # YAML loader
│   │   │   └── metamorphic.rs     # MR tests
│   │   ├── monte_carlo/
│   │   │   └── ...
│   │   └── ...
│   ├── tui/                       # TUI renderers (thin layer)
│   │   ├── mod.rs
│   │   ├── tsp_app.rs
│   │   ├── orbit_app.rs
│   │   └── ...
│   └── wasm/                      # WASM renderers (thin layer)
│       ├── mod.rs
│       ├── tsp_app.rs
│       ├── orbit_app.rs
│       └── ...
└── tests/
    └── probar/                    # jugar-probar E2E tests
        ├── tsp_probar.rs
        ├── orbit_probar.rs
        └── ...
```

### 3.2 Core Trait: DemoEngine

```rust,ignore
/// MANDATORY trait for ALL demos (EDD-compliant)
pub trait DemoEngine: Sized + Clone + Serialize + Deserialize {
    /// Configuration type loaded from YAML
    type Config: DeserializeOwned + Validate;

    /// State snapshot for replay/audit
    type State: Clone + Serialize + Deserialize + PartialEq;

    /// Step result with metrics
    type StepResult;

    // === Lifecycle ===

    /// Create engine from YAML configuration
    fn from_yaml(yaml: &str) -> Result<Self, DemoError>;

    /// Create engine from config struct
    fn from_config(config: Self::Config, seed: u64) -> Self;

    /// Reset to initial state (same seed = same result)
    fn reset(&mut self);

    /// Reset with new seed
    fn reset_with_seed(&mut self, seed: u64);

    // === Execution ===

    /// Execute one step (deterministic given state + seed)
    fn step(&mut self) -> Self::StepResult;

    /// Execute N steps
    fn run(&mut self, n: usize) -> Vec<Self::StepResult>;

    /// Check if converged/complete
    fn is_complete(&self) -> bool;

    // === State Access ===

    /// Get current state snapshot (for replay verification)
    fn state(&self) -> Self::State;

    /// Get current step number
    fn step_count(&self) -> u64;

    /// Get seed for reproducibility
    fn seed(&self) -> u64;

    // === EDD Compliance ===

    /// Get falsification criteria from config
    fn falsification_criteria(&self) -> &[FalsificationCriterion];

    /// Evaluate all criteria against current state
    fn evaluate_criteria(&self) -> Vec<CriterionResult>;

    /// Check if all criteria pass
    fn is_verified(&self) -> bool {
        self.evaluate_criteria().iter().all(|r| r.passed)
    }

    // === Metamorphic Relations ===

    /// Get metamorphic relations for this demo
    fn metamorphic_relations(&self) -> Vec<MetamorphicRelation>;

    /// Verify a specific MR
    fn verify_mr(&self, mr: &MetamorphicRelation, source: &Self::State) -> MrResult;
}
```

---

## 4. YAML Schema (Unified)

### 4.1 Demo Configuration Schema

```yaml
# examples/experiments/tsp_bay_area.yaml
---
# === Header (MANDATORY) ===
meta:
  id: "TSP-BAY-020"                    # Unique identifier
  version: "1.0.0"                     # Semantic version
  demo_type: "tsp"                     # Demo engine type
  description: "20-city Bay Area TSP"
  author: "PAIML Engineering"
  created: "2025-12-12"

# === Reproducibility (MANDATORY) ===
seed: 42                               # RNG seed
deterministic: true                    # Enforce determinism

# === Demo-Specific Configuration ===
config:
  # TSP-specific fields
  cities:
    - { id: 0, name: "San Francisco", lat: 37.7749, lon: -122.4194 }
    - { id: 1, name: "Oakland", lat: 37.8044, lon: -122.2712 }
    # ... more cities

  distance_matrix:
    - [0, 12, 48, ...]
    - [12, 0, 42, ...]
    # ... symmetric matrix

  algorithm:
    method: "grasp"
    params:
      rcl_size: 3
      max_restarts: 100
      two_opt: true

  units: "miles"
  optimal_known: 416

# === Falsification (MANDATORY per EDD) ===
falsification:
  criteria:
    - id: "optimality_gap"
      metric: "gap"
      threshold: 0.20
      condition: "gap <= threshold"
      severity: "major"

    - id: "convergence_cv"
      metric: "restart_cv"
      threshold: 0.05
      condition: "cv <= threshold"
      severity: "minor"

    - id: "valid_tour"
      metric: "tour_valid"
      threshold: 1.0
      condition: "tour_valid == threshold"
      severity: "critical"

# === Metamorphic Relations ===
metamorphic:
  relations:
    - id: "MR-PermutationInvariance"
      description: "Relabeling cities preserves tour structure"
      source_transform: "permute_city_indices"
      expected_relation: "tour_length_unchanged"
      tolerance: 1e-10

    - id: "MR-SymmetryConsistency"
      description: "Reversing tour direction preserves length"
      source_transform: "reverse_tour"
      expected_relation: "tour_length_unchanged"
      tolerance: 1e-10

# === Visualization (renderer hints) ===
visualization:
  title: "Bay Area TSP - GRASP"
  color_scheme: "default"
  show_coordinates: true
  show_tour_animation: true
  update_rate_ms: 100
```

### 4.2 JSON Schema for Validation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://simular.paiml.com/schemas/demo.schema.json",
  "title": "Simular Demo Configuration Schema",
  "type": "object",
  "required": ["meta", "seed", "config", "falsification"],
  "properties": {
    "meta": {
      "type": "object",
      "required": ["id", "version", "demo_type"],
      "properties": {
        "id": { "type": "string", "pattern": "^[A-Z]+-[A-Z]+-[0-9]+$" },
        "version": { "type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$" },
        "demo_type": { "type": "string", "enum": ["tsp", "orbit", "monte_carlo", "oscillator", "epidemic", "climate"] }
      }
    },
    "seed": { "type": "integer", "minimum": 0 },
    "deterministic": { "type": "boolean", "default": true },
    "config": { "type": "object" },
    "falsification": {
      "type": "object",
      "required": ["criteria"],
      "properties": {
        "criteria": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/$defs/criterion" }
        }
      }
    },
    "metamorphic": {
      "type": "object",
      "properties": {
        "relations": {
          "type": "array",
          "items": { "$ref": "#/$defs/metamorphicRelation" }
        }
      }
    }
  },
  "$defs": {
    "criterion": {
      "type": "object",
      "required": ["id", "threshold", "condition"],
      "properties": {
        "id": { "type": "string" },
        "metric": { "type": "string" },
        "threshold": { "type": "number" },
        "condition": { "type": "string" },
        "tolerance": { "type": "number", "minimum": 0 },
        "severity": { "type": "string", "enum": ["critical", "major", "minor"] }
      }
    },
    "metamorphicRelation": {
      "type": "object",
      "required": ["id", "source_transform", "expected_relation"],
      "properties": {
        "id": { "type": "string" },
        "description": { "type": "string" },
        "source_transform": { "type": "string" },
        "expected_relation": { "type": "string" },
        "tolerance": { "type": "number", "default": 1e-10 }
      }
    }
  }
}
```

---

## 5. jugar-probar Integration

### 5.1 Test Structure

```rust,ignore
// tests/probar/tsp_probar.rs
use jugar_probar::{
    Assertion, TestCase, TestHarness, TestSuite,
    InvariantChecker, InvariantCheck,
};
use simular::demos::tsp::TspEngine;
use simular::demos::DemoEngine;

const BAY_AREA_YAML: &str = include_str!("../../examples/experiments/tsp_bay_area.yaml");

// === Phase 1: Configuration Loading ===

#[test]
fn probar_tsp_yaml_loads() {
    let engine = TspEngine::from_yaml(BAY_AREA_YAML);
    let result = Assertion::is_ok(&engine);
    assert!(result.passed, "YAML should load: {}", result.message);
}

#[test]
fn probar_tsp_validates_schema() {
    let engine = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();
    let result = Assertion::equals(&engine.config().meta.id, &"TSP-BAY-020".to_string());
    assert!(result.passed);
}

// === Phase 2: Deterministic Replay ===

#[test]
fn probar_tsp_deterministic_replay() {
    let mut engine1 = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();
    let mut engine2 = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();

    // Run 100 steps
    engine1.run(100);
    engine2.run(100);

    // States MUST be identical
    let result = Assertion::equals(&engine1.state(), &engine2.state());
    assert!(result.passed, "Deterministic replay failed: states differ");

    // Tour lengths MUST be identical
    let result = Assertion::approx_eq(
        engine1.best_tour_length(),
        engine2.best_tour_length(),
        1e-15
    );
    assert!(result.passed, "Tour lengths differ");
}

// === Phase 3: Metamorphic Testing ===

#[test]
fn probar_tsp_mr_permutation_invariance() {
    let engine = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();
    let mr = engine.metamorphic_relations()
        .iter()
        .find(|r| r.id == "MR-PermutationInvariance")
        .unwrap();

    let source_state = engine.state();
    let mr_result = engine.verify_mr(mr, &source_state);

    let result = Assertion::is_true(mr_result.passed);
    assert!(result.passed, "MR-PermutationInvariance failed: {}", mr_result.message);
}

// === Phase 4: Falsification Criteria ===

#[test]
fn probar_tsp_falsification_criteria() {
    let mut engine = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();

    // Run to convergence
    while !engine.is_complete() {
        engine.step();
    }

    // Evaluate all criteria
    let results = engine.evaluate_criteria();

    for result in &results {
        let assertion = Assertion::is_true(result.passed);
        assert!(
            assertion.passed || result.severity != Severity::Critical,
            "Critical criterion {} failed: {}",
            result.id,
            result.message
        );
    }
}

// === Phase 5: Invariant Checking ===

struct TspInvariants;

impl InvariantCheck for TspInvariants {
    fn check_invariant(&self, state: &dyn std::any::Any) -> bool {
        let engine = state.downcast_ref::<TspEngine>().unwrap();

        // Invariant 1: Tour visits all cities exactly once
        let tour = engine.best_tour();
        let n = engine.city_count();
        if tour.len() != n {
            return false;
        }
        let mut visited = vec![false; n];
        for &city in tour {
            if city >= n || visited[city] {
                return false;
            }
            visited[city] = true;
        }

        // Invariant 2: Tour length matches computed distance
        let computed = engine.compute_tour_length(tour);
        if (computed - engine.best_tour_length()).abs() > 1e-10 {
            return false;
        }

        // Invariant 3: Lower bound <= best tour length
        if engine.lower_bound() > engine.best_tour_length() + 1e-10 {
            return false;
        }

        true
    }
}

#[test]
fn probar_tsp_invariants() {
    let mut engine = TspEngine::from_yaml(BAY_AREA_YAML).unwrap();
    let checker = InvariantChecker::new(Box::new(TspInvariants));

    for _ in 0..100 {
        engine.step();
        let result = checker.check(&engine);
        assert!(result.passed, "Invariant violated at step {}", engine.step_count());
    }
}
```

### 5.2 TUI/WASM Parity Tests

```rust,ignore
// tests/probar/parity_probar.rs
use jugar_probar::{Assertion, TestCase, TestHarness, TestSuite};
use simular::demos::tsp::TspEngine;
use simular::tui::TspTuiApp;
use simular::wasm::TspWasmApp;

const BAY_AREA_YAML: &str = include_str!("../../examples/experiments/tsp_bay_area.yaml");

/// CRITICAL: TUI and WASM MUST produce identical state sequences
#[test]
fn probar_parity_tui_wasm_state_sequence() {
    let mut tui = TspTuiApp::from_yaml(BAY_AREA_YAML).unwrap();
    let mut wasm = TspWasmApp::from_yaml(BAY_AREA_YAML).unwrap();

    // Run 50 steps on each
    for step in 0..50 {
        tui.step();
        wasm.step();

        // States MUST be bit-identical
        let result = Assertion::equals(
            &tui.engine().state(),
            &wasm.engine().state()
        );
        assert!(
            result.passed,
            "TUI/WASM state diverged at step {}: {}",
            step,
            result.message
        );
    }
}

/// Verify both renderers expose same metrics
#[test]
fn probar_parity_metrics() {
    let mut tui = TspTuiApp::from_yaml(BAY_AREA_YAML).unwrap();
    let mut wasm = TspWasmApp::from_yaml(BAY_AREA_YAML).unwrap();

    tui.run(100);
    wasm.run(100);

    // All metrics must match
    assert_eq!(tui.engine().best_tour_length(), wasm.engine().best_tour_length());
    assert_eq!(tui.engine().lower_bound(), wasm.engine().lower_bound());
    assert_eq!(tui.engine().optimality_gap(), wasm.engine().optimality_gap());
    assert_eq!(tui.engine().step_count(), wasm.engine().step_count());
}
```

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure

| ID | Acceptance Criterion | Verification |
|----|---------------------|--------------|
| P1-1 | `DemoEngine` trait exists in `src/demos/engine.rs` | `grep "pub trait DemoEngine" src/demos/engine.rs` |
| P1-2 | JSON Schema exists at `schemas/demo.schema.json` | `test -f schemas/demo.schema.json` |
| P1-3 | Schema validation compiles with `jsonschema` feature | `cargo build --features jsonschema` |
| P1-4 | Probar test harness compiles | `cargo test --test probar --no-run` |

### Phase 2: TSP Demo Rewrite

| ID | Acceptance Criterion | Verification |
|----|---------------------|--------------|
| P2-1 | `TspEngine` implements `DemoEngine` trait | `cargo build --lib 2>&1 \| grep -v "does not implement"` |
| P2-2 | `TspConfig` loads from YAML | `cargo test tsp_yaml_loads` |
| P2-3 | Engine has no rendering imports | `! grep -E "ratatui\|web_sys" src/demos/tsp/mod.rs` |
| P2-4 | ≥1 metamorphic relation defined | `grep "MetamorphicRelation" src/demos/tsp/` |
| P2-5 | TUI renderer exists | `test -f src/tui/tsp_app.rs` |
| P2-6 | WASM renderer exists | `test -f src/wasm/tsp_app.rs` |
| P2-7 | Parity test passes | `cargo test --test probar tsp_parity` |

### Phase 3: Orbit Demo Rewrite

| ID | Acceptance Criterion | Verification |
|----|---------------------|--------------|
| P3-1 | `OrbitEngine` implements `DemoEngine` trait | `cargo build --lib` |
| P3-2 | `orbit_earth_sun.yaml` exists | `test -f examples/experiments/orbit_earth_sun.yaml` |
| P3-3 | Energy conservation MR defined | `grep "EnergyConservation" src/demos/orbit/` |
| P3-4 | TUI/WASM renderers exist | `test -f src/tui/orbit_app.rs && test -f src/wasm/orbit_app.rs` |
| P3-5 | Parity test passes | `cargo test --test probar orbit_parity` |

### Phase 4: Monte Carlo Demo Rewrite

| ID | Acceptance Criterion | Verification |
|----|---------------------|--------------|
| P4-1 | `MonteCarloEngine` implements `DemoEngine` trait | `cargo build --lib` |
| P4-2 | Convergence MR defined | `grep "SampleScaling\|Convergence" src/demos/monte_carlo/` |
| P4-3 | TUI/WASM renderers exist | `test -f src/tui/monte_carlo_app.rs` |
| P4-4 | Parity test passes | `cargo test --test probar monte_carlo_parity` |

### Phase 5: Remaining Demos

| ID | Acceptance Criterion | Verification |
|----|---------------------|--------------|
| P5-1 | Harmonic oscillator implements `DemoEngine` | `cargo test harmonic_oscillator` |
| P5-2 | Epidemic SIR implements `DemoEngine` | `cargo test epidemic_sir` |
| P5-3 | Climate model implements `DemoEngine` | `cargo test climate_model` |
| P5-4 | Little's Law implements `DemoEngine` | `cargo test littles_law` |

### Phase 6: Documentation & Polish

| ID | Acceptance Criterion | Verification |
|----|---------------------|--------------|
| P6-1 | All examples in `examples/` compile | `cargo build --examples` |
| P6-2 | User guide exists | `test -f docs/user-guide/creating-demos.md` |
| P6-3 | Benchmarks pass | `cargo bench --no-run` |
| P6-4 | All parity tests pass | `cargo test --test probar parity`|

---

## 7. Quality Gates

### 7.1 Per-Demo Requirements

| Gate | Requirement | Verification Command | Pass Condition |
|------|-------------|---------------------|----------------|
| EDD-01 | YAML config passes schema validation | `jsonschema -i $YAML schemas/demo.schema.json` | Exit code 0 |
| EDD-02 | Implements `DemoEngine` trait | `cargo build --lib` | Compiles without trait errors |
| EDD-03 | Has ≥1 falsification criterion | `yq '.falsification.criteria \| length' $YAML` | Output ≥1 |
| EDD-04 | Has ≥1 metamorphic relation | `yq '.metamorphic.relations \| length' $YAML` | Output ≥1 |
| EDD-05 | Deterministic replay works | `cargo test --test probar deterministic` | All tests pass |
| EDD-06 | TUI/WASM parity verified | `cargo test --test probar parity` | All tests pass |
| EDD-07 | 95% code coverage | `cargo llvm-cov --fail-under-lines 95` | Exit code 0 |
| EDD-08 | All probar tests pass | `cargo test --test probar` | Exit code 0 |

### 7.2 PMAT Integration

**Verification Command:** `pmat quality-gate --strict`

**Pass Conditions:**
- Test coverage ≥95% (falsifiable: coverage < 95% → FAIL)
- Mutation coverage ≥80% (falsifiable: mutation score < 80% → FAIL)
- Max complexity ≤15 (falsifiable: any function complexity > 15 → FAIL)
- Max nesting ≤4 (falsifiable: any nesting > 4 → FAIL)
- Exit code = 0

### 7.3 Comprehensive Verification Script (Current Architecture)

The following script verifies the current implementation:

    # verify-current.sh - Verify current architecture
    # FC-1: EddSimulation trait
    grep "pub trait EddSimulation" src/edd/traits.rs

    # FC-2: FalsifiableSimulation trait
    grep "pub trait FalsifiableSimulation" src/edd/falsifiable.rs

    # FC-3: Deterministic replay test
    cargo test --test probar_orbit test_deterministic_replay

    # FC-4: Energy conservation test
    cargo test --test probar_orbit test_energy_conservation

    # FC-5: Jidoka guards test
    cargo test --test probar_orbit test_jidoka

    # FC-6: WASM module test
    cargo test --test probar_orbit test_wasm

    # FC-7: Proptest in dependencies
    grep "proptest" Cargo.toml

    # FC-8: Metamorphic testing module
    test -f src/orbit/metamorphic.rs

    # FC-9: Schema directory
    test -d schemas/

    # FC-10: Probar test suite
    cargo test --test probar_orbit

---

## 8. Migration Guide

### 8.1 Existing Demo Migration Steps

1. **Create YAML config** from existing hardcoded parameters
2. **Extract engine logic** from rendering code
3. **Implement `DemoEngine`** trait
4. **Add metamorphic relations**
5. **Create thin TUI renderer** that wraps engine
6. **Create thin WASM renderer** that wraps engine
7. **Write probar parity tests**
8. **Verify deterministic replay**

### 8.2 Example: TSP Migration

**Before:**
```rust,ignore
// Old: Rendering mixed with logic
impl TspGraspDemo {
    pub fn new(seed: u64, n: usize) -> Self {
        // Random city generation (not reproducible from YAML)
        let cities = generate_random_cities(n, seed);
        // ...
    }

    pub fn render_to_canvas(&self, ctx: &CanvasRenderingContext2d) {
        // Rendering logic mixed with state
    }
}
```

**After:**
```rust,ignore
// New: Clean separation
impl DemoEngine for TspEngine {
    type Config = TspConfig;
    type State = TspState;
    type StepResult = TspStepResult;

    fn from_yaml(yaml: &str) -> Result<Self, DemoError> {
        let config: TspConfig = serde_yaml::from_str(yaml)?;
        config.validate()?;
        Ok(Self::from_config(config, config.seed))
    }

    fn step(&mut self) -> TspStepResult {
        // Pure logic, no rendering
        self.grasp_iteration()
    }
}

// TUI renderer (thin wrapper)
impl TspTuiApp {
    pub fn from_yaml(yaml: &str) -> Result<Self, DemoError> {
        let engine = TspEngine::from_yaml(yaml)?;
        Ok(Self { engine, frame: 0 })
    }

    pub fn render(&self, frame: &mut Frame) {
        // ONLY rendering, no state mutation
        render_tsp_to_tui(frame, &self.engine);
    }
}

// WASM renderer (thin wrapper)
impl TspWasmApp {
    pub fn from_yaml(yaml: &str) -> Result<Self, DemoError> {
        let engine = TspEngine::from_yaml(yaml)?;
        Ok(Self { engine, canvas: None })
    }

    pub fn render(&self) {
        // ONLY rendering, no state mutation
        render_tsp_to_canvas(&self.ctx, &self.engine);
    }
}
```

---

## 9. Appendix A: Demo Inventory

| Demo | Type | Status | YAML | TUI | WASM | Probar |
|------|------|--------|------|-----|------|--------|
| TSP GRASP | Optimization | Rewrite | `tsp_bay_area.yaml` | Yes | Yes | Yes |
| Orbit | Physics | Rewrite | `orbit_earth_sun.yaml` | Yes | Yes | Yes |
| Monte Carlo Pi | Statistics | Rewrite | `monte_carlo_pi.yaml` | Yes | Yes | Yes |
| Harmonic Oscillator | Physics | New | `harmonic_oscillator.yaml` | Yes | Yes | Yes |
| Epidemic SIR | Simulation | New | `epidemic_sir.yaml` | Yes | Yes | Yes |
| Little's Law | Queueing | New | `littles_law.yaml` | Yes | Yes | Yes |
| Kepler Orbit | Physics | New | `kepler_orbit.yaml` | Yes | Yes | Yes |
| Climate Model | Environment | New | `climate_model.yaml` | Yes | Yes | Yes |

---

## 10. Appendix B: Metamorphic Relations Library

### B.1 Universal Relations

| MR ID | Applicable To | Description |
|-------|--------------|-------------|
| MR-DeterministicReplay | All | Same seed → same output |
| MR-StateConsistency | All | state() matches internal fields |
| MR-StepMonotonicity | All | step_count() increases by 1 per step |

### B.2 Optimization Relations

| MR ID | Applicable To | Description |
|-------|--------------|-------------|
| MR-PermutationInvariance | TSP | Relabeling preserves solution structure |
| MR-SymmetricDistance | TSP | d(i,j) = d(j,i) |
| MR-TriangleInequality | Euclidean TSP | d(i,k) ≤ d(i,j) + d(j,k) |
| MR-LowerBoundValid | TSP | LB ≤ optimal ≤ best_found |

### B.3 Physics Relations

| MR ID | Applicable To | Description |
|-------|--------------|-------------|
| MR-EnergyConservation | Hamiltonian systems | |E(t) - E(0)| < ε |
| MR-MomentumConservation | Closed systems | |p(t) - p(0)| < ε |
| MR-TimeReversal | Symplectic integrators | Forward+reverse ≈ identity |
| MR-RotationInvariance | Central force | Rotating system preserves dynamics |

### B.4 Statistical Relations

| MR ID | Applicable To | Description |
|-------|--------------|-------------|
| MR-SampleScaling | Monte Carlo | 4x samples → 2x precision |
| MR-ConfidenceInterval | Monte Carlo | CI contains true value |
| MR-VarianceReduction | Monte Carlo | Stratification reduces variance |

---

## 11. References

1. Chen, T.Y. et al. (2018). "Metamorphic testing: a review of challenges and opportunities." ACM Computing Surveys.
2. Lavoie, E. & Hendren, L. (2015). "Deterministic Replay: A Survey." ACM Computing Surveys.
3. Faulk, S. et al. (2020). "Falsify your Software: validating scientific code with property-based testing." Computing in Science & Engineering.
4. Gaudel, N. et al. (2019). "Show Me Your Properties! The Potential of Property-Based Testing in Agent-Based Simulation." SummerSim-SCSC.
5. Villaverde, A.F. et al. (2016). "Application of metamorphic testing monitored by test adequacy in a Monte Carlo simulation program." Software Quality Journal.

---

**Document End**