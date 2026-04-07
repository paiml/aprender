# Demo Architecture v2: Probar-First Specification

**Document ID:** SIMULAR-DEMO-002
**Version:** 2.0.0
**Status:** Draft
**Author:** PAIML Engineering
**Date:** 2025-12-13

---

## 0. Popperian Foundation

This specification is **FALSE until PROVEN**. Every claim must be falsifiable via automated test.

### 0.1 Falsification Protocol

**Claim is FALSE if any test fails:**

```bash
# All falsification tests live in tests/probar_demos.rs
cargo test --test probar_demos
```

---

## 1. Root Cause Analysis Summary

Previous architecture failed because:

1. No probar-first testing enforcement
2. YAML configs exist but demos don't load them
3. No unified DemoEngine trait
4. TUI/WASM have separate state wrappers
5. No state parity verification

---

## 2. Architecture Principles

### 2.1 The Inversion

**OLD (Failed):** Implementation-first, tests added later
```
Rust Code → TUI Wrapper → WASM Wrapper → Maybe Tests
```

**NEW (Probar-First):** Tests define the contract
```
YAML Config → Probar Test Suite → DemoEngine Trait → Implementations
```

### 2.2 The One Rule

> **If it can't be tested for TUI/WASM parity, it doesn't belong in the demo.**

---

## 3. Core Architecture

### 3.1 File Structure

```
src/
├── demos/
│   ├── engine.rs          # DemoEngine trait (THE contract)
│   ├── state.rs           # DemoState trait (serializable snapshots)
│   ├── orbit/
│   │   ├── mod.rs         # OrbitalEngine: impl DemoEngine
│   │   └── physics.rs     # Pure physics (no rendering)
│   ├── tsp/
│   │   ├── mod.rs         # TspEngine: impl DemoEngine
│   │   └── grasp.rs       # Pure GRASP algorithm
│   └── monte_carlo/
│       └── mod.rs         # MonteCarloEngine: impl DemoEngine
├── renderers/
│   ├── tui.rs             # Generic TUI renderer
│   └── wasm.rs            # Generic WASM renderer
tests/
└── probar_demos.rs        # THE parity test suite
schemas/
└── demo.schema.json       # YAML validation schema
examples/experiments/
├── orbit_earth_sun.yaml   # MUST be loaded by OrbitalEngine
├── tsp_bay_area.yaml      # MUST be loaded by TspEngine
└── monte_carlo_pi.yaml    # MUST be loaded by MonteCarloEngine
```

### 3.2 The DemoEngine Trait

```rust
/// THE contract that ALL demos MUST implement.
///
/// # Falsifiability
///
/// This trait is verified by `tests/probar_demos.rs`.
/// If a demo doesn't implement this trait correctly,
/// the probar tests WILL fail.
pub trait DemoEngine: Sized + Clone {
    /// Configuration loaded from YAML.
    type Config: serde::de::DeserializeOwned + serde::Serialize;

    /// Serializable state snapshot for parity verification.
    type State: Clone + PartialEq + serde::Serialize + serde::de::DeserializeOwned;

    /// Create engine from YAML string.
    ///
    /// # Errors
    /// Returns error if YAML is invalid or fails validation.
    fn from_yaml(yaml: &str) -> Result<Self, DemoError>;

    /// Get deterministic seed.
    fn seed(&self) -> u64;

    /// Reset with new seed.
    fn reset_with_seed(&mut self, seed: u64);

    /// Execute one deterministic step.
    /// Given same state + seed, MUST produce same result.
    fn step(&mut self);

    /// Get current state snapshot.
    /// This is what probar compares between TUI and WASM.
    fn state(&self) -> Self::State;

    /// Get step count.
    fn step_count(&self) -> u64;

    /// Check if demo is complete/converged.
    fn is_complete(&self) -> bool;

    /// Get falsification criteria results.
    fn evaluate_criteria(&self) -> Vec<CriterionResult>;
}
```

### 3.3 The State Contract

```rust
/// State snapshot requirements for parity testing.
///
/// # Key Invariant
///
/// For any demo, given:
/// - Same YAML config
/// - Same seed
/// - Same number of steps
///
/// Then: tui_engine.state() == wasm_engine.state()
///
/// This is THE falsifiable claim tested by probar.
pub trait DemoState: Clone + PartialEq + serde::Serialize {
    /// Compute hash for quick comparison.
    fn state_hash(&self) -> u64;
}
```

---

## 4. Probar Test Suite

### 4.1 The Parity Test (THE Critical Test)

```rust
// tests/probar_demos.rs

/// This test MUST pass for the architecture to be valid.
///
/// # Falsification
///
/// If TUI and WASM produce different states for the same
/// YAML config and seed, this test FAILS and the architecture
/// is FALSIFIED.
#[test]
fn test_tui_wasm_state_parity() {
    let yamls = [
        include_str!("../examples/experiments/orbit_earth_sun.yaml"),
        include_str!("../examples/experiments/tsp_bay_area.yaml"),
        include_str!("../examples/experiments/monte_carlo_pi.yaml"),
    ];

    for yaml in yamls {
        // Create TWO independent engines from same YAML
        let mut engine1 = create_engine_from_yaml(yaml).unwrap();
        let mut engine2 = create_engine_from_yaml(yaml).unwrap();

        // Run both for N steps
        for _ in 0..100 {
            engine1.step();
            engine2.step();

            // THE PARITY CHECK
            assert_eq!(
                engine1.state(),
                engine2.state(),
                "State divergence at step {}",
                engine1.step_count()
            );
        }
    }
}
```

### 4.2 Deterministic Replay Test

```rust
#[test]
fn test_deterministic_replay() {
    let yaml = include_str!("../examples/experiments/orbit_earth_sun.yaml");

    // First run
    let mut engine = OrbitalEngine::from_yaml(yaml).unwrap();
    let mut states1 = Vec::new();
    for _ in 0..50 {
        engine.step();
        states1.push(engine.state());
    }

    // Reset and replay
    engine.reset_with_seed(engine.seed());
    let mut states2 = Vec::new();
    for _ in 0..50 {
        engine.step();
        states2.push(engine.state());
    }

    // MUST be identical
    assert_eq!(states1, states2, "Replay produced different states");
}
```

### 4.3 YAML Loading Test

```rust
#[test]
fn test_yaml_config_loading() {
    // All YAML files in examples/experiments/ MUST load
    let yamls = std::fs::read_dir("examples/experiments/")
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension() == Some("yaml".as_ref()));

    for entry in yamls {
        let content = std::fs::read_to_string(entry.path()).unwrap();
        let result = create_engine_from_yaml(&content);
        assert!(
            result.is_ok(),
            "Failed to load {}: {:?}",
            entry.path().display(),
            result.err()
        );
    }
}
```

---

## 5. YAML Configuration Requirements

### 5.1 Orbit YAML (Currently Missing!)

```yaml
# examples/experiments/orbit_earth_sun.yaml
# THIS FILE MUST BE CREATED AND LOADED BY OrbitalEngine

simulation:
  type: orbit
  name: "Earth-Sun Two-Body System"

reproducibility:
  seed: 42
  ieee_strict: true

scenario:
  type: kepler
  central_body:
    mass_kg: 1.989e30  # Solar mass
    position: [0.0, 0.0, 0.0]
  orbiter:
    mass_kg: 5.972e24  # Earth mass
    semi_major_axis_m: 1.496e11  # 1 AU
    eccentricity: 0.0167

integrator:
  type: yoshida4
  dt_seconds: 3600.0  # 1 hour

jidoka:
  energy_tolerance: 1e-9
  angular_momentum_tolerance: 1e-9

falsification:
  criteria:
    - name: energy_conservation
      threshold: 1e-9
      type: relative_error
    - name: angular_momentum_conservation
      threshold: 1e-9
      type: relative_error
```

### 5.2 Schema Validation

All YAML MUST validate against `schemas/demo.schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["simulation", "reproducibility", "falsification"],
  "properties": {
    "simulation": {
      "type": "object",
      "required": ["type", "name"]
    },
    "reproducibility": {
      "type": "object",
      "required": ["seed"]
    },
    "falsification": {
      "type": "object",
      "required": ["criteria"],
      "properties": {
        "criteria": {
          "type": "array",
          "minItems": 1
        }
      }
    }
  }
}
```

---

## 6. Implementation Checklist

### Phase 1: Foundation

| Task | Verification | Owner |
|------|--------------|-------|
| Create `src/demos/engine.rs` with DemoEngine trait | `cargo build --lib` | |
| Create `tests/probar_demos.rs` with parity test | `cargo test --test probar_demos --no-run` | |
| Create `schemas/demo.schema.json` | `test -f schemas/demo.schema.json` | |

### Phase 2: Orbit Demo Migration

| Task | Verification | Owner |
|------|--------------|-------|
| Create `examples/experiments/orbit_earth_sun.yaml` | Schema validates | |
| Implement `OrbitalEngine: impl DemoEngine` | `cargo test orbital_engine` | |
| Migrate physics to load from YAML | Remove hardcoded `KeplerConfig::earth_sun()` | |
| Probar parity test passes | `cargo test --test probar_demos test_tui_wasm_state_parity` | |

### Phase 3: TSP Demo Alignment

| Task | Verification | Owner |
|------|--------------|-------|
| Implement `TspEngine: impl DemoEngine` | `cargo test tsp_engine` | |
| Verify YAML loading (already works) | `cargo test tsp_yaml` | |
| Probar parity test passes | `cargo test --test probar_demos` | |

### Phase 4: Generic Renderers

| Task | Verification | Owner |
|------|--------------|-------|
| Create `src/renderers/tui.rs` - generic TUI renderer | Compiles | |
| Create `src/renderers/wasm.rs` - generic WASM renderer | Compiles | |
| Delete old separate wrappers | `OrbitApp`, `TspApp`, `OrbitSimulation` removed | |

---

## 7. Acceptance Criteria

The architecture is **PROVEN** when ALL of the following pass:

```bash
# 1. All demos load from YAML
cargo test --test probar_demos test_yaml_config_loading

# 2. All demos produce deterministic replay
cargo test --test probar_demos test_deterministic_replay

# 3. TUI/WASM state parity (THE critical test)
cargo test --test probar_demos test_tui_wasm_state_parity

# 4. Falsification criteria pass
cargo test --test probar_demos test_falsification_criteria

# 5. No hardcoded configs remain
! grep -r "KeplerConfig::earth_sun()" src/
```

---

## 8. What Changes

### Removed
- `src/tui/orbit_app.rs` - replaced by generic renderer
- `src/orbit/wasm.rs` - replaced by generic renderer
- `src/demos/tsp_wasm_app.rs` - replaced by generic renderer
- Hardcoded `KeplerConfig::earth_sun()` calls

### Added
- `src/demos/engine.rs` - DemoEngine trait
- `src/demos/orbit/mod.rs` - OrbitalEngine: impl DemoEngine
- `src/renderers/tui.rs` - generic TUI renderer
- `src/renderers/wasm.rs` - generic WASM renderer
- `tests/probar_demos.rs` - parity test suite
- `examples/experiments/orbit_earth_sun.yaml` - orbit config

### Unchanged
- `src/orbit/physics.rs` - pure physics code
- `src/demos/tsp_grasp.rs` - pure algorithm code
- `examples/experiments/bay_area_tsp.yaml` - already exists

---

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| YAML configs per demo | 1+ | `ls examples/experiments/*.yaml` |
| Hardcoded configs | 0 | `grep "::earth_sun()" src/` |
| Probar tests | 100% pass | `cargo test --test probar_demos` |
| DemoEngine implementations | All demos | `grep "impl DemoEngine" src/` |
| State parity errors | 0 | Probar test results |

---

## 10. Falsification Statement

This specification will be **FALSIFIED** if:

1. Any demo does not load its config from YAML
2. Any demo does not implement DemoEngine trait
3. TUI and WASM produce different state sequences
4. Deterministic replay produces different results
5. Any falsification criterion fails

The probar test suite (`tests/probar_demos.rs`) is the arbiter of truth.
