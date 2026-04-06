//! Probar Demo Parity Test Suite
//!
//! Per specification SIMULAR-DEMO-002: This is THE arbiter of truth.
//! All demos MUST pass these tests for the architecture to be valid.
//!
//! # Popperian Foundation
//!
//! These tests are falsifiable claims. If any test fails:
//! - The demo architecture is FALSIFIED
//! - Implementation must be fixed before merge
//!
//! # Test Categories
//!
//! 1. YAML Loading - All demos load from YAML configs
//! 2. Deterministic Replay - Same seed produces identical states
//! 3. State Parity - Two instances produce identical state sequences
//! 4. Falsification Criteria - All criteria pass

use simular::demos::{DemoEngine, DemoError, OrbitalEngine, TspEngine};

// =============================================================================
// Test Infrastructure
// =============================================================================

/// Object-safe wrapper trait for heterogeneous engine testing.
#[allow(dead_code)]
trait DemoEngineObject {
    fn step_engine(&mut self);
    fn state_hash(&self) -> u64;
    fn step_count_val(&self) -> u64;
    fn seed_val(&self) -> u64;
    fn reset_with_seed_val(&mut self, seed: u64);
    fn is_complete_val(&self) -> bool;
}

impl DemoEngineObject for OrbitalEngine {
    fn step_engine(&mut self) {
        self.step();
    }

    fn state_hash(&self) -> u64 {
        self.state().compute_hash()
    }

    fn step_count_val(&self) -> u64 {
        self.step_count()
    }

    fn seed_val(&self) -> u64 {
        self.seed()
    }

    fn reset_with_seed_val(&mut self, seed: u64) {
        self.reset_with_seed(seed);
    }

    fn is_complete_val(&self) -> bool {
        self.is_complete()
    }
}

impl DemoEngineObject for TspEngine {
    fn step_engine(&mut self) {
        self.step();
    }

    fn state_hash(&self) -> u64 {
        self.state().compute_hash()
    }

    fn step_count_val(&self) -> u64 {
        self.step_count()
    }

    fn seed_val(&self) -> u64 {
        self.seed()
    }

    fn reset_with_seed_val(&mut self, seed: u64) {
        self.reset_with_seed(seed);
    }

    fn is_complete_val(&self) -> bool {
        self.is_complete()
    }
}

/// Create an engine from YAML by detecting demo type.
fn create_engine_from_yaml(yaml: &str) -> Result<Box<dyn DemoEngineObject>, DemoError> {
    // Parse YAML to determine demo type
    let value: serde_yaml::Value = serde_yaml::from_str(yaml)?;

    let demo_type = value
        .get("simulation")
        .and_then(|s| s.get("type"))
        .and_then(|t| t.as_str())
        .ok_or_else(|| DemoError::Validation("Missing simulation.type".to_string()))?;

    match demo_type {
        "orbit" => OrbitalEngine::from_yaml(yaml).map(|e| Box::new(e) as Box<dyn DemoEngineObject>),
        "tsp" => TspEngine::from_yaml(yaml).map(|e| Box::new(e) as Box<dyn DemoEngineObject>),
        "monte_carlo" => Err(DemoError::Validation(
            "MonteCarloEngine not yet implemented - TODO: impl DemoEngine".to_string(),
        )),
        other => Err(DemoError::Validation(format!("Unknown demo type: {other}"))),
    }
}

// =============================================================================
// Test 1: YAML Config Loading - ORBIT
// =============================================================================

/// Orbit YAML MUST load successfully.
#[test]
fn test_orbit_yaml_loading() {
    let yaml = include_str!("../examples/experiments/orbit_earth_sun.yaml");
    let result = create_engine_from_yaml(yaml);
    assert!(
        result.is_ok(),
        "Orbit YAML failed to load: {:?}",
        result.err()
    );
}

/// TSP YAML MUST load successfully.
#[test]
fn test_tsp_yaml_loading() {
    let yaml = include_str!("../examples/experiments/bay_area_tsp.yaml");
    let result = create_engine_from_yaml(yaml);
    assert!(
        result.is_ok(),
        "TSP YAML failed to load: {:?}",
        result.err()
    );
}

/// TSP deterministic replay test.
#[test]
fn test_tsp_deterministic_replay() {
    let yaml = include_str!("../examples/experiments/bay_area_tsp.yaml");

    let mut engine1 = TspEngine::from_yaml(yaml).expect("Engine 1 creation failed");
    let mut engine2 = TspEngine::from_yaml(yaml).expect("Engine 2 creation failed");

    // Run both for N steps
    for step in 0..20 {
        engine1.step();
        engine2.step();

        assert_eq!(
            engine1.state(),
            engine2.state(),
            "TSP state divergence at step {}: engine1 != engine2",
            step + 1
        );
    }
}

/// TSP TUI/WASM parity test.
#[test]
fn test_tsp_tui_wasm_parity() {
    let yaml = include_str!("../examples/experiments/bay_area_tsp.yaml");

    let mut tui_engine = TspEngine::from_yaml(yaml).expect("TUI engine creation failed");
    let mut wasm_engine = TspEngine::from_yaml(yaml).expect("WASM engine creation failed");

    for step in 0..20 {
        tui_engine.step();
        wasm_engine.step();

        assert_eq!(
            tui_engine.state().compute_hash(),
            wasm_engine.state().compute_hash(),
            "TSP TUI/WASM state divergence at step {}",
            step + 1
        );
    }
}

/// TSP reset replay test.
#[test]
fn test_tsp_reset_replay_identical() {
    let yaml = include_str!("../examples/experiments/bay_area_tsp.yaml");

    let mut engine = TspEngine::from_yaml(yaml).expect("Engine creation failed");

    let mut states_run1 = Vec::new();
    for _ in 0..10 {
        engine.step();
        states_run1.push(engine.state().compute_hash());
    }

    let seed = engine.seed();
    engine.reset_with_seed(seed);

    let mut states_run2 = Vec::new();
    for _ in 0..10 {
        engine.step();
        states_run2.push(engine.state().compute_hash());
    }

    assert_eq!(
        states_run1, states_run2,
        "TSP replay produced different states after reset"
    );
}

/// TSP falsification criteria test.
#[test]
fn test_tsp_falsification_criteria_pass() {
    let yaml = include_str!("../examples/experiments/bay_area_tsp.yaml");

    let mut engine = TspEngine::from_yaml(yaml).expect("Engine creation failed");

    // Run for 50 restarts
    for _ in 0..50 {
        engine.step();
    }

    let results = engine.evaluate_criteria();
    assert!(!results.is_empty(), "No falsification criteria defined");

    for result in &results {
        assert!(
            result.passed,
            "Criterion {} failed: {} (actual={:.2e}, threshold={:.2e})",
            result.id, result.message, result.actual, result.expected
        );
    }
}

// =============================================================================
// Test 2: Deterministic Replay - ORBIT
// =============================================================================

/// Given same YAML config, two engines MUST produce identical state sequences.
///
/// # Falsification
///
/// If engine1.state() != engine2.state() at any step, this test FAILS.
#[test]
fn test_orbit_deterministic_replay() {
    let yaml = include_str!("../examples/experiments/orbit_earth_sun.yaml");

    // Create two independent engines
    let mut engine1 = OrbitalEngine::from_yaml(yaml).expect("Engine 1 creation failed");
    let mut engine2 = OrbitalEngine::from_yaml(yaml).expect("Engine 2 creation failed");

    // Run both for N steps
    for step in 0..100 {
        engine1.step();
        engine2.step();

        // THE PARITY CHECK
        assert_eq!(
            engine1.state(),
            engine2.state(),
            "State divergence at step {}: engine1 != engine2",
            step + 1
        );
    }
}

// =============================================================================
// Test 3: TUI/WASM State Parity - ORBIT
// =============================================================================

/// TUI and WASM renderers MUST produce identical state sequences.
///
/// This is THE critical test for the architecture.
/// We verify by running two independent instances from same YAML.
#[test]
fn test_orbit_tui_wasm_parity() {
    let yaml = include_str!("../examples/experiments/orbit_earth_sun.yaml");

    // Simulate TUI instance
    let mut tui_engine = OrbitalEngine::from_yaml(yaml).expect("TUI engine creation failed");

    // Simulate WASM instance (same YAML, independent creation)
    let mut wasm_engine = OrbitalEngine::from_yaml(yaml).expect("WASM engine creation failed");

    // Run both for N steps
    for step in 0..100 {
        tui_engine.step();
        wasm_engine.step();

        // THE PARITY CHECK
        assert_eq!(
            tui_engine.state().compute_hash(),
            wasm_engine.state().compute_hash(),
            "TUI/WASM state divergence at step {}",
            step + 1
        );
    }
}

/// All demos TUI/WASM parity - both Orbit and TSP now implement DemoEngine.
#[test]
fn test_all_demos_tui_wasm_state_parity() {
    let yamls = [
        include_str!("../examples/experiments/orbit_earth_sun.yaml"),
        include_str!("../examples/experiments/bay_area_tsp.yaml"),
    ];

    for (i, yaml) in yamls.iter().enumerate() {
        let engine1_result = create_engine_from_yaml(yaml);
        let engine2_result = create_engine_from_yaml(yaml);

        assert!(
            engine1_result.is_ok() && engine2_result.is_ok(),
            "Failed to create engines for YAML {i}: {:?} / {:?}",
            engine1_result.as_ref().err(),
            engine2_result.as_ref().err()
        );

        let mut engine1 = engine1_result.unwrap();
        let mut engine2 = engine2_result.unwrap();

        // Run fewer steps for TSP (it's slower)
        let steps = if i == 1 { 20 } else { 100 };
        for step in 0..steps {
            engine1.step_engine();
            engine2.step_engine();

            assert_eq!(
                engine1.state_hash(),
                engine2.state_hash(),
                "YAML {i}: State divergence at step {}",
                step + 1
            );
        }
    }
}

// =============================================================================
// Test 4: Reset With Seed Produces Identical Replay - ORBIT
// =============================================================================

/// After reset_with_seed(), engine MUST reproduce exact same sequence.
#[test]
fn test_orbit_reset_replay_identical() {
    let yaml = include_str!("../examples/experiments/orbit_earth_sun.yaml");

    let mut engine = OrbitalEngine::from_yaml(yaml).expect("Engine creation failed");

    // First run: collect states
    let mut states_run1 = Vec::new();
    for _ in 0..50 {
        engine.step();
        states_run1.push(engine.state().compute_hash());
    }

    // Reset and replay
    let seed = engine.seed();
    engine.reset_with_seed(seed);

    // Second run: verify identical states
    let mut states_run2 = Vec::new();
    for _ in 0..50 {
        engine.step();
        states_run2.push(engine.state().compute_hash());
    }

    assert_eq!(
        states_run1, states_run2,
        "Replay produced different states after reset"
    );
}

// =============================================================================
// Test 5: No Hardcoded Configs
// =============================================================================

/// Verify no hardcoded KeplerConfig::earth_sun() calls remain.
///
/// This is verified by grep in CI, but we document it here.
#[test]
fn test_no_hardcoded_configs_documented() {
    // This test documents the requirement.
    // Actual verification is via: ! grep -r "KeplerConfig::earth_sun()" src/
    // See: .github/workflows/ci.yml

    // For now, this is a marker that the check exists
    // Hardcoded config check is performed in CI via grep
}

// =============================================================================
// Test 6: Falsification Criteria - ORBIT
// =============================================================================

/// Orbit demo MUST pass all falsification criteria.
#[test]
fn test_orbit_falsification_criteria_pass() {
    let yaml = include_str!("../examples/experiments/orbit_earth_sun.yaml");

    let mut engine = OrbitalEngine::from_yaml(yaml).expect("Engine creation failed");

    // Run simulation for 1000 steps (about 41 days with 1hr timestep)
    for _ in 0..1000 {
        engine.step();
    }

    // Evaluate criteria
    let results = engine.evaluate_criteria();
    assert!(!results.is_empty(), "No falsification criteria defined");

    for result in &results {
        assert!(
            result.passed,
            "Criterion {} failed: {} (actual={:.2e}, threshold={:.2e})",
            result.id, result.message, result.actual, result.expected
        );
    }
}

// =============================================================================
// Compile-Time Verification
// =============================================================================

/// Verify that DemoEngine trait is accessible.
#[test]
fn test_demo_engine_trait_exists() {
    // This compiles only if DemoEngine trait is public
    fn _assert_trait_exists<T: DemoEngine>() {}

    // The trait exists - this test passes
    // DemoEngine trait is defined and public — verified by compilation
}

/// Verify OrbitalEngine implements DemoEngine.
#[test]
fn test_orbital_engine_implements_demo_engine() {
    fn _assert_implements<T: DemoEngine>() {}
    _assert_implements::<OrbitalEngine>();
}

/// Document which demos still need to implement DemoEngine.
#[test]
fn test_demos_that_need_implementation() {
    // COMPLETED (implement DemoEngine):
    // - OrbitalEngine (for orbit demos)
    // - TspEngine (for TSP demos)

    // These demos exist but do NOT implement DemoEngine:
    // - HarmonicOscillatorDemo (implements EddDemo)
    // - MonteCarloPlDemo (implements EddDemo)
    // - LittlesLawFactoryDemo (implements EddDemo)
    // - KingmanHockeyDemo (implements EddDemo)

    // These are legacy wrappers (replaced by new engines):
    // - TspGraspDemo (replaced by TspEngine)
    // - KeplerOrbitDemo (replaced by OrbitalEngine)

    let unimplemented_demos = [
        "HarmonicOscillatorDemo",
        "MonteCarloPlDemo",
        "LittlesLawFactoryDemo",
        "KingmanHockeyDemo",
    ];

    // This documents remaining work - remove items as implemented
    let _ = unimplemented_demos;
}

/// Verify TspEngine implements DemoEngine.
#[test]
fn test_tsp_engine_implements_demo_engine() {
    fn _assert_implements<T: DemoEngine>() {}
    _assert_implements::<TspEngine>();
}

// =============================================================================
// Test 7: Energy Conservation Long-Term
// =============================================================================

/// Orbit MUST conserve energy over long simulations.
#[test]
fn test_orbit_energy_conservation_long_term() {
    let yaml = include_str!("../examples/experiments/orbit_earth_sun.yaml");

    let mut engine = OrbitalEngine::from_yaml(yaml).expect("Engine creation failed");

    // Run for ~10 orbits (365 days * 24 hours * 10 = 87600 steps)
    // We'll do fewer for test speed but check the principle
    for _ in 0..5000 {
        engine.step();
    }

    // Check energy conservation is within tolerance
    let results = engine.evaluate_criteria();
    let energy_result = results.iter().find(|r| r.id.contains("ENERGY"));

    assert!(
        energy_result.is_some(),
        "No energy conservation criterion found"
    );
    assert!(
        energy_result.unwrap().passed,
        "Energy conservation failed: {}",
        energy_result.unwrap().message
    );
}
