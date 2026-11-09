//! Unit tests for `tsp_engine` (moved out unchanged, #6025).

use super::*;

fn make_test_yaml() -> String {
    r#"
simulation:
  type: tsp
  name: "Test TSP"

meta:
  id: "TEST-TSP-001"
  version: "1.0.0"
  description: "Test TSP instance"

cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 0.0, lon: 1.0 }
  - id: 2
    name: "C"
    alias: "C"
    coords: { lat: 1.0, lon: 0.0 }
  - id: 3
    name: "D"
    alias: "D"
    coords: { lat: 1.0, lon: 1.0 }

matrix:
  - [0, 1, 1, 2]
  - [1, 0, 2, 1]
  - [1, 2, 0, 1]
  - [2, 1, 1, 0]

algorithm:
  method: "grasp"
  params:
    rcl_size: 2
    restarts: 10
    two_opt: true
    seed: 42
"#
    .to_string()
}

#[test]
fn test_from_yaml() {
    let yaml = make_test_yaml();
    let engine = TspEngine::from_yaml(&yaml);
    assert!(engine.is_ok(), "Failed to parse YAML: {:?}", engine.err());
}

#[test]
fn test_deterministic_state() {
    let yaml = make_test_yaml();
    let mut engine1 = TspEngine::from_yaml(&yaml).unwrap();
    let mut engine2 = TspEngine::from_yaml(&yaml).unwrap();

    for _ in 0..5 {
        engine1.step();
        engine2.step();
    }

    assert_eq!(
        engine1.state(),
        engine2.state(),
        "State divergence detected"
    );
}

#[test]
fn test_reset_replay() {
    let yaml = make_test_yaml();
    let mut engine = TspEngine::from_yaml(&yaml).unwrap();

    // Run 5 steps
    for _ in 0..5 {
        engine.step();
    }
    let state1 = engine.state();

    // Reset and replay
    engine.reset();
    for _ in 0..5 {
        engine.step();
    }
    let state2 = engine.state();

    assert_eq!(state1, state2, "Reset did not produce identical replay");
}

#[test]
fn test_valid_tour() {
    let yaml = make_test_yaml();
    let mut engine = TspEngine::from_yaml(&yaml).unwrap();

    // Run to completion
    while !engine.is_complete() {
        engine.step();
    }

    let results = engine.evaluate_criteria();
    assert!(!results.is_empty());
    assert!(results[0].passed, "Tour should be valid");
}

#[test]
fn test_tour_improves() {
    let yaml = make_test_yaml();
    let mut engine = TspEngine::from_yaml(&yaml).unwrap();

    let initial_length = engine.best_tour_length;

    // Run multiple steps
    for _ in 0..10 {
        engine.step();
    }

    // With GRASP + 2-opt, we should find the optimal tour
    // For this 4-city instance, optimal is 4 (square path)
    assert!(
        engine.best_tour_length <= initial_length,
        "Tour should improve or stay same"
    );
}

#[test]
fn test_step_count() {
    let yaml = make_test_yaml();
    let mut engine = TspEngine::from_yaml(&yaml).unwrap();

    assert_eq!(engine.step_count(), 0);
    engine.step();
    assert_eq!(engine.step_count(), 1);
}

#[test]
fn test_seed() {
    let yaml = make_test_yaml();
    let engine = TspEngine::from_yaml(&yaml).unwrap();
    assert_eq!(engine.seed(), 42);
}

#[test]
fn test_meta() {
    let yaml = make_test_yaml();
    let engine = TspEngine::from_yaml(&yaml).unwrap();
    assert_eq!(engine.meta().id, "TEST-TSP-001");
    assert_eq!(engine.meta().demo_type, "tsp");
}

#[test]
fn test_state_hash() {
    let yaml = make_test_yaml();
    let mut engine = TspEngine::from_yaml(&yaml).unwrap();
    engine.step();

    let state = engine.state();
    let hash1 = state.compute_hash();
    let hash2 = state.compute_hash();

    assert_eq!(hash1, hash2, "Hash should be deterministic");
}

#[test]
fn test_is_complete() {
    let yaml = make_test_yaml();
    let mut engine = TspEngine::from_yaml(&yaml).unwrap();

    // Run until complete
    while !engine.is_complete() {
        engine.step();
    }

    assert!(engine.is_complete());
}
