//! Generic WASM Renderer for `DemoEngine` Implementations
//!
//! Per specification SIMULAR-DEMO-002: This renderer is engine-agnostic.
//! Any `DemoEngine` implementation can be rendered through this interface.
//!
//! # Architecture
//!
//! WASM bindings cannot use generic types directly with wasm-bindgen.
//! This module provides:
//! 1. `WasmRunner<E>` - internal runner that wraps any `DemoEngine`
//! 2. JSON serialization for cross-boundary state transfer
//! 3. Common operations (step, pause, reset, get state)
//!
//! Concrete WASM exports (e.g., `WasmOrbitDemo`) wrap `WasmRunner` internally.

use crate::demos::{CriterionResult, DemoEngine, DemoMeta};
use serde::Serialize;

/// Generic WASM runner that wraps any `DemoEngine`.
///
/// This is used internally by concrete WASM exports.
/// It cannot be exported via wasm-bindgen directly due to generic type limitations.
#[derive(Debug)]
pub struct WasmRunner<E: DemoEngine> {
    engine: E,
    running: bool,
    paused: bool,
    step_count: u64,
}

impl<E: DemoEngine> WasmRunner<E> {
    /// Create a new WASM runner for the given engine.
    #[must_use]
    pub fn new(engine: E) -> Self {
        Self {
            engine,
            running: true,
            paused: false,
            step_count: 0,
        }
    }

    /// Create from YAML configuration.
    ///
    /// # Errors
    ///
    /// Returns error if YAML parsing fails.
    pub fn from_yaml(yaml: &str) -> Result<Self, crate::demos::DemoError> {
        let engine = E::from_yaml(yaml)?;
        Ok(Self::new(engine))
    }

    /// Get reference to the engine.
    #[must_use]
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Get mutable reference to the engine.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// Check if runner is running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running && !self.engine.is_complete()
    }

    /// Check if runner is paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Toggle pause state.
    pub fn toggle_pause(&mut self) -> bool {
        self.paused = !self.paused;
        self.paused
    }

    /// Set paused state directly.
    pub fn set_paused(&mut self, paused: bool) {
        self.paused = paused;
    }

    /// Stop the runner.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Resume running.
    pub fn resume(&mut self) {
        self.paused = false;
    }

    /// Reset the engine.
    pub fn reset(&mut self) {
        self.engine.reset();
        self.step_count = 0;
        self.paused = false;
    }

    /// Advance the simulation by one step.
    ///
    /// Returns false if paused or complete.
    pub fn step(&mut self) -> bool {
        if self.paused || self.engine.is_complete() {
            return false;
        }
        self.engine.step();
        self.step_count += 1;
        true
    }

    /// Run multiple steps.
    ///
    /// Returns the number of steps completed.
    pub fn run_steps(&mut self, num_steps: u32) -> u32 {
        let mut completed = 0;
        for _ in 0..num_steps {
            if !self.step() {
                break;
            }
            completed += 1;
        }
        completed
    }

    /// Get demo metadata.
    #[must_use]
    pub fn meta(&self) -> &DemoMeta {
        self.engine.meta()
    }

    /// Get current state.
    #[must_use]
    pub fn state(&self) -> E::State {
        self.engine.state()
    }

    /// Evaluate falsification criteria.
    #[must_use]
    pub fn evaluate_criteria(&self) -> Vec<CriterionResult> {
        self.engine.evaluate_criteria()
    }

    /// Get step count.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Get seed for reproducibility.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.engine.seed()
    }

    /// Check if complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.engine.is_complete()
    }
}

/// JSON-serializable state for cross-boundary transfer.
///
/// Used to send state from Rust to JavaScript.
#[derive(Debug, Clone, Serialize)]
pub struct WasmState {
    /// Demo ID.
    pub id: String,
    /// Demo type (e.g., "orbit", "tsp").
    pub demo_type: String,
    /// Current step number.
    pub step: u64,
    /// Seed for reproducibility.
    pub seed: u64,
    /// Whether demo is paused.
    pub paused: bool,
    /// Whether demo is complete.
    pub complete: bool,
    /// Engine-specific state as JSON.
    pub state_json: String,
    /// Falsification criteria results.
    pub criteria: Vec<CriterionResultJson>,
}

/// JSON-serializable criterion result.
#[derive(Debug, Clone, Serialize)]
pub struct CriterionResultJson {
    /// Criterion ID.
    pub id: String,
    /// Whether criterion passed.
    pub passed: bool,
    /// Actual value.
    pub actual: f64,
    /// Expected threshold.
    pub expected: f64,
    /// Human-readable message.
    pub message: String,
    /// Severity level.
    pub severity: String,
}

impl From<&CriterionResult> for CriterionResultJson {
    fn from(result: &CriterionResult) -> Self {
        Self {
            id: result.id.clone(),
            passed: result.passed,
            actual: result.actual,
            expected: result.expected,
            message: result.message.clone(),
            severity: format!("{:?}", result.severity),
        }
    }
}

impl<E: DemoEngine> WasmRunner<E>
where
    E::State: Serialize,
{
    /// Get state as JSON string for JavaScript consumption.
    #[must_use]
    pub fn state_json(&self) -> String {
        serde_json::to_string(&self.engine.state()).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get full WASM state for JavaScript.
    #[must_use]
    pub fn wasm_state(&self) -> WasmState {
        let meta = self.engine.meta();
        let criteria = self
            .engine
            .evaluate_criteria()
            .iter()
            .map(CriterionResultJson::from)
            .collect();

        WasmState {
            id: meta.id.clone(),
            demo_type: meta.demo_type.clone(),
            step: self.step_count,
            seed: self.engine.seed(),
            paused: self.paused,
            complete: self.engine.is_complete(),
            state_json: self.state_json(),
            criteria,
        }
    }

    /// Get full state as JSON string.
    #[must_use]
    pub fn wasm_state_json(&self) -> String {
        serde_json::to_string(&self.wasm_state()).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demos::OrbitalEngine;

    const TEST_YAML: &str = r#"
simulation:
  type: orbit
  name: "Test Orbit"

meta:
  id: "TEST-001"
  version: "1.0.0"
  demo_type: orbit

reproducibility:
  seed: 42

scenario:
  type: kepler
  central_body:
    name: "Sun"
    mass_kg: 1.989e30
    position: [0.0, 0.0, 0.0]
  orbiter:
    name: "Earth"
    mass_kg: 5.972e24
    semi_major_axis_m: 1.496e11
    eccentricity: 0.0167

integrator:
  type: stormer_verlet
  dt_seconds: 3600.0
"#;

    #[test]
    fn test_wasm_runner_creation() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let runner = WasmRunner::new(engine);

        assert!(runner.is_running());
        assert!(!runner.is_paused());
        assert_eq!(runner.step_count(), 0);
    }

    #[test]
    fn test_wasm_runner_from_yaml() {
        let runner = WasmRunner::<OrbitalEngine>::from_yaml(TEST_YAML).unwrap();
        assert_eq!(runner.meta().id, "TEST-001");
    }

    #[test]
    fn test_wasm_runner_step() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut runner = WasmRunner::new(engine);

        assert!(runner.step());
        assert_eq!(runner.step_count(), 1);

        assert!(runner.step());
        assert_eq!(runner.step_count(), 2);
    }

    #[test]
    fn test_wasm_runner_run_steps() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut runner = WasmRunner::new(engine);

        let completed = runner.run_steps(10);
        assert_eq!(completed, 10);
        assert_eq!(runner.step_count(), 10);
    }

    #[test]
    fn test_wasm_runner_pause() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut runner = WasmRunner::new(engine);

        assert!(!runner.is_paused());
        runner.toggle_pause();
        assert!(runner.is_paused());

        // Step should fail when paused
        assert!(!runner.step());
        assert_eq!(runner.step_count(), 0);

        runner.resume();
        assert!(!runner.is_paused());
        assert!(runner.step());
    }

    #[test]
    fn test_wasm_runner_reset() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut runner = WasmRunner::new(engine);

        runner.run_steps(5);
        assert_eq!(runner.step_count(), 5);

        runner.reset();
        assert_eq!(runner.step_count(), 0);
        assert!(!runner.is_paused());
    }

    #[test]
    fn test_wasm_runner_state_json() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let runner = WasmRunner::new(engine);

        let json = runner.state_json();
        assert!(json.contains("position"));
        assert!(json.contains("velocity"));
    }

    #[test]
    fn test_wasm_runner_wasm_state() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut runner = WasmRunner::new(engine);

        runner.step();
        let state = runner.wasm_state();

        assert_eq!(state.id, "TEST-001");
        assert_eq!(state.demo_type, "orbit");
        assert_eq!(state.step, 1);
        assert_eq!(state.seed, 42);
        assert!(!state.paused);
        assert!(!state.complete);
        assert!(!state.criteria.is_empty());
    }

    #[test]
    fn test_wasm_runner_wasm_state_json() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let runner = WasmRunner::new(engine);

        let json = runner.wasm_state_json();
        assert!(json.contains("TEST-001"));
        assert!(json.contains("orbit"));
    }

    #[test]
    fn test_wasm_runner_stop() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut runner = WasmRunner::new(engine);

        assert!(runner.is_running());
        runner.stop();
        assert!(!runner.is_running());
    }

    #[test]
    fn test_wasm_runner_set_paused() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut runner = WasmRunner::new(engine);

        runner.set_paused(true);
        assert!(runner.is_paused());
        runner.set_paused(false);
        assert!(!runner.is_paused());
    }

    #[test]
    fn test_wasm_runner_criteria() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let runner = WasmRunner::new(engine);

        let criteria = runner.evaluate_criteria();
        assert!(!criteria.is_empty());
    }

    #[test]
    fn test_wasm_runner_seed() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let runner = WasmRunner::new(engine);

        assert_eq!(runner.seed(), 42);
    }

    #[test]
    fn test_wasm_runner_is_complete() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let runner = WasmRunner::new(engine);

        assert!(!runner.is_complete());
    }
}
