//! Turn-by-Turn Audit Logging for EDD Simulations.
//!
//! **HARD REQUIREMENT (EDD-16, EDD-17, EDD-18):** Every simulation MUST produce
//! a complete audit trail of every step.
//!
//! This module provides:
//! - `SimulationAuditLog` trait for mandatory audit logging
//! - `StepEntry` for capturing complete step state
//! - `EquationEval` for logging equation computations
//! - `Decision` for logging algorithmic choices
//! - Automatic test case generation from logs
//! - Replay functionality with speed control
//!
//! # The Provability Chain
//!
//! ```text
//! Seed(42) → RNG State₀ → Decision₁ → RNG State₁ → Decision₂ → ... → Final
//!     ↓           ↓            ↓           ↓            ↓
//!   KNOWN     PROVABLE     PROVABLE    PROVABLE     PROVABLE
//! ```
//!
//! Every step is deterministic given the seed, therefore every step is provable.
//!
//! # References
//!
//! - EDD Spec Section 1.6: Turn-by-Turn Audit Logging (MANDATORY)
//! - Quality Gates: EDD-16, EDD-17, EDD-18

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::engine::SimTime;
use crate::error::{SimError, SimResult};

// =============================================================================
// Core Audit Types (EDD-16)
// =============================================================================

/// Equation evaluation record (EDD-17).
///
/// Every equation evaluation MUST be logged with inputs, result, and optional
/// expected value for verification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EquationEval {
    /// Equation identifier from EMC (e.g., `two_opt_delta`, `tour_length`)
    pub equation_id: String,

    /// Input values with variable names
    pub inputs: IndexMap<String, f64>,

    /// Computed result
    pub result: f64,

    /// Expected result (if known analytically)
    pub expected: Option<f64>,

    /// Absolute error (|result - expected| if expected is known)
    pub error: Option<f64>,

    /// Whether Z3 verified this computation
    #[serde(default)]
    pub z3_verified: Option<bool>,
}

impl EquationEval {
    /// Create a new equation evaluation record.
    #[must_use]
    pub fn new(equation_id: impl Into<String>, result: f64) -> Self {
        Self {
            equation_id: equation_id.into(),
            inputs: IndexMap::new(),
            result,
            expected: None,
            error: None,
            z3_verified: None,
        }
    }

    /// Add an input variable.
    #[must_use]
    pub fn with_input(mut self, name: impl Into<String>, value: f64) -> Self {
        self.inputs.insert(name.into(), value);
        self
    }

    /// Set expected value and compute error.
    #[must_use]
    pub fn with_expected(mut self, expected: f64) -> Self {
        self.expected = Some(expected);
        self.error = Some((self.result - expected).abs());
        self
    }

    /// Mark as Z3 verified.
    #[must_use]
    pub fn with_z3_verified(mut self, verified: bool) -> Self {
        self.z3_verified = Some(verified);
        self
    }

    /// Check if result matches expected within tolerance.
    #[must_use]
    pub fn is_correct(&self, tolerance: f64) -> bool {
        self.error.is_none_or(|err| err <= tolerance)
    }
}

/// Algorithmic decision record.
///
/// Every decision MUST be logged with options considered, choice made,
/// and the rationale (metrics that drove the choice).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Decision {
    /// Decision type (e.g., `rcl_selection`, `two_opt_apply`, `update_best`)
    pub decision_type: String,

    /// Options that were considered
    pub options: Vec<String>,

    /// The option that was chosen
    pub chosen: String,

    /// Rationale: metrics that drove the choice
    pub rationale: IndexMap<String, f64>,

    /// RNG value used (if decision involved randomness)
    pub rng_value: Option<f64>,
}

impl Decision {
    /// Create a new decision record.
    #[must_use]
    pub fn new(decision_type: impl Into<String>, chosen: impl Into<String>) -> Self {
        Self {
            decision_type: decision_type.into(),
            options: Vec::new(),
            chosen: chosen.into(),
            rationale: IndexMap::new(),
            rng_value: None,
        }
    }

    /// Add options that were considered.
    #[must_use]
    pub fn with_options(mut self, options: Vec<String>) -> Self {
        self.options = options;
        self
    }

    /// Add a rationale metric.
    #[must_use]
    pub fn with_rationale(mut self, name: impl Into<String>, value: f64) -> Self {
        self.rationale.insert(name.into(), value);
        self
    }

    /// Set the RNG value used.
    #[must_use]
    pub fn with_rng_value(mut self, value: f64) -> Self {
        self.rng_value = Some(value);
        self
    }
}

/// Complete step entry for audit logging (EDD-16).
///
/// Every step MUST include all required fields for full reproducibility
/// and verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepEntry<S: Clone> {
    /// Monotonic step counter
    pub step_id: u64,

    /// Simulation time at this step
    pub timestamp: SimTime,

    /// Blake3 hash of RNG state BEFORE this step
    pub rng_state_before: [u8; 32],

    /// Blake3 hash of RNG state AFTER this step
    pub rng_state_after: [u8; 32],

    /// State snapshot before step (for replay verification)
    pub input_state: S,

    /// State snapshot after step
    pub output_state: S,

    /// All equation evaluations performed in this step
    pub equation_evaluations: Vec<EquationEval>,

    /// All decisions made in this step
    pub decisions: Vec<Decision>,

    /// Step type identifier (simulation-specific)
    pub step_type: String,

    /// Duration of step computation (for profiling)
    pub compute_duration_us: u64,
}

impl<S: Clone + Serialize> StepEntry<S> {
    /// Create a new step entry.
    #[must_use]
    pub fn new(
        step_id: u64,
        timestamp: SimTime,
        step_type: impl Into<String>,
        input_state: S,
        output_state: S,
    ) -> Self {
        contract_pre_iterator!();
        contract_post_configuration!(&"ok");
        Self {
            step_id,
            timestamp,
            rng_state_before: [0; 32],
            rng_state_after: [0; 32],
            input_state,
            output_state,
            equation_evaluations: Vec::new(),
            decisions: Vec::new(),
            step_type: step_type.into(),
            compute_duration_us: 0,
        }
    }

    /// Set RNG state hashes.
    #[must_use]
    pub fn with_rng_states(mut self, before: [u8; 32], after: [u8; 32]) -> Self {
        self.rng_state_before = before;
        self.rng_state_after = after;
        self
    }

    /// Add an equation evaluation.
    pub fn add_equation_eval(&mut self, eval: EquationEval) {
        self.equation_evaluations.push(eval);
    }

    /// Add a decision.
    pub fn add_decision(&mut self, decision: Decision) {
        self.decisions.push(decision);
    }

    /// Set compute duration.
    #[must_use]
    pub fn with_duration(mut self, duration_us: u64) -> Self {
        self.compute_duration_us = duration_us;
        self
    }
}

// =============================================================================
// Generated Test Case (EDD-18)
// =============================================================================

/// Test case generated from audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedTestCase {
    /// Test case name
    pub name: String,

    /// Equation being tested
    pub equation_id: String,

    /// Input values
    pub inputs: IndexMap<String, f64>,

    /// Expected output
    pub expected_output: f64,

    /// Assertion description
    pub assertion: String,

    /// Source step ID
    pub source_step_id: u64,

    /// Tolerance for floating-point comparison
    pub tolerance: f64,
}

impl GeneratedTestCase {
    /// Create a new generated test case.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        equation_id: impl Into<String>,
        expected_output: f64,
        source_step_id: u64,
    ) -> Self {
        Self {
            name: name.into(),
            equation_id: equation_id.into(),
            inputs: IndexMap::new(),
            expected_output,
            assertion: String::new(),
            source_step_id,
            tolerance: 1e-10,
        }
    }

    /// Add input.
    #[must_use]
    pub fn with_input(mut self, name: impl Into<String>, value: f64) -> Self {
        self.inputs.insert(name.into(), value);
        self
    }

    /// Set assertion.
    #[must_use]
    pub fn with_assertion(mut self, assertion: impl Into<String>) -> Self {
        self.assertion = assertion.into();
        self
    }

    /// Set tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Generate Rust test code.
    #[must_use]
    pub fn to_rust_test(&self) -> String {
        let inputs_str: Vec<String> = self
            .inputs
            .iter()
            .map(|(k, v)| format!("    let {k} = {v}_f64;"))
            .collect();

        format!(
            r#"#[test]
fn {name}() {{
    // Generated from step {step_id}
    // Equation: {equation_id}
{inputs}
    let result = /* compute {equation_id} */;
    let expected = {expected}_f64;
    assert!(
        (result - expected).abs() < {tolerance},
        "{assertion}: got {{result}}, expected {{expected}}"
    );
}}
"#,
            name = self.name,
            step_id = self.source_step_id,
            equation_id = self.equation_id,
            inputs = inputs_str.join("\n"),
            expected = self.expected_output,
            tolerance = self.tolerance,
            assertion = self.assertion,
        )
    }
}

// =============================================================================
// Audit Log Trait (MANDATORY)
// =============================================================================

/// MANDATORY trait for all EDD simulations (EDD-16, EDD-17, EDD-18).
///
/// Every simulation MUST implement this trait to produce a complete
/// audit trail of every step.
pub trait SimulationAuditLog {
    /// State snapshot type (must be serializable and cloneable)
    type StateSnapshot: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Record a step with full state capture.
    fn log_step(&mut self, entry: StepEntry<Self::StateSnapshot>);

    /// Get all logged entries.
    fn audit_log(&self) -> &[StepEntry<Self::StateSnapshot>];

    /// Get mutable access to audit log.
    fn audit_log_mut(&mut self) -> &mut Vec<StepEntry<Self::StateSnapshot>>;

    /// Clear the audit log.
    fn clear_audit_log(&mut self);

    /// Export log as JSON for analysis.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    fn export_audit_json(&self) -> SimResult<String> {
        serde_json::to_string_pretty(self.audit_log())
            .map_err(|e| SimError::serialization(format!("Audit log JSON export: {e}")))
    }

    /// Export log as compact JSON (no pretty printing).
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    fn export_audit_json_compact(&self) -> SimResult<String> {
        serde_json::to_string(self.audit_log())
            .map_err(|e| SimError::serialization(format!("Audit log JSON export: {e}")))
    }

    /// Generate test cases from audit log (EDD-18).
    fn generate_test_cases(&self) -> Vec<GeneratedTestCase> {
        let mut test_cases = Vec::new();

        for entry in self.audit_log() {
            for eval in &entry.equation_evaluations {
                let mut tc = GeneratedTestCase::new(
                    format!("test_{}_{}", eval.equation_id, entry.step_id),
                    &eval.equation_id,
                    eval.result,
                    entry.step_id,
                );

                for (name, value) in &eval.inputs {
                    tc = tc.with_input(name, *value);
                }

                if let Some(expected) = eval.expected {
                    tc = tc.with_assertion(format!(
                        "{} should equal {expected} (computed {result})",
                        eval.equation_id,
                        result = eval.result
                    ));
                } else {
                    tc = tc.with_assertion(format!("{} computation", eval.equation_id));
                }

                test_cases.push(tc);
            }
        }

        test_cases
    }

    /// Get total number of equation evaluations in log.
    fn total_equation_evals(&self) -> usize {
        self.audit_log()
            .iter()
            .map(|e| e.equation_evaluations.len())
            .sum()
    }

    /// Get total number of decisions in log.
    fn total_decisions(&self) -> usize {
        self.audit_log().iter().map(|e| e.decisions.len()).sum()
    }

    /// Verify all equation evaluations are correct within tolerance.
    fn verify_all_equations(&self, tolerance: f64) -> Vec<(u64, String, f64)> {
        let mut failures = Vec::new();

        for entry in self.audit_log() {
            for eval in &entry.equation_evaluations {
                if !eval.is_correct(tolerance) {
                    if let Some(err) = eval.error {
                        failures.push((entry.step_id, eval.equation_id.clone(), err));
                    }
                }
            }
        }

        failures
    }
}

// =============================================================================
// Replay Support
// =============================================================================

/// Replay speed control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplaySpeed {
    /// Instant replay (no delays)
    #[default]
    Instant,
    /// Real-time replay with specified delay per step
    RealTime(Duration),
    /// Step-by-step (wait for external trigger)
    StepByStep,
    /// Fast forward (N times real speed)
    FastForward(u32),
}

/// Replay state for audit log playback.
#[derive(Debug, Clone)]
pub struct ReplayState<S: Clone> {
    /// Current step index in the log
    pub current_index: usize,
    /// Total steps in the log
    pub total_steps: usize,
    /// Current state snapshot
    pub current_state: Option<S>,
    /// Replay speed
    pub speed: ReplaySpeed,
    /// Whether replay is paused
    pub paused: bool,
}

impl<S: Clone> ReplayState<S> {
    /// Create a new replay state.
    #[must_use]
    pub fn new(total_steps: usize) -> Self {
        Self {
            current_index: 0,
            total_steps,
            current_state: None,
            speed: ReplaySpeed::default(),
            paused: false,
        }
    }

    /// Check if replay is complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current_index >= self.total_steps
    }

    /// Get progress as percentage.
    #[must_use]
    pub fn progress_percent(&self) -> f64 {
        if self.total_steps == 0 {
            100.0
        } else {
            (self.current_index as f64 / self.total_steps as f64) * 100.0
        }
    }

    /// Advance to next step.
    pub fn advance(&mut self) {
        if self.current_index < self.total_steps {
            self.current_index += 1;
        }
    }

    /// Seek to specific step.
    pub fn seek(&mut self, step_index: usize) {
        contract_pre_iterator!();
        self.current_index = step_index.min(self.total_steps);
    }

    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.current_state = None;
    }
}

/// Audit log replayer.
pub struct AuditLogReplayer<S: Clone + Serialize + for<'de> Deserialize<'de>> {
    /// The audit log entries
    entries: Vec<StepEntry<S>>,
    /// Current replay state
    state: ReplayState<S>,
}

impl<S: Clone + Serialize + for<'de> Deserialize<'de>> AuditLogReplayer<S> {
    /// Create a new replayer from audit log.
    #[must_use]
    pub fn new(entries: Vec<StepEntry<S>>) -> Self {
        let total = entries.len();
        Self {
            entries,
            state: ReplayState::new(total),
        }
    }

    /// Get current replay state.
    #[must_use]
    pub fn state(&self) -> &ReplayState<S> {
        &self.state
    }

    /// Get mutable replay state.
    pub fn state_mut(&mut self) -> &mut ReplayState<S> {
        &mut self.state
    }

    /// Get current step entry (if any).
    #[must_use]
    pub fn current_entry(&self) -> Option<&StepEntry<S>> {
        self.entries.get(self.state.current_index)
    }

    /// Step forward and return the entry.
    pub fn step_forward(&mut self) -> Option<&StepEntry<S>> {
        if self.state.paused || self.state.is_complete() {
            return None;
        }

        contract_pre_iterator!();
        let entry = self.entries.get(self.state.current_index);
        if entry.is_some() {
            self.state.current_state = entry.map(|e| e.output_state.clone());
            self.state.advance();
        }
        entry
    }

    /// Seek to a specific step index.
    pub fn seek_to(&mut self, index: usize) {
        self.state.seek(index);
        if let Some(entry) = self.entries.get(index) {
            self.state.current_state = Some(entry.output_state.clone());
        }
    }

    /// Get all entries.
    #[must_use]
    pub fn entries(&self) -> &[StepEntry<S>] {
        &self.entries
    }

    /// Find entry by step ID.
    #[must_use]
    pub fn find_by_step_id(&self, step_id: u64) -> Option<&StepEntry<S>> {
        contract_pre_iterator!();
        self.entries.iter().find(|e| e.step_id == step_id)
    }

    /// Get entries in time range.
    pub fn entries_in_range(&self, start: SimTime, end: SimTime) -> Vec<&StepEntry<S>> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }
}

// =============================================================================
// Verification Helpers
// =============================================================================

/// Verify RNG state consistency in audit log.
///
/// Returns list of step IDs where RNG state is inconsistent.
pub fn verify_rng_consistency<S: Clone + Serialize + for<'de> Deserialize<'de>>(
    log: &[StepEntry<S>],
) -> Vec<u64> {
    let mut inconsistencies = Vec::new();

    for i in 1..log.len() {
        // The "after" state of step i-1 should match "before" state of step i
        if log[i - 1].rng_state_after != log[i].rng_state_before {
            inconsistencies.push(log[i].step_id);
        }
    }

    inconsistencies
}

/// Compute Blake3 hash of serializable state.
///
/// # Errors
///
/// Returns error if serialization fails.
pub fn hash_state<S: Serialize>(state: &S) -> SimResult<[u8; 32]> {
    let bytes = bincode::serialize(state)
        .map_err(|e| SimError::serialization(format!("Hash state: {e}")))?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

// =============================================================================
// TSP-Specific Audit Types
// =============================================================================

/// TSP step type for audit logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TspStepType {
    /// Initial tour construction
    Construction,
    /// 2-opt improvement pass
    TwoOptPass,
    /// 2-opt improvement applied
    TwoOptImprove,
    /// Best tour updated
    BestUpdate,
    /// GRASP iteration complete
    GraspIteration,
}

impl std::fmt::Display for TspStepType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Construction => write!(f, "construction"),
            Self::TwoOptPass => write!(f, "two_opt_pass"),
            Self::TwoOptImprove => write!(f, "two_opt_improve"),
            Self::BestUpdate => write!(f, "best_update"),
            Self::GraspIteration => write!(f, "grasp_iteration"),
        }
    }
}

/// TSP state snapshot for audit logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspStateSnapshot {
    /// Current tour (city indices)
    pub tour: Vec<usize>,
    /// Current tour length
    pub tour_length: f64,
    /// Best tour found
    pub best_tour: Vec<usize>,
    /// Best tour length
    pub best_tour_length: f64,
    /// Number of restarts
    pub restarts: u64,
    /// Number of 2-opt iterations
    pub two_opt_iterations: u64,
    /// Number of 2-opt improvements
    pub two_opt_improvements: u64,
}

impl Default for TspStateSnapshot {
    fn default() -> Self {
        Self {
            tour: Vec::new(),
            tour_length: 0.0,
            best_tour: Vec::new(),
            best_tour_length: 0.0,
            restarts: 0,
            two_opt_iterations: 0,
            two_opt_improvements: 0,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // RED PHASE: Failing tests first (EXTREME TDD)
    // =========================================================================

    // -------------------------------------------------------------------------
    // EquationEval Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_equation_eval_creation() {
        let eval = EquationEval::new("two_opt_delta", 0.5);

        assert_eq!(eval.equation_id, "two_opt_delta");
        assert!((eval.result - 0.5).abs() < f64::EPSILON);
        assert!(eval.inputs.is_empty());
        assert!(eval.expected.is_none());
        assert!(eval.error.is_none());
    }

    #[test]
    fn test_equation_eval_with_inputs() {
        let eval = EquationEval::new("two_opt_delta", 0.5)
            .with_input("d_i_i1", 1.0)
            .with_input("d_j_j1", 0.8)
            .with_input("d_i_j", 0.7)
            .with_input("d_i1_j1", 0.6);

        assert_eq!(eval.inputs.len(), 4);
        assert!((eval.inputs["d_i_i1"] - 1.0).abs() < f64::EPSILON);
        assert!((eval.inputs["d_j_j1"] - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_equation_eval_with_expected() {
        let eval = EquationEval::new("tour_length", 4.5).with_expected(4.48);

        assert!(eval.expected.is_some());
        assert!((eval.expected.unwrap() - 4.48).abs() < f64::EPSILON);
        assert!(eval.error.is_some());
        // Use tolerance since 4.5 - 4.48 = 0.020000000000000018 due to floating point
        assert!((eval.error.unwrap() - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_equation_eval_is_correct() {
        let eval_correct = EquationEval::new("test", 1.0).with_expected(1.001);
        let eval_incorrect = EquationEval::new("test", 1.0).with_expected(2.0);

        assert!(eval_correct.is_correct(0.01));
        assert!(!eval_incorrect.is_correct(0.01));
    }

    #[test]
    fn test_equation_eval_z3_verified() {
        let eval = EquationEval::new("test", 1.0).with_z3_verified(true);

        assert_eq!(eval.z3_verified, Some(true));
    }

    // -------------------------------------------------------------------------
    // Decision Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_decision_creation() {
        let decision = Decision::new("rcl_selection", "city_7");

        assert_eq!(decision.decision_type, "rcl_selection");
        assert_eq!(decision.chosen, "city_7");
        assert!(decision.options.is_empty());
        assert!(decision.rationale.is_empty());
    }

    #[test]
    fn test_decision_with_options() {
        let decision = Decision::new("rcl_selection", "city_7").with_options(vec![
            "city_3".into(),
            "city_7".into(),
            "city_12".into(),
        ]);

        assert_eq!(decision.options.len(), 3);
        assert!(decision.options.contains(&"city_7".to_string()));
    }

    #[test]
    fn test_decision_with_rationale() {
        let decision = Decision::new("two_opt_apply", "apply")
            .with_rationale("delta", 0.15)
            .with_rationale("new_length", 4.35);

        assert_eq!(decision.rationale.len(), 2);
        assert!((decision.rationale["delta"] - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decision_with_rng_value() {
        let decision = Decision::new("rcl_selection", "city_7").with_rng_value(0.42);

        assert_eq!(decision.rng_value, Some(0.42));
    }

    // -------------------------------------------------------------------------
    // StepEntry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_step_entry_creation() {
        let entry: StepEntry<u64> =
            StepEntry::new(1, SimTime::from_secs(0.1), "construction", 0u64, 1u64);

        assert_eq!(entry.step_id, 1);
        assert_eq!(entry.step_type, "construction");
        assert_eq!(entry.input_state, 0);
        assert_eq!(entry.output_state, 1);
    }

    #[test]
    fn test_step_entry_with_rng_states() {
        let before = [1u8; 32];
        let after = [2u8; 32];

        let entry: StepEntry<u64> = StepEntry::new(1, SimTime::from_secs(0.1), "test", 0u64, 1u64)
            .with_rng_states(before, after);

        assert_eq!(entry.rng_state_before, before);
        assert_eq!(entry.rng_state_after, after);
    }

    #[test]
    fn test_step_entry_add_equation_eval() {
        let mut entry: StepEntry<u64> =
            StepEntry::new(1, SimTime::from_secs(0.1), "test", 0u64, 1u64);

        entry.add_equation_eval(EquationEval::new("test_eq", 1.5));

        assert_eq!(entry.equation_evaluations.len(), 1);
        assert_eq!(entry.equation_evaluations[0].equation_id, "test_eq");
    }

    #[test]
    fn test_step_entry_add_decision() {
        let mut entry: StepEntry<u64> =
            StepEntry::new(1, SimTime::from_secs(0.1), "test", 0u64, 1u64);

        entry.add_decision(Decision::new("test_decision", "option_a"));

        assert_eq!(entry.decisions.len(), 1);
        assert_eq!(entry.decisions[0].decision_type, "test_decision");
    }

    #[test]
    fn test_step_entry_with_duration() {
        let entry: StepEntry<u64> =
            StepEntry::new(1, SimTime::from_secs(0.1), "test", 0u64, 1u64).with_duration(1000);

        assert_eq!(entry.compute_duration_us, 1000);
    }

    // -------------------------------------------------------------------------
    // GeneratedTestCase Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_generated_test_case_creation() {
        let tc = GeneratedTestCase::new("test_two_opt_1", "two_opt_delta", 0.5, 42);

        assert_eq!(tc.name, "test_two_opt_1");
        assert_eq!(tc.equation_id, "two_opt_delta");
        assert!((tc.expected_output - 0.5).abs() < f64::EPSILON);
        assert_eq!(tc.source_step_id, 42);
    }

    #[test]
    fn test_generated_test_case_with_inputs() {
        let tc = GeneratedTestCase::new("test", "eq", 1.0, 1)
            .with_input("x", 2.0)
            .with_input("y", 3.0);

        assert_eq!(tc.inputs.len(), 2);
        assert!((tc.inputs["x"] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_generated_test_case_to_rust() {
        let tc = GeneratedTestCase::new("test_eq_1", "my_equation", 42.0, 5)
            .with_input("a", 1.0)
            .with_assertion("equation should compute correctly");

        let rust_code = tc.to_rust_test();

        assert!(rust_code.contains("#[test]"));
        assert!(rust_code.contains("fn test_eq_1()"));
        assert!(rust_code.contains("let a = 1"));
        assert!(rust_code.contains("let expected = 42"));
    }

    // -------------------------------------------------------------------------
    // ReplayState Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_replay_state_creation() {
        let state: ReplayState<u64> = ReplayState::new(100);

        assert_eq!(state.current_index, 0);
        assert_eq!(state.total_steps, 100);
        assert!(!state.is_complete());
        assert!(!state.paused);
    }

    #[test]
    fn test_replay_state_advance() {
        let mut state: ReplayState<u64> = ReplayState::new(10);

        state.advance();
        assert_eq!(state.current_index, 1);

        state.advance();
        assert_eq!(state.current_index, 2);
    }

    #[test]
    fn test_replay_state_progress() {
        let mut state: ReplayState<u64> = ReplayState::new(10);

        assert!((state.progress_percent() - 0.0).abs() < f64::EPSILON);

        state.seek(5);
        assert!((state.progress_percent() - 50.0).abs() < f64::EPSILON);

        state.seek(10);
        assert!((state.progress_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_replay_state_is_complete() {
        let mut state: ReplayState<u64> = ReplayState::new(3);

        assert!(!state.is_complete());

        state.seek(3);
        assert!(state.is_complete());
    }

    #[test]
    fn test_replay_state_reset() {
        let mut state: ReplayState<u64> = ReplayState::new(10);
        state.seek(5);
        state.current_state = Some(42);

        state.reset();

        assert_eq!(state.current_index, 0);
        assert!(state.current_state.is_none());
    }

    // -------------------------------------------------------------------------
    // AuditLogReplayer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_replayer_creation() {
        let entries: Vec<StepEntry<u64>> = vec![
            StepEntry::new(0, SimTime::ZERO, "step_0", 0, 1),
            StepEntry::new(1, SimTime::from_secs(0.1), "step_1", 1, 2),
        ];

        let replayer = AuditLogReplayer::new(entries);

        assert_eq!(replayer.entries().len(), 2);
        assert_eq!(replayer.state().total_steps, 2);
    }

    #[test]
    fn test_replayer_step_forward() {
        let entries: Vec<StepEntry<u64>> = vec![
            StepEntry::new(0, SimTime::ZERO, "step_0", 0, 1),
            StepEntry::new(1, SimTime::from_secs(0.1), "step_1", 1, 2),
        ];

        let mut replayer = AuditLogReplayer::new(entries);

        let entry = replayer.step_forward();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().step_id, 0);

        let entry = replayer.step_forward();
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().step_id, 1);

        // Should return None when complete
        let entry = replayer.step_forward();
        assert!(entry.is_none());
    }

    #[test]
    fn test_replayer_seek() {
        let entries: Vec<StepEntry<u64>> = vec![
            StepEntry::new(0, SimTime::ZERO, "step_0", 0, 1),
            StepEntry::new(1, SimTime::from_secs(0.1), "step_1", 1, 2),
            StepEntry::new(2, SimTime::from_secs(0.2), "step_2", 2, 3),
        ];

        let mut replayer = AuditLogReplayer::new(entries);

        replayer.seek_to(2);
        assert_eq!(replayer.state().current_index, 2);
        assert_eq!(replayer.state().current_state, Some(3));
    }

    #[test]
    fn test_replayer_find_by_step_id() {
        let entries: Vec<StepEntry<u64>> = vec![
            StepEntry::new(10, SimTime::ZERO, "step_0", 0, 1),
            StepEntry::new(20, SimTime::from_secs(0.1), "step_1", 1, 2),
        ];

        let replayer = AuditLogReplayer::new(entries);

        let found = replayer.find_by_step_id(20);
        assert!(found.is_some());
        assert_eq!(found.unwrap().step_id, 20);

        let not_found = replayer.find_by_step_id(99);
        assert!(not_found.is_none());
    }

    // -------------------------------------------------------------------------
    // Verification Helper Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_verify_rng_consistency_valid() {
        let entries: Vec<StepEntry<u64>> = vec![
            StepEntry::new(0, SimTime::ZERO, "s0", 0, 1).with_rng_states([1; 32], [2; 32]),
            StepEntry::new(1, SimTime::from_secs(0.1), "s1", 1, 2)
                .with_rng_states([2; 32], [3; 32]),
            StepEntry::new(2, SimTime::from_secs(0.2), "s2", 2, 3)
                .with_rng_states([3; 32], [4; 32]),
        ];

        let inconsistencies = verify_rng_consistency(&entries);
        assert!(inconsistencies.is_empty());
    }

    #[test]
    fn test_verify_rng_consistency_invalid() {
        let entries: Vec<StepEntry<u64>> = vec![
            StepEntry::new(0, SimTime::ZERO, "s0", 0, 1).with_rng_states([1; 32], [2; 32]),
            StepEntry::new(1, SimTime::from_secs(0.1), "s1", 1, 2)
                .with_rng_states([99; 32], [3; 32]), // Inconsistent!
        ];

        let inconsistencies = verify_rng_consistency(&entries);
        assert_eq!(inconsistencies.len(), 1);
        assert_eq!(inconsistencies[0], 1);
    }

    #[test]
    fn test_hash_state() {
        let state1 = 42u64;
        let state2 = 42u64;
        let state3 = 43u64;

        let hash1 = hash_state(&state1).expect("hash");
        let hash2 = hash_state(&state2).expect("hash");
        let hash3 = hash_state(&state3).expect("hash");

        assert_eq!(hash1, hash2); // Same state = same hash
        assert_ne!(hash1, hash3); // Different state = different hash
    }

    // -------------------------------------------------------------------------
    // SimulationAuditLog Trait Tests (via mock implementation)
    // -------------------------------------------------------------------------

    struct MockSimulation {
        log: Vec<StepEntry<u64>>,
    }

    impl MockSimulation {
        fn new() -> Self {
            Self { log: Vec::new() }
        }
    }

    impl SimulationAuditLog for MockSimulation {
        type StateSnapshot = u64;

        fn log_step(&mut self, entry: StepEntry<Self::StateSnapshot>) {
            self.log.push(entry);
        }

        fn audit_log(&self) -> &[StepEntry<Self::StateSnapshot>] {
            &self.log
        }

        fn audit_log_mut(&mut self) -> &mut Vec<StepEntry<Self::StateSnapshot>> {
            &mut self.log
        }

        fn clear_audit_log(&mut self) {
            self.log.clear();
        }
    }

    #[test]
    fn test_simulation_audit_log_trait() {
        let mut sim = MockSimulation::new();

        let mut entry = StepEntry::new(0, SimTime::ZERO, "test", 0u64, 1u64);
        entry.add_equation_eval(EquationEval::new("eq1", 1.0).with_expected(1.0));
        entry.add_decision(Decision::new("decide", "a"));

        sim.log_step(entry);

        assert_eq!(sim.audit_log().len(), 1);
        assert_eq!(sim.total_equation_evals(), 1);
        assert_eq!(sim.total_decisions(), 1);
    }

    #[test]
    fn test_simulation_audit_log_export_json() {
        let mut sim = MockSimulation::new();
        sim.log_step(StepEntry::new(0, SimTime::ZERO, "test", 0u64, 1u64));

        let json = sim.export_audit_json().expect("json export");
        assert!(json.contains("step_id"));
        assert!(json.contains("\"step_id\": 0"));
    }

    #[test]
    fn test_simulation_audit_log_generate_tests() {
        let mut sim = MockSimulation::new();

        let mut entry = StepEntry::new(0, SimTime::ZERO, "test", 0u64, 1u64);
        entry.add_equation_eval(
            EquationEval::new("my_equation", 42.0)
                .with_input("x", 1.0)
                .with_input("y", 2.0),
        );
        sim.log_step(entry);

        let test_cases = sim.generate_test_cases();

        assert_eq!(test_cases.len(), 1);
        assert_eq!(test_cases[0].equation_id, "my_equation");
        assert!((test_cases[0].expected_output - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_simulation_audit_log_verify_equations() {
        let mut sim = MockSimulation::new();

        let mut entry1 = StepEntry::new(0, SimTime::ZERO, "test", 0u64, 1u64);
        entry1.add_equation_eval(EquationEval::new("eq1", 1.0).with_expected(1.0)); // Correct

        let mut entry2 = StepEntry::new(1, SimTime::from_secs(0.1), "test", 1u64, 2u64);
        entry2.add_equation_eval(EquationEval::new("eq2", 1.0).with_expected(2.0)); // Wrong!

        sim.log_step(entry1);
        sim.log_step(entry2);

        let failures = sim.verify_all_equations(0.001);

        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, 1); // Step 1 failed
        assert_eq!(failures[0].1, "eq2"); // Equation eq2
    }

    #[test]
    fn test_simulation_audit_log_clear() {
        let mut sim = MockSimulation::new();
        sim.log_step(StepEntry::new(0, SimTime::ZERO, "test", 0u64, 1u64));

        assert_eq!(sim.audit_log().len(), 1);

        sim.clear_audit_log();
        assert!(sim.audit_log().is_empty());
    }

    // -------------------------------------------------------------------------
    // Serialization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_equation_eval_serialization() {
        let eval = EquationEval::new("test", 1.5)
            .with_input("x", 1.0)
            .with_expected(1.5);

        let json = serde_json::to_string(&eval).expect("serialize");
        let restored: EquationEval = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.equation_id, eval.equation_id);
        assert!((restored.result - eval.result).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decision_serialization() {
        let decision = Decision::new("test", "a")
            .with_options(vec!["a".into(), "b".into()])
            .with_rationale("score", 0.5);

        let json = serde_json::to_string(&decision).expect("serialize");
        let restored: Decision = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.decision_type, decision.decision_type);
        assert_eq!(restored.chosen, decision.chosen);
    }

    #[test]
    fn test_step_entry_serialization() {
        let entry: StepEntry<u64> =
            StepEntry::new(1, SimTime::from_secs(0.5), "test", 10u64, 20u64);

        let json = serde_json::to_string(&entry).expect("serialize");
        let restored: StepEntry<u64> = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.step_id, entry.step_id);
        assert_eq!(restored.input_state, entry.input_state);
        assert_eq!(restored.output_state, entry.output_state);
    }
}
