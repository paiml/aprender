//! Jidoka (自働化) - Autonomous anomaly detection.
//!
//! Implements Toyota's Jidoka principle: machines that detect problems
//! and stop automatically to prevent defect propagation.
//!
//! # Anomaly Types
//!
//! 1. **Non-finite values**: NaN or Inf in any state variable
//! 2. **Energy drift**: Total energy deviates from initial beyond tolerance
//! 3. **Constraint violations**: Physical constraints exceeded
//!
//! # Severity Levels
//!
//! Following the Batuta Stack Review, Jidoka uses graduated severity:
//! - **Acceptable**: Within tolerance, continue normally
//! - **Warning**: Approaching tolerance, log and continue
//! - **Critical**: Tolerance exceeded, stop the line
//! - **Fatal**: Unrecoverable state, halt immediately
//!
//! # Advanced TPS Kaizen (Section 4.3)
//!
//! - **Pre-flight Jidoka**: In-process anomaly detection during computation [49]
//! - **Andon vs Jidoka**: Self-healing auto-correction vs full stop [51][57]
//!
//! # Design
//!
//! The guard runs after every simulation step, ensuring immediate
//! detection of anomalies. This prevents error propagation and
//! enables root cause analysis.

use crate::config::SimConfig;
use crate::engine::state::SimState;
use crate::error::{SimError, SimResult};
use serde::{Deserialize, Serialize};

/// Severity levels for Jidoka violations.
///
/// Graduated response avoids false positives (Muda of over-processing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Acceptable variance within tolerance (continue).
    Acceptable,
    /// Warning: approaching tolerance boundary (log, continue).
    Warning,
    /// Critical: tolerance exceeded (stop the line).
    Critical,
    /// Fatal: unrecoverable state (halt immediately).
    Fatal,
}

/// Warning from Jidoka check (non-critical issue).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JidokaWarning {
    /// Energy drift approaching tolerance.
    EnergyDriftApproaching {
        /// Current drift value.
        drift: f64,
        /// Tolerance threshold.
        tolerance: f64,
    },
    /// Constraint approaching violation.
    ConstraintApproaching {
        /// Constraint name.
        name: String,
        /// Current violation amount.
        violation: f64,
        /// Tolerance threshold.
        tolerance: f64,
    },
}

/// Classifier for graduated Jidoka responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeverityClassifier {
    /// Warning threshold as fraction of tolerance (e.g., 0.8 = warn at 80%).
    pub warning_fraction: f64,
}

impl Default for SeverityClassifier {
    fn default() -> Self {
        Self {
            warning_fraction: 0.8,
        }
    }
}

impl SeverityClassifier {
    /// Create a new severity classifier.
    #[must_use]
    pub const fn new(warning_fraction: f64) -> Self {
        Self { warning_fraction }
    }

    /// Classify energy drift severity.
    #[must_use]
    pub fn classify_energy_drift(&self, drift: f64, tolerance: f64) -> ViolationSeverity {
        if drift.is_nan() || drift.is_infinite() {
            ViolationSeverity::Fatal
        } else if drift > tolerance {
            ViolationSeverity::Critical
        } else if drift > tolerance * self.warning_fraction {
            ViolationSeverity::Warning
        } else {
            ViolationSeverity::Acceptable
        }
    }

    /// Classify constraint violation severity.
    #[must_use]
    pub fn classify_constraint(&self, violation: f64, tolerance: f64) -> ViolationSeverity {
        let abs_violation = violation.abs();
        if abs_violation.is_nan() || abs_violation.is_infinite() {
            ViolationSeverity::Fatal
        } else if abs_violation > tolerance {
            ViolationSeverity::Critical
        } else if abs_violation > tolerance * self.warning_fraction {
            ViolationSeverity::Warning
        } else {
            ViolationSeverity::Acceptable
        }
    }
}

/// Jidoka violation types.
///
/// Each variant represents a specific anomaly that triggered the stop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JidokaViolation {
    /// Non-finite value (NaN or Inf) detected.
    NonFiniteValue {
        /// Location of the non-finite value (e.g., "position.x").
        location: String,
        /// The non-finite value itself.
        value: f64,
    },
    /// Energy conservation violated.
    EnergyDrift {
        /// Current energy.
        current: f64,
        /// Initial energy.
        initial: f64,
        /// Relative drift.
        drift: f64,
        /// Configured tolerance.
        tolerance: f64,
    },
    /// Constraint violated.
    ConstraintViolation {
        /// Constraint name.
        name: String,
        /// Current value.
        value: f64,
        /// Violation amount.
        violation: f64,
        /// Configured tolerance.
        tolerance: f64,
    },
}

impl From<JidokaViolation> for SimError {
    fn from(v: JidokaViolation) -> Self {
        match v {
            JidokaViolation::NonFiniteValue { location, .. } => Self::NonFiniteValue { location },
            JidokaViolation::EnergyDrift {
                drift, tolerance, ..
            } => Self::EnergyDrift { drift, tolerance },
            JidokaViolation::ConstraintViolation {
                name,
                violation,
                tolerance,
                ..
            } => Self::ConstraintViolation {
                name,
                violation,
                tolerance,
            },
        }
    }
}

/// Jidoka guard configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JidokaConfig {
    /// Maximum allowed relative energy drift.
    pub energy_tolerance: f64,
    /// NaN/Inf detection enabled.
    pub check_finite: bool,
    /// Constraint violation threshold.
    pub constraint_tolerance: f64,
    /// Enable energy conservation check.
    pub check_energy: bool,
    /// Severity classifier for graduated responses.
    #[serde(default)]
    pub severity_classifier: SeverityClassifier,
}

impl Default for JidokaConfig {
    fn default() -> Self {
        Self {
            energy_tolerance: 1e-6,
            check_finite: true,
            constraint_tolerance: 1e-8,
            check_energy: true,
            severity_classifier: SeverityClassifier::default(),
        }
    }
}

/// Jidoka guard for autonomous anomaly detection.
///
/// # Example
///
/// ```rust
/// use simular::engine::jidoka::{JidokaGuard, JidokaConfig};
/// use simular::engine::state::SimState;
///
/// let mut guard = JidokaGuard::new(JidokaConfig::default());
/// let state = SimState::default();
///
/// // Check will pass for valid state
/// assert!(guard.check(&state).is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct JidokaGuard {
    /// Configuration.
    config: JidokaConfig,
    /// Initial energy (set on first check).
    initial_energy: Option<f64>,
}

impl JidokaGuard {
    /// Create a new Jidoka guard with given configuration.
    #[must_use]
    pub const fn new(config: JidokaConfig) -> Self {
        Self {
            config,
            initial_energy: None,
        }
    }

    /// Create from simulation configuration.
    #[must_use]
    pub fn from_config(config: &SimConfig) -> Self {
        Self::new(config.jidoka.clone())
    }

    /// Check state for anomalies (Jidoka inspection).
    ///
    /// This method should be called after every simulation step.
    ///
    /// # Errors
    ///
    /// Returns `SimError` if any anomaly is detected:
    /// - `NonFiniteValue`: NaN or Inf found
    /// - `EnergyDrift`: Energy conservation violated
    /// - `ConstraintViolation`: Physical constraint exceeded
    pub fn check(&mut self, state: &SimState) -> SimResult<()> {
        // Check 1: Non-finite values (Poka-Yoke)
        if self.config.check_finite {
            self.check_finite(state)?;
        }

        // Check 2: Energy conservation
        if self.config.check_energy {
            self.check_energy(state)?;
        }

        // Check 3: Constraints
        self.check_constraints(state)?;

        Ok(())
    }

    /// Check for non-finite values in state.
    #[allow(clippy::unused_self)] // Consistent method signature with other checks
    fn check_finite(&self, state: &SimState) -> SimResult<()> {
        // Check all positions
        for (i, pos) in state.positions().iter().enumerate() {
            if !pos.x.is_finite() {
                return Err(SimError::NonFiniteValue {
                    location: format!("positions[{i}].x"),
                });
            }
            if !pos.y.is_finite() {
                return Err(SimError::NonFiniteValue {
                    location: format!("positions[{i}].y"),
                });
            }
            if !pos.z.is_finite() {
                return Err(SimError::NonFiniteValue {
                    location: format!("positions[{i}].z"),
                });
            }
        }

        // Check all velocities
        for (i, vel) in state.velocities().iter().enumerate() {
            if !vel.x.is_finite() {
                return Err(SimError::NonFiniteValue {
                    location: format!("velocities[{i}].x"),
                });
            }
            if !vel.y.is_finite() {
                return Err(SimError::NonFiniteValue {
                    location: format!("velocities[{i}].y"),
                });
            }
            if !vel.z.is_finite() {
                return Err(SimError::NonFiniteValue {
                    location: format!("velocities[{i}].z"),
                });
            }
        }

        Ok(())
    }

    /// Check energy conservation.
    fn check_energy(&mut self, state: &SimState) -> SimResult<()> {
        let current_energy = state.total_energy();

        // Skip if no energy defined
        if !current_energy.is_finite() || current_energy.abs() < f64::EPSILON {
            return Ok(());
        }

        match self.initial_energy {
            None => {
                // First check - record initial energy
                self.initial_energy = Some(current_energy);
                Ok(())
            }
            Some(initial) => {
                let drift = (current_energy - initial).abs() / initial.abs().max(f64::EPSILON);

                if drift > self.config.energy_tolerance {
                    Err(SimError::EnergyDrift {
                        drift,
                        tolerance: self.config.energy_tolerance,
                    })
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Check constraint violations.
    fn check_constraints(&self, state: &SimState) -> SimResult<()> {
        for (name, violation) in state.constraint_violations() {
            if violation.abs() > self.config.constraint_tolerance {
                return Err(SimError::ConstraintViolation {
                    name,
                    violation,
                    tolerance: self.config.constraint_tolerance,
                });
            }
        }

        Ok(())
    }

    /// Reset the guard (clear initial energy).
    #[allow(clippy::missing_const_for_fn)] // Mutable const not stable in all contexts
    pub fn reset(&mut self) {
        self.initial_energy = None;
    }

    /// Get current configuration.
    #[must_use]
    pub const fn config(&self) -> &JidokaConfig {
        &self.config
    }

    /// Check state with graduated severity (smart Jidoka).
    ///
    /// Returns warnings for approaching violations without stopping.
    /// Only returns errors for Critical or Fatal violations.
    ///
    /// # Errors
    ///
    /// Returns error for Critical/Fatal violations.
    pub fn check_with_warnings(
        &mut self,
        state: &SimState,
    ) -> Result<Vec<JidokaWarning>, SimError> {
        let mut warnings = Vec::new();

        // Check 1: Non-finite values (always Fatal)
        if self.config.check_finite {
            self.check_finite(state)?;
        }

        // Check 2: Energy conservation with graduated response
        if self.config.check_energy {
            if let Some(warning) = self.check_energy_graduated(state)? {
                warnings.push(warning);
            }
        }

        // Check 3: Constraints with graduated response
        warnings.extend(self.check_constraints_graduated(state)?);

        Ok(warnings)
    }

    /// Check energy with graduated severity.
    fn check_energy_graduated(
        &mut self,
        state: &SimState,
    ) -> Result<Option<JidokaWarning>, SimError> {
        let current_energy = state.total_energy();

        // Skip if no energy defined
        if !current_energy.is_finite() || current_energy.abs() < f64::EPSILON {
            return Ok(None);
        }

        match self.initial_energy {
            None => {
                self.initial_energy = Some(current_energy);
                Ok(None)
            }
            Some(initial) => {
                let drift = (current_energy - initial).abs() / initial.abs().max(f64::EPSILON);
                let severity = self
                    .config
                    .severity_classifier
                    .classify_energy_drift(drift, self.config.energy_tolerance);

                match severity {
                    ViolationSeverity::Acceptable => Ok(None),
                    ViolationSeverity::Warning => Ok(Some(JidokaWarning::EnergyDriftApproaching {
                        drift,
                        tolerance: self.config.energy_tolerance,
                    })),
                    ViolationSeverity::Critical | ViolationSeverity::Fatal => {
                        Err(SimError::EnergyDrift {
                            drift,
                            tolerance: self.config.energy_tolerance,
                        })
                    }
                }
            }
        }
    }

    /// Check constraints with graduated severity.
    fn check_constraints_graduated(
        &self,
        state: &SimState,
    ) -> Result<Vec<JidokaWarning>, SimError> {
        let mut warnings = Vec::new();

        for (name, violation) in state.constraint_violations() {
            let severity = self
                .config
                .severity_classifier
                .classify_constraint(violation, self.config.constraint_tolerance);

            match severity {
                ViolationSeverity::Acceptable => {}
                ViolationSeverity::Warning => {
                    warnings.push(JidokaWarning::ConstraintApproaching {
                        name,
                        violation,
                        tolerance: self.config.constraint_tolerance,
                    });
                }
                ViolationSeverity::Critical | ViolationSeverity::Fatal => {
                    return Err(SimError::ConstraintViolation {
                        name,
                        violation,
                        tolerance: self.config.constraint_tolerance,
                    });
                }
            }
        }

        Ok(warnings)
    }
}

// =============================================================================
// Pre-flight Jidoka (Section 4.3.1)
// =============================================================================

bitflags::bitflags! {
    /// Conditions that trigger immediate abort during computation [49][51].
    ///
    /// Pre-flight Jidoka prevents Muda of Processing by aborting before
    /// defects propagate through the computation graph.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AbortConditions: u32 {
        /// Abort on NaN or Infinity values.
        const NON_FINITE = 0b0001;
        /// Abort when gradient norm exceeds threshold.
        const GRADIENT_EXPLOSION = 0b0010;
        /// Abort when gradient norm drops below threshold.
        const GRADIENT_VANISHING = 0b0100;
        /// Abort on value exceeding physical bounds.
        const BOUND_VIOLATION = 0b1000;
    }
}

/// Pre-flight Jidoka guard for in-process anomaly detection [49][51].
///
/// Unlike post-process `JidokaGuard`, `PreflightJidoka` aborts *during*
/// computation to prevent wasted work (Muda of Processing).
///
/// # Example
///
/// ```rust
/// use simular::engine::jidoka::{PreflightJidoka, AbortConditions};
///
/// let mut preflight = PreflightJidoka::new();
/// assert!(preflight.check_value(1.0).is_ok());
/// assert!(preflight.check_value(f64::NAN).is_err());
/// ```
#[derive(Debug, Clone)]
pub struct PreflightJidoka {
    /// Abort conditions (OR'd together).
    abort_on: AbortConditions,
    /// Threshold for gradient explosion detection.
    gradient_explosion_threshold: f64,
    /// Threshold for gradient vanishing detection.
    gradient_vanishing_threshold: f64,
    /// Counter for early aborts (metrics).
    abort_count: u64,
    /// Upper bound for value checks.
    upper_bound: f64,
    /// Lower bound for value checks.
    lower_bound: f64,
}

impl Default for PreflightJidoka {
    fn default() -> Self {
        Self::new()
    }
}

impl PreflightJidoka {
    /// Create with default abort conditions.
    #[must_use]
    pub fn new() -> Self {
        Self {
            abort_on: AbortConditions::NON_FINITE | AbortConditions::GRADIENT_EXPLOSION,
            gradient_explosion_threshold: 1e6,
            gradient_vanishing_threshold: 1e-10,
            abort_count: 0,
            upper_bound: 1e12,
            lower_bound: -1e12,
        }
    }

    /// Create with custom abort conditions.
    #[must_use]
    pub const fn with_conditions(conditions: AbortConditions) -> Self {
        Self {
            abort_on: conditions,
            gradient_explosion_threshold: 1e6,
            gradient_vanishing_threshold: 1e-10,
            abort_count: 0,
            upper_bound: 1e12,
            lower_bound: -1e12,
        }
    }

    /// Set gradient explosion threshold.
    #[must_use]
    pub const fn with_explosion_threshold(mut self, threshold: f64) -> Self {
        self.gradient_explosion_threshold = threshold;
        self
    }

    /// Set gradient vanishing threshold.
    #[must_use]
    pub const fn with_vanishing_threshold(mut self, threshold: f64) -> Self {
        self.gradient_vanishing_threshold = threshold;
        self
    }

    /// Set value bounds.
    #[must_use]
    pub const fn with_bounds(mut self, lower: f64, upper: f64) -> Self {
        self.lower_bound = lower;
        self.upper_bound = upper;
        self
    }

    /// Check a single value for anomalies.
    ///
    /// # Errors
    ///
    /// Returns error if value violates abort conditions.
    pub fn check_value(&mut self, value: f64) -> SimResult<()> {
        // Check non-finite
        if self.abort_on.contains(AbortConditions::NON_FINITE) && !value.is_finite() {
            self.abort_count += 1;
            return Err(SimError::jidoka("Pre-flight: Non-finite value detected"));
        }

        // Check bounds
        if self.abort_on.contains(AbortConditions::BOUND_VIOLATION)
            && (value < self.lower_bound || value > self.upper_bound)
        {
            self.abort_count += 1;
            return Err(SimError::jidoka(format!(
                "Pre-flight: Value {value:.2e} outside bounds [{:.2e}, {:.2e}]",
                self.lower_bound, self.upper_bound
            )));
        }

        Ok(())
    }

    /// Check a slice of values for anomalies.
    ///
    /// # Errors
    ///
    /// Returns error if any value violates abort conditions.
    pub fn check_values(&mut self, values: &[f64]) -> SimResult<()> {
        for (i, &v) in values.iter().enumerate() {
            if self.abort_on.contains(AbortConditions::NON_FINITE) && !v.is_finite() {
                self.abort_count += 1;
                return Err(SimError::jidoka(format!(
                    "Pre-flight: Non-finite value at index {i}"
                )));
            }

            if self.abort_on.contains(AbortConditions::BOUND_VIOLATION)
                && (v < self.lower_bound || v > self.upper_bound)
            {
                self.abort_count += 1;
                return Err(SimError::jidoka(format!(
                    "Pre-flight: Value at index {i} ({v:.2e}) outside bounds"
                )));
            }
        }

        Ok(())
    }

    /// Check gradient norm for explosion/vanishing.
    ///
    /// # Errors
    ///
    /// Returns error if gradient is exploding or vanishing.
    pub fn check_gradient_norm(&mut self, norm: f64) -> SimResult<()> {
        if self.abort_on.contains(AbortConditions::NON_FINITE) && !norm.is_finite() {
            self.abort_count += 1;
            return Err(SimError::jidoka("Pre-flight: Non-finite gradient norm"));
        }

        if self.abort_on.contains(AbortConditions::GRADIENT_EXPLOSION)
            && norm > self.gradient_explosion_threshold
        {
            self.abort_count += 1;
            return Err(SimError::jidoka(format!(
                "Pre-flight: Gradient explosion detected (norm={norm:.2e})"
            )));
        }

        if self.abort_on.contains(AbortConditions::GRADIENT_VANISHING)
            && norm < self.gradient_vanishing_threshold
            && norm > 0.0
        {
            self.abort_count += 1;
            return Err(SimError::jidoka(format!(
                "Pre-flight: Gradient vanishing detected (norm={norm:.2e})"
            )));
        }

        Ok(())
    }

    /// Get total abort count.
    #[must_use]
    pub const fn abort_count(&self) -> u64 {
        self.abort_count
    }

    /// Reset abort count.
    pub fn reset_count(&mut self) {
        self.abort_count = 0;
    }
}

// =============================================================================
// Self-Healing Jidoka (Section 4.3.2)
// =============================================================================

/// Jidoka response type: Andon (stop) vs auto-correct [51][57].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JidokaResponse {
    /// Andon: Full stop, human intervention required.
    Andon,
    /// Auto-correct: Apply patch and continue.
    AutoCorrect,
    /// Monitor: Log warning, continue with observation.
    Monitor,
}

/// Training anomaly types for ML simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingAnomaly {
    /// NaN detected in parameters or loss.
    NaN {
        /// Location of NaN.
        location: String,
    },
    /// Model corruption detected.
    ModelCorruption {
        /// Description of corruption.
        description: String,
    },
    /// Loss spike detected.
    LossSpike {
        /// Current loss value.
        current: f64,
        /// Expected loss value.
        expected: f64,
        /// Z-score of spike.
        z_score: f64,
    },
    /// Gradient explosion detected.
    GradientExplosion {
        /// Gradient norm.
        norm: f64,
        /// Threshold that was exceeded.
        threshold: f64,
    },
    /// Gradient vanishing detected.
    GradientVanishing {
        /// Gradient norm.
        norm: f64,
        /// Threshold that was not met.
        threshold: f64,
    },
    /// Slow convergence detected.
    SlowConvergence {
        /// Recent loss values.
        recent_losses: Vec<f64>,
        /// Expected improvement rate.
        expected_rate: f64,
    },
    /// High variance in loss.
    HighVariance {
        /// Variance value.
        variance: f64,
        /// Threshold.
        threshold: f64,
    },
}

/// Corrective patch for self-healing [57].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RulePatch {
    /// Reduce learning rate.
    ReduceLearningRate {
        /// Factor to reduce by.
        factor: f64,
    },
    /// Enable gradient clipping.
    EnableGradientClipping {
        /// Max gradient norm.
        max_norm: f64,
    },
    /// Increase batch size.
    IncreaseBatchSize {
        /// Factor to increase by.
        factor: usize,
    },
    /// Enable learning rate warmup.
    EnableWarmup {
        /// Warmup steps.
        steps: usize,
    },
    /// Skip batch.
    SkipBatch,
    /// Rollback to checkpoint.
    Rollback {
        /// Number of steps to rollback.
        steps: u64,
    },
}

/// Self-healing Jidoka controller for ML training [51][57].
///
/// Distinguishes between Andon (full stop) and auto-correction based on
/// anomaly severity and correction history.
#[derive(Debug, Clone)]
pub struct SelfHealingJidoka {
    /// Maximum auto-corrections before escalating to Andon.
    max_auto_corrections: usize,
    /// Current correction count.
    correction_count: usize,
    /// Correction count per anomaly type.
    corrections_by_type: std::collections::HashMap<String, usize>,
    /// Applied patches history.
    applied_patches: Vec<RulePatch>,
    /// Maximum patches of same type before escalation.
    max_same_type_corrections: usize,
}

impl Default for SelfHealingJidoka {
    fn default() -> Self {
        Self::new(10)
    }
}

impl SelfHealingJidoka {
    /// Create with maximum auto-correction limit.
    #[must_use]
    pub fn new(max_auto_corrections: usize) -> Self {
        Self {
            max_auto_corrections,
            correction_count: 0,
            corrections_by_type: std::collections::HashMap::new(),
            applied_patches: Vec::new(),
            max_same_type_corrections: 3,
        }
    }

    /// Set maximum corrections of same type before escalation.
    #[must_use]
    pub const fn with_max_same_type(mut self, max: usize) -> Self {
        self.max_same_type_corrections = max;
        self
    }

    /// Determine response based on anomaly type and history.
    #[must_use]
    pub fn classify_response(&self, anomaly: &TrainingAnomaly) -> JidokaResponse {
        let anomaly_type = self.anomaly_type_key(anomaly);

        // Check type-specific correction count
        let type_count = self
            .corrections_by_type
            .get(&anomaly_type)
            .copied()
            .unwrap_or(0);
        if type_count >= self.max_same_type_corrections {
            return JidokaResponse::Andon;
        }

        match anomaly {
            // Fatal: Always Andon
            TrainingAnomaly::NaN { .. } | TrainingAnomaly::ModelCorruption { .. } => {
                JidokaResponse::Andon
            }

            // Recoverable: Auto-correct if under threshold
            TrainingAnomaly::LossSpike { z_score, .. } => {
                if *z_score > 5.0 || self.correction_count >= self.max_auto_corrections {
                    JidokaResponse::Andon
                } else {
                    JidokaResponse::AutoCorrect
                }
            }

            TrainingAnomaly::GradientExplosion { .. }
            | TrainingAnomaly::GradientVanishing { .. } => {
                if self.correction_count < self.max_auto_corrections {
                    JidokaResponse::AutoCorrect
                } else {
                    JidokaResponse::Andon
                }
            }

            // Minor: Monitor only
            TrainingAnomaly::SlowConvergence { .. } | TrainingAnomaly::HighVariance { .. } => {
                JidokaResponse::Monitor
            }
        }
    }

    /// Generate corrective patch for anomaly.
    #[must_use]
    pub fn generate_patch(&self, anomaly: &TrainingAnomaly) -> Option<RulePatch> {
        match anomaly {
            TrainingAnomaly::LossSpike { z_score, .. } => {
                if *z_score > 3.0 {
                    Some(RulePatch::SkipBatch)
                } else {
                    Some(RulePatch::ReduceLearningRate { factor: 0.5 })
                }
            }

            TrainingAnomaly::GradientExplosion { norm, .. } => {
                Some(RulePatch::EnableGradientClipping {
                    max_norm: norm / 10.0,
                })
            }

            TrainingAnomaly::GradientVanishing { .. } => {
                Some(RulePatch::ReduceLearningRate { factor: 2.0 }) // Increase LR
            }

            TrainingAnomaly::SlowConvergence { .. } => {
                Some(RulePatch::EnableWarmup { steps: 1000 })
            }

            TrainingAnomaly::HighVariance { .. } => {
                Some(RulePatch::IncreaseBatchSize { factor: 2 })
            }

            TrainingAnomaly::NaN { .. } | TrainingAnomaly::ModelCorruption { .. } => {
                Some(RulePatch::Rollback { steps: 100 })
            }
        }
    }

    /// Record that a correction was applied.
    pub fn record_correction(&mut self, anomaly: &TrainingAnomaly, patch: RulePatch) {
        let anomaly_type = self.anomaly_type_key(anomaly);
        *self.corrections_by_type.entry(anomaly_type).or_insert(0) += 1;
        self.correction_count += 1;
        self.applied_patches.push(patch);
    }

    /// Get total correction count.
    #[must_use]
    pub const fn correction_count(&self) -> usize {
        self.correction_count
    }

    /// Get applied patches.
    #[must_use]
    pub fn applied_patches(&self) -> &[RulePatch] {
        &self.applied_patches
    }

    /// Reset correction history.
    pub fn reset(&mut self) {
        self.correction_count = 0;
        self.corrections_by_type.clear();
        self.applied_patches.clear();
    }

    /// Get string key for anomaly type.
    #[allow(clippy::unused_self)]
    fn anomaly_type_key(&self, anomaly: &TrainingAnomaly) -> String {
        match anomaly {
            TrainingAnomaly::NaN { .. } => "nan".to_string(),
            TrainingAnomaly::ModelCorruption { .. } => "corruption".to_string(),
            TrainingAnomaly::LossSpike { .. } => "loss_spike".to_string(),
            TrainingAnomaly::GradientExplosion { .. } => "grad_explosion".to_string(),
            TrainingAnomaly::GradientVanishing { .. } => "grad_vanishing".to_string(),
            TrainingAnomaly::SlowConvergence { .. } => "slow_convergence".to_string(),
            TrainingAnomaly::HighVariance { .. } => "high_variance".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::state::Vec3;

    #[test]
    fn test_finite_check_passes_valid_state() {
        let mut guard = JidokaGuard::new(JidokaConfig::default());
        let state = SimState::default();

        assert!(guard.check(&state).is_ok());
    }

    #[test]
    fn test_finite_check_catches_nan() {
        let mut guard = JidokaGuard::new(JidokaConfig::default());
        let mut state = SimState::default();

        // Add a body with NaN position
        state.add_body(1.0, Vec3::new(f64::NAN, 0.0, 0.0), Vec3::zero());

        let result = guard.check(&state);
        assert!(result.is_err());

        if let Err(SimError::NonFiniteValue { location }) = result {
            assert!(location.contains("positions"));
        } else {
            panic!("Expected NonFiniteValue error");
        }
    }

    #[test]
    fn test_finite_check_catches_infinity() {
        let mut guard = JidokaGuard::new(JidokaConfig::default());
        let mut state = SimState::default();

        state.add_body(1.0, Vec3::zero(), Vec3::new(0.0, f64::INFINITY, 0.0));

        let result = guard.check(&state);
        assert!(result.is_err());

        if let Err(SimError::NonFiniteValue { location }) = result {
            assert!(location.contains("velocities"));
        } else {
            panic!("Expected NonFiniteValue error");
        }
    }

    #[test]
    fn test_energy_drift_detection() {
        let config = JidokaConfig {
            energy_tolerance: 0.01,
            check_energy: true,
            ..Default::default()
        };
        let mut guard = JidokaGuard::new(config);

        // Initial state with some energy
        let mut state = SimState::default();
        state.add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));

        // First check records initial energy
        assert!(guard.check(&state).is_ok());

        // Modify state to have significantly different energy
        state.set_velocity(0, Vec3::new(10.0, 0.0, 0.0)); // 100x kinetic energy

        let result = guard.check(&state);
        assert!(result.is_err());
        assert!(matches!(result, Err(SimError::EnergyDrift { .. })));
    }

    #[test]
    fn test_constraint_violation_detection() {
        let config = JidokaConfig {
            constraint_tolerance: 0.001,
            ..Default::default()
        };
        let mut guard = JidokaGuard::new(config);

        let mut state = SimState::default();
        state.add_constraint("test_constraint", 0.01); // Violation > tolerance

        let result = guard.check(&state);
        assert!(result.is_err());
        assert!(matches!(result, Err(SimError::ConstraintViolation { .. })));
    }

    #[test]
    fn test_guard_reset() {
        let mut guard = JidokaGuard::new(JidokaConfig::default());
        let mut state = SimState::default();

        // Add a body with some energy so initial_energy gets recorded
        state.add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        state.set_potential_energy(1.0);

        // Record initial energy - needs non-zero energy to record
        guard.check(&state).ok();
        assert!(
            guard.initial_energy.is_some(),
            "Initial energy should be recorded for non-zero energy state"
        );

        // Reset
        guard.reset();
        assert!(guard.initial_energy.is_none());
    }

    #[test]
    fn test_disabled_checks() {
        let config = JidokaConfig {
            check_finite: false,
            check_energy: false,
            ..Default::default()
        };
        let mut guard = JidokaGuard::new(config);

        let mut state = SimState::default();
        state.add_body(1.0, Vec3::new(f64::NAN, 0.0, 0.0), Vec3::zero());

        // Should pass because finite check is disabled
        assert!(guard.check(&state).is_ok());
    }

    // === Severity Classifier Tests ===

    #[test]
    fn test_severity_classifier_acceptable() {
        let classifier = SeverityClassifier::new(0.8);

        // 50% of tolerance = Acceptable
        let severity = classifier.classify_energy_drift(0.5, 1.0);
        assert_eq!(severity, ViolationSeverity::Acceptable);

        // 79% of tolerance = Acceptable (just under warning threshold)
        let severity = classifier.classify_energy_drift(0.79, 1.0);
        assert_eq!(severity, ViolationSeverity::Acceptable);
    }

    #[test]
    fn test_severity_classifier_warning() {
        let classifier = SeverityClassifier::new(0.8);

        // Just above 80% of tolerance = Warning (boundary is > not >=)
        let severity = classifier.classify_energy_drift(0.81, 1.0);
        assert_eq!(severity, ViolationSeverity::Warning);

        // 99% of tolerance = Warning
        let severity = classifier.classify_energy_drift(0.99, 1.0);
        assert_eq!(severity, ViolationSeverity::Warning);

        // Exactly at 80% boundary is still Acceptable
        let severity = classifier.classify_energy_drift(0.8, 1.0);
        assert_eq!(severity, ViolationSeverity::Acceptable);
    }

    #[test]
    fn test_severity_classifier_critical() {
        let classifier = SeverityClassifier::new(0.8);

        // 100% of tolerance = Critical (exactly at)
        let severity = classifier.classify_energy_drift(1.0, 1.0);
        assert_eq!(severity, ViolationSeverity::Warning); // At boundary, not over

        // 101% of tolerance = Critical (over)
        let severity = classifier.classify_energy_drift(1.01, 1.0);
        assert_eq!(severity, ViolationSeverity::Critical);

        // 200% of tolerance = Critical
        let severity = classifier.classify_energy_drift(2.0, 1.0);
        assert_eq!(severity, ViolationSeverity::Critical);
    }

    #[test]
    fn test_severity_classifier_fatal() {
        let classifier = SeverityClassifier::new(0.8);

        // NaN = Fatal
        let severity = classifier.classify_energy_drift(f64::NAN, 1.0);
        assert_eq!(severity, ViolationSeverity::Fatal);

        // Infinity = Fatal
        let severity = classifier.classify_energy_drift(f64::INFINITY, 1.0);
        assert_eq!(severity, ViolationSeverity::Fatal);

        // Negative Infinity = Fatal
        let severity = classifier.classify_energy_drift(f64::NEG_INFINITY, 1.0);
        assert_eq!(severity, ViolationSeverity::Fatal);
    }

    #[test]
    fn test_severity_classifier_constraint() {
        let classifier = SeverityClassifier::new(0.8);

        // Test positive violation
        assert_eq!(
            classifier.classify_constraint(0.5, 1.0),
            ViolationSeverity::Acceptable
        );
        assert_eq!(
            classifier.classify_constraint(0.85, 1.0),
            ViolationSeverity::Warning
        );
        assert_eq!(
            classifier.classify_constraint(1.5, 1.0),
            ViolationSeverity::Critical
        );

        // Test negative violation (abs applied)
        assert_eq!(
            classifier.classify_constraint(-0.5, 1.0),
            ViolationSeverity::Acceptable
        );
        assert_eq!(
            classifier.classify_constraint(-0.85, 1.0),
            ViolationSeverity::Warning
        );
        assert_eq!(
            classifier.classify_constraint(-1.5, 1.0),
            ViolationSeverity::Critical
        );
    }

    #[test]
    fn test_severity_classifier_default() {
        let classifier = SeverityClassifier::default();
        assert!((classifier.warning_fraction - 0.8).abs() < f64::EPSILON);
    }

    // === Check With Warnings Tests ===

    #[test]
    fn test_check_with_warnings_no_warnings() {
        let mut guard = JidokaGuard::new(JidokaConfig::default());
        let state = SimState::default();

        let result = guard.check_with_warnings(&state);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_check_with_warnings_energy_warning() {
        let config = JidokaConfig {
            energy_tolerance: 1.0,
            check_energy: true,
            severity_classifier: SeverityClassifier::new(0.8),
            ..Default::default()
        };
        let mut guard = JidokaGuard::new(config);

        // Initial state
        let mut state = SimState::default();
        state.add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        state.set_potential_energy(10.0);

        // First check records energy
        let _ = guard.check_with_warnings(&state);

        // Modify to have ~85% drift (in warning zone)
        // Initial energy ~= 10.5 (10 potential + 0.5 kinetic)
        // For 85% drift we need ~19.4 total energy
        state.set_potential_energy(18.9);

        let result = guard.check_with_warnings(&state);
        assert!(result.is_ok());
        let warnings = result.unwrap();
        assert!(!warnings.is_empty(), "Should have energy drift warning");

        match &warnings[0] {
            JidokaWarning::EnergyDriftApproaching { drift, .. } => {
                assert!(*drift > 0.8, "Drift should be > 80%");
                assert!(*drift <= 1.0, "Drift should be <= 100%");
            }
            JidokaWarning::ConstraintApproaching { .. } => {
                panic!("Expected EnergyDriftApproaching warning")
            }
        }
    }

    #[test]
    fn test_check_with_warnings_constraint_warning() {
        let config = JidokaConfig {
            constraint_tolerance: 1.0,
            severity_classifier: SeverityClassifier::new(0.8),
            check_energy: false, // Disable energy to isolate constraint test
            ..Default::default()
        };
        let mut guard = JidokaGuard::new(config);

        let mut state = SimState::default();
        state.add_constraint("test", 0.9); // 90% of tolerance = warning

        let result = guard.check_with_warnings(&state);
        assert!(result.is_ok());
        let warnings = result.unwrap();
        assert!(!warnings.is_empty(), "Should have constraint warning");

        match &warnings[0] {
            JidokaWarning::ConstraintApproaching {
                name, violation, ..
            } => {
                assert_eq!(name, "test");
                assert!((*violation - 0.9).abs() < f64::EPSILON);
            }
            JidokaWarning::EnergyDriftApproaching { .. } => {
                panic!("Expected ConstraintApproaching warning")
            }
        }
    }

    #[test]
    fn test_check_with_warnings_critical_error() {
        let config = JidokaConfig {
            constraint_tolerance: 1.0,
            severity_classifier: SeverityClassifier::new(0.8),
            check_energy: false,
            ..Default::default()
        };
        let mut guard = JidokaGuard::new(config);

        let mut state = SimState::default();
        state.add_constraint("critical", 1.5); // 150% = critical

        let result = guard.check_with_warnings(&state);
        assert!(result.is_err());
        assert!(matches!(result, Err(SimError::ConstraintViolation { .. })));
    }

    #[test]
    fn test_check_with_warnings_fatal_nan() {
        let mut guard = JidokaGuard::new(JidokaConfig::default());
        let mut state = SimState::default();
        state.add_body(1.0, Vec3::new(f64::NAN, 0.0, 0.0), Vec3::zero());

        let result = guard.check_with_warnings(&state);
        assert!(result.is_err());
        assert!(matches!(result, Err(SimError::NonFiniteValue { .. })));
    }

    #[test]
    fn test_violation_severity_ordering() {
        // Verify ordering: Acceptable < Warning < Critical < Fatal
        assert!(ViolationSeverity::Acceptable < ViolationSeverity::Warning);
        assert!(ViolationSeverity::Warning < ViolationSeverity::Critical);
        assert!(ViolationSeverity::Critical < ViolationSeverity::Fatal);
    }

    // === Pre-flight Jidoka Tests (Section 4.3.1) ===

    #[test]
    fn test_preflight_check_value_valid() {
        let mut preflight = PreflightJidoka::new();
        assert!(preflight.check_value(1.0).is_ok());
        assert!(preflight.check_value(-1.0).is_ok());
        assert!(preflight.check_value(0.0).is_ok());
        assert_eq!(preflight.abort_count(), 0);
    }

    #[test]
    fn test_preflight_check_value_nan() {
        let mut preflight = PreflightJidoka::new();
        assert!(preflight.check_value(f64::NAN).is_err());
        assert_eq!(preflight.abort_count(), 1);
    }

    #[test]
    fn test_preflight_check_value_infinity() {
        let mut preflight = PreflightJidoka::new();
        assert!(preflight.check_value(f64::INFINITY).is_err());
        assert_eq!(preflight.abort_count(), 1);

        assert!(preflight.check_value(f64::NEG_INFINITY).is_err());
        assert_eq!(preflight.abort_count(), 2);
    }

    #[test]
    fn test_preflight_check_values() {
        let mut preflight = PreflightJidoka::new();
        let values = vec![1.0, 2.0, 3.0, 4.0];
        assert!(preflight.check_values(&values).is_ok());

        let values_with_nan = vec![1.0, 2.0, f64::NAN, 4.0];
        assert!(preflight.check_values(&values_with_nan).is_err());
    }

    #[test]
    fn test_preflight_gradient_explosion() {
        let mut preflight = PreflightJidoka::new().with_explosion_threshold(100.0);

        assert!(preflight.check_gradient_norm(50.0).is_ok());
        assert!(preflight.check_gradient_norm(150.0).is_err());
        assert_eq!(preflight.abort_count(), 1);
    }

    #[test]
    fn test_preflight_gradient_vanishing() {
        let mut preflight = PreflightJidoka::with_conditions(
            AbortConditions::NON_FINITE | AbortConditions::GRADIENT_VANISHING,
        )
        .with_vanishing_threshold(1e-8);

        assert!(preflight.check_gradient_norm(1e-6).is_ok()); // Above threshold
        assert!(preflight.check_gradient_norm(1e-10).is_err()); // Below threshold
        assert!(preflight.check_gradient_norm(0.0).is_ok()); // Zero is ok (not > 0)
    }

    #[test]
    fn test_preflight_bounds() {
        let mut preflight = PreflightJidoka::with_conditions(AbortConditions::BOUND_VIOLATION)
            .with_bounds(-100.0, 100.0);

        assert!(preflight.check_value(50.0).is_ok());
        assert!(preflight.check_value(-50.0).is_ok());
        assert!(preflight.check_value(150.0).is_err());
        assert!(preflight.check_value(-150.0).is_err());
    }

    #[test]
    fn test_preflight_reset_count() {
        let mut preflight = PreflightJidoka::new();
        let _ = preflight.check_value(f64::NAN);
        assert_eq!(preflight.abort_count(), 1);

        preflight.reset_count();
        assert_eq!(preflight.abort_count(), 0);
    }

    // === Self-Healing Jidoka Tests (Section 4.3.2) ===

    #[test]
    fn test_self_healing_nan_always_andon() {
        let healer = SelfHealingJidoka::new(10);
        let anomaly = TrainingAnomaly::NaN {
            location: "loss".to_string(),
        };
        assert_eq!(healer.classify_response(&anomaly), JidokaResponse::Andon);
    }

    #[test]
    fn test_self_healing_corruption_always_andon() {
        let healer = SelfHealingJidoka::new(10);
        let anomaly = TrainingAnomaly::ModelCorruption {
            description: "CRC mismatch".to_string(),
        };
        assert_eq!(healer.classify_response(&anomaly), JidokaResponse::Andon);
    }

    #[test]
    fn test_self_healing_loss_spike_auto_correct() {
        let healer = SelfHealingJidoka::new(10);
        let anomaly = TrainingAnomaly::LossSpike {
            current: 10.0,
            expected: 1.0,
            z_score: 3.0,
        };
        assert_eq!(
            healer.classify_response(&anomaly),
            JidokaResponse::AutoCorrect
        );
    }

    #[test]
    fn test_self_healing_extreme_loss_spike_andon() {
        let healer = SelfHealingJidoka::new(10);
        let anomaly = TrainingAnomaly::LossSpike {
            current: 100.0,
            expected: 1.0,
            z_score: 6.0, // > 5.0 threshold
        };
        assert_eq!(healer.classify_response(&anomaly), JidokaResponse::Andon);
    }

    #[test]
    fn test_self_healing_gradient_explosion_auto_correct() {
        let healer = SelfHealingJidoka::new(10);
        let anomaly = TrainingAnomaly::GradientExplosion {
            norm: 1e7,
            threshold: 1e6,
        };
        assert_eq!(
            healer.classify_response(&anomaly),
            JidokaResponse::AutoCorrect
        );
    }

    #[test]
    fn test_self_healing_slow_convergence_monitor() {
        let healer = SelfHealingJidoka::new(10);
        let anomaly = TrainingAnomaly::SlowConvergence {
            recent_losses: vec![1.0, 0.99, 0.98],
            expected_rate: 0.1,
        };
        assert_eq!(healer.classify_response(&anomaly), JidokaResponse::Monitor);
    }

    #[test]
    fn test_self_healing_high_variance_monitor() {
        let healer = SelfHealingJidoka::new(10);
        let anomaly = TrainingAnomaly::HighVariance {
            variance: 0.5,
            threshold: 0.1,
        };
        assert_eq!(healer.classify_response(&anomaly), JidokaResponse::Monitor);
    }

    #[test]
    fn test_self_healing_escalation_after_max_corrections() {
        let mut healer = SelfHealingJidoka::new(2);

        let anomaly = TrainingAnomaly::GradientExplosion {
            norm: 1e7,
            threshold: 1e6,
        };

        // First two should auto-correct
        assert_eq!(
            healer.classify_response(&anomaly),
            JidokaResponse::AutoCorrect
        );
        let patch = healer.generate_patch(&anomaly).unwrap();
        healer.record_correction(&anomaly, patch);

        assert_eq!(
            healer.classify_response(&anomaly),
            JidokaResponse::AutoCorrect
        );
        let patch = healer.generate_patch(&anomaly).unwrap();
        healer.record_correction(&anomaly, patch);

        // Third should escalate to Andon
        assert_eq!(healer.classify_response(&anomaly), JidokaResponse::Andon);
    }

    #[test]
    fn test_self_healing_generate_patch() {
        let healer = SelfHealingJidoka::new(10);

        // Loss spike with high z-score -> skip batch
        let anomaly = TrainingAnomaly::LossSpike {
            current: 10.0,
            expected: 1.0,
            z_score: 4.0,
        };
        assert!(matches!(
            healer.generate_patch(&anomaly),
            Some(RulePatch::SkipBatch)
        ));

        // Gradient explosion -> gradient clipping
        let anomaly = TrainingAnomaly::GradientExplosion {
            norm: 1e7,
            threshold: 1e6,
        };
        assert!(matches!(
            healer.generate_patch(&anomaly),
            Some(RulePatch::EnableGradientClipping { .. })
        ));

        // Slow convergence -> warmup
        let anomaly = TrainingAnomaly::SlowConvergence {
            recent_losses: vec![],
            expected_rate: 0.1,
        };
        assert!(matches!(
            healer.generate_patch(&anomaly),
            Some(RulePatch::EnableWarmup { .. })
        ));
    }

    #[test]
    fn test_self_healing_reset() {
        let mut healer = SelfHealingJidoka::new(10);

        let anomaly = TrainingAnomaly::GradientExplosion {
            norm: 1e7,
            threshold: 1e6,
        };
        let patch = healer.generate_patch(&anomaly).unwrap();
        healer.record_correction(&anomaly, patch);

        assert_eq!(healer.correction_count(), 1);
        assert!(!healer.applied_patches().is_empty());

        healer.reset();

        assert_eq!(healer.correction_count(), 0);
        assert!(healer.applied_patches().is_empty());
    }

    #[test]
    fn test_self_healing_type_specific_escalation() {
        let mut healer = SelfHealingJidoka::new(100).with_max_same_type(2);

        let explosion = TrainingAnomaly::GradientExplosion {
            norm: 1e7,
            threshold: 1e6,
        };
        let spike = TrainingAnomaly::LossSpike {
            current: 10.0,
            expected: 1.0,
            z_score: 3.0,
        };

        // Record 2 gradient explosions
        for _ in 0..2 {
            let patch = healer.generate_patch(&explosion).unwrap();
            healer.record_correction(&explosion, patch);
        }

        // Third gradient explosion should be Andon (type limit exceeded)
        assert_eq!(healer.classify_response(&explosion), JidokaResponse::Andon);

        // But loss spike should still be AutoCorrect (different type)
        assert_eq!(
            healer.classify_response(&spike),
            JidokaResponse::AutoCorrect
        );
    }

    // === Clone and Debug Tests ===

    #[test]
    fn test_violation_severity_clone_debug() {
        let severity = ViolationSeverity::Warning;
        let cloned = severity.clone();
        assert_eq!(cloned, ViolationSeverity::Warning);

        let debug = format!("{:?}", severity);
        assert!(debug.contains("Warning"));
    }

    #[test]
    fn test_jidoka_warning_clone_debug() {
        let warning = JidokaWarning::EnergyDriftApproaching {
            drift: 0.9,
            tolerance: 1.0,
        };
        let cloned = warning.clone();
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("EnergyDriftApproaching"));

        let warning2 = JidokaWarning::ConstraintApproaching {
            name: "test".to_string(),
            violation: 0.5,
            tolerance: 1.0,
        };
        let debug2 = format!("{:?}", warning2);
        assert!(debug2.contains("ConstraintApproaching"));
    }

    #[test]
    fn test_severity_classifier_clone_debug() {
        let classifier = SeverityClassifier::new(0.85);
        let cloned = classifier.clone();
        assert!((cloned.warning_fraction - 0.85).abs() < f64::EPSILON);

        let debug = format!("{:?}", classifier);
        assert!(debug.contains("SeverityClassifier"));
    }

    #[test]
    fn test_jidoka_config_debug() {
        let config = JidokaConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("JidokaConfig"));
    }

    #[test]
    fn test_jidoka_guard_debug() {
        let guard = JidokaGuard::new(JidokaConfig::default());
        let debug = format!("{:?}", guard);
        assert!(debug.contains("JidokaGuard"));
    }

    #[test]
    fn test_violation_severity_ord_impl() {
        assert!(ViolationSeverity::Acceptable < ViolationSeverity::Warning);
        assert!(ViolationSeverity::Warning < ViolationSeverity::Critical);
        assert!(ViolationSeverity::Critical < ViolationSeverity::Fatal);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::engine::state::Vec3;
    use proptest::prelude::*;

    proptest! {
        /// Falsification: valid states should always pass.
        #[test]
        fn prop_valid_state_passes(
            x in -1e6f64..1e6,
            y in -1e6f64..1e6,
            z in -1e6f64..1e6,
            vx in -1e3f64..1e3,
            vy in -1e3f64..1e3,
            vz in -1e3f64..1e3,
            mass in 0.1f64..1e6,
        ) {
            let mut guard = JidokaGuard::new(JidokaConfig::default());
            let mut state = SimState::default();

            state.add_body(mass, Vec3::new(x, y, z), Vec3::new(vx, vy, vz));

            // All finite values should pass
            prop_assert!(guard.check(&state).is_ok());
        }

        /// Falsification: severity levels are monotonic with drift.
        #[test]
        fn prop_severity_monotonic(
            tolerance in 0.001f64..100.0,
            warning_fraction in 0.5f64..0.99,
        ) {
            let classifier = SeverityClassifier::new(warning_fraction);

            // Values below warning threshold
            let below_warning = tolerance * warning_fraction * 0.5;
            let at_warning = tolerance * warning_fraction;
            let above_tolerance = tolerance * 1.5;

            let sev_below = classifier.classify_energy_drift(below_warning, tolerance);
            let sev_at = classifier.classify_energy_drift(at_warning, tolerance);
            let sev_above = classifier.classify_energy_drift(above_tolerance, tolerance);

            // Monotonic: below_warning <= at_warning <= above_tolerance
            prop_assert!(sev_below <= sev_at);
            prop_assert!(sev_at <= sev_above);
        }

        /// Falsification: acceptable never exceeds warning threshold.
        #[test]
        fn prop_acceptable_boundary(
            tolerance in 0.001f64..100.0,
            warning_fraction in 0.5f64..0.99,
            drift_fraction in 0.0f64..0.99,
        ) {
            let classifier = SeverityClassifier::new(warning_fraction);
            let drift = tolerance * warning_fraction * drift_fraction;

            let severity = classifier.classify_energy_drift(drift, tolerance);
            prop_assert_eq!(severity, ViolationSeverity::Acceptable);
        }

        /// Falsification: critical always exceeds tolerance.
        #[test]
        fn prop_critical_boundary(
            tolerance in 0.001f64..100.0,
            excess_factor in 1.01f64..10.0,
        ) {
            let classifier = SeverityClassifier::default();
            let drift = tolerance * excess_factor;

            let severity = classifier.classify_energy_drift(drift, tolerance);
            prop_assert_eq!(severity, ViolationSeverity::Critical);
        }

        /// Falsification: constraint classification handles negative values.
        #[test]
        fn prop_constraint_abs_symmetry(
            violation in 0.001f64..100.0,
            tolerance in 0.01f64..100.0,
        ) {
            let classifier = SeverityClassifier::default();

            let pos_severity = classifier.classify_constraint(violation, tolerance);
            let neg_severity = classifier.classify_constraint(-violation, tolerance);

            // Absolute value applied, so both should be equal
            prop_assert_eq!(pos_severity, neg_severity);
        }
    }
}
