//! EDD Validator - Compliance checking for Equation-Driven Development.
//!
//! The validator ensures all simulations and experiments comply with the
//! four pillars of EDD:
//!
//! 1. **Prove It**: Simulation must have a governing equation (EMC)
//! 2. **Fail It**: Simulation must have failing tests derived from equations
//! 3. **Seed It**: Experiment must have explicit seed
//! 4. **Falsify It**: Experiment must have falsification criteria
//!
//! # EDD Violation Codes
//!
//! - EDD-01: Missing Equation Model Card
//! - EDD-02: Missing governing equation
//! - EDD-03: Missing citation
//! - EDD-04: Missing verification tests
//! - EDD-05: Missing explicit seed
//! - EDD-06: Missing falsification criteria
//! - EDD-07: Verification test failed
//! - EDD-08: Conservation laws violated (runtime monitoring)
//! - EDD-09: Cross-platform reproducibility failed
//! - EDD-10: Implementation without failing test (TDD violation)

use super::experiment::ExperimentSpec;
use super::model_card::EquationModelCard;

/// EDD violation codes and messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EddViolation {
    /// Violation code (e.g., "EDD-01")
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Context or additional details
    pub context: Option<String>,
}

/// Severity levels for violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    /// Informational - suggestion for improvement
    Info,
    /// Warning - should be addressed
    Warning,
    /// Error - must be fixed
    Error,
    /// Critical - halts execution (Jidoka)
    Critical,
}

impl EddViolation {
    /// Create a new violation.
    #[must_use]
    pub fn new(code: &str, message: &str, severity: ViolationSeverity) -> Self {
        Self {
            code: code.to_string(),
            message: message.to_string(),
            severity,
            context: None,
        }
    }

    /// Add context to the violation.
    #[must_use]
    pub fn with_context(mut self, context: &str) -> Self {
        self.context = Some(context.to_string());
        self
    }
}

/// Result type for EDD validation.
pub type EddResult<T> = Result<T, EddViolation>;

/// EDD Validator for checking compliance with the four pillars.
#[derive(Debug, Default)]
pub struct EddValidator {
    /// Collected violations
    violations: Vec<EddViolation>,
    /// Whether to halt on first critical error (Jidoka mode)
    #[allow(dead_code)]
    jidoka_mode: bool,
}

impl EddValidator {
    /// Create a new EDD validator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
            jidoka_mode: true,
        }
    }

    /// Create a validator in lenient mode (collects all violations).
    #[must_use]
    pub fn lenient() -> Self {
        Self {
            violations: Vec::new(),
            jidoka_mode: false,
        }
    }

    /// Clear all collected violations.
    pub fn clear(&mut self) {
        self.violations.clear();
    }

    /// Get collected violations.
    #[must_use]
    pub fn violations(&self) -> &[EddViolation] {
        &self.violations
    }

    /// Check if any critical violations exist.
    #[must_use]
    pub fn has_critical_violations(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Critical)
    }

    /// Check if any errors exist (including critical).
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity >= ViolationSeverity::Error)
    }

    // =========================================================================
    // Pillar 1: Prove It - EMC Validation
    // =========================================================================

    /// Validate that a simulation has an Equation Model Card.
    ///
    /// # EDD-01: Missing Equation Model Card
    ///
    /// # Errors
    /// Returns `EDD-01` violation if EMC is `None`.
    pub fn validate_simulation_has_emc(&self, emc: Option<&EquationModelCard>) -> EddResult<()> {
        match emc {
            Some(_) => Ok(()),
            None => Err(EddViolation::new(
                "EDD-01",
                "Simulation must have an Equation Model Card (Pillar 1: Prove It)",
                ViolationSeverity::Critical,
            )),
        }
    }

    /// Validate that an EMC has a governing equation.
    ///
    /// # EDD-02: Missing governing equation
    ///
    /// # Errors
    /// Returns `EDD-02` violation if equation is empty.
    pub fn validate_emc_has_equation(&self, emc: &EquationModelCard) -> EddResult<()> {
        if emc.equation.is_empty() {
            Err(EddViolation::new(
                "EDD-02",
                "EMC must have a governing equation in LaTeX format",
                ViolationSeverity::Critical,
            ))
        } else {
            Ok(())
        }
    }

    /// Validate that an EMC has a citation.
    ///
    /// # EDD-03: Missing citation
    ///
    /// # Errors
    /// Returns `EDD-03` violation if citation has no authors.
    pub fn validate_emc_has_citation(&self, emc: &EquationModelCard) -> EddResult<()> {
        if emc.citation.authors.is_empty() {
            Err(EddViolation::new(
                "EDD-03",
                "EMC must have a peer-reviewed citation",
                ViolationSeverity::Critical,
            ))
        } else {
            Ok(())
        }
    }

    /// Validate that an EMC has verification tests.
    ///
    /// # EDD-04: Missing verification tests
    ///
    /// # Errors
    /// Returns `EDD-04` violation if verification tests are empty.
    pub fn validate_emc_has_tests(&self, emc: &EquationModelCard) -> EddResult<()> {
        if emc.verification_tests.is_empty() {
            Err(EddViolation::new(
                "EDD-04",
                "EMC must have at least one verification test (Pillar 2: Fail It)",
                ViolationSeverity::Critical,
            ))
        } else {
            Ok(())
        }
    }

    /// Perform full EMC validation.
    ///
    /// # Errors
    /// Returns list of violations if any validation checks fail.
    pub fn validate_emc(&mut self, emc: &EquationModelCard) -> Result<(), Vec<EddViolation>> {
        let mut violations = Vec::new();

        if let Err(v) = self.validate_emc_has_equation(emc) {
            violations.push(v);
        }

        if let Err(v) = self.validate_emc_has_citation(emc) {
            violations.push(v);
        }

        if let Err(v) = self.validate_emc_has_tests(emc) {
            violations.push(v);
        }

        if violations.is_empty() {
            Ok(())
        } else {
            self.violations.extend(violations.clone());
            Err(violations)
        }
    }

    // =========================================================================
    // Pillar 3: Seed It - Reproducibility Validation
    // =========================================================================

    /// Validate that a seed is explicitly specified.
    ///
    /// # EDD-05: Missing explicit seed
    ///
    /// # Errors
    /// Returns `EDD-05` violation if seed is `None`.
    pub fn validate_seed_specified(&self, seed: Option<u64>) -> EddResult<()> {
        match seed {
            Some(_) => Ok(()),
            None => Err(EddViolation::new(
                "EDD-05",
                "Experiment must have an explicit seed (Pillar 3: Seed It)",
                ViolationSeverity::Critical,
            )),
        }
    }

    // =========================================================================
    // Pillar 4: Falsify It - Falsification Validation
    // =========================================================================

    /// Validate that an experiment has falsification criteria.
    ///
    /// # EDD-06: Missing falsification criteria
    ///
    /// # Errors
    /// Returns `EDD-06` warning if falsification criteria are empty.
    pub fn validate_has_falsification_criteria(&self, spec: &ExperimentSpec) -> EddResult<()> {
        if spec.falsification_criteria().is_empty() {
            // This is a warning, not critical - experiments can still run
            Err(EddViolation::new(
                "EDD-06",
                "Experiment should have falsification criteria (Pillar 4: Falsify It)",
                ViolationSeverity::Warning,
            ))
        } else {
            Ok(())
        }
    }

    // =========================================================================
    // Full Experiment Validation
    // =========================================================================

    /// Validate an experiment specification fully.
    ///
    /// # Errors
    /// Returns list of violations if any validation checks fail (error severity or higher).
    pub fn validate_experiment(&mut self, spec: &ExperimentSpec) -> Result<(), Vec<EddViolation>> {
        let mut violations = Vec::new();

        // Pillar 3: Seed It
        if let Err(v) = self.validate_seed_specified(Some(spec.seed())) {
            violations.push(v);
        }

        // Pillar 4: Falsify It (warning only)
        if let Err(v) = self.validate_has_falsification_criteria(spec) {
            violations.push(v);
        }

        // Standard validation
        if let Err(errs) = spec.validate() {
            for err in errs {
                violations.push(EddViolation::new("EDD-00", &err, ViolationSeverity::Error));
            }
        }

        if violations
            .iter()
            .any(|v| v.severity >= ViolationSeverity::Error)
        {
            self.violations.extend(violations.clone());
            Err(violations)
        } else {
            // Add warnings but return Ok
            self.violations.extend(violations);
            Ok(())
        }
    }

    // =========================================================================
    // Verification Test Validation
    // =========================================================================

    /// Validate that verification tests pass.
    ///
    /// # EDD-07: Verification test failed
    ///
    /// # Errors
    /// Returns `EDD-07` violations for each verification test that fails.
    pub fn validate_verification_tests<F>(
        &mut self,
        emc: &EquationModelCard,
        evaluator: F,
    ) -> Result<(), Vec<EddViolation>>
    where
        F: Fn(&std::collections::HashMap<String, f64>) -> f64,
    {
        let results = emc.run_verification_tests(evaluator);
        let failures: Vec<EddViolation> = results
            .into_iter()
            .filter(|(_, passed, _)| !passed)
            .map(|(name, _, msg)| {
                EddViolation::new(
                    "EDD-07",
                    &format!("Verification test failed: {name}"),
                    ViolationSeverity::Critical,
                )
                .with_context(&msg)
            })
            .collect();

        if failures.is_empty() {
            Ok(())
        } else {
            self.violations.extend(failures.clone());
            Err(failures)
        }
    }

    // =========================================================================
    // Conservation Law Validation (EDD-08)
    // =========================================================================

    /// Validate conservation laws are satisfied.
    ///
    /// # EDD-08: Conservation laws violated
    ///
    /// Monitors quantities that should be conserved (energy, momentum, etc.)
    /// and triggers Jidoka (stop-on-error) if drift exceeds tolerance.
    ///
    /// # Arguments
    ///
    /// * `quantity_name` - Name of the conserved quantity (e.g., "energy")
    /// * `initial_value` - Value at simulation start
    /// * `current_value` - Current value during simulation
    /// * `tolerance` - Maximum allowed relative drift
    ///
    /// # Errors
    ///
    /// Returns `EDD-08` violation if conservation law is violated.
    pub fn validate_conservation_law(
        &self,
        quantity_name: &str,
        initial_value: f64,
        current_value: f64,
        tolerance: f64,
    ) -> EddResult<()> {
        let relative_drift = if initial_value.abs() > f64::EPSILON {
            (current_value - initial_value).abs() / initial_value.abs()
        } else {
            (current_value - initial_value).abs()
        };

        if relative_drift > tolerance {
            Err(EddViolation::new(
                "EDD-08",
                &format!("Conservation law violated: {quantity_name} drifted beyond tolerance"),
                ViolationSeverity::Critical,
            ).with_context(&format!(
                "initial={initial_value:.6e}, current={current_value:.6e}, drift={relative_drift:.6e}, tolerance={tolerance:.6e}"
            )))
        } else {
            Ok(())
        }
    }

    // =========================================================================
    // Cross-Platform Reproducibility Validation (EDD-09)
    // =========================================================================

    /// Validate cross-platform reproducibility.
    ///
    /// # EDD-09: Cross-platform reproducibility failed
    ///
    /// Compares simulation results across different platforms to ensure
    /// bitwise-identical outputs given the same seed.
    ///
    /// # Arguments
    ///
    /// * `platform_a` - Name of first platform
    /// * `platform_b` - Name of second platform
    /// * `result_a` - Result from platform A
    /// * `result_b` - Result from platform B
    /// * `tolerance` - Maximum allowed difference (0.0 for exact match)
    ///
    /// # Errors
    ///
    /// Returns `EDD-09` violation if results differ beyond tolerance.
    pub fn validate_cross_platform_reproducibility(
        &self,
        platform_a: &str,
        platform_b: &str,
        result_a: f64,
        result_b: f64,
        tolerance: f64,
    ) -> EddResult<()> {
        let diff = (result_a - result_b).abs();

        if diff > tolerance {
            Err(EddViolation::new(
                "EDD-09",
                &format!("Cross-platform reproducibility failed: {platform_a} vs {platform_b}"),
                ViolationSeverity::Error,
            )
            .with_context(&format!(
                "{platform_a}={result_a:.15e}, {platform_b}={result_b:.15e}, diff={diff:.15e}"
            )))
        } else {
            Ok(())
        }
    }

    // =========================================================================
    // TDD Enforcement (EDD-10)
    // =========================================================================

    /// Validate that implementation has associated failing tests.
    ///
    /// # EDD-10: Implementation without failing test
    ///
    /// Enforces the TDD principle: no implementation without a failing test.
    /// This is verified by checking that test files exist and contain
    /// appropriate test cases.
    ///
    /// # Arguments
    ///
    /// * `implementation_name` - Name of the implementation
    /// * `has_test_file` - Whether a test file exists
    /// * `test_count` - Number of tests for this implementation
    ///
    /// # Errors
    ///
    /// Returns `EDD-10` violation if no tests exist.
    pub fn validate_tdd_compliance(
        &self,
        implementation_name: &str,
        has_test_file: bool,
        test_count: usize,
    ) -> EddResult<()> {
        if !has_test_file {
            return Err(EddViolation::new(
                "EDD-10",
                &format!("Implementation '{implementation_name}' has no test file"),
                ViolationSeverity::Critical,
            )
            .with_context("EDD requires TDD: write failing tests BEFORE implementation"));
        }

        if test_count == 0 {
            return Err(EddViolation::new(
                "EDD-10",
                &format!("Implementation '{implementation_name}' has no tests"),
                ViolationSeverity::Critical,
            )
            .with_context("Every implementation must have at least one test"));
        }

        Ok(())
    }

    // =========================================================================
    // Three Pillars Quality Gate (EDD-13, EDD-14, EDD-15)
    // =========================================================================

    /// # EDD-13: YAML-Only Configuration
    ///
    /// Validates that no hardcoded parameters exist - all configuration
    /// must come from YAML files.
    ///
    /// # Arguments
    ///
    /// * `has_yaml_config` - Whether the simulation has YAML configuration
    /// * `hardcoded_params` - List of detected hardcoded parameters
    ///
    /// # Errors
    ///
    /// Returns `EDD-13` violation if:
    /// - No YAML configuration is provided, or
    /// - Hardcoded parameters are detected
    pub fn validate_yaml_only_config(
        has_yaml_config: bool,
        hardcoded_params: &[String],
    ) -> Result<(), EddViolation> {
        if !has_yaml_config {
            return Err(EddViolation::new(
                "EDD-13",
                "Simulation requires YAML configuration",
                ViolationSeverity::Critical,
            )
            .with_context("Three Pillars: Pillar 2 requires YAML-only configuration"));
        }

        if !hardcoded_params.is_empty() {
            return Err(EddViolation::new(
                "EDD-13",
                &format!(
                    "Hardcoded parameters detected: {}",
                    hardcoded_params.join(", ")
                ),
                ViolationSeverity::Critical,
            )
            .with_context("All parameters must come from YAML configuration"));
        }

        Ok(())
    }

    /// # EDD-14: Probar TUI Verification
    ///
    /// Validates that TUI components have been tested with Probar.
    ///
    /// # Arguments
    ///
    /// * `probar_tests_passed` - Whether Probar TUI tests passed
    /// * `test_count` - Number of Probar TUI tests
    ///
    /// # Errors
    ///
    /// Returns `EDD-14` violation if:
    /// - No Probar tests exist, or
    /// - Probar tests failed
    pub fn validate_probar_tui(
        probar_tests_passed: bool,
        test_count: usize,
    ) -> Result<(), EddViolation> {
        if test_count == 0 {
            return Err(EddViolation::new(
                "EDD-14",
                "No Probar TUI tests found",
                ViolationSeverity::Critical,
            )
            .with_context("Three Pillars: Pillar 3 requires Probar TUI verification"));
        }

        if !probar_tests_passed {
            return Err(EddViolation::new(
                "EDD-14",
                "Probar TUI tests failed",
                ViolationSeverity::Critical,
            )
            .with_context("All Probar TUI tests must pass before release"));
        }

        Ok(())
    }

    /// # EDD-15: Probar WASM Verification
    ///
    /// Validates that WASM components have been tested with Probar.
    ///
    /// # Arguments
    ///
    /// * `probar_wasm_passed` - Whether Probar WASM tests passed
    /// * `test_count` - Number of Probar WASM tests
    ///
    /// # Errors
    ///
    /// Returns `EDD-15` violation if:
    /// - No Probar WASM tests exist, or
    /// - Probar WASM tests failed
    pub fn validate_probar_wasm(
        probar_wasm_passed: bool,
        test_count: usize,
    ) -> Result<(), EddViolation> {
        if test_count == 0 {
            return Err(EddViolation::new(
                "EDD-15",
                "No Probar WASM tests found",
                ViolationSeverity::Critical,
            )
            .with_context("Three Pillars: Pillar 3 requires Probar WASM verification"));
        }

        if !probar_wasm_passed {
            return Err(EddViolation::new(
                "EDD-15",
                "Probar WASM tests failed",
                ViolationSeverity::Critical,
            )
            .with_context("All Probar WASM tests must pass before release"));
        }

        Ok(())
    }

    /// # Three Pillars Quality Gate
    ///
    /// Validates all three pillars of provable simulation:
    /// 1. Z3 Equation Proofs (EDD-11, EDD-12)
    /// 2. YAML-Only Configuration (EDD-05, EDD-13)
    /// 3. Probar UX Verification (EDD-14, EDD-15)
    ///
    /// # Arguments
    ///
    /// * `z3_proofs_passed` - Whether Z3 proofs pass
    /// * `has_yaml_config` - Whether YAML config exists
    /// * `seed_specified` - Whether seed is specified
    /// * `probar_tui_passed` - Whether Probar TUI tests pass
    /// * `probar_wasm_passed` - Whether Probar WASM tests pass
    ///
    /// # Returns
    ///
    /// A vector of any violations found across all three pillars.
    #[must_use]
    #[allow(clippy::fn_params_excessive_bools)]
    pub fn validate_three_pillars(
        z3_proofs_passed: bool,
        has_yaml_config: bool,
        seed_specified: bool,
        probar_tui_passed: bool,
        probar_tui_test_count: usize,
        probar_wasm_passed: bool,
        probar_wasm_test_count: usize,
    ) -> Vec<EddViolation> {
        let mut violations = Vec::new();

        // Pillar 1: Z3 Equation Proofs
        if !z3_proofs_passed {
            violations.push(
                EddViolation::new(
                    "EDD-11",
                    "Z3 equation proofs did not pass",
                    ViolationSeverity::Critical,
                )
                .with_context("Pillar 1: All equations must be provable with Z3"),
            );
        }

        // Pillar 2: YAML-Only Configuration
        if !has_yaml_config {
            violations.push(
                EddViolation::new(
                    "EDD-13",
                    "No YAML configuration provided",
                    ViolationSeverity::Critical,
                )
                .with_context("Pillar 2: All configuration must be YAML-based"),
            );
        }

        if !seed_specified {
            violations.push(
                EddViolation::new(
                    "EDD-05",
                    "No seed specified in configuration",
                    ViolationSeverity::Critical,
                )
                .with_context("Pillar 2: Seed must be explicitly specified in YAML"),
            );
        }

        // Pillar 3: Probar UX Verification
        if let Err(e) = Self::validate_probar_tui(probar_tui_passed, probar_tui_test_count) {
            violations.push(e);
        }

        if let Err(e) = Self::validate_probar_wasm(probar_wasm_passed, probar_wasm_test_count) {
            violations.push(e);
        }

        violations
    }
}

/// Summary of EDD compliance status.
#[derive(Debug)]
pub struct EddComplianceSummary {
    /// Total number of violations
    pub total_violations: usize,
    /// Number of critical violations
    pub critical_count: usize,
    /// Number of error violations
    pub error_count: usize,
    /// Number of warning violations
    pub warning_count: usize,
    /// Number of info violations
    pub info_count: usize,
    /// Overall compliance status
    pub compliant: bool,
}

// =============================================================================
// TPS-Aligned Grades (Section 9.2)
// =============================================================================

/// TPS-aligned quality grades from Section 9.2 of EDD spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TpsGrade {
    /// 95-100%: Release OK
    ToyotaStandard,
    /// 85-94%: Beta with documented limitations
    KaizenRequired,
    /// 70-84%: Significant revision required
    AndonWarning,
    /// <70% or Critical failure: Block release
    StopTheLine,
}

impl TpsGrade {
    /// Calculate TPS grade from a compliance score (0.0 to 1.0).
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score >= 0.95 {
            Self::ToyotaStandard
        } else if score >= 0.85 {
            Self::KaizenRequired
        } else if score >= 0.70 {
            Self::AndonWarning
        } else {
            Self::StopTheLine
        }
    }

    /// Calculate TPS grade from violations.
    ///
    /// Any critical violation results in `StopTheLine` regardless of score.
    #[must_use]
    pub fn from_violations(violations: &[EddViolation], total_checks: usize) -> Self {
        // Any critical violation = STOP THE LINE
        if violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Critical)
        {
            return Self::StopTheLine;
        }

        // Calculate score based on errors (warnings don't affect score)
        let error_count = violations
            .iter()
            .filter(|v| v.severity >= ViolationSeverity::Error)
            .count();

        if total_checks == 0 {
            return Self::ToyotaStandard;
        }

        let score = 1.0 - (error_count as f64 / total_checks as f64);
        Self::from_score(score)
    }

    /// Get the decision text for this grade.
    #[must_use]
    pub const fn decision(&self) -> &'static str {
        match self {
            Self::ToyotaStandard => "Release OK",
            Self::KaizenRequired => "Beta with documented limitations",
            Self::AndonWarning => "Significant revision required",
            Self::StopTheLine => "Block release",
        }
    }

    /// Get the score range for this grade.
    #[must_use]
    pub const fn score_range(&self) -> &'static str {
        match self {
            Self::ToyotaStandard => "95-100%",
            Self::KaizenRequired => "85-94%",
            Self::AndonWarning => "70-84%",
            Self::StopTheLine => "<70% or Critical",
        }
    }
}

impl std::fmt::Display for TpsGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::ToyotaStandard => "Toyota Standard",
            Self::KaizenRequired => "Kaizen Required",
            Self::AndonWarning => "Andon Warning",
            Self::StopTheLine => "STOP THE LINE",
        };
        write!(f, "{name}")
    }
}

// =============================================================================
// Richardson Extrapolation for Convergence Order (EDD-07)
// =============================================================================

/// Result of Richardson extrapolation analysis.
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis {
    /// Computed convergence order
    pub order: f64,
    /// Expected convergence order
    pub expected_order: f64,
    /// Whether the order matches within tolerance
    pub order_matches: bool,
    /// Tolerance used for comparison
    pub tolerance: f64,
    /// Extrapolated value (improved estimate)
    pub extrapolated_value: f64,
    /// Error estimates at each refinement level
    pub error_estimates: Vec<f64>,
}

/// Compute convergence order using Richardson extrapolation.
///
/// Given function evaluations at successively refined step sizes,
/// estimate the order of convergence p where error ~ h^p.
///
/// # Arguments
///
/// * `values` - Function values at each refinement level (coarse to fine)
/// * `refinement_ratio` - Ratio between successive step sizes (default: 2.0)
/// * `expected_order` - Expected convergence order
/// * `tolerance` - Tolerance for order comparison
///
/// # Returns
///
/// `ConvergenceAnalysis` with computed order and extrapolated value.
///
/// # Panics
///
/// Panics if fewer than 3 values are provided.
#[must_use]
pub fn richardson_extrapolation(
    values: &[f64],
    refinement_ratio: f64,
    expected_order: f64,
    tolerance: f64,
) -> ConvergenceAnalysis {
    assert!(
        values.len() >= 3,
        "Richardson extrapolation requires at least 3 values"
    );

    let n = values.len();
    let r = refinement_ratio;

    // Compute error estimates (differences between successive values)
    let mut error_estimates = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        error_estimates.push((values[i] - values[i + 1]).abs());
    }

    // Compute observed convergence order using three finest values
    // p = log(|f_{h} - f_{h/r}| / |f_{h/r} - f_{h/r²}|) / log(r)
    let e1 = (values[n - 3] - values[n - 2]).abs();
    let e2 = (values[n - 2] - values[n - 1]).abs();

    let order = if e2 > f64::EPSILON && e1 > f64::EPSILON {
        (e1 / e2).ln() / r.ln()
    } else {
        expected_order // Assume expected if errors are tiny
    };

    // Richardson extrapolation formula for improved estimate
    // f_exact ≈ f_{h/r} + (f_{h/r} - f_h) / (r^p - 1)
    let extrapolated_value = if (r.powf(order) - 1.0).abs() > f64::EPSILON {
        values[n - 1] + (values[n - 1] - values[n - 2]) / (r.powf(order) - 1.0)
    } else {
        values[n - 1]
    };

    let order_matches = (order - expected_order).abs() <= tolerance;

    ConvergenceAnalysis {
        order,
        expected_order,
        order_matches,
        tolerance,
        extrapolated_value,
        error_estimates,
    }
}

impl EddComplianceSummary {
    /// Generate summary from violations.
    #[must_use]
    pub fn from_violations(violations: &[EddViolation]) -> Self {
        let critical_count = violations
            .iter()
            .filter(|v| v.severity == ViolationSeverity::Critical)
            .count();
        let error_count = violations
            .iter()
            .filter(|v| v.severity == ViolationSeverity::Error)
            .count();
        let warning_count = violations
            .iter()
            .filter(|v| v.severity == ViolationSeverity::Warning)
            .count();
        let info_count = violations
            .iter()
            .filter(|v| v.severity == ViolationSeverity::Info)
            .count();

        Self {
            total_violations: violations.len(),
            critical_count,
            error_count,
            warning_count,
            info_count,
            compliant: critical_count == 0 && error_count == 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edd::equation::Citation;
    use crate::edd::model_card::EmcBuilder;

    #[test]
    fn test_validator_new() {
        let validator = EddValidator::new();
        assert!(validator.violations().is_empty());
    }

    #[test]
    fn test_validate_simulation_has_emc_fails() {
        let validator = EddValidator::new();
        let result = validator.validate_simulation_has_emc(None);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-01");
        assert_eq!(err.severity, ViolationSeverity::Critical);
    }

    #[test]
    fn test_validate_simulation_has_emc_passes() {
        let validator = EddValidator::new();
        let emc = EmcBuilder::new()
            .name("Test")
            .equation("x = y")
            .citation(Citation::new(&["Test"], "Test", 2024))
            .add_verification_test("test", 1.0, 0.1)
            .build()
            .ok()
            .unwrap();

        let result = validator.validate_simulation_has_emc(Some(&emc));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_seed_specified_fails() {
        let validator = EddValidator::new();
        let result = validator.validate_seed_specified(None);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-05");
    }

    #[test]
    fn test_validate_seed_specified_passes() {
        let validator = EddValidator::new();
        let result = validator.validate_seed_specified(Some(42));
        assert!(result.is_ok());
    }

    #[test]
    fn test_compliance_summary() {
        let violations = vec![
            EddViolation::new("EDD-01", "test", ViolationSeverity::Critical),
            EddViolation::new("EDD-05", "test", ViolationSeverity::Error),
            EddViolation::new("EDD-06", "test", ViolationSeverity::Warning),
        ];

        let summary = EddComplianceSummary::from_violations(&violations);
        assert_eq!(summary.total_violations, 3);
        assert_eq!(summary.critical_count, 1);
        assert_eq!(summary.error_count, 1);
        assert_eq!(summary.warning_count, 1);
        assert!(!summary.compliant);
    }

    #[test]
    fn test_compliance_summary_compliant() {
        let violations = vec![
            EddViolation::new("EDD-06", "test", ViolationSeverity::Warning),
            EddViolation::new("EDD-00", "test", ViolationSeverity::Info),
        ];

        let summary = EddComplianceSummary::from_violations(&violations);
        assert!(summary.compliant);
    }

    #[test]
    fn test_violation_with_context() {
        let violation = EddViolation::new("EDD-07", "Test failed", ViolationSeverity::Critical)
            .with_context("Expected 10, got 15");

        assert!(violation.context.is_some());
        assert!(violation.context.unwrap().contains("Expected 10"));
    }

    #[test]
    fn test_validator_collects_violations() {
        let mut validator = EddValidator::lenient();

        // Create an EMC with missing tests
        let emc = EquationModelCard {
            name: "Test".to_string(),
            version: "1.0".to_string(),
            equation: String::new(), // Missing!
            class: crate::edd::equation::EquationClass::Queueing,
            citation: Citation::new(&[], "Test", 2024), // Missing authors!
            references: vec![],
            variables: vec![],
            verification_tests: vec![], // Missing!
            domain_constraints: vec![],
            falsification_criteria: vec![],
            implementation_notes: vec![],
            description: String::new(),
            lineage: vec![],
        };

        let result = validator.validate_emc(&emc);
        assert!(result.is_err());

        // Should have collected multiple violations
        assert!(validator.violations().len() >= 3);
    }

    #[test]
    fn test_has_critical_violations() {
        let mut validator = EddValidator::new();
        validator.violations.push(EddViolation::new(
            "EDD-01",
            "test",
            ViolationSeverity::Critical,
        ));

        assert!(validator.has_critical_violations());
        assert!(validator.has_errors());
    }

    #[test]
    fn test_clear_violations() {
        let mut validator = EddValidator::new();
        validator.violations.push(EddViolation::new(
            "EDD-01",
            "test",
            ViolationSeverity::Critical,
        ));

        validator.clear();
        assert!(validator.violations().is_empty());
    }

    // =========================================================================
    // EDD-08: Conservation Law Tests
    // =========================================================================

    #[test]
    fn test_validate_conservation_law_passes() {
        let validator = EddValidator::new();
        // Energy conserved within tolerance
        let result = validator.validate_conservation_law("energy", 100.0, 100.001, 1e-4);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_conservation_law_fails() {
        let validator = EddValidator::new();
        // Energy drifted beyond tolerance
        let result = validator.validate_conservation_law("energy", 100.0, 110.0, 1e-4);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-08");
        assert_eq!(err.severity, ViolationSeverity::Critical);
        assert!(err.message.contains("energy"));
    }

    #[test]
    fn test_validate_conservation_law_zero_initial() {
        let validator = EddValidator::new();
        // Handle case where initial value is zero
        let result = validator.validate_conservation_law("momentum", 0.0, 0.0001, 1e-4);
        assert!(result.is_ok());

        let result_fail = validator.validate_conservation_law("momentum", 0.0, 1.0, 1e-4);
        assert!(result_fail.is_err());
    }

    // =========================================================================
    // EDD-09: Cross-Platform Reproducibility Tests
    // =========================================================================

    #[test]
    fn test_validate_cross_platform_reproducibility_passes() {
        let validator = EddValidator::new();
        let result = validator.validate_cross_platform_reproducibility(
            "x86_64-linux",
            "aarch64-darwin",
            1.234567890123456,
            1.234567890123456,
            0.0,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_cross_platform_reproducibility_fails() {
        let validator = EddValidator::new();
        let result = validator.validate_cross_platform_reproducibility(
            "x86_64-linux",
            "aarch64-darwin",
            1.234567890123456,
            1.234567890123999,
            1e-15,
        );
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-09");
        assert_eq!(err.severity, ViolationSeverity::Error);
        assert!(err.context.is_some());
    }

    #[test]
    fn test_validate_cross_platform_with_tolerance() {
        let validator = EddValidator::new();
        // Allow small differences with relaxed IEEE mode
        let result = validator.validate_cross_platform_reproducibility(
            "x86_64-linux",
            "wasm32",
            1.0000000001,
            1.0000000002,
            1e-9,
        );
        assert!(result.is_ok());
    }

    // =========================================================================
    // EDD-10: TDD Compliance Tests
    // =========================================================================

    #[test]
    fn test_validate_tdd_compliance_passes() {
        let validator = EddValidator::new();
        let result = validator.validate_tdd_compliance("harmonic_oscillator", true, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_tdd_compliance_no_test_file() {
        let validator = EddValidator::new();
        let result = validator.validate_tdd_compliance("new_simulation", false, 0);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-10");
        assert!(err.message.contains("no test file"));
    }

    #[test]
    fn test_validate_tdd_compliance_no_tests() {
        let validator = EddValidator::new();
        let result = validator.validate_tdd_compliance("empty_simulation", true, 0);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-10");
        assert!(err.message.contains("no tests"));
    }

    // =========================================================================
    // TPS Grade Tests (Section 9.2)
    // =========================================================================

    #[test]
    fn test_tps_grade_from_score_toyota_standard() {
        assert_eq!(TpsGrade::from_score(1.0), TpsGrade::ToyotaStandard);
        assert_eq!(TpsGrade::from_score(0.95), TpsGrade::ToyotaStandard);
        assert_eq!(TpsGrade::from_score(0.99), TpsGrade::ToyotaStandard);
    }

    #[test]
    fn test_tps_grade_from_score_kaizen_required() {
        assert_eq!(TpsGrade::from_score(0.94), TpsGrade::KaizenRequired);
        assert_eq!(TpsGrade::from_score(0.85), TpsGrade::KaizenRequired);
        assert_eq!(TpsGrade::from_score(0.90), TpsGrade::KaizenRequired);
    }

    #[test]
    fn test_tps_grade_from_score_andon_warning() {
        assert_eq!(TpsGrade::from_score(0.84), TpsGrade::AndonWarning);
        assert_eq!(TpsGrade::from_score(0.70), TpsGrade::AndonWarning);
        assert_eq!(TpsGrade::from_score(0.75), TpsGrade::AndonWarning);
    }

    #[test]
    fn test_tps_grade_from_score_stop_the_line() {
        assert_eq!(TpsGrade::from_score(0.69), TpsGrade::StopTheLine);
        assert_eq!(TpsGrade::from_score(0.0), TpsGrade::StopTheLine);
        assert_eq!(TpsGrade::from_score(0.50), TpsGrade::StopTheLine);
    }

    #[test]
    fn test_tps_grade_from_violations_critical_always_stops() {
        let violations = vec![EddViolation::new(
            "EDD-01",
            "test",
            ViolationSeverity::Critical,
        )];
        // Even with only 1 violation out of 100 checks, critical = STOP
        assert_eq!(
            TpsGrade::from_violations(&violations, 100),
            TpsGrade::StopTheLine
        );
    }

    #[test]
    fn test_tps_grade_from_violations_no_violations() {
        let violations: Vec<EddViolation> = vec![];
        assert_eq!(
            TpsGrade::from_violations(&violations, 10),
            TpsGrade::ToyotaStandard
        );
    }

    #[test]
    fn test_tps_grade_from_violations_warnings_ignored() {
        let violations = vec![
            EddViolation::new("EDD-06", "test", ViolationSeverity::Warning),
            EddViolation::new("EDD-06", "test", ViolationSeverity::Warning),
        ];
        // Warnings don't affect score
        assert_eq!(
            TpsGrade::from_violations(&violations, 10),
            TpsGrade::ToyotaStandard
        );
    }

    #[test]
    fn test_tps_grade_decision_text() {
        assert_eq!(TpsGrade::ToyotaStandard.decision(), "Release OK");
        assert_eq!(
            TpsGrade::KaizenRequired.decision(),
            "Beta with documented limitations"
        );
        assert_eq!(
            TpsGrade::AndonWarning.decision(),
            "Significant revision required"
        );
        assert_eq!(TpsGrade::StopTheLine.decision(), "Block release");
    }

    #[test]
    fn test_tps_grade_display() {
        assert_eq!(format!("{}", TpsGrade::ToyotaStandard), "Toyota Standard");
        assert_eq!(format!("{}", TpsGrade::StopTheLine), "STOP THE LINE");
    }

    // =========================================================================
    // EDD-13: YAML-Only Configuration Tests
    // =========================================================================

    #[test]
    fn test_validate_yaml_only_config_passes() {
        let result = EddValidator::validate_yaml_only_config(true, &[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_yaml_only_config_no_yaml() {
        let result = EddValidator::validate_yaml_only_config(false, &[]);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-13");
        assert_eq!(err.severity, ViolationSeverity::Critical);
    }

    #[test]
    fn test_validate_yaml_only_config_hardcoded_params() {
        let hardcoded = vec!["omega".to_string(), "amplitude".to_string()];
        let result = EddValidator::validate_yaml_only_config(true, &hardcoded);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-13");
        assert!(err.message.contains("omega"));
        assert!(err.message.contains("amplitude"));
    }

    // =========================================================================
    // EDD-14: Probar TUI Verification Tests
    // =========================================================================

    #[test]
    fn test_validate_probar_tui_passes() {
        let result = EddValidator::validate_probar_tui(true, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_probar_tui_no_tests() {
        let result = EddValidator::validate_probar_tui(true, 0);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-14");
        assert_eq!(err.severity, ViolationSeverity::Critical);
    }

    #[test]
    fn test_validate_probar_tui_failed() {
        let result = EddValidator::validate_probar_tui(false, 5);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-14");
        assert!(err.message.contains("failed"));
    }

    // =========================================================================
    // EDD-15: Probar WASM Verification Tests
    // =========================================================================

    #[test]
    fn test_validate_probar_wasm_passes() {
        let result = EddValidator::validate_probar_wasm(true, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_probar_wasm_no_tests() {
        let result = EddValidator::validate_probar_wasm(true, 0);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-15");
        assert_eq!(err.severity, ViolationSeverity::Critical);
    }

    #[test]
    fn test_validate_probar_wasm_failed() {
        let result = EddValidator::validate_probar_wasm(false, 3);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err.code, "EDD-15");
        assert!(err.message.contains("failed"));
    }

    // =========================================================================
    // Three Pillars Quality Gate Tests
    // =========================================================================

    #[test]
    fn test_validate_three_pillars_all_pass() {
        let violations = EddValidator::validate_three_pillars(
            true, // z3_proofs_passed
            true, // has_yaml_config
            true, // seed_specified
            true, // probar_tui_passed
            5,    // probar_tui_test_count
            true, // probar_wasm_passed
            3,    // probar_wasm_test_count
        );
        assert!(violations.is_empty(), "All pillars should pass");
    }

    #[test]
    fn test_validate_three_pillars_z3_fails() {
        let violations = EddValidator::validate_three_pillars(
            false, // z3_proofs_passed
            true, true, true, 5, true, 3,
        );
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].code, "EDD-11");
    }

    #[test]
    fn test_validate_three_pillars_yaml_fails() {
        let violations = EddValidator::validate_three_pillars(
            true, false, // has_yaml_config
            true, true, 5, true, 3,
        );
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].code, "EDD-13");
    }

    #[test]
    fn test_validate_three_pillars_seed_missing() {
        let violations = EddValidator::validate_three_pillars(
            true, true, false, // seed_specified
            true, 5, true, 3,
        );
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].code, "EDD-05");
    }

    #[test]
    fn test_validate_three_pillars_probar_fails() {
        let violations = EddValidator::validate_three_pillars(
            true, true, true, false, // probar_tui_passed
            5, false, // probar_wasm_passed
            3,
        );
        assert_eq!(violations.len(), 2);
        assert!(violations.iter().any(|v| v.code == "EDD-14"));
        assert!(violations.iter().any(|v| v.code == "EDD-15"));
    }

    #[test]
    fn test_validate_three_pillars_multiple_failures() {
        let violations = EddValidator::validate_three_pillars(
            false, // z3_proofs_passed
            false, // has_yaml_config
            false, // seed_specified
            false, // probar_tui_passed
            0,     // probar_tui_test_count
            false, // probar_wasm_passed
            0,     // probar_wasm_test_count
        );
        // Should have multiple violations
        assert!(
            violations.len() >= 4,
            "Expected multiple violations: {:?}",
            violations
        );
    }

    // =========================================================================
    // Richardson Extrapolation Tests (EDD-07)
    // =========================================================================

    #[test]
    fn test_richardson_extrapolation_second_order() {
        // Simulate second-order convergence: error ~ h^2
        // f(h) = exact + C*h^2
        // With h = 1, 0.5, 0.25 and exact = 1.0, C = 1.0
        let values = vec![
            2.0,    // h=1:   1 + 1*1^2 = 2
            1.25,   // h=0.5: 1 + 1*0.25 = 1.25
            1.0625, // h=0.25: 1 + 1*0.0625 = 1.0625
        ];

        let result = richardson_extrapolation(&values, 2.0, 2.0, 0.1);

        // Should detect second-order convergence
        assert!(
            (result.order - 2.0).abs() < 0.1,
            "Expected order ~2.0, got {}",
            result.order
        );
        assert!(result.order_matches);

        // Extrapolated value should be close to exact (1.0)
        assert!(
            (result.extrapolated_value - 1.0).abs() < 0.1,
            "Expected extrapolated ~1.0, got {}",
            result.extrapolated_value
        );
    }

    #[test]
    fn test_richardson_extrapolation_first_order() {
        // Simulate first-order convergence: error ~ h
        // f(h) = exact + C*h
        // With h = 1, 0.5, 0.25 and exact = 0.0, C = 1.0
        let values = vec![
            1.0,   // h=1
            0.5,   // h=0.5
            0.25,  // h=0.25
            0.125, // h=0.125
        ];

        let result = richardson_extrapolation(&values, 2.0, 1.0, 0.1);

        // Should detect first-order convergence
        assert!(
            (result.order - 1.0).abs() < 0.1,
            "Expected order ~1.0, got {}",
            result.order
        );
        assert!(result.order_matches);
    }

    #[test]
    fn test_richardson_extrapolation_fourth_order() {
        // Simulate fourth-order convergence (RK4-like)
        // f(h) = exact + C*h^4
        let values = vec![
            1.0 + 1.0,            // h=1: 1 + 1^4 = 2
            1.0 + 0.0625,         // h=0.5: 1 + 0.5^4 = 1.0625
            1.0 + 0.00390625,     // h=0.25: 1 + 0.25^4
            1.0 + 0.000244140625, // h=0.125
        ];

        let result = richardson_extrapolation(&values, 2.0, 4.0, 0.2);

        // Should detect fourth-order convergence
        assert!(
            (result.order - 4.0).abs() < 0.3,
            "Expected order ~4.0, got {}",
            result.order
        );
    }

    #[test]
    fn test_richardson_extrapolation_error_estimates() {
        let values = vec![2.0, 1.25, 1.0625];
        let result = richardson_extrapolation(&values, 2.0, 2.0, 0.1);

        // Should have n-1 error estimates
        assert_eq!(result.error_estimates.len(), 2);

        // Errors should decrease
        assert!(
            result.error_estimates[0] > result.error_estimates[1],
            "Errors should decrease: {:?}",
            result.error_estimates
        );
    }

    #[test]
    #[should_panic(expected = "requires at least 3 values")]
    fn test_richardson_extrapolation_requires_minimum_values() {
        let values = vec![1.0, 0.5];
        let _ = richardson_extrapolation(&values, 2.0, 2.0, 0.1);
    }

    #[test]
    fn test_richardson_extrapolation_tolerance() {
        let values = vec![2.0, 1.25, 1.0625];

        // With tight tolerance, should not match if order slightly off
        let result_tight = richardson_extrapolation(&values, 2.0, 2.5, 0.01);
        assert!(!result_tight.order_matches);

        // With loose tolerance, should match
        let result_loose = richardson_extrapolation(&values, 2.0, 2.5, 1.0);
        assert!(result_loose.order_matches);
    }
}
