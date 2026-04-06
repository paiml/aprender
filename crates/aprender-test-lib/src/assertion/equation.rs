//! Equation Verification Assertions (Feature 22 - EDD Compliance)
//!
//! Provides assertions for verifying physics equations and game invariants.
//! Supports Equation-Driven Development (EDD) where game behavior is validated
//! against mathematical models.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Poka-Yoke**: Type-safe equation definitions prevent invalid formulas
//! - **Muda**: Fail-fast on equation violations with detailed diagnostics
//! - **Genchi Genbutsu**: Actual vs expected comparison with tolerance

use crate::result::{ProbarError, ProbarResult};
use std::collections::HashMap;
use std::fmt;

/// A variable binding for equation evaluation
#[derive(Debug, Clone)]
pub struct Variable {
    /// Variable name
    pub name: String,
    /// Current value
    pub value: f64,
    /// Optional unit (for documentation)
    pub unit: Option<String>,
}

impl Variable {
    /// Create a new variable
    #[must_use]
    pub fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
            unit: None,
        }
    }

    /// Create a variable with unit
    #[must_use]
    pub fn with_unit(name: &str, value: f64, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            value,
            unit: Some(unit.to_string()),
        }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.unit {
            Some(unit) => write!(f, "{} = {} {}", self.name, self.value, unit),
            None => write!(f, "{} = {}", self.name, self.value),
        }
    }
}

/// Context for equation evaluation
#[derive(Debug, Clone, Default)]
pub struct EquationContext {
    variables: HashMap<String, f64>,
}

impl EquationContext {
    /// Create a new empty context
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a variable value
    pub fn set(&mut self, name: &str, value: f64) -> &mut Self {
        self.variables.insert(name.to_string(), value);
        self
    }

    /// Get a variable value
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f64> {
        self.variables.get(name).copied()
    }

    /// Check if a variable exists
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    /// Get all variable names
    #[must_use]
    pub fn variables(&self) -> Vec<&str> {
        self.variables.keys().map(String::as_str).collect()
    }

    /// Create from a slice of variables
    #[must_use]
    pub fn from_variables(vars: &[Variable]) -> Self {
        let mut ctx = Self::new();
        for var in vars {
            ctx.set(&var.name, var.value);
        }
        ctx
    }
}

/// Result of an equation verification
#[derive(Debug, Clone)]
pub struct EquationResult {
    /// Name of the equation
    pub name: String,
    /// Whether the equation holds within tolerance
    pub passed: bool,
    /// Expected value
    pub expected: f64,
    /// Actual value
    pub actual: f64,
    /// Tolerance used
    pub tolerance: f64,
    /// Absolute difference
    pub difference: f64,
    /// Relative difference (percentage)
    pub relative_difference: f64,
    /// Diagnostic message
    pub message: String,
}

impl EquationResult {
    /// Create a new equation result
    #[must_use]
    fn new(name: &str, expected: f64, actual: f64, tolerance: f64) -> Self {
        let difference = (expected - actual).abs();
        let relative_difference = if expected.abs() > f64::EPSILON {
            (difference / expected.abs()) * 100.0
        } else {
            0.0
        };
        let passed = difference <= tolerance;

        let message = if passed {
            format!(
                "{}: expected {} ≈ {} (diff: {:.6}, tolerance: {})",
                name, expected, actual, difference, tolerance
            )
        } else {
            format!(
                "{}: FAILED - expected {} but got {} (diff: {:.6} > tolerance: {})",
                name, expected, actual, difference, tolerance
            )
        };

        Self {
            name: name.to_string(),
            passed,
            expected,
            actual,
            tolerance,
            difference,
            relative_difference,
            message,
        }
    }
}

impl fmt::Display for EquationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Equation verifier for testing physics and game invariants
#[derive(Debug)]
pub struct EquationVerifier {
    /// Name of the verification context
    name: String,
    /// Default tolerance for comparisons
    tolerance: f64,
    /// Results of verified equations
    results: Vec<EquationResult>,
}

impl EquationVerifier {
    /// Create a new equation verifier
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            tolerance: 1e-6,
            results: Vec::new(),
        }
    }

    /// Set the default tolerance
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Get the verifier name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get current tolerance
    #[must_use]
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Verify that two values are approximately equal
    pub fn verify_eq(&mut self, name: &str, expected: f64, actual: f64) -> &mut Self {
        let result = EquationResult::new(name, expected, actual, self.tolerance);
        self.results.push(result);
        self
    }

    /// Verify with custom tolerance
    pub fn verify_eq_with_tolerance(
        &mut self,
        name: &str,
        expected: f64,
        actual: f64,
        tolerance: f64,
    ) -> &mut Self {
        let result = EquationResult::new(name, expected, actual, tolerance);
        self.results.push(result);
        self
    }

    /// Verify a value is within a range
    pub fn verify_in_range(&mut self, name: &str, value: f64, min: f64, max: f64) -> &mut Self {
        let passed = value >= min && value <= max;
        let midpoint = (min + max) / 2.0;
        let difference = if passed {
            0.0
        } else if value < min {
            min - value
        } else {
            value - max
        };

        let message = if passed {
            format!("{}: {} is within [{}, {}]", name, value, min, max)
        } else {
            format!(
                "{}: FAILED - {} is outside range [{}, {}]",
                name, value, min, max
            )
        };

        self.results.push(EquationResult {
            name: name.to_string(),
            passed,
            expected: midpoint,
            actual: value,
            tolerance: (max - min) / 2.0,
            difference,
            relative_difference: 0.0,
            message,
        });
        self
    }

    /// Verify that a value is non-negative
    pub fn verify_non_negative(&mut self, name: &str, value: f64) -> &mut Self {
        let passed = value >= 0.0;
        let message = if passed {
            format!("{}: {} >= 0", name, value)
        } else {
            format!("{}: FAILED - {} < 0", name, value)
        };

        self.results.push(EquationResult {
            name: name.to_string(),
            passed,
            expected: 0.0,
            actual: value,
            tolerance: 0.0,
            difference: if passed { 0.0 } else { -value },
            relative_difference: 0.0,
            message,
        });
        self
    }

    /// Verify that a value is positive
    pub fn verify_positive(&mut self, name: &str, value: f64) -> &mut Self {
        let passed = value > 0.0;
        let message = if passed {
            format!("{}: {} > 0", name, value)
        } else {
            format!("{}: FAILED - {} <= 0", name, value)
        };

        self.results.push(EquationResult {
            name: name.to_string(),
            passed,
            expected: f64::EPSILON,
            actual: value,
            tolerance: 0.0,
            difference: if passed { 0.0 } else { -value },
            relative_difference: 0.0,
            message,
        });
        self
    }

    /// Get all results
    #[must_use]
    pub fn results(&self) -> &[EquationResult] {
        &self.results
    }

    /// Check if all verifications passed
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Get failed verifications
    #[must_use]
    pub fn failures(&self) -> Vec<&EquationResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Count passed verifications
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }

    /// Count failed verifications
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.passed).count()
    }

    /// Assert all verifications passed
    pub fn assert_all(&self) -> ProbarResult<()> {
        if self.all_passed() {
            Ok(())
        } else {
            let failures: Vec<String> = self.failures().iter().map(|r| r.message.clone()).collect();
            Err(ProbarError::AssertionFailed {
                message: format!(
                    "Equation verification '{}' failed:\n{}",
                    self.name,
                    failures.join("\n")
                ),
            })
        }
    }

    /// Clear all results
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

// =============================================================================
// Common Physics Equations
// =============================================================================

/// Verify kinematic equations (constant acceleration)
#[derive(Debug)]
pub struct KinematicVerifier {
    verifier: EquationVerifier,
}

impl KinematicVerifier {
    /// Create a new kinematic verifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            verifier: EquationVerifier::new("kinematics").with_tolerance(1e-4),
        }
    }

    /// With custom tolerance
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.verifier = self.verifier.with_tolerance(tolerance);
        self
    }

    /// Verify v = v0 + at
    pub fn verify_velocity(
        &mut self,
        v: f64,  // final velocity
        v0: f64, // initial velocity
        a: f64,  // acceleration
        t: f64,  // time
    ) -> &mut Self {
        let expected = v0 + a * t;
        self.verifier.verify_eq("v = v0 + at", expected, v);
        self
    }

    /// Verify x = x0 + v0*t + 0.5*a*t²
    pub fn verify_position(
        &mut self,
        x: f64,  // final position
        x0: f64, // initial position
        v0: f64, // initial velocity
        a: f64,  // acceleration
        t: f64,  // time
    ) -> &mut Self {
        let expected = x0 + v0 * t + 0.5 * a * t * t;
        self.verifier
            .verify_eq("x = x0 + v0*t + 0.5*a*t²", expected, x);
        self
    }

    /// Verify v² = v0² + 2a(x - x0)
    pub fn verify_velocity_squared(
        &mut self,
        v: f64,  // final velocity
        v0: f64, // initial velocity
        a: f64,  // acceleration
        x: f64,  // final position
        x0: f64, // initial position
    ) -> &mut Self {
        let expected = v0 * v0 + 2.0 * a * (x - x0);
        self.verifier
            .verify_eq("v² = v0² + 2a(x-x0)", expected, v * v);
        self
    }

    /// Get the underlying verifier
    #[must_use]
    pub fn verifier(&self) -> &EquationVerifier {
        &self.verifier
    }

    /// Assert all kinematic equations hold
    pub fn assert_all(&self) -> ProbarResult<()> {
        self.verifier.assert_all()
    }
}

impl Default for KinematicVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Verify energy conservation
#[derive(Debug)]
pub struct EnergyVerifier {
    verifier: EquationVerifier,
}

impl EnergyVerifier {
    /// Create a new energy verifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            verifier: EquationVerifier::new("energy").with_tolerance(1e-4),
        }
    }

    /// With custom tolerance
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.verifier = self.verifier.with_tolerance(tolerance);
        self
    }

    /// Verify kinetic energy: KE = 0.5 * m * v²
    pub fn verify_kinetic_energy(
        &mut self,
        ke: f64, // kinetic energy
        m: f64,  // mass
        v: f64,  // velocity
    ) -> &mut Self {
        let expected = 0.5 * m * v * v;
        self.verifier.verify_eq("KE = 0.5*m*v²", expected, ke);
        self
    }

    /// Verify potential energy: PE = m * g * h
    pub fn verify_potential_energy(
        &mut self,
        pe: f64, // potential energy
        m: f64,  // mass
        g: f64,  // gravity
        h: f64,  // height
    ) -> &mut Self {
        let expected = m * g * h;
        self.verifier.verify_eq("PE = m*g*h", expected, pe);
        self
    }

    /// Verify total mechanical energy conservation
    pub fn verify_conservation(
        &mut self,
        ke_initial: f64,
        pe_initial: f64,
        ke_final: f64,
        pe_final: f64,
    ) -> &mut Self {
        let total_initial = ke_initial + pe_initial;
        let total_final = ke_final + pe_final;
        self.verifier
            .verify_eq("E_total conserved", total_initial, total_final);
        self
    }

    /// Verify work-energy theorem: W = ΔKE
    pub fn verify_work_energy(&mut self, work: f64, ke_initial: f64, ke_final: f64) -> &mut Self {
        let delta_ke = ke_final - ke_initial;
        self.verifier.verify_eq("W = ΔKE", work, delta_ke);
        self
    }

    /// Get the underlying verifier
    #[must_use]
    pub fn verifier(&self) -> &EquationVerifier {
        &self.verifier
    }

    /// Assert all energy equations hold
    pub fn assert_all(&self) -> ProbarResult<()> {
        self.verifier.assert_all()
    }
}

impl Default for EnergyVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Verify momentum conservation
#[derive(Debug)]
pub struct MomentumVerifier {
    verifier: EquationVerifier,
}

impl MomentumVerifier {
    /// Create a new momentum verifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            verifier: EquationVerifier::new("momentum").with_tolerance(1e-4),
        }
    }

    /// With custom tolerance
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.verifier = self.verifier.with_tolerance(tolerance);
        self
    }

    /// Verify momentum: p = m * v
    pub fn verify_momentum(
        &mut self,
        p: f64, // momentum
        m: f64, // mass
        v: f64, // velocity
    ) -> &mut Self {
        let expected = m * v;
        self.verifier.verify_eq("p = m*v", expected, p);
        self
    }

    /// Verify momentum conservation in collision
    pub fn verify_conservation(
        &mut self,
        m1: f64,
        v1_initial: f64,
        m2: f64,
        v2_initial: f64,
        v1_final: f64,
        v2_final: f64,
    ) -> &mut Self {
        let p_initial = m1 * v1_initial + m2 * v2_initial;
        let p_final = m1 * v1_final + m2 * v2_final;
        self.verifier
            .verify_eq("p_total conserved", p_initial, p_final);
        self
    }

    /// Verify elastic collision (both momentum and KE conserved)
    pub fn verify_elastic_collision(
        &mut self,
        m1: f64,
        v1_initial: f64,
        m2: f64,
        v2_initial: f64,
        v1_final: f64,
        v2_final: f64,
    ) -> &mut Self {
        // Momentum conservation
        self.verify_conservation(m1, v1_initial, m2, v2_initial, v1_final, v2_final);

        // Kinetic energy conservation
        let ke_initial = 0.5 * m1 * v1_initial * v1_initial + 0.5 * m2 * v2_initial * v2_initial;
        let ke_final = 0.5 * m1 * v1_final * v1_final + 0.5 * m2 * v2_final * v2_final;
        self.verifier
            .verify_eq("KE conserved (elastic)", ke_initial, ke_final);
        self
    }

    /// Get the underlying verifier
    #[must_use]
    pub fn verifier(&self) -> &EquationVerifier {
        &self.verifier
    }

    /// Assert all momentum equations hold
    pub fn assert_all(&self) -> ProbarResult<()> {
        self.verifier.assert_all()
    }
}

impl Default for MomentumVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Game-specific invariant verifier
#[derive(Debug)]
pub struct InvariantVerifier {
    verifier: EquationVerifier,
}

impl InvariantVerifier {
    /// Create a new invariant verifier
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            verifier: EquationVerifier::new(name).with_tolerance(1e-6),
        }
    }

    /// With custom tolerance
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.verifier = self.verifier.with_tolerance(tolerance);
        self
    }

    /// Verify score is non-negative
    pub fn verify_score_non_negative(&mut self, score: f64) -> &mut Self {
        self.verifier.verify_non_negative("score >= 0", score);
        self
    }

    /// Verify health is in valid range [0, max_health]
    pub fn verify_health(&mut self, health: f64, max_health: f64) -> &mut Self {
        self.verifier
            .verify_in_range("health", health, 0.0, max_health);
        self
    }

    /// Verify position is within bounds
    pub fn verify_position_bounds(
        &mut self,
        x: f64,
        y: f64,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> &mut Self {
        self.verifier.verify_in_range("x position", x, min_x, max_x);
        self.verifier.verify_in_range("y position", y, min_y, max_y);
        self
    }

    /// Verify velocity is within speed limit
    pub fn verify_speed_limit(&mut self, vx: f64, vy: f64, max_speed: f64) -> &mut Self {
        let speed = (vx * vx + vy * vy).sqrt();
        self.verifier
            .verify_in_range("speed", speed, 0.0, max_speed);
        self
    }

    /// Verify entity count invariant
    pub fn verify_entity_count(&mut self, count: usize, expected: usize) -> &mut Self {
        self.verifier
            .verify_eq("entity count", expected as f64, count as f64);
        self
    }

    /// Custom invariant check
    pub fn verify_custom(&mut self, name: &str, expected: f64, actual: f64) -> &mut Self {
        self.verifier.verify_eq(name, expected, actual);
        self
    }

    /// Get the underlying verifier
    #[must_use]
    pub fn verifier(&self) -> &EquationVerifier {
        &self.verifier
    }

    /// Assert all invariants hold
    pub fn assert_all(&self) -> ProbarResult<()> {
        self.verifier.assert_all()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod variable_tests {
        use super::*;

        #[test]
        fn test_new() {
            let var = Variable::new("x", 10.0);
            assert_eq!(var.name, "x");
            assert!((var.value - 10.0).abs() < f64::EPSILON);
            assert!(var.unit.is_none());
        }

        #[test]
        fn test_with_unit() {
            let var = Variable::with_unit("velocity", 5.0, "m/s");
            assert_eq!(var.unit, Some("m/s".to_string()));
        }

        #[test]
        fn test_display() {
            let var1 = Variable::new("x", 10.0);
            assert_eq!(format!("{}", var1), "x = 10");

            let var2 = Variable::with_unit("v", 5.0, "m/s");
            assert_eq!(format!("{}", var2), "v = 5 m/s");
        }
    }

    mod equation_context_tests {
        use super::*;

        #[test]
        fn test_new() {
            let ctx = EquationContext::new();
            assert!(ctx.variables().is_empty());
        }

        #[test]
        fn test_set_and_get() {
            let mut ctx = EquationContext::new();
            ctx.set("x", 10.0);

            assert!(ctx.has("x"));
            assert_eq!(ctx.get("x"), Some(10.0));
            assert_eq!(ctx.get("y"), None);
        }

        #[test]
        fn test_from_variables() {
            let vars = vec![Variable::new("x", 1.0), Variable::new("y", 2.0)];
            let ctx = EquationContext::from_variables(&vars);

            assert_eq!(ctx.get("x"), Some(1.0));
            assert_eq!(ctx.get("y"), Some(2.0));
        }
    }

    mod equation_verifier_tests {
        use super::*;

        #[test]
        fn test_new() {
            let verifier = EquationVerifier::new("test");
            assert_eq!(verifier.name(), "test");
            assert!(verifier.results().is_empty());
        }

        #[test]
        fn test_verify_eq_pass() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_eq("1 + 1 = 2", 2.0, 2.0);

            assert!(verifier.all_passed());
            assert_eq!(verifier.passed_count(), 1);
        }

        #[test]
        fn test_verify_eq_fail() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_eq("1 + 1 = 3", 3.0, 2.0);

            assert!(!verifier.all_passed());
            assert_eq!(verifier.failed_count(), 1);
        }

        #[test]
        fn test_verify_eq_with_tolerance() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_eq_with_tolerance("approx", 1.0, 1.001, 0.01);

            assert!(verifier.all_passed());
        }

        #[test]
        fn test_verify_in_range_pass() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_in_range("value", 5.0, 0.0, 10.0);

            assert!(verifier.all_passed());
        }

        #[test]
        fn test_verify_in_range_fail() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_in_range("value", 15.0, 0.0, 10.0);

            assert!(!verifier.all_passed());
        }

        #[test]
        fn test_verify_non_negative_pass() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_non_negative("positive", 5.0);
            verifier.verify_non_negative("zero", 0.0);

            assert!(verifier.all_passed());
        }

        #[test]
        fn test_verify_non_negative_fail() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_non_negative("negative", -5.0);

            assert!(!verifier.all_passed());
        }

        #[test]
        fn test_verify_positive_pass() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_positive("positive", 5.0);

            assert!(verifier.all_passed());
        }

        #[test]
        fn test_verify_positive_fail() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_positive("zero", 0.0);

            assert!(!verifier.all_passed());
        }

        #[test]
        fn test_assert_all_pass() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_eq("test", 1.0, 1.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_assert_all_fail() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_eq("test", 1.0, 2.0);

            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_clear() {
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_eq("test", 1.0, 1.0);
            verifier.clear();

            assert!(verifier.results().is_empty());
        }
    }

    mod kinematic_verifier_tests {
        use super::*;

        #[test]
        fn test_verify_velocity() {
            let mut verifier = KinematicVerifier::new();
            // v = v0 + at = 10 + 2*5 = 20
            verifier.verify_velocity(20.0, 10.0, 2.0, 5.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_position() {
            let mut verifier = KinematicVerifier::new();
            // x = x0 + v0*t + 0.5*a*t² = 0 + 10*5 + 0.5*2*25 = 50 + 25 = 75
            verifier.verify_position(75.0, 0.0, 10.0, 2.0, 5.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_velocity_squared() {
            let mut verifier = KinematicVerifier::new();
            // v² = v0² + 2a(x-x0) = 100 + 2*2*50 = 100 + 200 = 300
            // v = sqrt(300) ≈ 17.32
            let v = (300.0_f64).sqrt();
            verifier.verify_velocity_squared(v, 10.0, 2.0, 50.0, 0.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_free_fall() {
            let mut verifier = KinematicVerifier::new();
            let g = 9.81;
            let t = 2.0;

            // Object dropped from rest
            let v = g * t; // v = 0 + g*t = 19.62 m/s
            let y = 0.5 * g * t * t; // y = 0 + 0 + 0.5*g*t² = 19.62 m

            verifier.verify_velocity(v, 0.0, g, t);
            verifier.verify_position(y, 0.0, 0.0, g, t);

            assert!(verifier.assert_all().is_ok());
        }
    }

    mod energy_verifier_tests {
        use super::*;

        #[test]
        fn test_verify_kinetic_energy() {
            let mut verifier = EnergyVerifier::new();
            // KE = 0.5 * m * v² = 0.5 * 2 * 9 = 9
            verifier.verify_kinetic_energy(9.0, 2.0, 3.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_potential_energy() {
            let mut verifier = EnergyVerifier::new();
            // PE = m * g * h = 1 * 10 * 5 = 50
            verifier.verify_potential_energy(50.0, 1.0, 10.0, 5.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_conservation() {
            let mut verifier = EnergyVerifier::new();
            // Total energy should be conserved
            // Initial: KE=50, PE=100, Total=150
            // Final: KE=100, PE=50, Total=150
            verifier.verify_conservation(50.0, 100.0, 100.0, 50.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_pendulum_energy() {
            let mut verifier = EnergyVerifier::new().with_tolerance(0.01);
            let m = 1.0;
            let g = 9.81;

            // At highest point: v=0, h=1m
            let pe_top = m * g * 1.0;
            let ke_top = 0.0;

            // At lowest point: h=0, v=sqrt(2gh)
            let v_bottom = (2.0_f64 * g * 1.0).sqrt();
            let ke_bottom = 0.5 * m * v_bottom * v_bottom;
            let pe_bottom = 0.0;

            verifier.verify_conservation(ke_top, pe_top, ke_bottom, pe_bottom);

            assert!(verifier.assert_all().is_ok());
        }
    }

    mod momentum_verifier_tests {
        use super::*;

        #[test]
        fn test_verify_momentum() {
            let mut verifier = MomentumVerifier::new();
            // p = m * v = 2 * 5 = 10
            verifier.verify_momentum(10.0, 2.0, 5.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_conservation() {
            let mut verifier = MomentumVerifier::new();
            // m1=1, v1=10, m2=1, v2=0
            // After collision: v1'=0, v2'=10 (perfect elastic)
            verifier.verify_conservation(1.0, 10.0, 1.0, 0.0, 0.0, 10.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_elastic_collision() {
            let mut verifier = MomentumVerifier::new();
            // Equal masses, one at rest: velocities exchange
            verifier.verify_elastic_collision(1.0, 10.0, 1.0, 0.0, 0.0, 10.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_pong_collision() {
            let mut verifier = MomentumVerifier::new().with_tolerance(0.1);

            // Pong ball hitting paddle (paddle effectively infinite mass)
            // Ball reverses direction, paddle doesn't move
            let ball_mass = 1.0;
            let ball_v_initial = 10.0;
            let ball_v_final = -10.0; // Reverses

            // For infinite mass paddle: momentum of ball alone isn't conserved
            // but this simulates elastic collision with immovable object
            let ke_initial = 0.5 * ball_mass * ball_v_initial * ball_v_initial;
            let ke_final = 0.5 * ball_mass * ball_v_final * ball_v_final;

            verifier
                .verifier
                .verify_eq("KE conserved", ke_initial, ke_final);
            assert!(verifier.assert_all().is_ok());
        }
    }

    mod invariant_verifier_tests {
        use super::*;

        #[test]
        fn test_verify_score_non_negative() {
            let mut verifier = InvariantVerifier::new("game");
            verifier.verify_score_non_negative(100.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_health() {
            let mut verifier = InvariantVerifier::new("game");
            verifier.verify_health(50.0, 100.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_position_bounds() {
            let mut verifier = InvariantVerifier::new("game");
            verifier.verify_position_bounds(400.0, 300.0, 0.0, 800.0, 0.0, 600.0);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_speed_limit() {
            let mut verifier = InvariantVerifier::new("game");
            verifier.verify_speed_limit(3.0, 4.0, 10.0); // speed = 5, limit = 10

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_entity_count() {
            let mut verifier = InvariantVerifier::new("game");
            verifier.verify_entity_count(10, 10);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_game_frame_invariants() {
            let mut verifier = InvariantVerifier::new("pong").with_tolerance(0.001);

            // Typical Pong game state
            let score = 5.0;
            let ball_x = 400.0;
            let ball_y = 300.0;
            let ball_vx = 5.0;
            let ball_vy = -3.0;
            let max_speed = 10.0;

            verifier
                .verify_score_non_negative(score)
                .verify_position_bounds(ball_x, ball_y, 0.0, 800.0, 0.0, 600.0)
                .verify_speed_limit(ball_vx, ball_vy, max_speed);

            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_verify_custom() {
            let mut verifier = InvariantVerifier::new("custom");
            verifier.verify_custom("custom_check", 42.0, 42.0);
            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_invariant_verifier_accessor() {
            let verifier = InvariantVerifier::new("test");
            assert_eq!(verifier.verifier().name(), "test");
        }
    }

    // Additional coverage tests for edge cases and uncovered branches
    mod additional_coverage_tests {
        use super::*;

        #[test]
        fn test_equation_result_display() {
            // Test Display trait for EquationResult
            let result = EquationResult::new("test_eq", 1.0, 1.0, 0.001);
            let display = format!("{}", result);
            assert!(display.contains("test_eq"));
            assert!(display.contains("expected"));
        }

        #[test]
        fn test_equation_result_relative_difference_zero_expected() {
            // Cover the branch when expected is near zero (relative_difference = 0.0)
            let result = EquationResult::new("zero_expected", 0.0, 0.0001, 0.001);
            assert!((result.relative_difference - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_equation_result_failed_message() {
            // Cover the failed message branch in EquationResult::new
            let result = EquationResult::new("fail_test", 10.0, 5.0, 0.001);
            assert!(!result.passed);
            assert!(result.message.contains("FAILED"));
            assert!(result.message.contains('>'));
        }

        #[test]
        fn test_equation_verifier_tolerance_accessor() {
            // Test the tolerance() accessor method
            let verifier = EquationVerifier::new("test").with_tolerance(0.01);
            assert!((verifier.tolerance() - 0.01).abs() < f64::EPSILON);
        }

        #[test]
        fn test_equation_verifier_failures() {
            // Test the failures() method
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_eq("pass", 1.0, 1.0);
            verifier.verify_eq("fail1", 10.0, 5.0);
            verifier.verify_eq("fail2", 20.0, 15.0);

            let failures = verifier.failures();
            assert_eq!(failures.len(), 2);
            assert!(failures.iter().any(|f| f.name == "fail1"));
            assert!(failures.iter().any(|f| f.name == "fail2"));
        }

        #[test]
        fn test_verify_in_range_below_min() {
            // Test the branch when value < min in verify_in_range
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_in_range("below_min", -5.0, 0.0, 10.0);

            assert!(!verifier.all_passed());
            let result = &verifier.results()[0];
            assert!(!result.passed);
            assert!((result.difference - 5.0).abs() < f64::EPSILON);
            assert!(result.message.contains("outside range"));
        }

        #[test]
        fn test_verify_positive_negative_value() {
            // Test verify_positive with a negative value (covers difference calculation)
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_positive("negative", -5.0);

            assert!(!verifier.all_passed());
            let result = &verifier.results()[0];
            assert!(!result.passed);
            assert!((result.difference - 5.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_kinematic_verifier_default() {
            // Test Default implementation for KinematicVerifier
            let verifier = KinematicVerifier::default();
            assert_eq!(verifier.verifier().name(), "kinematics");
        }

        #[test]
        fn test_kinematic_verifier_accessor() {
            // Test verifier() accessor for KinematicVerifier
            let verifier = KinematicVerifier::new();
            let inner = verifier.verifier();
            assert_eq!(inner.name(), "kinematics");
        }

        #[test]
        fn test_energy_verifier_default() {
            // Test Default implementation for EnergyVerifier
            let verifier = EnergyVerifier::default();
            assert_eq!(verifier.verifier().name(), "energy");
        }

        #[test]
        fn test_energy_verifier_work_energy() {
            // Test verify_work_energy method
            let mut verifier = EnergyVerifier::new();
            // Work = ΔKE = KE_final - KE_initial = 100 - 50 = 50
            verifier.verify_work_energy(50.0, 50.0, 100.0);
            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_energy_verifier_accessor() {
            // Test verifier() accessor for EnergyVerifier
            let verifier = EnergyVerifier::new();
            let inner = verifier.verifier();
            assert_eq!(inner.name(), "energy");
        }

        #[test]
        fn test_momentum_verifier_default() {
            // Test Default implementation for MomentumVerifier
            let verifier = MomentumVerifier::default();
            assert_eq!(verifier.verifier().name(), "momentum");
        }

        #[test]
        fn test_momentum_verifier_accessor() {
            // Test verifier() accessor for MomentumVerifier
            let verifier = MomentumVerifier::new();
            let inner = verifier.verifier();
            assert_eq!(inner.name(), "momentum");
        }

        #[test]
        fn test_equation_context_chaining() {
            // Test chaining set() calls
            let mut ctx = EquationContext::new();
            ctx.set("a", 1.0).set("b", 2.0).set("c", 3.0);

            assert_eq!(ctx.get("a"), Some(1.0));
            assert_eq!(ctx.get("b"), Some(2.0));
            assert_eq!(ctx.get("c"), Some(3.0));
        }

        #[test]
        fn test_equation_context_has_missing() {
            // Test has() for non-existent variable
            let ctx = EquationContext::new();
            assert!(!ctx.has("missing"));
        }

        #[test]
        fn test_equation_result_relative_difference_calculation() {
            // Test relative_difference calculation for non-zero expected
            let result = EquationResult::new("rel_diff", 100.0, 90.0, 1.0);
            // difference = 10, relative_difference = (10 / 100) * 100 = 10%
            assert!((result.relative_difference - 10.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_verifier_chaining() {
            // Test chaining of verify methods
            let mut verifier = EquationVerifier::new("chain");
            verifier
                .verify_eq("eq1", 1.0, 1.0)
                .verify_eq("eq2", 2.0, 2.0)
                .verify_non_negative("nn", 5.0)
                .verify_positive("pos", 1.0)
                .verify_in_range("range", 5.0, 0.0, 10.0);

            assert!(verifier.all_passed());
            assert_eq!(verifier.results().len(), 5);
        }

        #[test]
        fn test_invariant_verifier_health_out_of_range() {
            // Test health verification when health is out of range
            let mut verifier = InvariantVerifier::new("game");
            verifier.verify_health(150.0, 100.0);
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_invariant_verifier_speed_exceeds_limit() {
            // Test speed limit verification when speed exceeds limit
            let mut verifier = InvariantVerifier::new("game");
            verifier.verify_speed_limit(8.0, 6.0, 5.0); // speed = 10, limit = 5
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_kinematic_verifier_with_tolerance() {
            // Test KinematicVerifier with custom tolerance
            let mut verifier = KinematicVerifier::new().with_tolerance(0.1);
            verifier.verify_velocity(20.05, 10.0, 2.0, 5.0);
            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_momentum_verifier_with_tolerance() {
            // Test MomentumVerifier with custom tolerance
            let mut verifier = MomentumVerifier::new().with_tolerance(0.1);
            verifier.verify_momentum(10.05, 2.0, 5.0);
            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_invariant_verifier_with_tolerance() {
            // Test InvariantVerifier with custom tolerance
            let mut verifier = InvariantVerifier::new("test").with_tolerance(0.1);
            verifier.verify_custom("approx", 10.0, 10.05);
            assert!(verifier.assert_all().is_ok());
        }

        #[test]
        fn test_kinematic_verifier_failed_assertion() {
            // Test KinematicVerifier assert_all when failing
            let mut verifier = KinematicVerifier::new();
            verifier.verify_velocity(100.0, 10.0, 2.0, 5.0); // Expected: 20, got: 100
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_energy_verifier_failed_assertion() {
            // Test EnergyVerifier assert_all when failing
            let mut verifier = EnergyVerifier::new();
            verifier.verify_kinetic_energy(100.0, 2.0, 3.0); // Expected: 9, got: 100
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_momentum_verifier_failed_assertion() {
            // Test MomentumVerifier assert_all when failing
            let mut verifier = MomentumVerifier::new();
            verifier.verify_momentum(100.0, 2.0, 5.0); // Expected: 10, got: 100
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_invariant_verifier_failed_assertion() {
            // Test InvariantVerifier assert_all when failing
            let mut verifier = InvariantVerifier::new("test");
            verifier.verify_score_non_negative(-10.0);
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_verify_in_range_at_boundaries() {
            // Test verify_in_range at exact boundary values
            let mut verifier = EquationVerifier::new("boundaries");
            verifier.verify_in_range("at_min", 0.0, 0.0, 10.0);
            verifier.verify_in_range("at_max", 10.0, 0.0, 10.0);
            assert!(verifier.all_passed());
        }

        #[test]
        fn test_equation_context_variables_list() {
            // Test variables() accessor
            let mut ctx = EquationContext::new();
            ctx.set("x", 1.0).set("y", 2.0);

            let vars = ctx.variables();
            assert_eq!(vars.len(), 2);
            assert!(vars.contains(&"x"));
            assert!(vars.contains(&"y"));
        }

        #[test]
        fn test_equation_verifier_empty_results() {
            // Test all_passed and failures on empty verifier
            let verifier = EquationVerifier::new("empty");
            assert!(verifier.all_passed()); // No failures means all passed
            assert!(verifier.failures().is_empty());
            assert_eq!(verifier.passed_count(), 0);
            assert_eq!(verifier.failed_count(), 0);
        }

        #[test]
        fn test_equation_context_from_empty_variables() {
            // Test from_variables with empty slice
            let vars: Vec<Variable> = vec![];
            let ctx = EquationContext::from_variables(&vars);
            assert!(ctx.variables().is_empty());
        }

        #[test]
        fn test_variable_clone() {
            // Test Clone for Variable
            let var1 = Variable::with_unit("velocity", 10.0, "m/s");
            let var2 = var1;
            assert_eq!(var2.name, "velocity");
            assert!((var2.value - 10.0).abs() < f64::EPSILON);
            assert_eq!(var2.unit, Some("m/s".to_string()));
        }

        #[test]
        fn test_equation_context_clone() {
            // Test Clone for EquationContext
            let mut ctx1 = EquationContext::new();
            ctx1.set("x", 5.0);
            let ctx2 = ctx1.clone();
            assert_eq!(ctx2.get("x"), Some(5.0));
        }

        #[test]
        fn test_equation_result_clone() {
            // Test Clone for EquationResult
            let result1 = EquationResult::new("clone_test", 10.0, 10.0, 0.001);
            let result2 = result1;
            assert_eq!(result2.name, "clone_test");
            assert!(result2.passed);
        }

        #[test]
        fn test_equation_verifier_multiple_failures() {
            // Test verifier with multiple failures shows all in assert_all error
            let mut verifier = EquationVerifier::new("multi_fail");
            verifier.verify_eq("fail1", 1.0, 100.0);
            verifier.verify_eq("fail2", 2.0, 200.0);
            verifier.verify_eq("fail3", 3.0, 300.0);

            let result = verifier.assert_all();
            assert!(result.is_err());
            if let Err(ProbarError::AssertionFailed { message }) = result {
                assert!(message.contains("fail1"));
                assert!(message.contains("fail2"));
                assert!(message.contains("fail3"));
            }
        }

        #[test]
        fn test_verify_in_range_message_passed() {
            // Verify the message format when in_range passes
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_in_range("in_bounds", 5.0, 0.0, 10.0);

            let result = &verifier.results()[0];
            assert!(result.message.contains("is within"));
        }

        #[test]
        fn test_verify_non_negative_message() {
            // Verify message format for non_negative
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_non_negative("positive_val", 5.0);

            let result = &verifier.results()[0];
            assert!(result.message.contains(">= 0"));
        }

        #[test]
        fn test_verify_positive_message() {
            // Verify message format for positive
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_positive("pos_val", 5.0);

            let result = &verifier.results()[0];
            assert!(result.message.contains("> 0"));
        }

        #[test]
        fn test_equation_result_all_fields() {
            // Test all fields of EquationResult are correctly set
            let result = EquationResult::new("field_test", 100.0, 95.0, 1.0);
            assert_eq!(result.name, "field_test");
            assert!(!result.passed); // 5.0 > 1.0 tolerance
            assert!((result.expected - 100.0).abs() < f64::EPSILON);
            assert!((result.actual - 95.0).abs() < f64::EPSILON);
            assert!((result.tolerance - 1.0).abs() < f64::EPSILON);
            assert!((result.difference - 5.0).abs() < f64::EPSILON);
            assert!((result.relative_difference - 5.0).abs() < f64::EPSILON); // 5%
        }

        #[test]
        fn test_energy_verifier_conservation_fails() {
            // Test energy conservation failure
            let mut verifier = EnergyVerifier::new();
            verifier.verify_conservation(50.0, 100.0, 80.0, 80.0); // 150 != 160
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_kinematic_verifier_position_fails() {
            // Test position verification failure
            let mut verifier = KinematicVerifier::new();
            verifier.verify_position(100.0, 0.0, 10.0, 2.0, 5.0); // Expected: 75, got: 100
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_kinematic_verifier_velocity_squared_fails() {
            // Test velocity squared verification failure
            let mut verifier = KinematicVerifier::new();
            verifier.verify_velocity_squared(100.0, 10.0, 2.0, 50.0, 0.0); // Expected v²=300, v=100 gives v²=10000
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_momentum_conservation_fails() {
            // Test momentum conservation failure
            let mut verifier = MomentumVerifier::new();
            verifier.verify_conservation(1.0, 10.0, 1.0, 0.0, 5.0, 3.0); // p_initial=10, p_final=8
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_elastic_collision_ke_fails() {
            // Test elastic collision where KE is not conserved
            let mut verifier = MomentumVerifier::new();
            // Momentum conserved but KE not (inelastic collision)
            verifier.verify_elastic_collision(1.0, 10.0, 1.0, 0.0, 5.0, 5.0);
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_invariant_entity_count_fails() {
            // Test entity count mismatch
            let mut verifier = InvariantVerifier::new("game");
            verifier.verify_entity_count(5, 10);
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_invariant_position_out_of_bounds() {
            // Test position bounds failure
            let mut verifier = InvariantVerifier::new("game");
            verifier.verify_position_bounds(900.0, 700.0, 0.0, 800.0, 0.0, 600.0);
            assert!(verifier.assert_all().is_err());
        }

        #[test]
        fn test_verify_negative_value_non_negative() {
            // Verify the difference calculation for negative values in non_negative
            let mut verifier = EquationVerifier::new("test");
            verifier.verify_non_negative("neg", -10.0);

            let result = &verifier.results()[0];
            assert!((result.difference - 10.0).abs() < f64::EPSILON);
            assert!(result.message.contains("< 0"));
        }

        #[test]
        fn test_verifier_debug_impl() {
            // Test Debug implementation for EquationVerifier
            let verifier = EquationVerifier::new("debug_test");
            let debug_str = format!("{:?}", verifier);
            assert!(debug_str.contains("debug_test"));
        }

        #[test]
        fn test_kinematic_verifier_debug_impl() {
            // Test Debug implementation for KinematicVerifier
            let verifier = KinematicVerifier::new();
            let debug_str = format!("{:?}", verifier);
            assert!(debug_str.contains("KinematicVerifier"));
        }

        #[test]
        fn test_energy_verifier_debug_impl() {
            // Test Debug implementation for EnergyVerifier
            let verifier = EnergyVerifier::new();
            let debug_str = format!("{:?}", verifier);
            assert!(debug_str.contains("EnergyVerifier"));
        }

        #[test]
        fn test_momentum_verifier_debug_impl() {
            // Test Debug implementation for MomentumVerifier
            let verifier = MomentumVerifier::new();
            let debug_str = format!("{:?}", verifier);
            assert!(debug_str.contains("MomentumVerifier"));
        }

        #[test]
        fn test_invariant_verifier_debug_impl() {
            // Test Debug implementation for InvariantVerifier
            let verifier = InvariantVerifier::new("test");
            let debug_str = format!("{:?}", verifier);
            assert!(debug_str.contains("InvariantVerifier"));
        }

        #[test]
        fn test_variable_debug_impl() {
            // Test Debug implementation for Variable
            let var = Variable::new("x", 5.0);
            let debug_str = format!("{:?}", var);
            assert!(debug_str.contains("Variable"));
        }

        #[test]
        fn test_equation_context_debug_impl() {
            // Test Debug implementation for EquationContext
            let ctx = EquationContext::new();
            let debug_str = format!("{:?}", ctx);
            assert!(debug_str.contains("EquationContext"));
        }

        #[test]
        fn test_equation_result_debug_impl() {
            // Test Debug implementation for EquationResult
            let result = EquationResult::new("test", 1.0, 1.0, 0.001);
            let debug_str = format!("{:?}", result);
            assert!(debug_str.contains("EquationResult"));
        }

        #[test]
        fn test_equation_context_default() {
            // Test Default implementation for EquationContext
            let ctx = EquationContext::default();
            assert!(ctx.variables().is_empty());
        }

        #[test]
        fn test_energy_work_energy_fails() {
            // Test work-energy theorem failure
            let mut verifier = EnergyVerifier::new();
            verifier.verify_work_energy(100.0, 50.0, 100.0); // W=100, ΔKE=50, should fail
            assert!(verifier.assert_all().is_err());
        }
    }
}
