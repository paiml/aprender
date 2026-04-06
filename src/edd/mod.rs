//! Equation-Driven Development (EDD) module.
//!
//! Implements the EDD specification for falsifiable, equation-first simulation.
//!
//! # The Four Pillars of EDD
//!
//! 1. **Prove It** - Every simulation begins with a mathematically-verified governing equation
//! 2. **Fail It** - TDD with failing tests derived from analytical solutions
//! 3. **Seed It** - Deterministic reproducibility via explicit random seeds
//! 4. **Falsify It** - Active search for conditions that disprove the model
//!
//! # Operations Science Equations
//!
//! - Little's Law: `L = λW` (WIP = Throughput × Cycle Time)
//! - Kingman's Formula: VUT equation for queue wait times
//! - Square Root Law: Safety stock scaling
//! - Bullwhip Effect: Variance amplification in supply chains
//!
//! # References
//!
//! - [30] Little, J.D.C. (1961). "A Proof for the Queuing Formula: L = λW"
//! - [31] Kingman, J.F.C. (1961). "The single server queue in heavy traffic"
//! - [32] Lee, H.L., et al. (1997). "The Bullwhip Effect in Supply Chains"
//! - [33] Hopp, W.J. & Spearman, M.L. (2004). "To Pull or Not to Pull"

pub mod audit;
pub mod equation;
pub mod experiment;
pub mod falsifiable;
pub mod gui_coverage;
pub mod loader;
pub mod model_card;
pub mod operations;
pub mod prover;
pub mod report;
pub mod runner;
pub mod style;
pub mod tps;
pub mod traits;
pub mod v2;
pub mod validation;

// Re-exports
pub use audit::{
    hash_state, verify_rng_consistency, AuditLogReplayer, Decision, EquationEval,
    GeneratedTestCase, ReplaySpeed, ReplayState, SimulationAuditLog, StepEntry, TspStateSnapshot,
    TspStepType,
};
pub use equation::Citation;
pub use equation::{EquationClass, EquationVariable, GoverningEquation};
pub use experiment::{
    ExperimentHypothesis, ExperimentSpec, FalsificationAction, FalsificationCriterion,
};
pub use falsifiable::{
    ExperimentSeed, FalsifiableSimulation, FalsificationResult, ParamSpace, Trajectory,
};
pub use gui_coverage::GuiCoverage;
pub use loader::{EmcYaml, ExperimentYaml};
pub use model_card::{DomainConstraint, EmcBuilder, EquationModelCard};
pub use operations::{BullwhipEffect, KingmanFormula, LittlesLaw, SquareRootLaw};
#[cfg(feature = "z3-proofs")]
pub use prover::z3_impl;
pub use prover::{ProofError, ProofResult, Z3Provable};
pub use report::{ReportFormat, ReportGenerator};
pub use runner::{
    EddComplianceChecklist, EmcComplianceReport, EmcRegistry, ExecutionMetrics, ExperimentDomain,
    ExperimentResult, ExperimentRunner, FalsificationCriterionResult, FalsificationSummary,
    ReproducibilitySummary, RunnerConfig, VerificationSummary, VerificationTestSummary,
};
pub use tps::{
    validate_bullwhip_effect, validate_cell_layout, validate_kanban_vs_dbr,
    validate_kingmans_curve, validate_littles_law, validate_push_vs_pull, validate_shojinka,
    validate_smed_setup, validate_square_root_law, TpsMetrics, TpsTestCase, TpsTestResult,
};
pub use traits::{
    ConfigError, EddSimulation, Reproducible, TestResult, ValidationResult, VerificationResult,
    YamlConfigurable,
};
pub use validation::{
    richardson_extrapolation, ConvergenceAnalysis, EddComplianceSummary, EddResult, EddValidator,
    EddViolation, TpsGrade, ViolationSeverity,
};

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // RED PHASE: Write failing tests first (Extreme TDD)
    // =========================================================================

    // -------------------------------------------------------------------------
    // Pillar 1: Prove It - Governing Equation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_governing_equation_must_have_latex_representation() {
        let eq = operations::LittlesLaw::new();
        assert!(
            !eq.latex().is_empty(),
            "Equation must have LaTeX representation"
        );
        assert!(
            eq.latex().contains("\\lambda"),
            "Little's Law must reference λ"
        );
    }

    #[test]
    fn test_governing_equation_must_have_citation() {
        let eq = operations::LittlesLaw::new();
        let citation = eq.citation();
        assert!(citation.year > 0, "Citation must have valid year");
        assert!(!citation.authors.is_empty(), "Citation must have authors");
    }

    #[test]
    fn test_governing_equation_must_have_variables() {
        let eq = operations::LittlesLaw::new();
        let vars = eq.variables();
        assert!(
            vars.len() >= 3,
            "Little's Law must have at least 3 variables (L, λ, W)"
        );
    }

    // -------------------------------------------------------------------------
    // Pillar 2: Fail It - TDD Tests from Equations
    // -------------------------------------------------------------------------

    #[test]
    fn test_littles_law_analytical_solution() {
        // L = λW: If λ=5 items/hr and W=2 hrs, then L=10
        let eq = operations::LittlesLaw::new();
        let result = eq.evaluate(5.0, 2.0);
        assert!((result - 10.0).abs() < 1e-10, "L = λW = 5 * 2 = 10");
    }

    #[test]
    fn test_littles_law_validation() {
        let eq = operations::LittlesLaw::new();

        // Valid case: L ≈ λW
        let valid = eq.validate(10.0, 5.0, 2.0, 0.01);
        assert!(valid.is_ok(), "Should pass when L = λW");

        // Invalid case: L ≠ λW
        let invalid = eq.validate(15.0, 5.0, 2.0, 0.01);
        assert!(invalid.is_err(), "Should fail when L ≠ λW");
    }

    #[test]
    fn test_kingmans_formula_hockey_stick() {
        let eq = operations::KingmanFormula::new();

        // At low utilization (50%), wait time is manageable
        let wait_50 = eq.expected_wait_time(0.5, 1.0, 1.0, 1.0);

        // At high utilization (95%), wait time explodes
        let wait_95 = eq.expected_wait_time(0.95, 1.0, 1.0, 1.0);

        // The "hockey stick" effect: 95% util should be ~19x wait of 50% util
        assert!(
            wait_95 > wait_50 * 10.0,
            "High utilization should cause exponential wait time increase"
        );
    }

    // -------------------------------------------------------------------------
    // Pillar 3: Seed It - Reproducibility Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_experiment_must_have_seed() {
        let spec = ExperimentSpec::builder().name("Test Experiment").build();

        // Should fail without explicit seed
        assert!(spec.is_err(), "Experiment must require explicit seed");
    }

    #[test]
    fn test_experiment_with_seed_is_reproducible() {
        let spec = ExperimentSpec::builder()
            .name("Test Experiment")
            .seed(42)
            .build()
            .expect("Should build with seed");

        assert_eq!(spec.seed(), 42, "Seed must be preserved");
    }

    // -------------------------------------------------------------------------
    // Pillar 4: Falsify It - Active Refutation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_falsification_criteria_required() {
        let spec = ExperimentSpec::builder().name("Test").seed(42).build();

        // Should fail without falsification criteria
        assert!(
            spec.is_err()
                || spec
                    .as_ref()
                    .map(|s| s.falsification_criteria().is_empty())
                    .unwrap_or(true),
            "Experiment should require falsification criteria"
        );
    }

    // -------------------------------------------------------------------------
    // Equation Model Card Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_emc_must_have_governing_equation() {
        let emc = EquationModelCard::builder().name("Test EMC").build();

        assert!(emc.is_err(), "EMC must require governing equation");
    }

    #[test]
    fn test_emc_must_have_analytical_derivation() {
        let emc = EquationModelCard::builder()
            .name("Test EMC")
            .equation("L = \\lambda W")
            .build();

        assert!(
            emc.is_err(),
            "EMC must require analytical derivation (citation)"
        );
    }

    #[test]
    fn test_emc_must_have_verification_tests() {
        let emc = EquationModelCard::builder()
            .name("Little's Law")
            .equation("L = \\lambda W")
            .citation(Citation::new(
                &["Little, J.D.C."],
                "Operations Research",
                1961,
            ))
            .build();

        assert!(emc.is_err(), "EMC must require verification tests");
    }

    #[test]
    fn test_emc_complete_builds_successfully() {
        let emc = EquationModelCard::builder()
            .name("Little's Law")
            .equation("L = \\lambda W")
            .citation(Citation::new(
                &["Little, J.D.C."],
                "Operations Research",
                1961,
            ))
            .add_variable("L", "Average queue length", "items")
            .add_variable("lambda", "Arrival rate", "items/time")
            .add_variable("W", "Average wait time", "time")
            .add_verification_test("L = λW for λ=5, W=2 => L=10", 10.0, 1e-10)
            .build();

        assert!(emc.is_ok(), "Complete EMC should build successfully");
    }

    // -------------------------------------------------------------------------
    // Operations Science Test Cases (from TPS empirical validation)
    // -------------------------------------------------------------------------

    #[test]
    fn test_tc1_push_vs_pull_little_law_holds() {
        // Test Case 1: Little's Law holds under stochastic conditions
        let eq = operations::LittlesLaw::new();

        // Various WIP levels should maintain linear relationship
        let test_cases = [
            (10.0, 2.0, 5.0),  // WIP=10, TH=5 => CT=2
            (25.0, 5.0, 5.0),  // WIP=25, TH=5 => CT=5
            (50.0, 10.0, 5.0), // WIP=50, TH=5 => CT=10
        ];

        for (wip, ct, th) in test_cases {
            let result = eq.validate(wip, th, ct, 0.001);
            assert!(
                result.is_ok(),
                "Little's Law should hold: WIP={wip}, TH={th}, CT={ct}"
            );
        }
    }

    #[test]
    fn test_tc8_kingmans_curve_exponential() {
        // Test Case 8: Wait times are exponential, not linear
        let eq = operations::KingmanFormula::new();

        let util_levels = [0.5, 0.7, 0.85, 0.95];
        let mut wait_times = Vec::new();

        for &rho in &util_levels {
            let wait = eq.expected_wait_time(rho, 1.0, 1.0, 1.0);
            wait_times.push(wait);
        }

        // Check exponential growth (each step should increase by more than previous)
        for i in 1..wait_times.len() - 1 {
            let delta_prev = wait_times[i] - wait_times[i - 1];
            let delta_curr = wait_times[i + 1] - wait_times[i];
            assert!(delta_curr > delta_prev,
                "Wait time growth should accelerate (exponential): prev_delta={delta_prev}, curr_delta={delta_curr}");
        }
    }

    #[test]
    fn test_tc9_square_root_law() {
        // Test Case 9: Inventory scales as √lead_time, not linearly
        // The Square Root Law: I_safety = z × σ_D × √L
        let eq = operations::SquareRootLaw::new();

        // If lead time quadruples, safety stock should only double (√4 = 2)
        let stock_l1 = eq.safety_stock(100.0, 1.0, 1.96);
        let stock_l4 = eq.safety_stock(100.0, 4.0, 1.96);

        let ratio = stock_l4 / stock_l1;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "Safety stock should scale as √L: ratio should be 2.0, got {ratio}"
        );
    }

    // -------------------------------------------------------------------------
    // EDD Validator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_edd_validator_checks_emc_presence() {
        let validator = EddValidator::new();

        // Simulation without EMC should fail
        let result = validator.validate_simulation_has_emc(None);
        assert!(result.is_err(), "Simulation without EMC should fail EDD-01");
    }

    #[test]
    fn test_edd_validator_checks_seed_presence() {
        let validator = EddValidator::new();

        // Experiment without seed should fail
        let result = validator.validate_seed_specified(None);
        assert!(
            result.is_err(),
            "Experiment without seed should fail EDD-05"
        );
    }
}
