//! EDD (Equation-Driven Development) Checks (EDD-01 through EDD-03)
//!
//! These checks focus on equation verification and validation:
//! - Equation verification before implementation
//! - Equation Model Card (EMC) completeness
//! - Numerical vs analytical validation

use super::helpers::check_for_pattern;
use crate::falsification::helpers::{apply_check_outcome, CheckOutcome};
use crate::falsification::types::{CheckItem, Evidence, EvidenceType, Severity};
use std::path::Path;
use std::time::Instant;

/// EDD-01: Equation Verification Before Implementation
///
/// **Claim:** Every simulation has analytically verified governing equation.
///
/// **Rejection Criteria (Major):**
/// - Simulation without EMC or analytical derivation
pub fn check_equation_verification(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "EDD-01",
        "Equation Verification Before Implementation",
        "Simulations have verified governing equations",
    )
    .with_severity(Severity::Major)
    .with_tps("EDD - prove first, implement second");

    // Check for EMC (Equation Model Cards)
    let has_emc = project_path.join("docs/emc/").exists()
        || check_for_pattern(project_path, &["equation_model_card", "EMC", "governing_equation"]);

    // Check for mathematical documentation
    let has_math_docs =
        check_for_pattern(project_path, &["derivation", "proof", "analytical_solution", "LaTeX"]);

    // Check for simulation code
    let has_simulation =
        check_for_pattern(project_path, &["simulate", "Simulator", "physics", "dynamics"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "EDD: emc={}, math_docs={}, simulation={}",
            has_emc, has_math_docs, has_simulation
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (!has_simulation, CheckOutcome::Pass),
            (has_emc && has_math_docs, CheckOutcome::Pass),
            (has_emc || has_math_docs, CheckOutcome::Partial("Partial equation documentation")),
            (true, CheckOutcome::Partial("Simulation without equation verification")),
        ],
    );

    item.finish_timed(start)
}

/// EDD-02: Equation Model Card (EMC) Completeness
///
/// **Claim:** Every shared simulation includes complete EMC.
///
/// **Rejection Criteria (Major):**
/// - EMC missing required sections
pub fn check_emc_completeness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "EDD-02",
        "Equation Model Card Completeness",
        "Simulations include complete EMC",
    )
    .with_severity(Severity::Major)
    .with_tps("Governance - equation traceability");

    // Check for EMC directory
    let emc_dir = project_path.join("docs/emc/");
    let has_emc_dir = emc_dir.exists();

    // Check EMC completeness (governing equations, validity, derivation, stability, tests)
    let required_sections = ["governing", "validity", "derivation", "stability", "verification"];

    let mut sections_found = 0;
    if let Ok(entries) = glob::glob(&format!("{}/**/*.md", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                for section in &required_sections {
                    if content.to_lowercase().contains(section) {
                        sections_found += 1;
                    }
                }
            }
        }
    }

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "EMC: dir={}, sections_found={}/{}",
            has_emc_dir,
            sections_found,
            required_sections.len()
        ),
        data: None,
        files: Vec::new(),
    });

    let has_simulation = check_for_pattern(project_path, &["simulate", "Simulator"]);
    item = apply_check_outcome(
        item,
        &[
            (!has_simulation, CheckOutcome::Pass),
            (has_emc_dir && sections_found >= 4, CheckOutcome::Pass),
            (sections_found >= 2, CheckOutcome::Partial("Partial EMC documentation")),
            (true, CheckOutcome::Partial("Missing EMC documentation")),
        ],
    );

    item.finish_timed(start)
}

/// EDD-03: Numerical vs Analytical Validation
///
/// **Claim:** Simulation results match analytical solutions within tolerance.
///
/// **Rejection Criteria (Major):**
/// - Numerical-analytical deviation >1e-6
pub fn check_numerical_analytical_validation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "EDD-03",
        "Numerical vs Analytical Validation",
        "Simulation matches analytical solutions",
    )
    .with_severity(Severity::Major)
    .with_tps("Verification - known solutions");

    // Check for analytical validation tests
    let has_analytical_tests = check_for_pattern(
        project_path,
        &["analytical_solution", "exact_solution", "closed_form", "validate_against"],
    );

    // Check for tolerance specification
    let has_tolerance =
        check_for_pattern(project_path, &["tolerance", "epsilon", "1e-6", "assert_relative_eq"]);

    // Check for verification test suite
    let has_verification = check_for_pattern(
        project_path,
        &["verification_test", "numerical_validation", "convergence_test"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Validation: analytical={}, tolerance={}, verification={}",
            has_analytical_tests, has_tolerance, has_verification
        ),
        data: None,
        files: Vec::new(),
    });

    let has_simulation = check_for_pattern(project_path, &["simulate", "numerical"]);
    item = apply_check_outcome(
        item,
        &[
            (!has_simulation, CheckOutcome::Pass),
            (has_analytical_tests && has_tolerance, CheckOutcome::Pass),
            (
                has_verification || has_analytical_tests,
                CheckOutcome::Partial("Some validation (verify tolerance)"),
            ),
            (true, CheckOutcome::Partial("Numerical code without analytical validation")),
        ],
    );

    item.finish_timed(start)
}
