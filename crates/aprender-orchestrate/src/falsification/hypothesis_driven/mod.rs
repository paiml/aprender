//! Section 3: Hypothesis-Driven & Equation-Driven Development (HDD-01 to EDD-03)
//!
//! Implements Scientific Method integration for ML development.
//!
//! # TPS Principles
//!
//! - **Scientific Method**: Falsifiable hypotheses, pre-registration
//! - **Muda**: Overprocessing prevention via baseline comparison
//! - **Kaizen**: Learning from negative results

mod edd;
mod hdd_reproducibility;
mod hdd_scientific;
mod helpers;

#[cfg(test)]
mod tests;

use super::types::CheckItem;
use std::path::Path;

// Re-export all check functions for public API
pub use edd::{
    check_emc_completeness, check_equation_verification, check_numerical_analytical_validation,
};
pub use hdd_reproducibility::{
    check_baseline_comparison, check_data_version_control, check_environment_containerization,
    check_gold_reproducibility, check_hypothesis_statement, check_random_seed_documentation,
};
pub use hdd_scientific::{
    check_ablation_study, check_metric_preregistration, check_negative_result_documentation,
    check_statistical_significance,
};

/// Evaluate all Hypothesis-Driven Development checks.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_hypothesis_statement(project_path),
        check_baseline_comparison(project_path),
        check_gold_reproducibility(project_path),
        check_random_seed_documentation(project_path),
        check_environment_containerization(project_path),
        check_data_version_control(project_path),
        check_statistical_significance(project_path),
        check_ablation_study(project_path),
        check_negative_result_documentation(project_path),
        check_metric_preregistration(project_path),
        check_equation_verification(project_path),
        check_emc_completeness(project_path),
        check_numerical_analytical_validation(project_path),
    ]
}
