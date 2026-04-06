//! Tests for Hypothesis-Driven Development checks.

use super::*;
use crate::falsification::types::{CheckStatus, Severity};
use std::path::PathBuf;

#[test]
fn test_evaluate_all_returns_13_items() {
    let path = PathBuf::from(".");
    let items = evaluate_all(&path);
    assert_eq!(items.len(), 13);
}

#[test]
fn test_all_items_have_tps_principle() {
    let path = PathBuf::from(".");
    let items = evaluate_all(&path);
    for item in items {
        assert!(!item.tps_principle.is_empty(), "Item {} missing TPS principle", item.id);
    }
}

#[test]
fn test_all_items_have_evidence() {
    let path = PathBuf::from(".");
    let items = evaluate_all(&path);
    for item in items {
        assert!(!item.evidence.is_empty(), "Item {} missing evidence", item.id);
    }
}

#[test]
fn test_hdd_01_hypothesis() {
    let path = PathBuf::from(".");
    let item = check_hypothesis_statement(&path);
    assert_eq!(item.id, "HDD-01");
    assert_eq!(item.severity, Severity::Major);
}

#[test]
fn test_hdd_03_reproducibility() {
    let path = PathBuf::from(".");
    let item = check_gold_reproducibility(&path);
    assert_eq!(item.id, "HDD-03");
    assert_eq!(item.severity, Severity::Critical);
}

#[test]
fn test_edd_01_equation() {
    let path = PathBuf::from(".");
    let item = check_equation_verification(&path);
    assert_eq!(item.id, "EDD-01");
    assert_eq!(item.severity, Severity::Major);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_hdd_02_baseline() {
    let path = PathBuf::from(".");
    let item = check_baseline_comparison(&path);
    assert_eq!(item.id, "HDD-02");
}

#[test]
fn test_hdd_04_random_seed() {
    let path = PathBuf::from(".");
    let item = check_random_seed_documentation(&path);
    assert_eq!(item.id, "HDD-04");
}

#[test]
fn test_hdd_05_containerization() {
    let path = PathBuf::from(".");
    let item = check_environment_containerization(&path);
    assert_eq!(item.id, "HDD-05");
    assert_eq!(item.severity, Severity::Major);
}

#[test]
fn test_hdd_06_data_version_control() {
    let path = PathBuf::from(".");
    let item = check_data_version_control(&path);
    assert_eq!(item.id, "HDD-06");
}

#[test]
fn test_hdd_07_statistical_significance() {
    let path = PathBuf::from(".");
    let item = check_statistical_significance(&path);
    assert_eq!(item.id, "HDD-07");
}

#[test]
fn test_hdd_08_ablation_study() {
    let path = PathBuf::from(".");
    let item = check_ablation_study(&path);
    assert_eq!(item.id, "HDD-08");
}

#[test]
fn test_hdd_09_negative_results() {
    let path = PathBuf::from(".");
    let item = check_negative_result_documentation(&path);
    assert_eq!(item.id, "HDD-09");
}

#[test]
fn test_hdd_10_metric_preregistration() {
    let path = PathBuf::from(".");
    let item = check_metric_preregistration(&path);
    assert_eq!(item.id, "HDD-10");
}

#[test]
fn test_edd_02_emc_completeness() {
    let path = PathBuf::from(".");
    let item = check_emc_completeness(&path);
    assert_eq!(item.id, "EDD-02");
}

#[test]
fn test_edd_03_numerical_analytical() {
    let path = PathBuf::from(".");
    let item = check_numerical_analytical_validation(&path);
    assert_eq!(item.id, "EDD-03");
}

#[test]
fn test_nonexistent_path() {
    let path = PathBuf::from("/nonexistent/path");
    let items = evaluate_all(&path);
    assert_eq!(items.len(), 13);
    // All should have a status (pass or fail)
    for item in &items {
        assert!(!item.id.is_empty());
    }
}

#[test]
fn test_temp_dir_with_docker() {
    let temp_dir = std::env::temp_dir().join("test_docker_env");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create a Dockerfile
    std::fs::write(temp_dir.join("Dockerfile"), "FROM rust:1.70").unwrap();

    let item = check_environment_containerization(&temp_dir);
    assert_eq!(item.id, "HDD-05");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_temp_dir_with_dvc() {
    let temp_dir = std::env::temp_dir().join("test_dvc_env");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join(".dvc")).unwrap();

    std::fs::write(temp_dir.join(".dvc/config"), "[core]").unwrap();

    let item = check_data_version_control(&temp_dir);
    assert_eq!(item.id, "HDD-06");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_temp_dir_with_experiment_tracking() {
    let temp_dir = std::env::temp_dir().join("test_mlflow_env");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();
    std::fs::create_dir_all(temp_dir.join("mlruns")).unwrap();

    let item = check_negative_result_documentation(&temp_dir);
    assert_eq!(item.id, "HDD-09");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_temp_dir_with_cargo_and_seed() {
    let temp_dir = std::env::temp_dir().join("test_seed_env");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create a Rust file with seed constant
    std::fs::write(temp_dir.join("src/lib.rs"), r#"const SEED: u64 = 42;"#).unwrap();

    let item = check_random_seed_documentation(&temp_dir);
    assert_eq!(item.id, "HDD-04");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_all_severities_present() {
    let path = PathBuf::from(".");
    let items = evaluate_all(&path);

    let has_major = items.iter().any(|i| i.severity == Severity::Major);
    let has_critical = items.iter().any(|i| i.severity == Severity::Critical);

    assert!(has_major || has_critical, "Expected at least one Major or Critical check");
}

// =========================================================================
// Additional Coverage Tests for Branch Coverage
// =========================================================================

#[test]
fn test_hdd_cov_001_hypothesis_with_pr_template() {
    let temp_dir = std::env::temp_dir().join("test_pr_template");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join(".github")).unwrap();

    // Create PR template with hypothesis section
    std::fs::write(
        temp_dir.join(".github/PULL_REQUEST_TEMPLATE.md"),
        "## Hypothesis\n\nDescribe your hypothesis here\n",
    )
    .unwrap();

    let item = check_hypothesis_statement(&temp_dir);
    assert_eq!(item.id, "HDD-01");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_002_hypothesis_template_without_section() {
    let temp_dir = std::env::temp_dir().join("test_pr_no_hypo");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join(".github")).unwrap();

    // Create PR template without hypothesis section
    std::fs::write(
        temp_dir.join(".github/PULL_REQUEST_TEMPLATE.md"),
        "## Description\n\nDescribe your changes\n",
    )
    .unwrap();

    let item = check_hypothesis_statement(&temp_dir);
    assert_eq!(item.id, "HDD-01");
    // Should be partial since template exists but no hypothesis section

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_003_baseline_with_benchmarks() {
    let temp_dir = std::env::temp_dir().join("test_benches");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("benches")).unwrap();
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Create a benchmark file and baseline test
    std::fs::write(temp_dir.join("benches/bench.rs"), "#[bench]\nfn bench() {}").unwrap();
    std::fs::write(temp_dir.join("src/lib.rs"), "fn compare_to_baseline() {}").unwrap();

    let item = check_baseline_comparison(&temp_dir);
    assert_eq!(item.id, "HDD-02");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_004_reproducibility_with_makefile() {
    let temp_dir = std::env::temp_dir().join("test_reproduce");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join(".github/workflows")).unwrap();

    std::fs::write(temp_dir.join("Makefile"), "reproduce:\n\techo reproduce").unwrap();
    std::fs::write(
        temp_dir.join(".github/workflows/ci.yml"),
        "name: CI\njobs:\n  reproduce:\n    runs-on: ubuntu-latest",
    )
    .unwrap();

    let item = check_gold_reproducibility(&temp_dir);
    assert_eq!(item.id, "HDD-03");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_005_seed_with_rng() {
    let temp_dir = std::env::temp_dir().join("test_seeded_rng");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
use rand::SeedableRng;
const RANDOM_SEED: u64 = 42;
fn create_rng() -> StdRng {
    StdRng::seed_from_u64(RANDOM_SEED)
}
"#,
    )
    .unwrap();

    let item = check_random_seed_documentation(&temp_dir);
    assert_eq!(item.id, "HDD-04");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_006_containerization_with_nix() {
    let temp_dir = std::env::temp_dir().join("test_nix");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(temp_dir.join("Cargo.lock"), "# cargo lock").unwrap();
    std::fs::write(temp_dir.join("flake.nix"), "{ inputs = {}; }").unwrap();

    let item = check_environment_containerization(&temp_dir);
    assert_eq!(item.id, "HDD-05");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_007_data_version_with_dvc() {
    let temp_dir = std::env::temp_dir().join("test_dvc_yaml");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(temp_dir.join("dvc.yaml"), "stages:\n  train:").unwrap();
    std::fs::write(temp_dir.join("src/lib.rs"), "fn load_data() {}").unwrap();

    let item = check_data_version_control(&temp_dir);
    assert_eq!(item.id, "HDD-06");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_008_stats_with_effect_size() {
    let temp_dir = std::env::temp_dir().join("test_stats");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
fn accuracy() -> f64 { 0.95 }
fn p_value() -> f64 { 0.01 }
fn effect_size() -> f64 { 0.8 }
fn cohen_d(a: &[f64], b: &[f64]) -> f64 { 0.0 }
"#,
    )
    .unwrap();

    let item = check_statistical_significance(&temp_dir);
    assert_eq!(item.id, "HDD-07");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_009_ablation_study() {
    let temp_dir = std::env::temp_dir().join("test_ablation");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
mod neural;
fn ablation() {}
fn transformer() {}
"#,
    )
    .unwrap();

    let item = check_ablation_study(&temp_dir);
    assert_eq!(item.id, "HDD-08");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_010_negative_docs_with_adr() {
    let temp_dir = std::env::temp_dir().join("test_adr");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("docs/adr")).unwrap();

    std::fs::write(temp_dir.join("docs/adr/001-failed.md"), "# Decision: Did not work\n").unwrap();

    let item = check_negative_result_documentation(&temp_dir);
    assert_eq!(item.id, "HDD-09");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_011_metric_prereg() {
    let temp_dir = std::env::temp_dir().join("test_prereg");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        "const PRE_REGISTRATION: &str = \"metrics defined\";\n",
    )
    .unwrap();

    let item = check_metric_preregistration(&temp_dir);
    assert_eq!(item.id, "HDD-10");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_edd_cov_001_equation_with_emc() {
    let temp_dir = std::env::temp_dir().join("test_emc");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("docs/emc")).unwrap();
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // Must have both EMC and math docs (derivation, proof, analytical_solution, LaTeX)
    std::fs::write(
        temp_dir.join("docs/emc/equation.md"),
        "# Governing Equation\n\n## Derivation\n\n## Proof\n",
    )
    .unwrap();
    std::fs::write(temp_dir.join("src/lib.rs"), "fn simulate() {}\n// derivation included")
        .unwrap();

    let item = check_equation_verification(&temp_dir);
    assert_eq!(item.id, "EDD-01");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_edd_cov_002_emc_completeness_full() {
    let temp_dir = std::env::temp_dir().join("test_emc_full");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("docs/emc")).unwrap();
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("docs/emc/model.md"),
        r#"# Model
## Governing Equations
## Validity Domain
## Derivation
## Stability Analysis
## Verification Tests
"#,
    )
    .unwrap();
    std::fs::write(temp_dir.join("src/lib.rs"), "fn simulate() {}").unwrap();

    let item = check_emc_completeness(&temp_dir);
    assert_eq!(item.id, "EDD-02");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_edd_cov_003_numerical_validation() {
    let temp_dir = std::env::temp_dir().join("test_numerical");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(
        temp_dir.join("src/lib.rs"),
        r#"
fn simulate() {}
fn analytical_solution(t: f64) -> f64 { t.exp() }
const TOLERANCE: f64 = 1e-6;
fn assert_relative_eq(a: f64, b: f64) {}
"#,
    )
    .unwrap();

    let item = check_numerical_analytical_validation(&temp_dir);
    assert_eq!(item.id, "EDD-03");
    assert!(matches!(item.status, CheckStatus::Pass));

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_012_ci_pattern_gitlab() {
    let temp_dir = std::env::temp_dir().join("test_gitlab_ci");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(temp_dir.join(".gitlab-ci.yml"), "stages:\n  - reproduce\n").unwrap();

    let result = helpers::check_ci_for_pattern(&temp_dir, &["reproduce"]);
    assert!(result);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_013_no_random_usage() {
    let temp_dir = std::env::temp_dir().join("test_no_random");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    // No random usage at all
    std::fs::write(temp_dir.join("src/lib.rs"), "fn deterministic() -> i32 { 42 }").unwrap();

    let item = check_random_seed_documentation(&temp_dir);
    assert_eq!(item.id, "HDD-04");
    assert!(matches!(item.status, CheckStatus::Pass)); // Should pass if no randomness used

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_014_lock_only_no_container() {
    let temp_dir = std::env::temp_dir().join("test_lock_only");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(temp_dir.join("Cargo.lock"), "# lock file").unwrap();
    // No Dockerfile or nix

    let item = check_environment_containerization(&temp_dir);
    assert_eq!(item.id, "HDD-05");
    // Should be partial - locked but no container

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_015_docker_compose() {
    let temp_dir = std::env::temp_dir().join("test_compose");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    std::fs::write(temp_dir.join("Cargo.lock"), "# lock").unwrap();
    std::fs::write(temp_dir.join("docker-compose.yaml"), "version: '3'\nservices:").unwrap();

    let item = check_environment_containerization(&temp_dir);
    assert_eq!(item.id, "HDD-05");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_016_check_pattern_config_files() {
    let temp_dir = std::env::temp_dir().join("test_pattern_config");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create config with pattern
    std::fs::write(temp_dir.join("config.yaml"), "metrics:\n  - accuracy\n  - F1\n").unwrap();

    let result = helpers::check_for_pattern(&temp_dir, &["metrics:"]);
    assert!(result);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_017_experiments_dir() {
    let temp_dir = std::env::temp_dir().join("test_experiments");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("experiments")).unwrap();

    let item = check_negative_result_documentation(&temp_dir);
    assert_eq!(item.id, "HDD-09");
    // Should be partial - experiments dir exists

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_hdd_cov_018_sensitivity_no_ablation() {
    let temp_dir = std::env::temp_dir().join("test_sensitivity");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(temp_dir.join("src/lib.rs"), "fn neural() {}\nfn grid_search() {}\n").unwrap();

    let item = check_ablation_study(&temp_dir);
    assert_eq!(item.id, "HDD-08");
    // Should be partial - sensitivity but no formal ablation

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_edd_cov_004_partial_emc() {
    let temp_dir = std::env::temp_dir().join("test_partial_emc");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(temp_dir.join("src")).unwrap();

    std::fs::write(temp_dir.join("src/lib.rs"), "fn simulate() {}\n// governing equation").unwrap();

    let item = check_equation_verification(&temp_dir);
    assert_eq!(item.id, "EDD-01");
    // Should be partial - has simulation and some docs

    let _ = std::fs::remove_dir_all(&temp_dir);
}
