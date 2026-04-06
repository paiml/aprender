//! HDD Reproducibility Checks (HDD-01 through HDD-06)
//!
//! These checks focus on reproducibility and environment requirements:
//! - Hypothesis statements
//! - Baseline comparisons
//! - Gold standard reproducibility
//! - Random seed documentation
//! - Environment containerization
//! - Data version control

use super::helpers::{check_ci_for_pattern, check_for_pattern};
use crate::falsification::helpers::{apply_check_outcome, CheckOutcome};
use crate::falsification::types::{CheckItem, Evidence, EvidenceType, Severity};
use std::path::Path;
use std::time::Instant;

/// HDD-01: Hypothesis Statement Requirement
///
/// **Claim:** Every model change PR includes falsifiable hypothesis.
///
/// **Rejection Criteria (Major):**
/// - Model PR without "Hypothesis:" section
pub fn check_hypothesis_statement(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-01",
        "Hypothesis Statement Requirement",
        "Model change PRs include falsifiable hypothesis",
    )
    .with_severity(Severity::Major)
    .with_tps("Scientific Method integration");

    // Check for PR template with hypothesis section
    let pr_templates = [
        project_path.join(".github/PULL_REQUEST_TEMPLATE.md"),
        project_path.join(".github/pull_request_template.md"),
        project_path.join("docs/PR_TEMPLATE.md"),
    ];

    let has_pr_template = pr_templates.iter().any(|p| p.exists());
    let has_hypothesis_section = pr_templates.iter().filter(|p| p.exists()).any(|p| {
        std::fs::read_to_string(p)
            .ok()
            .map(|c| {
                c.to_lowercase().contains("hypothesis")
                    || c.contains("## Hypothesis")
                    || c.contains("### Hypothesis")
            })
            .unwrap_or(false)
    });

    // Check for hypothesis documentation in codebase
    let has_hypothesis_docs = check_for_pattern(
        project_path,
        &["hypothesis:", "Hypothesis:", "H0:", "H1:", "null_hypothesis"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Hypothesis: pr_template={}, hypothesis_section={}, docs={}",
            has_pr_template, has_hypothesis_section, has_hypothesis_docs
        ),
        data: None,
        files: Vec::new(),
    });

    let is_ml_project = check_for_pattern(project_path, &["model", "train", "predict"]);
    item = apply_check_outcome(
        item,
        &[
            (has_hypothesis_section, CheckOutcome::Pass),
            (
                has_pr_template || has_hypothesis_docs,
                CheckOutcome::Partial("PR template exists but missing hypothesis section"),
            ),
            (!is_ml_project, CheckOutcome::Pass),
            (true, CheckOutcome::Partial("No hypothesis requirement in PR workflow")),
        ],
    );

    item.finish_timed(start)
}

/// HDD-02: Baseline Comparison Requirement
///
/// **Claim:** Complex models must beat simple baselines to be merged.
///
/// **Rejection Criteria (Major):**
/// - Transformer without RF baseline, <5% improvement without justification
pub fn check_baseline_comparison(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-02",
        "Baseline Comparison Requirement",
        "Complex models beat simple baselines",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Overprocessing) prevention");

    // Check for baseline comparison in tests or benchmarks
    let has_baseline_tests = check_for_pattern(
        project_path,
        &["baseline", "Baseline", "simple_model", "compare_to_baseline"],
    );

    // Check for benchmark infrastructure
    let has_benchmarks = project_path.join("benches").exists()
        || check_for_pattern(project_path, &["#[bench]", "criterion", "benchmark"]);

    // Check for model comparison documentation
    let has_comparison_docs =
        check_for_pattern(project_path, &["vs_baseline", "improvement_over", "comparison"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Baseline: tests={}, benchmarks={}, comparison_docs={}",
            has_baseline_tests, has_benchmarks, has_comparison_docs
        ),
        data: None,
        files: Vec::new(),
    });

    let is_ml = check_for_pattern(project_path, &["neural", "transformer", "deep_learning"]);
    item = apply_check_outcome(
        item,
        &[
            (has_baseline_tests && has_benchmarks, CheckOutcome::Pass),
            (
                has_baseline_tests || has_comparison_docs,
                CheckOutcome::Partial("Some baseline comparison infrastructure"),
            ),
            (!is_ml, CheckOutcome::Pass),
            (true, CheckOutcome::Partial("Complex models without baseline comparison")),
        ],
    );

    item.finish_timed(start)
}

/// HDD-03: Gold Standard Reproducibility
///
/// **Claim:** `make reproduce` recreates training results from scratch.
///
/// **Rejection Criteria (Critical):**
/// - Build fails from clean state, metrics differ >1%
pub fn check_gold_reproducibility(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-03",
        "Gold Standard Reproducibility",
        "make reproduce recreates training results",
    )
    .with_severity(Severity::Critical)
    .with_tps("Scientific reproducibility");

    // Check for reproduce target in Makefile
    let makefile = project_path.join("Makefile");
    let has_reproduce_target = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("reproduce:") || c.contains("reproduce :"))
        .unwrap_or(false);

    // Check for reproduction documentation
    let has_repro_docs =
        check_for_pattern(project_path, &["REPRODUCIBILITY", "reproduce", "replication"]);

    // Check for CI reproduction
    let has_ci_repro = check_ci_for_pattern(project_path, &["reproduce", "replication"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Reproducibility: make_target={}, docs={}, ci={}",
            has_reproduce_target, has_repro_docs, has_ci_repro
        ),
        data: None,
        files: Vec::new(),
    });

    let is_ml = check_for_pattern(project_path, &["train", "model", "weights"]);
    item = apply_check_outcome(
        item,
        &[
            (has_reproduce_target && has_ci_repro, CheckOutcome::Pass),
            (
                has_reproduce_target || has_repro_docs,
                CheckOutcome::Partial("Reproduction target exists (not in CI)"),
            ),
            (!is_ml, CheckOutcome::Pass),
            (true, CheckOutcome::Partial("No reproduction infrastructure")),
        ],
    );

    item.finish_timed(start)
}

/// HDD-04: Random Seed Documentation
///
/// **Claim:** All stochastic operations have documented, pinned seeds.
///
/// **Rejection Criteria (Major):**
/// - Any stochastic operation without explicit seed
pub fn check_random_seed_documentation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-04",
        "Random Seed Documentation",
        "Stochastic operations have pinned seeds",
    )
    .with_severity(Severity::Major)
    .with_tps("Deterministic reproducibility");

    // Check for seed pinning in code
    let has_seed_pinning =
        check_for_pattern(project_path, &["seed", "Seed", "RANDOM_SEED", "rng_seed", "set_seed"]);

    // Check for StdRng usage (seeded RNG)
    let has_seeded_rng =
        check_for_pattern(project_path, &["StdRng", "SeedableRng", "from_seed", "seed_from_u64"]);

    // Check for seed documentation
    let has_seed_docs =
        check_for_pattern(project_path, &["seed=", "SEED:", "random seed", "deterministic"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Seeds: pinning={}, seeded_rng={}, docs={}",
            has_seed_pinning, has_seeded_rng, has_seed_docs
        ),
        data: None,
        files: Vec::new(),
    });

    let uses_random = check_for_pattern(project_path, &["rand::", "random", "Rng", "thread_rng"]);
    item = apply_check_outcome(
        item,
        &[
            (!uses_random, CheckOutcome::Pass),
            (has_seeded_rng && has_seed_pinning, CheckOutcome::Pass),
            (has_seed_pinning, CheckOutcome::Partial("Seed usage found (verify documentation)")),
            (true, CheckOutcome::Partial("Random usage without explicit seed pinning")),
        ],
    );

    item.finish_timed(start)
}

/// HDD-05: Environment Containerization
///
/// **Claim:** Training environment is fully containerized and versioned.
///
/// **Rejection Criteria (Major):**
/// - Dockerfile missing, unpinned dependencies
pub fn check_environment_containerization(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-05",
        "Environment Containerization",
        "Environment fully containerized and versioned",
    )
    .with_severity(Severity::Major)
    .with_tps("Silver -> Gold reproducibility");

    // Check for Dockerfile
    let has_dockerfile =
        project_path.join("Dockerfile").exists() || project_path.join("docker/Dockerfile").exists();

    // Check for docker-compose
    let has_compose = project_path.join("docker-compose.yml").exists()
        || project_path.join("docker-compose.yaml").exists();

    // Check for Cargo.lock (Rust dependency pinning)
    let has_lock_file = project_path.join("Cargo.lock").exists();

    // Check for nix/devenv
    let has_nix = project_path.join("flake.nix").exists()
        || project_path.join("shell.nix").exists()
        || project_path.join("devenv.nix").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Container: dockerfile={}, compose={}, lock={}, nix={}",
            has_dockerfile, has_compose, has_lock_file, has_nix
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            ((has_dockerfile || has_nix) && has_lock_file, CheckOutcome::Pass),
            (has_lock_file, CheckOutcome::Partial("Dependencies locked but no containerization")),
            (true, CheckOutcome::Partial("No environment containerization")),
        ],
    );

    item.finish_timed(start)
}

/// HDD-06: Data Version Control
///
/// **Claim:** Training data is versioned with content-addressable storage.
///
/// **Rejection Criteria (Major):**
/// - Any data modification not captured in version control
pub fn check_data_version_control(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-06",
        "Data Version Control",
        "Training data versioned with content-addressable storage",
    )
    .with_severity(Severity::Major)
    .with_tps("Reproducibility requirement");

    // Check for DVC
    let has_dvc = project_path.join(".dvc").exists() || project_path.join("dvc.yaml").exists();

    // Check for data versioning patterns
    let has_data_versioning =
        check_for_pattern(project_path, &["data_version", "dataset_hash", "content_hash"]);

    // Check for data directory with hashes
    let data_dir = project_path.join("data");
    let has_data_dir = data_dir.exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Data versioning: dvc={}, versioning_code={}, data_dir={}",
            has_dvc, has_data_versioning, has_data_dir
        ),
        data: None,
        files: Vec::new(),
    });

    let uses_data = check_for_pattern(project_path, &["load_data", "Dataset", "DataLoader"]);
    item = apply_check_outcome(
        item,
        &[
            (!uses_data, CheckOutcome::Pass),
            (has_dvc, CheckOutcome::Pass),
            (has_data_versioning, CheckOutcome::Partial("Data versioning patterns found (no DVC)")),
            (true, CheckOutcome::Partial("Data handling without version control")),
        ],
    );

    item.finish_timed(start)
}
