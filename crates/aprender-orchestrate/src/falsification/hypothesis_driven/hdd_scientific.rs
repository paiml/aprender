//! HDD Scientific Method Checks (HDD-07 through HDD-10)
//!
//! These checks focus on scientific rigor:
//! - Statistical significance
//! - Ablation studies
//! - Negative result documentation
//! - Metric pre-registration

use super::helpers::check_for_pattern;
use crate::falsification::helpers::{apply_check_outcome, CheckOutcome};
use crate::falsification::types::{CheckItem, Evidence, EvidenceType, Severity};
use std::path::Path;
use std::time::Instant;

/// HDD-07: Statistical Significance Requirement
///
/// **Claim:** Performance claims include statistical significance tests.
///
/// **Rejection Criteria (Major):**
/// - p >= 0.05, no confidence interval
pub fn check_statistical_significance(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-07",
        "Statistical Significance Requirement",
        "Performance claims include significance tests",
    )
    .with_severity(Severity::Major)
    .with_tps("Scientific rigor");

    // Check for statistical testing
    let has_stats = check_for_pattern(
        project_path,
        &["p_value", "p-value", "confidence_interval", "t_test", "significance"],
    );

    // Check for effect size
    let has_effect_size =
        check_for_pattern(project_path, &["effect_size", "cohen_d", "glass_delta", "hedges_g"]);

    // Check for statistical library
    let has_stats_lib =
        check_for_pattern(project_path, &["statrs", "statistical", "hypothesis_test"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Statistics: testing={}, effect_size={}, lib={}",
            has_stats, has_effect_size, has_stats_lib
        ),
        data: None,
        files: Vec::new(),
    });

    let has_perf_claims =
        check_for_pattern(project_path, &["accuracy", "F1", "precision", "recall"]);
    item = apply_check_outcome(
        item,
        &[
            (!has_perf_claims, CheckOutcome::Pass),
            (has_stats && has_effect_size, CheckOutcome::Pass),
            (has_stats, CheckOutcome::Partial("Statistical testing (missing effect size)")),
            (true, CheckOutcome::Partial("Performance metrics without significance testing")),
        ],
    );

    item.finish_timed(start)
}

/// HDD-08: Ablation Study Requirement
///
/// **Claim:** Multi-component changes include ablation studies.
///
/// **Rejection Criteria (Major):**
/// - >2 model changes without per-component analysis
pub fn check_ablation_study(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-08",
        "Ablation Study Requirement",
        "Multi-component changes include ablation studies",
    )
    .with_severity(Severity::Major)
    .with_tps("Scientific Method - isolation of variables");

    // Check for ablation documentation
    let has_ablation = check_for_pattern(
        project_path,
        &["ablation", "Ablation", "component_analysis", "feature_importance"],
    );

    // Check for sensitivity analysis
    let has_sensitivity =
        check_for_pattern(project_path, &["sensitivity", "hyperparameter_sweep", "grid_search"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Ablation: studies={}, sensitivity={}", has_ablation, has_sensitivity),
        data: None,
        files: Vec::new(),
    });

    let has_complex_models =
        check_for_pattern(project_path, &["neural", "transformer", "ensemble", "multi_layer"]);
    item = apply_check_outcome(
        item,
        &[
            (!has_complex_models, CheckOutcome::Pass),
            (has_ablation, CheckOutcome::Pass),
            (has_sensitivity, CheckOutcome::Partial("Sensitivity analysis (no formal ablation)")),
            (true, CheckOutcome::Partial("Complex models without ablation studies")),
        ],
    );

    item.finish_timed(start)
}

/// HDD-09: Negative Result Documentation
///
/// **Claim:** Failed experiments are documented, not just successes.
///
/// **Rejection Criteria (Minor):**
/// - Experiment log shows only successful attempts
pub fn check_negative_result_documentation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("HDD-09", "Negative Result Documentation", "Failed experiments documented")
            .with_severity(Severity::Minor)
            .with_tps("Kaizen - learning from failures");

    // Check for experiment logs
    let has_experiment_log = project_path.join("experiments/").exists()
        || project_path.join("logs/experiments/").exists()
        || check_for_pattern(project_path, &["experiment_log", "run_history"]);

    // Check for negative result documentation
    let has_negative_docs = check_for_pattern(
        project_path,
        &["failed_experiment", "negative_result", "did_not_work", "unsuccessful"],
    );

    // Check for ADR (Architecture Decision Records)
    let has_adr =
        project_path.join("docs/adr/").exists() || project_path.join("docs/decisions/").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Negative results: log={}, docs={}, adr={}",
            has_experiment_log, has_negative_docs, has_adr
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_negative_docs || has_adr, CheckOutcome::Pass),
            (
                has_experiment_log,
                CheckOutcome::Partial("Experiment logging (check for negative results)"),
            ),
            (true, CheckOutcome::Partial("No negative result documentation")),
        ],
    );

    item.finish_timed(start)
}

/// HDD-10: Pre-registration of Metrics
///
/// **Claim:** Evaluation metrics defined before experimentation.
///
/// **Rejection Criteria (Minor):**
/// - Metric definition after experiment commit
pub fn check_metric_preregistration(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "HDD-10",
        "Pre-registration of Metrics",
        "Metrics defined before experimentation",
    )
    .with_severity(Severity::Minor)
    .with_tps("Scientific pre-registration");

    // Check for metrics definition in config
    let has_metric_config =
        check_for_pattern(project_path, &["metrics:", "evaluation_metrics", "target_metric"]);

    // Check for pre-registration documentation
    let has_prereg = check_for_pattern(
        project_path,
        &["pre_registration", "preregistration", "planned_metrics"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Pre-registration: config={}, docs={}", has_metric_config, has_prereg),
        data: None,
        files: Vec::new(),
    });

    let is_ml = check_for_pattern(project_path, &["accuracy", "loss", "evaluate"]);
    item = apply_check_outcome(
        item,
        &[
            (has_prereg, CheckOutcome::Pass),
            (
                has_metric_config,
                CheckOutcome::Partial("Metrics in config (verify pre-registration)"),
            ),
            (!is_ml, CheckOutcome::Pass),
            (true, CheckOutcome::Partial("No metric pre-registration")),
        ],
    );

    item.finish_timed(start)
}
