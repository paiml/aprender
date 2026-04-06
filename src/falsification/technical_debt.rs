//! ML Technical Debt Prevention (Section 2)
//!
//! Implements MTD-01 through MTD-10 from the Popperian Falsification Checklist.
//! Based on Sculley et al. "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015).
//!
//! # Key Concepts
//!
//! - **CACE (Entanglement):** Changing Anything Changes Everything
//! - **Correction Cascades:** Models patching other models
//! - **Undeclared Consumers:** Unknown downstream dependencies
//! - **Feedback Loops:** Model outputs influencing future training

use std::path::Path;
use std::time::Instant;

use super::helpers::{apply_check_outcome, CheckOutcome};
use super::types::{CheckItem, CheckStatus, Evidence, EvidenceType, Severity};

/// Scan `src/**/*.rs` files, calling `detect` on each file's content to
/// accumulate pattern indicators. Returns a sorted, deduplicated list.
fn scan_src_files<F>(project_path: &Path, detect: F) -> Vec<&'static str>
where
    F: Fn(&str, &mut Vec<&'static str>),
{
    let mut indicators = Vec::new();
    let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) else {
        return indicators;
    };
    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        detect(&content, &mut indicators);
    }
    indicators.sort_unstable();
    indicators.dedup();
    indicators
}

/// Return `true` if any of the given relative paths exist under `base`.
fn path_exists_any(base: &Path, paths: &[&str]) -> bool {
    paths.iter().any(|p| base.join(p).exists())
}

/// Return `true` if the file at `path` exists and contains any of `patterns`.
fn file_contains_any(path: &Path, patterns: &[&str]) -> bool {
    std::fs::read_to_string(path).map(|c| patterns.iter().any(|p| c.contains(p))).unwrap_or(false)
}

/// Return `true` if the file at `path` exists and each pattern group has at
/// least one match.  Each inner slice is OR-ed; the outer slice is AND-ed.
fn file_contains_all(path: &Path, pattern_groups: &[&[&str]]) -> bool {
    std::fs::read_to_string(path)
        .map(|c| pattern_groups.iter().all(|group| group.iter().any(|p| c.contains(p))))
        .unwrap_or(false)
}

/// Evaluate all ML technical debt checks for a project.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_entanglement_detection(project_path),
        check_correction_cascade_prevention(project_path),
        check_undeclared_consumer_detection(project_path),
        check_data_dependency_freshness(project_path),
        check_pipeline_glue_code(project_path),
        check_configuration_debt(project_path),
        check_dead_code_elimination(project_path),
        check_abstraction_boundaries(project_path),
        check_feedback_loop_detection(project_path),
        check_technical_debt_quantification(project_path),
    ]
}

/// Classify isolation patterns in a single file's content.
fn classify_isolation_patterns(content: &str) -> Vec<&'static str> {
    let mut patterns = Vec::new();
    if content.contains("#[cfg(feature =")
        || content.contains("feature_enabled!")
        || content.contains("Feature::")
    {
        patterns.push("feature_flags");
    }
    if content.contains("impl<T>") && content.contains("T:") {
        patterns.push("generic_abstractions");
    }
    if content.contains("trait ") && content.contains("impl ") {
        patterns.push("trait_abstractions");
    }
    if content.contains("pub(crate)") || content.contains("pub(super)") {
        patterns.push("visibility_control");
    }
    patterns
}

/// Scan source files for feature isolation patterns.
fn scan_isolation_indicators(project_path: &Path) -> Vec<&'static str> {
    scan_src_files(project_path, |content, indicators| {
        indicators.extend(classify_isolation_patterns(content));
    })
}

/// MTD-01: Entanglement (CACE) Detection
///
/// **Claim:** Feature changes are isolated; changing one doesn't silently affect others.
///
/// **Rejection Criteria (Major):**
/// - Ablation study shows unexpected cross-feature impact
pub fn check_entanglement_detection(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-01",
        "Entanglement (CACE) Detection",
        "Feature changes are isolated; changing one doesn't silently affect others",
    )
    .with_severity(Severity::Major)
    .with_tps("Kaizen — root cause analysis");

    let isolation_indicators = scan_isolation_indicators(project_path);

    // Check for tests directory structure (indicates feature isolation)
    let tests_dir = project_path.join("tests");
    let has_test_structure = tests_dir.exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Isolation indicators: {:?}, test_structure={}",
            isolation_indicators, has_test_structure
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (isolation_indicators.len() >= 3, CheckOutcome::Pass),
            (
                !isolation_indicators.is_empty(),
                CheckOutcome::Partial("Partial feature isolation patterns detected"),
            ),
            (true, CheckOutcome::Partial("Consider adding feature isolation patterns")),
        ],
    );

    item.finish_timed(start)
}

/// Scan source files for correction cascade patterns.
fn scan_cascade_indicators(project_path: &Path) -> Vec<&'static str> {
    scan_src_files(project_path, |content, indicators| {
        if content.contains("post_process") && content.contains("model") {
            indicators.push("post_processing");
        }
        if content.contains("correction") || content.contains("fix_output") {
            indicators.push("correction_code");
        }
        if content.contains("ensemble") {
            indicators.push("ensemble (intentional)");
        }
    })
}

/// MTD-02: Correction Cascade Prevention
///
/// **Claim:** No model exists solely to correct another model's errors.
///
/// **Rejection Criteria (Major):**
/// - Model B exists only to patch Model A outputs
pub fn check_correction_cascade_prevention(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-02",
        "Correction Cascade Prevention",
        "No model exists solely to correct another model's errors",
    )
    .with_severity(Severity::Major)
    .with_tps("Kaizen — fix root cause in Model A");

    let cascade_indicators = scan_cascade_indicators(project_path);

    // Check for pipeline architecture documentation
    let has_architecture_doc = path_exists_any(
        project_path,
        &["docs/architecture.md", "ARCHITECTURE.md", "docs/pipeline.md"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Cascade indicators: {:?}, architecture_doc={}",
            cascade_indicators, has_architecture_doc
        ),
        data: None,
        files: Vec::new(),
    });

    let no_cascades =
        cascade_indicators.is_empty() || cascade_indicators.iter().all(|s| s.contains("ensemble"));
    item = apply_check_outcome(
        item,
        &[
            (no_cascades, CheckOutcome::Pass),
            (
                has_architecture_doc,
                CheckOutcome::Partial(
                    "Potential cascades - verify intentional in architecture doc",
                ),
            ),
            (true, CheckOutcome::Partial("Review for correction cascades")),
        ],
    );

    item.finish_timed(start)
}

/// MTD-03: Undeclared Consumer Detection
///
/// **Claim:** All model consumers are documented and access-controlled.
///
/// **Rejection Criteria (Major):**
/// - Any access from unregistered consumer
pub fn check_undeclared_consumer_detection(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-03",
        "Undeclared Consumer Detection",
        "All model consumers are documented and access-controlled",
    )
    .with_severity(Severity::Major)
    .with_tps("Visibility across downstream supply chain");

    // Check for API documentation
    let has_api_docs = path_exists_any(project_path, &["docs/api.md", "API.md"]);

    // Check for public API visibility control
    let mut pub_items = 0;
    let mut pub_crate_items = 0;

    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                pub_items += content.matches("pub fn ").count();
                pub_items += content.matches("pub struct ").count();
                pub_items += content.matches("pub enum ").count();
                pub_crate_items += content.matches("pub(crate)").count();
            }
        }
    }

    // Check for re-exports in lib.rs
    let lib_rs = project_path.join("src/lib.rs");
    let has_controlled_exports = file_contains_any(&lib_rs, &["pub use ", "pub mod "]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Consumer control: api_docs={}, pub_items={}, pub_crate={}, controlled_exports={}",
            has_api_docs, pub_items, pub_crate_items, has_controlled_exports
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_controlled_exports && pub_crate_items > 0, CheckOutcome::Pass),
            (
                has_controlled_exports,
                CheckOutcome::Partial(
                    "Controlled exports but consider pub(crate) for internal items",
                ),
            ),
            (true, CheckOutcome::Partial("Add explicit API boundary control")),
        ],
    );

    item.finish_timed(start)
}

/// MTD-04: Data Dependency Freshness
///
/// **Claim:** Training data dependencies are current and maintained.
///
/// **Rejection Criteria (Major):**
/// - Data source unchanged for >N days without explicit acknowledgment
pub fn check_data_dependency_freshness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-04",
        "Data Dependency Freshness",
        "Training data dependencies are current and maintained",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Inventory) — prevent data staleness");

    // Check for data versioning tools
    let has_dvc = path_exists_any(project_path, &[".dvc", "dvc.yaml"]);
    let has_data_dir = project_path.join("data").exists();

    // Check Cargo.toml for data-related dependencies
    let cargo_toml = project_path.join("Cargo.toml");
    let has_data_deps = file_contains_any(&cargo_toml, &["alimentar", "parquet", "arrow", "csv"]);

    // Check for data documentation
    let has_data_docs =
        path_exists_any(project_path, &["docs/data.md", "DATA.md", "data/README.md"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Data freshness: dvc={}, data_dir={}, data_deps={}, data_docs={}",
            has_dvc, has_data_dir, has_data_deps, has_data_docs
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_dvc && has_data_docs, CheckOutcome::Pass),
            (has_dvc || has_data_docs, CheckOutcome::Partial("Partial data management setup")),
            (!has_data_dir && !has_data_deps, CheckOutcome::Pass),
            (true, CheckOutcome::Partial("Consider adding data versioning (DVC or similar)")),
        ],
    );

    item.finish_timed(start)
}

/// Scan source files for standardization patterns
fn scan_standardization_indicators(project_path: &Path) -> Vec<&'static str> {
    scan_src_files(project_path, |content, indicators| {
        if content.contains("trait Pipeline") || content.contains("impl Pipeline") {
            indicators.push("pipeline_trait");
        }
        if content.contains("Stage") || content.contains("Step") {
            indicators.push("stage_abstraction");
        }
        if content.contains("Builder") {
            indicators.push("builder_pattern");
        }
        if content.contains("impl From<") || content.contains("impl Into<") {
            indicators.push("type_conversions");
        }
    })
}

/// MTD-05: Pipeline Glue Code Minimization
///
/// **Claim:** Pipeline code uses standardized connectors, not ad-hoc scripts.
///
/// **Rejection Criteria (Major):**
/// - >10% of pipeline LOC is custom data transformation
pub fn check_pipeline_glue_code(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-05",
        "Pipeline Glue Code Minimization",
        "Pipeline code uses standardized connectors, not ad-hoc scripts",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Motion) — standardization");

    let has_pipeline_module =
        path_exists_any(project_path, &["src/pipeline.rs", "src/pipeline/mod.rs"]);

    let standardization_indicators = scan_standardization_indicators(project_path);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Standardization: pipeline_module={}, indicators={:?}",
            has_pipeline_module, standardization_indicators
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_pipeline_module && standardization_indicators.len() >= 2, CheckOutcome::Pass),
            (
                has_pipeline_module || !standardization_indicators.is_empty(),
                CheckOutcome::Partial("Partial pipeline standardization"),
            ),
            (true, CheckOutcome::Partial("Consider standardized pipeline abstractions")),
        ],
    );

    item.finish_timed(start)
}

/// MTD-06: Configuration Debt Prevention
///
/// **Claim:** All hyperparameters and configurations are version-controlled.
///
/// **Rejection Criteria (Major):**
/// - Any configuration not in version control
pub fn check_configuration_debt(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-06",
        "Configuration Debt Prevention",
        "All hyperparameters and configurations are version-controlled",
    )
    .with_severity(Severity::Major)
    .with_tps("Reproducibility requirement");

    // Check for config files
    let config_files = [
        project_path.join("config"),
        project_path.join("configs"),
        project_path.join("batuta.toml"),
        project_path.join("config.toml"),
        project_path.join("settings.toml"),
    ];

    let mut config_found = Vec::new();
    for path in &config_files {
        if path.exists() {
            config_found.push(path.file_name().unwrap_or_default().to_string_lossy().to_string());
        }
    }

    // Check for typed config structs
    let has_config_struct = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| {
                        (c.contains("struct") && c.to_lowercase().contains("config"))
                            || c.contains("Deserialize")
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    // Check for environment variable documentation
    let has_env_docs = path_exists_any(project_path, &[".env.example", ".env.template"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Config: files={:?}, typed_struct={}, env_docs={}",
            config_found, has_config_struct, has_env_docs
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_config_struct && !config_found.is_empty(), CheckOutcome::Pass),
            (
                has_config_struct || !config_found.is_empty(),
                CheckOutcome::Partial("Configuration exists but consider typed structs"),
            ),
            (true, CheckOutcome::Partial("Add explicit configuration management")),
        ],
    );

    item.finish_timed(start)
}

/// MTD-07: Dead Code Elimination
///
/// **Claim:** No unused model code paths exist in production.
///
/// **Rejection Criteria (Major):**
/// - Any unreachable code path in model inference
pub fn check_dead_code_elimination(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-07",
        "Dead Code Elimination",
        "No unused model code paths exist in production",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Inventory) — code hygiene");

    // Check for dead code warnings in lib.rs
    let lib_rs = project_path.join("src/lib.rs");
    let main_rs = project_path.join("src/main.rs");
    let root_files = [&lib_rs, &main_rs];

    let allows_dead_code =
        root_files.iter().any(|p| file_contains_any(p, &["#![allow(dead_code)]"]));

    let denies_dead_code = root_files
        .iter()
        .any(|p| file_contains_any(p, &["#![deny(dead_code)]", "#![warn(dead_code)]"]));

    // Check CI for cargo udeps
    let has_udeps_ci = check_ci_for_content(project_path, "udeps");

    // Check Makefile for cleanup targets
    let makefile = project_path.join("Makefile");
    let has_cleanup = file_contains_any(&makefile, &["clean", "udeps"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Dead code: allows={}, denies={}, udeps_ci={}, cleanup={}",
            allows_dead_code, denies_dead_code, has_udeps_ci, has_cleanup
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (denies_dead_code || has_udeps_ci, CheckOutcome::Pass),
            (
                !allows_dead_code,
                CheckOutcome::Partial("Default dead code warnings (consider explicit deny)"),
            ),
            (
                has_cleanup,
                CheckOutcome::Partial(
                    "Dead code allowed (development phase), cleanup targets available",
                ),
            ),
            (
                true,
                CheckOutcome::Partial(
                    "Dead code warnings suppressed - verify intentional for development",
                ),
            ),
        ],
    );

    item.finish_timed(start)
}

/// MTD-08: Abstraction Boundary Verification
///
/// **Claim:** ML code respects clean abstraction boundaries.
///
/// **Rejection Criteria (Major):**
/// - Business logic leaks into model code or vice versa
pub fn check_abstraction_boundaries(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-08",
        "Abstraction Boundary Verification",
        "ML code respects clean abstraction boundaries",
    )
    .with_severity(Severity::Major)
    .with_tps("Clean Architecture principle");

    // Check for module structure
    let src_dir = project_path.join("src");
    let mut module_count = 0;
    let mut has_mod_files = false;

    if src_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&src_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    module_count += 1;
                } else if path.file_name().map(|n| n == "mod.rs").unwrap_or(false) {
                    has_mod_files = true;
                }
            }
        }
    }

    // Check for layer-based organization
    let common_layers = ["api", "domain", "service", "repository", "model", "types"];
    let layer_dirs: Vec<_> = common_layers
        .iter()
        .filter(|layer| src_dir.join(layer).exists())
        .map(|s| (*s).to_string())
        .collect();

    // Check for trait-based boundaries
    let has_trait_boundaries = glob::glob(&format!("{}/src/**/*.rs", project_path.display()))
        .ok()
        .map(|entries| {
            entries
                .flatten()
                .filter(|p| {
                    std::fs::read_to_string(p)
                        .ok()
                        .map(|c| c.contains("pub trait "))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Boundaries: modules={}, mod_files={}, layers={:?}, traits={}",
            module_count, has_mod_files, layer_dirs, has_trait_boundaries
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (module_count >= 3 && has_trait_boundaries >= 2, CheckOutcome::Pass),
            (
                module_count >= 2 || has_trait_boundaries > 0,
                CheckOutcome::Partial("Partial abstraction boundaries"),
            ),
            (true, CheckOutcome::Partial("Consider module-based architecture")),
        ],
    );

    item.finish_timed(start)
}

/// MTD-09: Feedback Loop Detection
///
/// **Claim:** No hidden feedback loops where model outputs influence future training.
///
/// **Rejection Criteria (Major):**
/// - Model output appears in training data pipeline
pub fn check_feedback_loop_detection(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-09",
        "Feedback Loop Detection",
        "No hidden feedback loops where model outputs influence future training",
    )
    .with_severity(Severity::Major)
    .with_tps("Entanglement prevention");

    // Check for training/inference separation
    let has_training_module =
        path_exists_any(project_path, &["src/training.rs", "src/train.rs", "src/training/mod.rs"]);

    let has_inference_module = path_exists_any(
        project_path,
        &["src/inference.rs", "src/infer.rs", "src/serve.rs", "src/inference/mod.rs"],
    );

    // Check for feedback loop documentation
    let has_feedback_docs = glob::glob(&format!("{}/docs/**/*.md", project_path.display()))
        .ok()
        .map(|entries| {
            entries.flatten().any(|p| {
                std::fs::read_to_string(&p)
                    .ok()
                    .map(|c| c.contains("feedback") || c.contains("loop"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    // Check Cargo.toml for ML training vs inference separation
    let cargo_toml = project_path.join("Cargo.toml");
    let has_feature_separation =
        file_contains_all(&cargo_toml, &[&["training", "train"], &["inference", "serve"]]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Feedback loops: training_module={}, inference_module={}, docs={}, features={}",
            has_training_module, has_inference_module, has_feedback_docs, has_feature_separation
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_training_module && has_inference_module, CheckOutcome::Pass),
            (
                !has_training_module && !has_inference_module,
                CheckOutcome::Partial("No explicit training/inference separation (verify N/A)"),
            ),
            (true, CheckOutcome::Partial("Consider separating training and inference paths")),
        ],
    );

    item.finish_timed(start)
}

/// MTD-10: Technical Debt Quantification
///
/// **Claim:** ML technical debt is measured and trending downward.
///
/// **Rejection Criteria (Major):**
/// - No TDG measurement
/// - TDG score declining over 3 releases
pub fn check_technical_debt_quantification(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MTD-10",
        "Technical Debt Quantification",
        "ML technical debt is measured and trending downward",
    )
    .with_severity(Severity::Major)
    .with_tps("Kaizen — continuous measurement");

    // Check for PMAT or similar tools
    let has_pmat_ci = check_ci_for_content(project_path, "pmat");
    let has_tdg_tracking = path_exists_any(project_path, &["tdg_history.json", "metrics/tdg.json"]);

    // Check Makefile for quality metrics
    let makefile = project_path.join("Makefile");
    let has_quality_targets = file_contains_any(&makefile, &["quality", "metrics", "pmat", "tdg"]);

    // Check for code quality CI
    let has_quality_ci = check_ci_for_content(project_path, "quality")
        || check_ci_for_content(project_path, "lint")
        || check_ci_for_content(project_path, "clippy");

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Debt tracking: pmat_ci={}, tdg_history={}, quality_targets={}, quality_ci={}",
            has_pmat_ci, has_tdg_tracking, has_quality_targets, has_quality_ci
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_pmat_ci || has_tdg_tracking, CheckOutcome::Pass),
            (
                has_quality_targets && has_quality_ci,
                CheckOutcome::Partial("Quality checks exist, consider TDG tracking"),
            ),
            (
                has_quality_ci,
                CheckOutcome::Partial("CI quality checks, consider formal debt tracking"),
            ),
            (true, CheckOutcome::Fail("No technical debt quantification")),
        ],
    );

    item.finish_timed(start)
}

/// Helper: Check if content exists in any CI configuration
fn check_ci_for_content(project_path: &Path, content: &str) -> bool {
    let ci_configs = [
        project_path.join(".github/workflows/ci.yml"),
        project_path.join(".github/workflows/test.yml"),
        project_path.join(".github/workflows/rust.yml"),
        project_path.join(".github/workflows/quality.yml"),
    ];

    for ci_path in &ci_configs {
        if ci_path.exists() {
            if let Ok(file_content) = std::fs::read_to_string(ci_path) {
                if file_content.contains(content) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
#[path = "technical_debt_tests.rs"]
mod tests;
