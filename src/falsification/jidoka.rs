//! Jidoka Automated Gates (Section 7)
//!
//! Implements JA-01 through JA-10 from the Popperian Falsification Checklist.
//! Focus: CI/CD circuit breakers, automated quality enforcement.
//!
//! # Jidoka Principle
//!
//! "Automation with human intelligence" - The CI/CD pipeline must detect
//! abnormalities and stop automatically before human review begins.

use std::path::Path;
use std::time::Instant;

use super::helpers::{apply_check_outcome, CheckOutcome};
use super::types::{CheckItem, CheckStatus, Evidence, EvidenceType, Severity};

/// Scan source files matching a glob pattern for a content predicate.
fn any_source_matches(project: &Path, glob_suffix: &str, pred: impl Fn(&str) -> bool) -> bool {
    glob::glob(&format!("{}/{glob_suffix}", project.display()))
        .ok()
        .map(|entries| {
            entries
                .flatten()
                .any(|p| std::fs::read_to_string(&p).ok().map(|c| pred(&c)).unwrap_or(false))
        })
        .unwrap_or(false)
}

/// Evaluate all Jidoka automated gates for a project.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_precommit_hooks(project_path),
        check_automated_sovereignty_linting(project_path),
        check_data_drift_circuit_breaker(project_path),
        check_performance_regression_gate(project_path),
        check_fairness_metric_circuit_breaker(project_path),
        check_latency_sla_circuit_breaker(project_path),
        check_memory_footprint_gate(project_path),
        check_security_scan_gate(project_path),
        check_license_compliance_gate(project_path),
        check_documentation_gate(project_path),
    ]
}

/// JA-01: Pre-Commit Hook Enforcement
///
/// **Claim:** Pre-commit hooks catch basic issues locally.
///
/// **Rejection Criteria (Major):**
/// - >5% of CI failures are pre-commit-detectable
pub fn check_precommit_hooks(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-01",
        "Pre-Commit Hook Enforcement",
        "Pre-commit hooks catch basic issues locally",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — early detection");

    // Check for pre-commit configuration
    let precommit_yaml = project_path.join(".pre-commit-config.yaml");
    let has_precommit = precommit_yaml.exists();

    // Check for git hooks directory
    let git_hooks = project_path.join(".git/hooks");
    let has_git_hooks = git_hooks.exists()
        && (git_hooks.join("pre-commit").exists() || git_hooks.join("pre-push").exists());

    // Check for husky (Node.js)
    let husky_dir = project_path.join(".husky");
    let has_husky = husky_dir.exists();

    // Check for cargo-husky or similar in Cargo.toml
    let cargo_toml = project_path.join("Cargo.toml");
    let has_cargo_hooks = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| c.contains("cargo-husky") || c.contains("[hooks]"))
        .unwrap_or(false);

    // Check for Makefile with pre-commit targets
    let makefile = project_path.join("Makefile");
    let has_make_hooks = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("pre-commit") || c.contains("precommit") || c.contains("tier1"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Pre-commit: yaml={}, git_hooks={}, husky={}, cargo_hooks={}, make_targets={}",
            has_precommit, has_git_hooks, has_husky, has_cargo_hooks, has_make_hooks
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_precommit || has_cargo_hooks, CheckOutcome::Pass),
            (
                has_git_hooks || has_husky || has_make_hooks,
                CheckOutcome::Partial("Pre-commit configured but not standardized"),
            ),
            (true, CheckOutcome::Fail("No pre-commit hooks configured")),
        ],
    );

    item.finish_timed(start)
}

/// JA-02: Automated Sovereignty Linting
///
/// **Claim:** Static analysis catches sovereignty violations.
///
/// **Rejection Criteria (Major):**
/// - Known violation pattern not flagged
pub fn check_automated_sovereignty_linting(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-02",
        "Automated Sovereignty Linting",
        "Static analysis catches sovereignty violations",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — automated sovereignty check");

    // Check for clippy configuration
    let clippy_toml = project_path.join("clippy.toml");
    let has_clippy_config = clippy_toml.exists();

    // Check CI for clippy
    let has_clippy_ci = check_ci_for_content(project_path, "clippy");

    // Check for custom lints
    let has_custom_lints = any_source_matches(project_path, "src/**/*.rs", |c| {
        c.contains("#[deny(")
            || c.contains("#![deny(")
            || c.contains("#[warn(")
            || c.contains("#![warn(")
    });

    // Check for deny.toml (cargo-deny)
    let deny_toml = project_path.join("deny.toml");
    let has_deny = deny_toml.exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Linting: clippy_config={}, clippy_ci={}, custom_lints={}, deny={}",
            has_clippy_config, has_clippy_ci, has_custom_lints, has_deny
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_clippy_ci && (has_deny || has_custom_lints), CheckOutcome::Pass),
            (
                has_clippy_ci,
                CheckOutcome::Partial("Clippy in CI but limited sovereignty-specific rules"),
            ),
            (true, CheckOutcome::Fail("No automated linting in CI")),
        ],
    );

    item.finish_timed(start)
}

/// JA-03: Data Drift Circuit Breaker
///
/// **Claim:** Training stops on significant data drift.
///
/// **Rejection Criteria (Major):**
/// - Training completes with >20% distribution shift
pub fn check_data_drift_circuit_breaker(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-03",
        "Data Drift Circuit Breaker",
        "Training stops on significant data drift",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — automatic stop");

    // Check for drift detection patterns in code
    let has_drift_detection = any_source_matches(project_path, "src/**/*.rs", |c| {
        c.contains("drift")
            || c.contains("distribution_shift")
            || c.contains("data_quality")
            || c.contains("schema_validation")
    });

    // Check for data validation in tests
    let has_data_validation = any_source_matches(project_path, "tests/**/*.rs", |c| {
        c.contains("data") && (c.contains("valid") || c.contains("schema"))
    });

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Data drift: detection={}, validation={}",
            has_drift_detection, has_data_validation
        ),
        data: None,
        files: Vec::new(),
    });

    // Check if project has ML training that needs drift detection
    let has_training = any_source_matches(project_path, "src/**/*.rs", |c| {
        c.contains("train") || c.contains("fit") || c.contains("epoch")
    });

    item = apply_check_outcome(
        item,
        &[
            (!has_training || has_drift_detection || has_data_validation, CheckOutcome::Pass),
            (true, CheckOutcome::Partial("Training without data drift detection")),
        ],
    );

    item.finish_timed(start)
}

/// JA-04: Model Performance Regression Gate
///
/// **Claim:** Deployment blocked on performance regression.
///
/// **Rejection Criteria (Major):**
/// - Model with <baseline metrics deploys
pub fn check_performance_regression_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-04",
        "Performance Regression Gate",
        "Deployment blocked on performance regression",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — quality gate");

    // Check for benchmark configuration
    let benches_dir = project_path.join("benches");
    let has_benches = benches_dir.exists();

    // Check for criterion in Cargo.toml
    let cargo_toml = project_path.join("Cargo.toml");
    let has_criterion = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| c.contains("criterion") || c.contains("divan") || c.contains("[bench]"))
        .unwrap_or(false);

    // Check CI for benchmarks
    let has_bench_ci = check_ci_for_content(project_path, "bench");

    // Check for hyperfine or similar
    let makefile = project_path.join("Makefile");
    let has_perf_make = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("hyperfine") || c.contains("bench") || c.contains("perf"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Performance: benches_dir={}, criterion={}, ci_bench={}, make_perf={}",
            has_benches, has_criterion, has_bench_ci, has_perf_make
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_benches && has_criterion && has_bench_ci, CheckOutcome::Pass),
            (
                has_benches || has_criterion,
                CheckOutcome::Partial("Benchmarks exist but not gated in CI"),
            ),
            (true, CheckOutcome::Partial("No performance regression detection")),
        ],
    );

    item.finish_timed(start)
}

/// JA-05: Fairness Metric Circuit Breaker
///
/// **Claim:** Training stops on fairness regression.
///
/// **Rejection Criteria (Major):**
/// - Protected class metric degrades >5% without alert
pub fn check_fairness_metric_circuit_breaker(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-05",
        "Fairness Metric Circuit Breaker",
        "Training stops on fairness regression",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — ethical safeguard");

    // Check for fairness-related code
    let has_fairness_code = any_source_matches(project_path, "src/**/*.rs", |c| {
        c.contains("fairness")
            || c.contains("bias")
            || c.contains("demographic_parity")
            || c.contains("equalized_odds")
            || c.contains("protected_class")
    });

    // Check for fairness testing
    let has_fairness_tests = any_source_matches(project_path, "tests/**/*.rs", |c| {
        c.contains("fairness") || c.contains("bias")
    });

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Fairness: code={}, tests={}", has_fairness_code, has_fairness_tests),
        data: None,
        files: Vec::new(),
    });

    // Check if project has ML that needs fairness monitoring
    let has_ml = any_source_matches(project_path, "src/**/*.rs", |c| {
        c.contains("classifier") || c.contains("predict") || c.contains("model")
    });

    item = apply_check_outcome(
        item,
        &[
            (!has_ml || has_fairness_code || has_fairness_tests, CheckOutcome::Pass),
            (true, CheckOutcome::Partial("ML without fairness monitoring")),
        ],
    );

    item.finish_timed(start)
}

/// JA-06: Latency SLA Circuit Breaker
///
/// **Claim:** Deployment blocked on latency regression.
///
/// **Rejection Criteria (Major):**
/// - P99 latency exceeds SLA in staging
pub fn check_latency_sla_circuit_breaker(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "JA-06",
        "Latency SLA Circuit Breaker",
        "Deployment blocked on latency regression",
    )
    .with_severity(Severity::Major)
    .with_tps("Jidoka — SLA enforcement");

    // Check for latency-related code
    let has_latency_monitoring = any_source_matches(project_path, "src/**/*.rs", |c| {
        c.contains("latency")
            || c.contains("p99")
            || c.contains("p95")
            || c.contains("percentile")
            || c.contains("sla")
    });

    // Check for timing/duration tracking
    let has_timing = any_source_matches(project_path, "src/**/*.rs", |c| {
        c.contains("Instant::") || c.contains("Duration::") || c.contains("elapsed")
    });

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Latency: monitoring={}, timing={}",
            has_latency_monitoring, has_timing
        ),
        data: None,
        files: Vec::new(),
    });

    // Check if project has serving that needs latency SLA
    let has_serving = any_source_matches(project_path, "src/**/*.rs", |c| {
        c.contains("serve") || c.contains("inference") || c.contains("api")
    });

    item = apply_check_outcome(
        item,
        &[
            (!has_serving || has_latency_monitoring, CheckOutcome::Pass),
            (has_timing, CheckOutcome::Partial("Timing code exists but no SLA enforcement")),
            (true, CheckOutcome::Partial("Serving without latency monitoring")),
        ],
    );

    item.finish_timed(start)
}

/// JA-07: Memory Footprint Gate
///
/// **Claim:** Deployment blocked on excessive memory.
///
/// **Rejection Criteria (Major):**
/// - Peak memory exceeds target by >20%
pub fn check_memory_footprint_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("JA-07", "Memory Footprint Gate", "Deployment blocked on excessive memory")
            .with_severity(Severity::Major)
            .with_tps("Muda (Inventory) prevention");

    // Check for memory profiling patterns
    let has_memory_profiling = any_source_matches(project_path, "src/**/*.rs", |c| {
        c.contains("memory")
            || c.contains("heap")
            || c.contains("allocator")
            || c.contains("mem::size_of")
    });

    // Check for memory limits in CI or config
    let has_memory_limits = check_ci_for_content(project_path, "memory")
        || check_ci_for_content(project_path, "ulimit");

    // Check Makefile for memory profiling
    let makefile = project_path.join("Makefile");
    let has_heaptrack = makefile
        .exists()
        .then(|| std::fs::read_to_string(&makefile).ok())
        .flatten()
        .map(|c| c.contains("heaptrack") || c.contains("valgrind") || c.contains("massif"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Memory: profiling={}, limits={}, heaptrack={}",
            has_memory_profiling, has_memory_limits, has_heaptrack
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_memory_profiling && (has_memory_limits || has_heaptrack), CheckOutcome::Pass),
            (
                has_memory_profiling || has_heaptrack,
                CheckOutcome::Partial("Memory profiling available but not gated"),
            ),
            (true, CheckOutcome::Partial("No memory footprint gate")),
        ],
    );

    item.finish_timed(start)
}

/// JA-08: Security Scan Gate
///
/// **Claim:** Build blocked on security findings.
///
/// **Rejection Criteria (Critical):**
/// - High/Critical vulnerability in build
pub fn check_security_scan_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("JA-08", "Security Scan Gate", "Build blocked on security findings")
            .with_severity(Severity::Critical)
            .with_tps("Jidoka — security gate");

    // Check for security scanning tools
    let has_audit_ci = check_ci_for_content(project_path, "cargo audit");
    let has_deny_ci = check_ci_for_content(project_path, "cargo deny");

    // Check for deny.toml
    let deny_toml = project_path.join("deny.toml");
    let has_deny_config = deny_toml.exists();

    // Check for security workflow
    let security_workflow = project_path.join(".github/workflows/security.yml");
    let has_security_workflow = security_workflow.exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Security: audit_ci={}, deny_ci={}, deny_config={}, security_workflow={}",
            has_audit_ci, has_deny_ci, has_deny_config, has_security_workflow
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_deny_config && (has_audit_ci || has_deny_ci), CheckOutcome::Pass),
            (
                has_audit_ci || has_deny_ci || has_deny_config,
                CheckOutcome::Partial("Security scanning partially configured"),
            ),
            (true, CheckOutcome::Fail("No security scanning in CI")),
        ],
    );

    item.finish_timed(start)
}

/// JA-09: License Compliance Gate
///
/// **Claim:** Build blocked on license violation.
///
/// **Rejection Criteria (Major):**
/// - Disallowed license in dependency tree
pub fn check_license_compliance_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("JA-09", "License Compliance Gate", "Build blocked on license violation")
            .with_severity(Severity::Major)
            .with_tps("Legal controls pillar");

    // Check for deny.toml with licenses section
    let deny_toml = project_path.join("deny.toml");
    let has_license_config = deny_toml
        .exists()
        .then(|| std::fs::read_to_string(&deny_toml).ok())
        .flatten()
        .map(|c| c.contains("[licenses]"))
        .unwrap_or(false);

    // Check for cargo-deny in CI
    let has_deny_licenses_ci = check_ci_for_content(project_path, "cargo deny check licenses");

    // Check for LICENSE file
    let has_license_file = project_path.join("LICENSE").exists()
        || project_path.join("LICENSE.md").exists()
        || project_path.join("LICENSE.txt").exists()
        || project_path.join("LICENSE-MIT").exists()
        || project_path.join("LICENSE-APACHE").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "License: config={}, ci_check={}, license_file={}",
            has_license_config, has_deny_licenses_ci, has_license_file
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_license_config && has_deny_licenses_ci, CheckOutcome::Pass),
            (
                has_license_file && (has_license_config || has_deny_licenses_ci),
                CheckOutcome::Partial("License file exists, partial enforcement"),
            ),
            (has_license_file, CheckOutcome::Partial("License file exists but no automated check")),
            (true, CheckOutcome::Fail("No license compliance setup")),
        ],
    );

    item.finish_timed(start)
}

/// JA-10: Documentation Gate
///
/// **Claim:** PR blocked without documentation updates.
///
/// **Rejection Criteria (Minor):**
/// - Public API change without doc update
pub fn check_documentation_gate(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("JA-10", "Documentation Gate", "PR blocked without documentation updates")
            .with_severity(Severity::Minor)
            .with_tps("Knowledge transfer");

    // Check for doc tests
    let has_doc_tests = check_ci_for_content(project_path, "cargo doc");

    // Check for deny warnings on missing docs
    let lib_rs = project_path.join("src/lib.rs");
    let has_deny_missing_docs = lib_rs
        .exists()
        .then(|| std::fs::read_to_string(&lib_rs).ok())
        .flatten()
        .map(|c| c.contains("#![deny(missing_docs)]") || c.contains("#![warn(missing_docs)]"))
        .unwrap_or(false);

    // Check for README
    let has_readme = project_path.join("README.md").exists();

    // Check for docs directory or book
    let has_docs_dir = project_path.join("docs").exists() || project_path.join("book").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Docs: ci_doc={}, deny_missing={}, readme={}, docs_dir={}",
            has_doc_tests, has_deny_missing_docs, has_readme, has_docs_dir
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (
                (has_doc_tests && has_deny_missing_docs) || (has_readme && has_docs_dir),
                CheckOutcome::Pass,
            ),
            (has_readme, CheckOutcome::Partial("README exists but no documentation enforcement")),
            (true, CheckOutcome::Fail("No documentation gate")),
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
        project_path.join(".github/workflows/security.yml"),
        project_path.join(".github/workflows/release.yml"),
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
#[path = "jidoka_tests.rs"]
mod tests;
