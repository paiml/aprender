//! Gate implementations: validate, audit, score.
//!
//! Each gate runs a specific quality check and returns a `GateResult`
//! plus a list of `LintFinding`s for downstream rendering.
//!
//! Extended gates (verify, enforce) are in `gates_extended.rs`.

use std::path::Path;
use std::time::Instant;

use crate::audit::audit_contract;
use crate::binding::{parse_binding, BindingRegistry};
use crate::error::Severity;
use crate::schema::{is_contract_yaml, parse_contract, validate_contract, Contract};
use crate::scoring::{score_contract, ContractScore};

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateResult};

/// Load and parse all YAML contracts from a directory.
///
/// Returns successfully parsed contracts and a list of parse errors.
#[allow(clippy::type_complexity)]
pub(crate) fn load_contracts(dir: &Path) -> (Vec<(String, Contract)>, Vec<(String, String)>) {
    let mut contracts = Vec::new();
    let mut parse_errors = Vec::new();
    let mut yaml_paths = Vec::new();
    collect_yaml_files(dir, &mut yaml_paths);

    for path in &yaml_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        match parse_contract(path) {
            Ok(c) => contracts.push((stem, c)),
            Err(e) => parse_errors.push((stem, e.to_string())),
        }
    }
    contracts.sort_by(|a, b| a.0.cmp(&b.0));
    (contracts, parse_errors)
}

/// Recursively collect `.yaml` contract files, skipping non-contract directories.
fn collect_yaml_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let dirname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // `publish-manifests/` holds PublishManifest artifacts (model
            // ship metadata keyed by `model_id:`), not Contract schemas;
            // validating them here would fail on missing `metadata.` root.
            if matches!(
                dirname,
                "kaizen" | "legacy" | "pipelines" | "publish-manifests"
            ) {
                continue;
            }
            collect_yaml_files(&path, out);
        } else if is_contract_yaml(&path) {
            out.push(path);
        }
    }
}

/// Load binding registry from an optional path.
pub(crate) fn load_binding(path: Option<&Path>) -> Option<BindingRegistry> {
    path.and_then(|p| parse_binding(p).ok())
}

pub(crate) fn run_validate_gate(
    contracts: &[(String, Contract)],
    parse_errors: &[(String, String)],
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut total_errors = 0usize;
    let mut total_warnings = 0usize;
    let mut error_messages = Vec::new();
    let mut findings = Vec::new();

    for (stem, err) in parse_errors {
        total_errors += 1;
        error_messages.push(format!("Parse error: {err} ({stem}.yaml)"));
        findings.push(
            LintFinding::new(
                "PV-VAL-001",
                RuleSeverity::Error,
                format!("YAML parse error: {err}"),
                format!("contracts/{stem}.yaml"),
            )
            .with_stem(stem.clone()),
        );
    }

    for (stem, contract) in contracts {
        let violations = validate_contract(contract);
        for v in &violations {
            let sev = match v.severity {
                Severity::Error => {
                    total_errors += 1;
                    error_messages.push(format!("{v} ({stem})"));
                    RuleSeverity::Error
                }
                Severity::Warning => {
                    total_warnings += 1;
                    RuleSeverity::Warning
                }
                Severity::Info => RuleSeverity::Info,
            };
            let rule_id = map_validation_rule(&v.rule);
            findings.push(
                LintFinding::new(
                    rule_id,
                    sev,
                    format!("{}: {}", v.rule, v.message),
                    format!("contracts/{stem}.yaml"),
                )
                .with_stem(stem.clone()),
            );
        }
    }

    let result = GateResult {
        name: "validate".into(),
        passed: total_errors == 0,
        skipped: false,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Validate {
            contracts: contracts.len() + parse_errors.len(),
            errors: total_errors,
            warnings: total_warnings,
            error_messages,
        },
    };
    (result, findings)
}

pub(crate) fn run_audit_gate(contracts: &[(String, Contract)]) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut total_findings = 0usize;
    let mut finding_messages = Vec::new();
    let mut findings = Vec::new();

    for (stem, contract) in contracts {
        let report = audit_contract(contract);
        for v in &report.violations {
            let sev = match v.severity {
                Severity::Error => {
                    total_findings += 1;
                    finding_messages.push(format!("{v} ({stem})"));
                    RuleSeverity::Error
                }
                Severity::Warning => RuleSeverity::Warning,
                Severity::Info => RuleSeverity::Info,
            };
            let rule_id = map_audit_rule(&v.rule);
            findings.push(
                LintFinding::new(
                    rule_id,
                    sev,
                    format!("{}: {}", v.rule, v.message),
                    format!("contracts/{stem}.yaml"),
                )
                .with_stem(stem.clone()),
            );
        }
    }

    let result = GateResult {
        name: "audit".into(),
        passed: total_findings == 0,
        skipped: false,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Audit {
            contracts: contracts.len(),
            findings: total_findings,
            finding_messages,
        },
    };
    (result, findings)
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn run_score_gate(
    contracts: &[(String, Contract)],
    binding: Option<&BindingRegistry>,
    threshold: f64,
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut scores: Vec<ContractScore> = Vec::new();
    let mut below_threshold = Vec::new();
    let mut findings = Vec::new();

    for (stem, contract) in contracts {
        let s = score_contract(contract, binding, stem);
        if s.composite < threshold {
            below_threshold.push(format!(
                "{} — {:.2} (Grade {}, threshold {:.2})",
                stem, s.composite, s.grade, threshold
            ));
            findings.push(
                LintFinding::new(
                    "PV-SCR-001",
                    RuleSeverity::Error,
                    format!(
                        "Score {:.2} (Grade {}) below threshold {:.2}",
                        s.composite, s.grade, threshold
                    ),
                    format!("contracts/{stem}.yaml"),
                )
                .with_stem(stem.clone())
                .with_evidence(format!(
                    "spec={:.2} falsify={:.2} kani={:.2} lean={:.2} bind={:.2}",
                    s.spec_depth,
                    s.falsification_coverage,
                    s.kani_coverage,
                    s.lean_coverage,
                    s.binding_coverage
                )),
            );
        }
        scores.push(s);
    }

    let min_score = scores
        .iter()
        .map(|s| s.composite)
        .fold(f64::INFINITY, f64::min);
    let mean_score = if scores.is_empty() {
        0.0
    } else {
        scores.iter().map(|s| s.composite).sum::<f64>() / scores.len() as f64
    };

    let passed = below_threshold.is_empty();

    let result = GateResult {
        name: "score".into(),
        passed,
        skipped: false,
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: GateDetail::Score {
            contracts: contracts.len(),
            min_score: if scores.is_empty() { 0.0 } else { min_score },
            mean_score,
            threshold,
            below_threshold,
        },
    };
    (result, findings)
}

/// Map existing validation rule IDs to PV-VAL-NNN catalog IDs.
fn map_validation_rule(rule: &str) -> String {
    match rule {
        "PROVABILITY-001" => "PV-PRV-001".into(),
        r if r.starts_with("SCHEMA-") => match r {
            "SCHEMA-004" => "PV-VAL-004".into(),
            "SCHEMA-005" => "PV-VAL-005".into(),
            "SCHEMA-006" => "PV-VAL-006".into(),
            _ => "PV-VAL-001".into(),
        },
        _ => "PV-VAL-001".into(),
    }
}

/// Map existing audit rule IDs to PV-AUD-NNN catalog IDs.
fn map_audit_rule(rule: &str) -> String {
    if rule.contains("paper") || rule.contains("reference") {
        "PV-AUD-002".into()
    } else if rule.contains("test") || rule.contains("falsification") {
        "PV-AUD-001".into()
    } else if rule.contains("domain") {
        "PV-AUD-004".into()
    } else if rule.contains("tolerance") {
        "PV-AUD-005".into()
    } else {
        "PV-AUD-003".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contracts_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
    }

    #[test]
    fn load_contracts_real() {
        let (contracts, errors) = load_contracts(&contracts_dir());
        assert!(contracts.len() > 100, "Expected 100+ contracts");
        assert!(errors.is_empty(), "No parse errors expected: {errors:?}");
    }

    #[test]
    fn load_contracts_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let (contracts, errors) = load_contracts(tmp.path());
        assert!(contracts.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn load_contracts_reports_parse_errors() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("bad.yaml"), "not: valid: yaml: {{{{").unwrap();
        let (contracts, errors) = load_contracts(tmp.path());
        assert!(contracts.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].0, "bad");
    }

    #[test]
    fn validate_gate_fails_on_parse_errors() {
        let parse_errors = vec![("bad".into(), "invalid YAML".into())];
        let (result, findings) = run_validate_gate(&[], &parse_errors);
        assert!(!result.passed);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].rule_id, "PV-VAL-001");
        assert!(findings[0].message.contains("parse error"));
    }

    #[test]
    fn validate_gate_passes() {
        let (contracts, errors) = load_contracts(&contracts_dir());
        let (result, _findings) = run_validate_gate(&contracts, &errors);
        assert!(result.passed);
    }

    #[test]
    fn audit_gate_passes() {
        let (contracts, _) = load_contracts(&contracts_dir());
        let (result, _findings) = run_audit_gate(&contracts);
        assert!(result.passed);
    }

    #[test]
    fn score_gate_passes_zero_threshold() {
        let (contracts, _) = load_contracts(&contracts_dir());
        let (result, findings) = run_score_gate(&contracts, None, 0.0);
        assert!(result.passed);
        assert!(findings.is_empty());
    }

    #[test]
    fn score_gate_fails_high_threshold() {
        let (contracts, _) = load_contracts(&contracts_dir());
        let (result, findings) = run_score_gate(&contracts, None, 0.99);
        assert!(!result.passed);
        assert!(!findings.is_empty());
        assert!(findings.iter().all(|f| f.rule_id == "PV-SCR-001"));
    }

    #[test]
    fn map_validation_rules() {
        assert_eq!(map_validation_rule("PROVABILITY-001"), "PV-PRV-001");
        assert_eq!(map_validation_rule("SCHEMA-001"), "PV-VAL-001");
        assert_eq!(map_validation_rule("SCHEMA-004"), "PV-VAL-004");
        assert_eq!(map_validation_rule("SCHEMA-005"), "PV-VAL-005");
        assert_eq!(map_validation_rule("SCHEMA-006"), "PV-VAL-006");
        assert_eq!(map_validation_rule("SCHEMA-999"), "PV-VAL-001");
        assert_eq!(map_validation_rule("UNKNOWN"), "PV-VAL-001");
    }

    #[test]
    fn map_audit_rules() {
        assert_eq!(map_audit_rule("missing paper reference"), "PV-AUD-002");
        assert_eq!(map_audit_rule("no falsification test"), "PV-AUD-001");
        assert_eq!(map_audit_rule("missing domain"), "PV-AUD-004");
        assert_eq!(map_audit_rule("no tolerance"), "PV-AUD-005");
        assert_eq!(map_audit_rule("other issue"), "PV-AUD-003");
    }
}
