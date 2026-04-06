//! Model Parity Gap Analysis (BH-27)
//!
//! Analyzes tiny-model-ground-truth directory for parity gaps:
//! missing oracle files, failed claims, and incomplete oracle-ops coverage.

use super::{DefectCategory, Finding, FindingEvidence, FindingSeverity, HuntMode};
use std::path::Path;

// ============================================================================
// Constants
// ============================================================================

const EXPECTED_MODELS: &[&str] = &["smollm-135m", "qwen2-0.5b", "gpt2-124m"];
const EXPECTED_PROMPTS: &[&str] = &["arithmetic", "code", "completion", "greeting"];
const EXPECTED_OPS: &[&str] = &["convert", "quantize", "finetune", "merge", "prune"];

// ============================================================================
// Public API
// ============================================================================

/// Discover the tiny-model-ground-truth directory.
///
/// Checks explicit path first, then auto-discovers `../tiny-model-ground-truth/`.
pub fn discover_model_parity_dir(
    project_path: &Path,
    explicit_path: Option<&Path>,
) -> Option<std::path::PathBuf> {
    if let Some(p) = explicit_path {
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    // Canonicalize to resolve "." correctly
    let resolved = project_path.canonicalize().ok()?;
    let parent = resolved.parent()?;
    let auto_path = parent.join("tiny-model-ground-truth");
    if auto_path.is_dir() {
        Some(auto_path)
    } else {
        None
    }
}

/// Analyze model parity gaps.
///
/// Produces `BH-PARITY-NNNN` findings for:
/// 1. Missing oracle files (model/prompt combinations)
/// 2. CLAIMS.md FAIL/Deferred claims
/// 3. Incomplete oracle-ops directories
pub fn analyze_model_parity_gaps(tmgt_dir: &Path, _project_path: &Path) -> Vec<Finding> {
    contract_pre_analyze!(tmgt_dir);
    let mut findings = Vec::new();
    let mut finding_id = 0u32;

    // Check 1: Oracle completeness
    check_oracle_completeness(tmgt_dir, &mut findings, &mut finding_id);

    // Check 2: CLAIMS.md status
    check_claims_status(tmgt_dir, &mut findings, &mut finding_id);

    // Check 3: Oracle-ops completeness
    check_oracle_ops(tmgt_dir, &mut findings, &mut finding_id);

    findings
}

// ============================================================================
// Internal helpers
// ============================================================================

fn check_oracle_completeness(tmgt_dir: &Path, findings: &mut Vec<Finding>, finding_id: &mut u32) {
    let oracle_dir = tmgt_dir.join("oracle");
    if !oracle_dir.is_dir() {
        *finding_id += 1;
        findings.push(
            Finding::new(
                format!("BH-PARITY-{:04}", finding_id),
                tmgt_dir,
                1,
                "Missing oracle directory",
            )
            .with_description("No oracle/ directory found in tiny-model-ground-truth")
            .with_severity(FindingSeverity::High)
            .with_category(DefectCategory::ModelParityGap)
            .with_suspiciousness(0.8)
            .with_discovered_by(HuntMode::Analyze)
            .with_evidence(FindingEvidence::model_parity(
                "all",
                "oracle_dir",
                "missing",
            )),
        );
        return;
    }

    for model in EXPECTED_MODELS {
        for prompt in EXPECTED_PROMPTS {
            let oracle_file = oracle_dir.join(model).join(format!("{}.json", prompt));
            if !oracle_file.exists() {
                *finding_id += 1;
                findings.push(
                    Finding::new(
                        format!("BH-PARITY-{:04}", finding_id),
                        &oracle_dir,
                        1,
                        format!("Missing oracle: {}/{}.json", model, prompt),
                    )
                    .with_description(format!(
                        "Oracle output for model `{}` prompt `{}` not generated",
                        model, prompt
                    ))
                    .with_severity(FindingSeverity::Medium)
                    .with_category(DefectCategory::ModelParityGap)
                    .with_suspiciousness(0.6)
                    .with_discovered_by(HuntMode::Analyze)
                    .with_evidence(FindingEvidence::model_parity(*model, *prompt, "missing")),
                );
            }
        }
    }
}

fn check_claims_status(tmgt_dir: &Path, findings: &mut Vec<Finding>, finding_id: &mut u32) {
    let claims_path = tmgt_dir.join("CLAIMS.md");
    let Ok(content) = std::fs::read_to_string(&claims_path) else {
        return;
    };

    for line in content.lines() {
        // Match "### Claim N: Title" headers
        let claim_header = line.strip_prefix("### Claim ");
        if claim_header.is_none() {
            continue;
        }
        let header = claim_header.expect("unexpected failure");
        let claim_title = header.to_string();

        // Check for (Deferred) in the header
        if header.contains("(Deferred)") || header.contains("Deferred") {
            *finding_id += 1;
            findings.push(
                Finding::new(
                    format!("BH-PARITY-{:04}", finding_id),
                    &claims_path,
                    1,
                    format!("Deferred claim: {}", claim_title.trim()),
                )
                .with_description("Claim is deferred — not yet testable or blocked")
                .with_severity(FindingSeverity::Low)
                .with_category(DefectCategory::ModelParityGap)
                .with_suspiciousness(0.4)
                .with_discovered_by(HuntMode::Analyze)
                .with_evidence(FindingEvidence::model_parity(
                    "claims",
                    &claim_title,
                    "deferred",
                )),
            );
        }
    }

    // Check for FAIL status in the content — match "Status" field lines only
    for line in content.lines() {
        let trimmed = line.trim();
        let is_status_line = trimmed.starts_with("- **Status**:")
            || trimmed.starts_with("**Status**:")
            || trimmed.starts_with("- Status:");
        if is_status_line && trimmed.contains("FAIL") {
            *finding_id += 1;
            findings.push(
                Finding::new(
                    format!("BH-PARITY-{:04}", finding_id),
                    &claims_path,
                    1,
                    "Failed claim detected in CLAIMS.md",
                )
                .with_description(line.trim().to_string())
                .with_severity(FindingSeverity::High)
                .with_category(DefectCategory::ModelParityGap)
                .with_suspiciousness(0.8)
                .with_discovered_by(HuntMode::Analyze)
                .with_evidence(FindingEvidence::model_parity("claims", "status", "FAIL")),
            );
        }
    }
}

fn check_oracle_ops(tmgt_dir: &Path, findings: &mut Vec<Finding>, finding_id: &mut u32) {
    let ops_dir = tmgt_dir.join("oracle-ops");
    if !ops_dir.is_dir() {
        *finding_id += 1;
        findings.push(
            Finding::new(
                format!("BH-PARITY-{:04}", finding_id),
                tmgt_dir,
                1,
                "Missing oracle-ops directory",
            )
            .with_description("No oracle-ops/ directory found in tiny-model-ground-truth")
            .with_severity(FindingSeverity::Medium)
            .with_category(DefectCategory::ModelParityGap)
            .with_suspiciousness(0.5)
            .with_discovered_by(HuntMode::Analyze)
            .with_evidence(FindingEvidence::model_parity(
                "ops",
                "oracle-ops",
                "missing",
            )),
        );
        return;
    }

    for op in EXPECTED_OPS {
        let op_dir = ops_dir.join(op);
        let is_empty = if op_dir.is_dir() {
            std::fs::read_dir(&op_dir).map(|mut d| d.next().is_none()).unwrap_or(true)
        } else {
            true
        };

        if is_empty {
            *finding_id += 1;
            findings.push(
                Finding::new(
                    format!("BH-PARITY-{:04}", finding_id),
                    &ops_dir,
                    1,
                    format!("Missing oracle-ops: {}/", op),
                )
                .with_description(format!("Oracle-ops `{}` directory is missing or empty", op))
                .with_severity(FindingSeverity::Low)
                .with_category(DefectCategory::ModelParityGap)
                .with_suspiciousness(0.4)
                .with_discovered_by(HuntMode::Analyze)
                .with_evidence(FindingEvidence::model_parity("ops", *op, "missing")),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_discover_explicit_path() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(&tmgt).expect("mkdir failed");
        let result = discover_model_parity_dir(dir.path(), Some(&tmgt));
        assert!(result.is_some());
        assert_eq!(result.expect("operation failed"), tmgt);
    }

    #[test]
    fn test_discover_explicit_path_missing() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let missing = dir.path().join("nonexistent");
        let result = discover_model_parity_dir(dir.path(), Some(&missing));
        assert!(result.is_none());
    }

    #[test]
    fn test_oracle_completeness_all_missing() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(tmgt.join("oracle")).expect("mkdir failed");
        // No model dirs → all 12 (3×4) combos missing
        let findings = analyze_model_parity_gaps(&tmgt, dir.path());
        let oracle_gaps: Vec<_> =
            findings.iter().filter(|f| f.title.contains("Missing oracle:")).collect();
        assert_eq!(oracle_gaps.len(), 12);
    }

    #[test]
    fn test_oracle_completeness_partial() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        let model_dir = tmgt.join("oracle").join("smollm-135m");
        std::fs::create_dir_all(&model_dir).expect("mkdir failed");
        // Create 2 of 4 prompts
        std::fs::write(model_dir.join("arithmetic.json"), "{}").expect("fs write failed");
        std::fs::write(model_dir.join("code.json"), "{}").expect("fs write failed");

        let findings = analyze_model_parity_gaps(&tmgt, dir.path());
        let smollm_gaps: Vec<_> =
            findings.iter().filter(|f| f.title.contains("smollm-135m")).collect();
        // 2 missing for smollm (completion, greeting)
        assert_eq!(smollm_gaps.len(), 2);
    }

    #[test]
    fn test_parse_claims_status_deferred() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(&tmgt).expect("mkdir failed");
        let claims = tmgt.join("CLAIMS.md");
        {
            let mut f = std::fs::File::create(&claims).expect("file open failed");
            write!(
                f,
                "# Claims\n\n### Claim 6: Cross-Runtime Parity (Deferred)\n- **Status**: Deferred.\n"
            )
            .expect("unexpected failure");
        }

        let mut findings = Vec::new();
        let mut id = 0;
        check_claims_status(&tmgt, &mut findings, &mut id);

        let deferred: Vec<_> = findings.iter().filter(|f| f.title.contains("Deferred")).collect();
        assert_eq!(deferred.len(), 1);
        assert_eq!(deferred[0].severity, FindingSeverity::Low);
    }

    #[test]
    fn test_parse_claims_status_fail() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(&tmgt).expect("mkdir failed");
        let claims = tmgt.join("CLAIMS.md");
        {
            let mut f = std::fs::File::create(&claims).expect("file open failed");
            write!(f, "# Claims\n\n### Claim 19: Throughput\n- **Status**: FAIL\n")
                .expect("unexpected failure");
        }

        let mut findings = Vec::new();
        let mut id = 0;
        check_claims_status(&tmgt, &mut findings, &mut id);

        let fails: Vec<_> = findings.iter().filter(|f| f.title.contains("Failed claim")).collect();
        assert_eq!(fails.len(), 1);
        assert_eq!(fails[0].severity, FindingSeverity::High);
    }

    #[test]
    fn test_oracle_ops_completeness() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        let ops_dir = tmgt.join("oracle-ops");
        // Create only convert and quantize with content
        std::fs::create_dir_all(ops_dir.join("convert")).expect("mkdir failed");
        std::fs::write(ops_dir.join("convert").join("smollm.json"), "{}").expect("fs write failed");
        std::fs::create_dir_all(ops_dir.join("quantize")).expect("mkdir failed");
        std::fs::write(ops_dir.join("quantize").join("smollm.json"), "{}")
            .expect("fs write failed");
        // finetune, merge, prune missing

        let mut findings = Vec::new();
        let mut id = 0;
        check_oracle_ops(&tmgt, &mut findings, &mut id);

        let ops_gaps: Vec<_> =
            findings.iter().filter(|f| f.title.contains("Missing oracle-ops:")).collect();
        assert_eq!(ops_gaps.len(), 3); // finetune, merge, prune
    }

    #[test]
    fn test_missing_oracle_directory() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(&tmgt).expect("mkdir failed");
        // No oracle/ dir at all

        let mut findings = Vec::new();
        let mut id = 0;
        check_oracle_completeness(&tmgt, &mut findings, &mut id);

        assert_eq!(findings.len(), 1);
        assert!(findings[0].title.contains("Missing oracle directory"));
    }

    // ===== Falsification tests =====

    #[test]
    fn test_falsify_fail_detection_rejects_description_lines() {
        // Falsifies: FAIL check must NOT match falsification criterion descriptions
        // Real bug: line 146 of CLAIMS.md has `status == "FAIL"` as criteria, not status
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(&tmgt).expect("mkdir failed");
        let claims = tmgt.join("CLAIMS.md");
        {
            let mut f = std::fs::File::create(&claims).expect("file open failed");
            write!(
                f,
                "# Claims\n\n\
                 ### Claim 20: QA Gate\n\
                 - **Falsification**: any gate with `status == \"FAIL\"`.\n"
            )
            .expect("unexpected failure");
        }

        let mut findings = Vec::new();
        let mut id = 0;
        check_claims_status(&tmgt, &mut findings, &mut id);

        let fails: Vec<_> = findings.iter().filter(|f| f.title.contains("Failed claim")).collect();
        assert_eq!(fails.len(), 0, "Should NOT match falsification criterion line");
    }

    #[test]
    fn test_falsify_fail_detection_matches_status_field() {
        // Must match actual Status field lines with FAIL
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(&tmgt).expect("mkdir failed");
        let claims = tmgt.join("CLAIMS.md");
        {
            let mut f = std::fs::File::create(&claims).expect("file open failed");
            write!(
                f,
                "# Claims\n\n\
                 ### Claim 19: Throughput\n\
                 - **Status**: FAIL (0 tok/s bug)\n\
                 ### Claim 20: QA Gate\n\
                 **Status**: FAIL — critical gate failed\n\
                 ### Claim 21: Other\n\
                 - Status: FAIL\n"
            )
            .expect("unexpected failure");
        }

        let mut findings = Vec::new();
        let mut id = 0;
        check_claims_status(&tmgt, &mut findings, &mut id);

        let fails: Vec<_> = findings.iter().filter(|f| f.title.contains("Failed claim")).collect();
        assert_eq!(fails.len(), 3, "All three Status formats should match");
    }

    #[test]
    fn test_falsify_missing_claims_file() {
        // No CLAIMS.md → 0 findings (not an error)
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(&tmgt).expect("mkdir failed");

        let mut findings = Vec::new();
        let mut id = 0;
        check_claims_status(&tmgt, &mut findings, &mut id);
        assert_eq!(findings.len(), 0);
    }

    #[test]
    fn test_falsify_empty_claims_file() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(&tmgt).expect("mkdir failed");
        std::fs::write(tmgt.join("CLAIMS.md"), "").expect("fs write failed");

        let mut findings = Vec::new();
        let mut id = 0;
        check_claims_status(&tmgt, &mut findings, &mut id);
        assert_eq!(findings.len(), 0);
    }

    #[test]
    fn test_falsify_oracle_all_present() {
        // Complete oracle → 0 findings
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        for model in EXPECTED_MODELS {
            for prompt in EXPECTED_PROMPTS {
                let model_dir = tmgt.join("oracle").join(model);
                std::fs::create_dir_all(&model_dir).expect("mkdir failed");
                std::fs::write(model_dir.join(format!("{}.json", prompt)), "{}")
                    .expect("fs write failed");
            }
        }

        let mut findings = Vec::new();
        let mut id = 0;
        check_oracle_completeness(&tmgt, &mut findings, &mut id);
        assert_eq!(findings.len(), 0, "All oracles present → 0 findings");
    }

    #[test]
    fn test_falsify_oracle_ops_all_present() {
        // All ops dirs populated → 0 findings
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        for op in EXPECTED_OPS {
            let op_dir = tmgt.join("oracle-ops").join(op);
            std::fs::create_dir_all(&op_dir).expect("mkdir failed");
            std::fs::write(op_dir.join("result.json"), "{}").expect("fs write failed");
        }

        let mut findings = Vec::new();
        let mut id = 0;
        check_oracle_ops(&tmgt, &mut findings, &mut id);
        assert_eq!(findings.len(), 0, "All ops present → 0 findings");
    }

    #[test]
    fn test_falsify_ops_dir_exists_but_empty() {
        // Dir exists but has no files → should still flag
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        for op in EXPECTED_OPS {
            std::fs::create_dir_all(tmgt.join("oracle-ops").join(op)).expect("mkdir failed");
        }

        let mut findings = Vec::new();
        let mut id = 0;
        check_oracle_ops(&tmgt, &mut findings, &mut id);
        assert_eq!(findings.len(), 5, "Empty dirs should be flagged");
    }

    #[test]
    fn test_falsify_discover_nonexistent_parent() {
        // Non-existent project path → None (canonicalize fails)
        let result = discover_model_parity_dir(Path::new("/nonexistent/path/xyz"), None);
        assert!(result.is_none());
    }

    #[test]
    fn test_falsify_full_pipeline_empty_tmgt() {
        // Empty tmgt dir → oracle missing + ops missing findings
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let tmgt = dir.path().join("tmgt");
        std::fs::create_dir_all(&tmgt).expect("mkdir failed");

        let findings = analyze_model_parity_gaps(&tmgt, dir.path());
        // No oracle dir → 1 finding; no oracle-ops dir → 1 finding; no CLAIMS.md → 0
        assert_eq!(findings.len(), 2);
        assert!(findings.iter().any(|f| f.title.contains("Missing oracle directory")));
        assert!(findings.iter().any(|f| f.title.contains("Missing oracle-ops directory")));
    }
}
