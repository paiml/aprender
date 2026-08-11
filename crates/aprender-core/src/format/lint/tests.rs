pub(crate) use super::*;

// ========================================================================
// Test LintLevel
// ========================================================================

#[test]
fn test_lint_level_ordering() {
    assert!(LintLevel::Info < LintLevel::Warn);
    assert!(LintLevel::Warn < LintLevel::Error);
}

#[test]
fn test_lint_level_display() {
    assert_eq!(LintLevel::Info.as_str(), "INFO");
    assert_eq!(LintLevel::Warn.as_str(), "WARN");
    assert_eq!(LintLevel::Error.as_str(), "ERROR");
}

// ========================================================================
// Test LintCategory
// ========================================================================

#[test]
fn test_lint_category_names() {
    assert_eq!(LintCategory::Metadata.name(), "Metadata");
    assert_eq!(LintCategory::Naming.name(), "Tensor Naming");
    assert_eq!(LintCategory::Efficiency.name(), "Efficiency");
}

// ========================================================================
// Test LintIssue
// ========================================================================

#[test]
fn test_lint_issue_creation() {
    let issue = LintIssue::new(LintLevel::Warn, LintCategory::Metadata, "Missing license");
    assert_eq!(issue.level, LintLevel::Warn);
    assert_eq!(issue.category, LintCategory::Metadata);
    assert_eq!(issue.message, "Missing license");
    assert!(issue.suggestion.is_none());
}

#[test]
fn test_lint_issue_with_suggestion() {
    let issue = LintIssue::naming_info("Use full name")
        .with_suggestion("Rename 'enc.w' to 'encoder.weight'");

    assert_eq!(issue.level, LintLevel::Info);
    assert_eq!(issue.category, LintCategory::Naming);
    assert!(issue.suggestion.is_some());
}

#[test]
fn test_lint_issue_display() {
    let issue = LintIssue::metadata_warn("Missing 'license' field");
    let display = format!("{}", issue);
    assert!(display.contains("[WARN]"));
    assert!(display.contains("Metadata"));
    assert!(display.contains("Missing 'license' field"));
}

// ========================================================================
// Test LintReport
// ========================================================================

#[test]
fn test_lint_report_empty() {
    let report = LintReport::new();
    assert!(report.passed());
    assert!(report.passed_strict());
    assert_eq!(report.total_issues(), 0);
}

#[test]
fn test_lint_report_add_issues() {
    let mut report = LintReport::new();

    report.add_issue(LintIssue::metadata_warn("Missing license"));
    report.add_issue(LintIssue::efficiency_info("Unaligned tensors"));

    assert_eq!(report.total_issues(), 2);
    assert_eq!(report.warn_count, 1);
    assert_eq!(report.info_count, 1);
    assert!(!report.passed()); // Has warning
    assert!(!report.passed_strict()); // Has issues
}

#[test]
fn test_lint_report_info_only_passes() {
    let mut report = LintReport::new();
    report.add_issue(LintIssue::efficiency_info("Suggestion"));

    assert!(report.passed()); // Info doesn't fail
    assert!(!report.passed_strict()); // But not strictly clean
}

#[test]
fn test_lint_report_issues_by_category() {
    let mut report = LintReport::new();
    report.add_issue(LintIssue::metadata_warn("Missing license"));
    report.add_issue(LintIssue::metadata_warn("Missing model_card"));
    report.add_issue(LintIssue::efficiency_info("Unaligned"));

    let metadata_issues = report.issues_in_category(LintCategory::Metadata);
    assert_eq!(metadata_issues.len(), 2);

    let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
    assert_eq!(efficiency_issues.len(), 1);
}

#[test]
fn test_lint_report_issues_by_level() {
    let mut report = LintReport::new();
    report.add_issue(LintIssue::metadata_warn("Warning 1"));
    report.add_issue(LintIssue::metadata_warn("Warning 2"));
    report.add_issue(LintIssue::efficiency_info("Info 1"));

    let warnings = report.issues_at_level(LintLevel::Warn);
    assert_eq!(warnings.len(), 2);

    let infos = report.issues_at_level(LintLevel::Info);
    assert_eq!(infos.len(), 1);
}

// ========================================================================
// Test Metadata Checks
// ========================================================================

#[test]
fn test_lint_missing_license() {
    let info = ModelLintInfo {
        has_license: false,
        has_model_card: true,
        has_provenance: true,
        ..Default::default()
    };

    let report = lint_model(&info);
    assert_eq!(report.warn_count, 1);

    let metadata_issues = report.issues_in_category(LintCategory::Metadata);
    assert_eq!(metadata_issues.len(), 1);
    assert!(metadata_issues[0].message.contains("license"));
}

#[test]
fn test_lint_missing_model_card() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: false,
        has_provenance: true,
        ..Default::default()
    };

    let report = lint_model(&info);
    assert_eq!(report.warn_count, 1);

    let metadata_issues = report.issues_in_category(LintCategory::Metadata);
    assert!(metadata_issues[0].message.contains("model_card"));
}

#[test]
fn test_lint_missing_provenance() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: false,
        ..Default::default()
    };

    let report = lint_model(&info);
    assert_eq!(report.warn_count, 1);

    let metadata_issues = report.issues_in_category(LintCategory::Metadata);
    assert!(metadata_issues[0].message.contains("provenance"));
}

#[test]
fn test_lint_all_metadata_present() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        ..Default::default()
    };

    let report = lint_model(&info);
    let metadata_issues = report.issues_in_category(LintCategory::Metadata);
    assert!(metadata_issues.is_empty());
}

#[test]
fn test_lint_all_metadata_missing() {
    let info = ModelLintInfo {
        has_license: false,
        has_model_card: false,
        has_provenance: false,
        ..Default::default()
    };

    let report = lint_model(&info);
    assert_eq!(report.warn_count, 3);

    let metadata_issues = report.issues_in_category(LintCategory::Metadata);
    assert_eq!(metadata_issues.len(), 3);
}

// ========================================================================
// Test Tensor Naming Checks
// ========================================================================

#[test]
fn test_lint_abbreviated_weight_name() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "encoder.conv1.w".to_string(),
            size_bytes: 1000,
            alignment: 64,
            is_compressed: false,
            shape: vec![],
        }],
        ..Default::default()
    };

    let report = lint_model(&info);
    let naming_issues = report.issues_in_category(LintCategory::Naming);
    assert!(!naming_issues.is_empty());
    assert!(naming_issues[0].message.contains(".weight"));
}

#[test]
fn test_lint_abbreviated_bias_name() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "encoder.conv1.b".to_string(),
            size_bytes: 1000,
            alignment: 64,
            is_compressed: false,
            shape: vec![],
        }],
        ..Default::default()
    };

    let report = lint_model(&info);
    let naming_issues = report.issues_in_category(LintCategory::Naming);
    assert!(!naming_issues.is_empty());
    assert!(naming_issues[0].message.contains(".bias"));
}

#[test]
fn test_lint_canonical_name_passes() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "encoder.conv1.weight".to_string(),
            size_bytes: 1000,
            alignment: 64,
            is_compressed: false,
            shape: vec![],
        }],
        ..Default::default()
    };

    let report = lint_model(&info);
    let naming_issues = report.issues_in_category(LintCategory::Naming);
    // Canonical name should not trigger abbreviated name warnings
    // (might trigger non-standard pattern check)
    let abbrev_issues: Vec<_> = naming_issues
        .iter()
        .filter(|i| i.message.contains("should be"))
        .collect();
    assert!(abbrev_issues.is_empty());
}

#[test]
fn test_lint_nonstandard_pattern_double_underscore() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "encoder__weight".to_string(),
            size_bytes: 1000,
            alignment: 64,
            is_compressed: false,
            shape: vec![],
        }],
        ..Default::default()
    };

    let report = lint_model(&info);
    let naming_issues = report.issues_in_category(LintCategory::Naming);
    let nonstandard: Vec<_> = naming_issues
        .iter()
        .filter(|i| i.message.contains("canonical naming"))
        .collect();
    assert!(!nonstandard.is_empty());
}

#[test]
fn test_lint_too_short_name() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "w".to_string(),
            size_bytes: 1000,
            alignment: 64,
            is_compressed: false,
            shape: vec![],
        }],
        ..Default::default()
    };

    let report = lint_model(&info);
    let naming_issues = report.issues_in_category(LintCategory::Naming);
    assert!(!naming_issues.is_empty());
}

// ========================================================================
// Test Efficiency Checks
// ========================================================================

#[test]
fn test_lint_unaligned_tensors() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![
            TensorLintInfo {
                name: "tensor1".to_string(),
                size_bytes: 1000,
                alignment: 32, // Unaligned
                is_compressed: false,
                shape: vec![],
            },
            TensorLintInfo {
                name: "tensor2".to_string(),
                size_bytes: 1000,
                alignment: 16, // Unaligned
                is_compressed: false,
                shape: vec![],
            },
        ],
        ..Default::default()
    };

    let report = lint_model(&info);
    let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
    let alignment_issues: Vec<_> = efficiency_issues
        .iter()
        .filter(|i| i.message.contains("aligned"))
        .collect();
    assert!(!alignment_issues.is_empty());
    assert!(alignment_issues[0].message.contains("2 tensors"));
}

#[test]
fn test_lint_aligned_tensors_pass() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "tensor1".to_string(),
            size_bytes: 1000,
            alignment: 64, // Aligned
            is_compressed: false,
            shape: vec![],
        }],
        ..Default::default()
    };

    let report = lint_model(&info);
    let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
    let alignment_issues: Vec<_> = efficiency_issues
        .iter()
        .filter(|i| i.message.contains("aligned"))
        .collect();
    assert!(alignment_issues.is_empty());
}

#[test]
fn test_lint_large_uncompressed_tensor() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "large_tensor".to_string(),
            size_bytes: 2 * 1024 * 1024, // 2MB
            alignment: 64,
            is_compressed: false, // Not compressed
            shape: vec![],
        }],
        ..Default::default()
    };

    let report = lint_model(&info);
    let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
    let compression_issues: Vec<_> = efficiency_issues
        .iter()
        .filter(|i| i.message.contains("compression"))
        .collect();
    assert!(!compression_issues.is_empty());
}

#[test]
fn test_lint_large_compressed_tensor_passes() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "large_tensor".to_string(),
            size_bytes: 2 * 1024 * 1024, // 2MB
            alignment: 64,
            is_compressed: true, // Compressed
            shape: vec![],
        }],
        ..Default::default()
    };

    let report = lint_model(&info);
    let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
    let compression_issues: Vec<_> = efficiency_issues
        .iter()
        .filter(|i| i.message.contains("compression"))
        .collect();
    assert!(compression_issues.is_empty());
}

#[path = "tests_model_lint.rs"]
mod tests_model_lint;
#[path = "tests_format_lint.rs"]
mod tests_format_lint;
#[path = "tests_magic_dispatch.rs"]
mod tests_magic_dispatch;

// ============================================================================
// #2394: the lint verdict must DISCRIMINATE, not fail unconditionally
// ============================================================================
//
// Dogfooding `cargo install aprender` 0.63.0 found `apr lint` exiting 5 on a
// healthy model with its own message reading `0 error(s)`:
//
//     ⚠ WARN │ Metadata │ Missing 'license' field
//     ⚠ WARN │ Metadata │ Missing 'model_card'
//     ⚠ WARN │ Metadata │ Missing 'provenance' information
//     ℹ INFO │ Efficiency │ 121 uncompressed tensors exceed 1MB
//     error: Lint failed with 0 error(s), 3 warning(s), 1 info(s)
//
// A corrupt GGUF produced the same exit 5. A verdict that is always "fail"
// distinguishes nothing, and `apr qualify` ran this as a gate that could
// therefore never pass.

fn report_with(errors: usize, warns: usize, infos: usize) -> LintReport {
    let mut r = LintReport::new();
    for _ in 0..errors {
        r.add_issue(LintIssue::new(
            LintLevel::Error,
            LintCategory::Metadata,
            "an error",
        ));
    }
    for _ in 0..warns {
        r.add_issue(LintIssue::new(
            LintLevel::Warn,
            LintCategory::Metadata,
            "a warning",
        ));
    }
    for _ in 0..infos {
        r.add_issue(LintIssue::new(
            LintLevel::Info,
            LintCategory::Efficiency,
            "a suggestion",
        ));
    }
    r
}

#[test]
fn advisory_findings_alone_do_not_fail_the_default_verdict() {
    // The exact shape observed on a healthy model in 0.63.0.
    let healthy = report_with(0, 3, 1);
    assert!(
        healthy.passed_at_level(LintLevel::Error),
        "3 metadata warnings and 1 efficiency suggestion are advice, not defects; \
         0.63.0 exited 5 here while printing `0 error(s)`"
    );
}

#[test]
fn an_error_fails_the_verdict_at_every_threshold() {
    let broken = report_with(1, 0, 0);
    assert!(!broken.passed_at_level(LintLevel::Error));
    assert!(!broken.passed_at_level(LintLevel::Warn));
    assert!(!broken.passed_at_level(LintLevel::Info));
}

#[test]
fn strict_promotes_warnings_to_failures() {
    let warned = report_with(0, 1, 0);
    assert!(
        warned.passed_at_level(LintLevel::Error),
        "a warning must not fail the default threshold"
    );
    assert!(
        !warned.passed_at_level(LintLevel::Warn),
        "--strict exists precisely so a caller CAN fail on warnings"
    );
}

#[test]
fn info_never_fails_anything_short_of_the_info_threshold() {
    let noted = report_with(0, 0, 5);
    assert!(noted.passed_at_level(LintLevel::Error));
    assert!(noted.passed_at_level(LintLevel::Warn));
    assert!(
        !noted.passed_at_level(LintLevel::Info),
        "the Info threshold means `no issues at all`"
    );
}

#[test]
fn the_verdict_actually_discriminates_between_two_models() {
    // The property that was missing: two models of different quality must not
    // receive the same verdict. In 0.63.0 a healthy .apr and a corrupt .gguf
    // both exited 5.
    let healthy = report_with(0, 3, 1);
    let broken = report_with(2, 3, 1);
    assert_ne!(
        healthy.passed_at_level(LintLevel::Error),
        broken.passed_at_level(LintLevel::Error),
        "a model with errors and one without must not share a verdict"
    );
}

#[test]
fn passed_at_level_subsumes_the_two_hardcoded_predicates() {
    for (e, w, i) in [(0, 0, 0), (0, 0, 2), (0, 1, 0), (1, 0, 0), (2, 3, 4)] {
        let r = report_with(e, w, i);
        assert_eq!(
            r.passed_at_level(LintLevel::Warn),
            r.passed(),
            "Warn threshold is the existing passed()"
        );
        assert_eq!(
            r.passed_at_level(LintLevel::Info),
            r.passed_strict(),
            "Info threshold is the existing passed_strict()"
        );
    }
}
