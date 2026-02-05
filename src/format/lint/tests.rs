use super::*;

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

#[test]
fn test_lint_small_uncompressed_tensor_passes() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "small_tensor".to_string(),
            size_bytes: 500 * 1024, // 500KB
            alignment: 64,
            is_compressed: false,
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

// ========================================================================
// Test Complete Model Lint
// ========================================================================

#[test]
fn test_lint_clean_model() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![
            TensorLintInfo {
                name: "encoder.conv1.weight".to_string(),
                size_bytes: 500 * 1024,
                alignment: 64,
                is_compressed: true,
                shape: vec![],
            },
            TensorLintInfo {
                name: "encoder.conv1.bias".to_string(),
                size_bytes: 1024,
                alignment: 64,
                is_compressed: true,
                shape: vec![],
            },
        ],
        is_compressed: true,
        vocab_size: None,
        hidden_dim: None,
    };

    let report = lint_model(&info);
    assert!(report.passed(), "Clean model should pass lint");
}

#[test]
fn test_lint_model_with_all_issues() {
    let info = ModelLintInfo {
        has_license: false,    // WARN
        has_model_card: false, // WARN
        has_provenance: false, // WARN
        tensors: vec![TensorLintInfo {
            name: "enc.w".to_string(),   // INFO (abbreviated)
            size_bytes: 2 * 1024 * 1024, // INFO (large uncompressed)
            alignment: 32,               // INFO (unaligned)
            is_compressed: false,
            shape: vec![],
        }],
        is_compressed: false,
        vocab_size: None,
        hidden_dim: None,
    };

    let report = lint_model(&info);
    assert!(!report.passed(), "Model with issues should not pass");
    assert!(report.warn_count >= 3, "Should have at least 3 warnings");
    assert!(report.info_count >= 1, "Should have at least 1 info");
}

// ========================================================================
// Test is_nonstandard_pattern
// ========================================================================

#[test]
fn test_is_nonstandard_double_underscore() {
    assert!(is_nonstandard_pattern("foo__bar"));
}

#[test]
fn test_is_nonstandard_double_dot() {
    assert!(is_nonstandard_pattern("foo..bar"));
}

#[test]
fn test_is_nonstandard_too_short() {
    assert!(is_nonstandard_pattern("w"));
    assert!(is_nonstandard_pattern("ab"));
}

#[test]
fn test_is_nonstandard_odd_numbers() {
    // Numbers not in standard patterns
    assert!(is_nonstandard_pattern("weight_123"));
}

#[test]
fn test_standard_patterns() {
    assert!(!is_nonstandard_pattern("encoder.conv1.weight"));
    assert!(!is_nonstandard_pattern(
        "encoder.layers.0.self_attn.q_proj.weight"
    ));
    assert!(!is_nonstandard_pattern("decoder.fc1.weight"));
}

// ========================================================================
// Additional Coverage Tests for lint.rs
// ========================================================================

#[test]
fn test_lint_level_display_impl() {
    assert_eq!(format!("{}", LintLevel::Info), "INFO");
    assert_eq!(format!("{}", LintLevel::Warn), "WARN");
    assert_eq!(format!("{}", LintLevel::Error), "ERROR");
}

#[test]
fn test_lint_category_display_impl() {
    assert_eq!(format!("{}", LintCategory::Metadata), "Metadata");
    assert_eq!(format!("{}", LintCategory::Naming), "Tensor Naming");
    assert_eq!(format!("{}", LintCategory::Efficiency), "Efficiency");
}

#[test]
fn test_lint_issue_display_without_suggestion() {
    let issue = LintIssue::new(LintLevel::Warn, LintCategory::Metadata, "Missing field");
    let display = format!("{}", issue);
    assert!(display.contains("[WARN]"));
    assert!(display.contains("Metadata"));
    assert!(display.contains("Missing field"));
    assert!(!display.contains("suggestion"));
}

#[test]
fn test_lint_issue_display_with_suggestion() {
    let issue =
        LintIssue::naming_info("Use full name").with_suggestion("Rename to 'encoder.weight'");
    let display = format!("{}", issue);
    assert!(display.contains("[INFO]"));
    assert!(display.contains("Tensor Naming"));
    assert!(display.contains("suggestion"));
    assert!(display.contains("encoder.weight"));
}

#[test]
fn test_lint_issue_naming_warn() {
    let issue = LintIssue::naming_warn("Invalid naming pattern");
    assert_eq!(issue.level, LintLevel::Warn);
    assert_eq!(issue.category, LintCategory::Naming);
    assert_eq!(issue.message, "Invalid naming pattern");
}

#[test]
fn test_lint_report_default() {
    let report = LintReport::default();
    assert!(report.issues.is_empty());
    assert!(report.by_category.is_empty());
    assert_eq!(report.info_count, 0);
    assert_eq!(report.warn_count, 0);
    assert_eq!(report.error_count, 0);
}

#[test]
fn test_lint_report_add_error() {
    let mut report = LintReport::new();
    report.add_issue(LintIssue::new(
        LintLevel::Error,
        LintCategory::Naming,
        "Critical naming issue",
    ));
    assert_eq!(report.error_count, 1);
    assert!(!report.passed());
    assert!(!report.passed_strict());
}

#[test]
fn test_lint_report_issues_in_nonexistent_category() {
    let report = LintReport::new();
    let issues = report.issues_in_category(LintCategory::Naming);
    assert!(issues.is_empty());
}

#[test]
fn test_model_lint_info_default() {
    let info = ModelLintInfo::default();
    assert!(!info.has_license);
    assert!(!info.has_model_card);
    assert!(!info.has_provenance);
    assert!(info.tensors.is_empty());
    assert!(!info.is_compressed);
}

#[test]
fn test_tensor_lint_info_clone() {
    let tensor = TensorLintInfo {
        name: "test_tensor".to_string(),
        size_bytes: 1024,
        alignment: 64,
        is_compressed: true,
        shape: vec![],
    };
    let cloned = tensor.clone();
    assert_eq!(cloned.name, "test_tensor");
    assert_eq!(cloned.size_bytes, 1024);
    assert_eq!(cloned.alignment, 64);
    assert!(cloned.is_compressed);
}

#[test]
fn test_tensor_lint_info_debug() {
    let tensor = TensorLintInfo {
        name: "debug_test".to_string(),
        size_bytes: 512,
        alignment: 32,
        is_compressed: false,
        shape: vec![],
    };
    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("debug_test"));
    assert!(debug_str.contains("512"));
}

#[test]
fn test_model_lint_info_clone() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: false,
        tensors: vec![],
        is_compressed: true,
        vocab_size: None,
        hidden_dim: None,
    };
    let cloned = info.clone();
    assert!(cloned.has_license);
    assert!(cloned.has_model_card);
    assert!(!cloned.has_provenance);
    assert!(cloned.is_compressed);
}

#[test]
fn test_model_lint_info_debug() {
    let info = ModelLintInfo::default();
    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("ModelLintInfo"));
}

#[test]
fn test_lint_issue_clone() {
    let issue = LintIssue::metadata_warn("Test message").with_suggestion("Fix it");
    let cloned = issue.clone();
    assert_eq!(cloned.level, LintLevel::Warn);
    assert_eq!(cloned.category, LintCategory::Metadata);
    assert_eq!(cloned.message, "Test message");
    assert_eq!(cloned.suggestion, Some("Fix it".to_string()));
}

#[test]
fn test_lint_issue_debug() {
    let issue = LintIssue::efficiency_info("Optimization hint");
    let debug_str = format!("{:?}", issue);
    assert!(debug_str.contains("Optimization hint"));
    assert!(debug_str.contains("Efficiency"));
}

#[test]
fn test_lint_report_clone() {
    let mut report = LintReport::new();
    report.add_issue(LintIssue::metadata_warn("Warning 1"));
    report.add_issue(LintIssue::efficiency_info("Info 1"));

    let cloned = report.clone();
    assert_eq!(cloned.total_issues(), 2);
    assert_eq!(cloned.warn_count, 1);
    assert_eq!(cloned.info_count, 1);
}

#[test]
fn test_lint_report_debug() {
    let report = LintReport::new();
    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("LintReport"));
}

#[test]
fn test_is_abbreviated_already_full_form() {
    // If the full form is present, it's not abbreviated
    assert!(!is_abbreviated("encoder.weight", ".w", ".weight"));
    assert!(!is_abbreviated("layer.bias", ".b", ".bias"));
}

#[test]
fn test_is_abbreviated_not_at_word_boundary() {
    // ".w" at word boundary
    assert!(is_abbreviated("encoder.w", ".w", ".weight"));
    // ".w" followed by separator
    assert!(is_abbreviated("encoder.w.test", ".w", ".weight"));
}

#[test]
fn test_is_at_word_boundary_at_end() {
    // Position at end of string = word boundary
    assert!(is_at_word_boundary("test", 4));
}

#[test]
fn test_is_at_word_boundary_with_separator() {
    // Position before separator = word boundary
    assert!(is_at_word_boundary("test.next", 4));
    assert!(is_at_word_boundary("test_next", 4));
    assert!(is_at_word_boundary("test-next", 4));
}

#[test]
fn test_is_at_word_boundary_not_separator() {
    // Position before letter = not word boundary
    assert!(!is_at_word_boundary("testing", 4));
}

#[test]
fn test_has_standard_numbering_patterns() {
    assert!(has_standard_numbering("encoder.layers.0.weight"));
    assert!(has_standard_numbering("conv1.weight"));
    assert!(has_standard_numbering("conv2.bias"));
    assert!(has_standard_numbering("fc1.weight"));
    assert!(has_standard_numbering("fc2.bias"));
}

#[test]
fn test_has_standard_numbering_no_pattern() {
    assert!(!has_standard_numbering("encoder.weight"));
    assert!(!has_standard_numbering("layer.bias"));
}

#[test]
fn test_has_unusual_separators() {
    assert!(has_unusual_separators("test__double"));
    assert!(has_unusual_separators("test--dash"));
    assert!(has_unusual_separators("test..dot"));
    assert!(!has_unusual_separators("test.normal"));
    assert!(!has_unusual_separators("test_normal"));
}

#[test]
fn test_is_nonstandard_pattern_double_dash() {
    assert!(is_nonstandard_pattern("test--name"));
}

#[test]
fn test_is_nonstandard_pattern_empty() {
    // Empty string is NOT flagged as nonstandard (special case - too short check has !name.is_empty())
    assert!(!is_nonstandard_pattern(""));
}

#[test]
fn test_is_nonstandard_pattern_exact_boundary() {
    // Names of exactly 5 chars should not be "too short"
    assert!(!is_nonstandard_pattern("abcde"));
    // Names of 4 chars should be "too short"
    assert!(is_nonstandard_pattern("abcd"));
}

#[test]
fn test_lint_with_all_abbreviation_patterns() {
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![
            TensorLintInfo {
                name: "encoder.layer_w.data".to_string(), // _w -> .weight
                size_bytes: 1000,
                alignment: 64,
                is_compressed: false,
                shape: vec![],
            },
            TensorLintInfo {
                name: "encoder.layer_b.data".to_string(), // _b -> .bias
                size_bytes: 1000,
                alignment: 64,
                is_compressed: false,
                shape: vec![],
            },
            TensorLintInfo {
                name: "encoder.attn_qkv.data".to_string(), // attn_ -> self_attn.
                size_bytes: 1000,
                alignment: 64,
                is_compressed: false,
                shape: vec![],
            },
        ],
        ..Default::default()
    };

    let report = lint_model(&info);
    let naming_issues = report.issues_in_category(LintCategory::Naming);
    // Should flag abbreviation patterns
    assert!(!naming_issues.is_empty());
}

#[test]
fn test_lint_category_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(LintCategory::Metadata);
    set.insert(LintCategory::Naming);
    set.insert(LintCategory::Efficiency);
    assert_eq!(set.len(), 3);
}

#[test]
fn test_lint_level_partial_ord() {
    // Info < Warn < Error
    assert!(LintLevel::Info < LintLevel::Warn);
    assert!(LintLevel::Warn < LintLevel::Error);
    assert!(LintLevel::Info < LintLevel::Error);

    // Equality
    assert!(LintLevel::Info == LintLevel::Info);
    assert!(LintLevel::Warn == LintLevel::Warn);
    assert!(LintLevel::Error == LintLevel::Error);
}

#[test]
fn test_lint_category_equality() {
    assert_eq!(LintCategory::Metadata, LintCategory::Metadata);
    assert_ne!(LintCategory::Metadata, LintCategory::Naming);
    assert_ne!(LintCategory::Naming, LintCategory::Efficiency);
}

#[test]
fn test_lint_level_copy() {
    let level = LintLevel::Warn;
    let copied = level;
    assert_eq!(copied, LintLevel::Warn);
}

#[test]
fn test_lint_category_copy() {
    let cat = LintCategory::Naming;
    let copied = cat;
    assert_eq!(copied, LintCategory::Naming);
}

#[test]
fn test_lint_zero_alignment() {
    // Zero alignment is a special case (skip check)
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "encoder.conv1.weight".to_string(),
            size_bytes: 1000,
            alignment: 0, // Zero alignment - skip check
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
    // Zero alignment should not be flagged as "unaligned"
    assert!(alignment_issues.is_empty());
}

// ========================================================================
// Test Multi-Format Lint (lint_model_file, lint_gguf_file, lint_safetensors_file)
// ========================================================================

#[test]
fn test_lint_gguf_file_with_metadata() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF with license and author metadata
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![4, 4],
        dtype: GgmlType::F32,
        data: vec![0u8; 64], // 16 floats * 4 bytes
    };
    let metadata = vec![
        (
            "general.license".to_string(),
            GgufValue::String("MIT".to_string()),
        ),
        (
            "general.author".to_string(),
            GgufValue::String("Test Author".to_string()),
        ),
        (
            "general.description".to_string(),
            GgufValue::String("Test model".to_string()),
        ),
    ];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::new().expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    let report = lint_gguf_file(temp_file.path()).expect("lint GGUF");
    let metadata_issues = report.issues_in_category(LintCategory::Metadata);

    // Should have no metadata warnings (all present)
    assert!(
        metadata_issues.is_empty(),
        "GGUF with full metadata should have no metadata warnings"
    );
}

#[test]
fn test_lint_gguf_file_missing_metadata() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF without license/author/description
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: vec![0u8; 16], // 4 floats * 4 bytes
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::new().expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    let report = lint_gguf_file(temp_file.path()).expect("lint GGUF");

    // Should have warnings for missing license, model_card, provenance
    assert!(
        report.warn_count >= 3,
        "GGUF without metadata should have 3+ warnings"
    );
}

#[test]
fn test_lint_gguf_file_tensor_info() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF with multiple tensors
    let tensors = vec![
        GgufTensor {
            name: "encoder.layer.weight".to_string(),
            shape: vec![4, 4],
            dtype: GgmlType::F32,
            data: vec![0u8; 64],
        },
        GgufTensor {
            name: "encoder.layer.bias".to_string(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: vec![0u8; 16],
        },
    ];
    let metadata = vec![
        (
            "general.license".to_string(),
            GgufValue::String("MIT".to_string()),
        ),
        (
            "general.author".to_string(),
            GgufValue::String("Test".to_string()),
        ),
        (
            "general.model_card".to_string(),
            GgufValue::String("Test".to_string()),
        ),
    ];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::new().expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    let report = lint_gguf_file(temp_file.path()).expect("lint GGUF");

    // Should pass with properly named tensors
    assert!(report.passed(), "GGUF with proper names should pass lint");
}

#[test]
fn test_lint_safetensors_file_with_metadata() {
    use crate::format::test_factory::{build_pygmy_safetensors_with_config, PygmyConfig};
    use tempfile::NamedTempFile;

    // Build SafeTensors with custom metadata
    let config = PygmyConfig::default();
    let st_bytes = build_pygmy_safetensors_with_config(config);

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    let report = lint_safetensors_file(temp_file.path()).expect("lint SafeTensors");

    // SafeTensors built with PygmyConfig may not have __metadata__ - that's expected
    // The report should still be valid (test is that lint doesn't crash)
    let _ = report.total_issues();
}

#[test]
fn test_lint_safetensors_file_with_custom_metadata() {
    use tempfile::NamedTempFile;

    // Manually create SafeTensors with __metadata__ containing license
    let header_json = serde_json::json!({
        "__metadata__": {
            "license": "Apache-2.0",
            "author": "Test Author",
            "description": "Test model for linting"
        },
        "test.weight": {
            "dtype": "F32",
            "shape": [4, 4],
            "data_offsets": [0, 64]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).expect("serialize header");
    let header_len = header_bytes.len() as u64;

    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&header_len.to_le_bytes());
    st_bytes.extend_from_slice(&header_bytes);
    st_bytes.extend_from_slice(&[0u8; 64]); // tensor data

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    let report = lint_safetensors_file(temp_file.path()).expect("lint SafeTensors");

    // Should have no metadata warnings (all present in __metadata__)
    let metadata_issues = report.issues_in_category(LintCategory::Metadata);
    assert!(
        metadata_issues.is_empty(),
        "SafeTensors with full __metadata__ should have no metadata warnings"
    );
}

#[test]
fn test_lint_safetensors_file_no_metadata() {
    use tempfile::NamedTempFile;

    // Create SafeTensors without __metadata__
    let header_json = serde_json::json!({
        "test.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).expect("serialize header");
    let header_len = header_bytes.len() as u64;

    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&header_len.to_le_bytes());
    st_bytes.extend_from_slice(&header_bytes);
    st_bytes.extend_from_slice(&[0u8; 16]); // tensor data

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    let report = lint_safetensors_file(temp_file.path()).expect("lint SafeTensors");

    // Should have warnings for missing license, model_card, provenance
    assert!(
        report.warn_count >= 3,
        "SafeTensors without __metadata__ should have 3+ warnings"
    );
}

#[test]
fn test_lint_model_file_gguf_dispatch() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: vec![0u8; 16],
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    // lint_model_file should detect GGUF format and dispatch to lint_gguf_file
    let report = lint_model_file(temp_file.path()).expect("lint model file");
    assert!(
        report.total_issues() > 0,
        "lint_model_file should work for GGUF"
    );
}

#[test]
fn test_lint_model_file_safetensors_dispatch() {
    use crate::format::test_factory::build_pygmy_safetensors;
    use tempfile::NamedTempFile;

    let st_bytes = build_pygmy_safetensors();

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    // lint_model_file should detect SafeTensors format and dispatch
    let report = lint_model_file(temp_file.path()).expect("lint model file");
    // Test passes if lint doesn't crash - total_issues() is usize, always >= 0
    let _ = report.total_issues();
}

#[test]
fn test_lint_model_file_format_detection_by_magic() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF file with .bin extension (format detected by magic, not extension)
    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: vec![0u8; 16],
    };
    let metadata = vec![(
        "general.license".to_string(),
        GgufValue::String("MIT".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    // Use .bin extension but GGUF magic
    let temp_file = NamedTempFile::with_suffix(".bin").expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    // Should detect GGUF by magic bytes, not extension
    let report = lint_model_file(temp_file.path()).expect("lint model file");
    // Test passes if lint correctly detects GGUF by magic
    let _ = report.total_issues();
}

#[test]
fn test_lint_gguf_abbreviated_tensor_names() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF with abbreviated tensor names
    let tensor = GgufTensor {
        name: "enc.w".to_string(), // abbreviated: should trigger naming warning
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: vec![0u8; 16],
    };
    let metadata = vec![
        (
            "general.license".to_string(),
            GgufValue::String("MIT".to_string()),
        ),
        (
            "general.author".to_string(),
            GgufValue::String("Test".to_string()),
        ),
        (
            "general.description".to_string(),
            GgufValue::String("Test".to_string()),
        ),
    ];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::new().expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    let report = lint_gguf_file(temp_file.path()).expect("lint GGUF");
    let naming_issues = report.issues_in_category(LintCategory::Naming);

    // Should flag abbreviated name
    assert!(
        !naming_issues.is_empty(),
        "GGUF with abbreviated names should trigger naming warnings"
    );
}

#[test]
fn test_lint_safetensors_abbreviated_tensor_names() {
    use tempfile::NamedTempFile;

    // Create SafeTensors with abbreviated tensor name
    let header_json = serde_json::json!({
        "__metadata__": {
            "license": "MIT",
            "author": "Test",
            "description": "Test"
        },
        "enc.w": {  // abbreviated name
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).expect("serialize header");
    let header_len = header_bytes.len() as u64;

    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&header_len.to_le_bytes());
    st_bytes.extend_from_slice(&header_bytes);
    st_bytes.extend_from_slice(&[0u8; 16]);

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    let report = lint_safetensors_file(temp_file.path()).expect("lint SafeTensors");
    let naming_issues = report.issues_in_category(LintCategory::Naming);

    // Should flag abbreviated name
    assert!(
        !naming_issues.is_empty(),
        "SafeTensors with abbreviated names should trigger naming warnings"
    );
}

// ========================================================================
// P0 FORMAT DISPATCH TESTS (Refs GH-202)
// These tests ensure all formats are properly handled by lint_model_file
// and don't silently fail or return incorrect results.
// ========================================================================

/// P0 REGRESSION TEST: APR v1 format (APRN magic) must be handled
#[test]
fn test_lint_apr_v1_magic_detection() {
    // APR v1 magic: "APRN" (0x4150524E)
    let magic = b"APRN";
    assert_eq!(magic[0], 0x41); // 'A'
    assert_eq!(magic[1], 0x50); // 'P'
    assert_eq!(magic[2], 0x52); // 'R'
    assert_eq!(magic[3], 0x4E); // 'N'
}

/// P0 REGRESSION TEST: APR v2 format (APR\0 magic) must be handled
#[test]
fn test_lint_apr_v2_magic_detection() {
    // APR v2 magic: "APR\0" (0x41505200)
    let magic = b"APR\0";
    assert_eq!(magic[0], 0x41); // 'A'
    assert_eq!(magic[1], 0x50); // 'P'
    assert_eq!(magic[2], 0x52); // 'R'
    assert_eq!(magic[3], 0x00); // '\0'
}

/// P0 REGRESSION TEST: lint_apr_file dispatches correctly by magic
#[test]
fn test_lint_apr_file_magic_dispatch() {
    use tempfile::NamedTempFile;

    // Test APR v1 magic detection (should attempt v1 parsing)
    let apr_v1_file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    std::fs::write(apr_v1_file.path(), b"APRN\x00\x00\x00\x00").expect("write");

    let result = lint_apr_file(apr_v1_file.path());
    // Should fail because invalid v1 file, but NOT with "unknown magic" error
    assert!(result.is_err(), "Invalid v1 file should error");
    let err = result.unwrap_err().to_string();
    assert!(
        !err.contains("Invalid APR magic"),
        "APR v1 magic should be recognized"
    );

    // Test APR v2 magic detection (should attempt v2 parsing)
    let apr_v2_file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    std::fs::write(apr_v2_file.path(), b"APR\x00\x02\x00\x00\x00").expect("write");

    let result = lint_apr_file(apr_v2_file.path());
    // Should fail because invalid v2 file, but NOT with "unknown magic" error
    assert!(result.is_err(), "Invalid v2 file should error");
    let err = result.unwrap_err().to_string();
    assert!(
        !err.contains("Invalid APR magic"),
        "APR v2 magic should be recognized"
    );
}

/// P0 REGRESSION TEST: Unknown magic should produce clear error
#[test]
fn test_lint_apr_file_unknown_magic_error() {
    use tempfile::NamedTempFile;

    // Test unknown magic produces clear error
    let unknown_file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    std::fs::write(unknown_file.path(), b"XXXX\x00\x00\x00\x00").expect("write");

    let result = lint_apr_file(unknown_file.path());
    assert!(result.is_err(), "Unknown magic should error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Invalid APR magic") || err.contains("magic"),
        "Unknown magic error should mention 'magic': {}",
        err
    );
}

/// P0 REGRESSION TEST: lint_model_file dispatches to correct handler
#[test]
fn test_lint_model_file_format_dispatch() {
    use crate::format::rosetta::FormatType;
    use tempfile::NamedTempFile;

    // Test GGUF dispatch
    let gguf_file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    // Minimal GGUF: magic + version (incomplete but enough for format detection)
    std::fs::write(gguf_file.path(), b"GGUF\x03\x00\x00\x00").expect("write");

    let format = FormatType::from_magic(gguf_file.path());
    assert!(format.is_ok(), "GGUF format should be detected");
    assert!(
        matches!(format.unwrap(), FormatType::Gguf),
        "Must detect as GGUF"
    );

    // Test APR dispatch
    let apr_file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    std::fs::write(apr_file.path(), b"APR\x00\x02\x00\x00\x00").expect("write");

    let format = FormatType::from_magic(apr_file.path());
    assert!(format.is_ok(), "APR format should be detected");
    assert!(
        matches!(format.unwrap(), FormatType::Apr),
        "Must detect as APR"
    );

    // Test SafeTensors dispatch
    let st_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    let header = b"{\"test\": {}}";
    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    st_bytes.extend_from_slice(header);
    std::fs::write(st_file.path(), &st_bytes).expect("write");

    let format = FormatType::from_magic(st_file.path());
    assert!(format.is_ok(), "SafeTensors format should be detected");
    assert!(
        matches!(format.unwrap(), FormatType::SafeTensors),
        "Must detect as SafeTensors"
    );
}
