//! Tests for validation types and ContentValidator

use crate::content::*;

#[test]
#[allow(non_snake_case)]
fn test_VALID_001_validation_result_pass() {
    let result = ValidationResult::pass(100);
    assert!(result.passed);
    assert_eq!(result.score, 100);
    assert!(result.violations.is_empty());
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_002_validation_result_fail() {
    let violations = vec![ValidationViolation {
        constraint: "test".to_string(),
        severity: ValidationSeverity::Error,
        location: "line 1".to_string(),
        text: "bad text".to_string(),
        suggestion: "fix it".to_string(),
    }];
    let result = ValidationResult::fail(violations);
    assert!(!result.passed);
    assert!(result.score < 100);
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_003_validation_score_calculation() {
    let mut result = ValidationResult::pass(100);
    result.add_violation(ValidationViolation {
        constraint: "test".to_string(),
        severity: ValidationSeverity::Warning,
        location: "line 1".to_string(),
        text: "text".to_string(),
        suggestion: "fix".to_string(),
    });
    assert_eq!(result.score, 90); // 100 - 10 for warning
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_004_validation_has_critical() {
    let mut result = ValidationResult::pass(100);
    assert!(!result.has_critical());
    result.add_violation(ValidationViolation {
        constraint: "test".to_string(),
        severity: ValidationSeverity::Critical,
        location: "line 1".to_string(),
        text: "text".to_string(),
        suggestion: "fix".to_string(),
    });
    assert!(result.has_critical());
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_005_validator_meta_commentary() {
    let validator = ContentValidator::new(ContentType::BookChapter);
    let content = "# Chapter 1\n\nIn this chapter, we will learn about Rust.";
    let result = validator.validate(content);
    assert!(result.violations.iter().any(|v| v.constraint == "no_meta_commentary"));
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_006_validator_code_block_no_language() {
    let validator = ContentValidator::new(ContentType::BookChapter);
    let content = "# Chapter 1\n\n```\nfn main() {}\n```";
    let result = validator.validate(content);
    assert!(result.violations.iter().any(|v| v.constraint == "code_block_language"));
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_007_validator_code_block_with_language() {
    let validator = ContentValidator::new(ContentType::BookChapter);
    let content = "# Chapter 1\n\n```rust\nfn main() {}\n```";
    let result = validator.validate(content);
    assert!(!result.violations.iter().any(|v| v.constraint == "code_block_language"));
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_008_validator_heading_hierarchy_ok() {
    let validator = ContentValidator::new(ContentType::BookChapter);
    let content = "# H1\n\n## H2\n\n### H3";
    let result = validator.validate(content);
    assert!(!result.violations.iter().any(|v| v.constraint == "heading_hierarchy"));
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_009_validator_heading_hierarchy_skipped() {
    let validator = ContentValidator::new(ContentType::BookChapter);
    let content = "# H1\n\n### H3 skipped H2";
    let result = validator.validate(content);
    assert!(result.violations.iter().any(|v| v.constraint == "heading_hierarchy"));
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_010_validator_unclosed_code_block() {
    let validator = ContentValidator::new(ContentType::BookChapter);
    let content = "# Chapter\n\n```rust\nfn main() {}";
    let result = validator.validate(content);
    assert!(result.violations.iter().any(|v| v.constraint == "code_block_closed"));
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_011_validator_blog_missing_frontmatter() {
    let validator = ContentValidator::new(ContentType::BlogPost);
    let content = "# Blog Post\n\nContent here.";
    let result = validator.validate(content);
    assert!(result.violations.iter().any(|v| v.constraint == "frontmatter_present"));
}

#[test]
#[allow(non_snake_case)]
fn test_VALID_012_format_display() {
    let mut result = ValidationResult::pass(100);
    result.add_violation(ValidationViolation {
        constraint: "test_constraint".to_string(),
        severity: ValidationSeverity::Warning,
        location: "line 5".to_string(),
        text: "some text".to_string(),
        suggestion: "fix this".to_string(),
    });
    let display = result.format_display();
    assert!(display.contains("90/100"));
    assert!(display.contains("WARNING"));
    assert!(display.contains("test_constraint"));
}
