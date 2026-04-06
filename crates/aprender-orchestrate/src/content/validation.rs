//! Validation (Jidoka)
//!
//! Content validation for quality gates with stop-on-error behavior.

use super::ContentType;
use serde::{Deserialize, Serialize};

/// Validation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Critical - must halt
    Critical,
    /// Error - should halt
    Error,
    /// Warning - flag for revision
    Warning,
    /// Info - informational
    Info,
}

/// A single validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// Constraint that was violated
    pub constraint: String,
    /// Severity level
    pub severity: ValidationSeverity,
    /// Location in content (e.g., "paragraph 3", "code block 5")
    pub location: String,
    /// The offending text
    pub text: String,
    /// Suggested fix
    pub suggestion: String,
}

impl ValidationViolation {
    pub(crate) fn new(
        constraint: &str,
        severity: ValidationSeverity,
        location: String,
        text: String,
        suggestion: &str,
    ) -> Self {
        Self {
            constraint: constraint.to_string(),
            severity,
            location,
            text,
            suggestion: suggestion.to_string(),
        }
    }
}

/// Validation result from content validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Quality score (0-100)
    pub score: u8,
    /// List of violations
    pub violations: Vec<ValidationViolation>,
}

impl ValidationResult {
    /// Create a passing result
    pub fn pass(score: u8) -> Self {
        Self { passed: true, score, violations: Vec::new() }
    }

    /// Create a failing result
    pub fn fail(violations: Vec<ValidationViolation>) -> Self {
        let score = Self::calculate_score(&violations);
        Self { passed: false, score, violations }
    }

    /// Add a violation
    pub fn add_violation(&mut self, violation: ValidationViolation) {
        self.violations.push(violation);
        self.score = Self::calculate_score(&self.violations);
        self.passed = !self.violations.iter().any(|v| {
            matches!(v.severity, ValidationSeverity::Critical | ValidationSeverity::Error)
        });
    }

    /// Calculate score based on violations
    fn calculate_score(violations: &[ValidationViolation]) -> u8 {
        let mut score = 100i32;
        for v in violations {
            match v.severity {
                ValidationSeverity::Critical => score -= 50,
                ValidationSeverity::Error => score -= 25,
                ValidationSeverity::Warning => score -= 10,
                ValidationSeverity::Info => score -= 2,
            }
        }
        score.max(0) as u8
    }

    /// Check if there are critical violations
    pub fn has_critical(&self) -> bool {
        self.violations.iter().any(|v| v.severity == ValidationSeverity::Critical)
    }

    /// Check if there are errors
    pub fn has_errors(&self) -> bool {
        self.violations.iter().any(|v| v.severity == ValidationSeverity::Error)
    }

    /// Format as display string
    pub fn format_display(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Quality Score: {}/100\n\n", self.score));

        if self.violations.is_empty() {
            output.push_str("No violations found. ✓\n");
            return output;
        }

        output.push_str(&format!("Violations ({}):\n", self.violations.len()));
        for (i, v) in self.violations.iter().enumerate() {
            let prefix = if i == self.violations.len() - 1 { "└──" } else { "├──" };
            let severity = match v.severity {
                ValidationSeverity::Critical => "CRITICAL",
                ValidationSeverity::Error => "ERROR",
                ValidationSeverity::Warning => "WARNING",
                ValidationSeverity::Info => "INFO",
            };
            output.push_str(&format!(
                "{} [{}] {} @ {}\n",
                prefix, severity, v.constraint, v.location
            ));
            output.push_str(&format!("    Text: \"{}\"\n", v.text));
            output.push_str(&format!("    Fix: {}\n", v.suggestion));
        }

        output
    }
}

/// Content validator for Jidoka quality gates
#[derive(Debug, Clone)]
pub struct ContentValidator {
    /// Content type being validated
    content_type: ContentType,
}

impl ContentValidator {
    /// Create a new validator for a content type
    pub fn new(content_type: ContentType) -> Self {
        Self { content_type }
    }

    /// Validate content against all rules
    pub fn validate(&self, content: &str) -> ValidationResult {
        let mut result = ValidationResult::pass(100);

        // Run all validation checks
        self.validate_instructor_voice(content, &mut result);
        self.validate_code_blocks(content, &mut result);
        self.validate_heading_hierarchy(content, &mut result);
        self.validate_meta_commentary(content, &mut result);

        // Content-type specific validation
        match self.content_type {
            ContentType::BookChapter | ContentType::BlogPost => {
                self.validate_frontmatter(content, &mut result);
            }
            _ => {}
        }

        result
    }

    /// Check for meta-commentary (Andon)
    fn validate_meta_commentary(&self, content: &str, result: &mut ValidationResult) {
        let meta_phrases = [
            "in this chapter",
            "in this section",
            "we will learn",
            "we will explore",
            "we will discuss",
            "this chapter covers",
            "this section covers",
            "as mentioned earlier",
            "as we discussed",
        ];

        for (line_num, line) in content.lines().enumerate() {
            let lower = line.to_lowercase();
            for phrase in &meta_phrases {
                if lower.contains(phrase) {
                    result.add_violation(ValidationViolation::new(
                        "no_meta_commentary",
                        ValidationSeverity::Warning,
                        format!("line {}", line_num + 1),
                        line.trim().chars().take(60).collect::<String>() + "...",
                        "Use direct instruction instead of meta-commentary",
                    ));
                }
            }
        }
    }

    /// Validate instructor voice
    fn validate_instructor_voice(&self, content: &str, result: &mut ValidationResult) {
        // Check for passive voice indicators in instruction contexts
        let passive_indicators =
            ["is being", "was being", "has been", "have been", "will be shown", "can be seen"];

        for (line_num, line) in content.lines().enumerate() {
            let lower = line.to_lowercase();
            // Only check non-code lines
            if !line.trim().starts_with("```") && !line.trim().starts_with("//") {
                for phrase in &passive_indicators {
                    if lower.contains(phrase) {
                        result.add_violation(ValidationViolation::new(
                            "instructor_voice",
                            ValidationSeverity::Info,
                            format!("line {}", line_num + 1),
                            line.trim().chars().take(60).collect::<String>(),
                            "Consider using active voice for clearer instruction",
                        ));
                    }
                }
            }
        }
    }

    /// Validate code blocks have language specifiers
    fn validate_code_blocks(&self, content: &str, result: &mut ValidationResult) {
        let mut in_code_block = false;
        let mut block_start = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.trim().starts_with("```") {
                if !in_code_block {
                    // Starting a code block
                    in_code_block = true;
                    block_start = line_num + 1;
                    let lang = line.trim().trim_start_matches('`');
                    if lang.is_empty() {
                        result.add_violation(ValidationViolation::new(
                            "code_block_language",
                            ValidationSeverity::Warning,
                            format!("line {}", line_num + 1),
                            "```".to_string(),
                            "Specify language: ```rust, ```python, ```bash, etc.",
                        ));
                    }
                } else {
                    // Ending a code block
                    in_code_block = false;
                }
            }
        }

        // Check for unclosed code block
        if in_code_block {
            result.add_violation(ValidationViolation::new(
                "code_block_closed",
                ValidationSeverity::Error,
                format!("line {}", block_start),
                "Unclosed code block".to_string(),
                "Add closing ``` to code block",
            ));
        }
    }

    /// Validate heading hierarchy (no skipped levels)
    fn validate_heading_hierarchy(&self, content: &str, result: &mut ValidationResult) {
        let mut last_level = 0;

        for (line_num, line) in content.lines().enumerate() {
            if line.starts_with('#') {
                let level = line.chars().take_while(|c| *c == '#').count();
                if last_level > 0 && level > last_level + 1 {
                    result.add_violation(ValidationViolation::new(
                        "heading_hierarchy",
                        ValidationSeverity::Error,
                        format!("line {}", line_num + 1),
                        line.trim().to_string(),
                        &format!(
                            "Heading level {} skips from level {}. Use H{}.",
                            level,
                            last_level,
                            last_level + 1
                        ),
                    ));
                }
                last_level = level;
            }
        }
    }

    /// Validate frontmatter presence (for BCH and BLP)
    fn validate_frontmatter(&self, content: &str, result: &mut ValidationResult) {
        let _has_yaml_frontmatter = content.starts_with("---");
        let has_toml_frontmatter = content.starts_with("+++");

        if self.content_type == ContentType::BlogPost && !has_toml_frontmatter {
            result.add_violation(ValidationViolation::new(
                "frontmatter_present",
                ValidationSeverity::Critical,
                "beginning".to_string(),
                "Missing TOML frontmatter".to_string(),
                "Add +++ frontmatter with title, date, description",
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ValidationSeverity Tests
    // =========================================================================

    #[test]
    fn test_validation_severity_equality() {
        assert_eq!(ValidationSeverity::Critical, ValidationSeverity::Critical);
        assert_ne!(ValidationSeverity::Critical, ValidationSeverity::Error);
    }

    #[test]
    fn test_validation_severity_serialization() {
        let severity = ValidationSeverity::Warning;
        let json = serde_json::to_string(&severity).expect("json serialize failed");
        let deserialized: ValidationSeverity =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized, severity);
    }

    // =========================================================================
    // ValidationViolation Tests
    // =========================================================================

    #[test]
    fn test_validation_violation_new() {
        let v = ValidationViolation::new(
            "test_constraint",
            ValidationSeverity::Error,
            "line 1".to_string(),
            "offending text".to_string(),
            "suggested fix",
        );
        assert_eq!(v.constraint, "test_constraint");
        assert_eq!(v.severity, ValidationSeverity::Error);
        assert_eq!(v.location, "line 1");
    }

    #[test]
    fn test_validation_violation_serialization() {
        let v = ValidationViolation::new(
            "test",
            ValidationSeverity::Info,
            "loc".to_string(),
            "text".to_string(),
            "fix",
        );
        let json = serde_json::to_string(&v).expect("json serialize failed");
        let deserialized: ValidationViolation =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized.constraint, v.constraint);
    }

    // =========================================================================
    // ValidationResult Tests
    // =========================================================================

    #[test]
    fn test_validation_result_pass() {
        let result = ValidationResult::pass(100);
        assert!(result.passed);
        assert_eq!(result.score, 100);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_validation_result_fail() {
        let violations = vec![ValidationViolation::new(
            "test",
            ValidationSeverity::Error,
            "loc".to_string(),
            "text".to_string(),
            "fix",
        )];
        let result = ValidationResult::fail(violations);
        assert!(!result.passed);
        assert_eq!(result.score, 75); // 100 - 25 for error
    }

    #[test]
    fn test_validation_result_add_violation() {
        let mut result = ValidationResult::pass(100);
        result.add_violation(ValidationViolation::new(
            "test",
            ValidationSeverity::Warning,
            "loc".to_string(),
            "text".to_string(),
            "fix",
        ));
        assert!(result.passed); // Warnings don't fail
        assert_eq!(result.score, 90); // 100 - 10 for warning
    }

    #[test]
    fn test_validation_result_add_critical() {
        let mut result = ValidationResult::pass(100);
        result.add_violation(ValidationViolation::new(
            "test",
            ValidationSeverity::Critical,
            "loc".to_string(),
            "text".to_string(),
            "fix",
        ));
        assert!(!result.passed); // Critical fails
        assert_eq!(result.score, 50); // 100 - 50 for critical
    }

    #[test]
    fn test_validation_result_has_critical() {
        let mut result = ValidationResult::pass(100);
        assert!(!result.has_critical());

        result.add_violation(ValidationViolation::new(
            "test",
            ValidationSeverity::Critical,
            "loc".to_string(),
            "text".to_string(),
            "fix",
        ));
        assert!(result.has_critical());
    }

    #[test]
    fn test_validation_result_has_errors() {
        let mut result = ValidationResult::pass(100);
        assert!(!result.has_errors());

        result.add_violation(ValidationViolation::new(
            "test",
            ValidationSeverity::Error,
            "loc".to_string(),
            "text".to_string(),
            "fix",
        ));
        assert!(result.has_errors());
    }

    #[test]
    fn test_validation_result_format_display_no_violations() {
        let result = ValidationResult::pass(100);
        let output = result.format_display();
        assert!(output.contains("Quality Score: 100/100"));
        assert!(output.contains("No violations found"));
    }

    #[test]
    fn test_validation_result_format_display_with_violations() {
        let violations = vec![ValidationViolation::new(
            "test_constraint",
            ValidationSeverity::Error,
            "line 1".to_string(),
            "bad text".to_string(),
            "use good text",
        )];
        let result = ValidationResult::fail(violations);
        let output = result.format_display();
        assert!(output.contains("[ERROR]"));
        assert!(output.contains("test_constraint"));
        assert!(output.contains("bad text"));
    }

    #[test]
    fn test_validation_result_default() {
        let result = ValidationResult::default();
        assert!(!result.passed);
        assert_eq!(result.score, 0);
    }

    #[test]
    fn test_validation_result_score_floor() {
        // Test that score doesn't go below 0
        let mut result = ValidationResult::pass(100);
        for _ in 0..10 {
            result.add_violation(ValidationViolation::new(
                "test",
                ValidationSeverity::Critical,
                "loc".to_string(),
                "text".to_string(),
                "fix",
            ));
        }
        assert_eq!(result.score, 0); // Should floor at 0
    }

    // =========================================================================
    // ContentValidator Tests
    // =========================================================================

    #[test]
    fn test_content_validator_new() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        // Just test it creates without panic
        assert!(std::mem::size_of_val(&validator) > 0);
    }

    #[test]
    fn test_content_validator_clean_content() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# Title\n\nSome clean content here.\n\n```rust\nfn main() {}\n```\n";
        let result = validator.validate(content);
        assert!(result.passed);
    }

    #[test]
    fn test_content_validator_meta_commentary() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# Title\n\nIn this chapter, we will learn about Rust.\n";
        let result = validator.validate(content);
        // Should have a warning for meta-commentary
        assert!(result.violations.iter().any(|v| v.constraint == "no_meta_commentary"));
    }

    #[test]
    fn test_content_validator_instructor_voice() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# Title\n\nThe code has been written and will be shown below.\n";
        let result = validator.validate(content);
        // Should have a warning for passive voice
        assert!(result.violations.iter().any(|v| v.constraint == "instructor_voice"));
    }

    #[test]
    fn test_content_validator_heading_hierarchy() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# Title\n\n### Skipped H2\n";
        let result = validator.validate(content);
        // Should have error for skipped heading level
        assert!(result.violations.iter().any(|v| v.constraint == "heading_hierarchy"));
    }

    #[test]
    fn test_content_validator_blog_post_missing_frontmatter() {
        let validator = ContentValidator::new(ContentType::BlogPost);
        let content = "# My Blog Post\n\nContent here.\n";
        let result = validator.validate(content);
        // Should fail for missing TOML frontmatter
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.constraint == "frontmatter_present"));
    }

    #[test]
    fn test_content_validator_blog_post_with_frontmatter() {
        let validator = ContentValidator::new(ContentType::BlogPost);
        let content = "+++\ntitle = \"Test\"\n+++\n\n# My Blog Post\n\nContent here.\n";
        let result = validator.validate(content);
        // Should not fail for frontmatter
        assert!(!result.violations.iter().any(|v| v.constraint == "frontmatter_present"));
    }

    #[test]
    fn test_content_validator_code_block_without_lang() {
        let validator = ContentValidator::new(ContentType::BookChapter);
        let content = "# Title\n\n```\ncode without language\n```\n";
        let result = validator.validate(content);
        // Should have warning for code block without language
        assert!(result.violations.iter().any(|v| v.constraint == "code_block_language"));
    }
}
