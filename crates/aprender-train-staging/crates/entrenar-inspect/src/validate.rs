//! Model validation (Andon principle - surface problems immediately).

use entrenar_common::Result;
use std::path::Path;

/// Result of model validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the model is valid
    pub valid: bool,
    /// List of issues found
    pub issues: Vec<ValidationIssue>,
    /// List of warnings
    pub warnings: Vec<String>,
    /// Validation checks performed
    pub checks: Vec<ValidationCheck>,
}

impl ValidationResult {
    /// Check if there are any errors (not just warnings).
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    /// Format as human-readable report.
    pub fn to_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!(
            "Validation Result: {}\n\n",
            if self.valid { "PASS" } else { "FAIL" }
        ));

        if !self.issues.is_empty() {
            report.push_str("Issues:\n");
            for issue in &self.issues {
                let prefix = match issue.severity {
                    Severity::Error => "✗",
                    Severity::Warning => "⚠",
                    Severity::Info => "ℹ",
                };
                report.push_str(&format!("  {} {}: {}\n", prefix, issue.code, issue.message));
                if let Some(suggestion) = &issue.suggestion {
                    report.push_str(&format!("    → {suggestion}\n"));
                }
            }
            report.push('\n');
        }

        report.push_str("Checks Performed:\n");
        for check in &self.checks {
            let status = if check.passed { "✓" } else { "✗" };
            report.push_str(&format!("  {} {}\n", status, check.name));
        }

        report
    }
}

/// A validation issue.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Issue code for programmatic handling
    pub code: String,
    /// Human-readable message
    pub message: String,
    /// Severity level
    pub severity: Severity,
    /// Actionable suggestion
    pub suggestion: Option<String>,
    /// Affected tensor name (if applicable)
    pub tensor: Option<String>,
}

/// Severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Critical error - model unusable
    Error,
    /// Warning - model usable but may have issues
    Warning,
    /// Informational - not a problem
    Info,
}

/// A validation check that was performed.
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    /// Check name
    pub name: String,
    /// Whether check passed
    pub passed: bool,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

/// Model integrity checker.
pub struct IntegrityChecker {
    strict: bool,
}

impl Default for IntegrityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegrityChecker {
    /// Create a new integrity checker.
    pub fn new() -> Self {
        Self { strict: false }
    }

    /// Enable strict mode (treat warnings as errors).
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Validate a model file.
    pub fn validate(&self, path: &Path) -> Result<ValidationResult> {
        let mut issues = Vec::new();
        let mut warnings = Vec::new();
        let mut checks = Vec::new();

        // Check file exists
        let file_check = self.check_file_exists(path);
        checks.push(file_check.clone());
        if !file_check.passed {
            issues.push(ValidationIssue {
                code: "V001".to_string(),
                message: format!("File not found: {}", path.display()),
                severity: Severity::Error,
                suggestion: Some("Check the file path".to_string()),
                tensor: None,
            });
            return Ok(ValidationResult { valid: false, issues, warnings, checks });
        }

        // Check format
        let format_check = self.check_format(path);
        checks.push(format_check.clone());
        if !format_check.passed {
            issues.push(ValidationIssue {
                code: "V002".to_string(),
                message: "Unsupported or potentially unsafe format".to_string(),
                severity: if self.strict { Severity::Error } else { Severity::Warning },
                suggestion: Some("Use SafeTensors format for security".to_string()),
                tensor: None,
            });
        }

        // Check file size
        let size_check = self.check_file_size(path);
        checks.push(size_check.clone());
        if !size_check.passed {
            warnings.push("File size is unusually small - may be corrupted".to_string());
        }

        // In real implementation, would also check:
        // - Tensor shapes consistency
        // - NaN/Inf values
        // - Data type consistency
        // - Architecture constraints

        let valid = !issues.iter().any(|i| i.severity == Severity::Error);

        Ok(ValidationResult { valid, issues, warnings, checks })
    }

    fn check_file_exists(&self, path: &Path) -> ValidationCheck {
        let start = std::time::Instant::now();
        let passed = path.exists();
        ValidationCheck {
            name: "File exists".to_string(),
            passed,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    fn check_format(&self, path: &Path) -> ValidationCheck {
        let start = std::time::Instant::now();
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let passed = matches!(extension.to_lowercase().as_str(), "safetensors" | "gguf" | "apr");
        ValidationCheck {
            name: "Safe format".to_string(),
            passed,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    fn check_file_size(&self, path: &Path) -> ValidationCheck {
        let start = std::time::Instant::now();
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let passed = size > 1000; // At least 1KB
        ValidationCheck {
            name: "Valid file size".to_string(),
            passed,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }
}

/// Validate a model file.
pub fn validate_model(path: impl AsRef<Path>) -> Result<ValidationResult> {
    IntegrityChecker::new().validate(path.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_validation_missing_file() {
        let result =
            validate_model("/nonexistent/model.safetensors").expect("operation should succeed");
        assert!(!result.valid);
        assert!(result.issues.iter().any(|i| i.code == "V001"));
    }

    #[test]
    fn test_validation_safe_format() {
        let mut file =
            NamedTempFile::with_suffix(".safetensors").expect("temp file creation should succeed");
        file.write_all(&[0u8; 2000]).expect("file write should succeed");

        let result = validate_model(file.path()).expect("operation should succeed");
        assert!(result.checks.iter().any(|c| c.name == "Safe format" && c.passed));
    }

    #[test]
    fn test_validation_unsafe_format() {
        let mut file =
            NamedTempFile::with_suffix(".pt").expect("temp file creation should succeed");
        file.write_all(&[0u8; 2000]).expect("file write should succeed");

        let result = validate_model(file.path()).expect("operation should succeed");
        // In non-strict mode, unsafe format is a warning not error
        let format_check = result
            .checks
            .iter()
            .find(|c| c.name == "Safe format")
            .expect("operation should succeed");
        assert!(!format_check.passed);
    }

    #[test]
    fn test_strict_mode() {
        let mut file =
            NamedTempFile::with_suffix(".pt").expect("temp file creation should succeed");
        file.write_all(&[0u8; 2000]).expect("file write should succeed");

        let result = IntegrityChecker::new()
            .strict()
            .validate(file.path())
            .expect("operation should succeed");
        assert!(!result.valid); // Unsafe format is error in strict mode
    }

    #[test]
    fn test_validation_report() {
        let result = ValidationResult {
            valid: false,
            issues: vec![ValidationIssue {
                code: "V001".to_string(),
                message: "Test error".to_string(),
                severity: Severity::Error,
                suggestion: Some("Fix it".to_string()),
                tensor: None,
            }],
            warnings: vec![],
            checks: vec![ValidationCheck {
                name: "Test check".to_string(),
                passed: false,
                duration_ms: 1,
            }],
        };

        let report = result.to_report();
        assert!(report.contains("FAIL"));
        assert!(report.contains("V001"));
        assert!(report.contains("Fix it"));
    }

    #[test]
    fn test_validation_report_pass() {
        let result = ValidationResult {
            valid: true,
            issues: vec![],
            warnings: vec![],
            checks: vec![ValidationCheck {
                name: "Test check".to_string(),
                passed: true,
                duration_ms: 1,
            }],
        };

        let report = result.to_report();
        assert!(report.contains("PASS"));
        assert!(report.contains("✓"));
    }

    #[test]
    fn test_has_errors_with_error() {
        let result = ValidationResult {
            valid: false,
            issues: vec![ValidationIssue {
                code: "V001".to_string(),
                message: "Error".to_string(),
                severity: Severity::Error,
                suggestion: None,
                tensor: None,
            }],
            warnings: vec![],
            checks: vec![],
        };
        assert!(result.has_errors());
    }

    #[test]
    fn test_has_errors_with_warning_only() {
        let result = ValidationResult {
            valid: true,
            issues: vec![ValidationIssue {
                code: "V002".to_string(),
                message: "Warning".to_string(),
                severity: Severity::Warning,
                suggestion: None,
                tensor: None,
            }],
            warnings: vec![],
            checks: vec![],
        };
        assert!(!result.has_errors());
    }

    #[test]
    fn test_has_errors_with_info_only() {
        let result = ValidationResult {
            valid: true,
            issues: vec![ValidationIssue {
                code: "V003".to_string(),
                message: "Info".to_string(),
                severity: Severity::Info,
                suggestion: None,
                tensor: None,
            }],
            warnings: vec![],
            checks: vec![],
        };
        assert!(!result.has_errors());
    }

    #[test]
    fn test_integrity_checker_default() {
        let checker = IntegrityChecker::default();
        // Should be non-strict by default
        let mut file =
            NamedTempFile::with_suffix(".pt").expect("temp file creation should succeed");
        file.write_all(&[0u8; 2000]).expect("file write should succeed");

        let result = checker.validate(file.path()).expect("operation should succeed");
        // Non-strict mode: unsafe format is warning, not error
        assert!(result.valid);
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(Severity::Error, Severity::Error);
        assert_ne!(Severity::Error, Severity::Warning);
        assert_ne!(Severity::Warning, Severity::Info);
    }

    #[test]
    fn test_report_warning_symbol() {
        let result = ValidationResult {
            valid: true,
            issues: vec![ValidationIssue {
                code: "V002".to_string(),
                message: "Warning".to_string(),
                severity: Severity::Warning,
                suggestion: None,
                tensor: None,
            }],
            warnings: vec![],
            checks: vec![],
        };

        let report = result.to_report();
        assert!(report.contains("⚠"));
    }

    #[test]
    fn test_report_info_symbol() {
        let result = ValidationResult {
            valid: true,
            issues: vec![ValidationIssue {
                code: "V003".to_string(),
                message: "Info".to_string(),
                severity: Severity::Info,
                suggestion: None,
                tensor: None,
            }],
            warnings: vec![],
            checks: vec![],
        };

        let report = result.to_report();
        assert!(report.contains("ℹ"));
    }

    #[test]
    fn test_validation_small_file_warning() {
        let mut file =
            NamedTempFile::with_suffix(".safetensors").expect("temp file creation should succeed");
        file.write_all(&[0u8; 100]).expect("file write should succeed"); // Very small file

        let result = validate_model(file.path()).expect("operation should succeed");
        // Small file should generate a warning
        assert!(!result.warnings.is_empty() || !result.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn test_validation_gguf_format() {
        let mut file =
            NamedTempFile::with_suffix(".gguf").expect("temp file creation should succeed");
        file.write_all(&[0u8; 2000]).expect("file write should succeed");

        let result = validate_model(file.path()).expect("operation should succeed");
        let format_check = result.checks.iter().find(|c| c.name == "Safe format");
        assert!(format_check.is_some());
        assert!(format_check.expect("operation should succeed").passed);
    }

    #[test]
    fn test_validation_apr_format() {
        let mut file =
            NamedTempFile::with_suffix(".apr").expect("temp file creation should succeed");
        file.write_all(&[0u8; 2000]).expect("file write should succeed");

        let result = validate_model(file.path()).expect("operation should succeed");
        let format_check = result.checks.iter().find(|c| c.name == "Safe format");
        assert!(format_check.is_some());
        assert!(format_check.expect("operation should succeed").passed);
    }
}
