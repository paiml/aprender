//! PMAT Bridge for Quality Gate Integration (PROBAR-PMAT-001)
//!
//! Integrates pmat static analysis results into probar compliance checking.
//!
//! ## Rationale
//!
//! Rather than duplicating pmat's capabilities (SATD, complexity, dead code),
//! probar delegates static analysis to pmat and integrates results into its
//! compliance framework.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use jugar_probar::comply::PmatBridge;
//!
//! let bridge = PmatBridge::new();
//! let result = bridge.run_quality_gate(Path::new("src/"))?;
//!
//! println!("SATD violations: {}", result.satd_count);
//! println!("Complexity violations: {}", result.complexity_count);
//! ```

use std::path::Path;
use std::process::Command;

use super::{ComplianceCheck, ComplianceResult};

/// Result of pmat quality gate analysis
#[derive(Debug, Default, Clone)]
pub struct PmatResult {
    /// SATD (Self-Admitted Technical Debt) violations
    pub satd_count: usize,
    /// SATD critical count
    pub satd_critical: usize,
    /// Complexity violations
    pub complexity_count: usize,
    /// Dead code violations
    pub dead_code_count: usize,
    /// Code duplication violations
    pub duplicate_count: usize,
    /// Security violations
    pub security_count: usize,
    /// Test coverage violations
    pub coverage_count: usize,
    /// Documentation violations
    pub documentation_count: usize,
    /// Total violations
    pub total_violations: usize,
    /// Whether quality gate passed
    pub passed: bool,
    /// Raw output from pmat
    pub raw_output: String,
}

impl PmatResult {
    /// Check if there are critical issues
    #[must_use]
    pub fn has_critical(&self) -> bool {
        self.satd_critical > 0 || self.security_count > 0
    }

    /// Get error-level count (critical + security)
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.satd_critical + self.security_count
    }

    /// Get warning-level count
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.total_violations.saturating_sub(self.error_count())
    }
}

/// Bridge to pmat for quality gate integration
#[derive(Debug, Default)]
pub struct PmatBridge {
    /// Path to pmat binary (defaults to "pmat")
    pmat_path: String,
    /// Additional flags for pmat
    extra_flags: Vec<String>,
}

impl PmatBridge {
    /// Create a new pmat bridge with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            pmat_path: "pmat".to_string(),
            extra_flags: Vec::new(),
        }
    }

    /// Set custom pmat binary path
    #[must_use]
    pub fn with_pmat_path(mut self, path: impl Into<String>) -> Self {
        self.pmat_path = path.into();
        self
    }

    /// Add extra flags for pmat
    #[must_use]
    pub fn with_flag(mut self, flag: impl Into<String>) -> Self {
        self.extra_flags.push(flag.into());
        self
    }

    /// Check if pmat is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        Command::new(&self.pmat_path)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    /// Run pmat quality-gate and parse results
    ///
    /// # Arguments
    /// * `path` - Path to analyze
    ///
    /// # Errors
    /// Returns error if pmat is not available or fails to run
    pub fn run_quality_gate(&self, path: &Path) -> Result<PmatResult, String> {
        if !self.is_available() {
            return Err("pmat not found. Install with: cargo install pmat".to_string());
        }

        let mut cmd = Command::new(&self.pmat_path);
        cmd.arg("quality-gate").arg(path);

        for flag in &self.extra_flags {
            cmd.arg(flag);
        }

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to run pmat: {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{stdout}\n{stderr}");

        Ok(self.parse_output(&combined, output.status.success()))
    }

    /// Parse pmat output into structured result
    fn parse_output(&self, output: &str, success: bool) -> PmatResult {
        let mut result = PmatResult {
            passed: success,
            raw_output: output.to_string(),
            ..Default::default()
        };

        // Parse violation counts from output
        // Format: "  🔍 Checking complexity... 64 violations found"
        for line in output.lines() {
            let line_lower = line.to_lowercase();

            if line_lower.contains("complexity") && line_lower.contains("violations") {
                result.complexity_count = Self::extract_count(line);
            } else if line_lower.contains("dead code") && line_lower.contains("violations") {
                result.dead_code_count = Self::extract_count(line);
            } else if line_lower.contains("technical debt") && line_lower.contains("violations") {
                result.satd_count = Self::extract_count(line);
                // Check for critical count in format "355 violations (17 critical)"
                if let Some(critical_start) = line.find('(') {
                    if let Some(critical_end) = line.find(" critical") {
                        if critical_start < critical_end {
                            let critical_str = &line[critical_start + 1..critical_end];
                            result.satd_critical = critical_str.trim().parse().unwrap_or(0);
                        }
                    }
                }
            } else if line_lower.contains("duplicat") && line_lower.contains("violations") {
                result.duplicate_count = Self::extract_count(line);
            } else if line_lower.contains("security") && line_lower.contains("violations") {
                result.security_count = Self::extract_count(line);
            } else if line_lower.contains("coverage") && line_lower.contains("violations") {
                result.coverage_count = Self::extract_count(line);
            } else if line_lower.contains("documentation") && line_lower.contains("violations") {
                result.documentation_count = Self::extract_count(line);
            } else if line_lower.contains("total violations:") {
                result.total_violations = Self::extract_count(line);
            }
        }

        // If total wasn't explicitly found, sum up
        if result.total_violations == 0 {
            result.total_violations = result.satd_count
                + result.complexity_count
                + result.dead_code_count
                + result.duplicate_count
                + result.security_count
                + result.coverage_count
                + result.documentation_count;
        }

        result
    }

    /// Extract numeric count from a line
    fn extract_count(line: &str) -> usize {
        // Find first number in the line
        let mut num_str = String::new();
        let mut found_digit = false;

        for c in line.chars() {
            if c.is_ascii_digit() {
                num_str.push(c);
                found_digit = true;
            } else if found_digit {
                break;
            }
        }

        num_str.parse().unwrap_or(0)
    }

    /// Convert pmat results to compliance checks
    #[must_use]
    pub fn to_compliance_checks(&self, result: &PmatResult) -> Vec<ComplianceCheck> {
        let mut checks = Vec::new();

        // SATD check
        if result.satd_count == 0 {
            checks.push(ComplianceCheck::pass("PMAT-SATD-001", "SATD Detection"));
        } else if result.satd_critical > 0 {
            checks.push(ComplianceCheck::fail(
                "PMAT-SATD-001",
                "SATD Detection",
                &format!(
                    "{} SATD violations ({} critical)",
                    result.satd_count, result.satd_critical
                ),
                result.satd_count,
            ));
        } else {
            checks.push(ComplianceCheck::warn(
                "PMAT-SATD-001",
                "SATD Detection",
                &format!("{} SATD violations", result.satd_count),
                result.satd_count,
            ));
        }

        // Complexity check
        if result.complexity_count == 0 {
            checks.push(ComplianceCheck::pass(
                "PMAT-COMPLEXITY-001",
                "Complexity Analysis",
            ));
        } else {
            checks.push(ComplianceCheck::warn(
                "PMAT-COMPLEXITY-001",
                "Complexity Analysis",
                &format!("{} complexity violations", result.complexity_count),
                result.complexity_count,
            ));
        }

        // Dead code check
        if result.dead_code_count == 0 {
            checks.push(ComplianceCheck::pass(
                "PMAT-DEADCODE-001",
                "Dead Code Detection",
            ));
        } else {
            checks.push(ComplianceCheck::warn(
                "PMAT-DEADCODE-001",
                "Dead Code Detection",
                &format!("{} dead code violations", result.dead_code_count),
                result.dead_code_count,
            ));
        }

        // Security check
        if result.security_count == 0 {
            checks.push(ComplianceCheck::pass(
                "PMAT-SECURITY-001",
                "Security Analysis",
            ));
        } else {
            checks.push(ComplianceCheck::fail(
                "PMAT-SECURITY-001",
                "Security Analysis",
                &format!("{} security violations", result.security_count),
                result.security_count,
            ));
        }

        // Duplicate check
        if result.duplicate_count == 0 {
            checks.push(ComplianceCheck::pass(
                "PMAT-DUPLICATE-001",
                "Code Duplication",
            ));
        } else {
            checks.push(ComplianceCheck::warn(
                "PMAT-DUPLICATE-001",
                "Code Duplication",
                &format!("{} duplicate code violations", result.duplicate_count),
                result.duplicate_count,
            ));
        }

        checks
    }

    /// Run quality gate and return compliance result
    ///
    /// # Arguments
    /// * `path` - Path to analyze
    ///
    /// # Errors
    /// Returns error if pmat fails
    pub fn check_compliance(&self, path: &Path) -> Result<ComplianceResult, String> {
        let pmat_result = self.run_quality_gate(path)?;
        let checks = self.to_compliance_checks(&pmat_result);

        let mut result = ComplianceResult::new();
        result.files_analyzed = 1; // pmat analyzes directory

        for check in checks {
            result.add_check(check);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comply::ComplianceStatus;

    #[test]
    fn test_parse_output_full() {
        let bridge = PmatBridge::new();
        let output = r#"
🔍 Running quality gate checks...

  🔍 Checking complexity... 64 violations found
  🔍 Checking dead code... 6 violations found
  🔍 Checking technical debt... 355 violations found
  🔍 Checking security... 0 violations found
  🔍 Checking duplicates... 88 violations found
Quality Gate: FAILED
Total violations: 513
"#;

        let result = bridge.parse_output(output, false);

        assert_eq!(result.complexity_count, 64);
        assert_eq!(result.dead_code_count, 6);
        assert_eq!(result.satd_count, 355);
        assert_eq!(result.security_count, 0);
        assert_eq!(result.duplicate_count, 88);
        assert_eq!(result.total_violations, 513);
        assert!(!result.passed);
    }

    #[test]
    fn test_parse_output_passing() {
        let bridge = PmatBridge::new();
        let output = r#"
🔍 Running quality gate checks...

  🔍 Checking complexity... 0 violations found
  🔍 Checking dead code... 0 violations found
  🔍 Checking technical debt... 0 violations found
  🔍 Checking security... 0 violations found
Quality Gate: PASSED
Total violations: 0
"#;

        let result = bridge.parse_output(output, true);

        assert_eq!(result.total_violations, 0);
        assert!(result.passed);
    }

    #[test]
    fn test_extract_count() {
        assert_eq!(PmatBridge::extract_count("64 violations found"), 64);
        assert_eq!(PmatBridge::extract_count("found 123 issues"), 123);
        assert_eq!(PmatBridge::extract_count("no numbers here"), 0);
        assert_eq!(
            PmatBridge::extract_count("355 violations (17 critical)"),
            355
        );
    }

    #[test]
    fn test_to_compliance_checks() {
        let bridge = PmatBridge::new();

        let result = PmatResult {
            satd_count: 10,
            satd_critical: 2,
            complexity_count: 5,
            security_count: 1,
            ..Default::default()
        };

        let checks = bridge.to_compliance_checks(&result);

        // Should have checks for SATD, complexity, dead code, security, duplicates
        assert!(checks.len() >= 4);

        // SATD with critical should be a failure
        let satd_check = checks.iter().find(|c| c.id == "PMAT-SATD-001").unwrap();
        assert_eq!(satd_check.status, ComplianceStatus::Fail);

        // Security violation should be a failure
        let security_check = checks.iter().find(|c| c.id == "PMAT-SECURITY-001").unwrap();
        assert_eq!(security_check.status, ComplianceStatus::Fail);

        // Complexity should be a warning
        let complexity_check = checks
            .iter()
            .find(|c| c.id == "PMAT-COMPLEXITY-001")
            .unwrap();
        assert_eq!(complexity_check.status, ComplianceStatus::Warn);
    }

    #[test]
    fn test_pmat_result_helpers() {
        let result = PmatResult {
            satd_count: 100,
            satd_critical: 5,
            security_count: 2,
            complexity_count: 50,
            total_violations: 157,
            ..Default::default()
        };

        assert!(result.has_critical());
        assert_eq!(result.error_count(), 7); // 5 critical + 2 security
        assert_eq!(result.warning_count(), 150); // 157 - 7
    }

    #[test]
    fn test_builder_pattern() {
        let bridge = PmatBridge::new()
            .with_pmat_path("/custom/pmat")
            .with_flag("--strict");

        assert_eq!(bridge.pmat_path, "/custom/pmat");
        assert!(bridge.extra_flags.contains(&"--strict".to_string()));
    }

    #[test]
    fn test_to_compliance_checks_all_pass() {
        let bridge = PmatBridge::new();
        let result = PmatResult::default(); // all zeros

        let checks = bridge.to_compliance_checks(&result);
        assert_eq!(checks.len(), 5);
        for check in &checks {
            assert_eq!(check.status, ComplianceStatus::Pass);
        }
    }

    #[test]
    fn test_to_compliance_checks_satd_warn_no_critical() {
        let bridge = PmatBridge::new();
        let result = PmatResult {
            satd_count: 5,
            satd_critical: 0, // no critical => warn, not fail
            ..Default::default()
        };

        let checks = bridge.to_compliance_checks(&result);
        let satd = checks.iter().find(|c| c.id == "PMAT-SATD-001").unwrap();
        assert_eq!(satd.status, ComplianceStatus::Warn);
    }

    #[test]
    fn test_to_compliance_checks_dead_code_warn() {
        let bridge = PmatBridge::new();
        let result = PmatResult {
            dead_code_count: 12,
            ..Default::default()
        };

        let checks = bridge.to_compliance_checks(&result);
        let dead = checks.iter().find(|c| c.id == "PMAT-DEADCODE-001").unwrap();
        assert_eq!(dead.status, ComplianceStatus::Warn);
    }

    #[test]
    fn test_to_compliance_checks_duplicates_warn() {
        let bridge = PmatBridge::new();
        let result = PmatResult {
            duplicate_count: 8,
            ..Default::default()
        };

        let checks = bridge.to_compliance_checks(&result);
        let dup = checks
            .iter()
            .find(|c| c.id == "PMAT-DUPLICATE-001")
            .unwrap();
        assert_eq!(dup.status, ComplianceStatus::Warn);
    }
}
