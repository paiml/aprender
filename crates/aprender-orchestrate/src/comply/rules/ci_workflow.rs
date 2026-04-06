//! CI Workflow Parity Rule
//!
//! Ensures consistent GitHub Actions workflow configuration across PAIML stack projects.

use crate::comply::rule::{
    FixResult, RuleCategory, RuleResult, RuleViolation, StackComplianceRule, Suggestion,
    ViolationLevel,
};
use std::path::Path;

/// Extract string values from a YAML sequence into a vec.
fn extract_string_seq(node: Option<&serde_yaml_ng::Value>, out: &mut Vec<String>) {
    if let Some(seq) = node.and_then(|n| n.as_sequence()) {
        for item in seq {
            if let Some(s) = item.as_str() {
                out.push(s.to_string());
            }
        }
    }
}

/// CI workflow parity rule
#[derive(Debug)]
pub struct CiWorkflowRule {
    /// Required workflow file names (any match is OK)
    workflow_files: Vec<String>,
    /// Required jobs in the workflow
    required_jobs: Vec<String>,
}

impl Default for CiWorkflowRule {
    fn default() -> Self {
        Self::new()
    }
}

impl CiWorkflowRule {
    /// Create a new CI workflow rule with default configuration
    pub fn new() -> Self {
        Self {
            workflow_files: vec![
                "ci.yml".to_string(),
                "ci.yaml".to_string(),
                "rust.yml".to_string(),
                "rust.yaml".to_string(),
                "test.yml".to_string(),
                "test.yaml".to_string(),
            ],
            required_jobs: vec!["fmt".to_string(), "clippy".to_string(), "test".to_string()],
        }
    }

    /// Find the CI workflow file
    fn find_workflow(&self, project_path: &Path) -> Option<std::path::PathBuf> {
        let workflows_dir = project_path.join(".github").join("workflows");

        if !workflows_dir.exists() {
            return None;
        }

        for name in &self.workflow_files {
            let path = workflows_dir.join(name);
            if path.exists() {
                return Some(path);
            }
        }

        None
    }

    /// Parse workflow and extract jobs
    fn parse_workflow(&self, path: &Path) -> anyhow::Result<WorkflowData> {
        let content = std::fs::read_to_string(path)?;
        let yaml: serde_yaml_ng::Value = serde_yaml_ng::from_str(&content)?;

        let mut jobs = Vec::new();
        let mut matrix_os = Vec::new();
        let mut matrix_rust = Vec::new();
        let mut uses_nextest = false;
        let mut uses_llvm_cov = false;

        if let Some(jobs_map) = yaml.get("jobs").and_then(|j| j.as_mapping()) {
            for (job_name, job_value) in jobs_map {
                if let Some(name) = job_name.as_str() {
                    jobs.push(name.to_string());

                    // Check for matrix
                    if let Some(matrix) = job_value.get("strategy").and_then(|s| s.get("matrix")) {
                        extract_string_seq(matrix.get("os"), &mut matrix_os);
                        let rust = matrix.get("rust").or_else(|| matrix.get("toolchain"));
                        extract_string_seq(rust, &mut matrix_rust);
                    }

                    // Check steps for specific tools
                    let run_cmds = job_value
                        .get("steps")
                        .and_then(|s| s.as_sequence())
                        .into_iter()
                        .flatten()
                        .filter_map(|step| step.get("run").and_then(|r| r.as_str()));
                    for run in run_cmds {
                        uses_nextest |= run.contains("nextest");
                        uses_llvm_cov |= run.contains("llvm-cov");
                    }
                }
            }
        }

        Ok(WorkflowData { jobs, matrix_os, matrix_rust, uses_nextest, uses_llvm_cov })
    }

    /// Return an error result when no workflow file is found.
    fn no_workflow_result(&self, project_path: &Path) -> anyhow::Result<RuleResult> {
        let workflows_dir = project_path.join(".github").join("workflows");
        if !workflows_dir.exists() {
            return Ok(RuleResult::fail(vec![RuleViolation::new(
                "CI-001",
                "No .github/workflows directory found",
            )
            .with_severity(ViolationLevel::Error)
            .with_location(project_path.display().to_string())]));
        }
        Ok(RuleResult::fail(vec![RuleViolation::new(
            "CI-002",
            format!(
                "No CI workflow file found (expected one of: {})",
                self.workflow_files.join(", ")
            ),
        )
        .with_severity(ViolationLevel::Error)
        .with_location(workflows_dir.display().to_string())]))
    }

    /// Check that all required job types exist in the workflow.
    fn check_required_jobs(&self, data: &WorkflowData, workflow_path: &Path) -> Vec<RuleViolation> {
        self.required_jobs
            .iter()
            .filter(|required_job| {
                !data.jobs.iter().any(|j| {
                    j.to_lowercase().contains(&required_job.to_lowercase())
                        || j.to_lowercase().contains(&required_job.replace('-', "_").to_lowercase())
                })
            })
            .map(|required_job| {
                RuleViolation::new("CI-003", format!("Missing required job type: {required_job}"))
                    .with_severity(ViolationLevel::Error)
                    .with_location(workflow_path.display().to_string())
            })
            .collect()
    }

    /// Collect improvement suggestions for the workflow.
    fn collect_suggestions(&self, data: &WorkflowData, workflow_path: &Path) -> Vec<Suggestion> {
        let mut suggestions = Vec::new();
        if !data.uses_nextest {
            suggestions.push(
                Suggestion::new("Consider using cargo-nextest for faster test execution")
                    .with_location(workflow_path.display().to_string())
                    .with_fix("cargo nextest run".to_string()),
            );
        }
        if !data.uses_llvm_cov {
            suggestions.push(
                Suggestion::new("Consider using cargo-llvm-cov for coverage (not tarpaulin)")
                    .with_location(workflow_path.display().to_string())
                    .with_fix("cargo llvm-cov --html".to_string()),
            );
        }
        if !data.matrix_rust.is_empty() && !data.matrix_rust.contains(&"stable".to_string()) {
            suggestions.push(
                Suggestion::new("Consider including 'stable' in Rust toolchain matrix")
                    .with_location(workflow_path.display().to_string()),
            );
        }
        suggestions
    }
}

#[derive(Debug)]
struct WorkflowData {
    jobs: Vec<String>,
    matrix_os: Vec<String>,
    matrix_rust: Vec<String>,
    uses_nextest: bool,
    uses_llvm_cov: bool,
}

impl StackComplianceRule for CiWorkflowRule {
    fn id(&self) -> &'static str {
        "ci-workflow-parity"
    }

    fn description(&self) -> &'static str {
        "Ensures consistent CI workflow configuration across stack projects"
    }

    fn help(&self) -> Option<&str> {
        Some(
            "Required jobs: fmt, clippy, test\n\
             Recommended: nextest for testing, llvm-cov for coverage",
        )
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Ci
    }

    fn check(&self, project_path: &Path) -> anyhow::Result<RuleResult> {
        let workflow_path = match self.find_workflow(project_path) {
            Some(p) => p,
            None => return self.no_workflow_result(project_path),
        };

        let data = self.parse_workflow(&workflow_path)?;
        let violations = self.check_required_jobs(&data, &workflow_path);
        let suggestions = self.collect_suggestions(&data, &workflow_path);

        if violations.is_empty() {
            if suggestions.is_empty() {
                Ok(RuleResult::pass())
            } else {
                Ok(RuleResult::pass_with_suggestions(suggestions))
            }
        } else {
            let mut result = RuleResult::fail(violations);
            result.suggestions = suggestions;
            Ok(result)
        }
    }

    fn can_fix(&self) -> bool {
        false // CI workflow changes need manual review
    }

    fn fix(&self, _project_path: &Path) -> anyhow::Result<FixResult> {
        Ok(FixResult::failure("Auto-fix not supported for CI workflows - manual review required"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_workflow_dir(temp: &TempDir) -> std::path::PathBuf {
        let workflows_dir = temp.path().join(".github").join("workflows");
        std::fs::create_dir_all(&workflows_dir).unwrap();
        workflows_dir
    }

    #[test]
    fn test_ci_workflow_rule_creation() {
        let rule = CiWorkflowRule::new();
        assert_eq!(rule.id(), "ci-workflow-parity");
        assert!(rule.required_jobs.contains(&"test".to_string()));
    }

    #[test]
    fn test_missing_workflows_dir() {
        let temp = TempDir::new().unwrap();
        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations[0].code, "CI-001");
    }

    #[test]
    fn test_missing_ci_file() {
        let temp = TempDir::new().unwrap();
        create_workflow_dir(&temp);

        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations[0].code, "CI-002");
    }

    #[test]
    fn test_valid_ci_workflow() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);
        let ci_file = workflows_dir.join("ci.yml");

        let content = r#"
name: CI

on: [push, pull_request]

jobs:
  fmt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo fmt --check

  clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo clippy -- -D warnings

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cargo nextest run
"#;
        std::fs::write(&ci_file, content).unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(result.passed, "Should pass: {:?}", result.violations);
        // Should have suggestion for llvm-cov
        assert!(!result.suggestions.is_empty());
    }

    #[test]
    fn test_missing_job() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);
        let ci_file = workflows_dir.join("ci.yml");

        let content = r#"
name: CI

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test
"#;
        std::fs::write(&ci_file, content).unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(!result.passed);
        // Should have violations for missing fmt and clippy
        assert!(result.violations.len() >= 2);
    }

    // -------------------------------------------------------------------------
    // Additional Coverage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ci_workflow_rule_default() {
        let rule = CiWorkflowRule::default();
        assert_eq!(rule.id(), "ci-workflow-parity");
    }

    #[test]
    fn test_ci_workflow_description() {
        let rule = CiWorkflowRule::new();
        assert!(rule.description().contains("CI workflow"));
    }

    #[test]
    fn test_ci_workflow_help() {
        let rule = CiWorkflowRule::new();
        let help = rule.help();
        assert!(help.is_some());
        assert!(help.unwrap().contains("fmt"));
        assert!(help.unwrap().contains("clippy"));
    }

    #[test]
    fn test_ci_workflow_category() {
        let rule = CiWorkflowRule::new();
        assert_eq!(rule.category(), RuleCategory::Ci);
    }

    #[test]
    fn test_ci_workflow_can_fix() {
        let rule = CiWorkflowRule::new();
        assert!(!rule.can_fix());
    }

    #[test]
    fn test_ci_workflow_fix() {
        let temp = TempDir::new().unwrap();
        let rule = CiWorkflowRule::new();
        let result = rule.fix(temp.path()).unwrap();
        assert!(!result.success);
    }

    #[test]
    fn test_ci_workflow_rule_debug() {
        let rule = CiWorkflowRule::new();
        let debug_str = format!("{:?}", rule);
        assert!(debug_str.contains("CiWorkflowRule"));
    }

    #[test]
    fn test_find_workflow_rust_yml() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);
        std::fs::write(workflows_dir.join("rust.yml"), "name: Rust").unwrap();

        let rule = CiWorkflowRule::new();
        let path = rule.find_workflow(temp.path());
        assert!(path.is_some());
        assert!(path.unwrap().ends_with("rust.yml"));
    }

    #[test]
    fn test_find_workflow_test_yaml() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);
        std::fs::write(workflows_dir.join("test.yaml"), "name: Test").unwrap();

        let rule = CiWorkflowRule::new();
        let path = rule.find_workflow(temp.path());
        assert!(path.is_some());
    }

    #[test]
    fn test_find_workflow_none() {
        let temp = TempDir::new().unwrap();
        let rule = CiWorkflowRule::new();
        let path = rule.find_workflow(temp.path());
        assert!(path.is_none());
    }

    #[test]
    fn test_workflow_with_matrix() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);
        let ci_file = workflows_dir.join("ci.yml");

        let content = r#"
name: CI

jobs:
  fmt:
    runs-on: ubuntu-latest
    steps:
      - run: cargo fmt --check

  clippy:
    runs-on: ubuntu-latest
    steps:
      - run: cargo clippy

  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        rust: [stable, nightly]
    runs-on: ${{ matrix.os }}
    steps:
      - run: cargo nextest run
"#;
        std::fs::write(&ci_file, content).unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_workflow_with_llvm_cov() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);
        let ci_file = workflows_dir.join("ci.yml");

        let content = r#"
name: CI

jobs:
  fmt:
    steps:
      - run: cargo fmt --check

  clippy:
    steps:
      - run: cargo clippy

  test:
    steps:
      - run: cargo llvm-cov --html
"#;
        std::fs::write(&ci_file, content).unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_workflow_missing_stable_rust() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);
        let ci_file = workflows_dir.join("ci.yml");

        let content = r#"
name: CI

jobs:
  fmt:
    steps:
      - run: cargo fmt --check

  clippy:
    steps:
      - run: cargo clippy

  test:
    strategy:
      matrix:
        rust: [nightly, beta]
    steps:
      - run: cargo nextest run
      - run: cargo llvm-cov
"#;
        std::fs::write(&ci_file, content).unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(result.passed);
        // Should have suggestion for stable rust
        assert!(result.suggestions.iter().any(|s| s.message.contains("stable")));
    }

    #[test]
    fn test_workflow_data_debug() {
        let data = WorkflowData {
            jobs: vec!["test".to_string()],
            matrix_os: vec!["ubuntu-latest".to_string()],
            matrix_rust: vec!["stable".to_string()],
            uses_nextest: true,
            uses_llvm_cov: false,
        };
        let debug_str = format!("{:?}", data);
        assert!(debug_str.contains("WorkflowData"));
    }

    #[test]
    fn test_parse_workflow_invalid_yaml() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("invalid.yml");
        std::fs::write(&file, "invalid: yaml: content: [").unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.parse_workflow(&file);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_workflow_empty_yaml() {
        let temp = TempDir::new().unwrap();
        let file = temp.path().join("empty.yml");
        std::fs::write(&file, "name: Empty").unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.parse_workflow(&file).unwrap();
        assert!(result.jobs.is_empty());
    }

    #[test]
    fn test_job_name_variations() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);
        let ci_file = workflows_dir.join("ci.yml");

        // Test with underscore variations
        let content = r#"
name: CI

jobs:
  rust_fmt:
    steps:
      - run: cargo fmt --check

  rust_clippy:
    steps:
      - run: cargo clippy

  unit_test:
    steps:
      - run: cargo nextest run
      - run: cargo llvm-cov
"#;
        std::fs::write(&ci_file, content).unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(
            result.passed,
            "Should recognize _fmt, _clippy, _test variations: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_ci_workflow_alternative_filenames() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);

        // Test ci.yaml (not ci.yml)
        let content = r#"
name: CI

jobs:
  fmt:
    steps:
      - run: cargo fmt

  clippy:
    steps:
      - run: cargo clippy

  test:
    steps:
      - run: cargo test
"#;
        std::fs::write(workflows_dir.join("ci.yaml"), content).unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(result.passed);
    }

    #[test]
    fn test_workflow_toolchain_matrix() {
        let temp = TempDir::new().unwrap();
        let workflows_dir = create_workflow_dir(&temp);
        let ci_file = workflows_dir.join("ci.yml");

        let content = r#"
name: CI

jobs:
  fmt:
    steps:
      - run: cargo fmt

  clippy:
    steps:
      - run: cargo clippy

  test:
    strategy:
      matrix:
        toolchain: [stable, nightly]
    steps:
      - run: cargo nextest run
      - run: cargo llvm-cov
"#;
        std::fs::write(&ci_file, content).unwrap();

        let rule = CiWorkflowRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(result.passed);
    }
}
