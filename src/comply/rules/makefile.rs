//! Makefile Target Consistency Rule
//!
//! Ensures all PAIML stack projects have consistent Makefile targets.

use crate::comply::rule::{
    FixDetail, FixResult, RuleCategory, RuleResult, RuleViolation, StackComplianceRule, Suggestion,
    ViolationLevel,
};
use std::collections::HashMap;
use std::path::Path;

/// Makefile target consistency rule
#[derive(Debug)]
pub struct MakefileRule {
    /// Required targets with expected patterns
    required_targets: HashMap<String, TargetSpec>,
    /// Prohibited commands
    prohibited_commands: Vec<String>,
}

#[derive(Debug, Clone)]
struct TargetSpec {
    pattern: Option<String>,
    description: String,
}

impl Default for MakefileRule {
    fn default() -> Self {
        Self::new()
    }
}

impl MakefileRule {
    /// Create a new Makefile rule with default configuration
    pub fn new() -> Self {
        let mut required_targets = HashMap::new();

        required_targets.insert(
            "test-fast".to_string(),
            TargetSpec {
                pattern: Some("cargo nextest run --lib".to_string()),
                description: "Fast unit tests".to_string(),
            },
        );

        required_targets.insert(
            "test".to_string(),
            TargetSpec {
                pattern: Some("cargo nextest run".to_string()),
                description: "Standard tests".to_string(),
            },
        );

        required_targets.insert(
            "lint".to_string(),
            TargetSpec {
                pattern: Some("cargo clippy".to_string()),
                description: "Clippy linting".to_string(),
            },
        );

        required_targets.insert(
            "fmt".to_string(),
            TargetSpec {
                pattern: Some("cargo fmt".to_string()),
                description: "Format code".to_string(),
            },
        );

        required_targets.insert(
            "coverage".to_string(),
            TargetSpec {
                pattern: Some("cargo llvm-cov".to_string()),
                description: "Coverage report".to_string(),
            },
        );

        Self {
            required_targets,
            prohibited_commands: vec!["cargo tarpaulin".to_string(), "cargo-tarpaulin".to_string()],
        }
    }

    fn check_required_targets(
        &self,
        targets: &HashMap<String, MakefileTarget>,
        violations: &mut Vec<RuleViolation>,
        suggestions: &mut Vec<Suggestion>,
    ) {
        for (target_name, spec) in &self.required_targets {
            let Some(target) = targets.get(target_name) else {
                violations.push(
                    RuleViolation::new("MK-002", format!("Missing required target: {target_name}"))
                        .with_severity(ViolationLevel::Error)
                        .with_location("Makefile".to_string())
                        .with_diff(format!("{target_name}: <command>"), "(not defined)".to_string())
                        .fixable(),
                );
                continue;
            };

            if let Some(pattern) = &spec.pattern {
                let has_pattern = target.commands.iter().any(|cmd| cmd.contains(pattern));
                if !has_pattern {
                    let msg = format!(
                        "Target '{target_name}' should include '{pattern}' for {}",
                        spec.description
                    );
                    suggestions.push(Suggestion::new(msg).with_location("Makefile".to_string()));
                }
            }

            self.check_prohibited_in_target(target_name, &target.commands, violations);
        }
    }

    fn check_prohibited_in_target(
        &self,
        target_name: &str,
        cmds: &[String],
        violations: &mut Vec<RuleViolation>,
    ) {
        for prohibited in &self.prohibited_commands {
            if cmds.iter().any(|cmd| cmd.contains(prohibited)) {
                let msg = format!("Target '{target_name}' uses prohibited command: {prohibited}");
                let diff_left = format!("cargo llvm-cov (for {target_name})");
                violations.push(
                    RuleViolation::new("MK-003", msg)
                        .with_severity(ViolationLevel::Critical)
                        .with_location("Makefile".to_string())
                        .with_diff(diff_left, prohibited.to_string()),
                );
            }
        }
    }

    fn check_all_prohibited(
        &self,
        targets: &HashMap<String, MakefileTarget>,
        violations: &mut Vec<RuleViolation>,
    ) {
        for target in targets.values() {
            if self.required_targets.contains_key(&target.name) {
                continue;
            }
            self.check_prohibited_in_target(&target.name, &target.commands, violations);
        }
    }

    /// Parse a Makefile and extract targets
    fn parse_makefile(&self, path: &Path) -> anyhow::Result<HashMap<String, MakefileTarget>> {
        let content = std::fs::read_to_string(path)?;
        let mut targets = HashMap::new();
        let mut current_target: Option<String> = None;
        let mut current_commands: Vec<String> = Vec::new();

        for line in content.lines() {
            // Skip comments and empty lines
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }

            // Check for target definition (name: [dependencies])
            if !line.starts_with('\t') && !line.starts_with(' ') && line.contains(':') {
                // Save previous target
                if let Some(name) = current_target.take() {
                    targets.insert(
                        name.clone(),
                        MakefileTarget { name, commands: std::mem::take(&mut current_commands) },
                    );
                }

                // Parse new target
                let parts: Vec<&str> = line.splitn(2, ':').collect();
                if !parts.is_empty() {
                    let target_name = parts[0].trim();
                    // Skip .PHONY and similar
                    if !target_name.starts_with('.') {
                        current_target = Some(target_name.to_string());
                    }
                }
            } else if (line.starts_with('\t') || line.starts_with(' ')) && current_target.is_some()
            {
                // Command line for current target
                let cmd = line.trim();
                if !cmd.is_empty() {
                    current_commands.push(cmd.to_string());
                }
            }
        }

        // Save last target
        if let Some(name) = current_target {
            targets.insert(name.clone(), MakefileTarget { name, commands: current_commands });
        }

        Ok(targets)
    }
}

#[derive(Debug)]
struct MakefileTarget {
    name: String,
    commands: Vec<String>,
}

impl StackComplianceRule for MakefileRule {
    fn id(&self) -> &'static str {
        "makefile-targets"
    }

    fn description(&self) -> &'static str {
        "Ensures consistent Makefile targets across stack projects"
    }

    fn help(&self) -> Option<&str> {
        Some(
            "Required targets: test-fast, test, lint, fmt, coverage\n\
             Prohibited commands: cargo tarpaulin",
        )
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Build
    }

    fn check(&self, project_path: &Path) -> anyhow::Result<RuleResult> {
        let makefile_path = project_path.join("Makefile");

        if !makefile_path.exists() {
            return Ok(RuleResult::fail(vec![RuleViolation::new("MK-001", "Makefile not found")
                .with_severity(ViolationLevel::Error)
                .with_location(project_path.display().to_string())
                .fixable()]));
        }

        let targets = self.parse_makefile(&makefile_path)?;
        let mut violations = Vec::new();
        let mut suggestions = Vec::new();

        self.check_required_targets(&targets, &mut violations, &mut suggestions);
        self.check_all_prohibited(&targets, &mut violations);

        if violations.is_empty() {
            if suggestions.is_empty() {
                Ok(RuleResult::pass())
            } else {
                Ok(RuleResult::pass_with_suggestions(suggestions))
            }
        } else {
            Ok(RuleResult::fail(violations))
        }
    }

    fn can_fix(&self) -> bool {
        true
    }

    fn fix(&self, project_path: &Path) -> anyhow::Result<FixResult> {
        let makefile_path = project_path.join("Makefile");
        let mut fixed = 0;
        let mut details = Vec::new();

        // Read existing content or start fresh
        let mut content = if makefile_path.exists() {
            std::fs::read_to_string(&makefile_path)?
        } else {
            ".PHONY: test-fast test lint fmt coverage build\n\n".to_string()
        };

        // Parse current targets
        let existing_targets = if makefile_path.exists() {
            self.parse_makefile(&makefile_path)?
        } else {
            HashMap::new()
        };

        // Add missing targets
        for (target_name, spec) in &self.required_targets {
            if !existing_targets.contains_key(target_name) {
                let default_cmd = spec.pattern.as_deref().unwrap_or("@echo 'TODO'");
                content.push_str(&format!("\n{0}:\n\t{1}\n", target_name, default_cmd));
                fixed += 1;
                details.push(FixDetail::Fixed {
                    code: "MK-002".to_string(),
                    description: format!("Added target '{}'", target_name),
                });
            }
        }

        // Write updated content
        if fixed > 0 {
            std::fs::write(&makefile_path, content)?;
        }

        Ok(FixResult::success(fixed).with_detail(FixDetail::Fixed {
            code: "MK-000".to_string(),
            description: format!("Updated Makefile with {} targets", fixed),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_makefile_rule_creation() {
        let rule = MakefileRule::new();
        assert_eq!(rule.id(), "makefile-targets");
        assert!(rule.required_targets.contains_key("test-fast"));
        assert!(rule.required_targets.contains_key("coverage"));
    }

    #[test]
    fn test_missing_makefile() {
        let temp = TempDir::new().unwrap();
        let rule = MakefileRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(!result.passed);
        assert_eq!(result.violations[0].code, "MK-001");
    }

    #[test]
    fn test_complete_makefile() {
        let temp = TempDir::new().unwrap();
        let makefile = temp.path().join("Makefile");

        let content = r#"
.PHONY: test-fast test lint fmt coverage

test-fast:
	cargo nextest run --lib

test:
	cargo nextest run

lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt --check

coverage:
	cargo llvm-cov --html
"#;
        std::fs::write(&makefile, content).unwrap();

        let rule = MakefileRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(result.passed, "Should pass: {:?}", result.violations);
    }

    #[test]
    fn test_missing_target() {
        let temp = TempDir::new().unwrap();
        let makefile = temp.path().join("Makefile");

        let content = r#"
test:
	cargo test

lint:
	cargo clippy
"#;
        std::fs::write(&makefile, content).unwrap();

        let rule = MakefileRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(!result.passed);
        // Should have violations for test-fast, fmt, coverage
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_prohibited_command() {
        let temp = TempDir::new().unwrap();
        let makefile = temp.path().join("Makefile");

        let content = r#"
coverage:
	cargo tarpaulin --out Html
"#;
        std::fs::write(&makefile, content).unwrap();

        let rule = MakefileRule::new();
        let result = rule.check(temp.path()).unwrap();
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.code == "MK-003"));
    }

    #[test]
    fn test_fix_creates_makefile() {
        let temp = TempDir::new().unwrap();
        let rule = MakefileRule::new();

        // Verify no makefile exists
        assert!(!temp.path().join("Makefile").exists());

        let result = rule.fix(temp.path()).unwrap();
        assert!(result.success);
        assert!(temp.path().join("Makefile").exists());
    }

    #[test]
    fn test_can_fix_returns_true() {
        let rule = MakefileRule::new();
        assert!(rule.can_fix());
    }

    #[test]
    fn test_rule_metadata() {
        let rule = MakefileRule::new();
        assert_eq!(rule.id(), "makefile-targets");
        assert!(!rule.description().is_empty());
        assert_eq!(rule.category(), RuleCategory::Build);
    }

    #[test]
    fn test_fix_with_existing_makefile() {
        let temp = TempDir::new().unwrap();
        let makefile = temp.path().join("Makefile");

        // Create a minimal Makefile
        let content = "test:\n\tcargo test\n";
        std::fs::write(&makefile, content).unwrap();

        let rule = MakefileRule::new();
        let result = rule.fix(temp.path()).unwrap();

        // Should succeed and add missing targets
        assert!(result.success);
        let new_content = std::fs::read_to_string(&makefile).unwrap();
        assert!(new_content.contains("test-fast:"));
    }

    #[test]
    fn test_prohibited_command_in_non_required_target() {
        let temp = TempDir::new().unwrap();
        let makefile = temp.path().join("Makefile");

        let content = r#"
custom-coverage:
	cargo tarpaulin --out Html

test-fast:
	cargo nextest run --lib
"#;
        std::fs::write(&makefile, content).unwrap();

        let rule = MakefileRule::new();
        let result = rule.check(temp.path()).unwrap();
        // Should fail because of prohibited command in custom-coverage
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.code == "MK-003"));
    }

    #[test]
    fn test_target_without_expected_pattern() {
        let temp = TempDir::new().unwrap();
        let makefile = temp.path().join("Makefile");

        // lint target without clippy
        let content = r#"
lint:
	echo "linting"

test-fast:
	cargo nextest run --lib

test:
	cargo test

fmt:
	cargo fmt --check

coverage:
	cargo llvm-cov
"#;
        std::fs::write(&makefile, content).unwrap();

        let rule = MakefileRule::new();
        let result = rule.check(temp.path()).unwrap();
        // Should pass but have suggestions
        assert!(result.passed);
        assert!(!result.suggestions.is_empty());
    }

    #[test]
    fn test_default_trait() {
        let rule = MakefileRule::default();
        assert_eq!(rule.id(), "makefile-targets");
    }
}
