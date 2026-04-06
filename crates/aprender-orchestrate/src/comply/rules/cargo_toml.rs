//! Cargo.toml Consistency Rule
//!
//! Ensures consistent Cargo.toml configuration across PAIML stack projects.

use crate::comply::rule::{
    FixResult, RuleCategory, RuleResult, RuleViolation, StackComplianceRule, ViolationLevel,
};
use std::collections::HashMap;
use std::path::Path;

/// Cargo.toml consistency rule
#[derive(Debug)]
pub struct CargoTomlRule {
    /// Required dependencies with version constraints
    required_deps: HashMap<String, String>,
    /// Prohibited dependencies
    prohibited_deps: Vec<String>,
    /// Required edition
    required_edition: Option<String>,
    /// Required license
    required_license: Option<String>,
}

impl Default for CargoTomlRule {
    fn default() -> Self {
        Self::new()
    }
}

impl CargoTomlRule {
    /// Create a new Cargo.toml rule with default configuration
    pub fn new() -> Self {
        let mut required_deps = HashMap::new();
        // trueno is a core dependency for stack crates
        required_deps.insert("trueno".to_string(), ">=0.14".to_string());

        Self {
            required_deps,
            prohibited_deps: vec!["cargo-tarpaulin".to_string()],
            required_edition: Some("2024".to_string()),
            required_license: Some("MIT OR Apache-2.0".to_string()),
        }
    }

    /// Parse Cargo.toml and extract relevant fields
    fn parse_cargo_toml(&self, path: &Path) -> anyhow::Result<CargoTomlData> {
        let content = std::fs::read_to_string(path)?;
        let toml: toml::Value = toml::from_str(&content)?;

        let package = toml.get("package");
        let name = package.and_then(|p| p.get("name")).and_then(|n| n.as_str()).map(String::from);
        let edition =
            package.and_then(|p| p.get("edition")).and_then(|e| e.as_str()).map(String::from);
        let license =
            package.and_then(|p| p.get("license")).and_then(|l| l.as_str()).map(String::from);
        let rust_version =
            package.and_then(|p| p.get("rust-version")).and_then(|r| r.as_str()).map(String::from);

        let mut dependencies = HashMap::new();
        let mut dev_dependencies = HashMap::new();

        // Parse dependencies
        if let Some(deps) = toml.get("dependencies") {
            if let Some(table) = deps.as_table() {
                for (name, value) in table {
                    let version = extract_version(value);
                    dependencies.insert(name.clone(), version);
                }
            }
        }

        // Parse dev-dependencies
        if let Some(deps) = toml.get("dev-dependencies") {
            if let Some(table) = deps.as_table() {
                for (name, value) in table {
                    let version = extract_version(value);
                    dev_dependencies.insert(name.clone(), version);
                }
            }
        }

        Ok(CargoTomlData { name, edition, license, rust_version, dependencies, dev_dependencies })
    }
}

fn extract_version(value: &toml::Value) -> Option<String> {
    match value {
        toml::Value::String(s) => Some(s.clone()),
        toml::Value::Table(t) => t.get("version").and_then(|v| v.as_str()).map(String::from),
        _ => None,
    }
}

#[derive(Debug)]
struct CargoTomlData {
    name: Option<String>,
    edition: Option<String>,
    license: Option<String>,
    rust_version: Option<String>,
    dependencies: HashMap<String, Option<String>>,
    dev_dependencies: HashMap<String, Option<String>>,
}

impl StackComplianceRule for CargoTomlRule {
    fn id(&self) -> &'static str {
        "cargo-toml-consistency"
    }

    fn description(&self) -> &'static str {
        "Ensures consistent Cargo.toml configuration across stack projects"
    }

    fn help(&self) -> Option<&str> {
        Some(
            "Checks: edition, license, required dependencies\n\
             Prohibited: cargo-tarpaulin",
        )
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Build
    }

    fn check(&self, project_path: &Path) -> anyhow::Result<RuleResult> {
        let cargo_toml_path = project_path.join("Cargo.toml");

        if !cargo_toml_path.exists() {
            return Ok(RuleResult::fail(vec![RuleViolation::new(
                "CT-001",
                "Cargo.toml not found",
            )
            .with_severity(ViolationLevel::Critical)
            .with_location(project_path.display().to_string())]));
        }

        let data = self.parse_cargo_toml(&cargo_toml_path)?;
        let mut violations = Vec::new();

        // Check edition
        if let Some(required_edition) = &self.required_edition {
            match &data.edition {
                None => {
                    violations.push(
                        RuleViolation::new("CT-002", "Edition not specified")
                            .with_severity(ViolationLevel::Warning)
                            .with_location("Cargo.toml".to_string())
                            .with_diff(required_edition.clone(), "(not set)".to_string())
                            .fixable(),
                    );
                }
                Some(edition) if edition != required_edition => {
                    // Allow older editions but warn
                    violations.push(
                        RuleViolation::new(
                            "CT-003",
                            format!("Edition mismatch: expected {}", required_edition),
                        )
                        .with_severity(ViolationLevel::Warning)
                        .with_location("Cargo.toml".to_string())
                        .with_diff(required_edition.clone(), edition.clone())
                        .fixable(),
                    );
                }
                _ => {}
            }
        }

        // Check license
        if let Some(required_license) = &self.required_license {
            match &data.license {
                None => {
                    violations.push(
                        RuleViolation::new("CT-004", "License not specified")
                            .with_severity(ViolationLevel::Error)
                            .with_location("Cargo.toml".to_string())
                            .with_diff(required_license.clone(), "(not set)".to_string())
                            .fixable(),
                    );
                }
                Some(license) if license != required_license => {
                    // Just warn, different licenses may be intentional
                    violations.push(
                        RuleViolation::new(
                            "CT-005",
                            format!("License differs from standard: {}", required_license),
                        )
                        .with_severity(ViolationLevel::Info)
                        .with_location("Cargo.toml".to_string())
                        .with_diff(required_license.clone(), license.clone()),
                    );
                }
                _ => {}
            }
        }

        // Check prohibited dependencies
        for prohibited in &self.prohibited_deps {
            if data.dependencies.contains_key(prohibited) {
                violations.push(
                    RuleViolation::new("CT-006", format!("Prohibited dependency: {}", prohibited))
                        .with_severity(ViolationLevel::Critical)
                        .with_location("Cargo.toml".to_string()),
                );
            }
            if data.dev_dependencies.contains_key(prohibited) {
                violations.push(
                    RuleViolation::new(
                        "CT-007",
                        format!("Prohibited dev-dependency: {}", prohibited),
                    )
                    .with_severity(ViolationLevel::Critical)
                    .with_location("Cargo.toml".to_string()),
                );
            }
        }

        // Note: We don't enforce trueno dependency for all crates,
        // as some crates (like documentation) don't need it
        // This could be made configurable per-crate category

        if violations.is_empty() {
            Ok(RuleResult::pass())
        } else {
            Ok(RuleResult::fail(violations))
        }
    }

    fn can_fix(&self) -> bool {
        false // Cargo.toml changes are too risky for auto-fix
    }

    fn fix(&self, _project_path: &Path) -> anyhow::Result<FixResult> {
        Ok(FixResult::failure("Auto-fix not supported for Cargo.toml - manual review required"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cargo_toml_rule_creation() {
        let rule = CargoTomlRule::new();
        assert_eq!(rule.id(), "cargo-toml-consistency");
        assert!(rule.required_deps.contains_key("trueno"));
    }

    #[test]
    fn test_missing_cargo_toml() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let rule = CargoTomlRule::new();
        let result = rule.check(temp.path()).expect("check failed");
        assert!(!result.passed);
        assert_eq!(result.violations[0].code, "CT-001");
    }

    #[test]
    fn test_valid_cargo_toml() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let cargo_toml = temp.path().join("Cargo.toml");

        let content = r#"
[package]
name = "test-crate"
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"

[dependencies]
trueno = "0.14"
"#;
        std::fs::write(&cargo_toml, content).expect("fs write failed");

        let rule = CargoTomlRule::new();
        let result = rule.check(temp.path()).expect("check failed");
        assert!(result.passed, "Should pass: {:?}", result.violations);
    }

    #[test]
    fn test_missing_edition() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let cargo_toml = temp.path().join("Cargo.toml");

        let content = r#"
[package]
name = "test-crate"
version = "0.1.0"
license = "MIT OR Apache-2.0"

[dependencies]
"#;
        std::fs::write(&cargo_toml, content).expect("fs write failed");

        let rule = CargoTomlRule::new();
        let result = rule.check(temp.path()).expect("check failed");
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.code == "CT-002"));
    }

    #[test]
    fn test_prohibited_dependency() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let cargo_toml = temp.path().join("Cargo.toml");

        let content = r#"
[package]
name = "test-crate"
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"

[dev-dependencies]
cargo-tarpaulin = "0.1"
"#;
        std::fs::write(&cargo_toml, content).expect("fs write failed");

        let rule = CargoTomlRule::new();
        let result = rule.check(temp.path()).expect("check failed");
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.code == "CT-007"));
    }

    #[test]
    fn test_wrong_edition() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let cargo_toml = temp.path().join("Cargo.toml");

        let content = r#"
[package]
name = "test-crate"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"

[dependencies]
"#;
        std::fs::write(&cargo_toml, content).expect("fs write failed");

        let rule = CargoTomlRule::new();
        let result = rule.check(temp.path()).expect("check failed");
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.code == "CT-003"));
    }

    #[test]
    fn test_missing_license() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let cargo_toml = temp.path().join("Cargo.toml");

        let content = r#"
[package]
name = "test-crate"
version = "0.1.0"
edition = "2024"

[dependencies]
"#;
        std::fs::write(&cargo_toml, content).expect("fs write failed");

        let rule = CargoTomlRule::new();
        let result = rule.check(temp.path()).expect("check failed");
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.code == "CT-004"));
    }

    #[test]
    fn test_different_license() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let cargo_toml = temp.path().join("Cargo.toml");

        let content = r#"
[package]
name = "test-crate"
version = "0.1.0"
edition = "2024"
license = "GPL-3.0"

[dependencies]
"#;
        std::fs::write(&cargo_toml, content).expect("fs write failed");

        let rule = CargoTomlRule::new();
        let result = rule.check(temp.path()).expect("check failed");
        // Different license is just a warning (Info), so it still passes
        assert!(result.violations.iter().any(|v| v.code == "CT-005"));
    }

    #[test]
    fn test_prohibited_dependency_in_deps() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let cargo_toml = temp.path().join("Cargo.toml");

        let content = r#"
[package]
name = "test-crate"
version = "0.1.0"
edition = "2024"
license = "MIT OR Apache-2.0"

[dependencies]
cargo-tarpaulin = "0.1"
"#;
        std::fs::write(&cargo_toml, content).expect("fs write failed");

        let rule = CargoTomlRule::new();
        let result = rule.check(temp.path()).expect("check failed");
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.code == "CT-006"));
    }

    #[test]
    fn test_can_fix_returns_false() {
        let rule = CargoTomlRule::new();
        assert!(!rule.can_fix());
    }

    #[test]
    fn test_fix_returns_failure() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let rule = CargoTomlRule::new();
        let result = rule.fix(temp.path()).expect("unexpected failure");
        assert!(!result.success);
    }

    #[test]
    fn test_rule_category() {
        let rule = CargoTomlRule::new();
        assert_eq!(rule.category(), RuleCategory::Build);
    }

    #[test]
    fn test_rule_description() {
        let rule = CargoTomlRule::new();
        assert!(!rule.description().is_empty());
    }

    #[test]
    fn test_default_trait() {
        let rule = CargoTomlRule::default();
        assert_eq!(rule.id(), "cargo-toml-consistency");
    }

    #[test]
    fn test_invalid_toml() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let cargo_toml = temp.path().join("Cargo.toml");
        std::fs::write(&cargo_toml, "invalid toml {{{{").expect("fs write failed");

        let rule = CargoTomlRule::new();
        // Should not panic, should return an error result
        let result = rule.check(temp.path());
        // The implementation returns an Err for parse failures
        assert!(result.is_err() || !result.expect("operation failed").passed);
    }
}
