//! Stack Compliance Engine - Cross-project consistency enforcement
//!
//! Ensures consistent patterns across the Sovereign AI Stack:
//! - Makefile target consistency
//! - Cargo.toml parity
//! - CI workflow alignment
//! - Code duplication detection (MinHash+LSH)
//!
//! # Toyota Production System Principles
//!
//! - **Heijunka**: Standardized targets across all projects
//! - **Poka-Yoke**: Configuration validation prevents drift
//! - **Jidoka**: Stop-on-error for critical violations
//! - **Kaizen**: Continuous improvement via compliance tracking

// Allow unused for public API items not yet consumed
pub mod config;
pub mod report;
pub mod rule;
pub mod rules;

pub use config::ComplyConfig;
#[allow(unused_imports)]
pub use config::ProjectOverride;
pub use report::{ComplyReport, ComplyReportFormat};
pub use rule::StackComplianceRule;
#[allow(unused_imports)]
pub use rule::{FixResult, RuleResult};

use crate::stack::PAIML_CRATES;
use std::path::{Path, PathBuf};

/// Stack Compliance Engine
///
/// Orchestrates compliance checks across all PAIML stack projects.
#[derive(Debug)]
pub struct StackComplyEngine {
    /// Compliance configuration
    config: ComplyConfig,
    /// Registered compliance rules
    rules: Vec<Box<dyn StackComplianceRule>>,
    /// Project discovery cache
    discovered_projects: Vec<ProjectInfo>,
}

/// Information about a discovered project
#[derive(Debug, Clone)]
pub struct ProjectInfo {
    /// Project name (crate name)
    pub name: String,
    /// Path to project root
    pub path: PathBuf,
    /// Whether it's a PAIML stack crate
    pub is_paiml_crate: bool,
}

impl StackComplyEngine {
    /// Create a new compliance engine with default rules
    pub fn new(config: ComplyConfig) -> Self {
        let mut engine = Self { config, rules: Vec::new(), discovered_projects: Vec::new() };

        // Register default rules
        engine.register_rule(Box::new(rules::MakefileRule::new()));
        engine.register_rule(Box::new(rules::CargoTomlRule::new()));
        engine.register_rule(Box::new(rules::CiWorkflowRule::new()));
        engine.register_rule(Box::new(rules::DuplicationRule::new()));

        engine
    }

    /// Create engine with default configuration
    pub fn default_for_workspace(workspace: &Path) -> Self {
        Self::new(ComplyConfig::default_for_workspace(workspace))
    }

    /// Register a custom compliance rule
    pub fn register_rule(&mut self, rule: Box<dyn StackComplianceRule>) {
        self.rules.push(rule);
    }

    /// Discover projects in the workspace
    pub fn discover_projects(&mut self, workspace: &Path) -> anyhow::Result<&[ProjectInfo]> {
        self.discovered_projects.clear();

        // Walk workspace looking for Cargo.toml files
        for entry in
            walkdir::WalkDir::new(workspace).max_depth(2).into_iter().filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.file_name() == Some(std::ffi::OsStr::new("Cargo.toml")) {
                if let Some(project) = self.parse_project(path)? {
                    self.discovered_projects.push(project);
                }
            }
        }

        Ok(&self.discovered_projects)
    }

    /// Parse a project from its Cargo.toml
    fn parse_project(&self, cargo_toml: &Path) -> anyhow::Result<Option<ProjectInfo>> {
        let content = std::fs::read_to_string(cargo_toml)?;
        let toml: toml::Value = toml::from_str(&content)?;

        let name = toml
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .map(String::from);

        match name {
            Some(name) => {
                let path = cargo_toml.parent().unwrap_or(Path::new(".")).to_path_buf();
                let is_paiml_crate = PAIML_CRATES.contains(&name.as_str());

                Ok(Some(ProjectInfo { name, path, is_paiml_crate }))
            }
            None => Ok(None),
        }
    }

    /// Run all compliance checks
    pub fn check_all(&self) -> ComplyReport {
        let mut report = ComplyReport::new();

        for project in &self.discovered_projects {
            // Skip non-PAIML crates unless explicitly included
            if !project.is_paiml_crate && !self.config.include_external {
                continue;
            }

            for rule in &self.rules {
                // Check if rule is enabled
                if !self.is_rule_enabled(rule.id()) {
                    continue;
                }

                // Check for project-specific override
                if self.has_rule_exemption(&project.name, rule.id()) {
                    report.add_exemption(&project.name, rule.id());
                    continue;
                }

                match rule.check(&project.path) {
                    Ok(result) => {
                        report.add_result(&project.name, rule.id(), result);
                    }
                    Err(e) => {
                        report.add_error(&project.name, rule.id(), e.to_string());
                    }
                }
            }
        }

        report.finalize();
        report
    }

    /// Run a specific rule
    pub fn check_rule(&self, rule_id: &str) -> ComplyReport {
        let mut report = ComplyReport::new();

        let rule = match self.rules.iter().find(|r| r.id() == rule_id) {
            Some(r) => r,
            None => {
                report.add_global_error(format!("Unknown rule: {}", rule_id));
                return report;
            }
        };

        for project in &self.discovered_projects {
            if !project.is_paiml_crate && !self.config.include_external {
                continue;
            }

            if self.has_rule_exemption(&project.name, rule_id) {
                report.add_exemption(&project.name, rule_id);
                continue;
            }

            match rule.check(&project.path) {
                Ok(result) => {
                    report.add_result(&project.name, rule_id, result);
                }
                Err(e) => {
                    report.add_error(&project.name, rule_id, e.to_string());
                }
            }
        }

        report.finalize();
        report
    }

    /// Attempt to fix violations
    pub fn fix_all(&self, dry_run: bool) -> ComplyReport {
        let mut report = ComplyReport::new();

        for project in &self.discovered_projects {
            if !project.is_paiml_crate && !self.config.include_external {
                continue;
            }

            for rule in &self.rules {
                if !self.is_rule_enabled(rule.id()) || !rule.can_fix() {
                    continue;
                }

                if self.has_rule_exemption(&project.name, rule.id()) {
                    continue;
                }

                // First check if there are violations to fix
                let check_result = match rule.check(&project.path) {
                    Ok(r) => r,
                    Err(e) => {
                        report.add_error(&project.name, rule.id(), e.to_string());
                        continue;
                    }
                };

                if check_result.passed {
                    report.add_result(&project.name, rule.id(), check_result);
                    continue;
                }

                // Attempt fix
                if dry_run {
                    report.add_dry_run_fix(&project.name, rule.id(), &check_result.violations);
                } else {
                    match rule.fix(&project.path) {
                        Ok(fix_result) => {
                            report.add_fix_result(&project.name, rule.id(), fix_result);
                        }
                        Err(e) => {
                            report.add_error(&project.name, rule.id(), e.to_string());
                        }
                    }
                }
            }
        }

        report.finalize();
        report
    }

    /// Check if a rule is enabled
    fn is_rule_enabled(&self, rule_id: &str) -> bool {
        if self.config.enabled_rules.is_empty() {
            // All rules enabled by default
            !self.config.disabled_rules.contains(&rule_id.to_string())
        } else {
            self.config.enabled_rules.contains(&rule_id.to_string())
        }
    }

    /// Check if a project has an exemption for a rule
    fn has_rule_exemption(&self, project_name: &str, rule_id: &str) -> bool {
        self.config
            .project_overrides
            .get(project_name)
            .map(|o| o.exempt_rules.contains(&rule_id.to_string()))
            .unwrap_or(false)
    }

    /// Get list of available rules
    pub fn available_rules(&self) -> Vec<(&str, &str)> {
        self.rules.iter().map(|r| (r.id(), r.description())).collect()
    }

    /// Get list of discovered projects
    pub fn projects(&self) -> &[ProjectInfo] {
        &self.discovered_projects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comply_engine_creation() {
        let config = ComplyConfig::default();
        let engine = StackComplyEngine::new(config);
        assert!(!engine.rules.is_empty());
    }

    #[test]
    fn test_available_rules() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let rules = engine.available_rules();
        assert!(rules.iter().any(|(id, _)| *id == "makefile-targets"));
        assert!(rules.iter().any(|(id, _)| *id == "cargo-toml-consistency"));
        assert!(rules.iter().any(|(id, _)| *id == "ci-workflow-parity"));
        assert!(rules.iter().any(|(id, _)| *id == "code-duplication"));
    }

    #[test]
    fn test_rule_enabled_check() {
        let mut config = ComplyConfig::default();
        config.disabled_rules.push("makefile-targets".to_string());
        let engine = StackComplyEngine::new(config);
        assert!(!engine.is_rule_enabled("makefile-targets"));
        assert!(engine.is_rule_enabled("cargo-toml-consistency"));
    }

    #[test]
    fn test_project_exemption() {
        let mut config = ComplyConfig::default();
        let mut override_config = ProjectOverride::default();
        override_config.exempt_rules.push("makefile-targets".to_string());
        config.project_overrides.insert("test-project".to_string(), override_config);

        let engine = StackComplyEngine::new(config);
        assert!(engine.has_rule_exemption("test-project", "makefile-targets"));
        assert!(!engine.has_rule_exemption("test-project", "cargo-toml-consistency"));
        assert!(!engine.has_rule_exemption("other-project", "makefile-targets"));
    }

    #[test]
    fn test_default_for_workspace() {
        let engine = StackComplyEngine::default_for_workspace(Path::new("."));
        assert!(!engine.rules.is_empty());
    }

    #[test]
    fn test_projects_empty_initially() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        assert!(engine.projects().is_empty());
    }

    #[test]
    fn test_enabled_rules_explicit() {
        let mut config = ComplyConfig::default();
        config.enabled_rules.push("makefile-targets".to_string());
        let engine = StackComplyEngine::new(config);
        assert!(engine.is_rule_enabled("makefile-targets"));
        assert!(!engine.is_rule_enabled("cargo-toml-consistency"));
    }

    #[test]
    fn test_project_info_fields() {
        let info = ProjectInfo {
            name: "test-project".to_string(),
            path: PathBuf::from("/path/to/project"),
            is_paiml_crate: true,
        };
        assert_eq!(info.name, "test-project");
        assert_eq!(info.path, PathBuf::from("/path/to/project"));
        assert!(info.is_paiml_crate);
    }

    #[test]
    fn test_check_rule_unknown() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let report = engine.check_rule("nonexistent-rule");
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn test_check_all_empty_projects() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let report = engine.check_all();
        // With no discovered projects, should return empty report
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_fix_all_empty_projects() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let report = engine.fix_all(true);
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_fix_all_dry_run() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let report = engine.fix_all(true); // dry_run = true
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_fix_all_actual_run() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let report = engine.fix_all(false); // dry_run = false
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_register_custom_rule() {
        use crate::comply::rules::MakefileRule;
        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        let initial_count = engine.rules.len();
        engine.register_rule(Box::new(MakefileRule::new()));
        assert_eq!(engine.rules.len(), initial_count + 1);
    }

    #[test]
    fn test_project_info_clone() {
        let info = ProjectInfo {
            name: "test-project".to_string(),
            path: PathBuf::from("/path/to/project"),
            is_paiml_crate: true,
        };
        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.path, info.path);
        assert_eq!(cloned.is_paiml_crate, info.is_paiml_crate);
    }

    #[test]
    fn test_project_info_non_paiml() {
        let info = ProjectInfo {
            name: "third-party-lib".to_string(),
            path: PathBuf::from("/external/lib"),
            is_paiml_crate: false,
        };
        assert!(!info.is_paiml_crate);
    }

    #[test]
    fn test_discover_projects_in_tempdir() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("my-project");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");

        // Create a minimal Cargo.toml
        let cargo_toml = r#"
[package]
name = "my-project"
version = "0.1.0"
"#;
        std::fs::write(project_dir.join("Cargo.toml"), cargo_toml).expect("fs write failed");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        let projects = engine.discover_projects(tempdir.path()).expect("unexpected failure");

        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].name, "my-project");
        assert!(!projects[0].is_paiml_crate);
    }

    #[test]
    fn test_discover_projects_paiml_crate() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");

        // Create Cargo.toml with PAIML crate name
        let cargo_toml = r#"
[package]
name = "trueno"
version = "0.1.0"
"#;
        std::fs::write(project_dir.join("Cargo.toml"), cargo_toml).expect("fs write failed");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        let projects = engine.discover_projects(tempdir.path()).expect("unexpected failure");

        assert_eq!(projects.len(), 1);
        assert!(projects[0].is_paiml_crate);
    }

    #[test]
    fn test_discover_projects_multiple() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");

        // Create two projects
        for name in &["proj-a", "proj-b"] {
            let project_dir = tempdir.path().join(name);
            std::fs::create_dir_all(&project_dir).expect("mkdir failed");
            let cargo_toml = format!(
                r#"
[package]
name = "{}"
version = "0.1.0"
"#,
                name
            );
            std::fs::write(project_dir.join("Cargo.toml"), cargo_toml).expect("fs write failed");
        }

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        let projects = engine.discover_projects(tempdir.path()).expect("unexpected failure");

        assert_eq!(projects.len(), 2);
    }

    #[test]
    fn test_discover_projects_no_name() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("unnamed");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");

        // Cargo.toml without name
        let cargo_toml = r#"
[package]
version = "0.1.0"
"#;
        std::fs::write(project_dir.join("Cargo.toml"), cargo_toml).expect("fs write failed");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        let projects = engine.discover_projects(tempdir.path()).expect("unexpected failure");

        assert_eq!(projects.len(), 0);
    }

    #[test]
    fn test_discover_projects_clears_cache() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("proj");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"proj\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());

        // First discovery
        engine.discover_projects(tempdir.path()).expect("unexpected failure");
        assert_eq!(engine.projects().len(), 1);

        // Second discovery on empty dir should clear
        let empty_dir = tempfile::tempdir().expect("tempdir creation failed");
        engine.discover_projects(empty_dir.path()).expect("unexpected failure");
        assert_eq!(engine.projects().len(), 0);
    }

    #[test]
    fn test_check_all_with_include_external() {
        let config = ComplyConfig { include_external: true, ..Default::default() };

        let engine = StackComplyEngine::new(config);
        let report = engine.check_all();
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_check_rule_with_discovered_projects() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_rule("makefile-targets");
        // Report should have processed the project
        assert!(!report.results.is_empty() || !report.errors.is_empty());
    }

    #[test]
    fn test_available_rules_descriptions() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let rules = engine.available_rules();

        for (id, desc) in &rules {
            assert!(!id.is_empty());
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn test_engine_debug() {
        let engine = StackComplyEngine::new(ComplyConfig::default());
        let debug_str = format!("{:?}", engine);
        assert!(debug_str.contains("StackComplyEngine"));
    }

    #[test]
    fn test_project_info_debug() {
        let info = ProjectInfo {
            name: "test".to_string(),
            path: PathBuf::from("/test"),
            is_paiml_crate: false,
        };
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("ProjectInfo"));
    }

    // =========================================================================
    // Coverage: check_all with discovered PAIML projects
    // =========================================================================

    #[test]
    fn test_check_all_with_paiml_project_no_makefile() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_all();
        // trueno is a PAIML crate, so it should be checked
        assert_eq!(report.summary.total_projects, 1);
        assert!(report.summary.total_checks > 0);
    }

    #[test]
    fn test_check_all_skips_non_paiml_without_include_external() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("some-lib");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"some-lib\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let config = ComplyConfig::default(); // include_external is false
        let mut engine = StackComplyEngine::new(config);
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_all();
        // Non-PAIML crate should be skipped
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_check_all_includes_non_paiml_with_include_external() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("external-lib");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"external-lib\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let config = ComplyConfig { include_external: true, ..Default::default() };
        let mut engine = StackComplyEngine::new(config);
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_all();
        // External crate should now be included
        assert_eq!(report.summary.total_projects, 1);
    }

    #[test]
    fn test_check_all_with_disabled_rule() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut config = ComplyConfig::default();
        config.disabled_rules.push("makefile-targets".to_string());
        let mut engine = StackComplyEngine::new(config);
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_all();
        // makefile-targets rule should be skipped
        let trueno_results = report.results.get("trueno");
        if let Some(results) = trueno_results {
            assert!(!results.contains_key("makefile-targets"));
        }
    }

    #[test]
    fn test_check_all_with_exemption() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut config = ComplyConfig::default();
        let mut override_cfg = ProjectOverride::default();
        override_cfg.exempt_rules.push("makefile-targets".to_string());
        config.project_overrides.insert("trueno".to_string(), override_cfg);

        let mut engine = StackComplyEngine::new(config);
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_all();
        // Should have exemption recorded
        assert!(report
            .exemptions
            .iter()
            .any(|e| e.project == "trueno" && e.rule == "makefile-targets"));
    }

    // =========================================================================
    // Coverage: fix_all with discovered projects
    // =========================================================================

    #[test]
    fn test_fix_all_dry_run_with_paiml_project() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");
        // No Makefile -> makefile-targets rule will have violations

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.fix_all(true); // dry_run = true
                                           // Should have processed the project
        assert_eq!(report.summary.total_projects, 1);
    }

    #[test]
    fn test_fix_all_actual_with_paiml_project() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.fix_all(false); // actual fix
        assert_eq!(report.summary.total_projects, 1);
    }

    #[test]
    fn test_fix_all_skips_non_paiml() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("external");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"external\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.fix_all(false);
        // Non-PAIML skipped
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_fix_all_skips_disabled_rules() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut config = ComplyConfig::default();
        // Disable all rules
        config.disabled_rules.push("makefile-targets".to_string());
        config.disabled_rules.push("cargo-toml-consistency".to_string());
        config.disabled_rules.push("ci-workflow-parity".to_string());
        config.disabled_rules.push("code-duplication".to_string());
        let mut engine = StackComplyEngine::new(config);
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.fix_all(false);
        // All rules disabled -> no checks run
        assert_eq!(report.summary.total_checks, 0);
    }

    #[test]
    fn test_fix_all_with_exemption() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut config = ComplyConfig::default();
        let mut override_cfg = ProjectOverride::default();
        override_cfg.exempt_rules.push("makefile-targets".to_string());
        config.project_overrides.insert("trueno".to_string(), override_cfg);

        let mut engine = StackComplyEngine::new(config);
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.fix_all(false);
        // fix_all only processes rules where can_fix() is true.
        // Only makefile-targets has can_fix()=true, and it's exempt here.
        // So zero fixable rules run -> total_projects is 0 in the report.
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_fix_all_passing_project_with_makefile() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        // Create a Makefile with required targets so makefile-targets passes
        let makefile = r#"test-fast:
	cargo nextest run --lib

test:
	cargo nextest run

lint:
	cargo clippy

fmt:
	cargo fmt

coverage:
	cargo llvm-cov
"#;
        std::fs::write(project_dir.join("Makefile"), makefile).expect("fs write failed");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.fix_all(false);
        // makefile-targets should pass (nothing to fix for that rule)
        assert_eq!(report.summary.total_projects, 1);
    }

    // =========================================================================
    // Coverage: check_rule with discovered projects
    // =========================================================================

    #[test]
    fn test_check_rule_skips_non_paiml() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("external");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"external\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_rule("makefile-targets");
        // Non-PAIML should be skipped
        assert_eq!(report.summary.total_projects, 0);
    }

    #[test]
    fn test_check_rule_with_exemption() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut config = ComplyConfig::default();
        let mut override_cfg = ProjectOverride::default();
        override_cfg.exempt_rules.push("makefile-targets".to_string());
        config.project_overrides.insert("trueno".to_string(), override_cfg);

        let mut engine = StackComplyEngine::new(config);
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_rule("makefile-targets");
        assert!(report
            .exemptions
            .iter()
            .any(|e| e.project == "trueno" && e.rule == "makefile-targets"));
    }

    #[test]
    fn test_check_rule_includes_non_paiml_with_external() {
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("ext-proj");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"ext-proj\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let config = ComplyConfig { include_external: true, ..Default::default() };
        let mut engine = StackComplyEngine::new(config);
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_rule("makefile-targets");
        // With include_external, non-PAIML projects should be checked
        assert!(report.summary.total_projects > 0 || !report.results.is_empty());
    }

    // =========================================================================
    // Coverage: parse_project edge cases
    // =========================================================================

    #[test]
    fn test_parse_project_workspace_toml() {
        // A workspace Cargo.toml has no [package] section
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("workspace");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        let cargo_toml = r#"
[workspace]
members = ["crate-a", "crate-b"]
"#;
        std::fs::write(project_dir.join("Cargo.toml"), cargo_toml).expect("fs write failed");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        let projects = engine.discover_projects(tempdir.path()).expect("unexpected failure");
        // Workspace toml has no package name, should be skipped
        assert_eq!(projects.len(), 0);
    }

    #[test]
    fn test_check_all_reports_rule_errors() {
        // Use a PAIML crate dir that causes rule check to produce an error
        // (e.g., invalid Makefile or missing directories)
        let tempdir = tempfile::tempdir().expect("tempdir creation failed");
        let project_dir = tempdir.path().join("trueno");
        std::fs::create_dir_all(&project_dir).expect("mkdir failed");
        std::fs::write(
            project_dir.join("Cargo.toml"),
            "[package]\nname = \"trueno\"\nversion = \"0.1.0\"",
        )
        .expect("unexpected failure");

        let mut engine = StackComplyEngine::new(ComplyConfig::default());
        engine.discover_projects(tempdir.path()).expect("unexpected failure");

        let report = engine.check_all();
        // Some rules may fail (no Makefile, no CI, etc.) but should produce results
        assert_eq!(report.summary.total_projects, 1);
        assert!(report.summary.total_checks > 0);
    }
}
