//! Core types for PAIML Stack Orchestration
//!
//! These types represent the domain model for dependency management,
//! health checking, and coordinated releases.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Represents a crate in the PAIML stack
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrateInfo {
    /// Crate name (e.g., "trueno")
    pub name: String,

    /// Local version from Cargo.toml
    pub local_version: semver::Version,

    /// Version published on crates.io (None if not published)
    pub crates_io_version: Option<semver::Version>,

    /// Path to the crate's Cargo.toml
    pub manifest_path: PathBuf,

    /// Dependencies on other PAIML crates
    pub paiml_dependencies: Vec<DependencyInfo>,

    /// External dependencies (non-PAIML)
    pub external_dependencies: Vec<DependencyInfo>,

    /// Health status
    pub status: CrateStatus,

    /// List of issues found
    pub issues: Vec<CrateIssue>,
}

impl CrateInfo {
    /// Create a new CrateInfo with minimal data
    pub fn new(name: impl Into<String>, version: semver::Version, manifest_path: PathBuf) -> Self {
        Self {
            name: name.into(),
            local_version: version,
            crates_io_version: None,
            manifest_path,
            paiml_dependencies: Vec::new(),
            external_dependencies: Vec::new(),
            status: CrateStatus::Unknown,
            issues: Vec::new(),
        }
    }

    /// Check if crate has any path dependencies
    pub fn has_path_dependencies(&self) -> bool {
        self.paiml_dependencies.iter().any(|d| d.is_path)
            || self.external_dependencies.iter().any(|d| d.is_path)
    }

    /// Check if crate version is ahead of crates.io
    pub fn is_ahead_of_crates_io(&self) -> bool {
        match &self.crates_io_version {
            Some(remote) => self.local_version > *remote,
            None => true, // Not published yet
        }
    }

    /// Check if crate is in sync with crates.io
    pub fn is_synced(&self) -> bool {
        match &self.crates_io_version {
            Some(remote) => self.local_version == *remote,
            None => false,
        }
    }
}

/// Information about a dependency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DependencyInfo {
    /// Dependency name
    pub name: String,

    /// Version requirement (e.g., "^1.0", ">=0.5")
    pub version_req: String,

    /// Whether this is a path dependency
    pub is_path: bool,

    /// Path if path dependency
    pub path: Option<PathBuf>,

    /// Whether this is a PAIML crate
    pub is_paiml: bool,

    /// Dependency kind (normal, dev, build)
    pub kind: DependencyKind,
}

impl DependencyInfo {
    /// Create a new dependency info
    pub fn new(name: impl Into<String>, version_req: impl Into<String>) -> Self {
        let name = name.into();
        let is_paiml = super::is_paiml_crate(&name);
        Self {
            name,
            version_req: version_req.into(),
            is_path: false,
            path: None,
            is_paiml,
            kind: DependencyKind::Normal,
        }
    }

    /// Create a path dependency
    pub fn path(name: impl Into<String>, path: PathBuf) -> Self {
        let name = name.into();
        let is_paiml = super::is_paiml_crate(&name);
        Self {
            name,
            version_req: String::new(),
            is_path: true,
            path: Some(path),
            is_paiml,
            kind: DependencyKind::Normal,
        }
    }
}

/// Kind of dependency
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum DependencyKind {
    #[default]
    Normal,
    Dev,
    Build,
}

/// Health status of a crate
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum CrateStatus {
    /// Crate is healthy - all checks pass
    Healthy,

    /// Crate has warnings but is functional
    Warning,

    /// Crate has errors that block release
    Error,

    /// Status not yet determined
    #[default]
    Unknown,
}

impl std::fmt::Display for CrateStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrateStatus::Healthy => write!(f, "healthy"),
            CrateStatus::Warning => write!(f, "warning"),
            CrateStatus::Error => write!(f, "error"),
            CrateStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Issue found during health check
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrateIssue {
    /// Issue severity
    pub severity: IssueSeverity,

    /// Issue type
    pub issue_type: IssueType,

    /// Human-readable message
    pub message: String,

    /// Suggested fix
    pub suggestion: Option<String>,
}

impl CrateIssue {
    /// Create a new issue
    pub fn new(severity: IssueSeverity, issue_type: IssueType, message: impl Into<String>) -> Self {
        Self { severity, issue_type, message: message.into(), suggestion: None }
    }

    /// Add a suggestion for fixing the issue
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// Severity of an issue
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
}

/// Type of issue
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IssueType {
    /// Path dependency that should be crates.io
    PathDependency,

    /// Version conflict between crates
    VersionConflict,

    /// Crate not published to crates.io
    NotPublished,

    /// Local version behind crates.io
    VersionBehind,

    /// Circular dependency detected
    CircularDependency,

    /// Missing dependency
    MissingDependency,

    /// Quality gate failure (lint, coverage)
    QualityGate,

    /// Uncommitted changes in git
    UncommittedChanges,
}

impl std::fmt::Display for IssueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueType::PathDependency => write!(f, "path dependency"),
            IssueType::VersionConflict => write!(f, "version conflict"),
            IssueType::NotPublished => write!(f, "not published"),
            IssueType::VersionBehind => write!(f, "version behind"),
            IssueType::CircularDependency => write!(f, "circular dependency"),
            IssueType::MissingDependency => write!(f, "missing dependency"),
            IssueType::QualityGate => write!(f, "quality gate"),
            IssueType::UncommittedChanges => write!(f, "uncommitted changes"),
        }
    }
}

/// Result of a stack health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackHealthReport {
    /// Timestamp of the check
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// All crates analyzed
    pub crates: Vec<CrateInfo>,

    /// Version conflicts detected
    pub conflicts: Vec<VersionConflict>,

    /// Summary statistics
    pub summary: HealthSummary,
}

impl StackHealthReport {
    /// Create a new health report
    pub fn new(crates: Vec<CrateInfo>, conflicts: Vec<VersionConflict>) -> Self {
        let summary = HealthSummary::from_crates(&crates);
        Self { timestamp: chrono::Utc::now(), crates, conflicts, summary }
    }

    /// Check if the stack is healthy (no errors)
    pub fn is_healthy(&self) -> bool {
        self.summary.error_count == 0 && self.conflicts.is_empty()
    }
}

/// Version conflict between crates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VersionConflict {
    /// Name of the conflicting dependency
    pub dependency: String,

    /// Crates involved and their required versions
    pub usages: Vec<ConflictUsage>,

    /// Recommended version to align on
    pub recommendation: Option<String>,
}

/// Usage in a version conflict
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConflictUsage {
    /// Crate that has the dependency
    pub crate_name: String,

    /// Version required
    pub version_req: String,
}

/// Summary of stack health
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthSummary {
    /// Total number of crates
    pub total_crates: usize,

    /// Number of healthy crates
    pub healthy_count: usize,

    /// Number of crates with warnings
    pub warning_count: usize,

    /// Number of crates with errors
    pub error_count: usize,

    /// Number of path dependencies found
    pub path_dependency_count: usize,

    /// Number of version conflicts
    pub conflict_count: usize,
}

impl HealthSummary {
    /// Create summary from crate list
    pub fn from_crates(crates: &[CrateInfo]) -> Self {
        let mut summary = Self { total_crates: crates.len(), ..Default::default() };

        for crate_info in crates {
            match crate_info.status {
                CrateStatus::Healthy => summary.healthy_count += 1,
                CrateStatus::Warning => summary.warning_count += 1,
                CrateStatus::Error => summary.error_count += 1,
                CrateStatus::Unknown => {}
            }

            if crate_info.has_path_dependencies() {
                summary.path_dependency_count += 1;
            }
        }

        summary
    }
}

/// Release plan for coordinated release
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleasePlan {
    /// Releases in topological order
    pub releases: Vec<PlannedRelease>,

    /// Whether this is a dry run
    pub dry_run: bool,

    /// Pre-flight check results
    pub preflight_results: HashMap<String, PreflightResult>,
}

/// A planned release for a single crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedRelease {
    /// Crate name
    pub crate_name: String,

    /// Current version
    pub current_version: semver::Version,

    /// New version to release
    pub new_version: semver::Version,

    /// Crates that depend on this release
    pub dependents: Vec<String>,

    /// Whether release is ready (all checks passed)
    pub ready: bool,
}

/// Result of pre-flight checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflightResult {
    /// Crate name
    pub crate_name: String,

    /// Individual check results
    pub checks: Vec<PreflightCheck>,

    /// Overall pass/fail
    pub passed: bool,
}

impl PreflightResult {
    /// Create a new preflight result
    pub fn new(crate_name: impl Into<String>) -> Self {
        Self { crate_name: crate_name.into(), checks: Vec::new(), passed: true }
    }

    /// Add a check result
    pub fn add_check(&mut self, check: PreflightCheck) {
        if !check.passed {
            self.passed = false;
        }
        self.checks.push(check);
    }
}

/// Individual pre-flight check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflightCheck {
    /// Check name
    pub name: String,

    /// Whether check passed
    pub passed: bool,

    /// Details/message
    pub message: String,
}

impl PreflightCheck {
    /// Create a passing check
    pub fn pass(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self { name: name.into(), passed: true, message: message.into() }
    }

    /// Create a failing check
    pub fn fail(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self { name: name.into(), passed: false, message: message.into() }
    }
}

/// Output format for stack commands
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Markdown,
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_info_new() {
        let info = CrateInfo::new(
            "trueno",
            semver::Version::new(1, 2, 0),
            PathBuf::from("/path/to/Cargo.toml"),
        );

        assert_eq!(info.name, "trueno");
        assert_eq!(info.local_version, semver::Version::new(1, 2, 0));
        assert_eq!(info.status, CrateStatus::Unknown);
        assert!(info.issues.is_empty());
    }

    #[test]
    fn test_crate_info_path_dependencies() {
        let mut info =
            CrateInfo::new("entrenar", semver::Version::new(0, 2, 2), PathBuf::from("Cargo.toml"));

        assert!(!info.has_path_dependencies());

        info.paiml_dependencies
            .push(DependencyInfo::path("alimentar", PathBuf::from("../alimentar")));

        assert!(info.has_path_dependencies());
    }

    #[test]
    fn test_crate_info_version_comparison() {
        let mut info =
            CrateInfo::new("trueno", semver::Version::new(1, 2, 0), PathBuf::from("Cargo.toml"));

        // Not published yet
        assert!(info.is_ahead_of_crates_io());
        assert!(!info.is_synced());

        // Same version
        info.crates_io_version = Some(semver::Version::new(1, 2, 0));
        assert!(!info.is_ahead_of_crates_io());
        assert!(info.is_synced());

        // Local ahead
        info.local_version = semver::Version::new(1, 3, 0);
        assert!(info.is_ahead_of_crates_io());
        assert!(!info.is_synced());
    }

    #[test]
    fn test_dependency_info_paiml_detection() {
        let dep = DependencyInfo::new("trueno", "^1.0");
        assert!(dep.is_paiml);
        assert!(!dep.is_path);

        let dep = DependencyInfo::new("serde", "1.0");
        assert!(!dep.is_paiml);

        let dep = DependencyInfo::path("aprender", PathBuf::from("../aprender"));
        assert!(dep.is_paiml);
        assert!(dep.is_path);
    }

    #[test]
    fn test_crate_issue_creation() {
        let issue = CrateIssue::new(
            IssueSeverity::Error,
            IssueType::PathDependency,
            "alimentar uses path dependency",
        )
        .with_suggestion("Change to: alimentar = \"0.3.0\"");

        assert_eq!(issue.severity, IssueSeverity::Error);
        assert_eq!(issue.issue_type, IssueType::PathDependency);
        assert!(issue.suggestion.is_some());
    }

    #[test]
    fn test_health_summary_from_crates() {
        let crates = vec![
            {
                let mut c = CrateInfo::new("trueno", semver::Version::new(1, 0, 0), PathBuf::new());
                c.status = CrateStatus::Healthy;
                c
            },
            {
                let mut c =
                    CrateInfo::new("aprender", semver::Version::new(0, 8, 0), PathBuf::new());
                c.status = CrateStatus::Warning;
                c
            },
            {
                let mut c =
                    CrateInfo::new("entrenar", semver::Version::new(0, 2, 0), PathBuf::new());
                c.status = CrateStatus::Error;
                c.paiml_dependencies
                    .push(DependencyInfo::path("alimentar", PathBuf::from("../alimentar")));
                c
            },
        ];

        let summary = HealthSummary::from_crates(&crates);

        assert_eq!(summary.total_crates, 3);
        assert_eq!(summary.healthy_count, 1);
        assert_eq!(summary.warning_count, 1);
        assert_eq!(summary.error_count, 1);
        assert_eq!(summary.path_dependency_count, 1);
    }

    #[test]
    fn test_preflight_result() {
        let mut result = PreflightResult::new("trueno");
        assert!(result.passed);

        result.add_check(PreflightCheck::pass("lint", "No errors"));
        assert!(result.passed);

        result.add_check(PreflightCheck::fail("coverage", "Coverage 85% < 90%"));
        assert!(!result.passed);

        assert_eq!(result.checks.len(), 2);
    }

    #[test]
    fn test_stack_health_report() {
        let crates = vec![{
            let mut c = CrateInfo::new("trueno", semver::Version::new(1, 0, 0), PathBuf::new());
            c.status = CrateStatus::Healthy;
            c
        }];

        let report = StackHealthReport::new(crates, vec![]);

        assert!(report.is_healthy());
        assert_eq!(report.summary.total_crates, 1);
        assert_eq!(report.summary.healthy_count, 1);
    }

    #[test]
    fn test_version_conflict() {
        let conflict = VersionConflict {
            dependency: "arrow".to_string(),
            usages: vec![
                ConflictUsage {
                    crate_name: "renacer".to_string(),
                    version_req: "54.0".to_string(),
                },
                ConflictUsage {
                    crate_name: "trueno-graph".to_string(),
                    version_req: "53.0".to_string(),
                },
            ],
            recommendation: Some("Upgrade to arrow 54.0".to_string()),
        };

        assert_eq!(conflict.usages.len(), 2);
        assert!(conflict.recommendation.is_some());
    }

    // ============================================================================
    // TYPES-001: OutputFormat tests
    // ============================================================================

    /// RED PHASE: Test OutputFormat default
    #[test]
    fn test_TYPES_001_output_format_default() {
        let format = OutputFormat::default();
        assert_eq!(format, OutputFormat::Text);
    }

    /// RED PHASE: Test OutputFormat equality
    #[test]
    fn test_TYPES_001_output_format_equality() {
        assert_eq!(OutputFormat::Text, OutputFormat::Text);
        assert_eq!(OutputFormat::Json, OutputFormat::Json);
        assert_eq!(OutputFormat::Markdown, OutputFormat::Markdown);
        assert_ne!(OutputFormat::Text, OutputFormat::Json);
    }

    /// RED PHASE: Test OutputFormat debug
    #[test]
    fn test_TYPES_001_output_format_debug() {
        assert!(format!("{:?}", OutputFormat::Text).contains("Text"));
        assert!(format!("{:?}", OutputFormat::Json).contains("Json"));
        assert!(format!("{:?}", OutputFormat::Markdown).contains("Markdown"));
    }

    /// RED PHASE: Test OutputFormat clone
    #[test]
    fn test_TYPES_001_output_format_clone() {
        let format = OutputFormat::Json;
        let cloned = format;
        assert_eq!(format, cloned);
    }

    // ============================================================================
    // TYPES-002: PreflightCheck tests
    // ============================================================================

    /// RED PHASE: Test PreflightCheck::pass
    #[test]
    fn test_TYPES_002_preflight_check_pass() {
        let check = PreflightCheck::pass("test_check", "All good");

        assert!(check.passed);
        assert_eq!(check.name, "test_check");
        assert_eq!(check.message, "All good");
    }

    /// RED PHASE: Test PreflightCheck::fail
    #[test]
    fn test_TYPES_002_preflight_check_fail() {
        let check = PreflightCheck::fail("lint_check", "Found 5 errors");

        assert!(!check.passed);
        assert_eq!(check.name, "lint_check");
        assert_eq!(check.message, "Found 5 errors");
    }

    /// RED PHASE: Test PreflightCheck serialization
    #[test]
    fn test_TYPES_002_preflight_check_serialization() {
        let check = PreflightCheck::pass("git", "clean");
        let json = serde_json::to_string(&check).expect("json serialize failed");

        assert!(json.contains("git"));
        assert!(json.contains("clean"));
        assert!(json.contains("true"));
    }

    /// RED PHASE: Test PreflightResult serialization
    #[test]
    fn test_TYPES_002_preflight_result_serialization() {
        let mut result = PreflightResult::new("test-crate");
        result.add_check(PreflightCheck::pass("a", "ok"));

        let json = serde_json::to_string(&result).expect("json serialize failed");
        assert!(json.contains("test-crate"));
        assert!(json.contains("passed"));
    }

    // ============================================================================
    // TYPES-003: CrateStatus tests
    // ============================================================================

    /// RED PHASE: Test CrateStatus variants
    #[test]
    fn test_TYPES_003_crate_status_variants() {
        assert_eq!(CrateStatus::Unknown, CrateStatus::Unknown);
        assert_eq!(CrateStatus::Healthy, CrateStatus::Healthy);
        assert_eq!(CrateStatus::Warning, CrateStatus::Warning);
        assert_eq!(CrateStatus::Error, CrateStatus::Error);
        assert_ne!(CrateStatus::Healthy, CrateStatus::Error);
    }

    /// RED PHASE: Test CrateStatus debug
    #[test]
    fn test_TYPES_003_crate_status_debug() {
        assert!(format!("{:?}", CrateStatus::Unknown).contains("Unknown"));
        assert!(format!("{:?}", CrateStatus::Healthy).contains("Healthy"));
        assert!(format!("{:?}", CrateStatus::Warning).contains("Warning"));
        assert!(format!("{:?}", CrateStatus::Error).contains("Error"));
    }

    /// RED PHASE: Test CrateStatus default
    #[test]
    fn test_TYPES_003_crate_status_default() {
        let status = CrateStatus::default();
        assert_eq!(status, CrateStatus::Unknown);
    }

    // ============================================================================
    // TYPES-004: IssueSeverity and IssueType tests
    // ============================================================================

    /// RED PHASE: Test IssueSeverity variants
    #[test]
    fn test_TYPES_004_issue_severity_variants() {
        assert_eq!(IssueSeverity::Info, IssueSeverity::Info);
        assert_eq!(IssueSeverity::Warning, IssueSeverity::Warning);
        assert_eq!(IssueSeverity::Error, IssueSeverity::Error);
        assert_ne!(IssueSeverity::Info, IssueSeverity::Error);
    }

    /// RED PHASE: Test IssueType variants
    #[test]
    fn test_TYPES_004_issue_type_variants() {
        assert_eq!(IssueType::PathDependency, IssueType::PathDependency);
        assert_eq!(IssueType::VersionConflict, IssueType::VersionConflict);
        assert_eq!(IssueType::NotPublished, IssueType::NotPublished);
        assert_eq!(IssueType::VersionBehind, IssueType::VersionBehind);
        assert_eq!(IssueType::CircularDependency, IssueType::CircularDependency);
        assert_eq!(IssueType::MissingDependency, IssueType::MissingDependency);
        assert_eq!(IssueType::QualityGate, IssueType::QualityGate);
    }

    /// RED PHASE: Test CrateIssue without suggestion
    #[test]
    fn test_TYPES_004_crate_issue_no_suggestion() {
        let issue =
            CrateIssue::new(IssueSeverity::Info, IssueType::NotPublished, "Crate not on crates.io");

        assert_eq!(issue.severity, IssueSeverity::Info);
        assert!(issue.suggestion.is_none());
    }

    // ============================================================================
    // TYPES-005: CrateInfo edge cases
    // ============================================================================

    /// RED PHASE: Test CrateInfo clone
    #[test]
    fn test_TYPES_005_crate_info_clone() {
        let info =
            CrateInfo::new("test", semver::Version::new(1, 0, 0), PathBuf::from("Cargo.toml"));
        let cloned = info.clone();

        assert_eq!(info.name, cloned.name);
        assert_eq!(info.local_version, cloned.local_version);
    }

    /// RED PHASE: Test CrateInfo debug
    #[test]
    fn test_TYPES_005_crate_info_debug() {
        let info = CrateInfo::new(
            "debug-test",
            semver::Version::new(2, 0, 0),
            PathBuf::from("Cargo.toml"),
        );
        let debug = format!("{:?}", info);

        assert!(debug.contains("CrateInfo"));
        assert!(debug.contains("debug-test"));
    }

    /// RED PHASE: Test CrateInfo serialization
    #[test]
    fn test_TYPES_005_crate_info_serialization() {
        let info = CrateInfo::new(
            "serializable",
            semver::Version::new(1, 2, 3),
            PathBuf::from("path/Cargo.toml"),
        );
        let json = serde_json::to_string(&info).expect("json serialize failed");

        assert!(json.contains("serializable"));
        assert!(json.contains("1.2.3"));
    }

    // ============================================================================
    // TYPES-006: DependencyInfo edge cases
    // ============================================================================

    /// RED PHASE: Test DependencyInfo clone
    #[test]
    fn test_TYPES_006_dependency_info_clone() {
        let dep = DependencyInfo::new("trueno", "^1.0");
        let cloned = dep.clone();

        assert_eq!(dep.name, cloned.name);
        assert_eq!(dep.is_paiml, cloned.is_paiml);
    }

    /// RED PHASE: Test DependencyInfo debug
    #[test]
    fn test_TYPES_006_dependency_info_debug() {
        let dep = DependencyInfo::path("aprender", PathBuf::from("../aprender"));
        let debug = format!("{:?}", dep);

        assert!(debug.contains("DependencyInfo"));
        assert!(debug.contains("aprender"));
    }

    /// RED PHASE: Test non-PAIML dependency
    #[test]
    fn test_TYPES_006_non_paiml_dependency() {
        let dep = DependencyInfo::new("tokio", "1.0");

        assert!(!dep.is_paiml);
        assert!(!dep.is_path);
        assert_eq!(dep.version_req, "1.0");
    }

    // ============================================================================
    // TYPES-007: StackHealthReport edge cases
    // ============================================================================

    /// RED PHASE: Test StackHealthReport with warnings (still healthy)
    #[test]
    fn test_TYPES_007_health_report_with_warnings() {
        let crates = vec![{
            let mut c = CrateInfo::new("test", semver::Version::new(1, 0, 0), PathBuf::new());
            c.status = CrateStatus::Warning;
            c
        }];

        let report = StackHealthReport::new(crates, vec![]);

        // Warnings don't prevent healthy status (only errors and conflicts do)
        assert!(report.is_healthy());
        assert_eq!(report.summary.warning_count, 1);
    }

    /// RED PHASE: Test StackHealthReport with errors
    #[test]
    fn test_TYPES_007_health_report_with_errors() {
        let crates = vec![{
            let mut c = CrateInfo::new("test", semver::Version::new(1, 0, 0), PathBuf::new());
            c.status = CrateStatus::Error;
            c
        }];

        let report = StackHealthReport::new(crates, vec![]);

        // Errors make the report unhealthy
        assert!(!report.is_healthy());
        assert_eq!(report.summary.error_count, 1);
    }

    /// RED PHASE: Test StackHealthReport with conflicts
    #[test]
    fn test_TYPES_007_health_report_with_conflicts() {
        let crates = vec![{
            let mut c = CrateInfo::new("test", semver::Version::new(1, 0, 0), PathBuf::new());
            c.status = CrateStatus::Healthy;
            c
        }];

        let conflicts = vec![VersionConflict {
            dependency: "arrow".to_string(),
            usages: vec![],
            recommendation: None,
        }];

        let report = StackHealthReport::new(crates, conflicts);

        assert!(!report.conflicts.is_empty());
    }

    // ============================================================================
    // TYPES-008: PlannedRelease and ReleasePlan edge cases
    // ============================================================================

    /// RED PHASE: Test PlannedRelease not ready
    #[test]
    fn test_TYPES_008_planned_release_not_ready() {
        let release = PlannedRelease {
            crate_name: "broken".to_string(),
            current_version: semver::Version::new(1, 0, 0),
            new_version: semver::Version::new(1, 0, 1),
            dependents: vec!["downstream".to_string()],
            ready: false,
        };

        assert!(!release.ready);
        assert_eq!(release.dependents.len(), 1);
    }

    /// RED PHASE: Test ReleasePlan clone
    #[test]
    fn test_TYPES_008_release_plan_clone() {
        let plan = ReleasePlan {
            releases: vec![PlannedRelease {
                crate_name: "test".to_string(),
                current_version: semver::Version::new(0, 1, 0),
                new_version: semver::Version::new(0, 1, 1),
                dependents: vec![],
                ready: true,
            }],
            dry_run: true,
            preflight_results: std::collections::HashMap::new(),
        };

        let cloned = plan.clone();
        assert_eq!(plan.releases.len(), cloned.releases.len());
        assert_eq!(plan.dry_run, cloned.dry_run);
    }

    /// RED PHASE: Test HealthSummary serialization
    #[test]
    fn test_TYPES_008_health_summary_serialization() {
        let crates = vec![{
            let mut c = CrateInfo::new("test", semver::Version::new(1, 0, 0), PathBuf::new());
            c.status = CrateStatus::Healthy;
            c
        }];
        let summary = HealthSummary::from_crates(&crates);

        let json = serde_json::to_string(&summary).expect("json serialize failed");
        assert!(json.contains("total_crates"));
        assert!(json.contains("healthy_count"));
    }
}
