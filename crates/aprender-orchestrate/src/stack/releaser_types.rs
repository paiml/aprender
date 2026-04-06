//! Release Orchestrator Types
//!
//! Core types for release orchestration extracted from releaser.rs.
//! Includes BumpType, ReleaseConfig, ReleaseResult, ReleasedCrate.

use crate::stack::types::*;

/// Bump type for version updates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BumpType {
    /// Increment patch version (0.0.x)
    Patch,
    /// Increment minor version (0.x.0)
    Minor,
    /// Increment major version (x.0.0)
    Major,
}

impl BumpType {
    /// Apply bump to a version
    pub fn apply(&self, version: &semver::Version) -> semver::Version {
        match self {
            BumpType::Patch => {
                semver::Version::new(version.major, version.minor, version.patch + 1)
            }
            BumpType::Minor => semver::Version::new(version.major, version.minor + 1, 0),
            BumpType::Major => semver::Version::new(version.major + 1, 0, 0),
        }
    }
}

/// Configuration for release orchestration
#[derive(Debug, Clone)]
pub struct ReleaseConfig {
    /// Version bump type
    pub bump_type: Option<BumpType>,

    /// Whether to skip quality gates
    pub no_verify: bool,

    /// Whether this is a dry run
    pub dry_run: bool,

    /// Whether to actually publish to crates.io
    pub publish: bool,

    /// Minimum coverage percentage required
    pub min_coverage: f64,

    /// Lint command to run
    pub lint_command: String,

    /// Coverage command to run
    pub coverage_command: String,

    /// PMAT comply command for defect detection (CB-XXX violations)
    pub comply_command: String,

    /// Whether to fail on PMAT comply violations
    pub fail_on_comply_violations: bool,

    // =========================================================================
    // PMAT Quality Gate Integration (PMAT-STACK-GATES)
    // =========================================================================
    /// PMAT quality-gate command for comprehensive checks
    pub quality_gate_command: String,

    /// Whether to fail on quality gate violations
    pub fail_on_quality_gate: bool,

    /// PMAT TDG (Technical Debt Grading) command
    pub tdg_command: String,

    /// Minimum TDG score required (0-100)
    pub min_tdg_score: f64,

    /// Whether to fail on TDG score below threshold
    pub fail_on_tdg: bool,

    /// PMAT dead-code analysis command
    pub dead_code_command: String,

    /// Whether to fail on dead code detection
    pub fail_on_dead_code: bool,

    /// PMAT complexity analysis command
    pub complexity_command: String,

    /// Maximum cyclomatic complexity allowed
    pub max_complexity: u32,

    /// Whether to fail on complexity violations
    pub fail_on_complexity: bool,

    /// PMAT SATD (Self-Admitted Technical Debt) command
    pub satd_command: String,

    /// Maximum SATD items allowed
    pub max_satd_items: u32,

    /// Whether to fail on SATD violations
    pub fail_on_satd: bool,

    /// PMAT Popper score command (falsifiability)
    pub popper_command: String,

    /// Minimum Popper score required (0-100)
    pub min_popper_score: f64,

    /// Whether to fail on Popper score below threshold
    pub fail_on_popper: bool,

    // =========================================================================
    // Book and Examples Verification (RELEASE-DOCS)
    // =========================================================================
    /// Book build command (e.g., "mdbook build book")
    pub book_command: String,

    /// Whether to fail on book build errors
    pub fail_on_book: bool,

    /// Examples verification command pattern (e.g., "cargo run --example")
    /// Will run for each example found in the project
    pub examples_command: String,

    /// Whether to fail on example execution errors
    pub fail_on_examples: bool,
}

impl Default for ReleaseConfig {
    fn default() -> Self {
        Self {
            bump_type: None,
            no_verify: false,
            dry_run: false,
            publish: false,
            min_coverage: 90.0,
            lint_command: "make lint".to_string(),
            coverage_command: "make coverage".to_string(),
            comply_command: "pmat comply".to_string(),
            fail_on_comply_violations: true,
            // PMAT Quality Gate Integration defaults
            quality_gate_command: "pmat quality-gate".to_string(),
            fail_on_quality_gate: true,
            tdg_command: "pmat tdg --format json".to_string(),
            min_tdg_score: 80.0,
            fail_on_tdg: true,
            dead_code_command: "pmat analyze dead-code --format json".to_string(),
            fail_on_dead_code: false, // Warning only by default
            complexity_command: "pmat analyze complexity --format json".to_string(),
            max_complexity: 20,
            fail_on_complexity: true,
            satd_command: "pmat analyze satd --format json".to_string(),
            max_satd_items: 10,
            fail_on_satd: false, // Warning only by default
            popper_command: "pmat popper-score --format json".to_string(),
            min_popper_score: 60.0,
            fail_on_popper: true,
            // Book and Examples Verification defaults
            book_command: "mdbook build book".to_string(),
            fail_on_book: true,
            examples_command: "cargo run --example".to_string(),
            fail_on_examples: true,
        }
    }
}

/// Result of a release execution
#[derive(Debug, Clone)]
pub struct ReleaseResult {
    /// Whether the release succeeded
    pub success: bool,

    /// Crates that were released
    pub released_crates: Vec<ReleasedCrate>,

    /// Message describing the result
    pub message: String,
}

/// Information about a released crate
#[derive(Debug, Clone)]
pub struct ReleasedCrate {
    /// Crate name
    pub name: String,

    /// Released version
    pub version: semver::Version,

    /// Whether it was published to crates.io
    pub published: bool,
}

/// Format a release plan as text
pub fn format_plan_text(plan: &ReleasePlan) -> String {
    let mut output = String::new();

    if plan.dry_run {
        output.push_str("📋 Release Plan (DRY RUN)\n");
    } else {
        output.push_str("📋 Release Plan\n");
    }
    output.push_str(&"═".repeat(50));
    output.push_str("\n\n");

    output.push_str("Release order (topological):\n");
    for (i, release) in plan.releases.iter().enumerate() {
        output.push_str(&format!(
            "  {}. {} {} → {}\n",
            i + 1,
            release.crate_name,
            release.current_version,
            release.new_version
        ));
    }

    output.push_str("\nPre-flight status:\n");
    for release in &plan.releases {
        let status = plan
            .preflight_results
            .get(&release.crate_name)
            .map(|r| if r.passed { "✓" } else { "✗" })
            .unwrap_or("?");

        output.push_str(&format!("  {} {}\n", status, release.crate_name));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // BumpType Tests
    // ============================================================================

    #[test]
    fn test_bump_type_patch() {
        let version = semver::Version::new(1, 2, 3);
        let bumped = BumpType::Patch.apply(&version);
        assert_eq!(bumped, semver::Version::new(1, 2, 4));
    }

    #[test]
    fn test_bump_type_minor() {
        let version = semver::Version::new(1, 2, 3);
        let bumped = BumpType::Minor.apply(&version);
        assert_eq!(bumped, semver::Version::new(1, 3, 0));
    }

    #[test]
    fn test_bump_type_major() {
        let version = semver::Version::new(1, 2, 3);
        let bumped = BumpType::Major.apply(&version);
        assert_eq!(bumped, semver::Version::new(2, 0, 0));
    }

    #[test]
    fn test_bump_type_clone() {
        let bump = BumpType::Minor;
        let cloned = bump;
        assert_eq!(bump, cloned);
    }

    // ============================================================================
    // ReleaseConfig Tests
    // ============================================================================

    #[test]
    fn test_release_config_default() {
        let config = ReleaseConfig::default();
        assert!(config.bump_type.is_none());
        assert!(!config.no_verify);
        assert!(!config.dry_run);
        assert!(!config.publish);
        assert_eq!(config.min_coverage, 90.0);
    }

    #[test]
    fn test_release_config_pmat_defaults() {
        let config = ReleaseConfig::default();
        assert_eq!(config.quality_gate_command, "pmat quality-gate");
        assert!(config.fail_on_quality_gate);
        assert_eq!(config.min_tdg_score, 80.0);
        assert!(config.fail_on_tdg);
        assert!(!config.fail_on_dead_code); // Warning only by default
        assert_eq!(config.max_complexity, 20);
        assert!(config.fail_on_complexity);
        assert!(!config.fail_on_satd); // Warning only by default
        assert_eq!(config.min_popper_score, 60.0);
        assert!(config.fail_on_popper);
    }

    #[test]
    fn test_release_config_book_defaults() {
        let config = ReleaseConfig::default();
        assert_eq!(config.book_command, "mdbook build book");
        assert!(config.fail_on_book);
        assert_eq!(config.examples_command, "cargo run --example");
        assert!(config.fail_on_examples);
    }

    #[test]
    fn test_release_config_custom_thresholds() {
        let config = ReleaseConfig {
            min_tdg_score: 90.0,
            min_popper_score: 70.0,
            max_complexity: 15,
            max_satd_items: 5,
            fail_on_dead_code: true,
            fail_on_satd: true,
            ..Default::default()
        };

        assert_eq!(config.min_tdg_score, 90.0);
        assert_eq!(config.min_popper_score, 70.0);
        assert_eq!(config.max_complexity, 15);
        assert_eq!(config.max_satd_items, 5);
        assert!(config.fail_on_dead_code);
        assert!(config.fail_on_satd);
    }

    #[test]
    fn test_release_config_disabled_checks() {
        let config = ReleaseConfig {
            quality_gate_command: String::new(),
            tdg_command: String::new(),
            dead_code_command: String::new(),
            complexity_command: String::new(),
            satd_command: String::new(),
            popper_command: String::new(),
            ..Default::default()
        };

        assert!(config.quality_gate_command.is_empty());
        assert!(config.tdg_command.is_empty());
        assert!(config.dead_code_command.is_empty());
        assert!(config.complexity_command.is_empty());
        assert!(config.satd_command.is_empty());
        assert!(config.popper_command.is_empty());
    }

    // ============================================================================
    // ReleaseResult Tests
    // ============================================================================

    #[test]
    fn test_release_result_success() {
        let result = ReleaseResult {
            success: true,
            released_crates: vec![ReleasedCrate {
                name: "test".to_string(),
                version: semver::Version::new(1, 0, 0),
                published: true,
            }],
            message: "Success".to_string(),
        };
        assert!(result.success);
        assert_eq!(result.released_crates.len(), 1);
    }

    #[test]
    fn test_released_crate_clone() {
        let crate_info = ReleasedCrate {
            name: "test".to_string(),
            version: semver::Version::new(1, 0, 0),
            published: true,
        };
        let cloned = crate_info.clone();
        assert_eq!(crate_info.name, cloned.name);
        assert_eq!(crate_info.version, cloned.version);
        assert_eq!(crate_info.published, cloned.published);
    }
}
