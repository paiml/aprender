//! Release Orchestrator
//!
//! Implements the `batuta stack release` command functionality.
//! Coordinates releases across multiple crates in topological order,
//! ensuring all quality gates pass before publishing.

use crate::stack::checker::StackChecker;
use crate::stack::types::*;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

// Re-export types from releaser_types module
pub use super::super::releaser_types::{
    format_plan_text, BumpType, ReleaseConfig, ReleaseResult, ReleasedCrate,
};

/// Release orchestrator for coordinated multi-crate releases
pub struct ReleaseOrchestrator {
    /// Release configuration (pub(super) for preflight module access)
    pub(in crate::stack) config: ReleaseConfig,

    /// Stack checker for health analysis
    checker: StackChecker,

    /// Pre-flight results per crate
    preflight_results: HashMap<String, PreflightResult>,
}

impl ReleaseOrchestrator {
    /// Create a new release orchestrator
    pub fn new(checker: StackChecker, config: ReleaseConfig) -> Self {
        Self { config, checker, preflight_results: HashMap::new() }
    }

    /// Create a release orchestrator from a workspace path
    #[cfg(feature = "native")]
    pub fn from_workspace(workspace_path: &Path, config: ReleaseConfig) -> Result<Self> {
        let checker = StackChecker::from_workspace(workspace_path)?;
        Ok(Self::new(checker, config))
    }

    /// Plan a release for a specific crate
    pub fn plan_release(&mut self, crate_name: &str) -> Result<ReleasePlan> {
        let release_order = self.checker.release_order_for(crate_name)?;

        let mut releases = Vec::new();

        for name in &release_order {
            let planned = self.plan_single_release(name)?;
            releases.push(planned);
        }

        Ok(ReleasePlan {
            releases,
            dry_run: self.config.dry_run,
            preflight_results: self.preflight_results.clone(),
        })
    }

    /// Plan a release for all crates with changes
    pub fn plan_all_releases(&mut self) -> Result<ReleasePlan> {
        let release_order = self.checker.topological_order()?;

        let mut releases = Vec::new();

        for name in &release_order {
            let planned = self.plan_single_release(name)?;
            releases.push(planned);
        }

        Ok(ReleasePlan {
            releases,
            dry_run: self.config.dry_run,
            preflight_results: self.preflight_results.clone(),
        })
    }

    /// Plan a single crate release
    fn plan_single_release(&self, crate_name: &str) -> Result<PlannedRelease> {
        let current_version = self
            .checker
            .get_crate(crate_name)
            .map(|c| c.local_version.clone())
            .unwrap_or_else(|| semver::Version::new(0, 0, 0));

        let new_version = match self.config.bump_type {
            Some(bump) => bump.apply(&current_version),
            None => semver::Version::new(
                current_version.major,
                current_version.minor,
                current_version.patch + 1,
            ),
        };

        let ready = self.preflight_results.get(crate_name).map(|r| r.passed).unwrap_or(true);

        Ok(PlannedRelease {
            crate_name: crate_name.to_string(),
            current_version,
            new_version,
            dependents: vec![],
            ready,
        })
    }

    /// Run pre-flight checks for a crate
    pub fn run_preflight(
        &mut self,
        crate_name: &str,
        crate_path: &Path,
    ) -> Result<PreflightResult> {
        let mut result = PreflightResult::new(crate_name);

        if self.config.no_verify {
            result.add_check(PreflightCheck::pass("verification", "Skipped (--no-verify)"));
            self.preflight_results.insert(crate_name.to_string(), result.clone());
            return Ok(result);
        }

        // Check 1: Git clean
        let git_check = self.check_git_clean(crate_path);
        result.add_check(git_check);

        // Check 2: Lint
        let lint_check = self.check_lint(crate_path);
        result.add_check(lint_check);

        // Check 3: Coverage
        let coverage_check = self.check_coverage(crate_path);
        result.add_check(coverage_check);

        // Check 4: PMAT comply (ComputeBrick defect detection)
        let comply_check = self.check_pmat_comply(crate_path);
        result.add_check(comply_check);

        // Check 5: No path dependencies
        let path_check = self.check_no_path_deps(crate_name);
        result.add_check(path_check);

        // Check 6: Version bumped
        let version_check = self.check_version_bumped(crate_name);
        result.add_check(version_check);

        // =====================================================================
        // PMAT Quality Gate Integration (PMAT-STACK-GATES)
        // =====================================================================

        // Check 7: PMAT quality-gate (comprehensive checks)
        let quality_gate_check = self.check_pmat_quality_gate(crate_path);
        result.add_check(quality_gate_check);

        // Check 8: PMAT TDG (Technical Debt Grading)
        let tdg_check = self.check_pmat_tdg(crate_path);
        result.add_check(tdg_check);

        // Check 9: PMAT dead-code analysis
        let dead_code_check = self.check_pmat_dead_code(crate_path);
        result.add_check(dead_code_check);

        // Check 10: PMAT complexity analysis
        let complexity_check = self.check_pmat_complexity(crate_path);
        result.add_check(complexity_check);

        // Check 11: PMAT SATD (Self-Admitted Technical Debt)
        let satd_check = self.check_pmat_satd(crate_path);
        result.add_check(satd_check);

        // Check 12: PMAT Popper score (falsifiability)
        let popper_check = self.check_pmat_popper(crate_path);
        result.add_check(popper_check);

        // =====================================================================
        // Book and Examples Verification (RELEASE-DOCS)
        // =====================================================================

        // Check 13: Book build
        let book_check = self.check_book_build(crate_path);
        result.add_check(book_check);

        // Check 14: Examples verification
        let examples_check = self.check_examples_run(crate_path);
        result.add_check(examples_check);

        self.preflight_results.insert(crate_name.to_string(), result.clone());
        Ok(result)
    }

    // Note: check_* methods are in releaser_preflight.rs module

    /// Execute the release plan
    #[cfg(feature = "native")]
    pub fn execute(&self, plan: &ReleasePlan) -> Result<ReleaseResult> {
        if plan.dry_run {
            return Ok(ReleaseResult {
                success: true,
                released_crates: vec![],
                message: "Dry run - no changes made".to_string(),
            });
        }

        let mut released = Vec::new();

        for release in &plan.releases {
            // Check preflight passed
            if let Some(preflight) = plan.preflight_results.get(&release.crate_name) {
                if !preflight.passed {
                    return Err(anyhow!(
                        "Pre-flight checks failed for {}: cannot release",
                        release.crate_name
                    ));
                }
            }

            // Get manifest path for version bump (only execute file ops when path exists)
            let manifest_path = self
                .checker
                .get_crate(&release.crate_name)
                .map(|c| c.manifest_path.clone())
                .filter(|p| p.exists());

            // Update Cargo.toml version (only if file exists)
            if let Some(ref path) = manifest_path {
                self.update_cargo_toml(path, &release.new_version)?;

                // Create git tag after version bump
                self.create_git_tag(&release.crate_name, &release.new_version)?;
            }

            if self.config.publish {
                if let Some(ref path) = manifest_path {
                    let crate_dir = path.parent().unwrap_or(Path::new("."));
                    self.cargo_publish(crate_dir)?;
                }
            }

            released.push(ReleasedCrate {
                name: release.crate_name.clone(),
                version: release.new_version.clone(),
                published: self.config.publish && manifest_path.is_some(),
            });
        }

        Ok(ReleaseResult {
            success: true,
            released_crates: released,
            message: format!("Successfully released {} crates", plan.releases.len()),
        })
    }

    /// Update version in Cargo.toml
    #[cfg(feature = "native")]
    pub(crate) fn update_cargo_toml(
        &self,
        manifest_path: &Path,
        new_version: &semver::Version,
    ) -> Result<()> {
        let content = std::fs::read_to_string(manifest_path)
            .map_err(|e| anyhow!("Failed to read {}: {}", manifest_path.display(), e))?;

        let version_str = new_version.to_string();

        // Replace version in [package] section using line-by-line rewrite
        // This preserves formatting better than full TOML parse/serialize
        let mut output = String::with_capacity(content.len());
        let mut in_package = false;
        let mut version_replaced = false;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed == "[package]" {
                in_package = true;
            } else if trimmed.starts_with('[') {
                in_package = false;
            }

            if in_package && !version_replaced && trimmed.starts_with("version") {
                if let Some(eq_pos) = line.find('=') {
                    let prefix = &line[..=eq_pos];
                    output.push_str(&format!("{} \"{}\"", prefix, version_str));
                    output.push('\n');
                    version_replaced = true;
                    continue;
                }
            }

            output.push_str(line);
            output.push('\n');
        }

        if !version_replaced {
            return Err(anyhow!(
                "Could not find version field in [package] section of {}",
                manifest_path.display()
            ));
        }

        std::fs::write(manifest_path, output)
            .map_err(|e| anyhow!("Failed to write {}: {}", manifest_path.display(), e))?;

        Ok(())
    }

    /// Create a git tag for the release
    #[cfg(feature = "native")]
    fn create_git_tag(&self, crate_name: &str, version: &semver::Version) -> Result<()> {
        let tag = format!("{}-v{}", crate_name, version);
        let message = format!("Release {} v{}", crate_name, version);

        let output = std::process::Command::new("git")
            .args(["tag", "-a", &tag, "-m", &message])
            .output()
            .map_err(|e| anyhow!("Failed to create git tag: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Git tag failed: {}", stderr));
        }

        Ok(())
    }

    /// Publish crate to crates.io
    #[cfg(feature = "native")]
    fn cargo_publish(&self, crate_dir: &Path) -> Result<()> {
        let mut cmd = std::process::Command::new("cargo");
        cmd.arg("publish").current_dir(crate_dir);

        if self.config.dry_run {
            cmd.arg("--dry-run");
        }

        let output = cmd.output().map_err(|e| anyhow!("Failed to run cargo publish: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("cargo publish failed: {}", stderr));
        }

        Ok(())
    }
}
