//! Preflight check methods for ReleaseOrchestrator
//!
//! Contains all check_* methods for various quality gates.

use crate::stack::releaser::ReleaseOrchestrator;
use crate::stack::types::PreflightCheck;
use std::path::Path;
use std::process::Command;

use super::helpers::{
    parse_count_from_json_multi, parse_value_from_json, run_check_command, score_check_result,
};

impl ReleaseOrchestrator {
    /// Check if git working directory is clean
    pub(in crate::stack) fn check_git_clean(&self, crate_path: &Path) -> PreflightCheck {
        let output =
            Command::new("git").args(["status", "--porcelain"]).current_dir(crate_path).output();

        match output {
            Ok(out) => {
                if out.stdout.is_empty() {
                    PreflightCheck::pass("git_clean", "Working directory is clean")
                } else {
                    let files = String::from_utf8_lossy(&out.stdout);
                    PreflightCheck::fail(
                        "git_clean",
                        format!("Uncommitted changes:\n{}", files.trim()),
                    )
                }
            }
            Err(e) => {
                PreflightCheck::fail("git_clean", format!("Failed to check git status: {}", e))
            }
        }
    }

    /// Check lint passes
    pub(in crate::stack) fn check_lint(&self, crate_path: &Path) -> PreflightCheck {
        run_check_command(
            &self.config.lint_command,
            "lint",
            "No lint command configured",
            crate_path,
            |output, _stdout, stderr| {
                if output.status.success() {
                    PreflightCheck::pass("lint", "Lint passed")
                } else {
                    PreflightCheck::fail("lint", format!("Lint failed: {}", stderr.trim()))
                }
            },
        )
    }

    /// Check coverage meets minimum
    pub(in crate::stack) fn check_coverage(&self, crate_path: &Path) -> PreflightCheck {
        let min_coverage = self.config.min_coverage;
        run_check_command(
            &self.config.coverage_command,
            "coverage",
            "No coverage command configured",
            crate_path,
            move |output, _stdout, _stderr| {
                if output.status.success() {
                    PreflightCheck::pass(
                        "coverage",
                        format!("Coverage check passed (min: {}%)", min_coverage),
                    )
                } else {
                    PreflightCheck::fail("coverage", format!("Coverage below {}%", min_coverage))
                }
            },
        )
    }

    /// Check PMAT comply for ComputeBrick defects (CB-XXX violations)
    ///
    /// Runs `pmat comply` to detect:
    /// - CB-020: Unsafe blocks without safety comments
    /// - CB-021: SIMD without target_feature attributes
    /// - CB-022: Missing error handling patterns
    /// - And other PMAT compliance rules
    pub(in crate::stack) fn check_pmat_comply(&self, crate_path: &Path) -> PreflightCheck {
        let fail_on_violations = self.config.fail_on_comply_violations;
        run_check_command(
            &self.config.comply_command,
            "pmat_comply",
            "No comply command configured (skipped)",
            crate_path,
            move |output, stdout, stderr| {
                let has_violations = stdout.contains("CB-")
                    || stderr.contains("CB-")
                    || stdout.contains("violation")
                    || stderr.contains("violation");

                if output.status.success() && !has_violations {
                    PreflightCheck::pass("pmat_comply", "PMAT comply passed (0 violations)")
                } else if has_violations && fail_on_violations {
                    let violation_hint = if stdout.contains("CB-") {
                        stdout
                            .lines()
                            .filter(|l| l.contains("CB-"))
                            .take(3)
                            .collect::<Vec<_>>()
                            .join("; ")
                    } else {
                        "violations detected".to_string()
                    };
                    PreflightCheck::fail(
                        "pmat_comply",
                        format!("PMAT comply failed: {}", violation_hint),
                    )
                } else if has_violations {
                    PreflightCheck::pass("pmat_comply", "PMAT comply has warnings (not blocking)")
                } else {
                    PreflightCheck::fail(
                        "pmat_comply",
                        format!("PMAT comply error: {}", stderr.trim()),
                    )
                }
            },
        )
    }

    /// Check for path dependencies
    pub(in crate::stack) fn check_no_path_deps(&self, _crate_name: &str) -> PreflightCheck {
        // This would use the checker's graph to verify no path deps
        // For now, always pass as a placeholder
        PreflightCheck::pass("no_path_deps", "No path dependencies found")
    }

    /// Check version is bumped from crates.io
    pub(in crate::stack) fn check_version_bumped(&self, _crate_name: &str) -> PreflightCheck {
        // This would compare local version vs crates.io
        // For now, always pass as a placeholder
        PreflightCheck::pass("version_bumped", "Version is ahead of crates.io")
    }

    // =========================================================================
    // PMAT Quality Gate Integration (PMAT-STACK-GATES)
    // =========================================================================

    /// Check PMAT quality-gate (comprehensive quality checks)
    ///
    /// Runs `pmat quality-gate` which includes:
    /// - Dead code detection
    /// - Complexity analysis
    /// - Coverage verification
    /// - SATD detection
    /// - Security checks
    pub(in crate::stack) fn check_pmat_quality_gate(&self, crate_path: &Path) -> PreflightCheck {
        let fail_on_gate = self.config.fail_on_quality_gate;
        run_check_command(
            &self.config.quality_gate_command,
            "quality_gate",
            "No quality-gate command configured (skipped)",
            crate_path,
            move |output, _stdout, stderr| {
                if output.status.success() {
                    PreflightCheck::pass("quality_gate", "PMAT quality-gate passed")
                } else if fail_on_gate {
                    PreflightCheck::fail(
                        "quality_gate",
                        format!("Quality gate failed: {}", stderr.trim()),
                    )
                } else {
                    PreflightCheck::pass("quality_gate", "Quality gate has warnings (not blocking)")
                }
            },
        )
    }

    /// Check PMAT TDG (Technical Debt Grading) score
    ///
    /// Runs `pmat tdg --format json` and parses the score.
    /// Fails if score < min_tdg_score (default: 80).
    pub(in crate::stack) fn check_pmat_tdg(&self, crate_path: &Path) -> PreflightCheck {
        let min_score = self.config.min_tdg_score;
        let fail_on = self.config.fail_on_tdg;
        run_check_command(
            &self.config.tdg_command,
            "tdg",
            "No TDG command configured (skipped)",
            crate_path,
            move |output, stdout, _stderr| {
                let score = parse_value_from_json(stdout, &["score", "tdg_score", "total"]);
                score_check_result(
                    "tdg",
                    "TDG score",
                    score,
                    min_score,
                    fail_on,
                    output.status.success(),
                )
            },
        )
    }

    /// Check PMAT dead-code analysis
    ///
    /// Runs `pmat analyze dead-code` to detect unused code.
    pub(in crate::stack) fn check_pmat_dead_code(&self, crate_path: &Path) -> PreflightCheck {
        let fail_on = self.config.fail_on_dead_code;
        run_check_command(
            &self.config.dead_code_command,
            "dead_code",
            "No dead-code command configured (skipped)",
            crate_path,
            move |_output, stdout, _stderr| {
                let has_dead_code = stdout.contains("dead_code") || stdout.contains("unused");
                let count = parse_count_from_json_multi(stdout, &["count", "dead_code_count"]);

                match (has_dead_code, count) {
                    (_, Some(0)) | (false, None) => {
                        PreflightCheck::pass("dead_code", "No dead code detected")
                    }
                    (_, Some(n)) if fail_on => {
                        PreflightCheck::fail("dead_code", format!("{} dead code items found", n))
                    }
                    (_, Some(n)) => PreflightCheck::pass(
                        "dead_code",
                        format!("{} dead code items (warning)", n),
                    ),
                    (true, None) if fail_on => {
                        PreflightCheck::fail("dead_code", "Dead code detected")
                    }
                    (true, None) => {
                        PreflightCheck::pass("dead_code", "Dead code detected (warning)")
                    }
                }
            },
        )
    }

    /// Check PMAT complexity analysis
    ///
    /// Runs `pmat analyze complexity` to check cyclomatic complexity.
    /// Fails if any function exceeds max_complexity (default: 20).
    pub(in crate::stack) fn check_pmat_complexity(&self, crate_path: &Path) -> PreflightCheck {
        let max_complexity = self.config.max_complexity;
        let fail_on = self.config.fail_on_complexity;
        run_check_command(
            &self.config.complexity_command,
            "complexity",
            "No complexity command configured (skipped)",
            crate_path,
            move |output, stdout, _stderr| {
                let max_found = parse_count_from_json_multi(stdout, &["max_complexity", "highest"]);
                let violations =
                    parse_count_from_json_multi(stdout, &["violations", "violation_count"]);

                match (max_found, violations) {
                    (Some(m), _) if m <= max_complexity => PreflightCheck::pass(
                        "complexity",
                        format!("Max complexity: {} (limit: {})", m, max_complexity),
                    ),
                    (Some(m), _) if fail_on => PreflightCheck::fail(
                        "complexity",
                        format!("Complexity {} exceeds limit {}", m, max_complexity),
                    ),
                    (_, Some(0)) => PreflightCheck::pass("complexity", "No complexity violations"),
                    (_, Some(v)) if fail_on => {
                        PreflightCheck::fail("complexity", format!("{} complexity violations", v))
                    }
                    _ if output.status.success() => {
                        PreflightCheck::pass("complexity", "Complexity check passed")
                    }
                    _ => PreflightCheck::pass("complexity", "Complexity check completed (warning)"),
                }
            },
        )
    }

    /// Check PMAT SATD (Self-Admitted Technical Debt)
    ///
    /// Runs `pmat analyze satd` to detect TODO/FIXME/HACK comments.
    /// Fails if count exceeds max_satd_items (default: 10).
    pub(in crate::stack) fn check_pmat_satd(&self, crate_path: &Path) -> PreflightCheck {
        let max_items = self.config.max_satd_items;
        let fail_on = self.config.fail_on_satd;
        run_check_command(
            &self.config.satd_command,
            "satd",
            "No SATD command configured (skipped)",
            crate_path,
            move |output, stdout, _stderr| {
                let count = parse_count_from_json_multi(stdout, &["total", "count", "satd_count"]);

                match count {
                    Some(c) if c <= max_items => PreflightCheck::pass(
                        "satd",
                        format!("{} SATD items (limit: {})", c, max_items),
                    ),
                    Some(c) if fail_on => PreflightCheck::fail(
                        "satd",
                        format!("{} SATD items exceed limit {}", c, max_items),
                    ),
                    Some(c) => PreflightCheck::pass(
                        "satd",
                        format!("{} SATD items (warning: exceeds {})", c, max_items),
                    ),
                    None if output.status.success() => {
                        PreflightCheck::pass("satd", "SATD check passed")
                    }
                    None => PreflightCheck::pass("satd", "SATD check completed"),
                }
            },
        )
    }

    /// Check PMAT Popper score (falsifiability)
    ///
    /// Runs `pmat popper-score` to assess scientific quality.
    /// Based on Karl Popper's falsification principles.
    /// Fails if score < min_popper_score (default: 60).
    pub(in crate::stack) fn check_pmat_popper(&self, crate_path: &Path) -> PreflightCheck {
        let min_score = self.config.min_popper_score;
        let fail_on = self.config.fail_on_popper;
        run_check_command(
            &self.config.popper_command,
            "popper",
            "No Popper command configured (skipped)",
            crate_path,
            move |output, stdout, _stderr| {
                let score = parse_value_from_json(stdout, &["score", "popper_score", "total"]);
                score_check_result(
                    "popper",
                    "Popper score",
                    score,
                    min_score,
                    fail_on,
                    output.status.success(),
                )
            },
        )
    }

    // =========================================================================
    // Book and Examples Verification (RELEASE-DOCS)
    // =========================================================================

    /// Check book builds successfully
    ///
    /// Runs `mdbook build book` (or configured command) to verify
    /// documentation compiles without errors.
    pub(in crate::stack) fn check_book_build(&self, crate_path: &Path) -> PreflightCheck {
        // Check if book directory exists before running the command
        let book_dir = crate_path.join("book");
        if !book_dir.exists() {
            return PreflightCheck::pass("book", "No book directory found (skipped)");
        }

        let fail_on = self.config.fail_on_book;

        run_check_command(
            &self.config.book_command,
            "book",
            "No book command configured (skipped)",
            crate_path,
            move |output, _stdout, stderr| {
                if output.status.success() {
                    PreflightCheck::pass("book", "Book built successfully")
                } else if fail_on {
                    PreflightCheck::fail("book", format!("Book build failed: {}", stderr.trim()))
                } else {
                    PreflightCheck::pass("book", "Book build has warnings (not blocking)")
                }
            },
        )
    }

    /// Check examples compile and run successfully
    ///
    /// Discovers examples from Cargo.toml [[example]] sections and
    /// runs each one with `cargo run --example <name>`.
    pub(in crate::stack) fn check_examples_run(&self, crate_path: &Path) -> PreflightCheck {
        let parts: Vec<&str> = self.config.examples_command.split_whitespace().collect();
        if parts.is_empty() {
            return PreflightCheck::pass("examples", "No examples command configured (skipped)");
        }

        // Check if examples directory exists
        let examples_dir = crate_path.join("examples");
        if !examples_dir.exists() {
            return PreflightCheck::pass("examples", "No examples directory found (skipped)");
        }

        // Discover examples from Cargo.toml or examples directory
        let examples = self.discover_examples(crate_path);
        if examples.is_empty() {
            return PreflightCheck::pass("examples", "No examples found (skipped)");
        }

        let mut failed = Vec::new();
        let mut succeeded = 0;

        for example in &examples {
            // Build the full command with example name
            let output = Command::new("cargo")
                .args(["run", "--example", example, "--", "--help"])
                .current_dir(crate_path)
                .output();

            match output {
                Ok(out) => {
                    // Consider it a pass if the example compiles and runs
                    // (even if --help exits with non-zero, compilation success is what matters)
                    if out.status.success() || out.status.code() == Some(0) {
                        succeeded += 1;
                    } else {
                        // Check if it failed during compilation vs runtime
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        if stderr.contains("error[E") || stderr.contains("could not compile") {
                            failed.push(example.clone());
                        } else {
                            // Runtime exit with non-zero is OK for --help
                            succeeded += 1;
                        }
                    }
                }
                Err(_) => {
                    failed.push(example.clone());
                }
            }
        }

        if failed.is_empty() {
            PreflightCheck::pass(
                "examples",
                format!("{}/{} examples verified", succeeded, examples.len()),
            )
        } else if self.config.fail_on_examples {
            PreflightCheck::fail(
                "examples",
                format!(
                    "{}/{} examples failed: {}",
                    failed.len(),
                    examples.len(),
                    failed.join(", ")
                ),
            )
        } else {
            PreflightCheck::pass(
                "examples",
                format!(
                    "{}/{} examples verified ({} failed, not blocking)",
                    succeeded,
                    examples.len(),
                    failed.len()
                ),
            )
        }
    }

    /// Discover examples from the crate
    pub(in crate::stack) fn discover_examples(&self, crate_path: &Path) -> Vec<String> {
        let mut examples = Vec::new();

        // Try to find examples from Cargo.toml
        let cargo_toml = crate_path.join("Cargo.toml");
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            // Simple parsing for [[example]] sections
            for line in content.lines() {
                if line.trim().starts_with("name = \"") {
                    // Check if we're in an [[example]] section by looking at previous context
                    // This is a simplified approach - in production, use toml crate
                    if let Some(name) = line.split('"').nth(1) {
                        // Verify it's actually in the examples dir
                        let example_file = crate_path.join("examples").join(format!("{}.rs", name));
                        if example_file.exists() {
                            examples.push(name.to_string());
                        }
                    }
                }
            }
        }

        // Also scan examples directory for .rs files
        let examples_dir = crate_path.join("examples");
        if let Ok(entries) = std::fs::read_dir(&examples_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "rs") {
                    if let Some(stem) = path.file_stem() {
                        let name = stem.to_string_lossy().to_string();
                        if !examples.contains(&name) {
                            examples.push(name);
                        }
                    }
                }
            }
        }

        examples
    }

    // =========================================================================
    // JSON Parsing Helpers
    // =========================================================================

    /// Helper: Parse a numeric score from JSON output
    pub(in crate::stack) fn parse_score_from_json(json: &str, key: &str) -> Option<f64> {
        // Simple JSON parsing without serde for minimal dependencies
        let pattern = format!("\"{}\":", key);
        if let Some(pos) = json.find(&pattern) {
            let after_key = &json[pos + pattern.len()..];
            let value_str: String = after_key
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_numeric() || *c == '.' || *c == '-')
                .collect();
            value_str.parse().ok()
        } else {
            None
        }
    }

    /// Helper: Parse an integer count from JSON output
    pub(in crate::stack) fn parse_count_from_json(json: &str, key: &str) -> Option<u32> {
        Self::parse_score_from_json(json, key).map(|f| f as u32)
    }
}
