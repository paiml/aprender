impl FailFastReporter {
    /// Generate a comprehensive markdown report from a fail-fast report
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn generate_markdown(&self, report: &FailFastReport) -> String {
        let mut md = String::new();

        // Header
        let _ = writeln!(md, "# Fail-Fast Report: {}\n", report.failure.gate_id);

        // Failure Summary Table
        md.push_str("## Failure Summary\n\n");
        md.push_str("| Field | Value |\n");
        md.push_str("|-------|-------|\n");
        let _ = writeln!(md, "| Gate | `{}` |", report.failure.gate_id);
        let _ = writeln!(md, "| Model | `{}` |", report.failure.model);
        let _ = writeln!(md, "| Format | {} |", report.failure.format);
        let _ = writeln!(md, "| Backend | {} |", report.failure.backend);
        let _ = writeln!(md, "| Outcome | {} |", report.failure.outcome);
        if let Some(code) = report.failure.exit_code {
            let _ = writeln!(md, "| Exit Code | {code} |");
        }
        let _ = writeln!(md, "| Duration | {}ms |", report.failure.duration_ms);
        md.push('\n');

        // Reason
        md.push_str("### Reason\n\n");
        let _ = writeln!(md, "{}\n", report.failure.reason);

        // Environment Table
        md.push_str("## Environment\n\n");
        md.push_str("| Field | Value |\n");
        md.push_str("|-------|-------|\n");
        let _ = writeln!(
            md,
            "| OS | {} {} |",
            report.environment.os, report.environment.arch
        );
        let _ = writeln!(md, "| apr-qa | {} |", report.environment.aprender_qa_version);
        let _ = writeln!(md, "| apr-cli | {} |", report.environment.apr_cli_version);
        let _ = writeln!(
            md,
            "| Git | {} ({}){}|",
            report.environment.git_commit,
            report.environment.git_branch,
            if report.environment.git_dirty {
                " [dirty]"
            } else {
                ""
            }
        );
        let _ = writeln!(md, "| Rust | {} |", report.environment.rustc_version);
        md.push('\n');

        // Pipeline Check Results
        if let Some(ref check) = report.diagnostics.check {
            md.push_str("## Pipeline Check Results\n\n");
            if check.success {
                md.push_str("All pipeline checks passed.\n\n");
            } else {
                md.push_str("**Pipeline check failed:**\n\n");
                md.push_str("```\n");
                md.push_str(&check.stderr);
                md.push_str("\n```\n\n");
            }
        }

        // Model Metadata
        if let Some(ref inspect) = report.diagnostics.inspect {
            md.push_str("## Model Metadata\n\n");
            md.push_str("<details>\n<summary>apr inspect output</summary>\n\n");
            md.push_str("```json\n");
            md.push_str(&inspect.stdout);
            md.push_str("\n```\n\n");
            md.push_str("</details>\n\n");
        }

        // Tensor Info
        if let Some(ref tensors) = report.diagnostics.tensors {
            md.push_str("## Tensor Inventory\n\n");
            md.push_str("<details>\n<summary>apr tensors output</summary>\n\n");
            md.push_str("```json\n");
            md.push_str(&tensors.stdout);
            md.push_str("\n```\n\n");
            md.push_str("</details>\n\n");
        }

        // Trace (if available)
        if let Some(ref trace) = report.diagnostics.trace {
            md.push_str("## Layer Trace\n\n");
            md.push_str("<details>\n<summary>apr trace output</summary>\n\n");
            md.push_str("```json\n");
            md.push_str(&trace.stdout);
            md.push_str("\n```\n\n");
            md.push_str("</details>\n\n");
        }

        // Error Explanation
        if let Some(ref explain) = report.diagnostics.explain {
            if !explain.stdout.is_empty() {
                md.push_str("## Error Analysis\n\n");
                md.push_str(&explain.stdout);
                md.push_str("\n\n");
            }
        }

        // Stderr Capture
        if let Some(ref stderr) = report.failure.stderr {
            if !stderr.is_empty() {
                md.push_str("## Stderr Capture\n\n");
                md.push_str("<details>\n<summary>Full stderr output</summary>\n\n");
                md.push_str("```\n");
                md.push_str(stderr);
                md.push_str("\n```\n\n");
                md.push_str("</details>\n\n");
            }
        }

        // Reproduction
        md.push_str("## Reproduction\n\n");
        md.push_str("```bash\n");
        md.push_str("# Reproduce this failure\n");
        let _ = writeln!(md, "{}\n", report.reproduction.command);
        md.push_str("# Run diagnostics manually\n");
        let _ = writeln!(md, "apr check {}", report.reproduction.model_path);
        let _ = writeln!(
            md,
            "apr trace {} --payload -v",
            report.reproduction.model_path
        );
        let _ = writeln!(md, "apr explain {}", report.failure.gate_id);
        md.push_str("```\n");

        md
    }
}

// Helper functions for environment collection

/// Get the apr CLI version string by running `apr --version`
fn get_apr_version() -> String {
    Command::new("apr")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .map(|s| s.replace("apr ", "").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get the short git commit hash for the current HEAD
fn get_git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .map_or_else(
            || "unknown".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}

/// Get the current git branch name
fn get_git_branch() -> String {
    Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .map_or_else(
            || "unknown".to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
}

/// Check if the git working tree has uncommitted changes
fn get_git_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output().is_ok_and(|o| !o.stdout.is_empty())
}

/// Get the rustc compiler version string
fn get_rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .map_or_else(
            || "unknown".to_string(),
            |o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .replace("rustc ", "")
            },
        )
}


#[cfg(test)]
#[path = "diagnostics_tests.rs"]
mod diagnostics_tests;
