//! Falsification command implementations
//!
//! This module contains the Popperian Falsification Checklist command extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use batuta::falsification::{CheckItem, CheckStatus, ChecklistResult, Severity, TpsGrade};
use std::path::{Path, PathBuf};

/// Falsify output format
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum FalsifyOutputFormat {
    /// Human-readable text output with colors
    Text,
    /// JSON output for CI/CD integration
    Json,
    /// Markdown output for documentation
    Markdown,
    /// GitHub Actions annotations
    GithubActions,
}

/// Run Popperian Falsification Checklist
pub fn cmd_falsify(
    path: PathBuf,
    critical_only: bool,
    format: FalsifyOutputFormat,
    output: Option<PathBuf>,
    min_grade: &str,
    verbose: bool,
) -> anyhow::Result<()> {
    use batuta::falsification::{evaluate_critical_only, evaluate_project};

    let min_grade_threshold = parse_grade_threshold(min_grade)?;
    let project_root = resolve_project_root(&path);
    let result = if critical_only {
        evaluate_critical_only(&project_root)
    } else {
        evaluate_project(&project_root)
    };

    let output_text = match format {
        FalsifyOutputFormat::Text => format_falsify_text(&result, verbose),
        FalsifyOutputFormat::Json => {
            serde_json::to_string_pretty(&result).unwrap_or_else(|e| format!("JSON error: {}", e))
        }
        FalsifyOutputFormat::Markdown => format_falsify_markdown(&result, verbose),
        FalsifyOutputFormat::GithubActions => format_falsify_github_actions(&result),
    };

    if let Some(output_path) = output {
        std::fs::write(&output_path, &output_text)?;
        println!("{} Report written to: {}", "✓".bright_green(), output_path.display());
    } else {
        println!("{}", output_text);
    }

    check_grade_threshold(&result, min_grade_threshold)?;
    Ok(())
}

fn parse_grade_threshold(min_grade: &str) -> anyhow::Result<TpsGrade> {
    match min_grade.to_lowercase().replace('-', "").as_str() {
        "toyotastandard" | "toyota" => Ok(TpsGrade::ToyotaStandard),
        "kaizenrequired" | "kaizen" => Ok(TpsGrade::KaizenRequired),
        "andonwarning" | "andon" => Ok(TpsGrade::AndonWarning),
        "stoptheline" | "stop" => Ok(TpsGrade::StopTheLine),
        _ => anyhow::bail!(
            "Invalid min-grade: '{}'. Valid: toyota-standard, kaizen-required, andon-warning, stop-the-line",
            min_grade
        ),
    }
}

fn check_grade_threshold(result: &ChecklistResult, threshold: TpsGrade) -> anyhow::Result<()> {
    let passes = match threshold {
        TpsGrade::ToyotaStandard => result.grade == TpsGrade::ToyotaStandard,
        TpsGrade::KaizenRequired => {
            matches!(result.grade, TpsGrade::ToyotaStandard | TpsGrade::KaizenRequired)
        }
        TpsGrade::AndonWarning => !matches!(result.grade, TpsGrade::StopTheLine),
        TpsGrade::StopTheLine => true,
    };
    if !passes {
        anyhow::bail!("Grade {} does not meet minimum threshold {}", result.grade, threshold);
    }
    Ok(())
}

/// Resolve a path to its project root directory.
///
/// If `path` is a file, walks up the directory tree looking for project markers
/// (Cargo.toml, .git, pyproject.toml). If `path` is already a directory, returns it.
fn resolve_project_root(path: &Path) -> PathBuf {
    let start = if path.is_file() { path.parent().unwrap_or(path) } else { path };

    let markers = ["Cargo.toml", ".git", "pyproject.toml"];
    let mut current = start;
    loop {
        for marker in &markers {
            if current.join(marker).exists() {
                return current.to_path_buf();
            }
        }
        match current.parent() {
            Some(parent) if parent != current => current = parent,
            _ => break,
        }
    }

    // Fallback: return the original path (directory or file's parent)
    start.to_path_buf()
}

// ============================================================================
// Text Formatting Helpers
// ============================================================================

fn status_icon_text(status: CheckStatus) -> String {
    match status {
        CheckStatus::Pass => "✓".bright_green().to_string(),
        CheckStatus::Partial => "◐".bright_yellow().to_string(),
        CheckStatus::Fail => "✗".bright_red().to_string(),
        CheckStatus::Skipped => "○".dimmed().to_string(),
    }
}

fn severity_tag_text(severity: Severity) -> String {
    match severity {
        Severity::Critical => "[CRITICAL]".on_red().white().to_string(),
        Severity::Major => "[MAJOR]".bright_red().to_string(),
        Severity::Minor => "[MINOR]".bright_yellow().to_string(),
        Severity::Info => "[INFO]".dimmed().to_string(),
    }
}

fn grade_color_text(grade: TpsGrade) -> String {
    match grade {
        TpsGrade::ToyotaStandard => "✓ Toyota Standard".bright_green().to_string(),
        TpsGrade::KaizenRequired => "◐ Kaizen Required".bright_yellow().to_string(),
        TpsGrade::AndonWarning => "⚠ Andon Warning".bright_red().to_string(),
        TpsGrade::StopTheLine => "✗ STOP THE LINE".on_red().white().bold().to_string(),
    }
}

fn format_item_text(item: &CheckItem, verbose: bool) -> String {
    let mut out = format!(
        "  {} {} {} {}\n",
        status_icon_text(item.status),
        item.id.bold(),
        item.name,
        severity_tag_text(item.severity)
    );

    if verbose {
        out.push_str(&format!("    Claim: {}\n", item.claim.dimmed()));
        if !item.tps_principle.is_empty() {
            out.push_str(&format!("    TPS: {}\n", item.tps_principle.cyan()));
        }
        if let Some(reason) = &item.rejection_reason {
            out.push_str(&format!("    Reason: {}\n", reason.bright_red()));
        }
        for evidence in &item.evidence {
            out.push_str(&format!("    Evidence: {}\n", evidence.description.dimmed()));
        }
    }
    out
}

fn format_falsify_text(result: &ChecklistResult, verbose: bool) -> String {
    let mut output = String::new();

    // Header
    output.push_str(&format!(
        "{}\n",
        "╔═══════════════════════════════════════════════════════════════════╗".bright_cyan()
    ));
    output.push_str(&format!(
        "{}\n",
        "║     POPPERIAN FALSIFICATION CHECKLIST - Sovereign AI Protocol    ║".bright_cyan()
    ));
    output.push_str(&format!(
        "{}\n\n",
        "╚═══════════════════════════════════════════════════════════════════╝".bright_cyan()
    ));

    // Project info
    output.push_str(&format!("Project: {}\n", result.project_path.display().to_string().cyan()));
    output.push_str(&format!("Evaluated: {}\n\n", result.timestamp.dimmed()));

    // Grade
    output.push_str(&format!("Grade: {}\n", grade_color_text(result.grade)));
    output.push_str(&format!("Score: {:.1}%\n", result.score));
    output.push_str(&format!(
        "Items: {}/{} passed, {} failed\n\n",
        result.passed_items, result.total_items, result.failed_items
    ));

    // Sections
    for (section_name, items) in &result.sections {
        output.push_str(&format!("{}\n", format!("─── {} ───", section_name).bold()));
        for item in items {
            output.push_str(&format_item_text(item, verbose));
        }
        output.push('\n');
    }

    // Summary
    if result.has_critical_failure {
        output.push_str(&format!(
            "{}\n",
            "⚠️  CRITICAL FAILURE DETECTED - Release blocked!".on_red().white().bold()
        ));
    } else if result.passes() {
        output.push_str(&format!(
            "{}\n",
            "✅ All critical checks passed - Release allowed".bright_green()
        ));
    }

    output
}

// ============================================================================
// Markdown Formatting Helpers
// ============================================================================

fn status_icon_md(status: CheckStatus) -> &'static str {
    match status {
        CheckStatus::Pass => "✅ Pass",
        CheckStatus::Partial => "⚠️ Partial",
        CheckStatus::Fail => "❌ Fail",
        CheckStatus::Skipped => "⏭️ Skipped",
    }
}

fn severity_icon_md(severity: Severity) -> &'static str {
    match severity {
        Severity::Critical => "🔴 Critical",
        Severity::Major => "🟠 Major",
        Severity::Minor => "🟡 Minor",
        Severity::Info => "🔵 Info",
    }
}

fn grade_badge_md(grade: TpsGrade) -> &'static str {
    match grade {
        TpsGrade::ToyotaStandard => {
            "![Grade](https://img.shields.io/badge/Grade-Toyota%20Standard-brightgreen)"
        }
        TpsGrade::KaizenRequired => {
            "![Grade](https://img.shields.io/badge/Grade-Kaizen%20Required-yellow)"
        }
        TpsGrade::AndonWarning => {
            "![Grade](https://img.shields.io/badge/Grade-Andon%20Warning-orange)"
        }
        TpsGrade::StopTheLine => {
            "![Grade](https://img.shields.io/badge/Grade-STOP%20THE%20LINE-red)"
        }
    }
}

fn format_section_md(section_name: &str, items: &[CheckItem], verbose: bool) -> String {
    let mut out = format!("## {}\n\n", section_name);
    out.push_str("| ID | Name | Status | Severity |\n");
    out.push_str("|----|------|--------|----------|\n");

    for item in items {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            item.id,
            item.name,
            status_icon_md(item.status),
            severity_icon_md(item.severity)
        ));
    }

    if verbose {
        let failures: Vec<_> = items
            .iter()
            .filter(|i| i.status == CheckStatus::Fail || i.status == CheckStatus::Partial)
            .collect();
        if !failures.is_empty() {
            out.push_str("\n### Details\n\n");
            for item in failures {
                out.push_str(&format!("#### {} - {}\n\n", item.id, item.name));
                out.push_str(&format!("**Claim:** {}\n\n", item.claim));
                if let Some(reason) = &item.rejection_reason {
                    out.push_str(&format!("**Rejection:** {}\n\n", reason));
                }
            }
        }
    }
    out.push('\n');
    out
}

fn format_falsify_markdown(result: &ChecklistResult, verbose: bool) -> String {
    let mut output = String::new();

    output.push_str("# Popperian Falsification Checklist Report\n\n");
    output.push_str(&format!("**Project:** `{}`\n\n", result.project_path.display()));
    output.push_str(&format!("**Evaluated:** {}\n\n", result.timestamp));
    output.push_str(&format!("{}\n\n", grade_badge_md(result.grade)));

    // Summary table
    output.push_str("## Summary\n\n");
    output.push_str("| Metric | Value |\n|--------|-------|\n");
    output.push_str(&format!("| Score | {:.1}% |\n", result.score));
    output.push_str(&format!("| Passed | {} |\n", result.passed_items));
    output.push_str(&format!("| Failed | {} |\n", result.failed_items));
    output.push_str(&format!("| Total | {} |\n", result.total_items));
    output.push_str(&format!(
        "| Critical Failure | {} |\n\n",
        if result.has_critical_failure { "Yes" } else { "No" }
    ));

    for (section_name, items) in &result.sections {
        output.push_str(&format_section_md(section_name, items, verbose));
    }

    output
}

// ============================================================================
// GitHub Actions Formatting
// ============================================================================

fn format_falsify_github_actions(result: &ChecklistResult) -> String {
    let mut output = String::new();

    output.push_str(&format!("::set-output name=score::{:.1}\n", result.score));
    output.push_str(&format!("::set-output name=grade::{}\n", result.grade));
    output.push_str(&format!("::set-output name=passes::{}\n", result.passes()));

    for items in result.sections.values() {
        for item in items.iter().filter(|i| i.status == CheckStatus::Fail) {
            let level = match item.severity {
                Severity::Critical | Severity::Major => "error",
                Severity::Minor => "warning",
                Severity::Info => "notice",
            };
            let reason =
                item.rejection_reason.as_ref().map(|r| format!(" - {}", r)).unwrap_or_default();
            output.push_str(&format!("::{} title={}::{}{}\n", level, item.id, item.name, reason));
        }
    }

    if result.has_critical_failure {
        output.push_str(
            "::error::Popperian Falsification Check FAILED - Critical failure detected\n",
        );
    } else if !result.passes() {
        output.push_str("::warning::Popperian Falsification Check - Grade below threshold\n");
    } else {
        output.push_str(&format!(
            "::notice::Popperian Falsification Check PASSED - Grade: {} ({:.1}%)\n",
            result.grade, result.score
        ));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_resolve_project_root_from_directory() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(resolve_project_root(&root), root);
    }

    #[test]
    fn test_resolve_project_root_from_file() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let file = root.join("src/cli/falsify.rs");
        assert_eq!(resolve_project_root(&file), root);
    }

    #[test]
    fn test_resolve_project_root_from_nested_dir() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let nested = root.join("src/cli");
        assert_eq!(resolve_project_root(&nested), root);
    }

    #[test]
    fn test_resolve_project_root_from_spec_file() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let spec_dir = root.join("docs/specifications");
        if spec_dir.exists() {
            assert_eq!(resolve_project_root(&spec_dir), root);
        }
    }

    #[test]
    fn test_resolve_project_root_nonexistent_falls_back() {
        let tmp = std::env::temp_dir().join("batuta_test_no_project_root");
        let _ = fs::create_dir_all(&tmp);
        let result = resolve_project_root(&tmp);
        assert_eq!(result, tmp);
        let _ = fs::remove_dir(&tmp);
    }

    #[test]
    fn test_parse_grade_threshold_valid() {
        assert!(parse_grade_threshold("toyota-standard").is_ok());
        assert!(parse_grade_threshold("kaizen").is_ok());
        assert!(parse_grade_threshold("andon").is_ok());
        assert!(parse_grade_threshold("stop").is_ok());
    }

    #[test]
    fn test_parse_grade_threshold_invalid() {
        assert!(parse_grade_threshold("invalid").is_err());
    }
}
