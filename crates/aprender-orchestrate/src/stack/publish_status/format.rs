//! Report formatting utilities.
//!
//! Provides text and JSON formatting for publish status reports.

use anyhow::Result;

use super::types::PublishStatusReport;

// ============================================================================
// PUB-007: Display Formatting
// ============================================================================

/// Format report as text table
pub fn format_report_text(report: &PublishStatusReport) -> String {
    use std::fmt::Write;

    let mut out = String::new();

    // Header
    writeln!(
        out,
        "{:<20} {:>10} {:>10} {:>10} {:>12}",
        "Crate", "Local", "crates.io", "Git", "Action"
    )
    .ok();
    writeln!(out, "{}", "─".repeat(65)).ok();

    // Rows
    for status in &report.crates {
        let local = status.local_version.as_deref().unwrap_or("-");
        let remote = status.crates_io_version.as_deref().unwrap_or("-");
        let git = status.git_status.summary();

        writeln!(
            out,
            "{:<20} {:>10} {:>10} {:>10} {:>2} {:>9}",
            status.name,
            local,
            remote,
            git,
            status.action.symbol(),
            status.action.description()
        )
        .ok();
    }

    writeln!(out, "{}", "─".repeat(65)).ok();

    // Summary
    writeln!(out).ok();
    writeln!(
        out,
        "\u{1F4CA} {} crates: {} publish, {} commit, {} up-to-date",
        report.total, report.needs_publish, report.needs_commit, report.up_to_date
    )
    .ok();
    writeln!(
        out,
        "\u{26A1} {}ms (cache: {} hits, {} misses)",
        report.elapsed_ms, report.cache_hits, report.cache_misses
    )
    .ok();

    out
}

/// Format report as JSON
pub fn format_report_json(report: &PublishStatusReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}
