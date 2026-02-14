//! Output formatting utilities
//!
//! Toyota Way: Visualization - Make information visible and clear.
//! Rich output primitives: Unicode box-drawing, semantic color coding, tables.

use colored::{ColoredString, Colorize};
use std::fmt::Display;
use std::path::Path;
use tabled::settings::{object::Columns, Alignment, Modify, Style};

/// APR format magic bytes
pub(crate) const MAGIC_APRN: [u8; 4] = [0x41, 0x50, 0x52, 0x4E]; // "APRN" - aprender v1
pub(crate) const MAGIC_APR1: [u8; 4] = [0x41, 0x50, 0x52, 0x31]; // "APR1" - whisper.apr
pub(crate) const MAGIC_APR2: [u8; 4] = [0x41, 0x50, 0x52, 0x32]; // "APR2" - aprender v2
pub(crate) const MAGIC_APR0: [u8; 4] = [0x41, 0x50, 0x52, 0x00]; // "APR\0" - ONE TRUE APR format (v2)

/// GGUF format magic bytes
pub(crate) const MAGIC_GGUF: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF" - llama.cpp quantized

/// Check if magic bytes are valid (supports APR, GGUF formats)
/// BUG-DEBUG-001 FIX: Now accepts GGUF format magic as valid
pub(crate) fn is_valid_magic(magic: &[u8]) -> bool {
    magic.len() >= 4
        && (magic[..4] == MAGIC_APRN
            || magic[..4] == MAGIC_APR1
            || magic[..4] == MAGIC_APR2
            || magic[..4] == MAGIC_APR0
            || magic[..4] == MAGIC_GGUF)
}

/// Get format name from magic bytes
/// BUG-PROBAR-001 FIX: Now returns correct name for GGUF format
pub(crate) fn format_name(magic: &[u8]) -> &'static str {
    if magic.len() < 4 {
        return "Unknown";
    }
    let m = &magic[..4];
    if m == MAGIC_APRN {
        return "APRN (aprender v1)";
    }
    if m == MAGIC_APR1 {
        return "APR1 (whisper.apr)";
    }
    if m == MAGIC_APR2 {
        return "APR2 (aprender v2)";
    }
    if m == MAGIC_APR0 {
        return "APR v2 (ONE TRUE format)";
    }
    if m == MAGIC_GGUF {
        return "GGUF (llama.cpp)";
    }
    "Unknown"
}

/// Print a section header
pub(crate) fn section(title: &str) {
    println!("\n{}", format!("=== {title} ===").cyan().bold());
}

// Note: These functions are used by various commands and may appear unused
// when only some commands are being compiled/used.

/// Print a key-value pair
pub(crate) fn kv(key: &str, value: impl std::fmt::Display) {
    println!("  {}: {}", key.white().bold(), value);
}

/// Print a success message
pub(crate) fn success(msg: &str) {
    println!("{} {}", "[PASS]".green().bold(), msg);
}

/// Print a warning message
pub(crate) fn warning(msg: &str) {
    println!("{} {}", "[WARN]".yellow().bold(), msg);
}

/// Print a failure message
#[allow(dead_code)]
pub(crate) fn fail(msg: &str) {
    println!("{} {}", "[FAIL]".red().bold(), msg);
}

/// Print an info message
#[allow(dead_code)]
pub(crate) fn info(msg: &str) {
    println!("{} {}", "[INFO]".blue(), msg);
}

/// Print an error message
#[allow(dead_code)]
pub(crate) fn error(msg: &str) {
    eprintln!("{} {}", "[ERROR]".red().bold(), msg);
}

/// Print a warning message (alias for backward compatibility)
pub(crate) fn warn(msg: &str) {
    warning(msg);
}

/// Format bytes as human-readable size
pub(crate) fn format_size(bytes: u64) -> String {
    humansize::format_size(bytes, humansize::BINARY)
}

// ============================================================================
// Rich Output Primitives
// ============================================================================

// ── Section Headers (Unicode box-drawing) ──

/// Print a bold section header with Unicode line drawing
pub(crate) fn header(title: &str) {
    let bar = "━".repeat(60);
    println!("\n{}", bar.cyan());
    println!("  {}", title.cyan().bold());
    println!("{}", bar.cyan());
}

/// Print a subsection header with lighter line drawing
pub(crate) fn subheader(title: &str) {
    println!("\n  {} {}", "───".dimmed(), title.cyan());
}

// ── Status Badges (Unicode + color) ──

/// Format a pass badge: "✓ label" in green bold
pub(crate) fn badge_pass(label: &str) -> String {
    format!("{} {}", "✓".green().bold(), label.green().bold())
}

/// Format a fail badge: "✗ label" in red bold
pub(crate) fn badge_fail(label: &str) -> String {
    format!("{} {}", "✗".red().bold(), label.red().bold())
}

/// Format a warning badge: "⚠ label" in yellow bold
pub(crate) fn badge_warn(label: &str) -> String {
    format!("{} {}", "⚠".yellow().bold(), label.yellow().bold())
}

/// Format a skip badge: "○ label" in dimmed
pub(crate) fn badge_skip(label: &str) -> String {
    format!("{} {}", "○".dimmed(), label.dimmed())
}

/// Format an info badge: "ℹ label" in blue
pub(crate) fn badge_info(label: &str) -> String {
    format!("{} {}", "ℹ".blue(), label.blue())
}

// ── Metrics Display ──

/// Print a metric with label, value, and unit
pub(crate) fn metric(label: &str, value: impl Display, unit: &str) {
    println!(
        "  {}: {}{}",
        label.dimmed(),
        format!("{value}").white().bold(),
        if unit.is_empty() {
            String::new()
        } else {
            format!(" {}", unit.dimmed())
        }
    );
}

/// Render a progress bar as a string
pub(crate) fn progress_bar(current: usize, total: usize, width: usize) -> String {
    if total == 0 {
        return format!("[{}]", " ".repeat(width));
    }
    let filled = (current * width) / total;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

/// Format a duration in human-readable form
pub(crate) fn duration_fmt(ms: u64) -> String {
    if ms >= 60_000 {
        format!("{:.1}m", ms as f64 / 60_000.0)
    } else if ms >= 1_000 {
        format!("{:.1}s", ms as f64 / 1_000.0)
    } else {
        format!("{ms}ms")
    }
}

/// Format a count with thousands separators
pub(crate) fn count_fmt(n: usize) -> String {
    if n < 1_000 {
        return n.to_string();
    }
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

// ── Semantic Coloring ──

/// Dimmed text for secondary information
#[allow(dead_code)]
pub(crate) fn dim(text: &str) -> ColoredString {
    text.dimmed()
}

/// Highlighted text for primary information
#[allow(dead_code)]
pub(crate) fn highlight(text: &str) -> ColoredString {
    text.white().bold()
}

/// Filepath in cyan
#[allow(dead_code)]
pub(crate) fn filepath(path: &Path) -> ColoredString {
    path.display().to_string().cyan()
}

/// Color a dtype string semantically
pub(crate) fn dtype_color(dtype: &str) -> ColoredString {
    match dtype.to_uppercase().as_str() {
        "F32" => dtype.green(),
        "F16" | "BF16" => dtype.yellow(),
        s if s.starts_with("Q4") => dtype.magenta().bold(),
        s if s.starts_with("Q5") => dtype.magenta(),
        s if s.starts_with("Q6") => dtype.blue(),
        s if s.starts_with("Q8") => dtype.cyan(),
        _ => dtype.white(),
    }
}

/// Color a grade string semantically
pub(crate) fn grade_color(grade: &str) -> ColoredString {
    match grade {
        "A+" | "A" => grade.green().bold(),
        "B+" | "B" => grade.green(),
        "C+" | "C" => grade.yellow(),
        "D" => grade.yellow().bold(),
        _ => grade.red().bold(),
    }
}

// ── Table Rendering (via tabled crate) ──

/// Render a table from headers and rows.
/// Returns the formatted table string with Unicode box-drawing borders.
pub(crate) fn table(headers: &[&str], rows: &[Vec<String>]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    // Build data: header row + data rows
    let mut data: Vec<Vec<String>> = Vec::with_capacity(rows.len() + 1);
    data.push(headers.iter().map(|h| (*h).to_string()).collect());
    for row in rows {
        data.push(row.clone());
    }

    // Use tabled's builder for dynamic column counts
    let mut builder = tabled::builder::Builder::new();
    for row in &data {
        builder.push_record(row.iter().map(String::as_str));
    }
    let mut tbl = builder.build();
    tbl.with(Style::rounded());
    tbl.to_string()
}

/// Render a key-value table (two columns: Key, Value)
pub(crate) fn kv_table(pairs: &[(&str, String)]) -> String {
    if pairs.is_empty() {
        return String::new();
    }
    let mut builder = tabled::builder::Builder::new();
    for (key, value) in pairs {
        builder.push_record([*key, value.as_str()]);
    }
    let mut tbl = builder.build();
    tbl.with(Style::rounded())
        .with(Modify::new(Columns::first()).with(Alignment::right()));
    tbl.to_string()
}

// ── Pipeline Stage Display ──

/// Status of a pipeline stage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum StageStatus {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
}

/// Print a pipeline stage with status indicator
pub(crate) fn pipeline_stage(name: &str, status: StageStatus) {
    let indicator = match status {
        StageStatus::Pending => "○".dimmed(),
        StageStatus::Running => "◉".yellow().bold(),
        StageStatus::Done => "✓".green().bold(),
        StageStatus::Failed => "✗".red().bold(),
        StageStatus::Skipped => "─".dimmed(),
    };
    let label = match status {
        StageStatus::Pending => name.dimmed(),
        StageStatus::Running => name.yellow(),
        StageStatus::Done => name.green(),
        StageStatus::Failed => name.red(),
        StageStatus::Skipped => name.dimmed(),
    };
    println!("  {} {}", indicator, label);
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
