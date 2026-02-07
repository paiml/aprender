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
    if magic.len() >= 4 {
        if magic[..4] == MAGIC_APRN {
            return "APRN (aprender v1)";
        }
        if magic[..4] == MAGIC_APR1 {
            return "APR1 (whisper.apr)";
        }
        if magic[..4] == MAGIC_APR2 {
            return "APR2 (aprender v2)";
        }
        if magic[..4] == MAGIC_APR0 {
            return "APR v2 (ONE TRUE format)";
        }
        if magic[..4] == MAGIC_GGUF {
            return "GGUF (llama.cpp)";
        }
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
#[allow(dead_code)]
pub(crate) fn success(msg: &str) {
    println!("{} {}", "[PASS]".green().bold(), msg);
}

/// Print a warning message
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) fn header(title: &str) {
    let bar = "━".repeat(60);
    println!("\n{}", bar.cyan());
    println!("  {}", title.cyan().bold());
    println!("{}", bar.cyan());
}

/// Print a subsection header with lighter line drawing
#[allow(dead_code)]
pub(crate) fn subheader(title: &str) {
    println!("\n  {} {}", "───".dimmed(), title.cyan());
}

// ── Status Badges (Unicode + color) ──

/// Format a pass badge: "✓ label" in green bold
#[allow(dead_code)]
pub(crate) fn badge_pass(label: &str) -> String {
    format!("{} {}", "✓".green().bold(), label.green().bold())
}

/// Format a fail badge: "✗ label" in red bold
#[allow(dead_code)]
pub(crate) fn badge_fail(label: &str) -> String {
    format!("{} {}", "✗".red().bold(), label.red().bold())
}

/// Format a warning badge: "⚠ label" in yellow bold
#[allow(dead_code)]
pub(crate) fn badge_warn(label: &str) -> String {
    format!("{} {}", "⚠".yellow().bold(), label.yellow().bold())
}

/// Format a skip badge: "○ label" in dimmed
#[allow(dead_code)]
pub(crate) fn badge_skip(label: &str) -> String {
    format!("{} {}", "○".dimmed(), label.dimmed())
}

/// Format an info badge: "ℹ label" in blue
#[allow(dead_code)]
pub(crate) fn badge_info(label: &str) -> String {
    format!("{} {}", "ℹ".blue(), label.blue())
}

// ── Metrics Display ──

/// Print a metric with label, value, and unit
#[allow(dead_code)]
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
#[allow(dead_code)]
pub(crate) fn progress_bar(current: usize, total: usize, width: usize) -> String {
    if total == 0 {
        return format!("[{}]", " ".repeat(width));
    }
    let filled = (current * width) / total;
    let empty = width.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

/// Format a duration in human-readable form
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
mod tests {
    use super::*;

    // ==================== Magic Byte Detection ====================

    #[test]
    fn test_is_valid_magic_aprn() {
        assert!(is_valid_magic(&MAGIC_APRN));
    }

    #[test]
    fn test_is_valid_magic_apr1() {
        assert!(is_valid_magic(&MAGIC_APR1));
    }

    #[test]
    fn test_is_valid_magic_apr2() {
        assert!(is_valid_magic(&MAGIC_APR2));
    }

    #[test]
    fn test_is_valid_magic_apr0() {
        assert!(is_valid_magic(&MAGIC_APR0));
    }

    #[test]
    fn test_is_valid_magic_gguf() {
        assert!(is_valid_magic(&MAGIC_GGUF));
    }

    #[test]
    fn test_is_valid_magic_unknown() {
        assert!(!is_valid_magic(&[0x00, 0x00, 0x00, 0x00]));
    }

    #[test]
    fn test_is_valid_magic_too_short() {
        assert!(!is_valid_magic(&[0x41, 0x50]));
    }

    #[test]
    fn test_is_valid_magic_empty() {
        assert!(!is_valid_magic(&[]));
    }

    #[test]
    fn test_is_valid_magic_extra_bytes_ignored() {
        let mut bytes = MAGIC_GGUF.to_vec();
        bytes.extend_from_slice(&[0xFF, 0xFF]);
        assert!(is_valid_magic(&bytes));
    }

    // ==================== Format Name ====================

    #[test]
    fn test_format_name_aprn() {
        assert_eq!(format_name(&MAGIC_APRN), "APRN (aprender v1)");
    }

    #[test]
    fn test_format_name_apr1() {
        assert_eq!(format_name(&MAGIC_APR1), "APR1 (whisper.apr)");
    }

    #[test]
    fn test_format_name_apr2() {
        assert_eq!(format_name(&MAGIC_APR2), "APR2 (aprender v2)");
    }

    #[test]
    fn test_format_name_apr0() {
        assert_eq!(format_name(&MAGIC_APR0), "APR v2 (ONE TRUE format)");
    }

    #[test]
    fn test_format_name_gguf() {
        assert_eq!(format_name(&MAGIC_GGUF), "GGUF (llama.cpp)");
    }

    #[test]
    fn test_format_name_unknown() {
        assert_eq!(format_name(&[0x00, 0x00, 0x00, 0x00]), "Unknown");
    }

    #[test]
    fn test_format_name_too_short() {
        assert_eq!(format_name(&[0x41]), "Unknown");
    }

    #[test]
    fn test_format_name_empty() {
        assert_eq!(format_name(&[]), "Unknown");
    }

    // ==================== Output Formatting (no-panic) ====================

    #[test]
    fn test_section_does_not_panic() {
        section("Test Section");
    }

    #[test]
    fn test_kv_does_not_panic() {
        kv("key", "value");
    }

    #[test]
    fn test_kv_with_number() {
        kv("count", 42);
    }

    #[test]
    fn test_success_does_not_panic() {
        success("operation completed");
    }

    #[test]
    fn test_warning_does_not_panic() {
        warning("something may be wrong");
    }

    #[test]
    fn test_fail_does_not_panic() {
        fail("operation failed");
    }

    #[test]
    fn test_info_does_not_panic() {
        info("informational message");
    }

    #[test]
    fn test_error_does_not_panic() {
        error("error message");
    }

    #[test]
    fn test_warn_alias_does_not_panic() {
        warn("warning via alias");
    }

    // ==================== Format Size ====================

    #[test]
    fn test_format_size_zero() {
        let s = format_size(0);
        assert!(s.contains('0'));
    }

    #[test]
    fn test_format_size_bytes() {
        let s = format_size(512);
        assert!(s.contains("512"));
    }

    #[test]
    fn test_format_size_kib() {
        let s = format_size(1024);
        assert!(s.contains("KiB") || s.contains("1"));
    }

    #[test]
    fn test_format_size_mib() {
        let s = format_size(1024 * 1024);
        assert!(s.contains("MiB") || s.contains("1"));
    }

    #[test]
    fn test_format_size_gib() {
        let s = format_size(1024 * 1024 * 1024);
        assert!(s.contains("GiB") || s.contains("1"));
    }

    // ==================== Magic Byte Constants ====================

    #[test]
    fn test_magic_constants_are_4_bytes() {
        assert_eq!(MAGIC_APRN.len(), 4);
        assert_eq!(MAGIC_APR1.len(), 4);
        assert_eq!(MAGIC_APR2.len(), 4);
        assert_eq!(MAGIC_APR0.len(), 4);
        assert_eq!(MAGIC_GGUF.len(), 4);
    }

    #[test]
    fn test_magic_aprn_is_ascii_aprn() {
        assert_eq!(&MAGIC_APRN, b"APRN");
    }

    #[test]
    fn test_magic_apr1_is_ascii_apr1() {
        assert_eq!(&MAGIC_APR1, b"APR1");
    }

    #[test]
    fn test_magic_apr2_is_ascii_apr2() {
        assert_eq!(&MAGIC_APR2, b"APR2");
    }

    #[test]
    fn test_magic_gguf_is_ascii_gguf() {
        assert_eq!(&MAGIC_GGUF, b"GGUF");
    }

    // ==================== Rich Output Primitives ====================

    #[test]
    fn test_header_does_not_panic() {
        header("Test Header");
    }

    #[test]
    fn test_subheader_does_not_panic() {
        subheader("Test Subheader");
    }

    #[test]
    fn test_badge_pass() {
        let s = badge_pass("OK");
        assert!(s.contains("OK"));
    }

    #[test]
    fn test_badge_fail() {
        let s = badge_fail("ERROR");
        assert!(s.contains("ERROR"));
    }

    #[test]
    fn test_badge_warn() {
        let s = badge_warn("WARNING");
        assert!(s.contains("WARNING"));
    }

    #[test]
    fn test_badge_skip() {
        let s = badge_skip("SKIPPED");
        assert!(s.contains("SKIPPED"));
    }

    #[test]
    fn test_badge_info() {
        let s = badge_info("INFO");
        assert!(s.contains("INFO"));
    }

    #[test]
    fn test_metric_does_not_panic() {
        metric("Throughput", 42.5, "tok/s");
    }

    #[test]
    fn test_metric_empty_unit() {
        metric("Count", 100, "");
    }

    #[test]
    fn test_progress_bar_full() {
        let bar = progress_bar(10, 10, 20);
        assert_eq!(bar.chars().count(), 22); // 20 fill chars + 2 brackets
        assert!(bar.contains("█"));
    }

    #[test]
    fn test_progress_bar_empty() {
        let bar = progress_bar(0, 10, 20);
        assert!(bar.contains("░"));
        assert!(!bar.contains("█"));
    }

    #[test]
    fn test_progress_bar_half() {
        let bar = progress_bar(5, 10, 20);
        assert!(bar.contains("█"));
        assert!(bar.contains("░"));
    }

    #[test]
    fn test_progress_bar_zero_total() {
        let bar = progress_bar(0, 0, 10);
        assert_eq!(bar, "[          ]");
    }

    #[test]
    fn test_duration_fmt_milliseconds() {
        assert_eq!(duration_fmt(42), "42ms");
        assert_eq!(duration_fmt(999), "999ms");
    }

    #[test]
    fn test_duration_fmt_seconds() {
        assert_eq!(duration_fmt(1_000), "1.0s");
        assert_eq!(duration_fmt(1_500), "1.5s");
        assert_eq!(duration_fmt(59_999), "60.0s");
    }

    #[test]
    fn test_duration_fmt_minutes() {
        assert_eq!(duration_fmt(60_000), "1.0m");
        assert_eq!(duration_fmt(90_000), "1.5m");
    }

    #[test]
    fn test_count_fmt_small() {
        assert_eq!(count_fmt(0), "0");
        assert_eq!(count_fmt(42), "42");
        assert_eq!(count_fmt(999), "999");
    }

    #[test]
    fn test_count_fmt_thousands() {
        assert_eq!(count_fmt(1_000), "1,000");
        assert_eq!(count_fmt(1_234), "1,234");
        assert_eq!(count_fmt(12_345), "12,345");
        assert_eq!(count_fmt(123_456), "123,456");
        assert_eq!(count_fmt(1_234_567), "1,234,567");
    }

    #[test]
    fn test_dim_does_not_panic() {
        let _ = dim("text");
    }

    #[test]
    fn test_highlight_does_not_panic() {
        let _ = highlight("text");
    }

    #[test]
    fn test_filepath_does_not_panic() {
        let _ = filepath(std::path::Path::new("/tmp/model.apr"));
    }

    #[test]
    fn test_dtype_color_f32() {
        let s = dtype_color("F32");
        assert!(s.to_string().contains("F32"));
    }

    #[test]
    fn test_dtype_color_f16() {
        let s = dtype_color("F16");
        assert!(s.to_string().contains("F16"));
    }

    #[test]
    fn test_dtype_color_q4k() {
        let s = dtype_color("Q4_K");
        assert!(s.to_string().contains("Q4_K"));
    }

    #[test]
    fn test_dtype_color_q6k() {
        let s = dtype_color("Q6_K");
        assert!(s.to_string().contains("Q6_K"));
    }

    #[test]
    fn test_dtype_color_q8_0() {
        let s = dtype_color("Q8_0");
        assert!(s.to_string().contains("Q8_0"));
    }

    #[test]
    fn test_dtype_color_unknown() {
        let s = dtype_color("INT8");
        assert!(s.to_string().contains("INT8"));
    }

    #[test]
    fn test_grade_color_all_grades() {
        for grade in &["A+", "A", "B+", "B", "C+", "C", "D", "F"] {
            let s = grade_color(grade);
            assert!(s.to_string().contains(grade));
        }
    }

    #[test]
    fn test_table_empty_rows() {
        let result = table(&["A", "B"], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_table_with_data() {
        let rows = vec![
            vec!["hello".to_string(), "world".to_string()],
            vec!["foo".to_string(), "bar".to_string()],
        ];
        let result = table(&["Col1", "Col2"], &rows);
        assert!(result.contains("Col1"));
        assert!(result.contains("hello"));
        assert!(result.contains("bar"));
    }

    #[test]
    fn test_kv_table_empty() {
        let result = kv_table(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_kv_table_with_data() {
        let pairs = vec![
            ("Format", "APR v2".to_string()),
            ("Size", "4.2 GiB".to_string()),
        ];
        let result = kv_table(&pairs);
        assert!(result.contains("Format"));
        assert!(result.contains("APR v2"));
        assert!(result.contains("Size"));
    }

    #[test]
    fn test_pipeline_stage_all_statuses() {
        pipeline_stage("Load", StageStatus::Pending);
        pipeline_stage("Parse", StageStatus::Running);
        pipeline_stage("Validate", StageStatus::Done);
        pipeline_stage("Export", StageStatus::Failed);
        pipeline_stage("Optional", StageStatus::Skipped);
    }
}
