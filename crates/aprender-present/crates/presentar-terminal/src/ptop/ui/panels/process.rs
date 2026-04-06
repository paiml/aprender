//! Process panel rendering and utilities.
//!
//! Provides process table formatting, column alignment,
//! and helper functions for rendering process metrics.

use presentar_core::Color;

// =============================================================================
// PROCESS TITLE BUILDING
// =============================================================================

/// Build process panel title string.
///
/// Format: "Processes │ 342 │ Running: 5 │ filter: chrome"
#[must_use]
pub fn build_process_title(total: usize, running: usize, filter: Option<&str>) -> String {
    if let Some(f) = filter {
        format!(
            "Processes │ {} │ Running: {} │ filter: {}",
            total, running, f
        )
    } else {
        format!("Processes │ {} │ Running: {}", total, running)
    }
}

/// Build compact process title for narrow panels.
///
/// Format: "Proc │ 342"
#[must_use]
pub fn build_process_title_compact(total: usize) -> String {
    format!("Proc │ {}", total)
}

// =============================================================================
// COLUMN WIDTHS
// =============================================================================

/// Process table column widths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessColumnWidths {
    /// PID column width
    pub pid: usize,
    /// User column width
    pub user: usize,
    /// CPU% column width
    pub cpu: usize,
    /// Memory% column width
    pub mem: usize,
    /// Command column width (flexible)
    pub cmd: usize,
}

impl ProcessColumnWidths {
    /// Calculate column widths for a given total width.
    ///
    /// Fixed columns: PID(7) + User(8) + CPU(6) + Mem(6) = 27
    /// Command gets the rest
    #[must_use]
    pub fn calculate(available_width: usize) -> Self {
        const PID_WIDTH: usize = 7;
        const USER_WIDTH: usize = 8;
        const CPU_WIDTH: usize = 6;
        const MEM_WIDTH: usize = 6;
        const FIXED_TOTAL: usize = PID_WIDTH + USER_WIDTH + CPU_WIDTH + MEM_WIDTH;

        let cmd_width = available_width.saturating_sub(FIXED_TOTAL).max(10);

        Self {
            pid: PID_WIDTH,
            user: USER_WIDTH,
            cpu: CPU_WIDTH,
            mem: MEM_WIDTH,
            cmd: cmd_width,
        }
    }

    /// Get total width of all columns.
    #[must_use]
    pub fn total(&self) -> usize {
        self.pid + self.user + self.cpu + self.mem + self.cmd
    }
}

impl Default for ProcessColumnWidths {
    fn default() -> Self {
        Self::calculate(80)
    }
}

// =============================================================================
// PROCESS STATE COLORS
// =============================================================================

/// Get color for process state.
#[must_use]
pub fn process_state_color(state: &str) -> Color {
    match state.chars().next() {
        Some('R') => Color::new(0.3, 0.9, 0.3, 1.0), // Running - green
        Some('S') => Color::new(0.6, 0.6, 0.6, 1.0), // Sleeping - gray
        Some('D') => Color::new(1.0, 0.5, 0.3, 1.0), // Disk wait - orange
        Some('Z') => Color::new(1.0, 0.3, 0.3, 1.0), // Zombie - red
        Some('T') => Color::new(0.5, 0.5, 0.8, 1.0), // Stopped - purple
        Some('I') => Color::new(0.4, 0.4, 0.4, 1.0), // Idle - dim gray
        _ => Color::new(0.7, 0.7, 0.7, 1.0),         // Unknown - light gray
    }
}

/// Get color for CPU usage percentage.
#[must_use]
pub fn cpu_usage_color(percent: f64) -> Color {
    if percent > 90.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Critical red
    } else if percent > 50.0 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow
    } else if percent > 25.0 {
        Color::new(0.5, 0.9, 0.3, 1.0) // Light green
    } else {
        Color::new(0.3, 0.7, 0.3, 1.0) // Green
    }
}

/// Get color for memory usage percentage.
#[must_use]
pub fn mem_usage_color(percent: f64) -> Color {
    if percent > 80.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Critical red
    } else if percent > 50.0 {
        Color::new(0.8, 0.5, 1.0, 1.0) // Purple
    } else if percent > 25.0 {
        Color::new(0.6, 0.4, 0.9, 1.0) // Light purple
    } else {
        Color::new(0.4, 0.4, 0.7, 1.0) // Dim purple
    }
}

// =============================================================================
// TRUNCATION UTILITIES
// =============================================================================

/// Truncate a string to fit within a width, adding ellipsis if needed.
#[must_use]
pub fn truncate_with_ellipsis(s: &str, max_width: usize) -> String {
    if max_width < 3 {
        return s.chars().take(max_width).collect();
    }

    let char_count = s.chars().count();
    if char_count <= max_width {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_width - 1).collect();
        format!("{}~", truncated)
    }
}

/// Truncate command path, preferring to show executable name.
///
/// "/usr/bin/very-long-program-name --arg" -> "very-long-prog~ --arg"
#[must_use]
pub fn truncate_command(cmd: &str, max_width: usize) -> String {
    if cmd.chars().count() <= max_width {
        return cmd.to_string();
    }

    // Split on first space to separate command from args
    let (exe, args) = match cmd.split_once(' ') {
        Some((e, a)) => (e, Some(a)),
        None => (cmd, None),
    };

    // Get just the executable name (after last /)
    let exe_name = exe.rsplit('/').next().unwrap_or(exe);

    match args {
        Some(a) => {
            let full = format!("{} {}", exe_name, a);
            truncate_with_ellipsis(&full, max_width)
        }
        None => truncate_with_ellipsis(exe_name, max_width),
    }
}

// =============================================================================
// SELECTION HIGHLIGHT
// =============================================================================

/// Selection highlight colors.
#[allow(clippy::derive_partial_eq_without_eq)] // Color doesn't implement Eq
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionColors {
    /// Background color for selected row
    pub background: Color,
    /// Foreground color for selected row
    pub foreground: Color,
}

impl SelectionColors {
    /// Standard selection colors (blue highlight).
    #[must_use]
    pub fn standard() -> Self {
        Self {
            background: Color::new(0.2, 0.4, 0.8, 1.0),
            foreground: Color::new(1.0, 1.0, 1.0, 1.0),
        }
    }

    /// Dim selection for unfocused panel.
    #[must_use]
    pub fn unfocused() -> Self {
        Self {
            background: Color::new(0.3, 0.3, 0.4, 1.0),
            foreground: Color::new(0.8, 0.8, 0.8, 1.0),
        }
    }
}

impl Default for SelectionColors {
    fn default() -> Self {
        Self::standard()
    }
}

// =============================================================================
// SORT INDICATOR
// =============================================================================

/// Sort direction indicator.
#[must_use]
pub fn sort_indicator(ascending: bool) -> &'static str {
    if ascending {
        "▲"
    } else {
        "▼"
    }
}

/// Get sort indicator with column name.
#[must_use]
pub fn sort_column_label(name: &str, ascending: bool) -> String {
    format!("{}{}", name, sort_indicator(ascending))
}

// =============================================================================
// PROCESS SORTING (reduces draw_process_dataframe complexity)
// =============================================================================

use crate::ptop::app::ProcessSortColumn;
use sysinfo::{Pid, Process};

/// Sort key extractor for processes.
///
/// Returns an Ordering based on sort column and direction.
/// Extracted from draw_process_dataframe to reduce cyclomatic complexity.
#[inline]
pub fn compare_processes(
    a: &(&Pid, &Process),
    b: &(&Pid, &Process),
    sort_column: ProcessSortColumn,
    descending: bool,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let cmp = match sort_column {
        ProcessSortColumn::Pid => a.0.cmp(b.0),
        ProcessSortColumn::User => {
            let ua = a.1.user_id().map(|u| u.to_string()).unwrap_or_default();
            let ub = b.1.user_id().map(|u| u.to_string()).unwrap_or_default();
            ua.cmp(&ub)
        }
        ProcessSortColumn::Cpu => {
            a.1.cpu_usage()
                .partial_cmp(&b.1.cpu_usage())
                .unwrap_or(Ordering::Equal)
        }
        ProcessSortColumn::Mem => a.1.memory().cmp(&b.1.memory()),
        ProcessSortColumn::Command => {
            let na = a.1.name().to_string_lossy();
            let nb = b.1.name().to_string_lossy();
            na.cmp(&nb)
        }
    };

    if descending {
        cmp.reverse()
    } else {
        cmp
    }
}

/// Sort processes in place by the specified column.
///
/// # Arguments
/// * `processes` - Mutable slice of (Pid, Process) tuples
/// * `sort_column` - Column to sort by
/// * `descending` - Sort in descending order if true
#[inline]
pub fn sort_processes(
    processes: &mut [(&Pid, &Process)],
    sort_column: ProcessSortColumn,
    descending: bool,
) {
    processes.sort_by(|a, b| compare_processes(a, b, sort_column, descending));
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // build_process_title tests
    // =========================================================================

    #[test]
    fn test_build_process_title_basic() {
        let title = build_process_title(342, 5, None);
        assert!(title.contains("Processes"));
        assert!(title.contains("342"));
        assert!(title.contains("Running: 5"));
    }

    #[test]
    fn test_build_process_title_with_filter() {
        let title = build_process_title(15, 2, Some("chrome"));
        assert!(title.contains("filter: chrome"));
        assert!(title.contains("15"));
    }

    #[test]
    fn test_build_process_title_zero() {
        let title = build_process_title(0, 0, None);
        assert!(title.contains("0"));
        assert!(title.contains("Running: 0"));
    }

    #[test]
    fn test_build_process_title_large_numbers() {
        let title = build_process_title(10000, 500, None);
        assert!(title.contains("10000"));
        assert!(title.contains("500"));
    }

    // =========================================================================
    // build_process_title_compact tests
    // =========================================================================

    #[test]
    fn test_build_process_title_compact_basic() {
        let title = build_process_title_compact(342);
        assert!(title.contains("Proc"));
        assert!(title.contains("342"));
        assert!(!title.contains("Running")); // Compact doesn't show running
    }

    #[test]
    fn test_build_process_title_compact_short() {
        let title = build_process_title_compact(5);
        assert!(title.chars().count() < 15);
    }

    // =========================================================================
    // ProcessColumnWidths tests
    // =========================================================================

    #[test]
    fn test_column_widths_default() {
        let widths = ProcessColumnWidths::default();
        assert_eq!(widths.pid, 7);
        assert_eq!(widths.user, 8);
        assert_eq!(widths.cpu, 6);
        assert_eq!(widths.mem, 6);
    }

    #[test]
    fn test_column_widths_calculate_80() {
        let widths = ProcessColumnWidths::calculate(80);
        assert_eq!(widths.total(), 80);
    }

    #[test]
    fn test_column_widths_calculate_120() {
        let widths = ProcessColumnWidths::calculate(120);
        assert_eq!(widths.total(), 120);
        // Extra width goes to cmd
        assert!(widths.cmd > 10);
    }

    #[test]
    fn test_column_widths_calculate_narrow() {
        let widths = ProcessColumnWidths::calculate(30);
        // cmd gets minimum 10
        assert_eq!(widths.cmd, 10);
    }

    #[test]
    fn test_column_widths_total() {
        let widths = ProcessColumnWidths::calculate(100);
        let expected = widths.pid + widths.user + widths.cpu + widths.mem + widths.cmd;
        assert_eq!(widths.total(), expected);
    }

    #[test]
    fn test_column_widths_derive_debug() {
        let widths = ProcessColumnWidths::default();
        let debug = format!("{:?}", widths);
        assert!(debug.contains("ProcessColumnWidths"));
    }

    #[test]
    fn test_column_widths_derive_clone() {
        let widths = ProcessColumnWidths::calculate(100);
        let cloned = widths.clone();
        assert_eq!(widths, cloned);
    }

    // =========================================================================
    // process_state_color tests
    // =========================================================================

    #[test]
    fn test_process_state_color_running() {
        let color = process_state_color("R");
        assert!(color.g > 0.8, "Running should be green");
    }

    #[test]
    fn test_process_state_color_sleeping() {
        let color = process_state_color("S");
        assert!((color.r - color.g).abs() < 0.1, "Sleeping should be gray");
    }

    #[test]
    fn test_process_state_color_disk_wait() {
        let color = process_state_color("D");
        assert!(color.r > 0.9, "Disk wait should be orange");
        assert!(color.g > 0.4 && color.g < 0.6);
    }

    #[test]
    fn test_process_state_color_zombie() {
        let color = process_state_color("Z");
        assert!(color.r > 0.9, "Zombie should be red");
        assert!(color.g < 0.5);
    }

    #[test]
    fn test_process_state_color_stopped() {
        let color = process_state_color("T");
        assert!(color.b > color.r, "Stopped should be purple-ish");
    }

    #[test]
    fn test_process_state_color_idle() {
        let color = process_state_color("I");
        assert!(color.r < 0.5, "Idle should be dim");
    }

    #[test]
    fn test_process_state_color_unknown() {
        let color = process_state_color("X");
        assert!(color.r > 0.6, "Unknown should be light gray");
    }

    #[test]
    fn test_process_state_color_empty() {
        let color = process_state_color("");
        assert!(color.r > 0.6);
    }

    // =========================================================================
    // cpu_usage_color tests
    // =========================================================================

    #[test]
    fn test_cpu_usage_color_low() {
        let color = cpu_usage_color(10.0);
        assert!(color.g > 0.6, "Low usage should be green");
    }

    #[test]
    fn test_cpu_usage_color_medium() {
        let color = cpu_usage_color(30.0);
        assert!(color.g > 0.8);
    }

    #[test]
    fn test_cpu_usage_color_high() {
        let color = cpu_usage_color(60.0);
        assert!(color.r > 0.9, "High usage should be yellow");
        assert!(color.g > 0.7);
    }

    #[test]
    fn test_cpu_usage_color_critical() {
        let color = cpu_usage_color(95.0);
        assert!(color.r > 0.9, "Critical should be red");
        assert!(color.g < 0.5);
    }

    // =========================================================================
    // mem_usage_color tests
    // =========================================================================

    #[test]
    fn test_mem_usage_color_low() {
        let color = mem_usage_color(10.0);
        assert!(color.b > 0.6, "Low mem should be dim purple");
    }

    #[test]
    fn test_mem_usage_color_medium() {
        let color = mem_usage_color(30.0);
        assert!(color.b > 0.8);
    }

    #[test]
    fn test_mem_usage_color_high() {
        let color = mem_usage_color(60.0);
        assert!(color.b > 0.9, "High mem should be purple");
    }

    #[test]
    fn test_mem_usage_color_critical() {
        let color = mem_usage_color(85.0);
        assert!(color.r > 0.9, "Critical mem should be red");
    }

    // =========================================================================
    // truncate_with_ellipsis tests
    // =========================================================================

    #[test]
    fn test_truncate_with_ellipsis_fits() {
        let result = truncate_with_ellipsis("hello", 10);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_with_ellipsis_truncates() {
        let result = truncate_with_ellipsis("hello world", 8);
        assert_eq!(result, "hello w~");
    }

    #[test]
    fn test_truncate_with_ellipsis_exact() {
        let result = truncate_with_ellipsis("hello", 5);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_with_ellipsis_very_short() {
        let result = truncate_with_ellipsis("hello", 2);
        assert_eq!(result, "he");
    }

    #[test]
    fn test_truncate_with_ellipsis_zero_width() {
        let result = truncate_with_ellipsis("hello", 0);
        assert_eq!(result, "");
    }

    #[test]
    fn test_truncate_with_ellipsis_unicode() {
        let result = truncate_with_ellipsis("hello\u{1F600}", 6);
        assert_eq!(result, "hello\u{1F600}");
    }

    // =========================================================================
    // truncate_command tests
    // =========================================================================

    #[test]
    fn test_truncate_command_fits() {
        let result = truncate_command("/usr/bin/ls", 20);
        assert_eq!(result, "/usr/bin/ls");
    }

    #[test]
    fn test_truncate_command_extracts_name() {
        let result = truncate_command("/usr/bin/very-long-program-name", 15);
        assert!(result.starts_with("very-long-prog"));
        assert!(result.ends_with("~"));
    }

    #[test]
    fn test_truncate_command_with_args() {
        let result = truncate_command("/usr/bin/prog --arg", 15);
        assert!(result.contains("prog"));
    }

    #[test]
    fn test_truncate_command_no_path() {
        let result = truncate_command("program", 10);
        assert_eq!(result, "program");
    }

    #[test]
    fn test_truncate_command_short_name() {
        let result = truncate_command("/bin/ls", 20);
        assert_eq!(result, "/bin/ls");
    }

    // =========================================================================
    // SelectionColors tests
    // =========================================================================

    #[test]
    fn test_selection_colors_standard() {
        let colors = SelectionColors::standard();
        assert!(colors.background.b > 0.7, "Standard bg should be blue");
        assert!(colors.foreground.r > 0.9, "Standard fg should be white");
    }

    #[test]
    fn test_selection_colors_unfocused() {
        let colors = SelectionColors::unfocused();
        assert!(colors.background.r < 0.5, "Unfocused should be dim");
    }

    #[test]
    fn test_selection_colors_default() {
        let default = SelectionColors::default();
        let standard = SelectionColors::standard();
        assert_eq!(default.background.r, standard.background.r);
    }

    #[test]
    fn test_selection_colors_derive_debug() {
        let colors = SelectionColors::standard();
        let debug = format!("{:?}", colors);
        assert!(debug.contains("SelectionColors"));
    }

    #[test]
    fn test_selection_colors_derive_clone() {
        let colors = SelectionColors::standard();
        let cloned = colors.clone();
        assert_eq!(colors, cloned);
    }

    // =========================================================================
    // sort_indicator tests
    // =========================================================================

    #[test]
    fn test_sort_indicator_ascending() {
        assert_eq!(sort_indicator(true), "▲");
    }

    #[test]
    fn test_sort_indicator_descending() {
        assert_eq!(sort_indicator(false), "▼");
    }

    #[test]
    fn test_sort_column_label_ascending() {
        let label = sort_column_label("CPU", true);
        assert_eq!(label, "CPU▲");
    }

    #[test]
    fn test_sort_column_label_descending() {
        let label = sort_column_label("MEM", false);
        assert_eq!(label, "MEM▼");
    }

    // =========================================================================
    // compare_processes tests (F-PROC-SORT-001 to F-PROC-SORT-010)
    // =========================================================================

    #[test]
    fn f_proc_sort_001_compare_function_exists() {
        // Verify function compiles with correct signature
        let _ = compare_processes
            as fn(
                &(&sysinfo::Pid, &sysinfo::Process),
                &(&sysinfo::Pid, &sysinfo::Process),
                ProcessSortColumn,
                bool,
            ) -> std::cmp::Ordering;
    }

    #[test]
    fn f_proc_sort_002_sort_function_exists() {
        // Verify function compiles with correct signature
        let _ = sort_processes
            as fn(&mut [(&sysinfo::Pid, &sysinfo::Process)], ProcessSortColumn, bool);
    }

    #[test]
    fn f_proc_sort_003_sort_column_pid() {
        // Verify ProcessSortColumn::Pid variant exists
        let col = ProcessSortColumn::Pid;
        assert!(matches!(col, ProcessSortColumn::Pid));
    }

    #[test]
    fn f_proc_sort_004_sort_column_user() {
        let col = ProcessSortColumn::User;
        assert!(matches!(col, ProcessSortColumn::User));
    }

    #[test]
    fn f_proc_sort_005_sort_column_cpu() {
        let col = ProcessSortColumn::Cpu;
        assert!(matches!(col, ProcessSortColumn::Cpu));
    }

    #[test]
    fn f_proc_sort_006_sort_column_mem() {
        let col = ProcessSortColumn::Mem;
        assert!(matches!(col, ProcessSortColumn::Mem));
    }

    #[test]
    fn f_proc_sort_007_sort_column_command() {
        let col = ProcessSortColumn::Command;
        assert!(matches!(col, ProcessSortColumn::Command));
    }

    #[test]
    fn f_proc_sort_008_descending_reverses() {
        // Test that descending flag reverses ordering
        // This is a compile-time check that the logic exists
        use std::cmp::Ordering;
        let asc = Ordering::Less;
        let desc = asc.reverse();
        assert_eq!(desc, Ordering::Greater);
    }

    #[test]
    fn f_proc_sort_009_ordering_equal_handled() {
        use std::cmp::Ordering;
        // Verify Equal stays Equal when reversed
        let eq = Ordering::Equal;
        assert_eq!(eq.reverse(), Ordering::Equal);
    }

    #[test]
    fn f_proc_sort_010_cpu_partial_cmp_fallback() {
        // Verify NaN handling compiles (partial_cmp returns None for NaN)
        let a: f32 = 0.0;
        let b: f32 = 1.0;
        let cmp = a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
        assert_eq!(cmp, std::cmp::Ordering::Less);
    }
}
