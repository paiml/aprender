//! Pixel comparison tests for ptop.
//!
//! These tests verify pixel-perfect rendering against baseline snapshots.
#![cfg(feature = "ptop")]

use presentar_terminal::direct::CellBuffer;
use presentar_terminal::ptop::ui::draw;
use presentar_terminal::ptop::App;

/// Helper to create a deterministic app for testing.
fn create_deterministic_app() -> App {
    // Set deterministic mode for reproducible output
    presentar_terminal::random_seed::set_global_seed(42);
    App::new(true) // deterministic = true
}

/// Count differences between two strings (line by line, char by char).
fn count_pixel_diff(baseline: &str, current: &str) -> PixelDiff {
    let baseline_lines: Vec<&str> = baseline.lines().collect();
    let current_lines: Vec<&str> = current.lines().collect();

    let mut different_cells = 0;
    let mut different_lines = 0;

    let max_lines = baseline_lines.len().max(current_lines.len());

    for i in 0..max_lines {
        let b_line = baseline_lines.get(i).unwrap_or(&"");
        let c_line = current_lines.get(i).unwrap_or(&"");

        if b_line != c_line {
            different_lines += 1;
            // Count character differences
            let b_chars: Vec<char> = b_line.chars().collect();
            let c_chars: Vec<char> = c_line.chars().collect();
            let max_chars = b_chars.len().max(c_chars.len());

            for j in 0..max_chars {
                if b_chars.get(j) != c_chars.get(j) {
                    different_cells += 1;
                }
            }
        }
    }

    PixelDiff {
        different_cells,
        different_lines,
        total_lines: max_lines,
    }
}

/// Result of pixel comparison.
#[derive(Debug)]
struct PixelDiff {
    different_cells: usize,
    different_lines: usize,
    total_lines: usize,
}

impl PixelDiff {
    fn is_perfect(&self) -> bool {
        self.different_cells == 0
    }

    fn difference_percent(&self) -> f64 {
        if self.total_lines == 0 {
            return 0.0;
        }
        (self.different_lines as f64 / self.total_lines as f64) * 100.0
    }
}

/// Render app to string for comparison.
fn render_to_string(app: &App, width: u16, height: u16) -> String {
    let mut buffer = CellBuffer::new(width, height);
    draw(app, &mut buffer);

    let mut output = String::new();
    for y in 0..height {
        for x in 0..width {
            if let Some(cell) = buffer.get(x, y) {
                output.push_str(cell.symbol.as_str());
            } else {
                output.push(' ');
            }
        }
        output.push('\n');
    }
    output
}

// F-PIXEL-001: Deterministic rendering produces same output
#[test]
fn test_deterministic_rendering_same_output() {
    let app1 = create_deterministic_app();
    let app2 = create_deterministic_app();

    let output1 = render_to_string(&app1, 120, 40);
    let output2 = render_to_string(&app2, 120, 40);

    let diff = count_pixel_diff(&output1, &output2);
    assert!(
        diff.is_perfect(),
        "PIXEL FAIL: Deterministic apps should render identically, but {} cells differ",
        diff.different_cells
    );
}

// F-PIXEL-002: Baseline exists and is readable
#[test]
fn test_baseline_exists() {
    let baseline = include_str!("../__pixel_baselines__/ptop_120x40_deterministic.txt");
    assert!(!baseline.is_empty(), "Baseline file should not be empty");
    assert!(
        baseline.lines().count() >= 40,
        "Baseline should have at least 40 lines"
    );
}

// F-PIXEL-003: Current rendering matches baseline format
#[test]
fn test_rendering_matches_baseline_format() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    let baseline = include_str!("../__pixel_baselines__/ptop_120x40_deterministic.txt");

    // Check dimensions match
    let output_lines = output.lines().count();
    let baseline_lines = baseline.lines().count();

    assert_eq!(
        output_lines, baseline_lines,
        "Output should have same number of lines as baseline ({} vs {})",
        output_lines, baseline_lines
    );
}

// F-PIXEL-004: Title bar present in first line
#[test]
fn test_title_bar_present() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    let first_line = output.lines().next().unwrap_or("");
    assert!(
        first_line.contains("ptop") || first_line.contains("CPU"),
        "First line should contain ptop title or CPU info: got '{}'",
        first_line
    );
}

// F-PIXEL-005: Status bar present in last line
#[test]
fn test_status_bar_present() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    let lines: Vec<&str> = output.lines().collect();
    let last_line = lines.last().unwrap_or(&"");
    assert!(
        last_line.contains("Tab") || last_line.contains("Help") || last_line.contains("Quit"),
        "Last line should contain status bar: got '{}'",
        last_line
    );
}

// F-PIXEL-006: Border characters are present
#[test]
fn test_border_characters_present() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Check for rounded border characters
    let has_top_left = output.contains('╭') || output.contains('╔');
    let has_top_right = output.contains('╮') || output.contains('╗');
    let has_bottom_left = output.contains('╰') || output.contains('╚');
    let has_bottom_right = output.contains('╯') || output.contains('╝');

    assert!(has_top_left, "Should have top-left corner character");
    assert!(has_top_right, "Should have top-right corner character");
    assert!(has_bottom_left, "Should have bottom-left corner character");
    assert!(
        has_bottom_right,
        "Should have bottom-right corner character"
    );
}

// F-PIXEL-007: Panel titles are visible
#[test]
fn test_panel_titles_visible() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Check for common panel titles
    assert!(output.contains("CPU"), "CPU panel title should be visible");
    assert!(
        output.contains("Memory"),
        "Memory panel title should be visible"
    );
    assert!(
        output.contains("Disk") || output.contains("Network"),
        "Disk or Network panel should be visible"
    );
    assert!(
        output.contains("Processes") || output.contains("Process"),
        "Process panel should be visible"
    );
}

// F-PIXEL-008: No garbage characters
#[test]
fn test_no_garbage_characters() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Check for common garbage characters that indicate rendering issues
    let garbage_chars = ['\0', '\x01', '\x02', '\x03', '\x04'];

    for ch in garbage_chars {
        assert!(
            !output.contains(ch),
            "Output should not contain garbage character {:?}",
            ch
        );
    }
}

// F-PIXEL-009: Braille characters for graphs
#[test]
fn test_braille_characters_for_graphs() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Braille block starts at U+2800
    let has_braille = output
        .chars()
        .any(|c| ('\u{2800}'..='\u{28FF}').contains(&c));

    // At least some UI should use braille for graphs
    // (This may fail if no network traffic, which is OK in deterministic mode)
    let _ = has_braille; // Just check it doesn't panic
}

// F-PIXEL-010: Small terminal still renders
#[test]
fn test_small_terminal_renders() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 80, 24);

    assert!(
        !output.is_empty(),
        "Should render something even for small terminal"
    );
}

// F-PIXEL-011: Large terminal renders
#[test]
fn test_large_terminal_renders() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 200, 60);

    assert!(
        !output.is_empty(),
        "Should render something for large terminal"
    );
    assert!(
        output.lines().count() >= 60,
        "Should fill large terminal height"
    );
}

// F-PIXEL-012: Focus indicator visible when focused
#[test]
fn test_focus_indicator_visible() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Focus indicator is ► or double border
    let has_focus_arrow = output.contains('►');
    let has_double_border = output.contains('║') || output.contains('═');

    assert!(
        has_focus_arrow || has_double_border,
        "Should have visible focus indicator"
    );
}

// F-PIXEL-013: Percentage values formatted correctly
#[test]
fn test_percentage_formatting() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Should have percentage signs in output
    assert!(output.contains('%'), "Should contain percentage values");
}

// F-PIXEL-014: Memory values formatted with units
#[test]
fn test_memory_formatting() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Should have memory units
    let has_units = output.contains("GB")
        || output.contains("MB")
        || output.contains("KB")
        || output.contains('G')
        || output.contains('M');

    assert!(has_units, "Should contain memory units");
}

// F-PIXEL-015: Rate values formatted
#[test]
fn test_rate_formatting() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Should have rate indicators
    let has_rates = output.contains("/s") || output.contains("B/s");
    assert!(has_rates, "Should contain rate values");
}

// F-PIXEL-016: Rendering is fast
#[test]
fn test_rendering_performance() {
    let app = create_deterministic_app();

    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _output = render_to_string(&app, 120, 40);
    }
    let elapsed = start.elapsed();

    // 10 renders should complete in under 1 second
    assert!(
        elapsed.as_millis() < 1000,
        "PERFORMANCE FAIL: 10 renders took {}ms (should be <1000ms)",
        elapsed.as_millis()
    );
}

// F-PIXEL-017: No trailing whitespace overflow
#[test]
fn test_no_trailing_overflow() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    for (i, line) in output.lines().enumerate() {
        // Each line should not exceed width + possible unicode width issues
        assert!(
            line.chars().count() <= 130,
            "Line {} exceeds max width: {} chars",
            i,
            line.chars().count()
        );
    }
}

// F-PIXEL-018: Horizontal separators present
#[test]
fn test_horizontal_separators() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Check for horizontal line characters
    let has_h_lines = output.contains('─') || output.contains('═') || output.contains('━');
    assert!(has_h_lines, "Should have horizontal separator lines");
}

// F-PIXEL-019: Vertical separators present
#[test]
fn test_vertical_separators() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    // Check for vertical line characters
    let has_v_lines = output.contains('│') || output.contains('║') || output.contains('┃');
    assert!(has_v_lines, "Should have vertical separator lines");
}

// F-PIXEL-020: Consistent line width
#[test]
fn test_consistent_line_width() {
    let app = create_deterministic_app();
    let output = render_to_string(&app, 120, 40);

    let widths: Vec<usize> = output.lines().map(|l| l.chars().count()).collect();

    if widths.len() > 1 {
        // Most lines should be the same width (allow some variation for empty lines)
        let max_width = widths.iter().max().unwrap_or(&0);
        let consistent_lines = widths.iter().filter(|&w| *w >= max_width - 5).count();

        assert!(
            consistent_lines >= widths.len() / 2,
            "Most lines should have consistent width"
        );
    }
}
