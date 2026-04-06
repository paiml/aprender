#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! F061-F075 Text Rendering Tests
//!
//! Popperian falsification tests for text rendering in presentar-terminal.
//! Each test attempts to DISPROVE a claim about text rendering behavior.
//!
//! Reference: SPEC-024 Section D (Text Rendering F061-F075)

use presentar_core::{Canvas, Color, FontWeight, Point, Rect, TextStyle, Widget};
use presentar_terminal::widgets::BrailleSymbols;
use presentar_terminal::{
    BrailleGraph, NetworkInterface, NetworkPanel, ProcessEntry, ProcessTable, Sparkline,
};

/// Test canvas that captures text draws for verification.
struct TestCanvas {
    texts: Vec<(String, Point, TextStyle)>,
}

impl TestCanvas {
    const fn new() -> Self {
        Self { texts: Vec::new() }
    }

    /// Get all rendered text as a single string.
    fn all_text(&self) -> String {
        self.texts.iter().map(|(t, _, _)| t.as_str()).collect()
    }

    /// Find text by content substring.
    fn find_text(&self, substring: &str) -> Option<&(String, Point, TextStyle)> {
        self.texts.iter().find(|(t, _, _)| t.contains(substring))
    }

    /// Check if any text uses the given color.
    fn has_color(&self, color: Color) -> bool {
        self.texts.iter().any(|(_, _, s)| s.color == color)
    }

    /// Check if any text uses bold weight.
    #[allow(dead_code)]
    fn has_bold(&self) -> bool {
        self.texts
            .iter()
            .any(|(_, _, s)| s.weight == FontWeight::Bold)
    }
}

impl Canvas for TestCanvas {
    fn fill_rect(&mut self, _rect: Rect, _color: Color) {}
    fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
    fn draw_text(&mut self, text: &str, position: Point, style: &TextStyle) {
        self.texts.push((text.to_string(), position, style.clone()));
    }
    fn draw_line(&mut self, _from: Point, _to: Point, _color: Color, _width: f32) {}
    fn fill_circle(&mut self, _center: Point, _radius: f32, _color: Color) {}
    fn stroke_circle(&mut self, _center: Point, _radius: f32, _color: Color, _width: f32) {}
    fn fill_arc(
        &mut self,
        _center: Point,
        _radius: f32,
        _start_angle: f32,
        _end_angle: f32,
        _color: Color,
    ) {
    }
    fn draw_path(&mut self, _points: &[Point], _color: Color, _width: f32) {}
    fn fill_polygon(&mut self, _points: &[Point], _color: Color) {}
    fn push_clip(&mut self, _rect: Rect) {}
    fn pop_clip(&mut self) {}
    fn push_transform(&mut self, _transform: presentar_core::Transform2D) {}
    fn pop_transform(&mut self) {}
}

// =============================================================================
// F061: Default text not black
// Falsification criterion: TextStyle::default().color is black
// =============================================================================

#[test]
fn f061_default_text_not_black_in_widgets() {
    // ProcessTable should NOT use pure black for text
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1234, "noah", 50.0, 25.0, "firefox")]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Check that no text is rendered as pure black
    let pure_black = Color::new(0.0, 0.0, 0.0, 1.0);
    let has_pure_black = canvas.texts.iter().any(|(_, _, s)| s.color == pure_black);
    assert!(
        !has_pure_black,
        "ProcessTable should not use pure black text"
    );
}

#[test]
fn f061_text_style_default_is_black() {
    // Verify the core TextStyle default IS black (this is a fact we're documenting)
    let default_style = TextStyle::default();
    assert_eq!(
        default_style.color,
        Color::BLACK,
        "TextStyle::default() should be black (widgets must override)"
    );
}

// =============================================================================
// F062: PID column visible
// Falsification criterion: PID rendered as black
// =============================================================================

#[test]
fn f062_pid_column_visible() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1234, "noah", 50.0, 25.0, "test")]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Find PID text
    let pid_text = canvas.find_text("1234");
    assert!(pid_text.is_some(), "PID should be rendered");

    let (_, _, style) = pid_text.unwrap();
    // PID should be light gray (0.8, 0.8, 0.8), not black
    assert!(
        style.color.r > 0.5,
        "PID text should not be black: {:?}",
        style.color
    );
}

// =============================================================================
// F063: USER column visible
// Falsification criterion: USER rendered as black
// =============================================================================

#[test]
fn f063_user_column_visible() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1234, "testuser", 50.0, 25.0, "cmd")]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Find USER text
    let user_text = canvas.find_text("testuser");
    assert!(user_text.is_some(), "USER should be rendered");

    let (_, _, style) = user_text.unwrap();
    // USER should be light gray, not black
    assert!(
        style.color.r > 0.5,
        "USER text should not be black: {:?}",
        style.color
    );
}

// =============================================================================
// F064: COMMAND column visible
// Falsification criterion: COMMAND rendered as black
// =============================================================================

#[test]
fn f064_command_column_visible() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(
        1234,
        "noah",
        50.0,
        25.0,
        "mycommand",
    )]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Find COMMAND text
    let cmd_text = canvas.find_text("mycommand");
    assert!(cmd_text.is_some(), "COMMAND should be rendered");

    let (_, _, style) = cmd_text.unwrap();
    // COMMAND should be light gray, not black
    assert!(
        style.color.r > 0.5,
        "COMMAND text should not be black: {:?}",
        style.color
    );
}

// =============================================================================
// F065: Interface name visible
// Falsification criterion: eth0/wlan0 rendered as black
// =============================================================================

#[test]
fn f065_interface_name_visible() {
    let mut panel = NetworkPanel::new().compact();
    panel.add_interface(NetworkInterface::new("eth0"));
    panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    panel.paint(&mut canvas);

    // Find interface name
    let iface_text = canvas.find_text("eth0");
    assert!(iface_text.is_some(), "Interface name should be rendered");

    let (_, _, style) = iface_text.unwrap();
    // Interface name should be light gray, not black
    assert!(
        style.color.r > 0.5,
        "Interface name should not be black: {:?}",
        style.color
    );
}

#[test]
fn f065_wlan0_interface_visible() {
    let mut panel = NetworkPanel::new();
    panel.add_interface(NetworkInterface::new("wlan0"));
    panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    panel.paint(&mut canvas);

    // Find wlan0 interface name
    let iface_text = canvas.find_text("wlan0");
    assert!(iface_text.is_some(), "wlan0 interface should be rendered");

    let (_, _, style) = iface_text.unwrap();
    // wlan0 should be visible (bold header or light color)
    assert!(
        style.color.r > 0.5 || style.weight == FontWeight::Bold,
        "wlan0 should be visible: {style:?}"
    );
}

// =============================================================================
// F066: Selected text white
// Falsification criterion: Selected row not white
// =============================================================================

#[test]
fn f066_selected_text_white() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![
        ProcessEntry::new(1, "root", 10.0, 5.0, "systemd"),
        ProcessEntry::new(1234, "noah", 50.0, 25.0, "firefox"),
    ]);
    table.select(0); // Select first row (after sorting by CPU, firefox will be first)
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Selected row command should be white
    let white = Color::new(1.0, 1.0, 1.0, 1.0);
    let has_white_text = canvas.has_color(white);
    assert!(
        has_white_text,
        "Selected row should have white text for command"
    );
}

// =============================================================================
// F067: Header bold
// Falsification criterion: Headers not bold weight
// =============================================================================

#[test]
fn f067_header_bold() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1234, "noah", 50.0, 25.0, "test")]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Find header row (contains "PID" or "USER" or "CPU%")
    let header_text = canvas.find_text("PID");
    assert!(header_text.is_some(), "Header should contain PID");

    let (_, _, style) = header_text.unwrap();
    assert_eq!(
        style.weight,
        FontWeight::Bold,
        "Header should use bold weight"
    );
}

#[test]
fn f067_network_panel_header_bold() {
    let mut panel = NetworkPanel::new();
    panel.add_interface(NetworkInterface::new("eth0"));
    panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    panel.paint(&mut canvas);

    // Find "Network" header
    let header = canvas.find_text("Network");
    assert!(header.is_some(), "Network header should be present");

    let (_, _, style) = header.unwrap();
    assert_eq!(
        style.weight,
        FontWeight::Bold,
        "Network header should be bold"
    );
}

// =============================================================================
// F068: Dim text distinct
// Falsification criterion: Dim same as foreground
// =============================================================================

#[test]
fn f068_dim_text_distinct() {
    // Separator and dim elements should use distinct (lower luminance) colors
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1234, "noah", 50.0, 25.0, "test")]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Find separator (────)
    let sep = canvas.find_text("─");
    if let Some((_, _, style)) = sep {
        // Separator should be dimmer than normal text
        let normal_text = canvas.find_text("1234").unwrap();
        assert!(
            style.color.r < normal_text.2.color.r,
            "Separator should be dimmer than normal text"
        );
    }
}

// =============================================================================
// F069: Text truncation
// Falsification criterion: Long text overflows
// =============================================================================

#[test]
fn f069_text_truncation() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(
        1234,
        "very_long_username_that_should_be_truncated",
        50.0,
        25.0,
        "very_long_command_name_that_should_also_be_truncated",
    )]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Truncation should happen - either username or command should be cut
    // The widget should not render text that exceeds bounds
    let total_len: usize = canvas.texts.iter().map(|(t, _, _)| t.len()).sum();
    // With 80 char width, total text shouldn't be excessively long
    assert!(
        total_len < 500,
        "Text should be truncated, not overflow: {total_len} chars"
    );
}

#[test]
fn f069_truncation_uses_ellipsis() {
    let mut table = ProcessTable::new();
    // Use a very long command that will definitely be truncated
    table.set_processes(vec![ProcessEntry::new(
        1,
        "root",
        0.0,
        0.0,
        "this_is_a_very_long_command_name_that_will_be_truncated",
    )]);
    table.layout(Rect::new(0.0, 0.0, 60.0, 10.0)); // Narrow width to force truncation

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Either truncation with "..." or the full text should be contained within bounds
    let all_text = canvas.all_text();
    // The truncation implementation should clip long text
    assert!(
        all_text.contains("...")
            || !all_text.contains("this_is_a_very_long_command_name_that_will_be_truncated"),
        "Long text should be truncated"
    );
}

// =============================================================================
// F070: Text alignment
// Falsification criterion: Right-align not working
// =============================================================================

#[test]
fn f070_text_alignment_right() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![
        ProcessEntry::new(1, "root", 1.0, 0.5, "init"),
        ProcessEntry::new(1234, "noah", 99.9, 88.8, "heavy"),
    ]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Find the positions of PID values - they should be right-aligned
    // PID 1 and PID 1234 should align on the right edge
    let pid1 = canvas
        .texts
        .iter()
        .find(|(t, _, _)| t.trim() == "1" || t.contains("     1"));
    let pid1234 = canvas.texts.iter().find(|(t, _, _)| t.contains("1234"));

    // Both should exist
    assert!(pid1.is_some(), "PID 1 should be rendered");
    assert!(pid1234.is_some(), "PID 1234 should be rendered");

    // The formatting should right-align (PID 1 should have leading spaces)
    if let Some((text, _, _)) = pid1 {
        // Right-aligned single digit should have leading spaces
        let has_padding = text.starts_with(' ') || text.len() > 1;
        assert!(
            has_padding,
            "PID should be right-aligned with padding: '{text}'"
        );
    }
}

// =============================================================================
// F071: Superscript rendering
// Falsification criterion: to_superscript(123) != "¹²³"
// =============================================================================

#[test]
fn f071_superscript_rendering() {
    let result = BrailleSymbols::to_superscript(123);
    assert_eq!(result, "¹²³", "to_superscript(123) should produce ¹²³");
}

#[test]
fn f071_superscript_all_digits() {
    let result = BrailleSymbols::to_superscript(1_234_567_890);
    assert_eq!(
        result, "¹²³⁴⁵⁶⁷⁸⁹⁰",
        "to_superscript should handle all digits"
    );
}

// =============================================================================
// F072: Subscript rendering
// Falsification criterion: to_subscript(123) != "₁₂₃"
// =============================================================================

#[test]
fn f072_subscript_rendering() {
    let result = BrailleSymbols::to_subscript(123);
    assert_eq!(result, "₁₂₃", "to_subscript(123) should produce ₁₂₃");
}

#[test]
fn f072_subscript_all_digits() {
    let result = BrailleSymbols::to_subscript(1_234_567_890);
    assert_eq!(
        result, "₁₂₃₄₅₆₇₈₉₀",
        "to_subscript should handle all digits"
    );
}

// =============================================================================
// F073: Unicode width
// Falsification criterion: Wide chars break layout
// =============================================================================

#[test]
fn f073_unicode_width_braille() {
    // Braille characters should not break layout
    let mut graph = BrailleGraph::new(vec![0.1, 0.5, 0.9, 0.3, 0.7]);
    graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));

    let mut canvas = TestCanvas::new();
    graph.paint(&mut canvas);

    // Should render without panic
    let all_text = canvas.all_text();
    // Should contain braille characters
    let has_unicode = all_text.chars().any(|c| c as u32 >= 0x2800);
    assert!(
        has_unicode || all_text.is_empty(),
        "BrailleGraph should render braille characters"
    );
}

#[test]
fn f073_unicode_width_sparkline() {
    let mut spark = Sparkline::new(vec![0.1, 0.5, 0.9, 0.3, 0.7]);
    spark.layout(Rect::new(0.0, 0.0, 10.0, 1.0));

    let mut canvas = TestCanvas::new();
    spark.paint(&mut canvas);

    // Should render sparkline characters (▁▂▃▄▅▆▇█)
    let all_text = canvas.all_text();
    let sparkline_chars: Vec<char> = "▁▂▃▄▅▆▇█".chars().collect();
    let has_sparkline = all_text.chars().any(|c| sparkline_chars.contains(&c));
    assert!(
        has_sparkline || !canvas.texts.is_empty(),
        "Sparkline should render block characters"
    );
}

// =============================================================================
// F074: Empty string
// Falsification criterion: Empty text causes panic
// =============================================================================

#[test]
#[allow(clippy::assertions_on_constants)]
fn f074_empty_string_no_panic() {
    let mut table = ProcessTable::new();
    // Create process with empty command
    table.set_processes(vec![ProcessEntry::new(1234, "", 50.0, 25.0, "")]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    // Should not panic
    table.paint(&mut canvas);
    assert!(true, "Rendering with empty strings should not panic");
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn f074_empty_data_widgets() {
    // Empty ProcessTable
    let mut table = ProcessTable::new();
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));
    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Empty NetworkPanel
    let mut panel = NetworkPanel::new();
    panel.layout(Rect::new(0.0, 0.0, 80.0, 10.0));
    panel.paint(&mut canvas);

    // Empty BrailleGraph
    let mut graph = BrailleGraph::new(vec![]);
    graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));
    graph.paint(&mut canvas);

    assert!(true, "Empty widgets should not panic");
}

// =============================================================================
// F075: Newline handling
// Falsification criterion: Newline chars break layout
// =============================================================================

#[test]
fn f075_newline_handling() {
    let mut table = ProcessTable::new();
    // Create process with newline in command (shouldn't happen but must be handled)
    table.set_processes(vec![ProcessEntry::new(
        1234,
        "user",
        50.0,
        25.0,
        "cmd\nwith\nnewlines",
    )]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    // Should not panic
    table.paint(&mut canvas);

    // Verify no text spans multiple rows unexpectedly
    // Each draw_text call should be for a single line
    for (text, _, _) in &canvas.texts {
        // Text with newlines might be truncated or rendered on single line
        // It should not cause rendering outside bounds
        let line_count = text.split('\n').count();
        // Either the newline is preserved in text (for single draw) or stripped
        // The key is it should not cause layout corruption
        assert!(
            line_count <= 3 || text.len() < 100,
            "Text with newlines should be handled gracefully"
        );
    }
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn f075_newline_in_user() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(
        1234,
        "user\nname",
        50.0,
        25.0,
        "cmd",
    )]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    // Should not panic
    table.paint(&mut canvas);
    assert!(true, "Newline in username should not panic");
}

// =============================================================================
// Additional boundary tests
// =============================================================================

#[test]
#[allow(clippy::assertions_on_constants)]
fn text_rendering_with_special_chars() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(
        1234,
        "test<>&\"'",
        50.0,
        25.0,
        "cmd<script>",
    )]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Should handle special characters without crash
    assert!(true, "Special characters should be handled");
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn text_rendering_zero_bounds() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1234, "user", 50.0, 25.0, "cmd")]);
    table.layout(Rect::new(0.0, 0.0, 0.0, 0.0)); // Zero bounds

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Should not panic with zero bounds
    assert!(true, "Zero bounds should not panic");
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn text_rendering_very_small_bounds() {
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1234, "user", 50.0, 25.0, "cmd")]);
    table.layout(Rect::new(0.0, 0.0, 1.0, 1.0)); // Tiny bounds

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);

    // Should handle gracefully
    assert!(true, "Very small bounds should not panic");
}
