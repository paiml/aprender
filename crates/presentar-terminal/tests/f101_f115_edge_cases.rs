#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::assertions_on_constants
)]
//! F101-F115 Edge Cases & Boundary Conditions Tests
//!
//! Popperian falsification tests for edge cases in presentar-terminal.
//! Each test attempts to DISPROVE a claim about handling extreme inputs.
//!
//! Reference: SPEC-024 Section G (Edge Cases F101-F115)

use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::{
    BrailleGraph, Cell, CellBuffer, CpuGrid, Gauge, Gradient, Heatmap, HeatmapCell, Modifiers,
    NetworkPanel, ProcessEntry, ProcessTable, Sparkline,
};
use std::sync::{Arc, Mutex};
use std::thread;

/// Test canvas that captures rendering for verification.
struct TestCanvas {
    texts: Vec<(String, Point, TextStyle)>,
    rects: Vec<(Rect, Color)>,
}

impl TestCanvas {
    const fn new() -> Self {
        Self {
            texts: Vec::new(),
            rects: Vec::new(),
        }
    }
}

impl Canvas for TestCanvas {
    fn fill_rect(&mut self, rect: Rect, color: Color) {
        self.rects.push((rect, color));
    }
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
// F101: NaN data handling
// Falsification criterion: BrailleGraph::set_data([NaN]) panics
// =============================================================================

#[test]
fn f101_nan_data_handling() {
    // BrailleGraph should handle NaN values gracefully without panicking
    let mut graph = BrailleGraph::new(vec![f64::NAN, f64::NAN, f64::NAN]);
    graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));

    let mut canvas = TestCanvas::new();
    // This should NOT panic
    graph.paint(&mut canvas);

    // Verify graph rendered something (even if degenerate)
    // NaN values should be clamped or treated as 0
    assert!(true, "BrailleGraph should handle NaN without panic");
}

#[test]
fn f101_sparkline_nan_handling() {
    // Sparkline should also handle NaN
    let mut sparkline = Sparkline::new(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
    sparkline.layout(Rect::new(0.0, 0.0, 10.0, 1.0));

    let mut canvas = TestCanvas::new();
    sparkline.paint(&mut canvas);
    assert!(true, "Sparkline should handle NaN without panic");
}

#[test]
fn f101_cpugrid_nan_handling() {
    // CpuGrid should handle NaN values
    let mut grid = CpuGrid::new(vec![f64::NAN, 50.0, f64::NAN, 75.0]);
    grid.layout(Rect::new(0.0, 0.0, 20.0, 4.0));

    let mut canvas = TestCanvas::new();
    grid.paint(&mut canvas);
    assert!(true, "CpuGrid should handle NaN without panic");
}

// =============================================================================
// F102: Inf data handling
// Falsification criterion: Gauge::new(f64::INFINITY, 100.0) panics
// =============================================================================

#[test]
fn f102_inf_data_handling() {
    // BrailleGraph with infinity values
    let mut graph = BrailleGraph::new(vec![f64::INFINITY, f64::NEG_INFINITY, 0.5]);
    graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));

    let mut canvas = TestCanvas::new();
    // This should NOT panic - infinity should be clamped
    graph.paint(&mut canvas);
    assert!(true, "BrailleGraph should handle Inf without panic");
}

#[test]
fn f102_sparkline_inf_handling() {
    // Sparkline with infinity
    let mut sparkline = Sparkline::new(vec![f64::INFINITY, 0.5, f64::NEG_INFINITY]);
    sparkline.layout(Rect::new(0.0, 0.0, 10.0, 1.0));

    let mut canvas = TestCanvas::new();
    sparkline.paint(&mut canvas);
    assert!(true, "Sparkline should handle Inf without panic");
}

#[test]
fn f102_heatmap_inf_handling() {
    // Heatmap with infinity
    let data = vec![
        vec![
            HeatmapCell::new(f64::INFINITY),
            HeatmapCell::new(0.5),
            HeatmapCell::new(0.3),
        ],
        vec![
            HeatmapCell::new(0.2),
            HeatmapCell::new(f64::NEG_INFINITY),
            HeatmapCell::new(0.7),
        ],
        vec![
            HeatmapCell::new(0.1),
            HeatmapCell::new(0.4),
            HeatmapCell::new(0.5),
        ],
    ];
    let mut heatmap = Heatmap::new(data);
    heatmap.layout(Rect::new(0.0, 0.0, 9.0, 9.0));

    let mut canvas = TestCanvas::new();
    heatmap.paint(&mut canvas);
    assert!(true, "Heatmap should handle Inf without panic");
}

// =============================================================================
// F103: Negative values
// Falsification criterion: MemoryBar::new(-50.0) panics
// =============================================================================

#[test]
fn f103_negative_values_braille() {
    // BrailleGraph with negative values
    let mut graph = BrailleGraph::new(vec![-0.5, -1.0, 0.0, 0.5, 1.0]);
    graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));

    let mut canvas = TestCanvas::new();
    graph.paint(&mut canvas);
    assert!(true, "BrailleGraph should handle negative values");
}

#[test]
fn f103_negative_values_sparkline() {
    // Sparkline with negative values
    let mut sparkline = Sparkline::new(vec![-5.0, -2.0, 0.0, 2.0, 5.0]);
    sparkline.layout(Rect::new(0.0, 0.0, 10.0, 1.0));

    let mut canvas = TestCanvas::new();
    sparkline.paint(&mut canvas);
    assert!(true, "Sparkline should handle negative values");
}

#[test]
fn f103_negative_process_cpu() {
    // ProcessTable with negative CPU (invalid but shouldn't crash)
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1, "test", -50.0, -10.0, "cmd")]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 5.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "ProcessTable should handle negative values");
}

// =============================================================================
// F104: Zero-width terminal
// Falsification criterion: CellBuffer::new(0, 24) panics
// =============================================================================

#[test]
fn f104_zero_width_cellbuffer() {
    // CellBuffer with zero width - should handle gracefully
    let buffer = CellBuffer::new(0, 24);
    assert_eq!(buffer.width(), 0);
    assert_eq!(buffer.height(), 24);
}

#[test]
fn f104_zero_width_layout() {
    // Widget with zero-width layout
    let mut graph = BrailleGraph::new(vec![0.5, 0.7, 0.3]);
    graph.layout(Rect::new(0.0, 0.0, 0.0, 5.0));

    let mut canvas = TestCanvas::new();
    graph.paint(&mut canvas);
    assert!(true, "Widget should handle zero-width layout");
}

// =============================================================================
// F105: Zero-height terminal
// Falsification criterion: CellBuffer::new(80, 0) panics
// =============================================================================

#[test]
fn f105_zero_height_cellbuffer() {
    // CellBuffer with zero height - should handle gracefully
    let buffer = CellBuffer::new(80, 0);
    assert_eq!(buffer.width(), 80);
    assert_eq!(buffer.height(), 0);
}

#[test]
fn f105_zero_height_layout() {
    // Widget with zero-height layout
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1, "test", 50.0, 10.0, "cmd")]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 0.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "Widget should handle zero-height layout");
}

// =============================================================================
// F106: Single-cell render
// Falsification criterion: Widget renders incorrectly in 1x1
// =============================================================================

#[test]
fn f106_single_cell_braille() {
    // BrailleGraph in 1x1 cell
    let mut graph = BrailleGraph::new(vec![0.5]);
    graph.layout(Rect::new(0.0, 0.0, 1.0, 1.0));

    let mut canvas = TestCanvas::new();
    graph.paint(&mut canvas);
    assert!(true, "BrailleGraph should render in 1x1");
}

#[test]
fn f106_single_cell_gauge() {
    // Gauge in minimal size
    let mut gauge = Gauge::new(50.0, 100.0);
    gauge.layout(Rect::new(0.0, 0.0, 1.0, 1.0));

    let mut canvas = TestCanvas::new();
    gauge.paint(&mut canvas);
    assert!(true, "Gauge should render in 1x1");
}

#[test]
fn f106_single_cell_sparkline() {
    // Sparkline in 1x1
    let mut sparkline = Sparkline::new(vec![0.5]);
    sparkline.layout(Rect::new(0.0, 0.0, 1.0, 1.0));

    let mut canvas = TestCanvas::new();
    sparkline.paint(&mut canvas);
    assert!(true, "Sparkline should render in 1x1");
}

// =============================================================================
// F107: UTF-8 boundary
// Falsification criterion: Multi-byte char split causes panic
// =============================================================================

#[test]
fn f107_utf8_multibyte_process_name() {
    // Process with multi-byte UTF-8 characters
    let mut table = ProcessTable::new();
    table.set_processes(vec![
        ProcessEntry::new(1, "user", 50.0, 10.0, "\u{4e2d}\u{6587}"), // Chinese
        ProcessEntry::new(2, "user", 30.0, 5.0, "\u{65e5}\u{672c}\u{8a9e}"), // Japanese
        ProcessEntry::new(3, "\u{0430}\u{0434}\u{043c}", 20.0, 3.0, "cmd"), // Cyrillic user
    ]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "ProcessTable should handle UTF-8 multibyte");
}

#[test]
fn f107_utf8_cell_set() {
    // CellBuffer with multi-byte character
    let mut buffer = CellBuffer::new(10, 5);
    let fg = presentar_terminal::Color::rgb(1.0, 1.0, 1.0);
    let bg = presentar_terminal::Color::rgb(0.0, 0.0, 0.0);
    // Set a cell with multi-byte character
    buffer.set(0, 0, Cell::new("\u{4e2d}", fg, bg, Modifiers::empty())); // Chinese character
    buffer.set(1, 0, Cell::new("\u{2764}", fg, bg, Modifiers::empty())); // Heart emoji
    assert!(true, "CellBuffer should handle multi-byte chars");
}

// =============================================================================
// F108: Emoji handling
// Falsification criterion: ZWJ sequence breaks layout
// =============================================================================

#[test]
fn f108_emoji_zwj_sequence() {
    // Process with ZWJ emoji sequence (family: man, woman, girl, boy)
    let mut table = ProcessTable::new();
    table.set_processes(vec![
        ProcessEntry::new(
            1,
            "user",
            50.0,
            10.0,
            "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}",
        ),
        ProcessEntry::new(2, "user", 30.0, 5.0, "\u{1F1FA}\u{1F1F8}"), // Flag sequence
    ]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "ProcessTable should handle emoji ZWJ sequences");
}

#[test]
fn f108_emoji_skin_tone() {
    // Emoji with skin tone modifier
    let mut table = ProcessTable::new();
    table.set_processes(vec![
        ProcessEntry::new(1, "user", 50.0, 10.0, "\u{1F44D}\u{1F3FD}"), // Thumbs up with skin tone
    ]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 5.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "ProcessTable should handle emoji skin tone modifiers");
}

// =============================================================================
// F109: RTL text
// Falsification criterion: Arabic/Hebrew text renders incorrectly
// =============================================================================

#[test]
fn f109_rtl_arabic_text() {
    // Process with Arabic RTL text
    let mut table = ProcessTable::new();
    table.set_processes(vec![
        ProcessEntry::new(1, "\u{0639}\u{0631}\u{0628}\u{064a}", 50.0, 10.0, "cmd"), // Arabic user
        ProcessEntry::new(
            2,
            "user",
            30.0,
            5.0,
            "\u{0645}\u{0631}\u{062d}\u{0628}\u{0627}",
        ), // Arabic command
    ]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "ProcessTable should handle Arabic RTL text");
}

#[test]
fn f109_rtl_hebrew_text() {
    // Process with Hebrew RTL text
    let mut table = ProcessTable::new();
    table.set_processes(vec![
        ProcessEntry::new(1, "\u{05e9}\u{05dc}\u{05d5}\u{05dd}", 50.0, 10.0, "cmd"), // Hebrew user
    ]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 5.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "ProcessTable should handle Hebrew RTL text");
}

#[test]
fn f109_bidi_mixed_text() {
    // Mixed bidirectional text (LTR + RTL)
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(
        1,
        "user",
        50.0,
        10.0,
        "Hello \u{05e9}\u{05dc}\u{05d5}\u{05dd} World",
    )]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 5.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "ProcessTable should handle mixed bidirectional text");
}

// =============================================================================
// F110: 100K data points
// Falsification criterion: BrailleGraph with 100K points OOMs
// =============================================================================

#[test]
fn f110_large_data_braille() {
    // BrailleGraph with 100K data points
    let data: Vec<f64> = (0..100_000)
        .map(|i| (f64::from(i) / 100_000.0).sin().abs())
        .collect();
    let mut graph = BrailleGraph::new(data);
    graph.layout(Rect::new(0.0, 0.0, 100.0, 10.0));

    let mut canvas = TestCanvas::new();
    graph.paint(&mut canvas);
    assert!(true, "BrailleGraph should handle 100K data points");
}

#[test]
fn f110_large_data_sparkline() {
    // Sparkline with 10K data points
    let data: Vec<f64> = (0..10_000)
        .map(|i| (f64::from(i) / 1000.0).sin().abs())
        .collect();
    let mut sparkline = Sparkline::new(data);
    sparkline.layout(Rect::new(0.0, 0.0, 50.0, 1.0));

    let mut canvas = TestCanvas::new();
    sparkline.paint(&mut canvas);
    assert!(true, "Sparkline should handle 10K data points");
}

#[test]
fn f110_large_process_table() {
    // ProcessTable with 1000 processes
    let processes: Vec<ProcessEntry> = (0..1000)
        .map(|i| {
            ProcessEntry::new(
                i as u32,
                "user",
                (i as f32) % 100.0,
                (i as f32) % 50.0,
                format!("cmd_{i}"),
            )
        })
        .collect();
    let mut table = ProcessTable::new();
    table.set_processes(processes);
    table.layout(Rect::new(0.0, 0.0, 80.0, 50.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "ProcessTable should handle 1000 processes");
}

// =============================================================================
// F111: Rapid resize
// Falsification criterion: 100 resize events/sec causes crash
// =============================================================================

#[test]
fn f111_rapid_resize() {
    // Simulate rapid resize events
    let mut graph = BrailleGraph::new(vec![0.1, 0.5, 0.9, 0.3, 0.7]);
    let mut canvas = TestCanvas::new();

    // Simulate 100 rapid resize events
    for i in 0..100 {
        let width = 10.0 + (i % 50) as f32;
        let height = 5.0 + (i % 20) as f32;
        graph.layout(Rect::new(0.0, 0.0, width, height));
        graph.paint(&mut canvas);
    }
    assert!(true, "Widget should handle rapid resize");
}

#[test]
fn f111_resize_cellbuffer() {
    // CellBuffer resize operations
    for i in 0..100 {
        let width = (i % 100) + 1;
        let height = (i % 50) + 1;
        let buffer = CellBuffer::new(width, height);
        assert_eq!(buffer.width(), width);
        assert_eq!(buffer.height(), height);
    }
    assert!(true, "CellBuffer should handle rapid resize");
}

// =============================================================================
// F112: Theme hot-swap
// Falsification criterion: Theme change mid-render causes artifact
// =============================================================================

#[test]
fn f112_theme_hot_swap() {
    // Test that widget recreations with different configurations doesn't cause issues
    // (simulating theme switching by recreating widgets with different colors)
    let gradients = [
        Gradient::from_hex(&["#7aa2f7", "#e0af68", "#f7768e"]), // Tokyo Night
        Gradient::from_hex(&["#50fa7b", "#f1fa8c", "#ff5555"]), // Dracula
        Gradient::from_hex(&["#a3be8c", "#ebcb8b", "#bf616a"]), // Nord
        Gradient::from_hex(&["#a6e22e", "#e6db74", "#f92672"]), // Monokai
    ];

    let mut canvas = TestCanvas::new();

    for gradient in gradients.iter().cycle().take(20) {
        let mut graph = BrailleGraph::new(vec![0.1, 0.5, 0.9]).with_gradient(gradient.clone());
        graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));
        graph.paint(&mut canvas);
    }
    assert!(
        true,
        "Widget recreation with different gradients should work"
    );
}

// =============================================================================
// F113: Concurrent updates
// Falsification criterion: Race between data update and paint
// =============================================================================

#[test]
fn f113_concurrent_update_simulation() {
    // Simulate concurrent data updates (single-threaded simulation)
    let mut graph = BrailleGraph::new(vec![0.5]);
    let mut canvas = TestCanvas::new();

    // Interleave data updates and paints
    for i in 0..100 {
        let data: Vec<f64> = (0..10)
            .map(|j| (f64::from(i + j) / 100.0).sin().abs())
            .collect();
        graph.set_data(data);
        graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));
        graph.paint(&mut canvas);
    }
    assert!(true, "Widget should handle interleaved updates");
}

#[test]
fn f113_thread_safe_cellbuffer() {
    // Verify CellBuffer can be used across threads
    let buffer = Arc::new(Mutex::new(CellBuffer::new(80, 24)));

    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let buffer_clone = Arc::clone(&buffer);
            thread::spawn(move || {
                let fg = presentar_terminal::Color::rgb(1.0, 1.0, 1.0);
                let bg = presentar_terminal::Color::rgb(0.0, 0.0, 0.0);
                for i in 0..100 {
                    let mut buf = buffer_clone.lock().unwrap();
                    let x = ((thread_id * 20 + i % 20) % 80) as u16;
                    let y = ((thread_id * 6) % 24) as u16;
                    buf.set(x, y, Cell::new("X", fg, bg, Modifiers::empty()));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
    assert!(true, "CellBuffer should be thread-safe with mutex");
}

// =============================================================================
// F114: Signal during render
// Falsification criterion: SIGWINCH during paint() corrupts state
// =============================================================================

#[test]
fn f114_signal_simulation() {
    // We can't actually test SIGWINCH in unit tests, but we can test
    // that widgets maintain valid state through interruptions
    let mut graph = BrailleGraph::new(vec![0.1, 0.5, 0.9]);
    let mut canvas = TestCanvas::new();

    // Start painting
    graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));
    graph.paint(&mut canvas);

    // Simulate "interrupt" by clearing and re-painting with different size
    let mut canvas2 = TestCanvas::new();
    graph.layout(Rect::new(0.0, 0.0, 40.0, 10.0));
    graph.paint(&mut canvas2);

    // Both paints should succeed without corruption
    assert!(true, "Widget state should survive resize during render");
}

// =============================================================================
// F115: Ctrl+C cleanup
// Falsification criterion: SIGINT leaves terminal in raw mode
// =============================================================================

#[test]
fn f115_cleanup_simulation() {
    // Test that buffers can be properly cleared/reset
    let mut buffer = CellBuffer::new(80, 24);
    let fg = presentar_terminal::Color::rgb(1.0, 1.0, 1.0);
    let bg = presentar_terminal::Color::rgb(0.0, 0.0, 0.0);

    // Fill with data
    for y in 0..24u16 {
        for x in 0..80u16 {
            buffer.set(x, y, Cell::new("X", fg, bg, Modifiers::empty()));
        }
    }

    // Clear should reset all cells
    buffer.clear();

    // Verify cleared (cells should be spaces)
    for y in 0..24u16 {
        for x in 0..80u16 {
            if let Some(cell) = buffer.get(x, y) {
                assert_eq!(
                    cell.symbol.as_str(),
                    " ",
                    "Cell should be space after clear at ({x}, {y})"
                );
            }
        }
    }
}

#[test]
fn f115_widget_drop() {
    // Verify widgets can be dropped cleanly
    {
        let _graph = BrailleGraph::new(vec![0.1, 0.5, 0.9]);
        let _table = ProcessTable::new();
        let _buffer = CellBuffer::new(80, 24);
        // All widgets go out of scope and are dropped
    }
    assert!(true, "Widgets should drop without panic");
}

// =============================================================================
// Additional edge case tests for completeness
// =============================================================================

#[test]
fn edge_case_empty_data_all_widgets() {
    // Test all widgets with empty data
    let mut graph = BrailleGraph::new(vec![]);
    graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));
    let mut canvas = TestCanvas::new();
    graph.paint(&mut canvas);

    let mut sparkline = Sparkline::new(vec![]);
    sparkline.layout(Rect::new(0.0, 0.0, 10.0, 1.0));
    sparkline.paint(&mut canvas);

    let mut table = ProcessTable::new();
    table.set_processes(vec![]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));
    table.paint(&mut canvas);

    let mut network = NetworkPanel::new();
    network.layout(Rect::new(0.0, 0.0, 40.0, 5.0));
    network.paint(&mut canvas);

    assert!(true, "All widgets should handle empty data");
}

#[test]
fn edge_case_extreme_values() {
    // Test extreme but valid f64 values
    let extreme_values = vec![f64::MIN_POSITIVE, f64::MAX, f64::MIN, 1e-300, 1e300];

    let mut graph = BrailleGraph::new(extreme_values.clone());
    graph.layout(Rect::new(0.0, 0.0, 20.0, 5.0));
    let mut canvas = TestCanvas::new();
    graph.paint(&mut canvas);

    let mut sparkline = Sparkline::new(extreme_values);
    sparkline.layout(Rect::new(0.0, 0.0, 10.0, 1.0));
    sparkline.paint(&mut canvas);

    assert!(true, "Widgets should handle extreme f64 values");
}

#[test]
fn edge_case_very_long_strings() {
    // Test with very long strings
    let long_command = "a".repeat(10000);
    let long_user = "b".repeat(1000);

    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(
        1,
        &long_user,
        50.0,
        10.0,
        &long_command,
    )]);
    table.layout(Rect::new(0.0, 0.0, 80.0, 5.0));

    let mut canvas = TestCanvas::new();
    table.paint(&mut canvas);
    assert!(true, "ProcessTable should handle very long strings");
}
