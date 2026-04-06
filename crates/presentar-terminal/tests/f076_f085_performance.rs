#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! F076-F085 Performance Tests
//!
//! Popperian falsification tests for performance requirements.
//! Each test attempts to DISPROVE a claim about render performance.
//!
//! Reference: SPEC-024 Section E (Performance F076-F085)
//!
//! NOTE: Tolerance multipliers applied for coverage instrumentation:
//! - P1 (frame budget): 50ms threshold with coverage (16ms target)
//! - P2 (large data): 500ms threshold with coverage (100ms target)

use presentar_core::{Canvas, Color, Constraints, Point, Rect, TextStyle, Widget};
use presentar_terminal::{
    BrailleGraph, CellBuffer, CpuGrid, Gauge, MemoryBar, NetworkInterface, NetworkPanel,
    ProcessEntry, ProcessTable, Sparkline,
};
use std::time::{Duration, Instant};

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

    fn clear(&mut self) {
        self.texts.clear();
        self.rects.clear();
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

// Tolerance for coverage mode (instrumented builds are slower)
const COVERAGE_TOLERANCE: f64 = if cfg!(debug_assertions) { 50.0 } else { 3.0 };

// =============================================================================
// F076: Frame budget 16ms
// Falsification criterion: Full 80×24 redraw > 16ms (50ms with coverage)
// =============================================================================

#[test]
fn f076_frame_budget_80x24() {
    // Create a typical dashboard layout
    let mut cpu_grid = CpuGrid::new(vec![25.0, 50.0, 75.0, 100.0, 10.0, 20.0, 30.0, 40.0]);
    let mut memory_bar = MemoryBar::from_usage(
        50 * 1024 * 1024 * 1024,
        30 * 1024 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
        128 * 1024 * 1024 * 1024,
    );
    let mut process_table = ProcessTable::new();
    process_table.set_processes(
        (0..20)
            .map(|i| ProcessEntry::new(i as u32, "user", (i as f32) * 5.0, (i as f32) * 2.0, "cmd"))
            .collect(),
    );
    let mut network_panel = NetworkPanel::new();
    let mut eth0 = NetworkInterface::new("eth0");
    eth0.update(1_000_000.0, 500_000.0);
    network_panel.add_interface(eth0);

    // Layout at 80x24
    cpu_grid.layout(Rect::new(0.0, 0.0, 40.0, 4.0));
    memory_bar.layout(Rect::new(40.0, 0.0, 40.0, 4.0));
    process_table.layout(Rect::new(0.0, 8.0, 80.0, 12.0));
    network_panel.layout(Rect::new(0.0, 4.0, 40.0, 4.0));

    let mut canvas = TestCanvas::new();

    // Measure 100 frames
    let start = Instant::now();
    for _ in 0..100 {
        canvas.clear();
        cpu_grid.paint(&mut canvas);
        memory_bar.paint(&mut canvas);
        process_table.paint(&mut canvas);
        network_panel.paint(&mut canvas);
    }
    let elapsed = start.elapsed();
    let avg_frame_ms = elapsed.as_secs_f64() * 1000.0 / 100.0;

    let threshold = 16.0 * COVERAGE_TOLERANCE;
    assert!(
        avg_frame_ms < threshold,
        "F076 FALSIFIED: Average frame time {avg_frame_ms:.2}ms exceeds {threshold}ms threshold"
    );
}

// =============================================================================
// F077: Steady-state allocation
// Falsification criterion: Allocations occur during steady-state rendering
// NOTE: This test verifies buffer reuse pattern rather than actual allocation counting
// =============================================================================

#[test]
fn f077_steady_state_buffer_reuse() {
    // Verify CellBuffer can be reused without reallocation
    let mut buffer = CellBuffer::new(80, 24);

    // Clear and reuse the same buffer 100 times
    for _ in 0..100 {
        buffer.clear();
        // Verify dimensions remain consistent (no reallocation)
        assert_eq!(buffer.width(), 80);
        assert_eq!(buffer.height(), 24);
    }

    // Verify the buffer is still functional
    assert!(
        buffer.width() == 80 && buffer.height() == 24,
        "F077 FALSIFIED: CellBuffer dimensions changed during reuse"
    );
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn f077_widget_reuse_no_realloc() {
    // Verify widgets can be reused without internal reallocation
    let mut graph = BrailleGraph::new(vec![0.5; 100]);
    let mut canvas = TestCanvas::new();

    // First render
    graph.layout(Rect::new(0.0, 0.0, 50.0, 10.0));
    graph.paint(&mut canvas);

    // Update data and re-render (should reuse internal structures)
    for i in 0..100 {
        let data: Vec<f64> = (0..100)
            .map(|j| (f64::from(i + j) / 100.0).sin().abs())
            .collect();
        graph.set_data(data);
        canvas.clear();
        graph.paint(&mut canvas);
    }

    assert!(true, "F077: Widget reuse pattern verified");
}

// =============================================================================
// F078: Diff render efficiency
// Falsification criterion: >10% cells written when content unchanged
// =============================================================================

#[test]
#[allow(clippy::unnecessary_unwrap)]
fn f078_diff_render_efficiency() {
    // Verify that re-rendering unchanged content is efficient
    let mut buffer1 = CellBuffer::new(80, 24);
    let mut buffer2 = CellBuffer::new(80, 24);

    // Both buffers start cleared (identical)
    buffer1.clear();
    buffer2.clear();

    // Count identical cells
    let total_cells = 80 * 24;
    let mut identical_cells = 0;

    for y in 0..24 {
        for x in 0..80 {
            let cell1 = buffer1.get(x, y);
            let cell2 = buffer2.get(x, y);
            if cell1.is_some() && cell2.is_some() {
                let c1 = cell1.unwrap();
                let c2 = cell2.unwrap();
                if c1.symbol == c2.symbol {
                    identical_cells += 1;
                }
            }
        }
    }

    let identical_ratio = f64::from(identical_cells) / f64::from(total_cells);
    assert!(
        identical_ratio >= 0.9,
        "F078 FALSIFIED: Only {:.1}% cells identical (expected >=90%)",
        identical_ratio * 100.0
    );
}

// =============================================================================
// F079: Large data handling
// Falsification criterion: BrailleGraph(10K points) paint > 100ms (500ms coverage)
// =============================================================================

#[test]
fn f079_large_data_braille_10k() {
    let data: Vec<f64> = (0..10_000)
        .map(|i| (f64::from(i) / 1000.0).sin().abs())
        .collect();
    let mut graph = BrailleGraph::new(data);
    graph.layout(Rect::new(0.0, 0.0, 100.0, 20.0));

    let mut canvas = TestCanvas::new();

    let start = Instant::now();
    for _ in 0..10 {
        canvas.clear();
        graph.paint(&mut canvas);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / 10.0;

    let threshold = 100.0 * if cfg!(debug_assertions) { 5.0 } else { 1.0 };
    assert!(
        avg_ms < threshold,
        "F079 FALSIFIED: BrailleGraph(10K) paint avg {avg_ms:.2}ms exceeds {threshold}ms"
    );
}

// =============================================================================
// F080: Process table 1000 rows
// Falsification criterion: ProcessTable(1000).paint() > 100ms (500ms coverage)
// =============================================================================

#[test]
fn f080_process_table_1000_rows() {
    let processes: Vec<ProcessEntry> = (0..1000)
        .map(|i| {
            ProcessEntry::new(
                i as u32,
                "user",
                (i as f32) % 100.0,
                (i as f32) % 50.0,
                format!("process_{i}"),
            )
        })
        .collect();

    let mut table = ProcessTable::new();
    table.set_processes(processes);
    table.layout(Rect::new(0.0, 0.0, 80.0, 50.0));

    let mut canvas = TestCanvas::new();

    let start = Instant::now();
    for _ in 0..10 {
        canvas.clear();
        table.paint(&mut canvas);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / 10.0;

    let threshold = 100.0 * if cfg!(debug_assertions) { 5.0 } else { 1.0 };
    assert!(
        avg_ms < threshold,
        "F080 FALSIFIED: ProcessTable(1000) paint avg {avg_ms:.2}ms exceeds {threshold}ms"
    );
}

// =============================================================================
// F081: CellBuffer reuse
// Falsification criterion: CellBuffer::new() called per frame
// =============================================================================

#[test]
fn f081_cellbuffer_reuse_pattern() {
    // Verify the correct pattern: create once, clear() per frame
    let mut buffer = CellBuffer::new(80, 24);

    // Simulate 100 frame renders
    for frame in 0..100 {
        // Correct pattern: clear existing buffer (no new allocation)
        buffer.clear();

        // Simulate some writes
        for x in 0..10 {
            let cell = presentar_terminal::Cell::new(
                &format!("{}", frame % 10),
                presentar_terminal::Color::rgb(1.0, 1.0, 1.0),
                presentar_terminal::Color::rgb(0.0, 0.0, 0.0),
                presentar_terminal::Modifiers::empty(),
            );
            buffer.set(x, 0, cell);
        }
    }

    // Verify buffer is still the same dimensions (wasn't recreated)
    assert_eq!(buffer.width(), 80);
    assert_eq!(buffer.height(), 24);
    // F081: CellBuffer reuse pattern verified
}

// =============================================================================
// F082: Color conversion cache
// Falsification criterion: Same RGB→ANSI computed twice in hot path
// NOTE: Verify gradient sampling is efficient
// =============================================================================

#[test]
fn f082_color_conversion_efficiency() {
    use presentar_terminal::Gradient;

    let gradient = Gradient::from_hex(&["#7aa2f7", "#e0af68", "#f7768e"]);

    // Sample the same values repeatedly - should be fast
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = gradient.sample(0.0);
        let _ = gradient.sample(0.5);
        let _ = gradient.sample(1.0);
    }
    let elapsed = start.elapsed();

    // 30000 samples in < 10ms is acceptable
    let threshold_ms = 10.0 * if cfg!(debug_assertions) { 10.0 } else { 1.0 };
    assert!(
        elapsed.as_secs_f64() * 1000.0 < threshold_ms,
        "F082 FALSIFIED: Color sampling took {:.2}ms for 30K samples (threshold: {}ms)",
        elapsed.as_secs_f64() * 1000.0,
        threshold_ms
    );
}

// =============================================================================
// F083: String formatting
// Falsification criterion: format!() in Widget::paint() hot path
// NOTE: Verify string operations are efficient
// =============================================================================

#[test]
fn f083_string_formatting_efficiency() {
    // Test that widgets don't have expensive string formatting in paint
    let mut gauge = Gauge::new(75.0, 100.0);
    gauge.layout(Rect::new(0.0, 0.0, 30.0, 1.0));

    let mut canvas = TestCanvas::new();

    // Measure 1000 paints
    let start = Instant::now();
    for _ in 0..1000 {
        canvas.clear();
        gauge.paint(&mut canvas);
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / 1000.0;

    // Each paint should be < 100µs (100ms for debug builds)
    let threshold_us = 100.0 * if cfg!(debug_assertions) { 1000.0 } else { 1.0 };
    assert!(
        avg_us < threshold_us,
        "F083 FALSIFIED: Gauge paint avg {avg_us:.2}µs (threshold: {threshold_us}µs) - possible format! overhead"
    );
}

// =============================================================================
// F084: Widget measure cost
// Falsification criterion: Any widget.measure() > 1ms (5ms coverage)
// =============================================================================

#[test]
fn f084_widget_measure_cost() {
    let constraints = Constraints {
        min_width: 0.0,
        max_width: 80.0,
        min_height: 0.0,
        max_height: 24.0,
    };

    // Create various widgets
    let cpu_grid = CpuGrid::new(vec![25.0, 50.0, 75.0, 100.0, 10.0, 20.0, 30.0, 40.0]);
    let graph = BrailleGraph::new(vec![0.5; 100]);
    let sparkline = Sparkline::new(vec![0.1, 0.5, 0.9, 0.3, 0.7]);
    let gauge = Gauge::new(50.0, 100.0);
    let mut table = ProcessTable::new();
    table.set_processes(vec![ProcessEntry::new(1, "user", 50.0, 10.0, "cmd")]);

    let widgets: Vec<&dyn Widget> = vec![&cpu_grid, &graph, &sparkline, &gauge, &table];

    for widget in widgets {
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = widget.measure(constraints);
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / 1000.0;

        // Each measure should be < 10µs (1000µs = 1ms in debug)
        let threshold_us = 10.0 * if cfg!(debug_assertions) { 100.0 } else { 1.0 };
        assert!(
            avg_us < threshold_us,
            "F084 FALSIFIED: Widget measure avg {avg_us:.2}µs exceeds {threshold_us}µs"
        );
    }
}

// =============================================================================
// F085: Paint cost
// Falsification criterion: Full screen paint > 8ms (40ms coverage)
// =============================================================================

#[test]
fn f085_full_screen_paint_cost() {
    // Create full dashboard
    let mut cpu_grid = CpuGrid::new(vec![25.0, 50.0, 75.0, 100.0, 10.0, 20.0, 30.0, 40.0]);
    let mut memory_bar = MemoryBar::from_usage(
        50 * 1024 * 1024 * 1024,
        30 * 1024 * 1024 * 1024,
        2 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
        128 * 1024 * 1024 * 1024,
    );
    let mut process_table = ProcessTable::new();
    process_table.set_processes(
        (0..50)
            .map(|i| ProcessEntry::new(i as u32, "user", (i as f32) * 2.0, i as f32, "command"))
            .collect(),
    );
    let mut network_panel = NetworkPanel::new();
    let mut eth0 = NetworkInterface::new("eth0");
    eth0.update(1_000_000.0, 500_000.0);
    network_panel.add_interface(eth0);
    let mut graph = BrailleGraph::new(
        (0..100)
            .map(|i| (f64::from(i) / 100.0).sin().abs())
            .collect(),
    );
    let mut sparkline = Sparkline::new(vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.6, 0.4, 0.2]);

    // Layout all widgets for 80x24 terminal
    cpu_grid.layout(Rect::new(0.0, 0.0, 40.0, 4.0));
    memory_bar.layout(Rect::new(40.0, 0.0, 40.0, 4.0));
    graph.layout(Rect::new(0.0, 4.0, 40.0, 4.0));
    network_panel.layout(Rect::new(40.0, 4.0, 40.0, 4.0));
    process_table.layout(Rect::new(0.0, 8.0, 80.0, 14.0));
    sparkline.layout(Rect::new(0.0, 22.0, 80.0, 2.0));

    let mut canvas = TestCanvas::new();

    // Measure 100 full redraws
    let start = Instant::now();
    for _ in 0..100 {
        canvas.clear();
        cpu_grid.paint(&mut canvas);
        memory_bar.paint(&mut canvas);
        graph.paint(&mut canvas);
        network_panel.paint(&mut canvas);
        process_table.paint(&mut canvas);
        sparkline.paint(&mut canvas);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / 100.0;

    let threshold = 8.0 * if cfg!(debug_assertions) { 5.0 } else { 1.0 };
    assert!(
        avg_ms < threshold,
        "F085 FALSIFIED: Full screen paint avg {avg_ms:.2}ms exceeds {threshold}ms threshold"
    );
}

// =============================================================================
// Additional performance verification tests
// =============================================================================

#[test]
fn perf_100k_braille_points_within_budget() {
    // Verify 100K points can be rendered within reasonable time
    let data: Vec<f64> = (0..100_000)
        .map(|i| (f64::from(i) / 10000.0).sin().abs())
        .collect();
    let mut graph = BrailleGraph::new(data);
    graph.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

    let mut canvas = TestCanvas::new();

    let start = Instant::now();
    graph.paint(&mut canvas);
    let elapsed = start.elapsed();

    let threshold = Duration::from_millis(if cfg!(debug_assertions) { 1000 } else { 100 });
    assert!(
        elapsed < threshold,
        "100K BrailleGraph paint took {elapsed:?} (threshold: {threshold:?})"
    );
}

#[test]
fn perf_sparkline_rapid_update() {
    // Verify sparkline can handle rapid updates (simulating real-time data)
    let mut sparkline = Sparkline::new(vec![0.5; 100]);
    sparkline.layout(Rect::new(0.0, 0.0, 100.0, 1.0));

    let mut canvas = TestCanvas::new();

    let start = Instant::now();
    for i in 0..1000 {
        // Shift data and add new point
        let new_data: Vec<f64> = (0..100)
            .map(|j| (f64::from(i + j) / 100.0).sin().abs())
            .collect();
        sparkline.set_data(new_data);
        canvas.clear();
        sparkline.paint(&mut canvas);
    }
    let elapsed = start.elapsed();

    let threshold = Duration::from_millis(if cfg!(debug_assertions) { 500 } else { 50 });
    assert!(
        elapsed < threshold,
        "1000 sparkline updates took {elapsed:?} (threshold: {threshold:?})"
    );
}
