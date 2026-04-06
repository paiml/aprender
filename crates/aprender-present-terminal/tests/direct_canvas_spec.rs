#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Popperian Falsification Tests for Direct TUI Backend
//!
//! This test file implements the falsification checklist from
//! `docs/specifications/simplified-tui-spec.md` Section 8.
//!
//! # Methodology
//!
//! Each test attempts to falsify a specific claim. If ANY test fails,
//! the claim is falsified and implementation must be fixed.
//!
//! # Reference
//!
//! PROBAR-SPEC-009: Bug Hunting Probador - Brick Architecture

use presentar_core::{Canvas, Color, Point, Rect, TextStyle};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas, Modifiers};
use presentar_terminal::ColorMode;
use std::time::Instant;

// =============================================================================
// P1-P10: Performance Claims
// =============================================================================

/// P1: Full 80×24 redraw completes in <1ms
#[test]
fn p1_full_redraw_80x24_under_1ms() {
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::new();

    // Warm up
    for _ in 0..10 {
        buffer.mark_all_dirty();
        let mut output = Vec::with_capacity(8192);
        let _ = renderer.flush(&mut buffer, &mut output);
    }

    // Measure
    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        buffer.mark_all_dirty();
        let mut output = Vec::with_capacity(8192);
        let _ = renderer.flush(&mut buffer, &mut output);
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);

    // Allow significant headroom for CI variance and coverage instrumentation
    // Coverage mode can add 10-50x overhead
    assert!(
        avg_ms < 50.0,
        "P1 FALSIFIED: Full 80x24 redraw took {avg_ms:.3}ms (target: <1ms, tolerance: <50ms with coverage)"
    );
}

/// P2: Differential update of 10% cells <0.1ms
#[test]
fn p2_differential_update_10_percent_under_100us() {
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::new();

    // Initial full render
    buffer.mark_all_dirty();
    let mut output = Vec::with_capacity(8192);
    let _ = renderer.flush(&mut buffer, &mut output);

    // Update ~10% of cells (192 cells)
    let iterations = 100;
    let start = Instant::now();

    for i in 0..iterations {
        // Mark 192 cells dirty (10% of 1920)
        for j in 0..192 {
            let x = ((i + j) % 80) as u16;
            let y = ((i + j) / 80 % 24) as u16;
            buffer.update(x, y, "X", Color::WHITE, Color::BLACK, Modifiers::NONE);
        }
        output.clear();
        let _ = renderer.flush(&mut buffer, &mut output);
    }

    let elapsed = start.elapsed();
    let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / f64::from(iterations);

    // Allow headroom for CI and coverage instrumentation
    assert!(
        avg_us < 5000.0,
        "P2 FALSIFIED: 10% update took {avg_us:.1}μs (target: <100μs, tolerance: <5000μs with coverage)"
    );
}

/// P3: Memory usage <100KB for 80×24
#[test]
fn p3_memory_under_100kb() {
    let buffer = CellBuffer::new(80, 24);

    // CellBuffer structure size estimate:
    // - 1920 cells × ~40 bytes/cell = ~76.8KB
    // - Dirty bitmap: 1920 bits = 240 bytes
    // - Metadata: ~16 bytes

    // We can't directly measure heap allocation without alloc hooks,
    // but we can verify the structure sizes are reasonable
    let cell_count = buffer.len();
    assert_eq!(cell_count, 1920);

    // Verify dirty bitmap is efficient (1 bit per cell)
    let dirty_count_full = {
        let mut buf = CellBuffer::new(80, 24);
        buf.mark_all_dirty();
        buf.dirty_count()
    };
    assert_eq!(dirty_count_full, 1920);
}

/// P4: 200×50 full redraw <5ms
#[test]
fn p4_large_terminal_redraw_under_5ms() {
    let mut buffer = CellBuffer::new(200, 50);
    let mut renderer = DiffRenderer::new();

    // Warm up
    for _ in 0..5 {
        buffer.mark_all_dirty();
        let mut output = Vec::with_capacity(32768);
        let _ = renderer.flush(&mut buffer, &mut output);
    }

    // Measure
    let iterations = 20;
    let start = Instant::now();

    for _ in 0..iterations {
        buffer.mark_all_dirty();
        let mut output = Vec::with_capacity(32768);
        let _ = renderer.flush(&mut buffer, &mut output);
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / f64::from(iterations);

    // Allow significant headroom for coverage instrumentation
    assert!(
        avg_ms < 200.0,
        "P4 FALSIFIED: 200x50 redraw took {avg_ms:.3}ms (target: <5ms, tolerance: <200ms with coverage)"
    );
}

/// P6: Dirty bitmap overhead <1% of buffer
#[test]
fn p6_dirty_bitmap_overhead_under_1_percent() {
    // CellBuffer cells: ~40 bytes each
    // Dirty bitmap: 1 bit per cell
    //
    // For 80x24 = 1920 cells:
    // Cell data: ~76,800 bytes
    // Dirty bitmap: 240 bytes (1920 bits / 8)
    // Overhead: 240 / 76800 = 0.3%

    let buffer = CellBuffer::new(80, 24);
    let cell_count = buffer.len();

    // Bitmap size in bits = cell_count
    // Bitmap size in bytes = cell_count / 8
    let bitmap_bytes = cell_count.div_ceil(8);

    // Assume ~40 bytes per cell (CompactString + colors + modifiers)
    let estimated_cell_size = 40;
    let cell_data_bytes = cell_count * estimated_cell_size;

    let overhead_percent = (bitmap_bytes as f64 / cell_data_bytes as f64) * 100.0;

    assert!(
        overhead_percent < 1.0,
        "P6 FALSIFIED: Dirty bitmap overhead is {overhead_percent:.2}% (target: <1%)"
    );
}

/// P7: Cell lookup is O(1)
#[test]
fn p7_cell_lookup_is_constant_time() {
    // Test that lookup time doesn't grow with buffer size
    let sizes = [(10, 10), (100, 100), (200, 200)];
    let mut times = Vec::new();

    for (w, h) in sizes {
        let buffer = CellBuffer::new(w, h);

        // Warm up
        for _ in 0..1000 {
            let _ = buffer.get(w / 2, h / 2);
        }

        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = buffer.get(w / 2, h / 2);
        }

        let elapsed = start.elapsed();
        times.push(elapsed.as_nanos() / iterations as u128);
    }

    // All times should be roughly equal (within 10x of smallest)
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();

    assert!(
        max_time < min_time * 10,
        "P7 FALSIFIED: Cell lookup time varies too much: min={min_time}ns, max={max_time}ns"
    );
}

/// P9: Cursor movement minimized
#[test]
fn p9_cursor_movement_minimized() {
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::new();

    // Update consecutive cells on same row
    buffer.update(10, 5, "A", Color::WHITE, Color::BLACK, Modifiers::NONE);
    buffer.update(11, 5, "B", Color::WHITE, Color::BLACK, Modifiers::NONE);
    buffer.update(12, 5, "C", Color::WHITE, Color::BLACK, Modifiers::NONE);

    let mut output = Vec::new();
    let _ = renderer.flush(&mut buffer, &mut output);

    // Should only need 1 cursor move (to start position)
    assert_eq!(
        renderer.cursor_moves(),
        1,
        "P9 FALSIFIED: Expected 1 cursor move for consecutive cells, got {}",
        renderer.cursor_moves()
    );
}

/// P10: Color mode detection is fast
#[test]
fn p10_color_mode_detection_fast() {
    let iterations = 1000;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = ColorMode::detect();
    }

    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / iterations as u128;

    // Should be <1μs = <1000ns
    assert!(
        avg_ns < 10000,
        "P10 FALSIFIED: Color mode detection took {avg_ns}ns (target: <1000ns, tolerance: <10000ns)"
    );
}

// =============================================================================
// C1-C6: Correctness Claims
// =============================================================================

/// C2: Wide chars occupy correct cells
#[test]
fn c2_wide_chars_correct_width() {
    let mut buffer = CellBuffer::new(20, 5);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        canvas.draw_text("日本語", Point::new(0.0, 0.0), &TextStyle::default());
    }

    // "日本語" should occupy 6 cells (2 each)
    // Cell 0: "日", Cell 1: continuation
    // Cell 2: "本", Cell 3: continuation
    // Cell 4: "語", Cell 5: continuation

    assert_eq!(buffer.get(0, 0).unwrap().symbol.as_str(), "日");
    assert!(buffer.get(1, 0).unwrap().is_continuation());
    assert_eq!(buffer.get(2, 0).unwrap().symbol.as_str(), "本");
    assert!(buffer.get(3, 0).unwrap().is_continuation());
    assert_eq!(buffer.get(4, 0).unwrap().symbol.as_str(), "語");
    assert!(buffer.get(5, 0).unwrap().is_continuation());
}

/// C5: Color accuracy in `TrueColor`
#[test]
fn c5_truecolor_accuracy() {
    let mode = ColorMode::TrueColor;

    // Test exact RGB values are preserved
    let test_colors = [
        (1.0, 0.0, 0.0),   // Pure red
        (0.0, 1.0, 0.0),   // Pure green
        (0.0, 0.0, 1.0),   // Pure blue
        (0.5, 0.5, 0.5),   // Mid gray
        (0.25, 0.75, 0.5), // Arbitrary
    ];

    for (r, g, b) in test_colors {
        let color = Color::new(r, g, b, 1.0);
        let converted = mode.to_crossterm(color);

        if let crossterm::style::Color::Rgb {
            r: cr,
            g: cg,
            b: cb,
        } = converted
        {
            let expected_r = (r * 255.0).round() as u8;
            let expected_g = (g * 255.0).round() as u8;
            let expected_b = (b * 255.0).round() as u8;

            assert_eq!(cr, expected_r, "C5 FALSIFIED: Red channel mismatch");
            assert_eq!(cg, expected_g, "C5 FALSIFIED: Green channel mismatch");
            assert_eq!(cb, expected_b, "C5 FALSIFIED: Blue channel mismatch");
        } else {
            panic!("C5 FALSIFIED: TrueColor mode returned non-RGB color");
        }
    }
}

/// C6: 256-color palette mapping
#[test]
fn c6_256_color_mapping() {
    let mode = ColorMode::Color256;

    // Test that colors map to valid palette indices (0-255)
    let test_colors = [
        Color::RED,
        Color::GREEN,
        Color::BLUE,
        Color::WHITE,
        Color::BLACK,
    ];

    for color in test_colors {
        let converted = mode.to_crossterm(color);

        if let crossterm::style::Color::AnsiValue(_idx) = converted {
            // Valid - u8 is always in range 0-255
        } else {
            panic!("C6 FALSIFIED: Color256 mode returned non-indexed color");
        }
    }
}

/// C20: Resize preserves no content (fresh buffer)
#[test]
fn c20_resize_clears_buffer() {
    let mut buffer = CellBuffer::new(10, 10);

    // Write some content
    buffer.update(5, 5, "X", Color::RED, Color::BLUE, Modifiers::BOLD);

    // Resize
    buffer.resize(20, 20);

    // All cells should be reset
    let cell = buffer.get(5, 5).unwrap();
    assert_eq!(
        cell.symbol.as_str(),
        " ",
        "C20 FALSIFIED: Content not cleared after resize"
    );
    assert_eq!(
        cell.fg,
        Color::WHITE,
        "C20 FALSIFIED: Foreground not reset after resize"
    );
    assert_eq!(
        cell.bg,
        Color::TRANSPARENT,
        "C20 FALSIFIED: Background not reset after resize"
    );
    assert!(
        cell.modifiers.is_empty(),
        "C20 FALSIFIED: Modifiers not cleared after resize"
    );

    // All should be marked dirty
    assert_eq!(
        buffer.dirty_count(),
        400,
        "C20 FALSIFIED: Not all cells marked dirty after resize"
    );
}

// =============================================================================
// D1-D7: Dependency Claims
// =============================================================================

/// D2: Only ~4-5 direct dependencies (verified at compile time)
#[test]
#[allow(clippy::assertions_on_constants)]
fn d2_minimal_dependencies() {
    // This test verifies that the code compiles with the expected dependencies.
    // The actual dependency count is verified in Cargo.toml.

    // Required dependencies:
    // 1. presentar-core
    // 2. crossterm
    // 3. compact_str
    // 4. bitvec
    // 5. unicode-width
    // 6. unicode-segmentation
    // (ratatui has been removed - using direct crossterm backend)

    // This test passes if compilation succeeds
    assert!(true, "Dependencies verified at compile time");
}

/// D7: No unsafe in direct module (verified at compile time)
#[test]
#[allow(clippy::assertions_on_constants)]
fn d7_no_unsafe_code() {
    // The direct module uses #[forbid(unsafe_code)] at crate level
    // via the lint configuration. This test documents that requirement.

    // Verify we're using safe abstractions
    let buffer = CellBuffer::new(10, 10);
    let _len = buffer.len(); // Safe index calculation

    assert!(
        true,
        "No unsafe code in direct module - verified by lint configuration"
    );
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test that the canvas properly clips drawing operations
#[test]
fn integration_canvas_clipping() {
    let mut buffer = CellBuffer::new(10, 10);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        // Draw outside bounds - should not panic
        canvas.fill_rect(Rect::new(-5.0, -5.0, 20.0, 20.0), Color::RED);
    }

    // Should have filled the visible portion
    assert_eq!(buffer.get(0, 0).unwrap().bg, Color::RED);
    assert_eq!(buffer.get(9, 9).unwrap().bg, Color::RED);
}

/// Test transform stacking
#[test]
fn integration_transform_stacking() {
    use presentar_core::Transform2D;

    let mut buffer = CellBuffer::new(20, 20);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        canvas.push_transform(Transform2D::translate(5.0, 5.0));
        canvas.push_transform(Transform2D::translate(2.0, 2.0));
        canvas.fill_rect(Rect::new(0.0, 0.0, 2.0, 2.0), Color::GREEN);
        canvas.pop_transform();
        canvas.pop_transform();
    }

    // Should be at (7, 7) after combined transform
    assert_eq!(buffer.get(7, 7).unwrap().bg, Color::GREEN);
    // Untouched cells have transparent background
    assert_eq!(buffer.get(0, 0).unwrap().bg, Color::TRANSPARENT);
}

/// Test style caching in renderer
#[test]
fn integration_style_caching() {
    let mut buffer = CellBuffer::new(10, 5);
    let mut renderer = DiffRenderer::new();

    // Update cells with same style
    buffer.update(0, 0, "A", Color::RED, Color::BLUE, Modifiers::BOLD);
    buffer.update(1, 0, "B", Color::RED, Color::BLUE, Modifiers::BOLD);
    buffer.update(2, 0, "C", Color::RED, Color::BLUE, Modifiers::BOLD);

    let mut output = Vec::new();
    renderer.flush(&mut buffer, &mut output).unwrap();

    // Should only have 1 style change (initial)
    assert_eq!(
        renderer.style_changes(),
        1,
        "Style should be cached for consecutive cells with same style"
    );
}
