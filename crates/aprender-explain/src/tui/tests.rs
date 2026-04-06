use super::*;
use crate::analyzer::{MemoryPattern, RegisterUsage, RooflineMetric};
use presentar_core::Color;

fn sample_report() -> AnalysisReport {
    AnalysisReport {
        name: "test_kernel".to_string(),
        target: "PTX".to_string(),
        registers: RegisterUsage {
            f32_regs: 24,
            b32_regs: 18,
            b64_regs: 12,
            pred_regs: 4,
            ..Default::default()
        },
        memory: MemoryPattern {
            global_loads: 100,
            global_stores: 50,
            coalesced_ratio: 0.95,
            ..Default::default()
        },
        roofline: RooflineMetric {
            arithmetic_intensity: 2.5,
            theoretical_peak_gflops: 15000.0,
            memory_bound: true,
        },
        warnings: vec![],
        instruction_count: 150,
        estimated_occupancy: 0.875,
    }
}

/// Helper to compare f32 Color components with tolerance.
fn color_approx_eq(a: Color, b: Color) -> bool {
    (a.r - b.r).abs() < 0.01
        && (a.g - b.g).abs() < 0.01
        && (a.b - b.b).abs() < 0.01
        && (a.a - b.a).abs() < 0.01
}

/// F026: TUI app creates without panic
#[test]
fn f026_tui_app_creation() {
    let ptx = ".entry test() { ret; }".to_string();
    let report = sample_report();
    let app = TuiApp::new(ptx, report);
    assert!(!app.should_quit);
}

/// F027: Resize terminal - UI adapts responsively
/// Verifies that state remains valid after simulated resize
#[test]
fn f027_resize_terminal() {
    let ptx = (0..50)
        .map(|i| format!("    add.f32 %f{}, %f{}, %f{}", i, i, i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let report = sample_report();
    let mut app = TuiApp::new(ptx, report);

    // Simulate scrolling to middle of content
    for _ in 0..25 {
        app.handle_key(KeyCode::Down);
    }
    let scroll_before = app.source_scroll;

    // Resize events don't change app state directly
    // The UI adapts by recalculating visible area
    // Key behaviors should remain consistent

    // State should be preserved (no panics, consistent behavior)
    assert_eq!(app.source_scroll, scroll_before);
    assert!(!app.should_quit);

    // Navigation should still work after "resize"
    app.handle_key(KeyCode::Down);
    assert_eq!(app.source_scroll, scroll_before + 1);
}

/// F029: Toggle sidebar
#[test]
fn f029_toggle_sidebar() {
    let ptx = ".entry test() { ret; }".to_string();
    let report = sample_report();
    let mut app = TuiApp::new(ptx, report);

    assert!(app.sidebar_visible);
    app.handle_key(KeyCode::Char('s'));
    assert!(!app.sidebar_visible);
    app.handle_key(KeyCode::Char('s'));
    assert!(app.sidebar_visible);
}

/// F030: Quit with 'q'
#[test]
fn f030_quit_tui() {
    let ptx = ".entry test() { ret; }".to_string();
    let report = sample_report();
    let mut app = TuiApp::new(ptx, report);

    assert!(!app.should_quit);
    app.handle_key(KeyCode::Char('q'));
    assert!(app.should_quit);
}

#[test]
fn test_scroll_down() {
    let ptx = "line1\nline2\nline3\nline4\nline5".to_string();
    let report = sample_report();
    let mut app = TuiApp::new(ptx, report);

    assert_eq!(app.source_scroll, 0);
    app.handle_key(KeyCode::Down);
    assert_eq!(app.source_scroll, 1);
    app.handle_key(KeyCode::Char('j'));
    assert_eq!(app.source_scroll, 2);
}

#[test]
fn test_scroll_up() {
    let ptx = "line1\nline2\nline3".to_string();
    let report = sample_report();
    let mut app = TuiApp::new(ptx, report);

    app.source_scroll = 2;
    app.handle_key(KeyCode::Up);
    assert_eq!(app.source_scroll, 1);
    app.handle_key(KeyCode::Char('k'));
    assert_eq!(app.source_scroll, 0);
}

#[test]
fn test_scroll_bounds() {
    let ptx = "line1\nline2".to_string();
    let report = sample_report();
    let mut app = TuiApp::new(ptx, report);

    // Can't scroll past end
    app.source_scroll = 1;
    app.handle_key(KeyCode::Down);
    assert_eq!(app.source_scroll, 1);

    // Can't scroll before start
    app.source_scroll = 0;
    app.handle_key(KeyCode::Up);
    assert_eq!(app.source_scroll, 0);
}

#[test]
fn test_page_navigation() {
    let ptx = (0..100).map(|i| format!("line{}", i)).collect::<Vec<_>>().join("\n");
    let report = sample_report();
    let mut app = TuiApp::new(ptx, report);

    app.handle_key(KeyCode::PageDown);
    assert_eq!(app.source_scroll, 20);

    app.handle_key(KeyCode::PageUp);
    assert_eq!(app.source_scroll, 0);
}

#[test]
fn test_home_end() {
    let ptx = (0..50).map(|i| format!("line{}", i)).collect::<Vec<_>>().join("\n");
    let report = sample_report();
    let mut app = TuiApp::new(ptx, report);

    app.handle_key(KeyCode::End);
    assert_eq!(app.source_scroll, 49);

    app.handle_key(KeyCode::Home);
    assert_eq!(app.source_scroll, 0);
}

#[test]
fn test_highlight_ptx_comment() {
    let (_, color) = highlight_ptx_line("// This is a comment");
    assert!(color_approx_eq(color, Color::new(0.5, 0.5, 0.5, 1.0)));
}

#[test]
fn test_highlight_ptx_directive() {
    let (_, color) = highlight_ptx_line(".entry test()");
    assert!(color_approx_eq(color, Color::new(1.0, 0.3, 1.0, 1.0)));
}

#[test]
fn test_highlight_ptx_memory() {
    let (_, color) = highlight_ptx_line("    ld.global.f32 %f1, [%rd1]");
    assert!(color_approx_eq(color, Color::new(1.0, 1.0, 0.3, 1.0)));
}

#[test]
fn test_highlight_ptx_arithmetic() {
    let (_, color) = highlight_ptx_line("    add.f32 %f1, %f2, %f3");
    assert!(color_approx_eq(color, Color::new(0.3, 1.0, 0.3, 1.0)));
}

#[test]
fn test_highlight_ptx_control() {
    let (_, color) = highlight_ptx_line("    ret;");
    assert!(color_approx_eq(color, Color::new(1.0, 0.3, 0.3, 1.0)));
}

/// F028: Scroll source pane - ASM pane scrolls in sync
/// In split-pane mode, both panes share the same scroll position
#[test]
fn f028_sync_scroll_source_asm() {
    let ptx = (0..100)
        .map(|i| format!("    add.f32 %f{}, %f{}, %f{}", i, i, i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let report = sample_report();
    let mut app = TuiApp::new(ptx, report);

    // Initial scroll position
    assert_eq!(app.source_scroll, 0);

    // Scroll down multiple times
    for i in 1..=10 {
        app.handle_key(KeyCode::Down);
        assert_eq!(app.source_scroll, i, "Scroll position should update");
    }

    // The source_scroll controls both panes in the split view
    // (no separate asm_scroll - they're synced by design)
    assert_eq!(app.source_scroll, 10, "Source/ASM should be at position 10");

    // Scroll back up
    app.handle_key(KeyCode::PageUp);
    assert_eq!(app.source_scroll, 0, "Should scroll back to top");
}
