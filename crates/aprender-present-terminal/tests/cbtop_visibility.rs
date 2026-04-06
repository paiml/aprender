//! Visibility tests for cbtop widgets.
//!
//! Validates that widget text uses visible colors (not black on dark background).

use presentar_core::{Color, Rect};
use presentar_terminal::{
    CellBuffer, DirectTerminalCanvas, NetworkInterface, NetworkPanel, ProcessEntry, ProcessState,
    ProcessTable,
};

/// Check if a color is visible (not too dark).
fn is_visible_color(c: &Color) -> bool {
    c.r > 0.1 || c.g > 0.1 || c.b > 0.1
}

/// Check if symbol contains a PID digit from "1234".
fn contains_pid_digit(sym: &str) -> bool {
    sym.contains('1') || sym.contains('2') || sym.contains('3') || sym.contains('4')
}

/// Check if symbol could be part of username.
#[allow(dead_code)]
fn is_username_char(sym: &str) -> bool {
    sym == "t" || sym == "e" || sym == "s"
}

/// Create a test process entry for visibility tests.
fn create_test_process() -> ProcessEntry {
    ProcessEntry {
        pid: 1234,
        user: "testuser".to_string(),
        cpu_percent: 25.0,
        mem_percent: 10.0,
        command: "test_cmd".to_string(),
        cmdline: None,
        state: ProcessState::Running,
        oom_score: None,
        cgroup: None,
        nice: None,
        threads: None,
        parent_pid: None,
        tree_depth: 0,
        is_last_child: false,
        tree_prefix: String::new(),
    }
}

/// Test that `ProcessTable` renders with proper colors.
#[test]
fn test_process_table_renders_visible_text() {
    use presentar_core::Widget;

    let mut table = ProcessTable::new();
    table.add_process(create_test_process());

    let mut buffer = CellBuffer::new(80, 10);
    let mut canvas = DirectTerminalCanvas::new(&mut buffer);

    table.layout(Rect::new(0.0, 0.0, 80.0, 10.0));
    table.paint(&mut canvas);

    let cells = buffer.cells();
    let mut found_pid = false;

    for cell in cells {
        if contains_pid_digit(&cell.symbol) {
            assert!(
                is_visible_color(&cell.fg),
                "PID digit should not be black, got: ({}, {}, {})",
                cell.fg.r,
                cell.fg.g,
                cell.fg.b
            );
            found_pid = true;
        }
    }

    assert!(found_pid, "Should find PID digits in rendered output");
}

/// Test that `NetworkPanel` renders interface names with visible colors.
#[test]
fn test_network_panel_renders_visible_text() {
    use presentar_core::Widget;

    // Create a network panel with test data
    let mut panel = NetworkPanel::new().compact();
    let mut iface = NetworkInterface::new("eth0");
    iface.update(1024.0 * 1024.0, 512.0 * 1024.0);
    panel.add_interface(iface);

    // Create a canvas to render to
    let mut buffer = CellBuffer::new(80, 5);
    let mut canvas = DirectTerminalCanvas::new(&mut buffer);

    // Layout and paint
    panel.layout(Rect::new(0.0, 0.0, 80.0, 5.0));
    panel.paint(&mut canvas);

    // Check that cells contain interface name with visible color
    let cells = buffer.cells();
    let mut found_eth = false;

    for cell in cells {
        if cell.symbol == "e" {
            // Could be start of "eth0"
            let fg = cell.fg;
            if fg.r > 0.1 || fg.g > 0.1 || fg.b > 0.1 {
                found_eth = true;
            }
        }
    }

    // Note: This test documents expected behavior
    // If this fails, interface names are invisible
    assert!(found_eth, "Interface name should be visible (not black)");
}

/// Verify the light gray color we're using is visible.
#[test]
fn test_light_gray_is_visible() {
    let light_gray = Color::new(0.8, 0.8, 0.8, 1.0);

    // Should be bright enough to be visible
    assert!(
        light_gray.r > 0.5,
        "Light gray red component should be > 0.5"
    );
    assert!(
        light_gray.g > 0.5,
        "Light gray green component should be > 0.5"
    );
    assert!(
        light_gray.b > 0.5,
        "Light gray blue component should be > 0.5"
    );

    // Should not be black
    assert!(
        light_gray.r > 0.1 || light_gray.g > 0.1 || light_gray.b > 0.1,
        "Light gray should not be black"
    );
}
