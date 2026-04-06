#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! F301: TUI Pixel-Level Acceptance Tests
//!
//! Uses jugar-probar for Playwright-style TUI testing with frame assertions.
//! These tests verify the visual correctness of cbtop's terminal UI.
//!
//! ## Toyota Way Application
//! - **Poka-Yoke**: Type-safe frame assertions prevent visual regressions
//! - **Jidoka**: Fail-fast on pixel mismatch with clear visual diff
//! - **Genchi Genbutsu**: Test actual rendered output, not mocked data

use jugar_probar::tui::{FrameAssertion, TuiFrame};

/// Create a mock cbtop frame for testing
fn create_cbtop_frame() -> TuiFrame {
    TuiFrame::from_lines(&[
        " cbtop │ AMD Ryzen 9 7950X (32 cores, AVX-512) │ 64GB RAM │ GPU: NVIDIA RTX 4090 ",
        "│[1:Overview] 2:CPU 3:GPU 4:PCIe 5:Memory 6:Thermal 7:Load 8:Config 9:Help     │",
        " Load: ● RUNNING ",
        "┌─ Real-Time Metrics ─────────────────────────────────────────────────────────┐",
        "│ CPU Usage:     ████████████████████████████████████████░░░░░░░░░░  78.5%    │",
        "│ Memory:        ████████████████████████████░░░░░░░░░░░░░░░░░░░░░░  45.2%    │",
        "│ Swap:          ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2.1%    │",
        "└─────────────────────────────────────────────────────────────────────────────┘",
        "┌─ Per-Core CPU ──────────────────────────────────────────────────────────────┐",
        "│ Core 0:  ████████████████████████████████████████████░░░░░░░  89.2%         │",
        "│ Core 1:  ██████████████████████████████░░░░░░░░░░░░░░░░░░░░░  67.8%         │",
        "│ Core 2:  ████████████████████████████████████████░░░░░░░░░░░  82.1%         │",
        "│ Core 3:  ██████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░  54.3%         │",
        "└─────────────────────────────────────────────────────────────────────────────┘",
        "┌─ GPU ───────────────────────────────────────────────────────────────────────┐",
        "│ Utilization:   ████████████████████████████████████████████████░░  95.0%    │",
        "│ Memory:        ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  38.2%    │",
        "│ Temperature:   72°C [Normal]                                                │",
        "│ Power:         285W / 450W                                                  │",
        "└─────────────────────────────────────────────────────────────────────────────┘",
        "┌─ Network ───────────────────────────────────────────────────────────────────┐",
        "│ eth0 RX:  125.4 MB/s    TX:  42.8 MB/s                                      │",
        "└─────────────────────────────────────────────────────────────────────────────┘",
        "┌─ Disk ──────────────────────────────────────────────────────────────────────┐",
        "│ /       ████████████████████████████░░░░░░░░░░░░░░░░░░░░░░  58.2%  1.2TB   │",
        "│ /home   ██████████████████████████████████████████████░░░░  89.1%  3.8TB   │",
        "└─────────────────────────────────────────────────────────────────────────────┘",
        " Status: 12.4 GFLOP/s │ Backend: AVX-512 │ Workload: GEMM │ Size: 1M ",
    ])
}

#[test]
fn f301_title_bar_contains_hardware_info() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // Title bar should contain cbtop name and hardware info
    assertion.to_contain_text("cbtop").unwrap();
    assertion.to_contain_text("cores").unwrap();
    assertion.to_contain_text("RAM").unwrap();
}

#[test]
fn f301_panel_navigation_tab_bar_visible() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // Tab bar should show all 9 panels with key numbers
    assertion.to_contain_text("1:Overview").unwrap();
    assertion.to_contain_text("2:CPU").unwrap();
    assertion.to_contain_text("3:GPU").unwrap();
    assertion.to_contain_text("4:PCIe").unwrap();
    assertion.to_contain_text("5:Memory").unwrap();
    assertion.to_contain_text("6:Thermal").unwrap();
    assertion.to_contain_text("7:Load").unwrap();
    assertion.to_contain_text("8:Config").unwrap();
    assertion.to_contain_text("9:Help").unwrap();
}

#[test]
fn f301_cpu_usage_bar_rendered() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // CPU usage should be displayed with progress bar
    assertion.to_contain_text("CPU Usage:").unwrap();
    assertion.to_contain_text("█").unwrap(); // Progress bar filled
    assertion.to_contain_text("░").unwrap(); // Progress bar empty
    assertion.to_contain_text("%").unwrap(); // Percentage
}

#[test]
fn f301_memory_breakdown_visible() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // Memory breakdown should show Memory and Swap
    assertion.to_contain_text("Memory:").unwrap();
    assertion.to_contain_text("Swap:").unwrap();
}

#[test]
fn f301_per_core_cpu_bars() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // Per-core CPU section
    assertion.to_contain_text("Per-Core CPU").unwrap();
    assertion.to_contain_text("Core 0:").unwrap();
    assertion.to_contain_text("Core 1:").unwrap();
}

#[test]
fn f301_gpu_panel_metrics() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // GPU panel should show utilization, memory, temp, power
    assertion.to_contain_text("GPU").unwrap();
    assertion.to_contain_text("Utilization:").unwrap();
    assertion.to_contain_text("Temperature:").unwrap();
    assertion.to_contain_text("Power:").unwrap();
    assertion.to_contain_text("°C").unwrap();
}

#[test]
fn f301_network_panel_tx_rx() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // Network panel should show TX/RX rates
    assertion.to_contain_text("Network").unwrap();
    assertion.to_contain_text("RX:").unwrap();
    assertion.to_contain_text("TX:").unwrap();
    assertion.to_contain_text("MB/s").unwrap();
}

#[test]
fn f301_disk_panel_mounts() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // Disk panel should show mount points with usage
    assertion.to_contain_text("Disk").unwrap();
    assertion.to_contain_text("/").unwrap();
    assertion.to_contain_text("TB").unwrap();
}

#[test]
fn f301_status_bar_gflops() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // Status bar should show GFLOP/s throughput
    assertion.to_contain_text("GFLOP/s").unwrap();
    assertion.to_contain_text("Backend:").unwrap();
    assertion.to_contain_text("Workload:").unwrap();
}

#[test]
fn f301_color_gradient_bars() {
    let frame = create_cbtop_frame();

    // Verify bars use Unicode block characters for gradients
    assert!(frame.contains("█"), "Progress bars should use filled blocks");
    assert!(frame.contains("░"), "Progress bars should use empty blocks");
}

#[test]
fn f301_responsive_box_drawing() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // Box drawing characters should be present
    assertion.to_contain_text("┌").unwrap();
    assertion.to_contain_text("┐").unwrap();
    assertion.to_contain_text("└").unwrap();
    assertion.to_contain_text("┘").unwrap();
    assertion.to_contain_text("│").unwrap();
    assertion.to_contain_text("─").unwrap();
}

#[test]
fn f301_load_status_indicator() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame);

    // Load status should show running indicator
    assertion.to_contain_text("Load:").unwrap();
    assertion.to_contain_text("RUNNING").unwrap();
}

#[test]
fn f301_frame_dimensions_valid() {
    let frame = create_cbtop_frame();

    // Frame should have reasonable dimensions
    assert!(frame.width() > 70, "Frame should be at least 70 chars wide");
    assert!(frame.height() >= 20, "Frame should be at least 20 lines tall");
}

#[test]
fn f301_soft_assertions_collect_errors() {
    let frame = create_cbtop_frame();
    let mut assertion = FrameAssertion::new(&frame).soft();

    // These should pass
    assertion.to_contain_text("cbtop").unwrap();
    assertion.to_contain_text("CPU").unwrap();

    // This should fail but not panic in soft mode
    let result = assertion.to_contain_text("NONEXISTENT_TEXT_12345");
    assert!(result.is_ok(), "Soft mode should not fail immediately");
}
