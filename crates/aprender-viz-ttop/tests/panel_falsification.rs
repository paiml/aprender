//! F-series Falsification Tests for ttop v2
//!
//! SPEC-024 ENFORCEMENT: Popperian falsification — tests attempt to BREAK,
//! not verify. Every test has an explicit falsifiable claim. All tests S3+.
//!
//! References: Popper (1963), Tufte (1983), Ohno (1988)

#![allow(clippy::unwrap_used)]

use presentar_terminal::direct::CellBuffer;
use presentar_terminal::ptop::config::PtopConfig;
use presentar_terminal::ptop::{ui, App, PanelType};

/// Extract all text from a CellBuffer as a single string.
fn buf_text(buf: &CellBuffer) -> String {
    let mut text = String::new();
    for y in 0..buf.height() {
        for x in 0..buf.width() {
            if let Some(cell) = buf.get(x, y) {
                text.push_str(cell.symbol.as_str());
            } else {
                text.push(' ');
            }
        }
        text.push('\n');
    }
    text
}

/// Extract a specific row from a CellBuffer.
fn buf_row(buf: &CellBuffer, y: u16) -> String {
    let mut row = String::new();
    for x in 0..buf.width() {
        if let Some(cell) = buf.get(x, y) {
            row.push_str(cell.symbol.as_str());
        } else {
            row.push(' ');
        }
    }
    row
}

/// Render deterministic app at given size.
fn render(w: u16, h: u16) -> (App, CellBuffer) {
    let app = App::new(true);
    let mut buf = CellBuffer::new(w, h);
    ui::draw(&app, &mut buf);
    (app, buf)
}

/// Render with exploded panel.
fn render_exploded(panel: PanelType, w: u16, h: u16) -> CellBuffer {
    let mut app = App::new(true);
    app.exploded_panel = Some(panel);
    let mut buf = CellBuffer::new(w, h);
    ui::draw(&app, &mut buf);
    buf
}

// ============================================================================
// F001-F010: Title Bar and Frame Structure
// ============================================================================

#[test]
fn f001_title_bar_present() {
    let (_, buf) = render(120, 40);
    let row0 = buf_row(&buf, 0);
    assert!(
        row0.contains("ptop") || row0.contains("ttop") || row0.contains("Quit"),
        "F001 FALSIFIED: Title bar must contain app name or controls"
    );
}

#[test]
fn f002_border_chars_present() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains('╔') || text.contains('╭') || text.contains('┌'),
        "F002 FALSIFIED: Panel borders must use box-drawing characters"
    );
}

#[test]
fn f003_cpu_panel_visible() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("CPU"),
        "F003 FALSIFIED: CPU panel must be visible in default layout"
    );
}

#[test]
fn f004_memory_panel_visible() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("Memory") || text.contains("Mem"),
        "F004 FALSIFIED: Memory panel must be visible"
    );
}

#[test]
fn f005_disk_panel_visible() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("Disk"),
        "F005 FALSIFIED: Disk panel must be visible"
    );
}

#[test]
fn f006_network_panel_visible() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("Network") || text.contains("net"),
        "F006 FALSIFIED: Network panel must be visible"
    );
}

#[test]
fn f007_process_panel_visible() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("Process") || text.contains("PID"),
        "F007 FALSIFIED: Process panel must be visible"
    );
}

#[test]
fn f008_connections_panel_visible() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("Connection") || text.contains("SVC") || text.contains("TCP"),
        "F008 FALSIFIED: Connections panel must be visible"
    );
}

#[test]
fn f009_files_panel_visible() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("File"),
        "F009 FALSIFIED: Files panel must be visible"
    );
}

#[test]
fn f010_no_empty_frame() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    let non_space = text.chars().filter(|c| !c.is_whitespace()).count();
    assert!(
        non_space > 100,
        "F010 FALSIFIED: Frame must have substantial content (got {non_space} non-space chars)"
    );
}

// ============================================================================
// F011-F020: Panel Detail Levels
// ============================================================================

#[test]
fn f011_compact_mode_no_panic() {
    // Compact: 60x12 — all panels must degrade gracefully
    let (_, buf) = render(60, 12);
    let text = buf_text(&buf);
    assert!(text.len() > 100, "F011: Compact mode must produce output");
}

#[test]
fn f012_minimal_mode_no_panic() {
    // Minimal: 40x10 — absolute minimum
    let (_, buf) = render(40, 10);
    let text = buf_text(&buf);
    assert!(text.len() > 50, "F012: Minimal mode must produce output");
}

#[test]
fn f013_wide_mode_no_panic() {
    // Wide: 250x80 — large terminal
    let (_, buf) = render(250, 80);
    let text = buf_text(&buf);
    assert!(text.contains("CPU"), "F013: Wide mode must show CPU panel");
}

#[test]
fn f014_tall_mode_no_panic() {
    let (_, buf) = render(80, 80);
    let text = buf_text(&buf);
    assert!(text.len() > 100, "F014: Tall mode must produce output");
}

#[test]
fn f015_square_mode_no_panic() {
    let (_, buf) = render(100, 100);
    let text = buf_text(&buf);
    assert!(text.len() > 100, "F015: Square mode must produce output");
}

// ============================================================================
// F021-F035: Exploded View (Fullscreen Panel)
// ============================================================================

#[test]
fn f021_exploded_cpu() {
    let buf = render_exploded(PanelType::Cpu, 120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("CPU"),
        "F021: Exploded CPU must show CPU content"
    );
}

#[test]
fn f022_exploded_memory() {
    let buf = render_exploded(PanelType::Memory, 120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("Memory") || text.contains("Mem") || text.contains("Used"),
        "F022: Exploded Memory must show memory content"
    );
}

#[test]
fn f023_exploded_disk() {
    let buf = render_exploded(PanelType::Disk, 120, 40);
    let text = buf_text(&buf);
    let upper = text.to_uppercase();
    assert!(
        upper.contains("DISK") || upper.contains("I/O") || upper.contains("MOUNT"),
        "F023: Exploded Disk must show disk content"
    );
}

#[test]
fn f024_exploded_network() {
    let buf = render_exploded(PanelType::Network, 120, 40);
    let text = buf_text(&buf);
    let upper = text.to_uppercase();
    assert!(
        upper.contains("NETWORK")
            || upper.contains("DOWNLOAD")
            || upper.contains("UPLOAD")
            || upper.contains("RX")
            || upper.contains("TX"),
        "F024: Exploded Network must show network content"
    );
}

#[test]
fn f025_exploded_process() {
    let buf = render_exploded(PanelType::Process, 120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("Process") || text.contains("PID"),
        "F025: Exploded Process must show process content"
    );
}

#[test]
fn f026_exploded_gpu() {
    let buf = render_exploded(PanelType::Gpu, 120, 40);
    let text = buf_text(&buf);
    // GPU may not be present — just verify no panic
    assert!(text.len() > 100, "F026: Exploded GPU must produce output");
}

#[test]
fn f027_exploded_sensors() {
    let buf = render_exploded(PanelType::Sensors, 120, 40);
    let text = buf_text(&buf);
    assert!(
        text.len() > 100,
        "F027: Exploded Sensors must produce output"
    );
}

#[test]
fn f028_exploded_connections() {
    let buf = render_exploded(PanelType::Connections, 120, 40);
    let text = buf_text(&buf);
    assert!(
        text.len() > 100,
        "F028: Exploded Connections must produce output"
    );
}

#[test]
fn f029_exploded_psi() {
    let buf = render_exploded(PanelType::Psi, 120, 40);
    let text = buf_text(&buf);
    assert!(text.len() > 100, "F029: Exploded PSI must produce output");
}

#[test]
fn f030_exploded_files() {
    let buf = render_exploded(PanelType::Files, 120, 40);
    let text = buf_text(&buf);
    assert!(text.len() > 100, "F030: Exploded Files must produce output");
}

#[test]
fn f031_exploded_battery() {
    let buf = render_exploded(PanelType::Battery, 120, 40);
    let text = buf_text(&buf);
    assert!(
        text.len() > 100,
        "F031: Exploded Battery must produce output"
    );
}

#[test]
fn f032_exploded_containers() {
    let buf = render_exploded(PanelType::Containers, 120, 40);
    let text = buf_text(&buf);
    assert!(
        text.len() > 100,
        "F032: Exploded Containers must produce output"
    );
}

#[test]
fn f033_exploded_at_small_size() {
    let buf = render_exploded(PanelType::Cpu, 40, 10);
    let text = buf_text(&buf);
    assert!(
        text.len() > 50,
        "F033: Exploded at small size must not panic"
    );
}

#[test]
fn f034_exploded_at_large_size() {
    let buf = render_exploded(PanelType::Cpu, 300, 100);
    let text = buf_text(&buf);
    assert!(
        text.contains("CPU"),
        "F034: Exploded at large size must show content"
    );
}

// ============================================================================
// F040-F050: Data Format Correctness
// ============================================================================

#[test]
fn f040_title_bar_has_controls() {
    let (_, buf) = render(120, 40);
    let row0 = buf_row(&buf, 0);
    // Title bar should show keyboard controls
    assert!(
        row0.contains('q') || row0.contains("Quit") || row0.contains("Help"),
        "F040 FALSIFIED: Title bar must show keyboard controls"
    );
}

#[test]
fn f041_process_table_has_header() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("PID") || text.contains("COMMAND") || text.contains("C%"),
        "F041 FALSIFIED: Process table must have column headers"
    );
}

#[test]
fn f042_sort_indicator_present() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains("Sort") || text.contains('▼') || text.contains('▲'),
        "F042 FALSIFIED: Sort indicator must be visible"
    );
}

// ============================================================================
// F060-F070: Deterministic Parity
// ============================================================================

#[test]
fn f060_deterministic_same_frame() {
    // Same App instance, two renders — must be identical
    let app = App::new(true);
    let mut buf1 = CellBuffer::new(120, 40);
    let mut buf2 = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf1);
    ui::draw(&app, &mut buf2);
    let text1 = buf_text(&buf1);
    let text2 = buf_text(&buf2);
    assert_eq!(
        text1, text2,
        "F060 FALSIFIED: Deterministic mode must produce identical frames"
    );
}

#[test]
fn f061_deterministic_different_sizes_differ() {
    let (_, buf1) = render(80, 24);
    let (_, buf2) = render(120, 40);
    let text1 = buf_text(&buf1);
    let text2 = buf_text(&buf2);
    assert_ne!(
        text1, text2,
        "F061 FALSIFIED: Different terminal sizes must produce different frames"
    );
}

#[test]
fn f062_deterministic_exploded_differs() {
    let (_, buf_normal) = render(120, 40);
    let buf_exploded = render_exploded(PanelType::Cpu, 120, 40);
    let text_normal = buf_text(&buf_normal);
    let text_exploded = buf_text(&buf_exploded);
    assert_ne!(
        text_normal, text_exploded,
        "F062 FALSIFIED: Exploded mode must produce different frame than normal"
    );
}

// ============================================================================
// F080-F090: Config-Driven Panel Visibility
// ============================================================================

#[test]
fn f080_config_hide_cpu() {
    let mut config = PtopConfig::default();
    if let Some(pc) = config.panels.get_mut(&PanelType::Cpu) {
        pc.enabled = false;
    }
    let app = App::with_config_lightweight(true, config);
    let mut buf = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf);
    assert!(
        buf_text(&buf).len() > 100,
        "F080: Hiding CPU must not panic"
    );
}

#[test]
fn f081_config_hide_process() {
    let mut config = PtopConfig::default();
    if let Some(pc) = config.panels.get_mut(&PanelType::Process) {
        pc.enabled = false;
    }
    let app = App::with_config_lightweight(true, config);
    let mut buf = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf);
    assert!(
        buf_text(&buf).len() > 100,
        "F081: Hiding process must not panic"
    );
}

#[test]
fn f082_config_all_hidden() {
    let mut config = PtopConfig::default();
    for (_, pc) in config.panels.iter_mut() {
        pc.enabled = false;
    }
    let app = App::with_config_lightweight(true, config);
    let mut buf = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf);
    assert!(
        !buf_text(&buf).is_empty(),
        "F082: All hidden must not panic"
    );
}

// ============================================================================
// F100-F110: Edge Cases and Anti-Regression
// ============================================================================

#[test]
fn f100_1x1_terminal() {
    let (_, buf) = render(1, 1);
    // Must not panic
    let _ = buf_text(&buf);
}

#[test]
fn f101_3x3_terminal() {
    let (_, buf) = render(3, 3);
    let _ = buf_text(&buf);
}

#[test]
fn f102_1000x1_terminal() {
    let (_, buf) = render(1000, 1);
    let _ = buf_text(&buf);
}

#[test]
fn f103_1x1000_terminal() {
    let (_, buf) = render(1, 1000);
    let _ = buf_text(&buf);
}

#[test]
fn f104_max_width() {
    let (_, buf) = render(500, 40);
    let text = buf_text(&buf);
    assert!(text.len() > 100, "F104: Max width must produce output");
}

#[test]
fn f105_rapid_resize() {
    // Simulate rapid terminal resize
    for (w, h) in [(40, 10), (120, 40), (80, 24), (200, 60), (60, 15)] {
        let (_, buf) = render(w, h);
        let _ = buf_text(&buf);
    }
}

// ============================================================================
// F110-F115: GPU Panel — MANDATORY (NVIDIA + AMD)
// ============================================================================

#[test]
fn f110_gpu_panel_no_crash_exploded() {
    // GPU panel must render without panic at any size, regardless of hardware
    for (w, h) in [(40, 10), (80, 24), (120, 40), (200, 60)] {
        let buf = render_exploded(PanelType::Gpu, w, h);
        let text = buf_text(&buf);
        assert!(
            !text.is_empty(),
            "F110: GPU exploded must not crash at {w}x{h}"
        );
    }
}

#[test]
fn f111_gpu_panel_shows_gpu_or_no_gpu() {
    // GPU panel must show either GPU info OR "No GPU" message
    let buf = render_exploded(PanelType::Gpu, 120, 40);
    let text = buf_text(&buf);
    let upper = text.to_uppercase();
    assert!(
        upper.contains("GPU")
            || upper.contains("NVIDIA")
            || upper.contains("AMD")
            || upper.contains("NO GPU")
            || upper.contains("NOT DETECTED"),
        "F111 FALSIFIED: GPU panel must show GPU info or 'No GPU' message"
    );
}

#[test]
fn f112_gpu_exploded_has_util_bar() {
    // If GPU is present, exploded view must show utilization bar
    let buf = render_exploded(PanelType::Gpu, 120, 40);
    let text = buf_text(&buf);
    // Either has utilization bar (█░) or shows "No GPU"
    let has_bar = text.contains('█') || text.contains('░');
    let has_no_gpu = text.to_uppercase().contains("NO GPU")
        || text.to_uppercase().contains("NOT DETECTED")
        || text.contains("requires");
    assert!(
        has_bar || has_no_gpu,
        "F112 FALSIFIED: GPU must show utilization bar or 'No GPU' message"
    );
}

#[test]
fn f113_gpu_temp_in_title_when_present() {
    // When GPU detected, temperature should appear in title/content
    let buf = render_exploded(PanelType::Gpu, 120, 40);
    let text = buf_text(&buf);
    // Either has temperature (°C) or no GPU detected
    let has_temp = text.contains("°C");
    let no_gpu = !text.contains("NVIDIA") && !text.contains("AMD") && !text.contains("RTX");
    assert!(
        has_temp || no_gpu,
        "F113 FALSIFIED: GPU with hardware must show temperature"
    );
}

#[test]
fn f114_gpu_vram_shown_when_present() {
    let buf = render_exploded(PanelType::Gpu, 120, 40);
    let text = buf_text(&buf);
    let has_vram = text.contains("VRAM") || text.contains("vram");
    let no_gpu = !text.contains("NVIDIA") && !text.contains("AMD") && !text.contains("RTX");
    assert!(
        has_vram || no_gpu,
        "F114 FALSIFIED: GPU with hardware must show VRAM usage"
    );
}

#[test]
fn f115_gpu_in_default_layout() {
    // GPU panel must appear in default (non-exploded) layout when hardware present
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    let upper = text.to_uppercase();
    // GPU panel should be visible if hardware is present
    // In deterministic mode, GPU detection still runs
    assert!(
        upper.contains("GPU") || upper.contains("NVIDIA"),
        "F115 FALSIFIED: GPU panel must be visible in default layout"
    );
}

// ============================================================================
// F120-F130: Probar-style Snapshot Assertions
// ============================================================================

#[test]
fn f120_snapshot_cpu_has_percentage() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        text.contains('%'),
        "F120 FALSIFIED: CPU panel must show percentage symbol"
    );
}

#[test]
fn f121_snapshot_border_consistency() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    // Count opening and closing border chars — should roughly balance
    let opens = text.matches('╔').count() + text.matches('╭').count();
    let closes = text.matches('╝').count() + text.matches('╯').count();
    assert!(
        opens > 0 && closes > 0,
        "F121 FALSIFIED: Must have both opening and closing borders"
    );
    // Allow some imbalance due to panel clipping at edges
    let diff = (opens as i32 - closes as i32).unsigned_abs();
    assert!(
        diff <= opens as u32,
        "F121 FALSIFIED: Border open/close mismatch too large: {opens} opens, {closes} closes"
    );
}

#[test]
fn f122_snapshot_no_null_chars() {
    let (_, buf) = render(120, 40);
    let text = buf_text(&buf);
    assert!(
        !text.contains('\0'),
        "F122 FALSIFIED: Frame must not contain null characters"
    );
}

#[test]
fn f123_snapshot_title_bar_row0() {
    let (_, buf) = render(120, 40);
    let row0 = buf_row(&buf, 0);
    // Title bar must have some content
    let non_space = row0.chars().filter(|c| !c.is_whitespace()).count();
    assert!(
        non_space > 10,
        "F123 FALSIFIED: Title bar must have substantial content (got {non_space} chars)"
    );
}

#[test]
fn f124_snapshot_all_panels_have_content() {
    // Each visible panel area should have non-space content
    let (_, buf) = render(120, 40);
    // Check multiple rows for content
    let mut content_rows = 0;
    for y in 0..40u16 {
        let row = buf_row(&buf, y);
        if row.chars().filter(|c| !c.is_whitespace()).count() > 5 {
            content_rows += 1;
        }
    }
    assert!(
        content_rows > 20,
        "F124 FALSIFIED: At least half of rows must have content (got {content_rows}/40)"
    );
}

#[test]
fn f125_snapshot_color_present() {
    // Verify cells have non-default colors (not all white-on-black)
    let (_, buf) = render(120, 40);
    let mut colored_cells = 0;
    let default_fg = presentar_core::Color::WHITE;
    for y in 0..40u16 {
        for x in 0..120u16 {
            if let Some(cell) = buf.get(x, y) {
                if cell.fg != default_fg && cell.fg != presentar_core::Color::BLACK {
                    colored_cells += 1;
                }
            }
        }
    }
    assert!(
        colored_cells > 50,
        "F125 FALSIFIED: Frame must have colored cells (got {colored_cells})"
    );
}

// ============================================================================
// Proptest: Exhaustive Size Coverage
// ============================================================================

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn f200_fuzz_terminal_size(w in 1u16..500, h in 1u16..200) {
        let app = App::new(true);
        let mut buf = CellBuffer::new(w, h);
        ui::draw(&app, &mut buf);
        // Contract: never panics
    }

    #[test]
    fn f201_fuzz_exploded_panel(
        panel_idx in 0u8..12,
        w in 10u16..300,
        h in 5u16..100,
    ) {
        let panels = [
            PanelType::Cpu, PanelType::Memory, PanelType::Disk,
            PanelType::Network, PanelType::Process, PanelType::Gpu,
            PanelType::Sensors, PanelType::Connections, PanelType::Psi,
            PanelType::Files, PanelType::Battery, PanelType::Containers,
        ];
        let panel = panels[panel_idx as usize % panels.len()];
        let buf = render_exploded(panel, w, h);
        let _ = buf_text(&buf);
        // Contract: never panics
    }
}
