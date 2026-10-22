// #5594: behavioural tests for widgets::render_register_panel, driven through
// the pub(super) render_sidebar entry point (the panel occupies rows 0..8).
// Included from tests.rs so `super::*` items (TuiApp, sample_report) are in scope.

fn render_panel_rows(regs: RegisterUsage, occupancy: f32) -> (CellBuffer, Vec<String>) {
    let mut report = sample_report();
    report.registers = regs;
    report.estimated_occupancy = occupancy;
    let app = TuiApp::new(".entry t() { ret; }".to_string(), report);
    let mut buffer = CellBuffer::new(44, 30);
    {
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        widgets::render_sidebar(&mut canvas, &app, 0.0, 0.0, 44, 30);
    }
    let rows = (0..8u16)
        .map(|y| {
            (0..44u16)
                .filter_map(|x| buffer.get(x, y).map(|c| c.symbol.to_string()))
                .collect::<String>()
        })
        .collect();
    (buffer, rows)
}

fn regs_with_total(f32_regs: u32) -> RegisterUsage {
    RegisterUsage {
        f32_regs,
        ..Default::default()
    }
}

fn border_fg(buffer: &CellBuffer) -> Color {
    buffer.get(0, 0).expect("cell (0,0) exists").fg
}

const PANEL_GREEN: Color = Color {
    r: 0.3,
    g: 1.0,
    b: 0.3,
    a: 1.0,
};
const PANEL_YELLOW: Color = Color {
    r: 1.0,
    g: 1.0,
    b: 0.3,
    a: 1.0,
};
const PANEL_RED: Color = Color {
    r: 1.0,
    g: 0.3,
    b: 0.3,
    a: 1.0,
};

#[test]
fn register_panel_lines_show_each_register_class() {
    let regs = RegisterUsage {
        f32_regs: 24,
        b32_regs: 18,
        b64_regs: 12,
        pred_regs: 4,
        ..Default::default()
    };
    let (_, rows) = render_panel_rows(regs, 0.875);
    assert!(rows[0].starts_with("┌─ Registers "), "{}", rows[0]);
    assert!(rows[0].trim_end().ends_with('┐'), "{}", rows[0]);
    assert!(rows[1].contains(".f32:  24 / 255"), "{}", rows[1]);
    assert!(rows[2].contains(".b32:  18 / 255"), "{}", rows[2]);
    assert!(rows[3].contains(".b64:  12 / 255"), "{}", rows[3]);
    assert!(rows[4].contains(".pred:  4 / 8"), "{}", rows[4]);
    assert!(rows[5].contains("Total: 58 -> 88% occ"), "{}", rows[5]);
    assert!(
        rows[7].starts_with('└') && rows[7].trim_end().ends_with('┘'),
        "{}",
        rows[7]
    );
}

#[test]
fn register_panel_total_below_64_is_green() {
    let (buf, _) = render_panel_rows(regs_with_total(63), 0.5);
    assert!(color_approx_eq(border_fg(&buf), PANEL_GREEN));
}

#[test]
fn register_panel_total_64_is_yellow() {
    let (buf, _) = render_panel_rows(regs_with_total(64), 0.5);
    assert!(color_approx_eq(border_fg(&buf), PANEL_YELLOW));
}

#[test]
fn register_panel_total_127_is_yellow() {
    let (buf, _) = render_panel_rows(regs_with_total(127), 0.5);
    assert!(color_approx_eq(border_fg(&buf), PANEL_YELLOW));
}

#[test]
fn register_panel_total_128_is_red() {
    let (buf, rows) = render_panel_rows(regs_with_total(128), 0.25);
    assert!(color_approx_eq(border_fg(&buf), PANEL_RED));
    assert!(rows[5].contains("Total: 128 -> 25% occ"), "{}", rows[5]);
}
