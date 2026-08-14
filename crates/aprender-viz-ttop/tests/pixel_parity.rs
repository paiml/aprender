//! Pixel Parity Tests — Spec Section 11, Phase 4
//!
//! Validates deterministic rendering produces identical frames across
//! invocations and that frame content matches expected patterns.
//!
//! CLD (Character-Level Diff): threshold < 0.001 (0.1%)

#![allow(clippy::unwrap_used)]

use presentar_terminal::direct::CellBuffer;
use presentar_terminal::ptop::{ui, App, PanelType};

fn buf_to_chars(buf: &CellBuffer) -> Vec<char> {
    let mut chars = Vec::with_capacity(buf.width() as usize * buf.height() as usize);
    for y in 0..buf.height() {
        for x in 0..buf.width() {
            if let Some(cell) = buf.get(x, y) {
                chars.push(cell.symbol.chars().next().unwrap_or(' '));
            } else {
                chars.push(' ');
            }
        }
    }
    chars
}

/// Character-Level Diff: fraction of cells that differ between two buffers.
fn cld(a: &CellBuffer, b: &CellBuffer) -> f64 {
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());
    let ca = buf_to_chars(a);
    let cb = buf_to_chars(b);
    let total = ca.len();
    if total == 0 {
        return 0.0;
    }
    let diffs = ca.iter().zip(cb.iter()).filter(|(a, b)| a != b).count();
    diffs as f64 / total as f64
}

// ============================================================================
// Deterministic Parity (same App → identical frames)
// ============================================================================

#[test]
fn parity_deterministic_80x24() {
    let app = App::new(true);
    let mut buf1 = CellBuffer::new(80, 24);
    let mut buf2 = CellBuffer::new(80, 24);
    ui::draw(&app, &mut buf1);
    ui::draw(&app, &mut buf2);
    let diff = cld(&buf1, &buf2);
    assert!(
        diff < 0.001,
        "CLD {diff:.6} exceeds 0.1% threshold at 80x24"
    );
}

#[test]
fn parity_deterministic_120x40() {
    let app = App::new(true);
    let mut buf1 = CellBuffer::new(120, 40);
    let mut buf2 = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf1);
    ui::draw(&app, &mut buf2);
    let diff = cld(&buf1, &buf2);
    assert!(
        diff < 0.001,
        "CLD {diff:.6} exceeds 0.1% threshold at 120x40"
    );
}

#[test]
fn parity_deterministic_200x60() {
    let app = App::new(true);
    let mut buf1 = CellBuffer::new(200, 60);
    let mut buf2 = CellBuffer::new(200, 60);
    ui::draw(&app, &mut buf1);
    ui::draw(&app, &mut buf2);
    let diff = cld(&buf1, &buf2);
    assert!(
        diff < 0.001,
        "CLD {diff:.6} exceeds 0.1% threshold at 200x60"
    );
}

// ============================================================================
// Cross-invocation parity (separate App instances)
// ============================================================================

#[test]
fn parity_cross_invocation_120x40() {
    // Two App::new(true) snapshot live system state, so they may differ slightly.
    // Allow up to 20% CLD (system metrics like CPU% change between calls).
    let app1 = App::new(true);
    let app2 = App::new(true);
    let mut buf1 = CellBuffer::new(120, 40);
    let mut buf2 = CellBuffer::new(120, 40);
    ui::draw(&app1, &mut buf1);
    ui::draw(&app2, &mut buf2);
    let diff = cld(&buf1, &buf2);
    assert!(
        diff < 0.20,
        "Cross-invocation CLD {diff:.6} exceeds 20% — frames wildly different"
    );
}

// ============================================================================
// Exploded panel parity
// ============================================================================

#[test]
fn parity_exploded_cpu() {
    // Same app, two draws — must be identical
    let mut app = App::new(true);
    app.exploded_panel = Some(PanelType::Cpu);
    let mut buf1 = CellBuffer::new(120, 40);
    let mut buf2 = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf1);
    ui::draw(&app, &mut buf2);
    let diff = cld(&buf1, &buf2);
    assert!(diff < 0.001, "Exploded CPU CLD {diff:.6} exceeds threshold");
}

#[test]
fn parity_exploded_gpu() {
    let mut app = App::new(true);
    app.exploded_panel = Some(PanelType::Gpu);
    let mut buf1 = CellBuffer::new(120, 40);
    let mut buf2 = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf1);
    ui::draw(&app, &mut buf2);
    let diff = cld(&buf1, &buf2);
    assert!(diff < 0.001, "Exploded GPU CLD {diff:.6} exceeds threshold");
}

#[test]
fn parity_exploded_memory() {
    let mut app = App::new(true);
    app.exploded_panel = Some(PanelType::Memory);
    let mut buf1 = CellBuffer::new(120, 40);
    let mut buf2 = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf1);
    ui::draw(&app, &mut buf2);
    let diff = cld(&buf1, &buf2);
    assert!(
        diff < 0.001,
        "Exploded Memory CLD {diff:.6} exceeds threshold"
    );
}

// ============================================================================
// Content structure parity (frame has expected structure)
// ============================================================================

#[test]
fn parity_title_bar_position() {
    let app = App::new(true);
    let mut buf = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf);
    // Title bar is always row 0
    let mut row0_content = 0;
    for x in 0..120u16 {
        if let Some(cell) = buf.get(x, 0) {
            if cell.symbol.as_str() != " " {
                row0_content += 1;
            }
        }
    }
    assert!(
        row0_content > 20,
        "Title bar (row 0) must have content: {row0_content} chars"
    );
}

#[test]
fn parity_panel_borders_aligned() {
    let app = App::new(true);
    let mut buf = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf);
    // Panels should have vertical borders (│ or ║) in consistent columns
    let mut border_cols = std::collections::HashSet::new();
    for y in 1..39u16 {
        for x in 0..120u16 {
            if let Some(cell) = buf.get(x, y) {
                let s = cell.symbol.as_str();
                if s == "│" || s == "║" {
                    border_cols.insert(x);
                }
            }
        }
    }
    // Should have at least 2 vertical border columns (panel separators)
    assert!(
        border_cols.len() >= 2,
        "Must have vertical panel borders: found {} columns",
        border_cols.len()
    );
}

#[test]
fn parity_no_truncated_unicode() {
    let app = App::new(true);
    let mut buf = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf);
    // No cell should contain replacement character U+FFFD
    for y in 0..40u16 {
        for x in 0..120u16 {
            if let Some(cell) = buf.get(x, y) {
                assert!(
                    !cell.symbol.contains('\u{FFFD}'),
                    "Replacement char at ({x},{y}): truncated unicode"
                );
            }
        }
    }
}
