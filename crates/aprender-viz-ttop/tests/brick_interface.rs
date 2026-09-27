//! Brick Architecture interface tests for ttop v2.
//!
//! SPEC-024 ENFORCEMENT: Tests define the interface. Implementation follows.
//! This file MUST exist — lib.rs enforces via include_str!().
//!
//! Contract: panel-render-v1.yaml
//! References: Meyer (1992), Popper (1963), Ohno (1988)

#![allow(clippy::unwrap_used)]

use presentar_terminal::direct::CellBuffer;
use presentar_terminal::ptop::{ui, App};

// ============================================================================
// P1: No panic on minimum terminal size (40x10)
// ============================================================================

#[test]
fn p1_draw_min_size() {
    let app = App::new(true); // deterministic
    let mut buf = CellBuffer::new(40, 10);
    ui::draw(&app, &mut buf);
    // Contract: no panic
}

// ============================================================================
// P2: No panic on large terminal size (300x100)
// ============================================================================

#[test]
fn p2_draw_large_size() {
    let app = App::new(true);
    let mut buf = CellBuffer::new(300, 100);
    ui::draw(&app, &mut buf);
    // Contract: no panic
}

// ============================================================================
// P3: Deterministic mode produces identical frames
// ============================================================================

#[test]
fn p3_deterministic_parity() {
    let app = App::new(true);

    let mut buf1 = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf1);

    let mut buf2 = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf2);

    // Both frames should be identical in deterministic mode
    for y in 0..40u16 {
        for x in 0..120u16 {
            let c1 = buf1.get(x, y);
            let c2 = buf2.get(x, y);
            assert_eq!(
                c1.map(|c| c.symbol.as_str().to_string()),
                c2.map(|c| c.symbol.as_str().to_string()),
                "Deterministic parity failed at ({x}, {y})"
            );
        }
    }
}

// ============================================================================
// P4: Buffer bounds contract
// ============================================================================

#[test]
fn p4_buffer_bounds_contract() {
    let buf = CellBuffer::new(80, 24);
    // Contract: cells.len() == width * height
    assert_eq!(buf.width(), 80);
    assert_eq!(buf.height(), 24);
}

// ============================================================================
// P5: Draw produces visible content
// ============================================================================

#[test]
fn p5_draw_produces_content() {
    let app = App::new(true);
    let mut buf = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf);

    // At least some cells should have non-space content
    let mut has_content = false;
    for y in 0..40u16 {
        for x in 0..120u16 {
            if let Some(cell) = buf.get(x, y) {
                if cell.symbol.as_str() != " " && !cell.symbol.is_empty() {
                    has_content = true;
                    break;
                }
            }
        }
        if has_content {
            break;
        }
    }
    assert!(has_content, "Draw should produce visible content");
}

// ============================================================================
// P8-P10: Frame Budget Contracts (Spec Section 10)
// ============================================================================

#[test]
fn p8_frame_budget_80x24() {
    // Spec: render time (80x24) < 1ms
    let app = App::new(true);
    let mut buf = CellBuffer::new(80, 24);

    // Warm up
    ui::draw(&app, &mut buf);

    let start = std::time::Instant::now();
    for _ in 0..100 {
        ui::draw(&app, &mut buf);
    }
    let avg_us = start.elapsed().as_micros() / 100;
    assert!(avg_us < 5_000,
        "P8: 80x24 frame budget exceeded: {avg_us}us avg (target <1000us, tolerance 5x for debug build)");
}

#[test]
fn p9_frame_budget_120x40() {
    let app = App::new(true);
    let mut buf = CellBuffer::new(120, 40);
    ui::draw(&app, &mut buf);

    let start = std::time::Instant::now();
    for _ in 0..50 {
        ui::draw(&app, &mut buf);
    }
    let avg_us = start.elapsed().as_micros() / 50;
    assert!(
        avg_us < 16_000,
        "P9: 120x40 must render within 16ms frame budget: {avg_us}us avg"
    );
}

#[test]
fn p10_frame_budget_200x50() {
    let app = App::new(true);
    let mut buf = CellBuffer::new(200, 50);
    ui::draw(&app, &mut buf);

    let start = std::time::Instant::now();
    for _ in 0..20 {
        ui::draw(&app, &mut buf);
    }
    let avg_us = start.elapsed().as_micros() / 20;
    assert!(
        avg_us < 16_000,
        "P10: 200x50 must render within 16ms budget: {avg_us}us avg"
    );
}

// ============================================================================
// P6: Proptest — no panic for any terminal size
// ============================================================================

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn p6_draw_any_size(width in 10u16..300, height in 5u16..100) {
        let app = App::new(true);
        let mut buf = CellBuffer::new(width, height);
        ui::draw(&app, &mut buf);
        // Contract: no panic for any valid size
    }

    #[test]
    fn p7_draw_edge_sizes(width in 1u16..20, height in 1u16..10) {
        let app = App::new(true);
        let mut buf = CellBuffer::new(width, height);
        ui::draw(&app, &mut buf);
        // Contract: no panic even for tiny sizes
    }
}
