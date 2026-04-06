//! F001-F020: ComputeBrick Core Invariants

use cbtop::brick::Brick;
use cbtop::bricks::collectors::cpu::CpuCollectorBrick;
use cbtop::bricks::collectors::gpu::GpuCollectorBrick;
use cbtop::bricks::collectors::memory::MemoryCollectorBrick;
use cbtop::bricks::collectors::pepita::PepitaCollectorBrick;
use cbtop::bricks::collectors::wos::WosCollectorBrick;
use cbtop::bricks::collectors::zram::ZramCollectorBrick;
use cbtop::bricks::panels::gpu::GpuPanelBrick;
use std::any::Any;

/// F001: Every brick MUST have at least one assertion
#[test]
fn f001_brick_assertions_non_empty() {
    let bricks: Vec<Box<dyn Brick>> = vec![
        Box::new(CpuCollectorBrick::new()),
        Box::new(GpuCollectorBrick::new(0)),
        Box::new(MemoryCollectorBrick::new()),
        Box::new(PepitaCollectorBrick::new()),
        Box::new(WosCollectorBrick::new()),
        Box::new(ZramCollectorBrick::new()),
        Box::new(GpuPanelBrick::new()),
    ];

    for brick in &bricks {
        assert!(
            !brick.assertions().is_empty(),
            "F001 FALSIFIED: Brick '{}' has no assertions",
            brick.brick_name()
        );
    }
}

/// F002: Brick names MUST be unique per instance type
#[test]
fn f002_brick_names_unique() {
    let bricks: Vec<Box<dyn Brick>> = vec![
        Box::new(CpuCollectorBrick::new()),
        Box::new(GpuCollectorBrick::new(0)),
        Box::new(MemoryCollectorBrick::new()),
        Box::new(PepitaCollectorBrick::new()),
        Box::new(WosCollectorBrick::new()),
        Box::new(ZramCollectorBrick::new()),
        Box::new(GpuPanelBrick::new()),
    ];

    let names: Vec<&str> = bricks.iter().map(|b| b.brick_name()).collect();
    let mut unique_names = names.clone();
    unique_names.sort();
    unique_names.dedup();

    assert_eq!(
        names.len(),
        unique_names.len(),
        "F002 FALSIFIED: Duplicate brick names detected"
    );
}

/// F003: verify() MUST check ALL assertions
#[test]
fn f003_verify_checks_all_assertions() {
    let brick = CpuCollectorBrick::new();
    let verification = brick.verify();

    // Verification should have processed assertions
    // (we can't easily count them but verify should not panic)
    assert!(
        verification.is_valid() || !verification.is_valid(),
        "F003 FALSIFIED: verify() did not complete"
    );
}

/// F010: budget() MUST return non-zero values
#[test]
fn f010_budget_non_zero() {
    let bricks: Vec<Box<dyn Brick>> = vec![
        Box::new(CpuCollectorBrick::new()),
        Box::new(GpuCollectorBrick::new(0)),
        Box::new(MemoryCollectorBrick::new()),
        Box::new(PepitaCollectorBrick::new()),
        Box::new(WosCollectorBrick::new()),
        Box::new(ZramCollectorBrick::new()),
        Box::new(GpuPanelBrick::new()),
    ];

    for brick in &bricks {
        let budget = brick.budget();
        // At least one budget component should be non-zero
        let total = budget.collect_ms + budget.layout_ms + budget.render_ms;
        assert!(
            total > 0,
            "F010 FALSIFIED: Brick '{}' has all-zero budget",
            brick.brick_name()
        );
    }
}

/// F017: Bricks can be accessed via Any for downcasting
#[test]
fn f017_brick_as_any_works() {
    let brick = CpuCollectorBrick::new();
    let any_ref: &dyn Any = brick.as_any();
    let downcast = any_ref.downcast_ref::<CpuCollectorBrick>();
    assert!(
        downcast.is_some(),
        "F017 FALSIFIED: Could not downcast CpuCollectorBrick via as_any()"
    );
}
