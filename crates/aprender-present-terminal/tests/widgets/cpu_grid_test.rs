//! SPEC-024: CpuGrid Widget Interface Tests
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

use presentar_terminal::widgets::CpuGrid;
use presentar_terminal::Widget;

/// CpuGrid MUST be constructable with core usage data
#[test]
fn creates() {
    let _grid = CpuGrid::new(vec![25.0, 50.0, 75.0, 100.0]);
}

/// CpuGrid MUST handle many cores (48+)
#[test]
fn handles_many_cores() {
    let percentages: Vec<f64> = (0..48).map(|i| (i as f64 * 2.0) % 100.0).collect();
    let _grid = CpuGrid::new(percentages);
}

/// CpuGrid MUST handle single core
#[test]
fn handles_single_core() {
    let _grid = CpuGrid::new(vec![50.0]);
}

/// CpuGrid MUST handle empty data
#[test]
fn handles_empty_data() {
    let _grid = CpuGrid::new(vec![]);
}

/// CpuGrid MUST implement Widget trait
#[test]
fn implements_widget() {
    fn assert_widget<T: Widget>() {}
    assert_widget::<CpuGrid>();
}
