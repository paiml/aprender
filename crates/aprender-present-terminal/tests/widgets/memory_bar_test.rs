//! SPEC-024: MemoryBar Widget Interface Tests
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

use presentar_terminal::widgets::MemoryBar;
use presentar_terminal::Widget;

/// MemoryBar MUST be constructable with total bytes
#[test]
fn creates() {
    let _bar = MemoryBar::new(16_000_000_000);
}

/// MemoryBar MUST handle zero total
#[test]
fn handles_zero_total() {
    let _bar = MemoryBar::new(0);
}

/// MemoryBar MUST implement Widget trait
#[test]
fn implements_widget() {
    fn assert_widget<T: Widget>() {}
    assert_widget::<MemoryBar>();
}
