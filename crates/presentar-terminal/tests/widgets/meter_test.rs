//! SPEC-024: Meter Widget Interface Tests
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

use presentar_terminal::widgets::Meter;
use presentar_terminal::Widget;

/// Meter MUST be constructable with value and max
#[test]
fn creates() {
    let _meter = Meter::new(75.0, 100.0);
}

/// Meter MUST handle various values
#[test]
fn handles_values() {
    let _zero = Meter::new(0.0, 100.0);
    let _full = Meter::new(100.0, 100.0);
    let _half = Meter::new(50.0, 100.0);
}

/// Meter MUST handle overflow (value > max)
#[test]
fn handles_overflow() {
    let _meter = Meter::new(150.0, 100.0);
}

/// Meter MUST implement Widget trait
#[test]
fn implements_widget() {
    fn assert_widget<T: Widget>() {}
    assert_widget::<Meter>();
}
