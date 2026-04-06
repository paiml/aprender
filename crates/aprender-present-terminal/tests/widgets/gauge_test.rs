//! SPEC-024: Gauge Widget Interface Tests
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

use presentar_terminal::widgets::Gauge;
use presentar_terminal::{Color, Widget};

/// Gauge MUST be constructable with value and max
#[test]
fn creates() {
    let _gauge = Gauge::new(75.0, 100.0);
}

/// Gauge MUST accept label
#[test]
fn accepts_label() {
    let gauge = Gauge::new(75.0, 100.0).with_label("CPU");
    let _ = gauge;
}

/// Gauge MUST accept color
#[test]
fn accepts_color() {
    let gauge = Gauge::new(75.0, 100.0).with_color(Color::GREEN);
    let _ = gauge;
}

/// Gauge MUST handle edge cases
#[test]
fn handles_edge_cases() {
    let _zero = Gauge::new(0.0, 100.0);
    let _full = Gauge::new(100.0, 100.0);
    let _overflow = Gauge::new(150.0, 100.0);
}

/// Gauge MUST implement Widget trait
#[test]
fn implements_widget() {
    fn assert_widget<T: Widget>() {}
    assert_widget::<Gauge>();
}
