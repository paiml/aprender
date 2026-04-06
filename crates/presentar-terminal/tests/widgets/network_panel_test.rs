//! SPEC-024: NetworkPanel Widget Interface Tests
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

use presentar_terminal::widgets::{NetworkInterface, NetworkPanel};
use presentar_terminal::Widget;

/// NetworkPanel MUST be constructable
#[test]
fn creates() {
    let _panel = NetworkPanel::new();
}

/// NetworkInterface MUST be constructable
#[test]
fn interface_creates() {
    let _iface = NetworkInterface::new("eth0");
}

/// NetworkPanel MUST implement Widget trait
#[test]
fn implements_widget() {
    fn assert_widget<T: Widget>() {}
    assert_widget::<NetworkPanel>();
}
