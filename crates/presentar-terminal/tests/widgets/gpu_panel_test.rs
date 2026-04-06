//! SPEC-024: GpuPanel Widget Interface Tests
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

use presentar_terminal::widgets::GpuPanel;
use presentar_terminal::Widget;

/// GpuPanel MUST be constructable
#[test]
fn creates() {
    let _panel = GpuPanel::new();
}

/// GpuPanel MUST implement Widget trait
#[test]
fn implements_widget() {
    fn assert_widget<T: Widget>() {}
    assert_widget::<GpuPanel>();
}
