//! SPEC-024: TitleBar Widget Interface Tests
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

use presentar_terminal::widgets::TitleBar;
use presentar_terminal::Widget;

/// TitleBar MUST be constructable with app name
#[test]
fn creates() {
    let _bar = TitleBar::new("ptop");
}

/// TitleBar MUST implement Widget trait
#[test]
fn implements_widget() {
    fn assert_widget<T: Widget>() {}
    assert_widget::<TitleBar>();
}
