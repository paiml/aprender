//! SPEC-024: ProcessTable Widget Interface Tests
//!
//! **TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.**

use presentar_terminal::widgets::{ProcessEntry, ProcessTable};
use presentar_terminal::Widget;

/// ProcessTable MUST be constructable
#[test]
fn creates() {
    let _table = ProcessTable::new();
}

/// ProcessEntry MUST be constructable with required fields
#[test]
fn entry_creates() {
    let _entry = ProcessEntry::new(1234, "root", 5.5, 2.3, "bash");
}

/// ProcessTable MUST implement Widget trait
#[test]
fn implements_widget() {
    fn assert_widget<T: Widget>() {}
    assert_widget::<ProcessTable>();
}
