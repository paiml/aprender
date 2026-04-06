//! Integration tests for presentar_test_macros.
//!
//! Tests that the proc macros generate correct code.

use presentar_test_macros::presentar_test;

// =============================================================================
// Basic #[presentar_test] tests
// =============================================================================

#[presentar_test]
fn test_basic_macro() {
    assert_eq!(2 + 2, 4);
}

#[presentar_test(timeout = 1000)]
fn test_with_timeout() {
    // Verify timeout attribute is parsed
    assert_eq!(1 + 1, 2);
}

#[presentar_test(ignore)]
fn test_ignored() {
    // This test is ignored by default
    panic!("This should not run unless explicitly enabled");
}

#[presentar_test(should_panic)]
fn test_should_panic() {
    panic!("This panic is expected");
}

#[presentar_test(timeout = 2000, should_panic)]
fn test_combined_attrs() {
    panic!("Expected panic with timeout");
}

// =============================================================================
// Compile tests (ensure generated code is valid)
// =============================================================================

#[presentar_test]
fn test_with_setup() {
    let data = [1, 2, 3];
    assert_eq!(data.len(), 3);
}

#[presentar_test]
fn test_with_assertions() {
    let value = 42;
    assert!(value > 0);
    assert!(value < 100);
    assert_eq!(value, 42);
}

// =============================================================================
// Test attribute preservation
// =============================================================================

#[presentar_test]
#[allow(clippy::assertions_on_constants)]
fn test_preserves_other_attrs() {
    assert!(true);
}

// =============================================================================
// Test async support (when implemented)
// =============================================================================

#[presentar_test]
fn test_sync_function() {
    let result = std::thread::spawn(|| 42)
        .join()
        .expect("thread join failed");
    assert_eq!(result, 42);
}
