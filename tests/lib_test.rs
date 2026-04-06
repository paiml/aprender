//! Tests for top-level Database API

use trueno_db::{Backend, Database};

#[test]
fn test_database_builder() {
    // Test Database::builder() returns DatabaseBuilder
    let _builder = Database::builder();
}

#[test]
fn test_database_builder_with_backend() {
    // Test backend configuration
    let _builder = Database::builder().backend(Backend::CostBased);

    let _builder = Database::builder().backend(Backend::Gpu);

    let _builder = Database::builder().backend(Backend::Simd);
}

#[test]
fn test_database_builder_with_morsel_size() {
    // Test morsel size configuration
    let _builder = Database::builder().morsel_size_mb(256);
}

#[test]
fn test_database_builder_chain() {
    // Test method chaining
    let _builder = Database::builder().backend(Backend::CostBased).morsel_size_mb(512);
}

#[test]
fn test_database_build() {
    // Test building database
    let result = Database::builder().build();
    assert!(result.is_ok(), "Database build should succeed");
}

#[test]
fn test_database_build_with_config() {
    // Test building database with configuration
    let result = Database::builder().backend(Backend::Simd).morsel_size_mb(128).build();

    assert!(result.is_ok(), "Database build with config should succeed");
}

#[test]
fn test_backend_enum_clone() {
    // Test Backend enum is Clone
    let backend = Backend::CostBased;
    let _cloned = backend;
}

#[test]
fn test_backend_enum_copy() {
    // Test Backend enum is Copy
    let backend = Backend::Gpu;
    let _copied = backend;
    let _another = backend; // Should compile if Copy
}

#[test]
fn test_backend_enum_debug() {
    // Test Backend enum Debug implementation
    let backend = Backend::Simd;
    let debug_str = format!("{backend:?}");
    assert!(debug_str.contains("Simd"));
}
