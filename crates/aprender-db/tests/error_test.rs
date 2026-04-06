//! Tests for error types

use trueno_db::Error;

#[test]
fn test_gpu_init_failed_error() {
    let error = Error::GpuInitFailed("test error".to_string());
    let error_str = format!("{error}");
    assert!(error_str.contains("GPU initialization failed"));
    assert!(error_str.contains("Falling back to SIMD"));
}

#[test]
fn test_vram_exhausted_error() {
    let error = Error::VramExhausted("test exhaustion".to_string());
    let error_str = format!("{error}");
    assert!(error_str.contains("VRAM exhausted"));
    assert!(error_str.contains("Please report this issue"));
}

#[test]
fn test_backend_mismatch_error() {
    let error =
        Error::BackendMismatch { gpu_result: "42.5".to_string(), simd_result: "42.6".to_string() };
    let error_str = format!("{error}");
    assert!(error_str.contains("Backend equivalence failed"));
    assert!(error_str.contains("42.5"));
    assert!(error_str.contains("42.6"));
}

#[test]
fn test_parse_error() {
    let error = Error::ParseError("invalid SQL".to_string());
    let error_str = format!("{error}");
    assert!(error_str.contains("SQL parse error"));
    assert!(error_str.contains("invalid SQL"));
}

#[test]
fn test_storage_error() {
    let error = Error::StorageError("file not found".to_string());
    let error_str = format!("{error}");
    assert!(error_str.contains("Storage error"));
    assert!(error_str.contains("file not found"));
}

#[test]
fn test_queue_closed_error() {
    let error = Error::QueueClosed;
    let error_str = format!("{error}");
    assert!(error_str.contains("GPU transfer queue closed"));
}

#[test]
fn test_invalid_input_error() {
    let error = Error::InvalidInput("k must be positive".to_string());
    let error_str = format!("{error}");
    assert!(error_str.contains("Invalid input"));
    assert!(error_str.contains("k must be positive"));
}

#[test]
fn test_io_error_conversion() {
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let error: Error = io_error.into();
    let error_str = format!("{error}");
    assert!(error_str.contains("IO error"));
}

#[test]
fn test_other_error() {
    let error = Error::Other("custom error message".to_string());
    let error_str = format!("{error}");
    assert_eq!(error_str, "custom error message");
}

#[test]
fn test_error_debug() {
    let error = Error::QueueClosed;
    let debug_str = format!("{error:?}");
    assert!(debug_str.contains("QueueClosed"));
}

#[test]
fn test_result_type_alias() {
    // Test that Result<T> can be used
    #[allow(clippy::unnecessary_wraps)]
    fn returns_result() -> trueno_db::Result<i32> {
        Ok(42)
    }

    let result = returns_result();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_result_type_alias_error() {
    fn returns_error() -> trueno_db::Result<i32> {
        Err(Error::Other("test error".to_string()))
    }

    let result = returns_error();
    assert!(result.is_err());
}
