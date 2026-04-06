//! API Fault Injection Tests (Spec 1.5.1)
//!
//! Tests error propagation in the API layer when underlying operations fail.
//! Uses mock/faulty data sources to verify graceful error handling.
//!
//! Coverage targets:
//! - Model loading errors
//! - State initialization failures
//! - Request validation errors
//! - Response serialization errors

use realizar::api::AppState;
use realizar::error::RealizarError;
use std::sync::Arc;

// ============================================================================
// A. AppState Initialization Error Tests
// ============================================================================

#[test]
fn test_demo_state_creation() {
    // Demo state should always succeed (no external dependencies)
    let result = AppState::demo();
    assert!(result.is_ok(), "Demo state should create successfully");
}

#[test]
fn test_state_without_model_is_demo() {
    let state = AppState::demo().expect("Demo should work");
    // Demo state has limited functionality
    assert!(!state.has_quantized_model());
    assert!(!state.has_gpu_model());
    assert!(!state.has_cached_model());
}

// ============================================================================
// B. State Method Error Propagation Tests
// ============================================================================

#[test]
fn test_quantized_model_none_when_not_set() {
    let state = AppState::demo().expect("Demo should work");
    assert!(state.quantized_model().is_none());
}

#[test]
fn test_gpu_model_none_when_not_set() {
    let state = AppState::demo().expect("Demo should work");
    assert!(state.gpu_model().is_none());
}

#[test]
fn test_cached_model_none_when_not_set() {
    let state = AppState::demo().expect("Demo should work");
    assert!(state.cached_model().is_none());
}

#[test]
fn test_dispatch_metrics_none_when_not_set() {
    let state = AppState::demo().expect("Demo should work");
    assert!(state.dispatch_metrics().is_none());
}

#[test]
fn test_batch_request_tx_none_when_not_set() {
    let state = AppState::demo().expect("Demo should work");
    assert!(state.batch_request_tx().is_none());
}

#[test]
fn test_batch_config_none_when_not_set() {
    let state = AppState::demo().expect("Demo should work");
    assert!(state.batch_config().is_none());
}

#[test]
fn test_batch_enabled_false_when_not_configured() {
    let state = AppState::demo().expect("Demo should work");
    assert!(!state.batch_enabled());
}

// ============================================================================
// C. Error Type Coverage Tests
// ============================================================================

#[test]
fn test_error_display_shape_mismatch() {
    let err = RealizarError::ShapeMismatch {
        expected: vec![10, 20],
        actual: vec![10, 30],
    };
    let msg = err.to_string();
    assert!(msg.contains("Shape mismatch") || msg.contains("shape"));
}

#[test]
fn test_error_display_invalid_shape() {
    let err = RealizarError::InvalidShape {
        reason: "negative dimension".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("Invalid shape") || msg.contains("negative"));
}

#[test]
fn test_error_display_data_shape_mismatch() {
    let err = RealizarError::DataShapeMismatch {
        data_size: 100,
        shape: vec![10, 20],
        expected: 200,
    };
    let msg = err.to_string();
    assert!(msg.contains("100") || msg.contains("200"));
}

#[test]
fn test_error_display_invalid_dimension() {
    let err = RealizarError::InvalidDimension { dim: 5, ndim: 3 };
    let msg = err.to_string();
    assert!(msg.contains('5') || msg.contains('3') || msg.contains("dimension"));
}

#[test]
fn test_error_display_matmul_mismatch() {
    let err = RealizarError::MatmulDimensionMismatch {
        m: 10,
        k: 20,
        k2: 30,
        n: 40,
    };
    let msg = err.to_string();
    assert!(msg.contains("10") || msg.contains("dimension") || msg.contains("mismatch"));
}

#[test]
fn test_error_display_unsupported_operation() {
    let err = RealizarError::UnsupportedOperation {
        operation: "test_op".to_string(),
        reason: "not implemented".to_string(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("test_op") || msg.contains("not implemented") || msg.contains("Unsupported")
    );
}

// ============================================================================
// D. Faulty Reader Simulation
// ============================================================================

/// A reader that fails after N bytes
struct FaultyReader {
    data: Vec<u8>,
    position: usize,
    fail_after: usize,
}

impl FaultyReader {
    fn new(data: Vec<u8>, fail_after: usize) -> Self {
        Self {
            data,
            position: 0,
            fail_after,
        }
    }
}

impl std::io::Read for FaultyReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.position >= self.fail_after {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Simulated read failure",
            ));
        }

        let remaining = self.fail_after.saturating_sub(self.position);
        let to_read = buf
            .len()
            .min(remaining)
            .min(self.data.len() - self.position);

        if to_read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Simulated EOF",
            ));
        }

        buf[..to_read].copy_from_slice(&self.data[self.position..self.position + to_read]);
        self.position += to_read;
        Ok(to_read)
    }
}

#[test]
fn test_faulty_reader_fails_immediately() {
    let reader = FaultyReader::new(vec![1, 2, 3, 4], 0);
    let mut buf = [0u8; 4];
    let result = std::io::Read::read(&mut { reader }, &mut buf);
    assert!(result.is_err());
}

#[test]
fn test_faulty_reader_fails_after_partial() {
    let mut reader = FaultyReader::new(vec![1, 2, 3, 4, 5, 6, 7, 8], 4);
    let mut buf = [0u8; 8];

    // First read succeeds (partial)
    let n = std::io::Read::read(&mut reader, &mut buf[..4]).unwrap();
    assert_eq!(n, 4);

    // Second read fails
    let result = std::io::Read::read(&mut reader, &mut buf[4..]);
    assert!(result.is_err());
}

#[test]
fn test_faulty_reader_with_gguf_header() {
    use realizar::gguf::GGUFModel;

    // Create valid GGUF header
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&0x4655_4747u32.to_le_bytes()); // GGUF magic
    data[4..8].copy_from_slice(&3u32.to_le_bytes()); // version 3

    // Test truncation at various points
    for fail_point in [0, 4, 8, 12, 16, 20, 23] {
        let truncated = &data[..fail_point.min(data.len())];
        let result = GGUFModel::from_bytes(truncated);
        if fail_point < 24 {
            assert!(result.is_err(), "Should fail with {} bytes", fail_point);
        }
    }
}

// ============================================================================
// E. Concurrent Access Safety Tests
// ============================================================================

#[test]
fn test_state_clone_is_safe() {
    let state = AppState::demo().expect("Demo should work");
    let state_clone = state.clone();

    // Both should be usable
    assert!(!state.has_quantized_model());
    assert!(!state_clone.has_quantized_model());
}

// ============================================================================
// F. Memory Safety Tests
// ============================================================================

#[test]
fn test_state_drop_is_safe() {
    {
        let state = AppState::demo().expect("Demo should work");
        // State should drop without panic
        drop(state);
    }
    // If we get here, drop was safe
}

#[test]
fn test_nested_arc_state() {
    let state = AppState::demo().expect("Demo should work");
    let arc_state = Arc::new(state);
    let arc_clone = Arc::clone(&arc_state);

    // Both refs should work
    assert!(!arc_state.has_quantized_model());
    assert!(!arc_clone.has_quantized_model());

    // Drop one
    drop(arc_clone);

    // Original should still work
    assert!(!arc_state.has_quantized_model());
}

// ============================================================================
// G. Error Chaining Tests
// ============================================================================

#[test]
fn test_error_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<RealizarError>();
    assert_sync::<RealizarError>();
}

#[test]
fn test_error_debug_impl() {
    let err = RealizarError::InvalidShape {
        reason: "test".to_string(),
    };
    let debug_str = format!("{:?}", err);
    assert!(!debug_str.is_empty());
}

#[test]
fn test_error_display_impl() {
    let err = RealizarError::InvalidShape {
        reason: "test".to_string(),
    };
    let display_str = format!("{}", err);
    assert!(!display_str.is_empty());
}

// ============================================================================
// H. GGUF Error Path Integration
// ============================================================================

#[test]
fn test_gguf_from_bytes_empty() {
    use realizar::gguf::GGUFModel;
    let result = GGUFModel::from_bytes(&[]);
    assert!(result.is_err());
}

#[test]
fn test_gguf_from_bytes_too_small() {
    use realizar::gguf::GGUFModel;
    let result = GGUFModel::from_bytes(&[1, 2, 3, 4]);
    assert!(result.is_err());
}

#[test]
fn test_gguf_from_bytes_wrong_magic() {
    use realizar::gguf::GGUFModel;
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&0x12345678u32.to_le_bytes());
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_gguf_from_bytes_wrong_version() {
    use realizar::gguf::GGUFModel;
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(&0x4655_4747u32.to_le_bytes()); // GGUF magic
    data[4..8].copy_from_slice(&1u32.to_le_bytes()); // version 1 (unsupported)
    let result = GGUFModel::from_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// I. State Builder Pattern Tests
// ============================================================================

#[test]
fn test_state_has_methods_consistency() {
    let state = AppState::demo().expect("Demo should work");

    // All "has_" methods should be consistent with their corresponding getters
    if state.has_quantized_model() {
        assert!(state.quantized_model().is_some());
    } else {
        assert!(state.quantized_model().is_none());
    }

    if state.has_gpu_model() {
        assert!(state.gpu_model().is_some());
    } else {
        assert!(state.gpu_model().is_none());
    }

    if state.has_cached_model() {
        assert!(state.cached_model().is_some());
    } else {
        assert!(state.cached_model().is_none());
    }
}

// ============================================================================
// J. Edge Case Value Tests
// ============================================================================

#[test]
fn test_error_with_empty_strings() {
    let err = RealizarError::InvalidShape {
        reason: String::new(),
    };
    let _ = err.to_string();
}

#[test]
fn test_error_with_unicode() {
    let err = RealizarError::InvalidShape {
        reason: "日本語エラー 🔥".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("日本語") || msg.contains("Invalid"));
}

#[test]
fn test_error_with_long_string() {
    let long_reason = "x".repeat(10000);
    let err = RealizarError::InvalidShape {
        reason: long_reason,
    };
    let msg = err.to_string();
    assert!(!msg.is_empty());
}
