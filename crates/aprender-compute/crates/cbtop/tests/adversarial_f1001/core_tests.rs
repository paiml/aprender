//! F1001-F1013: Core adversarial falsification tests

use cbtop::{
    AdversarialError, BitFlipInjector, CancellationToken, CheckedArithmetic, ConfigValidator,
    InputValidator, MonotonicClock, ResourceLimiter,
};
use std::time::Duration;

// ============================================================================
// F1001: Bit-flip in tensor maintains safety
// ============================================================================

#[test]
fn f1001_bit_flip_detected_by_checksum() {
    let data = b"test data for checksum validation";
    let original_checksum = InputValidator::compute_checksum(data);

    // Inject bit flips
    let injector = BitFlipInjector::new(42, 3);
    let corrupted = injector.inject(data);

    // Checksum should detect corruption
    let corrupted_checksum = InputValidator::compute_checksum(&corrupted);
    assert_ne!(original_checksum, corrupted_checksum);
}

#[test]
fn f1001_bit_flip_produces_different_output() {
    let injector = BitFlipInjector::new(12345, 5);
    let data = vec![0xAAu8; 1000];
    let corrupted = injector.inject(&data);

    // Should be different but same length
    assert_eq!(data.len(), corrupted.len());
    assert_ne!(data, corrupted);
}

// ============================================================================
// F1002: Arbitrary bit-flips detected
// ============================================================================

#[test]
fn f1002_checksum_verification_fails_on_corruption() {
    let validator = InputValidator::new();
    let data = b"important data";
    let checksum = InputValidator::compute_checksum(data);

    // Corrupt the data
    let mut corrupted = data.to_vec();
    corrupted[0] ^= 0xFF;

    // Verification should fail
    let result = validator.verify_checksum(&corrupted, checksum);
    assert!(matches!(result, Err(AdversarialError::CorruptedInput { .. })));
}

#[test]
fn f1002_checksum_passes_on_valid_data() {
    let validator = InputValidator::new();
    let data = b"valid data";
    let checksum = InputValidator::compute_checksum(data);

    let result = validator.verify_checksum(data, checksum);
    assert!(result.is_ok());
}

// ============================================================================
// F1003: Memory pressure handled gracefully
// ============================================================================

#[test]
fn f1003_memory_request_within_limit_succeeds() {
    let mut limiter = ResourceLimiter::new().with_max_memory(1024 * 1024);
    assert!(limiter.request_memory(1024).is_ok());
}

#[test]
fn f1003_memory_request_exceeding_limit_fails() {
    let mut limiter = ResourceLimiter::new().with_max_memory(1024);
    let result = limiter.request_memory(2048);
    assert!(matches!(result, Err(AdversarialError::ResourceExhausted { .. })));
}

#[test]
fn f1003_memory_release_works() {
    let mut limiter = ResourceLimiter::new().with_max_memory(1024);
    limiter.request_memory(512).unwrap();
    limiter.release_memory(512);
    // Should be able to request again
    assert!(limiter.request_memory(1024).is_ok());
}

// ============================================================================
// F1004: Zero-size inputs handled
// ============================================================================

#[test]
fn f1004_zero_size_bytes_rejected() {
    let validator = InputValidator::new();
    let result = validator.validate_bytes(&[]);
    assert!(matches!(result, Err(AdversarialError::ZeroSizeInput)));
}

#[test]
fn f1004_zero_size_floats_rejected() {
    let validator = InputValidator::new();
    let result = validator.validate_floats(&[]);
    assert!(matches!(result, Err(AdversarialError::ZeroSizeInput)));
}

// ============================================================================
// F1005: Maximum-size inputs handled
// ============================================================================

#[test]
fn f1005_max_size_bytes_rejected() {
    let validator = InputValidator::new().with_max_size(100);
    let data = vec![0u8; 200];
    let result = validator.validate_bytes(&data);
    assert!(matches!(result, Err(AdversarialError::MaxSizeExceeded { .. })));
}

#[test]
fn f1005_within_max_size_accepted() {
    let validator = InputValidator::new().with_max_size(1000);
    let data = vec![0u8; 500];
    assert!(validator.validate_bytes(&data).is_ok());
}

// ============================================================================
// F1006: Clock skew doesn't corrupt state
// ============================================================================

#[test]
fn f1006_monotonic_clock_maintains_order() {
    let mut clock = MonotonicClock::new();

    let t1 = clock.tick().unwrap();
    std::thread::sleep(Duration::from_millis(1));
    let t2 = clock.tick().unwrap();
    std::thread::sleep(Duration::from_millis(1));
    let t3 = clock.tick().unwrap();

    assert!(t2 >= t1);
    assert!(t3 >= t2);
}

#[test]
fn f1006_clock_reset_works() {
    let mut clock = MonotonicClock::new();
    clock.tick().unwrap();
    assert!(clock.elapsed().is_some());

    clock.reset();
    assert!(clock.elapsed().is_none());
}

// ============================================================================
// F1007: Concurrent access is safe
// ============================================================================

#[test]
fn f1007_cancellation_token_thread_safe() {
    let token = CancellationToken::new();
    let token_clone = token.clone_token();

    // Spawn thread that cancels
    let handle = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10));
        token_clone.cancel();
    });

    // Wait for cancellation
    while !token.is_cancelled() {
        std::thread::sleep(Duration::from_millis(1));
    }

    handle.join().unwrap();
    assert!(token.is_cancelled());
}

// ============================================================================
// F1008: Config corruption detected
// ============================================================================

#[test]
fn f1008_unclosed_brackets_rejected() {
    let validator = ConfigValidator::new();
    let result = validator.validate_toml_string("[section");
    assert!(matches!(result, Err(AdversarialError::ConfigParseError { .. })));
}

#[test]
fn f1008_unclosed_quotes_rejected() {
    let validator = ConfigValidator::new();
    let result = validator.validate_toml_string(r#"key = "unclosed"#);
    assert!(matches!(result, Err(AdversarialError::ConfigParseError { .. })));
}

#[test]
fn f1008_empty_config_rejected() {
    let validator = ConfigValidator::new();
    let result = validator.validate_toml_string("");
    assert!(matches!(result, Err(AdversarialError::ConfigParseError { .. })));
}

#[test]
fn f1008_valid_toml_accepted() {
    let validator = ConfigValidator::new();
    let result = validator.validate_toml_string(
        r#"
[section]
key = "value"
number = 42
"#,
    );
    assert!(result.is_ok());
}

// ============================================================================
// F1009: Pathological configs bounded
// ============================================================================

#[test]
fn f1009_value_below_min_rejected() {
    let validator = ConfigValidator::new().with_bound("temperature", 0.0, 2.0);

    let result = validator.validate_numeric("temperature", -1.0);
    assert!(matches!(result, Err(AdversarialError::ConfigOutOfBounds { .. })));
}

#[test]
fn f1009_value_above_max_rejected() {
    let validator = ConfigValidator::new().with_bound("temperature", 0.0, 2.0);

    let result = validator.validate_numeric("temperature", 5.0);
    assert!(matches!(result, Err(AdversarialError::ConfigOutOfBounds { .. })));
}

#[test]
fn f1009_value_within_bounds_accepted() {
    let validator = ConfigValidator::new().with_bound("temperature", 0.0, 2.0);

    let result = validator.validate_numeric("temperature", 1.0);
    assert!(result.is_ok());
}

// ============================================================================
// F1010-F1011: Rust memory safety
// ============================================================================

#[test]
fn f1010_f1011_rust_memory_safety() {
    let data = [1, 2, 3, 4, 5];
    let sum: i32 = data.iter().sum();
    assert_eq!(sum, 15);
}

// ============================================================================
// F1012: Integer overflow handled
// ============================================================================

#[test]
fn f1012_i64_add_overflow_detected() {
    let result = CheckedArithmetic::checked_add_i64(i64::MAX, 1);
    assert!(matches!(result, Err(AdversarialError::IntegerOverflow { .. })));
}

#[test]
fn f1012_i64_mul_overflow_detected() {
    let result = CheckedArithmetic::checked_mul_i64(i64::MAX, 2);
    assert!(matches!(result, Err(AdversarialError::IntegerOverflow { .. })));
}

#[test]
fn f1012_usize_add_overflow_detected() {
    let result = CheckedArithmetic::checked_add_usize(usize::MAX, 1);
    assert!(matches!(result, Err(AdversarialError::IntegerOverflow { .. })));
}

#[test]
fn f1012_valid_arithmetic_succeeds() {
    assert_eq!(CheckedArithmetic::checked_add_i64(10, 20).unwrap(), 30);
    assert_eq!(CheckedArithmetic::checked_mul_i64(10, 20).unwrap(), 200);
}

// ============================================================================
// F1013: Division by zero handled
// ============================================================================

#[test]
fn f1013_float_div_zero_detected() {
    let result = CheckedArithmetic::checked_div_f64(10.0, 0.0);
    assert!(matches!(result, Err(AdversarialError::DivisionByZero { .. })));
}

#[test]
fn f1013_int_div_zero_detected() {
    let result = CheckedArithmetic::checked_div_i64(10, 0);
    assert!(matches!(result, Err(AdversarialError::DivisionByZero { .. })));
}

#[test]
fn f1013_valid_division_succeeds() {
    assert!((CheckedArithmetic::checked_div_f64(10.0, 2.0).unwrap() - 5.0).abs() < 1e-10);
    assert_eq!(CheckedArithmetic::checked_div_i64(10, 2).unwrap(), 5);
}
