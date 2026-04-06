use super::*;

#[test]
fn test_input_validator_zero_size() {
    let validator = InputValidator::new();
    let result = validator.validate_bytes(&[]);
    assert!(matches!(result, Err(AdversarialError::ZeroSizeInput)));
}

#[test]
fn test_input_validator_max_size() {
    let validator = InputValidator::new().with_max_size(100);
    let data = vec![0u8; 200];
    let result = validator.validate_bytes(&data);
    assert!(matches!(
        result,
        Err(AdversarialError::MaxSizeExceeded { .. })
    ));
}

#[test]
fn test_checksum_consistency() {
    let data = b"hello world";
    let checksum1 = InputValidator::compute_checksum(data);
    let checksum2 = InputValidator::compute_checksum(data);
    assert_eq!(checksum1, checksum2);
}

#[test]
fn test_bit_flip_injector() {
    let injector = BitFlipInjector::new(42, 1);
    let data = vec![0u8; 100];
    let corrupted = injector.inject(&data);

    // At least one bit should be different
    assert_ne!(data, corrupted);
}

#[test]
fn test_checked_arithmetic_overflow() {
    let result = CheckedArithmetic::checked_add_i64(i64::MAX, 1);
    assert!(matches!(
        result,
        Err(AdversarialError::IntegerOverflow { .. })
    ));
}

#[test]
fn test_checked_div_zero() {
    let result = CheckedArithmetic::checked_div_f64(10.0, 0.0);
    assert!(matches!(
        result,
        Err(AdversarialError::DivisionByZero { .. })
    ));
}

#[test]
fn test_monotonic_clock() {
    let mut clock = MonotonicClock::new();
    let t1 = clock.tick().unwrap();
    let t2 = clock.tick().unwrap();
    assert!(t2 >= t1);
}

#[test]
fn test_resource_limiter_stack() {
    let mut limiter = ResourceLimiter::new().with_max_depth(5);

    for _ in 0..5 {
        assert!(limiter.enter_recursion().is_ok());
    }

    // 6th should fail
    assert!(matches!(
        limiter.enter_recursion(),
        Err(AdversarialError::StackOverflow { .. })
    ));
}

#[test]
fn test_cancellation_token() {
    let token = CancellationToken::new();
    assert!(!token.is_cancelled());
    assert!(token.check("test").is_ok());

    token.cancel();
    assert!(token.is_cancelled());
    assert!(matches!(
        token.check("test"),
        Err(AdversarialError::Cancelled { .. })
    ));
}

#[test]
fn test_config_validator_bounds() {
    let validator = ConfigValidator::new().with_bound("learning_rate", 0.0001, 1.0);

    assert!(validator.validate_numeric("learning_rate", 0.01).is_ok());
    assert!(matches!(
        validator.validate_numeric("learning_rate", 2.0),
        Err(AdversarialError::ConfigOutOfBounds { .. })
    ));
}

#[test]
fn test_recovery_handler() {
    let mut handler: RecoveryHandler<i32> = RecoveryHandler::new();

    // No checkpoint - recovery should fail
    assert!(handler.recover().is_err());

    // With checkpoint - recovery should work
    handler.checkpoint(42);
    assert_eq!(handler.recover().unwrap(), 42);
}
