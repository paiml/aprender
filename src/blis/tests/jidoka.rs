use super::super::*;

// ========================================================================
// Jidoka Tests
// ========================================================================

#[test]
fn test_jidoka_guard_catches_nan() {
    let guard = JidokaGuard::strict();
    let result = guard.validate(f32::NAN, 1.0);
    assert!(matches!(result, Err(JidokaError::NaNDetected { .. })));
}

#[test]
fn test_jidoka_guard_catches_inf() {
    let guard = JidokaGuard::strict();
    let result = guard.validate(f32::INFINITY, 1.0);
    assert!(matches!(result, Err(JidokaError::InfDetected { .. })));
}

#[test]
fn test_jidoka_guard_passes_valid() {
    let guard = JidokaGuard::strict();
    let result = guard.validate(1.0, 1.0);
    assert!(result.is_ok());
}

#[test]
fn test_jidoka_guard_catches_deviation() {
    let guard = JidokaGuard { epsilon: 0.01, check_special: true, sample_rate: 1 };
    let result = guard.validate(1.0, 2.0); // 50% error
    assert!(matches!(result, Err(JidokaError::NumericalDeviation { .. })));
}

#[test]
fn test_gemm_with_jidoka_nan_input() {
    let a = vec![1.0, f32::NAN, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0; 4];
    let guard = JidokaGuard::strict();

    let result = gemm_reference_with_jidoka(2, 2, 2, &a, &b, &mut c, &guard);
    assert!(matches!(result, Err(JidokaError::NaNDetected { .. })));
}
