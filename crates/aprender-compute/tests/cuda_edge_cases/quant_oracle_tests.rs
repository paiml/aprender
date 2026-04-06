// ============================================================================
// F4: Quantization Parity Oracle -- SIMD/GPU Numerical Accuracy
// ============================================================================

use trueno_cuda_edge::quant_oracle::{
    check_values_parity, BoundaryValueGenerator, ParityConfig, QuantFormat,
};

/// Test format-specific tolerances for trueno's quantization.
#[test]
fn quantization_format_tolerances() {
    // 4-bit quantization: ~5% tolerance
    assert!((QuantFormat::Q4K.tolerance() - 0.05).abs() < f64::EPSILON);

    // 5-bit quantization: ~2% tolerance
    assert!((QuantFormat::Q5K.tolerance() - 0.02).abs() < f64::EPSILON);

    // 6-bit quantization: ~1% tolerance
    assert!((QuantFormat::Q6K.tolerance() - 0.01).abs() < f64::EPSILON);

    // 8-bit quantization: ~0.5% tolerance
    assert!((QuantFormat::Q8_0.tolerance() - 0.005).abs() < f64::EPSILON);

    // F16: ~0.1% tolerance
    assert!((QuantFormat::F16.tolerance() - 0.001).abs() < f64::EPSILON);

    // F32: machine epsilon
    assert!((QuantFormat::F32.tolerance() - f64::EPSILON).abs() < f64::EPSILON);
}

/// Test boundary value generation for edge cases.
#[test]
fn boundary_value_generation() {
    let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);

    // Universal boundaries
    let universal = gen.universal_boundaries();
    assert!(universal.contains(&0.0));
    assert!(universal.iter().any(|v| v.is_nan()));
    assert!(universal.iter().any(|v| v.is_infinite()));

    // Format-specific boundaries
    let format_bounds = gen.format_boundaries();
    // Q4K has 16 levels x 2 (+/-) = 32 values
    assert_eq!(format_bounds.len(), 32);

    // All boundaries
    let all = gen.all_boundaries();
    assert_eq!(all.len(), universal.len() + format_bounds.len());
}

/// Test parity checking for CPU/GPU comparison.
#[test]
fn parity_check_cpu_gpu() {
    let config = ParityConfig::new(QuantFormat::Q4K);

    // Identical values: pass
    let cpu = vec![1.0, 2.0, 3.0];
    let gpu = vec![1.0, 2.0, 3.0];
    let report = check_values_parity(&cpu, &gpu, &config);
    assert!(report.passed());

    // Small difference within tolerance: pass
    let gpu_close = vec![1.01, 2.01, 3.01];
    let report = check_values_parity(&cpu, &gpu_close, &config);
    assert!(report.passed());

    // Large difference: fail
    let gpu_far = vec![1.0, 2.5, 3.0];
    let report = check_values_parity(&cpu, &gpu_far, &config);
    assert!(!report.passed());
    assert_eq!(report.violations.len(), 1);
}

/// Test NaN handling in parity checks.
#[test]
fn parity_nan_handling() {
    let config = ParityConfig::new(QuantFormat::F32);

    // NaN vs NaN: OK (both are NaN)
    let cpu = vec![f64::NAN];
    let gpu = vec![f64::NAN];
    let report = check_values_parity(&cpu, &gpu, &config);
    assert!(report.passed());

    // NaN vs number: violation
    let gpu_num = vec![1.0];
    let report = check_values_parity(&cpu, &gpu_num, &config);
    assert!(!report.passed());
}
