//! Integration tests for the Quantization Parity Oracle (TCE-QUANT).
//!
//! Falsification tests from Section 8.2, F4 (claims 31-40).

#![allow(clippy::unwrap_used)]

use trueno_cuda_edge::quant_oracle::{
    check_values_parity, roundtrip_idempotence, BoundaryValueGenerator, MockQuantizer,
    ParityConfig, QuantFormat,
};

/// Claim 31: Q4K tolerance is 0.05
#[test]
fn claim_31_q4k_tolerance() {
    let tolerance = QuantFormat::Q4K.tolerance();
    assert!((tolerance - 0.05).abs() < f64::EPSILON, "Q4K tolerance must be 0.05");
}

/// Claim 32: Q5K tolerance is 0.02
#[test]
fn claim_32_q5k_tolerance() {
    let tolerance = QuantFormat::Q5K.tolerance();
    assert!((tolerance - 0.02).abs() < f64::EPSILON, "Q5K tolerance must be 0.02");
}

/// Claim 33: Q6K tolerance is 0.01
#[test]
fn claim_33_q6k_tolerance() {
    let tolerance = QuantFormat::Q6K.tolerance();
    assert!((tolerance - 0.01).abs() < f64::EPSILON, "Q6K tolerance must be 0.01");
}

/// Claim 34: Parity check detects differences above tolerance
#[test]
fn claim_34_parity_detects_violations() {
    let cpu = vec![1.0, 2.0, 3.0];
    let gpu = vec![1.0, 2.5, 3.0]; // 0.5 difference exceeds Q4K tolerance
    let config = ParityConfig::new(QuantFormat::Q4K);
    let report = check_values_parity(&cpu, &gpu, &config);

    assert!(!report.passed(), "Parity check must fail for large differences");
    assert_eq!(report.violations.len(), 1);
    assert_eq!(report.violations[0].index, 1);
}

/// Claim 35: NaN vs NaN is not a violation
#[test]
fn claim_35_nan_vs_nan_not_violation() {
    let cpu = vec![f64::NAN, 1.0];
    let gpu = vec![f64::NAN, 1.0];
    let config = ParityConfig::new(QuantFormat::F32);
    let report = check_values_parity(&cpu, &gpu, &config);
    assert!(report.passed(), "NaN vs NaN must not be a violation");
}

/// Claim 36: NaN vs number is a violation
#[test]
fn claim_36_nan_vs_number_is_violation() {
    let cpu = vec![f64::NAN];
    let gpu = vec![1.0];
    let config = ParityConfig::new(QuantFormat::F32);
    let report = check_values_parity(&cpu, &gpu, &config);
    assert!(!report.passed(), "NaN vs number must be a violation");
}

/// Claim 37: Boundary generator includes universal values (NaN, Inf, 0)
#[test]
fn claim_37_boundary_includes_universal() {
    let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);
    let bounds = gen.universal_boundaries();

    assert!(bounds.iter().any(|v| v.is_nan()), "Must include NaN");
    assert!(
        bounds.iter().any(|v| v.is_infinite() && v.is_sign_positive()),
        "Must include positive infinity"
    );
    assert!(
        bounds.iter().any(|v| v.is_infinite() && v.is_sign_negative()),
        "Must include negative infinity"
    );
    assert!(bounds.contains(&0.0), "Must include zero");
}

/// Claim 38: Roundtrip is idempotent for zero
#[test]
fn claim_38_roundtrip_idempotent_zero() {
    let quantizer = MockQuantizer::for_format(QuantFormat::Q4K);
    let result = roundtrip_idempotence(&quantizer, 0.0);
    assert!(result.idempotent, "Zero must be idempotent under roundtrip");
    assert!(
        (result.after_first - 0.0).abs() < f64::EPSILON,
        "Zero must remain zero after roundtrip"
    );
}

/// Claim 39: Tolerance is positive for all formats
#[test]
fn claim_39_tolerance_positive_all_formats() {
    let formats = [
        QuantFormat::Q4K,
        QuantFormat::Q5K,
        QuantFormat::Q6K,
        QuantFormat::Q8_0,
        QuantFormat::F16,
        QuantFormat::F32,
    ];

    for format in formats {
        assert!(format.tolerance() > 0.0, "Tolerance for {format} must be positive");
    }
}

/// Claim 40: Identical values always pass parity
#[test]
fn claim_40_identical_values_pass() {
    let values = vec![1.0, 2.0, 3.0, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
    let config = ParityConfig::new(QuantFormat::F32);
    let report = check_values_parity(&values, &values, &config);
    assert!(report.passed(), "Identical values must always pass parity");
}

/// Test `Q8_0` has 256 levels
#[test]
fn q8_0_has_256_levels() {
    assert_eq!(QuantFormat::Q8_0.levels(), 256);
}

/// Test format display strings
#[test]
fn format_display_strings() {
    assert_eq!(QuantFormat::Q4K.to_string(), "Q4_K");
    assert_eq!(QuantFormat::Q5K.to_string(), "Q5_K");
    assert_eq!(QuantFormat::Q6K.to_string(), "Q6_K");
    assert_eq!(QuantFormat::Q8_0.to_string(), "Q8_0");
}

/// Test parity violation rate calculation
#[test]
fn violation_rate_calculation() {
    let cpu = vec![1.0, 2.0, 3.0, 4.0];
    let gpu = vec![1.0, 3.0, 3.0, 5.0]; // 2 violations at index 1 and 3
    let config = ParityConfig::new(QuantFormat::Q4K);
    let report = check_values_parity(&cpu, &gpu, &config);
    assert!((report.violation_rate() - 0.5).abs() < 0.01);
}

/// Test format-specific boundary counts
#[test]
fn format_boundary_counts() {
    let gen_q4k = BoundaryValueGenerator::new(QuantFormat::Q4K);
    let bounds_q4k = gen_q4k.format_boundaries();
    assert_eq!(bounds_q4k.len(), 32); // 16 levels × 2

    let gen_q5k = BoundaryValueGenerator::new(QuantFormat::Q5K);
    let bounds_q5k = gen_q5k.format_boundaries();
    assert_eq!(bounds_q5k.len(), 64); // 32 levels × 2
}
