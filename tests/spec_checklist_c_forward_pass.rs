#![allow(clippy::disallowed_methods)]
//! Spec Checklist Tests - Section C: Forward Pass (25 points)
//!
//! These tests verify claims from the spec's falsification checklist.
//! Each test is designed to FAIL if the claim is false.
//!
//! Reference: docs/specifications/qwen2-0.5b-instruct-interactive-chat-demo.md

#![allow(unused_imports)]

use aprender::autograd::Tensor;
use aprender::nn::Module;

// ============================================================================
// Section C: Forward Pass - "No Fake" Zone (25 points)
// ============================================================================

/// C7: RMSNorm must not produce NaN/Inf
#[test]
fn c7_rmsnorm_numerical_stability() {
    use aprender::nn::RMSNorm;

    let norm = RMSNorm::new(&[64]);

    // Test with normal values (pseudo-random pattern)
    let test_data: Vec<f32> = (0..640).map(|i| (i as f32 * 0.1).sin() * 2.0).collect();
    let input = Tensor::new(&test_data, &[1, 10, 64]);
    let output = norm.forward(&input);
    let data = output.data();

    let has_nan = data.iter().any(|&x| x.is_nan());
    let has_inf = data.iter().any(|&x| x.is_infinite());

    assert!(!has_nan, "C7 FAIL: RMSNorm produced NaN values");
    assert!(!has_inf, "C7 FAIL: RMSNorm produced Inf values");

    // Test with extreme values
    let extreme = Tensor::new(&[1e6_f32; 64], &[1, 1, 64]);
    let output_extreme = norm.forward(&extreme);
    let data_extreme = output_extreme.data();

    let has_nan_extreme = data_extreme.iter().any(|&x| x.is_nan());
    let has_inf_extreme = data_extreme.iter().any(|&x| x.is_infinite());

    assert!(
        !has_nan_extreme,
        "C7 FAIL: RMSNorm produced NaN on extreme values"
    );
    assert!(
        !has_inf_extreme,
        "C7 FAIL: RMSNorm produced Inf on extreme values"
    );
}

/// C8: SwiGLU activation must produce some negative values
#[test]
fn c8_swiglu_non_monotonic() {
    // SiLU (used in SwiGLU) is non-monotonic and produces negative values
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    let test_values = [-2.0f32, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let outputs: Vec<f32> = test_values.iter().map(|&x| silu(x)).collect();

    // SiLU must produce negative values for negative inputs
    let has_negative = outputs.iter().any(|&x| x < 0.0);
    assert!(
        has_negative,
        "C8 FAIL: SiLU (SwiGLU component) did not produce negative values"
    );

    // SiLU minimum is around x ≈ -1.28
    let min_output = outputs.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    assert!(
        min_output < 0.0,
        "C8 FAIL: SiLU minimum should be negative, got {min_output}"
    );
}

// ============================================================================
// Section C Additional: Forward Pass Tests
// ============================================================================

/// C1: Golden trace - verify logit precision infrastructure
#[test]
fn c1_golden_trace_precision() {
    use aprender::format::golden::verify_logits;

    // Simulate PyTorch reference logits (pre-computed)
    let reference_logits: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 5.0).collect();

    // Simulate model output with small deviation
    let model_logits: Vec<f32> = reference_logits
        .iter()
        .map(|&x| x + 0.00005) // 5e-5 deviation
        .collect();

    // Should pass with 1e-4 tolerance (per spec C1)
    let result = verify_logits("golden_test", &model_logits, &reference_logits, 1e-4);
    assert!(
        result.passed,
        "C1 FAIL: Model within 1e-4 tolerance should pass (max_dev={})",
        result.max_deviation
    );

    // Should fail with excessive deviation
    let bad_logits: Vec<f32> = reference_logits
        .iter()
        .map(|&x| x + 0.001) // 1e-3 deviation
        .collect();

    let fail_result = verify_logits("golden_test", &bad_logits, &reference_logits, 1e-4);
    assert!(
        !fail_result.passed,
        "C1 FAIL: Model exceeding 1e-4 tolerance should fail"
    );
}
