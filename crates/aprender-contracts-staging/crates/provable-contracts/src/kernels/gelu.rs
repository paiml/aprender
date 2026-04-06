//! GELU kernel (standalone module).
//!
//! Matches `gelu-kernel-v1.yaml`.
//! `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
//!
//! The implementation is in `activation.rs` (`gelu_scalar`, `gelu_avx2`, `gelu_ptx`).
//! This module re-exports those functions and adds GELU-specific tests that go
//! beyond the generic activation tests.

#[cfg(target_arch = "x86_64")]
pub use super::activation::gelu_avx2;
pub use super::activation::{gelu_ptx, gelu_scalar};

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_gelu_zero_preservation() {
        let input = [0.0f32];
        let mut output = [999.0f32];
        gelu_scalar(&input, &mut output);
        assert!(
            (output[0]).abs() < 1e-7,
            "GELU(0) should be 0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_gelu_positive_nonneg() {
        let input = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        let mut output = [0.0f32; 6];
        gelu_scalar(&input, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(val >= 0.0, "GELU({}) = {} should be >= 0", input[i], val);
        }
    }

    #[test]
    fn test_gelu_asymptotic_linearity() {
        // For large positive x, GELU(x) ≈ x
        let input = [100.0f32];
        let mut output = [0.0f32];
        gelu_scalar(&input, &mut output);
        assert!(
            (output[0] - 100.0).abs() < 0.1,
            "GELU(100) should be ≈ 100, got {}",
            output[0]
        );
    }

    #[test]
    fn test_gelu_monotonic_positive() {
        // GELU is monotonically increasing for x > 0
        let input: Vec<f32> = (1..=20).map(|i| i as f32 * 0.5).collect();
        let mut output = vec![0.0f32; input.len()];
        gelu_scalar(&input, &mut output);

        for i in 1..output.len() {
            assert!(
                output[i] > output[i - 1],
                "GELU not monotonic at x={}: {:.6} vs {:.6}",
                input[i],
                output[i],
                output[i - 1]
            );
        }
    }

    #[test]
    fn test_gelu_lower_bound() {
        // GELU(x) >= -0.171 for all x
        let input: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let mut output = vec![0.0f32; input.len()];
        gelu_scalar(&input, &mut output);

        for (i, &val) in output.iter().enumerate() {
            assert!(
                val >= -0.18,
                "GELU({}) = {} below lower bound",
                input[i],
                val
            );
        }
    }

    proptest! {
        #[test]
        fn prop_gelu_finite(x in -100.0f32..100.0) {
            let input = [x];
            let mut output = [0.0f32];
            gelu_scalar(&input, &mut output);
            prop_assert!(output[0].is_finite(), "GELU({x}) not finite: {}", output[0]);
        }

        #[test]
        fn prop_gelu_positive_nonneg(x in 0.001f32..1000.0) {
            let input = [x];
            let mut output = [0.0f32];
            gelu_scalar(&input, &mut output);
            prop_assert!(output[0] >= 0.0, "GELU({x}) = {} should be >= 0", output[0]);
        }
    }
}
