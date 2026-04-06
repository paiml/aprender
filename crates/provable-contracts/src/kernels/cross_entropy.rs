//! Cross-entropy loss kernel with log-softmax.
//!
//! Matches `cross-entropy-kernel-v1.yaml`.
//! loss = -sum(targets_i * log_softmax(logits)_i)
//! where log_softmax uses the log-sum-exp trick for numerical stability:
//!   log_softmax(x)_i = x_i - max(x) - log(sum(exp(x_j - max(x))))
//!
//! Each function provides one of three backends:
//! - `fn {name}_scalar(...)` — Pure Rust scalar reference (ground truth)
//! - `unsafe fn {name}_avx2(...)` — AVX2 SIMD implementation
//! - `fn cross_entropy_ptx() -> &'static str` — PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementations
// ────────────────────────────────────────────────────────────────────────────

/// Compute numerically stable log-softmax via log-sum-exp.
///
/// For each element: `output_i = logits_i - max(logits) - log(sum(exp(logits_j - max(logits))))`.
/// All output values are <= 0.
///
/// # Panics
/// Panics if `logits.len() != output.len()` or `logits.is_empty()`.
pub fn log_softmax_scalar(logits: &[f32], output: &mut [f32]) {
    assert_eq!(logits.len(), output.len(), "logits/output length mismatch");
    assert!(!logits.is_empty(), "logits must not be empty");

    // Find max for numerical stability
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute log-sum-exp: log(sum(exp(x_i - max)))
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let lse = max + sum_exp.ln();

    // log_softmax(x)_i = x_i - lse
    for (x, y) in logits.iter().zip(output.iter_mut()) {
        *y = x - lse;
    }
}

/// Compute cross-entropy loss: -sum(targets_i * log_softmax(logits)_i).
///
/// Targets should form a valid probability distribution (non-negative, sum to 1).
/// Returns a non-negative scalar loss value.
///
/// # Panics
/// Panics if `targets.len() != logits.len()` or `logits.is_empty()`.
pub fn cross_entropy_scalar(targets: &[f32], logits: &[f32]) -> f32 {
    assert_eq!(
        targets.len(),
        logits.len(),
        "targets/logits length mismatch"
    );
    assert!(!logits.is_empty(), "logits must not be empty");

    let mut log_sm = vec![0.0f32; logits.len()];
    log_softmax_scalar(logits, &mut log_sm);

    let loss: f32 = targets
        .iter()
        .zip(log_sm.iter())
        .map(|(&t, &ls)| t * ls)
        .sum();
    -loss
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementations
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 log-softmax — delegates to scalar (no hardware `log`/`exp` in AVX2).
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if `logits.len() != output.len()` or `logits.is_empty()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn log_softmax_avx2(logits: &[f32], output: &mut [f32]) {
    log_softmax_scalar(logits, output);
}

/// AVX2 cross-entropy — delegates to scalar (no hardware `log`/`exp` in AVX2).
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if `targets.len() != logits.len()` or `logits.is_empty()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn cross_entropy_avx2(targets: &[f32], logits: &[f32]) -> f32 {
    cross_entropy_scalar(targets, logits)
}

include!("cross_entropy_ptx.rs");

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::ulp::assert_ulp_eq;
    use super::*;
    use proptest::prelude::*;

    // ── log_softmax known-answer tests ────────────────────────────────────

    #[test]
    fn test_log_softmax_uniform() {
        // log_softmax([0, 0]) = [-ln2, -ln2]
        let logits = [0.0f32, 0.0];
        let mut output = vec![0.0f32; 2];
        log_softmax_scalar(&logits, &mut output);
        let expected = -(2.0f32.ln());
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "log_softmax([0,0])[0] should be -ln(2) ~ {expected}, got {}",
            output[0]
        );
        assert!(
            (output[1] - expected).abs() < 1e-6,
            "log_softmax([0,0])[1] should be -ln(2) ~ {expected}, got {}",
            output[1]
        );
    }

    #[test]
    fn test_log_softmax_dominant() {
        // log_softmax([100, 0]) ~ [0, -100] (approximately)
        let logits = [100.0f32, 0.0];
        let mut output = vec![0.0f32; 2];
        log_softmax_scalar(&logits, &mut output);
        assert!(
            output[0].abs() < 1e-4,
            "log_softmax for dominant class should be ~0, got {}",
            output[0]
        );
        assert!(
            output[1] < -99.0,
            "log_softmax for non-dominant class should be << 0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_log_softmax_single_element() {
        // log_softmax([x]) = [0] for any x (only one class)
        let logits = [42.0f32];
        let mut output = vec![0.0f32; 1];
        log_softmax_scalar(&logits, &mut output);
        assert!(
            output[0].abs() < 1e-6,
            "log_softmax of single element should be 0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_log_softmax_shift_invariance() {
        // log_softmax(x + c) = log_softmax(x) for any constant c
        let logits = [1.0f32, 2.0, 3.0];
        let shifted = [101.0f32, 102.0, 103.0];
        let mut out1 = vec![0.0f32; 3];
        let mut out2 = vec![0.0f32; 3];
        log_softmax_scalar(&logits, &mut out1);
        log_softmax_scalar(&shifted, &mut out2);
        for i in 0..3 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-5,
                "log_softmax should be shift-invariant, index {i}: {} vs {}",
                out1[i],
                out2[i]
            );
        }
    }

    #[test]
    fn test_log_softmax_three_classes() {
        // log_softmax([1, 2, 3]): known values
        let logits = [1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        log_softmax_scalar(&logits, &mut output);
        // softmax([1,2,3]) = [e^1, e^2, e^3] / (e^1 + e^2 + e^3)
        let e1 = 1.0f32.exp();
        let e2 = 2.0f32.exp();
        let e3 = 3.0f32.exp();
        let total = e1 + e2 + e3;
        let expected = [(e1 / total).ln(), (e2 / total).ln(), (e3 / total).ln()];
        for i in 0..3 {
            assert!(
                (output[i] - expected[i]).abs() < 1e-5,
                "log_softmax([1,2,3])[{i}]: expected {}, got {}",
                expected[i],
                output[i]
            );
        }
    }

    // ── cross_entropy known-answer tests ──────────────────────────────────

    #[test]
    fn test_cross_entropy_one_hot() {
        // CE(one_hot(0), [2, 1, 0]) = -log_softmax([2,1,0])[0]
        let targets = [1.0f32, 0.0, 0.0];
        let logits = [2.0f32, 1.0, 0.0];
        let loss = cross_entropy_scalar(&targets, &logits);

        let mut log_sm = vec![0.0f32; 3];
        log_softmax_scalar(&logits, &mut log_sm);
        let expected = -log_sm[0];

        assert!(
            (loss - expected).abs() < 1e-6,
            "CE with one-hot should be -log_softmax(correct_class), expected {expected}, got {loss}"
        );
    }

    #[test]
    fn test_cross_entropy_uniform_logits() {
        // CE(one_hot(0), [0,0]) = -log_softmax([0,0])[0] = ln(2)
        let targets = [1.0f32, 0.0];
        let logits = [0.0f32, 0.0];
        let loss = cross_entropy_scalar(&targets, &logits);
        let expected = 2.0f32.ln();
        assert!(
            (loss - expected).abs() < 1e-6,
            "CE with uniform logits and 2 classes should be ln(2) ~ {expected}, got {loss}"
        );
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        // CE approaches 0 when the correct class has a very high logit
        let targets = [1.0f32, 0.0, 0.0];
        let logits = [100.0f32, 0.0, 0.0];
        let loss = cross_entropy_scalar(&targets, &logits);
        assert!(
            loss < 1e-4,
            "CE with perfect prediction should be ~0, got {loss}"
        );
    }

    #[test]
    fn test_cross_entropy_soft_targets() {
        // CE with uniform targets over uniform logits = ln(n)
        let n = 4;
        let targets = vec![1.0 / n as f32; n];
        let logits = vec![0.0f32; n];
        let loss = cross_entropy_scalar(&targets, &logits);
        let expected = (n as f32).ln();
        assert!(
            (loss - expected).abs() < 1e-5,
            "CE(uniform, uniform) should be ln({n}) ~ {expected}, got {loss}"
        );
    }

    #[test]
    fn test_cross_entropy_second_class() {
        // CE(one_hot(1), [0,0,0]) = -log_softmax([0,0,0])[1] = ln(3)
        let targets = [0.0f32, 1.0, 0.0];
        let logits = [0.0f32, 0.0, 0.0];
        let loss = cross_entropy_scalar(&targets, &logits);
        let expected = 3.0f32.ln();
        assert!(
            (loss - expected).abs() < 1e-5,
            "CE(one_hot(1), [0,0,0]) should be ln(3) ~ {expected}, got {loss}"
        );
    }

    #[test]
    #[should_panic(expected = "targets/logits length mismatch")]
    fn test_cross_entropy_length_mismatch() {
        let targets = [1.0f32, 0.0];
        let logits = [1.0f32, 2.0, 3.0];
        cross_entropy_scalar(&targets, &logits);
    }

    #[test]
    #[should_panic(expected = "logits must not be empty")]
    fn test_cross_entropy_empty() {
        let targets: [f32; 0] = [];
        let logits: [f32; 0] = [];
        cross_entropy_scalar(&targets, &logits);
    }

    #[test]
    #[should_panic(expected = "logits must not be empty")]
    fn test_log_softmax_empty() {
        let logits: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        log_softmax_scalar(&logits, &mut output);
    }

    #[test]
    #[should_panic(expected = "logits/output length mismatch")]
    fn test_log_softmax_length_mismatch() {
        let logits = [1.0f32, 2.0];
        let mut output = vec![0.0f32; 3];
        log_softmax_scalar(&logits, &mut output);
    }

    // ── Property-based tests ──────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_log_softmax_all_nonpositive(
            logits in proptest::collection::vec(-100.0f32..100.0, 2..64),
        ) {
            let n = logits.len();
            let mut output = vec![0.0f32; n];
            log_softmax_scalar(&logits, &mut output);
            for (i, &y) in output.iter().enumerate() {
                prop_assert!(
                    y <= 0.0 + 1e-7,
                    "log_softmax should be <= 0, index {i}: got {y}"
                );
            }
        }

        #[test]
        fn prop_log_softmax_exp_sums_to_one(
            logits in proptest::collection::vec(-50.0f32..50.0, 2..32),
        ) {
            let n = logits.len();
            let mut output = vec![0.0f32; n];
            log_softmax_scalar(&logits, &mut output);
            let sum: f32 = output.iter().map(|&y| y.exp()).sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "exp(log_softmax) should sum to 1.0, got {sum}"
            );
        }

        #[test]
        fn prop_cross_entropy_nonnegative(
            logits in proptest::collection::vec(-20.0f32..20.0, 2..32),
        ) {
            let n = logits.len();
            // One-hot target at index 0
            let mut targets = vec![0.0f32; n];
            targets[0] = 1.0;
            let loss = cross_entropy_scalar(&targets, &logits);
            prop_assert!(
                loss >= -1e-6,
                "cross-entropy must be non-negative, got {loss}"
            );
        }

        #[test]
        fn prop_cross_entropy_finite(
            logits in proptest::collection::vec(-100.0f32..100.0, 2..32),
        ) {
            let n = logits.len();
            let mut targets = vec![0.0f32; n];
            targets[0] = 1.0;
            let loss = cross_entropy_scalar(&targets, &logits);
            prop_assert!(
                loss.is_finite(),
                "cross-entropy must be finite for finite inputs, got {loss}"
            );
        }

        #[test]
        fn prop_log_softmax_shift_invariant(
            logits in proptest::collection::vec(-50.0f32..50.0, 2..16),
            shift in -100.0f32..100.0,
        ) {
            let n = logits.len();
            let shifted: Vec<f32> = logits.iter().map(|&x| x + shift).collect();
            let mut out1 = vec![0.0f32; n];
            let mut out2 = vec![0.0f32; n];
            log_softmax_scalar(&logits, &mut out1);
            log_softmax_scalar(&shifted, &mut out2);
            for i in 0..n {
                prop_assert!(
                    (out1[i] - out2[i]).abs() < 1e-4,
                    "log_softmax should be shift-invariant, index {i}: {} vs {}",
                    out1[i], out2[i]
                );
            }
        }
    }

    // ── AVX2 parity tests ─────────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_log_softmax_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let logits: Vec<f32> = (-10..10).map(|i| i as f32 * 0.5).collect();
        let mut scalar_out = vec![0.0f32; logits.len()];
        let mut avx2_out = vec![0.0f32; logits.len()];

        log_softmax_scalar(&logits, &mut scalar_out);
        unsafe { log_softmax_avx2(&logits, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_cross_entropy_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let logits: Vec<f32> = (-10..10).map(|i| i as f32 * 0.3).collect();
        let n = logits.len();
        let mut targets = vec![0.0f32; n];
        targets[0] = 1.0;

        let scalar_loss = cross_entropy_scalar(&targets, &logits);
        let avx2_loss = unsafe { cross_entropy_avx2(&targets, &logits) };

        assert_ulp_eq(&[scalar_loss], &[avx2_loss], 2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_cross_entropy_avx2_soft_targets() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 10;
        let logits: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let targets: Vec<f32> = vec![1.0 / n as f32; n];

        let scalar_loss = cross_entropy_scalar(&targets, &logits);
        let avx2_loss = unsafe { cross_entropy_avx2(&targets, &logits) };

        assert_ulp_eq(&[scalar_loss], &[avx2_loss], 2);
    }

    // ── PTX structural tests ──────────────────────────────────────────────

    #[test]
    fn test_cross_entropy_ptx_structure() {
        let ptx = cross_entropy_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(
            ptx.contains(".entry cross_entropy_kernel"),
            "missing entry point"
        );
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains("ex2.approx.f32"), "missing ex2.approx for exp");
        assert!(ptx.contains("lg2.approx.f32"), "missing lg2.approx for log");
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }

    #[test]
    fn test_cross_entropy_ptx_nonempty() {
        assert!(!cross_entropy_ptx().is_empty());
    }

    #[test]
    fn test_cross_entropy_ptx_has_params() {
        let ptx = cross_entropy_ptx();
        assert!(ptx.contains(".param .u64 targets"), "missing targets param");
        assert!(ptx.contains(".param .u64 logits"), "missing logits param");
        assert!(ptx.contains(".param .u64 output"), "missing output param");
        assert!(ptx.contains(".param .u32 n"), "missing n param");
    }

    #[test]
    fn test_cross_entropy_ptx_has_shared_memory() {
        let ptx = cross_entropy_ptx();
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
    }

    #[test]
    fn test_cross_entropy_ptx_has_barrier() {
        let ptx = cross_entropy_ptx();
        assert!(ptx.contains("bar.sync"), "missing barrier synchronization");
    }
}
