// SHIP-TWO-001 — `cuda-classify-training-v1` algorithm-level PARTIAL
// discharge for FALSIFY-CUDA_CLASSIFY_TRAINING_V1_001..002.
//
// Contract: `contracts/cuda-classify-training-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Two SSC classifier-training kernel parity gates:
//
// - CCT-001 (forward parity): CUDA-computed logits/probabilities match
//   CPU reference within ε.
// - CCT-002 (backward parity): CUDA-computed gradients match CPU
//   reference within ε.
//
// In-module reference: `linear_softmax_forward` (linear projection
// then softmax) and `cross_entropy_backward` (gradient w.r.t. logits
// for one-hot label).

/// Forward-parity tolerance.
pub const AC_CCT_001_FORWARD_EPS: f32 = 1e-5;

/// Backward-parity tolerance.
pub const AC_CCT_002_BACKWARD_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CctVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference forward and backward.
// -----------------------------------------------------------------------------

/// Numerically stable softmax over a row.
#[must_use]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for v in &mut exps {
            *v /= sum;
        }
    }
    exps
}

/// Reference forward: y = softmax(W·x + b) for one sample.
///
/// `weights` is row-major `[n_classes, n_features]`; `bias` is
/// `[n_classes]`; `x` is `[n_features]`.
#[must_use]
pub fn linear_softmax_forward(
    weights: &[f32],
    bias: &[f32],
    x: &[f32],
    n_classes: usize,
    n_features: usize,
) -> Option<Vec<f32>> {
    if weights.len() != n_classes * n_features
        || bias.len() != n_classes
        || x.len() != n_features
    {
        return None;
    }
    let mut logits = bias.to_vec();
    for c in 0..n_classes {
        for j in 0..n_features {
            logits[c] += weights[c * n_features + j] * x[j];
        }
    }
    Some(softmax(&logits))
}

/// Reference backward: ∂CE/∂logits = softmax(logits) - one_hot(label).
///
/// Returns `None` if `label >= n_classes`.
#[must_use]
pub fn cross_entropy_backward(probs: &[f32], label: usize) -> Option<Vec<f32>> {
    if label >= probs.len() {
        return None;
    }
    let mut grad = probs.to_vec();
    grad[label] -= 1.0;
    Some(grad)
}

/// Maximum elementwise absolute difference.
#[must_use]
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut max = 0.0_f32;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if !ai.is_finite() || !bi.is_finite() {
            return None;
        }
        let d = (ai - bi).abs();
        if d > max {
            max = d;
        }
    }
    Some(max)
}

// -----------------------------------------------------------------------------
// Verdict 1: CCT-001 — CUDA forward matches CPU.
// -----------------------------------------------------------------------------

/// Pass iff `|cuda_probs - cpu_probs|` is < `AC_CCT_001_FORWARD_EPS`
/// elementwise.
#[must_use]
pub fn verdict_from_forward_parity(cuda_probs: &[f32], cpu_probs: &[f32]) -> CctVerdict {
    match max_abs_diff(cuda_probs, cpu_probs) {
        Some(d) if d < AC_CCT_001_FORWARD_EPS => CctVerdict::Pass,
        _ => CctVerdict::Fail,
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: CCT-002 — CUDA gradients match CPU.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_backward_parity(cuda_grad: &[f32], cpu_grad: &[f32]) -> CctVerdict {
    match max_abs_diff(cuda_grad, cpu_grad) {
        Some(d) if d < AC_CCT_002_BACKWARD_EPS => CctVerdict::Pass,
        _ => CctVerdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_forward_eps_1e_5() {
        assert_eq!(AC_CCT_001_FORWARD_EPS, 1e-5);
    }

    #[test]
    fn provenance_backward_eps_1e_5() {
        assert_eq!(AC_CCT_002_BACKWARD_EPS, 1e-5);
    }

    // -------------------------------------------------------------------------
    // Section 2: Domain — reference forward.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_forward_uniform_logits_uniform_probs() {
        // weights=0, bias=0 ⇒ logits=0 ⇒ softmax = uniform = [1/4]*4.
        let weights = vec![0.0_f32; 4 * 2];
        let bias = vec![0.0_f32; 4];
        let x = vec![1.0_f32, 2.0];
        let probs = linear_softmax_forward(&weights, &bias, &x, 4, 2).unwrap();
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn domain_forward_known_logits() {
        // 2 classes, 1 feature: w=[1, -1], b=[0, 0], x=[1] ⇒ logits=[1, -1]
        // softmax(1, -1) = [exp(1)/(exp(1)+exp(-1)), exp(-1)/(exp(1)+exp(-1))]
        //                ≈ [0.8808, 0.1192]
        let weights = vec![1.0_f32, -1.0];
        let bias = vec![0.0_f32, 0.0];
        let x = vec![1.0_f32];
        let probs = linear_softmax_forward(&weights, &bias, &x, 2, 1).unwrap();
        assert!((probs[0] - 0.8808).abs() < 0.001, "p[0]={}", probs[0]);
        assert!((probs[1] - 0.1192).abs() < 0.001, "p[1]={}", probs[1]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn domain_forward_invalid_shapes() {
        // Wrong weight buffer size.
        let weights = vec![1.0_f32, 2.0]; // expected 4*2
        let bias = vec![0.0_f32; 4];
        let x = vec![1.0_f32, 2.0];
        assert!(linear_softmax_forward(&weights, &bias, &x, 4, 2).is_none());
    }

    // -------------------------------------------------------------------------
    // Section 3: Domain — reference backward.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_backward_subtract_one_at_label() {
        // softmax = [0.7, 0.2, 0.1]; label = 0 ⇒ grad = [-0.3, 0.2, 0.1].
        let probs = vec![0.7_f32, 0.2, 0.1];
        let grad = cross_entropy_backward(&probs, 0).unwrap();
        assert!((grad[0] - (-0.3)).abs() < 1e-6);
        assert!((grad[1] - 0.2).abs() < 1e-6);
        assert!((grad[2] - 0.1).abs() < 1e-6);
        // Sum should be ≈ 0 (probabilities sum to 1 so subtracting 1 zeroes total).
        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-6);
    }

    #[test]
    fn domain_backward_at_uniform_label_2() {
        let probs = vec![0.25_f32, 0.25, 0.25, 0.25];
        let grad = cross_entropy_backward(&probs, 2).unwrap();
        assert!((grad[2] - (-0.75)).abs() < 1e-6);
        for (i, g) in grad.iter().enumerate() {
            if i != 2 {
                assert!((g - 0.25).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn domain_backward_label_out_of_range() {
        let probs = vec![0.5_f32, 0.5];
        assert!(cross_entropy_backward(&probs, 5).is_none());
    }

    // -------------------------------------------------------------------------
    // Section 4: CCT-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn cct001_pass_identical_probs() {
        let cuda = vec![0.7_f32, 0.2, 0.1];
        let cpu = vec![0.7_f32, 0.2, 0.1];
        assert_eq!(verdict_from_forward_parity(&cuda, &cpu), CctVerdict::Pass);
    }

    #[test]
    fn cct001_pass_within_tolerance() {
        let cuda = vec![0.700001_f32, 0.199998, 0.100001];
        let cpu = vec![0.7_f32, 0.2, 0.1];
        assert_eq!(verdict_from_forward_parity(&cuda, &cpu), CctVerdict::Pass);
    }

    #[test]
    fn cct001_pass_realistic_uniform() {
        let cuda = vec![0.250003_f32, 0.249998, 0.250001, 0.249998];
        let cpu = vec![0.25_f32, 0.25, 0.25, 0.25];
        assert_eq!(verdict_from_forward_parity(&cuda, &cpu), CctVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 5: CCT-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn cct001_fail_above_tolerance() {
        let cuda = vec![0.8_f32, 0.15, 0.05];
        let cpu = vec![0.7_f32, 0.2, 0.1];
        assert_eq!(verdict_from_forward_parity(&cuda, &cpu), CctVerdict::Fail);
    }

    #[test]
    fn cct001_fail_length_mismatch() {
        let cuda = vec![0.5_f32, 0.5];
        let cpu = vec![0.5_f32, 0.5, 0.0];
        assert_eq!(verdict_from_forward_parity(&cuda, &cpu), CctVerdict::Fail);
    }

    #[test]
    fn cct001_fail_nan() {
        let cuda = vec![f32::NAN];
        let cpu = vec![0.5_f32];
        assert_eq!(verdict_from_forward_parity(&cuda, &cpu), CctVerdict::Fail);
    }

    #[test]
    fn cct001_fail_inf() {
        let cuda = vec![f32::INFINITY];
        let cpu = vec![1.0_f32];
        assert_eq!(verdict_from_forward_parity(&cuda, &cpu), CctVerdict::Fail);
    }

    #[test]
    fn cct001_fail_empty_both() {
        let v: Vec<f32> = vec![];
        assert_eq!(verdict_from_forward_parity(&v, &v), CctVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: CCT-002 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn cct002_pass_identical_grads() {
        let cuda = vec![-0.3_f32, 0.2, 0.1];
        let cpu = vec![-0.3_f32, 0.2, 0.1];
        assert_eq!(verdict_from_backward_parity(&cuda, &cpu), CctVerdict::Pass);
    }

    #[test]
    fn cct002_pass_within_tolerance() {
        let cuda = vec![-0.299998_f32, 0.200001, 0.099998];
        let cpu = vec![-0.3_f32, 0.2, 0.1];
        assert_eq!(verdict_from_backward_parity(&cuda, &cpu), CctVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 7: CCT-002 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn cct002_fail_above_tolerance() {
        let cuda = vec![-0.5_f32, 0.3, 0.2];
        let cpu = vec![-0.3_f32, 0.2, 0.1];
        assert_eq!(verdict_from_backward_parity(&cuda, &cpu), CctVerdict::Fail);
    }

    #[test]
    fn cct002_fail_sign_flip() {
        // Bug: gradient sign inverted.
        let cuda = vec![0.3_f32, -0.2, -0.1]; // wrong sign
        let cpu = vec![-0.3_f32, 0.2, 0.1];
        assert_eq!(verdict_from_backward_parity(&cuda, &cpu), CctVerdict::Fail);
    }

    #[test]
    fn cct002_fail_length_mismatch() {
        let cuda = vec![-0.3_f32, 0.2];
        let cpu = vec![-0.3_f32, 0.2, 0.1];
        assert_eq!(verdict_from_backward_parity(&cuda, &cpu), CctVerdict::Fail);
    }

    #[test]
    fn cct002_fail_nan() {
        let cuda = vec![f32::NAN, 0.2];
        let cpu = vec![-0.3_f32, 0.2];
        assert_eq!(verdict_from_backward_parity(&cuda, &cpu), CctVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: Sweep — forward parity at various drifts.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_forward_parity_drift_band() {
        // Use exact f32 boundaries to avoid quirks.
        let cpu = vec![0.5_f32];
        let test_cases: Vec<(f32, CctVerdict)> = vec![
            (0.5, CctVerdict::Pass),
            (0.5 + 5e-6, CctVerdict::Pass),  // drift 5e-6 < 1e-5
            (0.5 + 5e-5, CctVerdict::Fail),  // drift 5e-5 >= 1e-5
            (0.5 + 1e-3, CctVerdict::Fail),
            (0.5 + 1e-2, CctVerdict::Fail),
        ];
        for (cuda_val, expected) in test_cases {
            let cuda = vec![cuda_val];
            let v = verdict_from_forward_parity(&cuda, &cpu);
            assert_eq!(v, expected, "cuda={cuda_val}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 9: Realistic — full forward+backward pipeline.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_full_classifier_step_passes_both_gates() {
        // Synthetic 4-class, 3-feature classifier.
        let weights = vec![
            1.0_f32, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            -1.0, -1.0, -1.0,
        ];
        let bias = vec![0.0_f32; 4];
        let x = vec![2.0_f32, 1.0, 0.5];

        // CPU forward.
        let cpu_probs = linear_softmax_forward(&weights, &bias, &x, 4, 3).unwrap();

        // "CUDA" forward — same impl with synthesized 1e-7 drift.
        let cuda_probs: Vec<f32> = cpu_probs
            .iter()
            .enumerate()
            .map(|(i, p)| if i == 0 { p + 1e-7 } else { *p })
            .collect();

        assert_eq!(
            verdict_from_forward_parity(&cuda_probs, &cpu_probs),
            CctVerdict::Pass
        );

        // CPU backward at label = 0.
        let cpu_grad = cross_entropy_backward(&cpu_probs, 0).unwrap();
        let cuda_grad: Vec<f32> = cpu_grad.iter().map(|g| g + 1e-7).collect();
        assert_eq!(
            verdict_from_backward_parity(&cuda_grad, &cpu_grad),
            CctVerdict::Pass
        );
    }

    #[test]
    fn realistic_kernel_arithmetic_drift_caught() {
        // CCT-001 if_fails: CUDA kernel uses different reduction order
        // ⇒ accumulates to wrong probabilities.
        let cuda = vec![0.71_f32, 0.19, 0.10];
        let cpu = vec![0.7_f32, 0.2, 0.1];
        assert_eq!(verdict_from_forward_parity(&cuda, &cpu), CctVerdict::Fail);
    }

    #[test]
    fn realistic_grad_sign_bug_caught() {
        // CCT-002 if_fails: backward kernel computes p - one_hot
        // instead of one_hot - p (sign flip).
        let cuda = vec![0.3_f32, -0.2, -0.1];
        let cpu = vec![-0.3_f32, 0.2, 0.1];
        assert_eq!(verdict_from_backward_parity(&cuda, &cpu), CctVerdict::Fail);
    }
}
