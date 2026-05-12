// `batch-training-v1` algorithm-level PARTIAL discharge for the 6
// mini-batch + gradient accumulation falsifiers.
//
// Contract: `contracts/batch-training-v1.yaml`.
// Refs: Goyal et al. (2017) "Accurate, Large Minibatch SGD"
// (arXiv:1706.02677).

/// Tolerance for "single-batch vs accumulated micro-batch" gradient
/// equivalence (1e-5 per contract F-BATCH-001).
pub const AC_BTRAIN_GRAD_TOLERANCE: f32 = 1.0e-5;

/// Tolerance for clipped-norm bound check (1e-6 per FALSIFY-BATCH-003).
pub const AC_BTRAIN_CLIP_TOLERANCE: f32 = 1.0e-6;

/// Tolerance for "params changed" check (1e-10 — strict per contract).
pub const AC_BTRAIN_PARAM_CHANGE_TOLERANCE: f32 = 1.0e-10;

// =============================================================================
// FALSIFY-BATCH-001 — gradient equivalence (single batch == accumulated)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtrainGradEquivalenceVerdict {
    /// |single_batch_grad[i] - accumulated_grad[i]| < 1e-5 ∀ i.
    Pass,
    /// Some gradient component diverges — normalization error.
    Fail,
}

#[must_use]
pub fn verdict_from_btrain_grad_equivalence(
    single_batch: &[f32],
    accumulated: &[f32],
) -> BtrainGradEquivalenceVerdict {
    if single_batch.len() != accumulated.len() {
        return BtrainGradEquivalenceVerdict::Fail;
    }
    if single_batch.is_empty() {
        return BtrainGradEquivalenceVerdict::Fail;
    }
    for (a, b) in single_batch.iter().zip(accumulated.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return BtrainGradEquivalenceVerdict::Fail;
        }
        if (a - b).abs() >= AC_BTRAIN_GRAD_TOLERANCE {
            return BtrainGradEquivalenceVerdict::Fail;
        }
    }
    BtrainGradEquivalenceVerdict::Pass
}

// =============================================================================
// FALSIFY-BATCH-002 — loss is finite AND non-negative
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtrainLossFiniteVerdict {
    /// Batch loss is finite (no NaN, no Inf) AND >= 0.
    Pass,
    /// NaN/Inf or negative loss.
    Fail,
}

#[must_use]
pub fn verdict_from_btrain_loss_finite(loss: f32) -> BtrainLossFiniteVerdict {
    if !loss.is_finite() {
        return BtrainLossFiniteVerdict::Fail;
    }
    if loss < 0.0 {
        return BtrainLossFiniteVerdict::Fail;
    }
    BtrainLossFiniteVerdict::Pass
}

// =============================================================================
// FALSIFY-BATCH-003 — gradient norm bounded by clip_norm
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtrainGradClipVerdict {
    /// post_clip_grad_norm <= clip_norm + 1e-6 (with rounding tolerance).
    Pass,
    /// Norm exceeds clip threshold — clipping not applied.
    Fail,
}

#[must_use]
pub fn verdict_from_btrain_grad_clip(post_clip_norm: f32, clip_norm: f32) -> BtrainGradClipVerdict {
    if !post_clip_norm.is_finite() || !clip_norm.is_finite() {
        return BtrainGradClipVerdict::Fail;
    }
    if clip_norm <= 0.0 {
        return BtrainGradClipVerdict::Fail;
    }
    if post_clip_norm <= clip_norm + AC_BTRAIN_CLIP_TOLERANCE {
        BtrainGradClipVerdict::Pass
    } else {
        BtrainGradClipVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-BATCH-004 — single optimizer step changed parameters
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtrainOptimizerStepVerdict {
    /// At least one parameter changed AFTER train_batch (optimizer fired).
    Pass,
    /// Parameters identical — optimizer not called.
    Fail,
}

#[must_use]
pub fn verdict_from_btrain_optimizer_step(
    params_before: &[f32],
    params_after: &[f32],
) -> BtrainOptimizerStepVerdict {
    if params_before.len() != params_after.len() {
        return BtrainOptimizerStepVerdict::Fail;
    }
    if params_before.is_empty() {
        return BtrainOptimizerStepVerdict::Fail;
    }
    for (a, b) in params_before.iter().zip(params_after.iter()) {
        if (a - b).abs() > AC_BTRAIN_PARAM_CHANGE_TOLERANCE {
            return BtrainOptimizerStepVerdict::Pass;
        }
    }
    BtrainOptimizerStepVerdict::Fail
}

// =============================================================================
// FALSIFY-BATCH-005 — no stale gradients between batches
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtrainNoStaleGradVerdict {
    /// Gradients differ between back-to-back batches (zeroing happened).
    Pass,
    /// Gradients identical — stale from previous batch.
    Fail,
}

#[must_use]
pub fn verdict_from_btrain_no_stale_grad(
    grads_after_a: &[f32],
    grads_after_b: &[f32],
) -> BtrainNoStaleGradVerdict {
    if grads_after_a.len() != grads_after_b.len() {
        return BtrainNoStaleGradVerdict::Fail;
    }
    if grads_after_a.is_empty() {
        return BtrainNoStaleGradVerdict::Fail;
    }
    for (a, b) in grads_after_a.iter().zip(grads_after_b.iter()) {
        if (a - b).abs() > AC_BTRAIN_PARAM_CHANGE_TOLERANCE {
            return BtrainNoStaleGradVerdict::Pass;
        }
    }
    BtrainNoStaleGradVerdict::Fail
}

// =============================================================================
// FALSIFY-BATCH-006 — gradient scaling (all post-scaling values finite)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtrainGradScalingVerdict {
    /// All accumulated gradients are finite after divide-by-K.
    Pass,
    /// Any non-finite element.
    Fail,
}

#[must_use]
pub fn verdict_from_btrain_grad_scaling(
    accumulated_grads: &[f32],
    accumulation_steps: u32,
) -> BtrainGradScalingVerdict {
    if accumulation_steps == 0 {
        return BtrainGradScalingVerdict::Fail;
    }
    if accumulated_grads.is_empty() {
        return BtrainGradScalingVerdict::Fail;
    }
    for &g in accumulated_grads {
        if !g.is_finite() {
            return BtrainGradScalingVerdict::Fail;
        }
    }
    BtrainGradScalingVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_grad_tolerance_1e_neg5() {
        assert!((AC_BTRAIN_GRAD_TOLERANCE - 1.0e-5).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_clip_tolerance_1e_neg6() {
        assert!((AC_BTRAIN_CLIP_TOLERANCE - 1.0e-6).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_param_tolerance_1e_neg10() {
        assert!((AC_BTRAIN_PARAM_CHANGE_TOLERANCE - 1.0e-10).abs() < f32::EPSILON);
    }

    // -------------------------------------------------------------------------
    // Section 2: BATCH-001 gradient equivalence.
    // -------------------------------------------------------------------------
    #[test]
    fn fbt001_pass_exact_match() {
        let g = vec![0.5_f32, -0.3, 0.1];
        assert_eq!(
            verdict_from_btrain_grad_equivalence(&g, &g),
            BtrainGradEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fbt001_pass_within_tolerance() {
        let single = vec![0.5_f32];
        let acc = vec![0.5_f32 + 5e-6];
        assert_eq!(
            verdict_from_btrain_grad_equivalence(&single, &acc),
            BtrainGradEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fbt001_fail_outside_tolerance() {
        assert_eq!(
            verdict_from_btrain_grad_equivalence(&[0.5], &[0.6]),
            BtrainGradEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fbt001_fail_length_mismatch() {
        assert_eq!(
            verdict_from_btrain_grad_equivalence(&[0.1], &[0.1, 0.2]),
            BtrainGradEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fbt001_fail_nan() {
        assert_eq!(
            verdict_from_btrain_grad_equivalence(&[f32::NAN], &[0.0]),
            BtrainGradEquivalenceVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: BATCH-002 loss finite.
    // -------------------------------------------------------------------------
    #[test]
    fn fbt002_pass_positive_loss() {
        assert_eq!(verdict_from_btrain_loss_finite(0.5), BtrainLossFiniteVerdict::Pass);
    }

    #[test]
    fn fbt002_pass_zero_loss() {
        assert_eq!(verdict_from_btrain_loss_finite(0.0), BtrainLossFiniteVerdict::Pass);
    }

    #[test]
    fn fbt002_fail_nan() {
        assert_eq!(verdict_from_btrain_loss_finite(f32::NAN), BtrainLossFiniteVerdict::Fail);
    }

    #[test]
    fn fbt002_fail_inf() {
        assert_eq!(verdict_from_btrain_loss_finite(f32::INFINITY), BtrainLossFiniteVerdict::Fail);
    }

    #[test]
    fn fbt002_fail_negative() {
        assert_eq!(verdict_from_btrain_loss_finite(-0.001), BtrainLossFiniteVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: BATCH-003 gradient clipping.
    // -------------------------------------------------------------------------
    #[test]
    fn fbt003_pass_under_clip() {
        assert_eq!(verdict_from_btrain_grad_clip(0.8, 1.0), BtrainGradClipVerdict::Pass);
    }

    #[test]
    fn fbt003_pass_at_clip() {
        assert_eq!(verdict_from_btrain_grad_clip(1.0, 1.0), BtrainGradClipVerdict::Pass);
    }

    #[test]
    fn fbt003_fail_over_clip() {
        assert_eq!(verdict_from_btrain_grad_clip(2.0, 1.0), BtrainGradClipVerdict::Fail);
    }

    #[test]
    fn fbt003_fail_clip_zero() {
        assert_eq!(verdict_from_btrain_grad_clip(0.5, 0.0), BtrainGradClipVerdict::Fail);
    }

    #[test]
    fn fbt003_fail_nan_norm() {
        assert_eq!(
            verdict_from_btrain_grad_clip(f32::NAN, 1.0),
            BtrainGradClipVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: BATCH-004 single optimizer step.
    // -------------------------------------------------------------------------
    #[test]
    fn fbt004_pass_params_changed() {
        let before = vec![1.0_f32, 2.0];
        let after = vec![1.001_f32, 2.001];
        assert_eq!(
            verdict_from_btrain_optimizer_step(&before, &after),
            BtrainOptimizerStepVerdict::Pass
        );
    }

    #[test]
    fn fbt004_fail_params_unchanged() {
        let v = vec![1.0_f32, 2.0];
        assert_eq!(
            verdict_from_btrain_optimizer_step(&v, &v),
            BtrainOptimizerStepVerdict::Fail
        );
    }

    #[test]
    fn fbt004_fail_length_mismatch() {
        assert_eq!(
            verdict_from_btrain_optimizer_step(&[1.0], &[1.0, 2.0]),
            BtrainOptimizerStepVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: BATCH-005 no stale gradients.
    // -------------------------------------------------------------------------
    #[test]
    fn fbt005_pass_grads_differ() {
        let a = vec![0.5_f32, -0.3];
        let b = vec![0.4_f32, 0.1];
        assert_eq!(
            verdict_from_btrain_no_stale_grad(&a, &b),
            BtrainNoStaleGradVerdict::Pass
        );
    }

    #[test]
    fn fbt005_fail_grads_identical() {
        let g = vec![0.5_f32, -0.3];
        assert_eq!(
            verdict_from_btrain_no_stale_grad(&g, &g),
            BtrainNoStaleGradVerdict::Fail
        );
    }

    #[test]
    fn fbt005_fail_empty() {
        assert_eq!(
            verdict_from_btrain_no_stale_grad(&[], &[]),
            BtrainNoStaleGradVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: BATCH-006 gradient scaling.
    // -------------------------------------------------------------------------
    #[test]
    fn fbt006_pass_finite_scaled_grads() {
        let g = vec![0.125_f32, -0.0625, 0.03125];
        assert_eq!(
            verdict_from_btrain_grad_scaling(&g, 4),
            BtrainGradScalingVerdict::Pass
        );
    }

    #[test]
    fn fbt006_fail_nan_after_scaling() {
        let g = vec![f32::NAN];
        assert_eq!(
            verdict_from_btrain_grad_scaling(&g, 4),
            BtrainGradScalingVerdict::Fail
        );
    }

    #[test]
    fn fbt006_fail_zero_steps() {
        assert_eq!(
            verdict_from_btrain_grad_scaling(&[0.5], 0),
            BtrainGradScalingVerdict::Fail
        );
    }

    #[test]
    fn fbt006_fail_empty() {
        assert_eq!(
            verdict_from_btrain_grad_scaling(&[], 4),
            BtrainGradScalingVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: Realistic — full healthy batch training.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_batch_training_passes_all_6() {
        // 001: gradient equivalence within tolerance.
        let single = vec![0.1_f32, -0.2, 0.3];
        let acc = vec![0.1_f32 + 1e-7, -0.2 + 1e-7, 0.3 - 1e-7];
        assert_eq!(
            verdict_from_btrain_grad_equivalence(&single, &acc),
            BtrainGradEquivalenceVerdict::Pass
        );
        // 002: positive finite loss.
        assert_eq!(
            verdict_from_btrain_loss_finite(2.5),
            BtrainLossFiniteVerdict::Pass
        );
        // 003: clipped norm under threshold.
        assert_eq!(
            verdict_from_btrain_grad_clip(0.95, 1.0),
            BtrainGradClipVerdict::Pass
        );
        // 004: params changed.
        assert_eq!(
            verdict_from_btrain_optimizer_step(&[1.0_f32], &[1.001_f32]),
            BtrainOptimizerStepVerdict::Pass
        );
        // 005: grads differ between batches.
        assert_eq!(
            verdict_from_btrain_no_stale_grad(&[0.5_f32], &[0.4_f32]),
            BtrainNoStaleGradVerdict::Pass
        );
        // 006: scaled grads finite.
        assert_eq!(
            verdict_from_btrain_grad_scaling(&[0.125_f32, -0.0625], 8),
            BtrainGradScalingVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        // 001: forgot to divide by K.
        assert_eq!(
            verdict_from_btrain_grad_equivalence(&[0.1], &[0.4]),
            BtrainGradEquivalenceVerdict::Fail
        );
        // 002: loss is NaN.
        assert_eq!(
            verdict_from_btrain_loss_finite(f32::NAN),
            BtrainLossFiniteVerdict::Fail
        );
        // 003: clipping skipped, norm = 100.
        assert_eq!(
            verdict_from_btrain_grad_clip(100.0, 1.0),
            BtrainGradClipVerdict::Fail
        );
        // 004: optimizer not wired.
        let v = vec![1.0_f32, 2.0];
        assert_eq!(
            verdict_from_btrain_optimizer_step(&v, &v),
            BtrainOptimizerStepVerdict::Fail
        );
        // 005: stale gradients.
        let g = vec![0.5_f32];
        assert_eq!(
            verdict_from_btrain_no_stale_grad(&g, &g),
            BtrainNoStaleGradVerdict::Fail
        );
        // 006: scaled grads have NaN.
        assert_eq!(
            verdict_from_btrain_grad_scaling(&[f32::NAN], 4),
            BtrainGradScalingVerdict::Fail
        );
    }
}
