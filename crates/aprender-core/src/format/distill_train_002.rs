// SHIP-TWO-001 §35 — `apr-cli-distill-train-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-DISTILL-TRAIN-002.
//
// Contract: `contracts/apr-cli-distill-train-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §35.
//
// ## What FALSIFY-APR-DISTILL-TRAIN-002 says
//
//   rule: KL loss decreases over epochs
//   prediction: Per-epoch metadata.json shows
//               kl_loss[epoch=N+1] < kl_loss[epoch=N] (with
//               batch-noise tolerance ≤ 5%).
//
// Today this gate cannot run because the §35 stub never produces
// per-epoch kl_loss values. Once §35.3 implementation lands, the
// live test passes.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// The decision rule — "for every consecutive pair `(losses[i],
// losses[i+1])`, the next-epoch loss is at most `1.05 ×`
// previous-epoch loss" — is pinned. Future implementations cannot
// silently weaken the tolerance (e.g., to 50%) or relax the
// monotonicity check entirely. Sibling to `distill_train_001`
// (#1139, stub-detection) — both falsifiers must Pass for the
// distill loop to be considered "real training".

/// Maximum tolerated batch-noise factor on the per-epoch KL-loss curve.
///
/// Per the contract `falsification_test`:
///
/// ```text
/// assert losses[1] < losses[0] * 1.05  // allow 5% batch noise
/// assert losses[2] < losses[1] * 1.05
/// ```
///
/// Bound here as `0.05` (additive over `1.0`). A future implementation
/// regressing to a looser tolerance (e.g. `0.50`, allowing 50% loss
/// regrowth between epochs) would silently pass a buggy training loop;
/// pinning the constant catches that drift.
pub const AC_DISTILL_TRAIN_002_BATCH_NOISE_TOLERANCE: f32 = 0.05;

/// Multiplier applied to `losses[i]` when checking `losses[i+1]`.
///
/// Equals `1.0 + AC_DISTILL_TRAIN_002_BATCH_NOISE_TOLERANCE = 1.05`.
/// Bound here so the relationship between the additive tolerance and
/// the multiplicative threshold is provable from the test surface.
pub const AC_DISTILL_TRAIN_002_NEXT_EPOCH_MAX_MULTIPLIER: f32 =
    1.0 + AC_DISTILL_TRAIN_002_BATCH_NOISE_TOLERANCE;

/// Binary verdict for `FALSIFY-APR-DISTILL-TRAIN-002`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillTrain002Verdict {
    /// Per-epoch KL-loss curve is monotonically decreasing modulo the
    /// 5% batch-noise tolerance. Real gradient-based training is
    /// converging.
    Pass,
    /// One or more of:
    /// - Slice has fewer than 2 epochs (no comparison possible).
    /// - At least one loss is non-finite (NaN, ±∞).
    /// - At least one loss is negative (KL divergence is non-negative
    ///   by definition of probability ratios; negative implies a
    ///   buggy loss reduction or unit confusion — conservative `Fail`).
    /// - For some `i`, `losses[i+1] > losses[i] * 1.05` — the loss
    ///   regrew by more than the tolerated batch noise.
    Fail,
}

/// Pure verdict function for FALSIFY-APR-DISTILL-TRAIN-002.
///
/// Input: per-epoch KL-loss values from
/// `<output_dir>/epoch-*.metadata.json` files, in chronological order.
///
/// # Examples
///
/// Clean monotonic decrease — `Pass`:
/// ```
/// use aprender::format::distill_train_002::{
///     verdict_from_per_epoch_kl_losses, DistillTrain002Verdict,
/// };
/// let losses = vec![10.0_f32, 8.0, 6.5, 5.0, 4.2];
/// assert_eq!(
///     verdict_from_per_epoch_kl_losses(&losses),
///     DistillTrain002Verdict::Pass,
/// );
/// ```
///
/// Loss regrowth beyond 5% — `Fail`:
/// ```
/// use aprender::format::distill_train_002::{
///     verdict_from_per_epoch_kl_losses, DistillTrain002Verdict,
/// };
/// let losses = vec![1.0_f32, 1.5, 1.2]; // 1.0 → 1.5 = +50% regrowth
/// assert_eq!(
///     verdict_from_per_epoch_kl_losses(&losses),
///     DistillTrain002Verdict::Fail,
/// );
/// ```
#[must_use]
pub fn verdict_from_per_epoch_kl_losses(losses: &[f32]) -> DistillTrain002Verdict {
    if losses.len() < 2 {
        return DistillTrain002Verdict::Fail;
    }
    for &v in losses {
        if !v.is_finite() || v < 0.0 {
            return DistillTrain002Verdict::Fail;
        }
    }
    for i in 0..losses.len() - 1 {
        let prev = losses[i];
        let next = losses[i + 1];
        let ceiling = prev * AC_DISTILL_TRAIN_002_NEXT_EPOCH_MAX_MULTIPLIER;
        if next > ceiling {
            return DistillTrain002Verdict::Fail;
        }
    }
    DistillTrain002Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_up_f32(x: f32) -> f32 {
        f32::from_bits(x.to_bits() + 1)
    }

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — tolerance + multiplier match contract.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_batch_noise_tolerance_is_five_percent() {
        assert_eq!(AC_DISTILL_TRAIN_002_BATCH_NOISE_TOLERANCE, 0.05);
    }

    #[test]
    fn provenance_next_epoch_max_multiplier_is_one_oh_five() {
        assert_eq!(AC_DISTILL_TRAIN_002_NEXT_EPOCH_MAX_MULTIPLIER, 1.05);
    }

    #[test]
    fn provenance_multiplier_equals_one_plus_tolerance() {
        // Catches a silent drift where one constant changes but not the other.
        assert_eq!(
            AC_DISTILL_TRAIN_002_NEXT_EPOCH_MAX_MULTIPLIER,
            1.0 + AC_DISTILL_TRAIN_002_BATCH_NOISE_TOLERANCE
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — clean / noisy-but-tolerant decreases.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_clean_monotonic_decrease() {
        let losses = vec![10.0_f32, 8.0, 6.5, 5.0, 4.2, 3.5];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Pass
        );
    }

    #[test]
    fn pass_noisy_but_within_tolerance() {
        // 1.0 → 1.04 is +4% (≤ 5%) → Pass; 1.04 → 1.0 is decrease → Pass.
        let losses = vec![1.0_f32, 1.04, 1.0, 0.95, 0.92];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Pass
        );
    }

    #[test]
    fn pass_strict_two_epochs() {
        let losses = vec![5.0_f32, 3.0];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Pass
        );
    }

    #[test]
    fn pass_two_epochs_within_tolerance() {
        let losses = vec![1.0_f32, 1.04];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Pass
        );
    }

    #[test]
    fn pass_zero_loss_then_zero_loss() {
        // Degenerate but mathematically sound: zero loss carries through.
        // 0.0 * 1.05 = 0.0; next 0.0 ≤ 0.0 ✓.
        let losses = vec![0.0_f32, 0.0, 0.0];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — clear loss regrowth beyond tolerance.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_loss_doubled_between_epochs() {
        // 1.0 → 2.0 is +100% (way above 5% tolerance).
        let losses = vec![1.0_f32, 2.0, 0.5];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Fail
        );
    }

    #[test]
    fn fail_loss_regrowth_at_any_position() {
        // Regrowth must trip the gate regardless of position in the curve.
        for bad_i in [0_usize, 2, 4] {
            let mut losses = vec![1.0_f32, 0.9, 0.8, 0.7, 0.6, 0.5];
            losses[bad_i + 1] = losses[bad_i] * 2.0;
            assert_eq!(
                verdict_from_per_epoch_kl_losses(&losses),
                DistillTrain002Verdict::Fail,
                "regrowth between index {bad_i} and {} must Fail",
                bad_i + 1
            );
        }
    }

    #[test]
    fn fail_just_above_tolerance() {
        // 1.0 → 1.05 + 1 ULP must Fail (strict `>` on the ceiling).
        let just_above = next_up_f32(AC_DISTILL_TRAIN_002_NEXT_EPOCH_MAX_MULTIPLIER);
        assert!(just_above > AC_DISTILL_TRAIN_002_NEXT_EPOCH_MAX_MULTIPLIER);
        let losses = vec![1.0_f32, just_above];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Fail
        );
    }

    #[test]
    fn pass_exactly_at_tolerance() {
        // 1.0 → 1.05 exactly is `next == prev * 1.05` (NOT `>`) → Pass.
        // Pinned because the contract uses strict `<` in the assertion
        // (`losses[1] < losses[0] * 1.05`); our `>` mirror is the
        // negation — passes when not strictly above.
        let losses = vec![1.0_f32, AC_DISTILL_TRAIN_002_NEXT_EPOCH_MAX_MULTIPLIER];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Insufficient input — fewer than 2 epochs → Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_input() {
        let losses: Vec<f32> = vec![];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Fail
        );
    }

    #[test]
    fn fail_single_epoch_no_comparison_possible() {
        let losses = vec![3.0_f32];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Fail,
            "single-epoch input cannot prove decrease; conservative Fail"
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Domain violation — non-finite or negative losses Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_nan_in_any_position() {
        for pos in [0_usize, 1, 2, 4] {
            let mut losses = vec![1.0_f32, 0.9, 0.8, 0.7, 0.6];
            losses[pos] = f32::NAN;
            assert_eq!(
                verdict_from_per_epoch_kl_losses(&losses),
                DistillTrain002Verdict::Fail,
                "NaN at position {pos} must Fail"
            );
        }
    }

    #[test]
    fn fail_positive_infinity() {
        let losses = vec![f32::INFINITY, 1.0, 0.5];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Fail
        );
    }

    #[test]
    fn fail_negative_infinity() {
        let losses = vec![1.0_f32, f32::NEG_INFINITY];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Fail
        );
    }

    #[test]
    fn fail_negative_kl_loss_is_domain_violation() {
        // KL(p || q) ≥ 0 always (Gibbs' inequality). Negative implies bug.
        let losses = vec![1.0_f32, -0.5];
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Three-epoch contract assertion shape (matches contract test).
    // -------------------------------------------------------------------------
    #[test]
    fn three_epoch_contract_pattern_pass() {
        // From contract: `assert losses[1] < losses[0] * 1.05`
        //                `assert losses[2] < losses[1] * 1.05`
        let losses = vec![10.0_f32, 9.5, 8.0]; // strict decrease
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Pass
        );
    }

    #[test]
    fn three_epoch_first_pair_fails() {
        let losses = vec![10.0_f32, 11.0, 9.0]; // 10 → 11 = +10% > 5%
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Fail
        );
    }

    #[test]
    fn three_epoch_second_pair_fails() {
        let losses = vec![10.0_f32, 9.0, 10.0]; // 9 → 10 = +11% > 5%
        assert_eq!(
            verdict_from_per_epoch_kl_losses(&losses),
            DistillTrain002Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Boundary sweep at the 5% threshold.
    // -------------------------------------------------------------------------
    #[test]
    fn boundary_sweep_two_epochs() {
        let probes: Vec<(f32, DistillTrain002Verdict)> = vec![
            (0.5, DistillTrain002Verdict::Pass),
            (0.99, DistillTrain002Verdict::Pass),
            (1.0, DistillTrain002Verdict::Pass),
            (1.04, DistillTrain002Verdict::Pass),
            (
                AC_DISTILL_TRAIN_002_NEXT_EPOCH_MAX_MULTIPLIER,
                DistillTrain002Verdict::Pass,
            ),
            (
                next_up_f32(AC_DISTILL_TRAIN_002_NEXT_EPOCH_MAX_MULTIPLIER),
                DistillTrain002Verdict::Fail,
            ),
            (1.06, DistillTrain002Verdict::Fail),
            (1.10, DistillTrain002Verdict::Fail),
            (2.0, DistillTrain002Verdict::Fail),
        ];
        for (next, expected) in probes {
            let losses = vec![1.0_f32, next];
            assert_eq!(
                verdict_from_per_epoch_kl_losses(&losses),
                expected,
                "1.0 → {next} expected {expected:?}"
            );
        }
    }
}
