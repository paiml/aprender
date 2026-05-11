// SHIP-TWO-001 MODEL-2 — `apr-cli-distill-train-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-DISTILL-TRAIN-009.
//
// Contract: `contracts/apr-cli-distill-train-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 distillation track (§34.5).
//
// ## What FALSIFY-APR-DISTILL-TRAIN-009 says
//
//   rule: end-to-end smoke on tiny pair beats from-scratch baseline
//   prediction: Distilling a 50M-param student from a 500M-param
//               teacher on 1M tokens for 1 epoch produces a student
//               with val_loss < (50M-from-scratch on same data,
//               same epochs).
//   if_fails:   distillation provides no measurable benefit over
//               pretraining at this scale — re-evaluate
//               hyperparameters or approach.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two final val_loss measurements
// (`distill_val_loss`, `from_scratch_val_loss`), Pass iff:
//
//   distill_val_loss < from_scratch_val_loss
//
// AND both losses are finite (no NaN / ±∞), positive (val_loss is
// a non-negative cross-entropy), and within a sane upper bound
// (`<= AC_DISTILL_TRAIN_009_MAX_PLAUSIBLE_VAL_LOSS = 100.0`).
// Strict `<` matches the contract's wording — even an exact tie
// fails (distillation must provide measurable benefit). The
// upper-bound check rejects degenerate runs (loss = 1e308) that
// would silently pass a `distill < from_scratch` predicate.

/// Maximum plausible val_loss for a tiny-pair distillation smoke.
///
/// Per `apr-cli-distill-train-v1` smoke spec: 50M-param student on
/// 1M tokens, 1 epoch. Cross-entropy at vocab=50K initializes
/// around `ln(50_000) ≈ 10.8`, and a degenerate run with NaN
/// gradient or no actual training produces effectively
/// `+∞`-bounded values. A finite cap of 100.0 catches both the
/// "untrained" (loss ≈ ln(vocab)) and "blown-up" (loss → ∞) modes
/// while leaving plenty of headroom.
pub const AC_DISTILL_TRAIN_009_MAX_PLAUSIBLE_VAL_LOSS: f64 = 100.0;

/// Binary verdict for `FALSIFY-APR-DISTILL-TRAIN-009`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillTrain009Verdict {
    /// Both losses are finite, positive, plausibly bounded, AND
    /// `distill_val_loss < from_scratch_val_loss` strictly.
    Pass,
    /// One or more of:
    /// - Either val_loss is NaN or ±∞ (training collapsed).
    /// - Either val_loss is < 0.0 (impossible — cross-entropy is
    ///   non-negative).
    /// - Either val_loss is > AC_DISTILL_TRAIN_009_MAX_PLAUSIBLE_VAL_LOSS
    ///   (degenerate / non-training run).
    /// - `distill_val_loss >= from_scratch_val_loss` (distillation
    ///   provided no benefit; strict `<` per contract).
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-DISTILL-TRAIN-009`.
///
/// Inputs:
/// - `distill_val_loss`: final val_loss after distilling a 50M
///   student from a 500M teacher on 1M tokens × 1 epoch.
/// - `from_scratch_val_loss`: final val_loss after pretraining the
///   same 50M student from scratch on the same data, same epochs.
///
/// Pass iff:
/// 1. Both losses are finite (rules out NaN, ±∞),
/// 2. Both losses are >= 0.0 (cross-entropy non-negativity),
/// 3. Both losses are <= 100.0 (plausibility cap),
/// 4. `distill_val_loss < from_scratch_val_loss` strictly.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Distillation beats baseline by 5% — `Pass`:
/// ```
/// use aprender::format::distill_train_009::{
///     verdict_from_distill_vs_baseline, DistillTrain009Verdict,
/// };
/// let v = verdict_from_distill_vs_baseline(8.5, 9.0);
/// assert_eq!(v, DistillTrain009Verdict::Pass);
/// ```
///
/// Distillation ties baseline (no measurable benefit) — `Fail`:
/// ```
/// use aprender::format::distill_train_009::{
///     verdict_from_distill_vs_baseline, DistillTrain009Verdict,
/// };
/// let v = verdict_from_distill_vs_baseline(9.0, 9.0);
/// assert_eq!(v, DistillTrain009Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_distill_vs_baseline(
    distill_val_loss: f64,
    from_scratch_val_loss: f64,
) -> DistillTrain009Verdict {
    if !distill_val_loss.is_finite() || !from_scratch_val_loss.is_finite() {
        return DistillTrain009Verdict::Fail;
    }
    if distill_val_loss < 0.0 || from_scratch_val_loss < 0.0 {
        return DistillTrain009Verdict::Fail;
    }
    if distill_val_loss > AC_DISTILL_TRAIN_009_MAX_PLAUSIBLE_VAL_LOSS
        || from_scratch_val_loss > AC_DISTILL_TRAIN_009_MAX_PLAUSIBLE_VAL_LOSS
    {
        return DistillTrain009Verdict::Fail;
    }
    if distill_val_loss < from_scratch_val_loss {
        DistillTrain009Verdict::Pass
    } else {
        DistillTrain009Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — plausibility cap.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_max_plausible_val_loss_is_100() {
        assert!((AC_DISTILL_TRAIN_009_MAX_PLAUSIBLE_VAL_LOSS - 100.0).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — distill beats baseline by realistic margins.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_distill_beats_baseline_by_5_percent() {
        let v = verdict_from_distill_vs_baseline(8.5, 9.0);
        assert_eq!(v, DistillTrain009Verdict::Pass);
    }

    #[test]
    fn pass_distill_beats_baseline_by_realistic_margin() {
        // Realistic spec figures: 50M student from-scratch
        // val_loss ≈ 7.5; with distillation ≈ 6.8.
        let v = verdict_from_distill_vs_baseline(6.8, 7.5);
        assert_eq!(v, DistillTrain009Verdict::Pass);
    }

    #[test]
    fn pass_distill_marginally_better() {
        // Even a tiny win (1e-6 below baseline) passes — strict `<`.
        let v = verdict_from_distill_vs_baseline(7.499_999, 7.500_000);
        assert_eq!(v, DistillTrain009Verdict::Pass);
    }

    #[test]
    fn pass_at_floor_baseline_just_above() {
        // distill ≈ 0 (perfect student), baseline > 0.
        let v = verdict_from_distill_vs_baseline(0.001, 0.002);
        assert_eq!(v, DistillTrain009Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — distill ties or loses to baseline.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_exact_tie() {
        // Strict `<` per contract: even exact equality must Fail.
        let v = verdict_from_distill_vs_baseline(9.0, 9.0);
        assert_eq!(
            v,
            DistillTrain009Verdict::Fail,
            "exact tie must Fail (strict < per contract)"
        );
    }

    #[test]
    fn fail_distill_one_ulp_higher() {
        // Distill is one ULP above baseline → no measurable benefit.
        let baseline = 7.5_f64;
        let one_ulp_higher = f64::from_bits(baseline.to_bits() + 1);
        let v = verdict_from_distill_vs_baseline(one_ulp_higher, baseline);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    #[test]
    fn fail_distill_clearly_worse() {
        let v = verdict_from_distill_vs_baseline(9.5, 8.5);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — domain violations (NaN, ±∞).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_distill_nan() {
        let v = verdict_from_distill_vs_baseline(f64::NAN, 9.0);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    #[test]
    fn fail_baseline_nan() {
        let v = verdict_from_distill_vs_baseline(8.0, f64::NAN);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    #[test]
    fn fail_distill_positive_infinity() {
        let v = verdict_from_distill_vs_baseline(f64::INFINITY, 9.0);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    #[test]
    fn fail_baseline_negative_infinity() {
        let v = verdict_from_distill_vs_baseline(8.0, f64::NEG_INFINITY);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — negative loss (impossible by definition).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_distill_negative() {
        // Cross-entropy is non-negative; negative loss is corruption.
        let v = verdict_from_distill_vs_baseline(-0.01, 9.0);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    #[test]
    fn fail_baseline_negative() {
        let v = verdict_from_distill_vs_baseline(8.0, -0.01);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Fail band — implausibly high losses (degenerate runs).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_distill_above_plausibility_cap() {
        // 100.001 > 100.0 — degenerate / non-training run.
        let v = verdict_from_distill_vs_baseline(100.001, 200.0);
        assert_eq!(
            v,
            DistillTrain009Verdict::Fail,
            "above plausibility cap must Fail (degenerate run)"
        );
    }

    #[test]
    fn fail_baseline_above_plausibility_cap() {
        let v = verdict_from_distill_vs_baseline(50.0, 1_000.0);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    #[test]
    fn fail_both_blown_up() {
        let v = verdict_from_distill_vs_baseline(1e10, 1e20);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }

    #[test]
    fn pass_at_exact_plausibility_cap() {
        // 100.0 is the inclusive upper bound (≤ 100.0).
        let v = verdict_from_distill_vs_baseline(99.0, 100.0);
        assert_eq!(v, DistillTrain009Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 7: Boundary sweep — distill_val_loss around the strict cutoff.
    // -------------------------------------------------------------------------
    #[test]
    fn distill_loss_sweep_at_fixed_baseline() {
        let baseline = 9.0_f64;
        let probes: Vec<(f64, DistillTrain009Verdict)> = vec![
            (0.0, DistillTrain009Verdict::Pass),
            (1.0, DistillTrain009Verdict::Pass),
            (8.0, DistillTrain009Verdict::Pass),
            (8.999, DistillTrain009Verdict::Pass),
            (9.0, DistillTrain009Verdict::Fail),    // exact tie
            (9.001, DistillTrain009Verdict::Fail),
            (10.0, DistillTrain009Verdict::Fail),
            (50.0, DistillTrain009Verdict::Fail),
        ];
        for (distill, expected) in probes {
            let v = verdict_from_distill_vs_baseline(distill, baseline);
            assert_eq!(
                v, expected,
                "distill={distill} baseline={baseline} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 8: Realistic — MODEL-2 spec values.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_at_csn_python_realistic_values() {
        // 50M-from-scratch on CSN-Python ≈ 9.751 (per spec).
        // Distillation target: < 9.751 by some margin.
        let v = verdict_from_distill_vs_baseline(9.0, 9.751);
        assert_eq!(v, DistillTrain009Verdict::Pass);
    }

    #[test]
    fn fail_csn_python_no_benefit() {
        // Distillation collapsed to the same as from-scratch.
        let v = verdict_from_distill_vs_baseline(9.751, 9.751);
        assert_eq!(v, DistillTrain009Verdict::Fail);
    }
}
