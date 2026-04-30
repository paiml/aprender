// SHIP-TWO-001 §35 — `apr-cli-distill-train-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-DISTILL-TRAIN-001.
//
// Contract: `contracts/apr-cli-distill-train-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §35
// (`apr distill` Standard strategy is currently a stub at distill.rs:1464,
// just `tensor_clone()`, no gradient training).
//
// ## What FALSIFY-APR-DISTILL-TRAIN-001 says
//
//   rule: real training (not stub) — student tensors differ post-train
//   prediction: After `apr distill --stage train`, at least one tensor
//   in `student.apr` differs from input student by >Q4K tolerance.
//
// Today this FAILS — distill is `tensor_clone()` so all max_diffs are 0.
// Once §35.3 implementation lands (KL+CE training loop), the live test
// passes and this contract flips to ACTIVE.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// The decision rule — "at least one tensor's max_diff% exceeds Q4K
// tolerance" — is pinned. A future implementation cannot regress it
// silently. The mutation survey covers the empty-input, all-zero
// (stub), one-large, all-small, just-at-boundary, and non-finite cases.

/// Q4K-tolerance percentage threshold per CLAUDE.md (±5% element-wise).
///
/// A real-training pass MUST produce at least one tensor whose
/// `max(|new - input|) / max(|input|) * 100` exceeds this threshold.
/// Stub `tensor_clone()` behaviour produces all-zero diffs and so
/// fails this gate — exactly the §35 finding.
pub const AC_DISTILL_TRAIN_001_Q4K_TOLERANCE_PCT: f32 = 5.0;

/// Binary verdict for `FALSIFY-APR-DISTILL-TRAIN-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillTrain001Verdict {
    /// At least one tensor's `max_diff%` exceeds
    /// `AC_DISTILL_TRAIN_001_Q4K_TOLERANCE_PCT` (5.0). Real gradient-
    /// based training has flowed to weight updates; student.apr is no
    /// longer a metadata-only clone of the input.
    Pass,
    /// One or more of:
    /// - Empty `max_diffs_pct` slice (no tensors compared — caller error).
    /// - At least one diff is non-finite (NaN, ±∞).
    /// - At least one diff is negative (a percentage cannot be negative;
    ///   conservative `Fail` — implies a buggy `apr diff --values`).
    /// - **Every** diff is ≤ 5% — the stub-behaviour signature: the
    ///   shipped distill loop just cloned tensors, never updated weights.
    Fail,
}

/// Pure verdict function for FALSIFY-APR-DISTILL-TRAIN-001.
///
/// Input: per-tensor `max(|new - input|) / max(|input|) * 100` percentages,
/// one per tensor in the student.apr (e.g., one per `apr diff --values`
/// line). Typical Qwen2.5-Coder-7B has 339 tensors.
///
/// # Examples
///
/// Stub (`tensor_clone()`) — all zeros — is `Fail`:
/// ```
/// use aprender::format::distill_train_001::{
///     verdict_from_max_diff_pct, DistillTrain001Verdict,
/// };
/// let stub_diffs = vec![0.0_f32; 339];
/// assert_eq!(
///     verdict_from_max_diff_pct(&stub_diffs),
///     DistillTrain001Verdict::Fail,
/// );
/// ```
///
/// Real training — at least one tensor moved by >5% — is `Pass`:
/// ```
/// use aprender::format::distill_train_001::{
///     verdict_from_max_diff_pct, DistillTrain001Verdict,
/// };
/// let mut diffs = vec![0.5_f32; 339];
/// diffs[100] = 12.4; // one tensor moved 12.4%
/// assert_eq!(
///     verdict_from_max_diff_pct(&diffs),
///     DistillTrain001Verdict::Pass,
/// );
/// ```
#[must_use]
pub fn verdict_from_max_diff_pct(max_diffs_pct: &[f32]) -> DistillTrain001Verdict {
    if max_diffs_pct.is_empty() {
        return DistillTrain001Verdict::Fail;
    }
    let mut any_above = false;
    for &d in max_diffs_pct {
        if !d.is_finite() || d < 0.0 {
            return DistillTrain001Verdict::Fail;
        }
        if d > AC_DISTILL_TRAIN_001_Q4K_TOLERANCE_PCT {
            any_above = true;
        }
    }
    if any_above {
        DistillTrain001Verdict::Pass
    } else {
        DistillTrain001Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_up_f32(x: f32) -> f32 {
        f32::from_bits(x.to_bits() + 1)
    }

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — threshold matches CLAUDE.md ±5% Q4K spec.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_q4k_tolerance_is_five_percent() {
        assert_eq!(AC_DISTILL_TRAIN_001_Q4K_TOLERANCE_PCT, 5.0);
    }

    // -------------------------------------------------------------------------
    // Section 2: §35 stub-behaviour signature — `tensor_clone()` produces
    //            all-zero max_diffs → Fail (catches future regression to stub).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_all_zeros_is_stub_signature() {
        // 339 = canonical Qwen2.5-Coder-7B tensor count.
        let diffs = vec![0.0_f32; 339];
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Fail,
            "all-zero max_diffs is the §35 tensor_clone stub signature; must Fail"
        );
    }

    #[test]
    fn fail_all_below_threshold() {
        // Even small noisy diffs from quantization round-trip are not
        // sufficient evidence of real training.
        let diffs = vec![1.0_f32; 339];
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Fail
        );
    }

    #[test]
    fn fail_all_at_or_below_boundary() {
        // Exact-boundary 5.0% is NOT above the threshold (strict `>`),
        // so a uniform 5.0% is still stub-shaped Fail.
        let diffs = vec![AC_DISTILL_TRAIN_001_Q4K_TOLERANCE_PCT; 339];
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Fail,
            "exact 5.0 must Fail (strict `>` on threshold)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: Pass band — at least one tensor moved by > Q4K tolerance.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_one_large_move_among_many_small() {
        let mut diffs = vec![0.1_f32; 339];
        diffs[42] = 12.4;
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Pass
        );
    }

    #[test]
    fn pass_just_above_threshold() {
        let just_above = next_up_f32(AC_DISTILL_TRAIN_001_Q4K_TOLERANCE_PCT);
        assert!(just_above > AC_DISTILL_TRAIN_001_Q4K_TOLERANCE_PCT);
        let mut diffs = vec![0.0_f32; 339];
        diffs[100] = just_above;
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Pass,
            "5.0 + 1 ULP must Pass (strict `>` boundary)"
        );
    }

    #[test]
    fn pass_all_above_threshold() {
        let diffs = vec![15.0_f32; 339];
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Empty-input — caller error → conservative Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_input() {
        let diffs: Vec<f32> = vec![];
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Fail,
            "empty max_diffs implies no tensors compared — caller error"
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Domain violation — non-finite or negative percentages Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_nan_in_any_position() {
        for pos in [0_usize, 50, 338] {
            let mut diffs = vec![10.0_f32; 339];
            diffs[pos] = f32::NAN;
            assert_eq!(
                verdict_from_max_diff_pct(&diffs),
                DistillTrain001Verdict::Fail,
                "NaN at position {pos} must Fail (domain violation)"
            );
        }
    }

    #[test]
    fn fail_positive_infinity() {
        let mut diffs = vec![10.0_f32; 339];
        diffs[7] = f32::INFINITY;
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Fail
        );
    }

    #[test]
    fn fail_negative_infinity() {
        let mut diffs = vec![10.0_f32; 339];
        diffs[7] = f32::NEG_INFINITY;
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Fail
        );
    }

    #[test]
    fn fail_negative_diff_is_domain_violation() {
        // A percentage cannot be negative; `apr diff --values` emitting
        // negative max_diff implies a tooling bug, not real training.
        let mut diffs = vec![10.0_f32; 339];
        diffs[7] = -1.0;
        assert_eq!(
            verdict_from_max_diff_pct(&diffs),
            DistillTrain001Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Single-tensor sweep — flipping any one tensor above
    //            threshold flips the whole verdict to Pass.
    // -------------------------------------------------------------------------
    #[test]
    fn single_tensor_above_threshold_flips_to_pass_at_each_index() {
        for i in [0_usize, 1, 100, 169, 338] {
            let mut diffs = vec![0.5_f32; 339];
            diffs[i] = 50.0;
            assert_eq!(
                verdict_from_max_diff_pct(&diffs),
                DistillTrain001Verdict::Pass,
                "single tensor above threshold at index {i} must Pass"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Monotonicity sweep at uniform diff.
    // -------------------------------------------------------------------------
    #[test]
    fn monotonicity_sweep_uniform_diff() {
        let probes: Vec<(f32, DistillTrain001Verdict)> = vec![
            (0.0, DistillTrain001Verdict::Fail),
            (0.5, DistillTrain001Verdict::Fail),
            (4.999, DistillTrain001Verdict::Fail),
            (
                AC_DISTILL_TRAIN_001_Q4K_TOLERANCE_PCT,
                DistillTrain001Verdict::Fail,
            ),
            (
                next_up_f32(AC_DISTILL_TRAIN_001_Q4K_TOLERANCE_PCT),
                DistillTrain001Verdict::Pass,
            ),
            (5.001, DistillTrain001Verdict::Pass),
            (10.0, DistillTrain001Verdict::Pass),
            (100.0, DistillTrain001Verdict::Pass),
        ];
        for (d, expected) in probes {
            let diffs = vec![d; 339];
            assert_eq!(
                verdict_from_max_diff_pct(&diffs),
                expected,
                "uniform diff {d} expected {expected:?}"
            );
        }
    }
}
