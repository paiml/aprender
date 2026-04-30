// SHIP-TWO-001 §35 — `apr-cli-distill-train-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-DISTILL-TRAIN-006.
//
// Contract: `contracts/apr-cli-distill-train-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §35.
//
// ## What FALSIFY-APR-DISTILL-TRAIN-006 says
//
//   rule: stage train can resume from precompute cache
//   prediction: If `teacher_logits/` cache exists, stage train SHOULD
//               skip recomputation and proceed to gradient updates.
//   test:       rm -rf run/ckpt; apr distill --stage train ... → uses
//               existing cache, no teacher forward.
//   if_fails:   missing idempotency — would force re-running expensive
//               precompute on every train.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule on the 4-state truth table over
// `(cache_exists, teacher_forward_invoked)`:
//
//   | cache_exists | teacher_forward_invoked | Verdict | Reason                                      |
//   |--------------|-------------------------|---------|---------------------------------------------|
//   | true         | true                    | Fail    | idempotency violated — recomputed anyway    |
//   | true         | false                   | Pass    | cache hit; skipped recomputation            |
//   | false        | true                    | Pass    | cache miss; honestly recomputed             |
//   | false        | false                   | Fail    | no cache, no compute — no work done         |
//
// Future implementations cannot:
// - Always recompute (would Fail the cache-hit case)
// - Skip computation when no cache exists (would Fail the cache-miss case)
// - Treat "cache exists" implicitly as "skip everything including the
//   gradient updates" (separate concern; not what this gate checks).

/// Binary verdict for `FALSIFY-APR-DISTILL-TRAIN-006`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistillTrain006Verdict {
    /// Cache resume worked correctly:
    /// - `cache_exists ∧ ¬teacher_forward_invoked` (cache hit), OR
    /// - `¬cache_exists ∧ teacher_forward_invoked` (cache miss → honest recompute).
    Pass,
    /// Cache resume violated idempotency:
    /// - `cache_exists ∧ teacher_forward_invoked` (recomputed anyway), OR
    /// - `¬cache_exists ∧ ¬teacher_forward_invoked` (no cache, no compute).
    Fail,
}

/// Pure verdict function for FALSIFY-APR-DISTILL-TRAIN-006.
///
/// Inputs:
/// - `cache_exists`: was `<output_dir>/teacher_logits/` populated with a
///   non-empty SHA-256-hashable set of files when stage `train` started?
/// - `teacher_forward_invoked`: did the train loop actually run a teacher
///   forward pass during this run? (Live impl: a counter or a process-level
///   flag set by the teacher-forward kernel dispatcher.)
///
/// Pass iff exactly one of the two booleans is true (XOR semantics).
///
/// # Examples
///
/// Cache hit — `Pass`:
/// ```
/// use aprender::format::distill_train_006::{
///     verdict_from_cache_resume_observation, DistillTrain006Verdict,
/// };
/// let v = verdict_from_cache_resume_observation(true, false);
/// assert_eq!(v, DistillTrain006Verdict::Pass);
/// ```
///
/// Cache exists but teacher forward ran anyway — `Fail`:
/// ```
/// use aprender::format::distill_train_006::{
///     verdict_from_cache_resume_observation, DistillTrain006Verdict,
/// };
/// let v = verdict_from_cache_resume_observation(true, true);
/// assert_eq!(v, DistillTrain006Verdict::Fail);
/// ```
#[must_use]
pub const fn verdict_from_cache_resume_observation(
    cache_exists: bool,
    teacher_forward_invoked: bool,
) -> DistillTrain006Verdict {
    // Pass iff exactly one is true (XOR).
    if cache_exists ^ teacher_forward_invoked {
        DistillTrain006Verdict::Pass
    } else {
        DistillTrain006Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: 4-state truth table — exhaustive enumeration of bool inputs.
    // -------------------------------------------------------------------------
    #[test]
    fn truth_table_t_t_fails_idempotency_violated() {
        assert_eq!(
            verdict_from_cache_resume_observation(true, true),
            DistillTrain006Verdict::Fail,
            "cache exists ∧ recomputed anyway → idempotency violation"
        );
    }

    #[test]
    fn truth_table_t_f_passes_cache_hit() {
        assert_eq!(
            verdict_from_cache_resume_observation(true, false),
            DistillTrain006Verdict::Pass,
            "cache hit — recomputation skipped → Pass"
        );
    }

    #[test]
    fn truth_table_f_t_passes_cache_miss() {
        assert_eq!(
            verdict_from_cache_resume_observation(false, true),
            DistillTrain006Verdict::Pass,
            "cache miss → honest recompute → Pass"
        );
    }

    #[test]
    fn truth_table_f_f_fails_no_work_done() {
        assert_eq!(
            verdict_from_cache_resume_observation(false, false),
            DistillTrain006Verdict::Fail,
            "no cache, no compute → train did no work — Fail"
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: XOR symmetry — verdict invariant under input swap.
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_symmetric_under_input_swap() {
        for (a, b) in [(true, true), (true, false), (false, true), (false, false)] {
            assert_eq!(
                verdict_from_cache_resume_observation(a, b),
                verdict_from_cache_resume_observation(b, a),
                "XOR is symmetric: (a={a}, b={b}) == (a={b}, b={a})"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 3: Mutation-resistance — verdict flips under single-bit flip.
    // -------------------------------------------------------------------------
    #[test]
    fn single_bit_flip_changes_verdict_each_axis() {
        // For every input, flipping exactly one of the two bits must flip
        // the verdict (Pass ↔ Fail). This is exactly the XOR property — so
        // the test catches a future drift to AND/OR semantics.
        for (a, b) in [(true, true), (true, false), (false, true), (false, false)] {
            let baseline = verdict_from_cache_resume_observation(a, b);
            let flip_a = verdict_from_cache_resume_observation(!a, b);
            let flip_b = verdict_from_cache_resume_observation(a, !b);
            assert_ne!(
                baseline, flip_a,
                "flipping cache_exists at ({a},{b}) must flip verdict"
            );
            assert_ne!(
                baseline, flip_b,
                "flipping teacher_forward at ({a},{b}) must flip verdict"
            );
        }
    }

    #[test]
    fn double_bit_flip_preserves_verdict() {
        // Flipping BOTH bits returns to the same equivalence class.
        for (a, b) in [(true, true), (true, false), (false, true), (false, false)] {
            let baseline = verdict_from_cache_resume_observation(a, b);
            let flip_both = verdict_from_cache_resume_observation(!a, !b);
            assert_eq!(
                baseline, flip_both,
                "flipping both bits at ({a},{b}) must preserve verdict"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 4: Documentation-rule sanity — verdict matches the truth-table
    //            comment in the source.
    // -------------------------------------------------------------------------
    #[test]
    fn truth_table_pass_count_is_exactly_two() {
        let mut pass_count = 0;
        let mut fail_count = 0;
        for cache in [true, false] {
            for forward in [true, false] {
                match verdict_from_cache_resume_observation(cache, forward) {
                    DistillTrain006Verdict::Pass => pass_count += 1,
                    DistillTrain006Verdict::Fail => fail_count += 1,
                }
            }
        }
        // XOR over two booleans yields true in exactly half the cases.
        assert_eq!(pass_count, 2);
        assert_eq!(fail_count, 2);
    }

    // -------------------------------------------------------------------------
    // Section 5: Const evaluability — verdict can be computed at compile time.
    // -------------------------------------------------------------------------
    #[test]
    fn const_eval_works_in_static_context() {
        // The verdict fn is `pub const fn` — exercise it at compile time
        // so a future regression to runtime-only would break this test.
        const PASS: DistillTrain006Verdict = verdict_from_cache_resume_observation(true, false);
        const FAIL: DistillTrain006Verdict = verdict_from_cache_resume_observation(true, true);
        assert_eq!(PASS, DistillTrain006Verdict::Pass);
        assert_eq!(FAIL, DistillTrain006Verdict::Fail);
    }
}
