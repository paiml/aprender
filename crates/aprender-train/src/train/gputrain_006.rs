//! FALSIFY-GPUTRAIN-006 / INV-GPUTRAIN-006 — algorithm-level PARTIAL discharge.
//!
//! Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §14
//! (task #132 CUDA training backend gap).
//!
//! Contract: `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 → v1.1.0
//! binds INV-GPUTRAIN-006 at PARTIAL_ALGORITHM_LEVEL via two pure threshold
//! functions:
//!
//!   1. `verdict_from_loss_delta(delta_abs, tolerance) -> Gputrain006Verdict`
//!      — single-step inequality: Pass iff both inputs finite, both ≥ 0, and
//!      `delta_abs <= tolerance`.
//!
//!   2. `verdict_from_loss_trajectories(run_a, run_b, tolerance) -> Verdict`
//!      — aggregate: both slices same non-zero length, every pair finite,
//!      every `|a[k] - b[k]| <= tolerance`. Empty or mismatched-length is
//!      conservatively Fail.
//!
//! INV-GPUTRAIN-006 states: "Same-device seed reproducibility holds (two
//! `cuda:0` runs at seed=0, `|Δloss[k]| ≤ 1e-5`)." The 1e-5 floor is looser
//! than CPU's 1e-6 (peer contract INV-TRAIN-006) to allow cuBLAS non-
//! determinism but still tight enough to catch a seed-plumbing regression.
//!
//! The compute-heavy portion (actually replaying two 100-step cuda:0 runs
//! through `CudaTransformerTrainer` and capturing per-step losses) is
//! intentionally out of scope here; the threshold rule is what the live
//! parity-runner calls, and changing the 1e-5 constant or the pair-wise
//! inequality breaks this test before any CUDA kernel launches.

/// Maximum tolerated absolute loss delta at any step k between two
/// same-device runs at the same seed. Looser than CPU's 1e-6 per peer
/// contract INV-TRAIN-006 to accommodate cuBLAS warp-reduction non-
/// determinism, but tight enough that a seed-plumbing regression (e.g.
/// `rand::thread_rng()` leaked into a supposedly deterministic path)
/// will fail the gate.
pub const AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA: f32 = 1e-5;

/// Binary verdict for FALSIFY-GPUTRAIN-006.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gputrain006Verdict {
    /// Both runs' losses agree within tolerance at every step.
    Pass,
    /// Any single-step violation, any non-finite value, empty input, or
    /// length mismatch — all conservatively Fail.
    Fail,
}

/// Single-step threshold rule: given a pre-computed absolute loss delta
/// and the tolerance, Pass iff both are finite, both non-negative, and
/// the delta is at most the tolerance (inclusive). `const fn` so the
/// boundary at exactly `AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA` is const-
/// evaluable.
#[must_use]
pub const fn verdict_from_loss_delta(delta_abs: f32, tolerance: f32) -> Gputrain006Verdict {
    if !delta_abs.is_finite() || !tolerance.is_finite() {
        return Gputrain006Verdict::Fail;
    }
    if delta_abs < 0.0 || tolerance < 0.0 {
        return Gputrain006Verdict::Fail;
    }
    if delta_abs <= tolerance {
        Gputrain006Verdict::Pass
    } else {
        Gputrain006Verdict::Fail
    }
}

/// Aggregate trajectory rule: given two per-step loss arrays and a
/// tolerance, Pass iff both have the same non-zero length, every element
/// in both is finite, and every pair-wise `|a[k] - b[k]|` is at most the
/// tolerance. Empty arrays, length mismatch, or any non-finite element is
/// Fail — all three are legitimate counter-examples for a broken
/// reproducibility harness.
#[must_use]
pub fn verdict_from_loss_trajectories(
    run_a: &[f32],
    run_b: &[f32],
    tolerance: f32,
) -> Gputrain006Verdict {
    if run_a.is_empty() || run_b.is_empty() || run_a.len() != run_b.len() {
        return Gputrain006Verdict::Fail;
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Gputrain006Verdict::Fail;
    }
    for (a, b) in run_a.iter().zip(run_b.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return Gputrain006Verdict::Fail;
        }
        let delta = (a - b).abs();
        if delta > tolerance {
            return Gputrain006Verdict::Fail;
        }
    }
    Gputrain006Verdict::Pass
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GPUTRAIN-006 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-GPUTRAIN-006 algorithm-level PARTIAL discharge: prove the
    /// same-device seed reproducibility threshold rule + trajectory
    /// aggregate. Any mutation that flips the comparison direction,
    /// relaxes the finiteness guard, silently accepts a length mismatch,
    /// or defaults the tolerance to infinity must break this test before
    /// the live CUDA parity run.
    #[test]
    fn falsify_gputrain_006_seed_reproducibility_threshold_logic() {
        let tol = AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA;

        // Section 1: boundary — delta exactly equal to tolerance. Pass
        // per the `<=` inclusive-ceiling rule. Any mutation to strict
        // `<` flips this to Fail.
        assert_eq!(
            verdict_from_loss_delta(tol, tol),
            Gputrain006Verdict::Pass,
            "delta == tolerance (1e-5) must Pass per inclusive ceiling",
        );

        // Section 2: above tolerance by ULP. Any mutation that relaxed
        // to a ±epsilon compare or flipped the inequality would make
        // this Pass.
        let one_ulp_above = f32::from_bits(tol.to_bits() + 1);
        assert!(one_ulp_above > tol);
        assert_eq!(
            verdict_from_loss_delta(one_ulp_above, tol),
            Gputrain006Verdict::Fail,
            "one ULP above tolerance must Fail",
        );
        // A larger overshoot — the defect shape where a seed plumbing
        // regression breaks determinism outright.
        assert_eq!(
            verdict_from_loss_delta(1e-3, tol),
            Gputrain006Verdict::Fail,
            "100× tolerance must Fail (visible seed plumbing regression)",
        );

        // Section 3: trajectory — single-step fail. 99 steps within
        // tolerance plus ONE step above must Fail. Mirrors the real
        // failure mode: a reproducibility regression often shows up at
        // a specific layer depth (e.g. the first LayerNorm backward
        // where cuBLAS warp-reduction order leaked).
        let mut run_a = vec![1.0f32; 100];
        let mut run_b = vec![1.0f32; 100];
        run_b[42] = 1.0 + 1e-3; // delta = 1e-3 > tol
        assert_eq!(
            verdict_from_loss_trajectories(&run_a, &run_b, tol),
            Gputrain006Verdict::Fail,
            "single-step trajectory violation at k=42 must Fail",
        );
        // Restore k=42 to within tolerance — everything else unchanged
        // must now Pass.
        run_b[42] = 1.0 + (tol / 2.0);
        assert_eq!(
            verdict_from_loss_trajectories(&run_a, &run_b, tol),
            Gputrain006Verdict::Pass,
            "all-within-tolerance trajectory must Pass",
        );
        // Sanity: a tiny drift on every step is still Pass as long as
        // each delta is within tolerance.
        for i in 0..run_a.len() {
            run_a[i] = 2.0 + (i as f32) * 1e-3;
            run_b[i] = run_a[i] + (tol / 10.0);
        }
        assert_eq!(
            verdict_from_loss_trajectories(&run_a, &run_b, tol),
            Gputrain006Verdict::Pass,
            "uniform within-tolerance drift across 100 steps must Pass",
        );

        // Section 4: length mismatch. Two runs of different length can't
        // be compared pairwise — conservative Fail (some other bug in
        // the harness cut one run short).
        let short = vec![1.0f32; 50];
        let long = vec![1.0f32; 100];
        assert_eq!(
            verdict_from_loss_trajectories(&short, &long, tol),
            Gputrain006Verdict::Fail,
            "length mismatch (50 vs 100) must Fail",
        );
        assert_eq!(
            verdict_from_loss_trajectories(&long, &short, tol),
            Gputrain006Verdict::Fail,
            "reverse length mismatch must also Fail",
        );

        // Section 5: empty input. A defensive `is_empty()` check
        // prevents a vacuously-true "no steps" from passing the gate.
        let empty: Vec<f32> = vec![];
        let one = vec![1.0f32];
        assert_eq!(
            verdict_from_loss_trajectories(&empty, &empty, tol),
            Gputrain006Verdict::Fail,
            "both-empty trajectories must Fail (no steps compared)",
        );
        assert_eq!(
            verdict_from_loss_trajectories(&empty, &one, tol),
            Gputrain006Verdict::Fail,
            "one-empty one-nonempty must Fail",
        );

        // Section 6: non-finite elements. A NaN or ±∞ anywhere in
        // either run must propagate to Fail. Catches the failure mode
        // where a GradScaler overflow emitted NaN and the harness kept
        // plotting.
        let mut nan_a = vec![1.0f32; 10];
        let nan_b = vec![1.0f32; 10];
        nan_a[3] = f32::NAN;
        assert_eq!(
            verdict_from_loss_trajectories(&nan_a, &nan_b, tol),
            Gputrain006Verdict::Fail,
            "NaN in run_a must Fail",
        );
        let mut inf_b = vec![1.0f32; 10];
        inf_b[7] = f32::INFINITY;
        assert_eq!(
            verdict_from_loss_trajectories(&nan_b, &inf_b, tol),
            Gputrain006Verdict::Fail,
            "+inf in run_b must Fail",
        );
        // Non-finite single-step delta.
        assert_eq!(
            verdict_from_loss_delta(f32::NAN, tol),
            Gputrain006Verdict::Fail,
            "NaN delta must Fail",
        );
        assert_eq!(
            verdict_from_loss_delta(1e-6, f32::INFINITY),
            Gputrain006Verdict::Fail,
            "infinite tolerance must Fail (no rubber-stamp Pass)",
        );
        // Negative tolerance / delta.
        assert_eq!(
            verdict_from_loss_delta(-1e-6, tol),
            Gputrain006Verdict::Fail,
            "negative delta must Fail (caller passed raw a-b, not |a-b|)",
        );
        assert_eq!(
            verdict_from_loss_delta(1e-6, -1e-5),
            Gputrain006Verdict::Fail,
            "negative tolerance must Fail (nonsense threshold)",
        );

        // Section 7: provenance pin — the 1e-5 tolerance is load-
        // bearing and lockstep with the YAML contract rule and peer
        // INV-TRAIN-006 (CPU 1e-6, CUDA 1e-5). Any future tightening
        // (e.g. after trueno#203 lands deterministic kernels) or
        // relaxation must move the constant, the YAML rule, and this
        // test together.
        assert!(
            (AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA - 1e-5).abs() < 1e-9,
            "INV-GPUTRAIN-006 tolerance is 1e-5 \
             (spec §14.4 / gpu-training-backend-v1 INV-GPUTRAIN-006)",
        );
    }
}
