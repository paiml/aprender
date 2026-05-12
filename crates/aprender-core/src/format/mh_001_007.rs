// SHIP-TWO-001 — `metaheuristics-v1` algorithm-level PARTIAL discharge
// for FALSIFY-MH-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/metaheuristics-v1.yaml`.

// ===========================================================================
// MH-001 — best-so-far is non-increasing across iterations
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mh001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_best_monotone(best_history: &[f64]) -> Mh001Verdict {
    if best_history.is_empty() { return Mh001Verdict::Fail; }
    if best_history.iter().any(|v| !v.is_finite()) { return Mh001Verdict::Fail; }
    for w in best_history.windows(2) {
        if w[1] > w[0] + 1e-12 { return Mh001Verdict::Fail; }
    }
    Mh001Verdict::Pass
}

// ===========================================================================
// MH-002 — SA: final_best <= initial_best
// MH-003 — GA: final_best <= initial_best
// MH-004 — PSO: final_best <= initial_best
//
// All three reduce to the same monotonic-improvement decision rule;
// share one verdict fn so identical-shape gates aren't duplicated.
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mh002Verdict { Pass, Fail }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mh003Verdict { Pass, Fail }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mh004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_final_best_le_initial(initial: f64, final_: f64) -> bool {
    if !initial.is_finite() || !final_.is_finite() { return false; }
    final_ <= initial + 1e-12
}

#[must_use]
pub fn verdict_from_sa_improvement(initial: f64, final_: f64) -> Mh002Verdict {
    if verdict_from_final_best_le_initial(initial, final_) { Mh002Verdict::Pass } else { Mh002Verdict::Fail }
}

#[must_use]
pub fn verdict_from_ga_improvement(initial: f64, final_: f64) -> Mh003Verdict {
    if verdict_from_final_best_le_initial(initial, final_) { Mh003Verdict::Pass } else { Mh003Verdict::Fail }
}

#[must_use]
pub fn verdict_from_pso_improvement(initial: f64, final_: f64) -> Mh004Verdict {
    if verdict_from_final_best_le_initial(initial, final_) { Mh004Verdict::Pass } else { Mh004Verdict::Fail }
}

// ===========================================================================
// MH-005 — SA acceptance probability in (0, 1] for any (Δ, T>0)
//
// P(accept) = exp(-max(0, Δ) / T); strict 0 < P <= 1 only when Δ <
// +inf, T > 0.
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mh005Verdict { Pass, Fail }

#[must_use]
pub fn sa_acceptance(delta_e: f64, t: f64) -> Option<f64> {
    if !delta_e.is_finite() || !t.is_finite() || t <= 0.0 { return None; }
    if delta_e <= 0.0 { return Some(1.0); } // always accept improvements
    Some((-delta_e / t).exp())
}

#[must_use]
pub fn verdict_from_sa_acceptance_bounds(delta_e: f64, t: f64) -> Mh005Verdict {
    match sa_acceptance(delta_e, t) {
        Some(p) if p > 0.0 && p <= 1.0 + 1e-12 => Mh005Verdict::Pass,
        _ => Mh005Verdict::Fail,
    }
}

// ===========================================================================
// MH-006 — PSO velocity clamping: |v_i| <= v_max
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mh006Verdict { Pass, Fail }

#[must_use]
pub fn clamp_velocity(v: &[f64], v_max: f64) -> Option<Vec<f64>> {
    if v_max <= 0.0 || !v_max.is_finite() { return None; }
    if v.iter().any(|x| !x.is_finite()) { return None; }
    Some(v.iter().map(|x| x.clamp(-v_max, v_max)).collect())
}

#[must_use]
pub fn verdict_from_velocity_clamping(v_clamped: &[f64], v_max: f64) -> Mh006Verdict {
    if v_clamped.is_empty() || v_max <= 0.0 || !v_max.is_finite() { return Mh006Verdict::Fail; }
    for x in v_clamped {
        if !x.is_finite() { return Mh006Verdict::Fail; }
        if x.abs() > v_max + 1e-12 { return Mh006Verdict::Fail; }
    }
    Mh006Verdict::Pass
}

// ===========================================================================
// MH-007 — GA SBX crossover: children within search-space bounds
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mh007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_crossover_bounds(
    children: &[f64],
    lower: f64,
    upper: f64,
) -> Mh007Verdict {
    if children.is_empty() || !lower.is_finite() || !upper.is_finite() { return Mh007Verdict::Fail; }
    if lower > upper { return Mh007Verdict::Fail; }
    for c in children {
        if !c.is_finite() { return Mh007Verdict::Fail; }
        if *c < lower - 1e-12 || *c > upper + 1e-12 { return Mh007Verdict::Fail; }
    }
    Mh007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // MH-001
    #[test] fn mh001_pass_decreasing() {
        let h = vec![10.0_f64, 8.0, 5.0, 3.0, 2.0];
        assert_eq!(verdict_from_best_monotone(&h), Mh001Verdict::Pass);
    }
    #[test] fn mh001_pass_constant() {
        let h = vec![5.0_f64; 10];
        assert_eq!(verdict_from_best_monotone(&h), Mh001Verdict::Pass);
    }
    #[test] fn mh001_fail_increase() {
        let h = vec![10.0_f64, 5.0, 7.0]; // regressed at step 2
        assert_eq!(verdict_from_best_monotone(&h), Mh001Verdict::Fail);
    }
    #[test] fn mh001_fail_empty() {
        assert_eq!(verdict_from_best_monotone(&[]), Mh001Verdict::Fail);
    }
    #[test] fn mh001_fail_nan() {
        let h = vec![10.0_f64, f64::NAN];
        assert_eq!(verdict_from_best_monotone(&h), Mh001Verdict::Fail);
    }

    // MH-002 / 003 / 004 (same shape)
    #[test] fn mh002_pass_improvement() {
        assert_eq!(verdict_from_sa_improvement(10.0, 5.0), Mh002Verdict::Pass);
    }
    #[test] fn mh002_pass_no_change() {
        assert_eq!(verdict_from_sa_improvement(5.0, 5.0), Mh002Verdict::Pass);
    }
    #[test] fn mh002_fail_regression() {
        assert_eq!(verdict_from_sa_improvement(5.0, 7.0), Mh002Verdict::Fail);
    }
    #[test] fn mh003_pass_ga() {
        assert_eq!(verdict_from_ga_improvement(100.0, 80.0), Mh003Verdict::Pass);
    }
    #[test] fn mh003_fail_ga_loses_elite() {
        assert_eq!(verdict_from_ga_improvement(100.0, 105.0), Mh003Verdict::Fail);
    }
    #[test] fn mh004_pass_pso() {
        assert_eq!(verdict_from_pso_improvement(50.0, 25.0), Mh004Verdict::Pass);
    }
    #[test] fn mh004_fail_pso() {
        assert_eq!(verdict_from_pso_improvement(50.0, 60.0), Mh004Verdict::Fail);
    }

    // MH-005
    #[test] fn mh005_pass_improvement_always_accepted() {
        // Δ <= 0 → P = 1 (always accept).
        assert_eq!(verdict_from_sa_acceptance_bounds(-1.0, 100.0), Mh005Verdict::Pass);
    }
    #[test] fn mh005_pass_worse_finite_t() {
        // Δ > 0, T > 0 → P = exp(-Δ/T) in (0, 1).
        assert_eq!(verdict_from_sa_acceptance_bounds(2.0, 5.0), Mh005Verdict::Pass);
    }
    #[test] fn mh005_pass_high_temp() {
        // High T → P close to 1.
        assert_eq!(verdict_from_sa_acceptance_bounds(0.001, 1e6), Mh005Verdict::Pass);
    }
    #[test] fn mh005_fail_zero_t() {
        assert_eq!(verdict_from_sa_acceptance_bounds(1.0, 0.0), Mh005Verdict::Fail);
    }
    #[test] fn mh005_fail_negative_t() {
        assert_eq!(verdict_from_sa_acceptance_bounds(1.0, -1.0), Mh005Verdict::Fail);
    }
    #[test] fn mh005_fail_extreme_underflow() {
        // Δ=1e10, T=1 → P ≈ exp(-1e10) underflows to 0; verdict requires P > 0.
        assert_eq!(verdict_from_sa_acceptance_bounds(1e10, 1.0), Mh005Verdict::Fail);
    }

    // MH-006
    #[test] fn mh006_pass_clamped() {
        let v = clamp_velocity(&[10.0_f64, -10.0, 0.5, -0.3], 5.0).unwrap();
        assert_eq!(verdict_from_velocity_clamping(&v, 5.0), Mh006Verdict::Pass);
    }
    #[test] fn mh006_fail_unclamped() {
        // Inject an unclamped velocity.
        let v = vec![10.0_f64, -10.0, 0.5, -0.3];
        assert_eq!(verdict_from_velocity_clamping(&v, 5.0), Mh006Verdict::Fail);
    }
    #[test] fn mh006_fail_zero_max() {
        let v = vec![0.0_f64];
        assert_eq!(verdict_from_velocity_clamping(&v, 0.0), Mh006Verdict::Fail);
    }

    // MH-007
    #[test] fn mh007_pass_in_bounds() {
        let children = vec![0.5_f64, 1.0, 0.0, 0.7];
        assert_eq!(verdict_from_crossover_bounds(&children, 0.0, 1.0), Mh007Verdict::Pass);
    }
    #[test] fn mh007_fail_above_upper() {
        let children = vec![0.5_f64, 1.5];
        assert_eq!(verdict_from_crossover_bounds(&children, 0.0, 1.0), Mh007Verdict::Fail);
    }
    #[test] fn mh007_fail_below_lower() {
        let children = vec![-0.1_f64, 0.5];
        assert_eq!(verdict_from_crossover_bounds(&children, 0.0, 1.0), Mh007Verdict::Fail);
    }
    #[test] fn mh007_fail_swapped_bounds() {
        let children = vec![0.5_f64];
        assert_eq!(verdict_from_crossover_bounds(&children, 1.0, 0.0), Mh007Verdict::Fail);
    }
    #[test] fn mh007_fail_empty() {
        assert_eq!(verdict_from_crossover_bounds(&[], 0.0, 1.0), Mh007Verdict::Fail);
    }
}
