// `lbfgs-kernel-v1` algorithm-level PARTIAL discharge for
// FALSIFY-LB-001..006.
//
// Contract: `contracts/lbfgs-kernel-v1.yaml`.
//
// LB-001: descent direction — g^T * direction < 0 for non-zero gradient
// LB-002: curvature condition — y^T * s > 0 for accepted (s, y) pairs
// LB-003: history length bounded by m
// LB-004: objective decrease on Rosenbrock — f(x_{k+1}) < f(x_k)
// LB-005: SIMD vs scalar within 8 ULP
// LB-006: empty history → direction == -g (steepest descent)

/// LB-005 SIMD vs scalar tolerance (8 ULP * EPSILON).
pub const AC_LB_SIMD_ULP_BUDGET: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbVerdict {
    Pass,
    Fail,
}

/// Reference dot product (returns NaN on length mismatch / non-finite).
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return f32::NAN;
    }
    let mut sum = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return f32::NAN;
        }
        sum += x * y;
    }
    sum
}

/// LB-001: g^T * direction < 0 strictly (descent direction).
#[must_use]
pub fn verdict_from_descent_direction(grad: &[f32], direction: &[f32]) -> LbVerdict {
    let prod = dot(grad, direction);
    if !prod.is_finite() {
        return LbVerdict::Fail;
    }
    // Skip if grad is effectively zero (gate doesn't apply).
    let g_norm_sq = dot(grad, grad);
    if g_norm_sq <= 0.0 || !g_norm_sq.is_finite() {
        return LbVerdict::Fail;
    }
    if prod < 0.0 {
        LbVerdict::Pass
    } else {
        LbVerdict::Fail
    }
}

/// LB-002: y^T * s > 0 for every accepted (s, y) pair.
#[must_use]
pub fn verdict_from_curvature(s_y_pairs: &[(&[f32], &[f32])]) -> LbVerdict {
    if s_y_pairs.is_empty() {
        return LbVerdict::Fail;
    }
    for (s, y) in s_y_pairs {
        let p = dot(s, y);
        if !p.is_finite() || p <= 0.0 {
            return LbVerdict::Fail;
        }
    }
    LbVerdict::Pass
}

/// LB-003: history length ≤ m.
#[must_use]
pub fn verdict_from_history_bound(history_len: usize, m: usize) -> LbVerdict {
    if m == 0 {
        return LbVerdict::Fail;
    }
    if history_len <= m {
        LbVerdict::Pass
    } else {
        LbVerdict::Fail
    }
}

/// LB-004: f(x_{k+1}) < f(x_k) (objective strictly decreased).
#[must_use]
pub fn verdict_from_objective_decrease(f_k: f32, f_k_plus_1: f32) -> LbVerdict {
    if !f_k.is_finite() || !f_k_plus_1.is_finite() {
        return LbVerdict::Fail;
    }
    if f_k_plus_1 < f_k {
        LbVerdict::Pass
    } else {
        LbVerdict::Fail
    }
}

/// LB-005: SIMD vs scalar within `AC_LB_SIMD_ULP_BUDGET * EPSILON`.
#[must_use]
pub fn verdict_from_simd_scalar_parity(
    simd_direction: &[f32],
    scalar_direction: &[f32],
) -> LbVerdict {
    if simd_direction.is_empty() || simd_direction.len() != scalar_direction.len() {
        return LbVerdict::Fail;
    }
    let bound = (AC_LB_SIMD_ULP_BUDGET as f32) * f32::EPSILON;
    for (a, b) in simd_direction.iter().zip(scalar_direction.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return LbVerdict::Fail;
        }
        let scale = a.abs().max(b.abs()).max(1.0);
        if (a - b).abs() > bound * scale {
            return LbVerdict::Fail;
        }
    }
    LbVerdict::Pass
}

/// LB-006: empty history → direction == -gradient (steepest descent).
#[must_use]
pub fn verdict_from_empty_history_steepest(grad: &[f32], direction: &[f32]) -> LbVerdict {
    if grad.is_empty() || grad.len() != direction.len() {
        return LbVerdict::Fail;
    }
    for (g, d) in grad.iter().zip(direction.iter()) {
        if !g.is_finite() || !d.is_finite() {
            return LbVerdict::Fail;
        }
        if (-g).to_bits() != d.to_bits() {
            return LbVerdict::Fail;
        }
    }
    LbVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_simd_ulp_budget_8() {
        assert_eq!(AC_LB_SIMD_ULP_BUDGET, 8);
    }

    // -----------------------------------------------------------------
    // Section 2: LB-001 descent direction.
    // -----------------------------------------------------------------
    #[test]
    fn flb001_pass_descent() {
        let g = vec![1.0_f32, 2.0];
        let d = vec![-1.0_f32, -2.0];
        let v = verdict_from_descent_direction(&g, &d);
        assert_eq!(v, LbVerdict::Pass);
    }

    #[test]
    fn flb001_fail_ascent() {
        let g = vec![1.0_f32, 2.0];
        let d = vec![1.0_f32, 2.0]; // same direction as grad
        let v = verdict_from_descent_direction(&g, &d);
        assert_eq!(v, LbVerdict::Fail);
    }

    #[test]
    fn flb001_fail_orthogonal() {
        let g = vec![1.0_f32, 0.0];
        let d = vec![0.0_f32, 1.0];
        let v = verdict_from_descent_direction(&g, &d);
        assert_eq!(v, LbVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: LB-002 curvature.
    // -----------------------------------------------------------------
    #[test]
    fn flb002_pass_positive_inner() {
        let s = vec![1.0_f32];
        let y = vec![2.0_f32];
        let pairs: &[(&[f32], &[f32])] = &[(&s, &y)];
        let v = verdict_from_curvature(pairs);
        assert_eq!(v, LbVerdict::Pass);
    }

    #[test]
    fn flb002_fail_zero_inner() {
        let s = vec![1.0_f32, 1.0];
        let y = vec![1.0_f32, -1.0]; // dot = 0
        let pairs: &[(&[f32], &[f32])] = &[(&s, &y)];
        let v = verdict_from_curvature(pairs);
        assert_eq!(v, LbVerdict::Fail);
    }

    #[test]
    fn flb002_fail_negative_inner() {
        let s = vec![1.0_f32];
        let y = vec![-1.0_f32];
        let pairs: &[(&[f32], &[f32])] = &[(&s, &y)];
        let v = verdict_from_curvature(pairs);
        assert_eq!(v, LbVerdict::Fail);
    }

    #[test]
    fn flb002_fail_empty() {
        let pairs: &[(&[f32], &[f32])] = &[];
        let v = verdict_from_curvature(pairs);
        assert_eq!(v, LbVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: LB-003 history bound.
    // -----------------------------------------------------------------
    #[test]
    fn flb003_pass_at_capacity() {
        let v = verdict_from_history_bound(5, 5);
        assert_eq!(v, LbVerdict::Pass);
    }

    #[test]
    fn flb003_pass_under_capacity() {
        let v = verdict_from_history_bound(3, 5);
        assert_eq!(v, LbVerdict::Pass);
    }

    #[test]
    fn flb003_fail_over_capacity() {
        let v = verdict_from_history_bound(6, 5);
        assert_eq!(v, LbVerdict::Fail);
    }

    #[test]
    fn flb003_fail_zero_m() {
        let v = verdict_from_history_bound(0, 0);
        assert_eq!(v, LbVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: LB-004..005.
    // -----------------------------------------------------------------
    #[test]
    fn flb004_pass_decreased() {
        let v = verdict_from_objective_decrease(10.0, 5.0);
        assert_eq!(v, LbVerdict::Pass);
    }

    #[test]
    fn flb004_fail_increased() {
        let v = verdict_from_objective_decrease(5.0, 10.0);
        assert_eq!(v, LbVerdict::Fail);
    }

    #[test]
    fn flb004_fail_unchanged() {
        let v = verdict_from_objective_decrease(5.0, 5.0);
        assert_eq!(v, LbVerdict::Fail);
    }

    #[test]
    fn flb005_pass_within_ulp_budget() {
        let simd = vec![1.0_f32, 2.0, 3.0];
        let scalar = vec![1.0_f32, 2.0, 3.0];
        let v = verdict_from_simd_scalar_parity(&simd, &scalar);
        assert_eq!(v, LbVerdict::Pass);
    }

    #[test]
    fn flb005_fail_far_drift() {
        let simd = vec![1.0_f32];
        let scalar = vec![1.5_f32];
        let v = verdict_from_simd_scalar_parity(&simd, &scalar);
        assert_eq!(v, LbVerdict::Fail);
    }

    #[test]
    fn flb005_pass_8_ulp() {
        // 8 ULPs at 1.0 = 8 * f32::EPSILON ~= 9.5e-7
        let simd = vec![1.0_f32];
        let bumped = f32::from_bits(1.0_f32.to_bits() + 7);
        let scalar = vec![bumped];
        let v = verdict_from_simd_scalar_parity(&simd, &scalar);
        assert_eq!(v, LbVerdict::Pass);
    }

    // -----------------------------------------------------------------
    // Section 6: LB-006 steepest descent.
    // -----------------------------------------------------------------
    #[test]
    fn flb006_pass_neg_gradient() {
        let g = vec![1.0_f32, -2.0, 3.0];
        let d = vec![-1.0_f32, 2.0, -3.0];
        let v = verdict_from_empty_history_steepest(&g, &d);
        assert_eq!(v, LbVerdict::Pass);
    }

    #[test]
    fn flb006_fail_not_negated() {
        let g = vec![1.0_f32, -2.0];
        let d = vec![1.0_f32, -2.0]; // same as g, not -g
        let v = verdict_from_empty_history_steepest(&g, &d);
        assert_eq!(v, LbVerdict::Fail);
    }

    #[test]
    fn flb006_fail_one_ulp_drift() {
        let g = vec![1.0_f32];
        let bumped = f32::from_bits((-1.0_f32).to_bits() + 1);
        let d = vec![bumped];
        let v = verdict_from_empty_history_steepest(&g, &d);
        assert_eq!(v, LbVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation surveys + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_lb003_history_band() {
        for m in [1_usize, 5, 10, 100] {
            for h in 0..=2 * m {
                let v = verdict_from_history_bound(h, m);
                let want = if h <= m {
                    LbVerdict::Pass
                } else {
                    LbVerdict::Fail
                };
                assert_eq!(v, want, "m={m} h={h}");
            }
        }
    }

    #[test]
    fn realistic_healthy_passes_all_6() {
        let g = vec![1.0_f32, -2.0];
        let d_descent = vec![-1.0_f32, 2.0];
        let v1 = verdict_from_descent_direction(&g, &d_descent);
        let s = vec![1.0_f32, 1.0];
        let y = vec![2.0_f32, 3.0];
        let v2 = verdict_from_curvature(&[(&s, &y)]);
        let v3 = verdict_from_history_bound(3, 5);
        let v4 = verdict_from_objective_decrease(100.0, 50.0);
        let simd = vec![1.0_f32, 2.0];
        let scalar = vec![1.0_f32, 2.0];
        let v5 = verdict_from_simd_scalar_parity(&simd, &scalar);
        let v6 = verdict_from_empty_history_steepest(&g, &d_descent);
        for v in [v1, v2, v3, v4, v5, v6] {
            assert_eq!(v, LbVerdict::Pass);
        }
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        let g = vec![1.0_f32];
        let d_ascent = vec![1.0_f32]; // wrong sign (LB-001 sign error)
        let v1 = verdict_from_descent_direction(&g, &d_ascent);
        let s = vec![1.0_f32];
        let y = vec![-2.0_f32]; // negative inner (curvature violated)
        let v2 = verdict_from_curvature(&[(&s, &y)]);
        let v3 = verdict_from_history_bound(10, 5); // overflow
        let v4 = verdict_from_objective_decrease(50.0, 100.0); // ascended
        let simd = vec![1.0_f32];
        let scalar = vec![1.5_f32]; // ULP budget exceeded
        let v5 = verdict_from_simd_scalar_parity(&simd, &scalar);
        let v6 = verdict_from_empty_history_steepest(&[1.0], &[1.0]); // not -g
        for v in [v1, v2, v3, v4, v5, v6] {
            assert_eq!(v, LbVerdict::Fail);
        }
    }
}
