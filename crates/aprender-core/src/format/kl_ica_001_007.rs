// Bundles two sister contracts in one verdict module:
//
//   `kernel-launch-budget-v1` (FALSIFY-KL-001..004)
//   `ica-v1` (FALSIFY-ICA-001..003)
//
// KL-001: per-token launch formula matches predicted count
// KL-002: per-layer decomposition sum == 12 kernels
// KL-003: more layers → strictly more launches (monotone)
// KL-004: SIMD vs scalar budget calculation bit-equal
// ICA-001: transformed shape == (n_samples, n_components)
// ICA-002: transform(X) == transform(X) deterministic
// ICA-003: every output finite for finite input

/// KL-002 per-layer kernel decomposition sum.
pub const AC_KL_PER_LAYER_KERNELS: u32 = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KlIcaVerdict {
    Pass,
    Fail,
}

/// Reference per-token launch formula: 12 * num_layers + final_kernels (3).
#[must_use]
pub fn launch_count_formula(num_layers: u32, final_kernels: u32) -> u32 {
    AC_KL_PER_LAYER_KERNELS
        .saturating_mul(num_layers)
        .saturating_add(final_kernels)
}

/// KL-001: observed launches matches formula prediction.
#[must_use]
pub fn verdict_from_launch_formula(
    num_layers: u32,
    final_kernels: u32,
    observed_launches: u32,
) -> KlIcaVerdict {
    if observed_launches == launch_count_formula(num_layers, final_kernels) {
        KlIcaVerdict::Pass
    } else {
        KlIcaVerdict::Fail
    }
}

/// KL-002: per-layer decomposition sum == 12.
#[must_use]
pub fn verdict_from_decomposition_sum(per_layer_components: &[u32]) -> KlIcaVerdict {
    if per_layer_components.is_empty() {
        return KlIcaVerdict::Fail;
    }
    let sum: u32 = per_layer_components.iter().sum();
    if sum == AC_KL_PER_LAYER_KERNELS {
        KlIcaVerdict::Pass
    } else {
        KlIcaVerdict::Fail
    }
}

/// KL-003: more layers → strictly more launches (monotone).
#[must_use]
pub fn verdict_from_launch_monotone(
    layers_a: u32,
    launches_a: u32,
    layers_b: u32,
    launches_b: u32,
) -> KlIcaVerdict {
    let monotone = match layers_a.cmp(&layers_b) {
        std::cmp::Ordering::Less => launches_a < launches_b,
        std::cmp::Ordering::Greater => launches_a > launches_b,
        std::cmp::Ordering::Equal => launches_a == launches_b,
    };
    if monotone {
        KlIcaVerdict::Pass
    } else {
        KlIcaVerdict::Fail
    }
}

/// KL-004: SIMD vs scalar budget bit-equal.
#[must_use]
pub fn verdict_from_simd_budget_parity(simd: u32, scalar: u32) -> KlIcaVerdict {
    if simd == scalar {
        KlIcaVerdict::Pass
    } else {
        KlIcaVerdict::Fail
    }
}

/// ICA-001: transformed shape == (n_samples, n_components).
#[must_use]
pub fn verdict_from_ica_output_shape(
    n_samples: usize,
    n_components: usize,
    actual_rows: usize,
    actual_cols: usize,
) -> KlIcaVerdict {
    if n_samples == 0 || n_components == 0 {
        return KlIcaVerdict::Fail;
    }
    if actual_rows == n_samples && actual_cols == n_components {
        KlIcaVerdict::Pass
    } else {
        KlIcaVerdict::Fail
    }
}

/// ICA-002: deterministic transform.
#[must_use]
pub fn verdict_from_ica_deterministic(call_a: &[f32], call_b: &[f32]) -> KlIcaVerdict {
    if call_a.is_empty() || call_a.len() != call_b.len() {
        return KlIcaVerdict::Fail;
    }
    for (x, y) in call_a.iter().zip(call_b.iter()) {
        if x.to_bits() != y.to_bits() {
            return KlIcaVerdict::Fail;
        }
    }
    KlIcaVerdict::Pass
}

/// ICA-003: every output finite.
#[must_use]
pub fn verdict_from_ica_finite_output(output: &[f32]) -> KlIcaVerdict {
    if output.is_empty() {
        return KlIcaVerdict::Fail;
    }
    if output.iter().all(|x| x.is_finite()) {
        KlIcaVerdict::Pass
    } else {
        KlIcaVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_per_layer_kernels_12() {
        assert_eq!(AC_KL_PER_LAYER_KERNELS, 12);
    }

    #[test]
    fn launch_count_formula_zero_layers() {
        assert_eq!(launch_count_formula(0, 3), 3);
    }

    #[test]
    fn launch_count_formula_24_layers() {
        // 24 * 12 + 3 = 291
        assert_eq!(launch_count_formula(24, 3), 291);
    }

    // -----------------------------------------------------------------
    // Section 2: KL-001 launch formula.
    // -----------------------------------------------------------------
    #[test]
    fn fkl001_pass_24_layers() {
        let v = verdict_from_launch_formula(24, 3, 291);
        assert_eq!(v, KlIcaVerdict::Pass);
    }

    #[test]
    fn fkl001_fail_off_by_one() {
        let v = verdict_from_launch_formula(24, 3, 292);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    #[test]
    fn fkl001_pass_zero_layers() {
        let v = verdict_from_launch_formula(0, 3, 3);
        assert_eq!(v, KlIcaVerdict::Pass);
    }

    // -----------------------------------------------------------------
    // Section 3: KL-002 decomposition.
    // -----------------------------------------------------------------
    #[test]
    fn fkl002_pass_canonical_decomposition() {
        // 4 attention + 4 ffn + 2 norms + 2 misc = 12
        let v = verdict_from_decomposition_sum(&[4, 4, 2, 2]);
        assert_eq!(v, KlIcaVerdict::Pass);
    }

    #[test]
    fn fkl002_fail_missing_kernel() {
        let v = verdict_from_decomposition_sum(&[4, 4, 2, 1]);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    #[test]
    fn fkl002_fail_extra_kernel() {
        let v = verdict_from_decomposition_sum(&[4, 4, 2, 3]);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    #[test]
    fn fkl002_fail_empty() {
        let v = verdict_from_decomposition_sum(&[]);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: KL-003 monotonicity.
    // -----------------------------------------------------------------
    #[test]
    fn fkl003_pass_more_layers_more_launches() {
        let v = verdict_from_launch_monotone(12, 147, 24, 291);
        assert_eq!(v, KlIcaVerdict::Pass);
    }

    #[test]
    fn fkl003_pass_same_layers_same_launches() {
        let v = verdict_from_launch_monotone(24, 291, 24, 291);
        assert_eq!(v, KlIcaVerdict::Pass);
    }

    #[test]
    fn fkl003_fail_more_layers_fewer_launches() {
        let v = verdict_from_launch_monotone(12, 200, 24, 100);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    #[test]
    fn fkl003_fail_same_layers_different_launches() {
        let v = verdict_from_launch_monotone(24, 291, 24, 290);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: KL-004 + ICA-001.
    // -----------------------------------------------------------------
    #[test]
    fn fkl004_pass_simd_matches_scalar() {
        let v = verdict_from_simd_budget_parity(291, 291);
        assert_eq!(v, KlIcaVerdict::Pass);
    }

    #[test]
    fn fkl004_fail_simd_drift() {
        let v = verdict_from_simd_budget_parity(291, 290);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    #[test]
    fn fica001_pass_canonical_shape() {
        let v = verdict_from_ica_output_shape(100, 5, 100, 5);
        assert_eq!(v, KlIcaVerdict::Pass);
    }

    #[test]
    fn fica001_fail_wrong_n_components() {
        let v = verdict_from_ica_output_shape(100, 5, 100, 10);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    #[test]
    fn fica001_fail_zero_components() {
        let v = verdict_from_ica_output_shape(100, 0, 100, 0);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: ICA-002 + ICA-003.
    // -----------------------------------------------------------------
    #[test]
    fn fica002_pass_bit_identical() {
        let a = vec![1.0_f32, 2.5, -3.0];
        let b = a.clone();
        let v = verdict_from_ica_deterministic(&a, &b);
        assert_eq!(v, KlIcaVerdict::Pass);
    }

    #[test]
    fn fica002_fail_drift() {
        let a = vec![1.0_f32, 2.5];
        let bumped = f32::from_bits(2.5_f32.to_bits() + 1);
        let b = vec![1.0_f32, bumped];
        let v = verdict_from_ica_deterministic(&a, &b);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    #[test]
    fn fica003_pass_finite() {
        let v = verdict_from_ica_finite_output(&[1.0, -2.0, 3.0]);
        assert_eq!(v, KlIcaVerdict::Pass);
    }

    #[test]
    fn fica003_fail_nan() {
        let v = verdict_from_ica_finite_output(&[1.0, f32::NAN]);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    #[test]
    fn fica003_fail_infinity() {
        let v = verdict_from_ica_finite_output(&[1.0, f32::INFINITY]);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    #[test]
    fn fica003_fail_empty() {
        let v = verdict_from_ica_finite_output(&[]);
        assert_eq!(v, KlIcaVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation surveys + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_kl003_layer_band() {
        for la in [0_u32, 1, 12, 24, 50] {
            for lb in [0_u32, 1, 12, 24, 50] {
                let launches_a = launch_count_formula(la, 3);
                let launches_b = launch_count_formula(lb, 3);
                let v = verdict_from_launch_monotone(la, launches_a, lb, launches_b);
                assert_eq!(v, KlIcaVerdict::Pass, "(la={la}, lb={lb})");
            }
        }
    }

    #[test]
    fn realistic_healthy_passes_all_7() {
        let v1 = verdict_from_launch_formula(24, 3, 291);
        let v2 = verdict_from_decomposition_sum(&[4, 4, 2, 2]);
        let v3 = verdict_from_launch_monotone(12, 147, 24, 291);
        let v4 = verdict_from_simd_budget_parity(291, 291);
        let v5 = verdict_from_ica_output_shape(1000, 5, 1000, 5);
        let a = vec![1.0_f32, 2.0];
        let v6 = verdict_from_ica_deterministic(&a, &a);
        let v7 = verdict_from_ica_finite_output(&a);
        for v in [v1, v2, v3, v4, v5, v6, v7] {
            assert_eq!(v, KlIcaVerdict::Pass);
        }
    }

    #[test]
    fn realistic_pre_fix_all_7_failures() {
        let v1 = verdict_from_launch_formula(24, 3, 100); // wrong constant
        let v2 = verdict_from_decomposition_sum(&[4, 4, 2, 1]); // missing kernel
        let v3 = verdict_from_launch_monotone(12, 200, 24, 100); // non-monotone
        let v4 = verdict_from_simd_budget_parity(291, 290); // SIMD drift
        let v5 = verdict_from_ica_output_shape(1000, 5, 1000, 10); // wrong dim
        let bumped = f32::from_bits(2.5_f32.to_bits() + 1);
        let a = vec![1.0_f32, 2.5];
        let b = vec![1.0_f32, bumped];
        let v6 = verdict_from_ica_deterministic(&a, &b); // random state leak
        let v7 = verdict_from_ica_finite_output(&[1.0, f32::NAN]); // NaN leak
        for v in [v1, v2, v3, v4, v5, v6, v7] {
            assert_eq!(v, KlIcaVerdict::Fail);
        }
    }
}
