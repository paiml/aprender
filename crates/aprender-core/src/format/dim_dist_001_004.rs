// Bundles two sister contracts in one verdict module:
//
//   `dimension-independent-kernels-v1` (FALSIFY-DIMENSION..._001..002)
//   `distributed-training-v1` (FALSIFY-DISTRIBUTED..._001..002)
//
// DIM-001: ∀ M,K,N: dim_independent_output ≈ specialized_output within ε
// DIM-002: kernel binary loaded once, M/K/N passed as launch params
// DIST-001: ∀ rank: params_after_step identical across workers
// DIST-002: loss(distributed, N) ≈ loss(single, N) within ε

/// DIM-001 numeric tolerance.
pub const AC_DIM_OUTPUT_EPSILON: f32 = 1e-5;
/// DIST-002 loss tolerance.
pub const AC_DIST_LOSS_EPSILON: f32 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimDistVerdict {
    Pass,
    Fail,
}

/// DIM-001: dim-independent output matches specialized output within ε.
#[must_use]
pub fn verdict_from_dim_independence_parity(
    dim_independent: &[f32],
    specialized: &[f32],
) -> DimDistVerdict {
    if dim_independent.is_empty() || dim_independent.len() != specialized.len() {
        return DimDistVerdict::Fail;
    }
    for (a, b) in dim_independent.iter().zip(specialized.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return DimDistVerdict::Fail;
        }
        if (a - b).abs() > AC_DIM_OUTPUT_EPSILON {
            return DimDistVerdict::Fail;
        }
    }
    DimDistVerdict::Pass
}

/// DIM-002: kernel loaded once, not recompiled per-launch.
#[must_use]
pub fn verdict_from_no_recompile(
    kernel_load_count: u32,
    launch_count: u32,
) -> DimDistVerdict {
    if launch_count == 0 {
        return DimDistVerdict::Fail;
    }
    if kernel_load_count == 1 {
        DimDistVerdict::Pass
    } else {
        DimDistVerdict::Fail
    }
}

/// DIST-001: params identical across all workers after sync.
#[must_use]
pub fn verdict_from_gradient_sync(
    rank0_params: &[f32],
    other_ranks_params: &[&[f32]],
) -> DimDistVerdict {
    if rank0_params.is_empty() || other_ranks_params.is_empty() {
        return DimDistVerdict::Fail;
    }
    for &other in other_ranks_params {
        if other.len() != rank0_params.len() {
            return DimDistVerdict::Fail;
        }
        for (a, b) in rank0_params.iter().zip(other.iter()) {
            if a.to_bits() != b.to_bits() {
                return DimDistVerdict::Fail;
            }
        }
    }
    DimDistVerdict::Pass
}

/// DIST-002: distributed loss equivalent to single-worker loss within ε.
#[must_use]
pub fn verdict_from_loss_equivalence(
    distributed_loss: f32,
    single_loss: f32,
) -> DimDistVerdict {
    if !distributed_loss.is_finite() || !single_loss.is_finite() {
        return DimDistVerdict::Fail;
    }
    if (distributed_loss - single_loss).abs() <= AC_DIST_LOSS_EPSILON {
        DimDistVerdict::Pass
    } else {
        DimDistVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_DIM_OUTPUT_EPSILON, 1e-5);
        assert_eq!(AC_DIST_LOSS_EPSILON, 1e-4);
    }

    // -----------------------------------------------------------------
    // Section 2: DIM-001..002.
    // -----------------------------------------------------------------
    #[test]
    fn fdim001_pass_within_epsilon() {
        let dim_indep = vec![1.0_f32, 2.0, 3.0];
        let specialized = vec![1.000001_f32, 1.999999, 3.000005];
        let v = verdict_from_dim_independence_parity(&dim_indep, &specialized);
        assert_eq!(v, DimDistVerdict::Pass);
    }

    #[test]
    fn fdim001_fail_drift() {
        let dim_indep = vec![1.0_f32];
        let specialized = vec![1.5_f32];
        let v = verdict_from_dim_independence_parity(&dim_indep, &specialized);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    #[test]
    fn fdim001_fail_length_mismatch() {
        let dim_indep = vec![1.0_f32];
        let specialized = vec![1.0_f32, 2.0];
        let v = verdict_from_dim_independence_parity(&dim_indep, &specialized);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    #[test]
    fn fdim001_fail_nan() {
        let dim_indep = vec![1.0_f32, f32::NAN];
        let specialized = vec![1.0_f32, 2.0];
        let v = verdict_from_dim_independence_parity(&dim_indep, &specialized);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    #[test]
    fn fdim002_pass_load_once() {
        let v = verdict_from_no_recompile(1, 1000);
        assert_eq!(v, DimDistVerdict::Pass);
    }

    #[test]
    fn fdim002_fail_recompile_per_launch() {
        let v = verdict_from_no_recompile(1000, 1000);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    #[test]
    fn fdim002_fail_zero_launches() {
        let v = verdict_from_no_recompile(1, 0);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: DIST-001..002.
    // -----------------------------------------------------------------
    #[test]
    fn fdist001_pass_4_ranks_synced() {
        let rank0 = vec![1.0_f32, 2.0, 3.0];
        let r1 = rank0.clone();
        let r2 = rank0.clone();
        let r3 = rank0.clone();
        let other: Vec<&[f32]> = vec![&r1, &r2, &r3];
        let v = verdict_from_gradient_sync(&rank0, &other);
        assert_eq!(v, DimDistVerdict::Pass);
    }

    #[test]
    fn fdist001_fail_one_rank_diverged() {
        let rank0 = vec![1.0_f32, 2.0];
        let r1 = vec![1.0_f32, 2.0];
        let r2 = vec![1.0_f32, 2.5]; // diverged
        let other: Vec<&[f32]> = vec![&r1, &r2];
        let v = verdict_from_gradient_sync(&rank0, &other);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    #[test]
    fn fdist001_fail_one_ulp_drift() {
        let rank0 = vec![1.0_f32];
        let bumped = f32::from_bits(1.0_f32.to_bits() + 1);
        let r1 = vec![bumped];
        let other: Vec<&[f32]> = vec![&r1];
        let v = verdict_from_gradient_sync(&rank0, &other);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    #[test]
    fn fdist001_fail_length_mismatch() {
        let rank0 = vec![1.0_f32];
        let r1 = vec![1.0_f32, 2.0];
        let other: Vec<&[f32]> = vec![&r1];
        let v = verdict_from_gradient_sync(&rank0, &other);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    #[test]
    fn fdist001_fail_empty_others() {
        let rank0 = vec![1.0_f32];
        let other: Vec<&[f32]> = vec![];
        let v = verdict_from_gradient_sync(&rank0, &other);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    #[test]
    fn fdist002_pass_within_epsilon() {
        let v = verdict_from_loss_equivalence(0.50001, 0.50000);
        assert_eq!(v, DimDistVerdict::Pass);
    }

    #[test]
    fn fdist002_fail_drift() {
        let v = verdict_from_loss_equivalence(0.5, 1.0);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    #[test]
    fn fdist002_fail_nan() {
        let v = verdict_from_loss_equivalence(f32::NAN, 0.5);
        assert_eq!(v, DimDistVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_dist_loss_band() {
        let single = 1.0_f32;
        for delta_x10000 in [0_i32, 50, 99, 100, 101, 200, 1000] {
            let delta = delta_x10000 as f32 / 10_000.0;
            let dist = single + delta;
            let v = verdict_from_loss_equivalence(dist, single);
            let want = if delta.abs() <= 1e-4 {
                DimDistVerdict::Pass
            } else {
                DimDistVerdict::Fail
            };
            assert_eq!(v, want, "delta={delta}");
        }
    }

    // -----------------------------------------------------------------
    // Section 5: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_4() {
        let v1 = verdict_from_dim_independence_parity(
            &[1.0_f32, 2.0],
            &[1.0_f32, 2.000001],
        );
        let v2 = verdict_from_no_recompile(1, 100_000);
        let rank0 = vec![1.0_f32, 2.0];
        let r1 = rank0.clone();
        let other: Vec<&[f32]> = vec![&r1];
        let v3 = verdict_from_gradient_sync(&rank0, &other);
        let v4 = verdict_from_loss_equivalence(0.75005, 0.75);
        for v in [v1, v2, v3, v4] {
            assert_eq!(v, DimDistVerdict::Pass);
        }
    }

    // -----------------------------------------------------------------
    // Section 6: Pre-fix regressions.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_all_4_failures() {
        let v1 = verdict_from_dim_independence_parity(&[1.0_f32], &[2.0]);
        let v2 = verdict_from_no_recompile(50, 100); // recompile per-launch
        let rank0 = vec![1.0_f32];
        let r1 = vec![2.0_f32]; // diverged
        let other: Vec<&[f32]> = vec![&r1];
        let v3 = verdict_from_gradient_sync(&rank0, &other);
        let v4 = verdict_from_loss_equivalence(0.5, 1.0);
        for v in [v1, v2, v3, v4] {
            assert_eq!(v, DimDistVerdict::Fail);
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Edge cases.
    // -----------------------------------------------------------------
    #[test]
    fn edge_empty_inputs_fail() {
        let v1 = verdict_from_dim_independence_parity(&[], &[]);
        let other: Vec<&[f32]> = vec![];
        let v3 = verdict_from_gradient_sync(&[], &other);
        for v in [v1, v3] {
            assert_eq!(v, DimDistVerdict::Fail);
        }
    }
}
