// Bundles three sister contracts in one verdict module:
//
//   `glm-v1` (FALSIFY-GLM-001..004)
//   `gnn-v1` (FALSIFY-GNN-001..006)
//   `learned-position-embedding-v1` (FALSIFY-POS-001..003)
//
// GLM-001: link round-trip g(g^{-1}(eta)) ≈ eta within tolerance
// GLM-002: predicted mean lies in valid range per family
// GLM-003: IRLS deviance monotone non-increasing
// GLM-004: predictions finite for bounded input
// GNN-001: GCN preserves node count
// GNN-002: message-passing preserves node count
// GNN-003: global mean-pool produces finite output
// GNN-004: global max-pool ≤ per-feature max of nodes
// GNN-005: pooling preserves feature dimension
// GNN-006: GCN output finite for finite input
// POS-001: pos ≥ max_positions returns Err (no silent truncation)
// POS-002: PE(pos) deterministic across calls
// POS-003: PE(pos).len() == d_model

/// GLM-001 link round-trip tolerance.
pub const AC_GLM_LINK_TOLERANCE: f32 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmGnnPosVerdict {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmFamily {
    Poisson,
    Gamma,
    Binomial,
    Gaussian,
}

// ----------------------------------------------------------------
// GLM-001..004
// ----------------------------------------------------------------

#[must_use]
pub fn verdict_from_glm_link_roundtrip(eta: f32, roundtrip: f32) -> GlmGnnPosVerdict {
    if !eta.is_finite() || !roundtrip.is_finite() {
        return GlmGnnPosVerdict::Fail;
    }
    if (eta - roundtrip).abs() <= AC_GLM_LINK_TOLERANCE {
        GlmGnnPosVerdict::Pass
    } else {
        GlmGnnPosVerdict::Fail
    }
}

#[must_use]
pub fn verdict_from_glm_mean_in_range(family: GlmFamily, mu: f32) -> GlmGnnPosVerdict {
    if !mu.is_finite() {
        return GlmGnnPosVerdict::Fail;
    }
    match family {
        GlmFamily::Poisson | GlmFamily::Gamma => {
            if mu > 0.0 {
                GlmGnnPosVerdict::Pass
            } else {
                GlmGnnPosVerdict::Fail
            }
        }
        GlmFamily::Binomial => {
            if mu > 0.0 && mu < 1.0 {
                GlmGnnPosVerdict::Pass
            } else {
                GlmGnnPosVerdict::Fail
            }
        }
        GlmFamily::Gaussian => GlmGnnPosVerdict::Pass, // unbounded
    }
}

#[must_use]
pub fn verdict_from_glm_irls_monotone(deviance: &[f32]) -> GlmGnnPosVerdict {
    if deviance.len() < 2 {
        return GlmGnnPosVerdict::Fail;
    }
    for window in deviance.windows(2) {
        if !window[0].is_finite() || !window[1].is_finite() {
            return GlmGnnPosVerdict::Fail;
        }
        if window[1] > window[0] {
            return GlmGnnPosVerdict::Fail;
        }
    }
    GlmGnnPosVerdict::Pass
}

#[must_use]
pub fn verdict_from_glm_finite_predictions(predictions: &[f32]) -> GlmGnnPosVerdict {
    if predictions.is_empty() {
        return GlmGnnPosVerdict::Fail;
    }
    if predictions.iter().all(|x| x.is_finite()) {
        GlmGnnPosVerdict::Pass
    } else {
        GlmGnnPosVerdict::Fail
    }
}

// ----------------------------------------------------------------
// GNN-001..006
// ----------------------------------------------------------------

#[must_use]
pub fn verdict_from_gnn_node_count(input_nodes: usize, output_nodes: usize) -> GlmGnnPosVerdict {
    if input_nodes == 0 {
        return GlmGnnPosVerdict::Fail;
    }
    if input_nodes == output_nodes {
        GlmGnnPosVerdict::Pass
    } else {
        GlmGnnPosVerdict::Fail
    }
}

#[must_use]
pub fn verdict_from_gnn_finite(features: &[f32]) -> GlmGnnPosVerdict {
    if features.is_empty() {
        return GlmGnnPosVerdict::Fail;
    }
    if features.iter().all(|x| x.is_finite()) {
        GlmGnnPosVerdict::Pass
    } else {
        GlmGnnPosVerdict::Fail
    }
}

#[must_use]
pub fn verdict_from_gnn_max_pool_bound(
    pooled: &[f32],
    per_feature_max: &[f32],
) -> GlmGnnPosVerdict {
    if pooled.is_empty() || pooled.len() != per_feature_max.len() {
        return GlmGnnPosVerdict::Fail;
    }
    for (p, m) in pooled.iter().zip(per_feature_max.iter()) {
        if !p.is_finite() || !m.is_finite() {
            return GlmGnnPosVerdict::Fail;
        }
        if *p > *m {
            return GlmGnnPosVerdict::Fail;
        }
    }
    GlmGnnPosVerdict::Pass
}

#[must_use]
pub fn verdict_from_gnn_pool_dim(input_dim: usize, output_dim: usize) -> GlmGnnPosVerdict {
    if input_dim == 0 {
        return GlmGnnPosVerdict::Fail;
    }
    if input_dim == output_dim {
        GlmGnnPosVerdict::Pass
    } else {
        GlmGnnPosVerdict::Fail
    }
}

// ----------------------------------------------------------------
// POS-001..003
// ----------------------------------------------------------------

#[must_use]
pub fn verdict_from_pos_oob(
    pos: usize,
    max_positions: usize,
    returned_err: bool,
) -> GlmGnnPosVerdict {
    if pos < max_positions {
        // In-range: must NOT return Err
        if !returned_err {
            GlmGnnPosVerdict::Pass
        } else {
            GlmGnnPosVerdict::Fail
        }
    } else {
        // Out-of-range: must return Err
        if returned_err {
            GlmGnnPosVerdict::Pass
        } else {
            GlmGnnPosVerdict::Fail
        }
    }
}

#[must_use]
pub fn verdict_from_pos_deterministic(call_a: &[f32], call_b: &[f32]) -> GlmGnnPosVerdict {
    if call_a.is_empty() || call_a.len() != call_b.len() {
        return GlmGnnPosVerdict::Fail;
    }
    for (x, y) in call_a.iter().zip(call_b.iter()) {
        if x.to_bits() != y.to_bits() {
            return GlmGnnPosVerdict::Fail;
        }
    }
    GlmGnnPosVerdict::Pass
}

#[must_use]
pub fn verdict_from_pos_output_dim(actual_len: usize, d_model: usize) -> GlmGnnPosVerdict {
    if d_model == 0 {
        return GlmGnnPosVerdict::Fail;
    }
    if actual_len == d_model {
        GlmGnnPosVerdict::Pass
    } else {
        GlmGnnPosVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_glm_link_tolerance() {
        assert_eq!(AC_GLM_LINK_TOLERANCE, 1e-4);
    }

    // -----------------------------------------------------------------
    // Section 2: GLM-001..004.
    // -----------------------------------------------------------------
    #[test]
    fn fglm001_pass_within_tolerance() {
        let v = verdict_from_glm_link_roundtrip(2.5, 2.5001);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fglm001_fail_far_drift() {
        let v = verdict_from_glm_link_roundtrip(2.5, 5.0);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fglm002_pass_poisson_positive() {
        let v = verdict_from_glm_mean_in_range(GlmFamily::Poisson, 2.5);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fglm002_fail_poisson_zero() {
        let v = verdict_from_glm_mean_in_range(GlmFamily::Poisson, 0.0);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fglm002_pass_binomial_in_range() {
        let v = verdict_from_glm_mean_in_range(GlmFamily::Binomial, 0.5);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fglm002_fail_binomial_at_one() {
        let v = verdict_from_glm_mean_in_range(GlmFamily::Binomial, 1.0);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fglm002_pass_gaussian_unbounded() {
        let v = verdict_from_glm_mean_in_range(GlmFamily::Gaussian, -100.0);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fglm003_pass_monotone_decrease() {
        let v = verdict_from_glm_irls_monotone(&[10.0, 5.0, 2.0, 1.5]);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fglm003_pass_plateau() {
        let v = verdict_from_glm_irls_monotone(&[5.0, 5.0, 5.0]);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fglm003_fail_increased() {
        let v = verdict_from_glm_irls_monotone(&[5.0, 10.0]);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fglm004_pass_finite() {
        let v = verdict_from_glm_finite_predictions(&[1.0, -2.0, 3.0]);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fglm004_fail_inf() {
        let v = verdict_from_glm_finite_predictions(&[1.0, f32::INFINITY]);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: GNN-001..006.
    // -----------------------------------------------------------------
    #[test]
    fn fgnn001_pass_node_count_preserved() {
        let v = verdict_from_gnn_node_count(100, 100);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fgnn001_fail_dropped_node() {
        let v = verdict_from_gnn_node_count(100, 99);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fgnn003_pass_finite_pool() {
        let v = verdict_from_gnn_finite(&[1.0, 2.0, 3.0]);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fgnn003_fail_nan_pool() {
        let v = verdict_from_gnn_finite(&[1.0, f32::NAN]);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fgnn004_pass_below_max() {
        let v = verdict_from_gnn_max_pool_bound(&[5.0, 3.0], &[5.0, 4.0]);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fgnn004_fail_exceeds_max() {
        // pooled = 6 but per-feature max = 5 — impossible by definition.
        let v = verdict_from_gnn_max_pool_bound(&[6.0], &[5.0]);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fgnn005_pass_dim_preserved() {
        let v = verdict_from_gnn_pool_dim(64, 64);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fgnn005_fail_dim_changed() {
        let v = verdict_from_gnn_pool_dim(64, 32);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fgnn006_pass_finite_gcn() {
        let v = verdict_from_gnn_finite(&[1.0, 2.0]);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    // -----------------------------------------------------------------
    // Section 4: POS-001..003.
    // -----------------------------------------------------------------
    #[test]
    fn fpos001_pass_in_range_no_err() {
        let v = verdict_from_pos_oob(50, 100, false);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fpos001_pass_oob_returns_err() {
        let v = verdict_from_pos_oob(150, 100, true);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fpos001_fail_oob_silent() {
        // The regression class — silent truncation
        let v = verdict_from_pos_oob(150, 100, false);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fpos001_fail_at_boundary() {
        // pos == max_positions is OOB (max excluded)
        let v = verdict_from_pos_oob(100, 100, true);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fpos002_pass_bit_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = a.clone();
        let v = verdict_from_pos_deterministic(&a, &b);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fpos002_fail_drift() {
        let a = vec![1.0_f32];
        let bumped = f32::from_bits(1.0_f32.to_bits() + 1);
        let b = vec![bumped];
        let v = verdict_from_pos_deterministic(&a, &b);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fpos003_pass_correct_dim() {
        let v = verdict_from_pos_output_dim(768, 768);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
    }

    #[test]
    fn fpos003_fail_wrong_dim() {
        let v = verdict_from_pos_output_dim(512, 768);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    #[test]
    fn fpos003_fail_zero_d_model() {
        let v = verdict_from_pos_output_dim(0, 0);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_pos_oob_band() {
        let max_pos = 100_usize;
        for pos in [0_usize, 50, 99, 100, 101, 200] {
            let in_range = pos < max_pos;
            // Both branches: in-range no-err, OOB returns err
            let v = verdict_from_pos_oob(pos, max_pos, !in_range);
            assert_eq!(v, GlmGnnPosVerdict::Pass, "pos={pos}");
        }
    }

    #[test]
    fn mutation_survey_glm_irls_monotone_sweep() {
        // Strictly decreasing
        let v = verdict_from_glm_irls_monotone(&[10.0, 9.0, 8.0, 7.0]);
        assert_eq!(v, GlmGnnPosVerdict::Pass);
        // Strictly increasing
        let v = verdict_from_glm_irls_monotone(&[1.0, 2.0, 3.0]);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
        // Mixed: decrease then increase
        let v = verdict_from_glm_irls_monotone(&[5.0, 3.0, 7.0]);
        assert_eq!(v, GlmGnnPosVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_13() {
        let v1 = verdict_from_glm_link_roundtrip(2.5, 2.5);
        let v2 = verdict_from_glm_mean_in_range(GlmFamily::Poisson, 3.0);
        let v3 = verdict_from_glm_irls_monotone(&[10.0, 5.0, 2.0]);
        let v4 = verdict_from_glm_finite_predictions(&[1.0, 2.0]);
        let v5 = verdict_from_gnn_node_count(100, 100);
        let v6 = verdict_from_gnn_node_count(100, 100); // message-pass same shape
        let v7 = verdict_from_gnn_finite(&[1.0, 2.0]);
        let v8 = verdict_from_gnn_max_pool_bound(&[3.0], &[3.0]);
        let v9 = verdict_from_gnn_pool_dim(64, 64);
        let v10 = verdict_from_gnn_finite(&[1.0]);
        let v11 = verdict_from_pos_oob(50, 100, false);
        let v12 = verdict_from_pos_deterministic(&[1.0_f32], &[1.0_f32]);
        let v13 = verdict_from_pos_output_dim(768, 768);
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13] {
            assert_eq!(v, GlmGnnPosVerdict::Pass);
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Pre-fix regressions.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_pre_fix_all_13_failures() {
        let v1 = verdict_from_glm_link_roundtrip(2.5, 5.0);
        let v2 = verdict_from_glm_mean_in_range(GlmFamily::Binomial, 1.5);
        let v3 = verdict_from_glm_irls_monotone(&[5.0, 10.0]);
        let v4 = verdict_from_glm_finite_predictions(&[1.0, f32::NAN]);
        let v5 = verdict_from_gnn_node_count(100, 90);
        let v6 = verdict_from_gnn_node_count(100, 110);
        let v7 = verdict_from_gnn_finite(&[1.0, f32::NAN]);
        let v8 = verdict_from_gnn_max_pool_bound(&[10.0], &[5.0]);
        let v9 = verdict_from_gnn_pool_dim(64, 32);
        let v10 = verdict_from_gnn_finite(&[1.0, f32::INFINITY]);
        let v11 = verdict_from_pos_oob(150, 100, false); // silent OOB
        let bumped = f32::from_bits(1.0_f32.to_bits() + 1);
        let v12 = verdict_from_pos_deterministic(&[1.0_f32], &[bumped]);
        let v13 = verdict_from_pos_output_dim(512, 768);
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13] {
            assert_eq!(v, GlmGnnPosVerdict::Fail);
        }
    }
}
