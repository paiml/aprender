// SHIP-TWO-001 — `absolute-position-v1` algorithm-level PARTIAL
// discharge for FALSIFY-AP-001..004 (closes 4/4 sweep).
//
// Contract: `contracts/absolute-position-v1.yaml`.
// Spec: Absolute position embeddings — learned additive positional
// encoding (Vaswani et al. 2017 Attention Is All You Need).
//
// NOTE: Module name `iso_001_004` (initial-stamp-only) disambiguates from
// AP-* prefix collisions in adamw-kernel (already bound earlier).

// ===========================================================================
// AP-001 — Shape preservation: output.shape == token_embed.shape
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ap001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_shape_preservation(
    token_embed_shape: &[u64],
    pos_embed_shape: &[u64],
    output_shape: &[u64],
) -> Ap001Verdict {
    if token_embed_shape.is_empty() || pos_embed_shape.is_empty() || output_shape.is_empty() {
        return Ap001Verdict::Fail;
    }
    // All three shapes must match (additive embedding, no broadcasting allowed).
    if token_embed_shape != pos_embed_shape { return Ap001Verdict::Fail; }
    if token_embed_shape != output_shape { return Ap001Verdict::Fail; }
    Ap001Verdict::Pass
}

// ===========================================================================
// AP-002 — Additive identity: pos_embed == 0 ⟹ output == token_embed
// ===========================================================================

pub const AC_AP_002_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ap002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_additive_identity(
    token_embed: &[f32],
    pos_embed: &[f32],
    output: &[f32],
) -> Ap002Verdict {
    if token_embed.is_empty() || pos_embed.is_empty() || output.is_empty() {
        return Ap002Verdict::Fail;
    }
    if token_embed.len() != pos_embed.len() || token_embed.len() != output.len() {
        return Ap002Verdict::Fail;
    }
    // Verify pos_embed is all-zero (precondition for the additive identity).
    for &p in pos_embed {
        if !p.is_finite() { return Ap002Verdict::Fail; }
        if p.abs() > AC_AP_002_TOLERANCE { return Ap002Verdict::Fail; }
    }
    // Verify output == token_embed within tolerance.
    for (&t, &o) in token_embed.iter().zip(output.iter()) {
        if !t.is_finite() || !o.is_finite() { return Ap002Verdict::Fail; }
        if (t - o).abs() > AC_AP_002_TOLERANCE { return Ap002Verdict::Fail; }
    }
    Ap002Verdict::Pass
}

// ===========================================================================
// AP-003 — Max position bound: all t < max_position
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ap003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_position_bound(positions: &[u64], max_position: u64) -> Ap003Verdict {
    if positions.is_empty() || max_position == 0 { return Ap003Verdict::Fail; }
    for &t in positions {
        if t >= max_position { return Ap003Verdict::Fail; }
    }
    Ap003Verdict::Pass
}

// ===========================================================================
// AP-004 — Finite output: finite inputs ⟹ finite output
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ap004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_finite_output(
    token_embed: &[f32],
    pos_embed: &[f32],
    output: &[f32],
) -> Ap004Verdict {
    if token_embed.is_empty() || pos_embed.is_empty() || output.is_empty() {
        return Ap004Verdict::Fail;
    }
    if token_embed.len() != pos_embed.len() || token_embed.len() != output.len() {
        return Ap004Verdict::Fail;
    }
    // Precondition: all inputs finite.
    if !token_embed.iter().all(|v| v.is_finite()) { return Ap004Verdict::Fail; }
    if !pos_embed.iter().all(|v| v.is_finite()) { return Ap004Verdict::Fail; }
    // Postcondition: all outputs finite.
    if !output.iter().all(|v| v.is_finite()) { return Ap004Verdict::Fail; }
    Ap004Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // AP-001 (shape preservation)
    #[test] fn ap001_pass_canonical() {
        let s = vec![16_u64, 4096];
        assert_eq!(verdict_from_shape_preservation(&s, &s, &s), Ap001Verdict::Pass);
    }
    #[test] fn ap001_fail_token_pos_mismatch() {
        let token = vec![16_u64, 4096];
        let pos = vec![16_u64, 4097];
        let out = vec![16_u64, 4096];
        assert_eq!(verdict_from_shape_preservation(&token, &pos, &out), Ap001Verdict::Fail);
    }
    #[test] fn ap001_fail_output_drift() {
        let s = vec![16_u64, 4096];
        let out = vec![16_u64, 4097];
        assert_eq!(verdict_from_shape_preservation(&s, &s, &out), Ap001Verdict::Fail);
    }
    #[test] fn ap001_fail_empty() {
        assert_eq!(verdict_from_shape_preservation(&[], &[], &[]), Ap001Verdict::Fail);
    }

    // AP-002 (additive identity)
    #[test] fn ap002_pass_canonical() {
        let token = vec![1.0_f32, 2.0, 3.0];
        let pos = vec![0.0_f32, 0.0, 0.0];
        let output = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_additive_identity(&token, &pos, &output), Ap002Verdict::Pass);
    }
    #[test] fn ap002_fail_nonzero_pos_embed() {
        // Precondition violated: pos_embed is not zero.
        let token = vec![1.0_f32];
        let pos = vec![0.5_f32];
        let output = vec![1.5_f32];
        assert_eq!(verdict_from_additive_identity(&token, &pos, &output), Ap002Verdict::Fail);
    }
    #[test] fn ap002_fail_output_drift() {
        // pos is zero but output drifts from token (the canonical regression:
        // "Replace addition with multiplication" — multiplying by 0 gives 0).
        let token = vec![1.0_f32, 2.0, 3.0];
        let pos = vec![0.0_f32, 0.0, 0.0];
        let output = vec![0.0_f32, 0.0, 0.0]; // multiplication regression
        assert_eq!(verdict_from_additive_identity(&token, &pos, &output), Ap002Verdict::Fail);
    }
    #[test] fn ap002_fail_length_mismatch() {
        let token = vec![1.0_f32];
        let pos = vec![0.0_f32, 0.0];
        let output = vec![1.0_f32];
        assert_eq!(verdict_from_additive_identity(&token, &pos, &output), Ap002Verdict::Fail);
    }
    #[test] fn ap002_fail_nan() {
        let token = vec![f32::NAN];
        let pos = vec![0.0_f32];
        let output = vec![f32::NAN];
        assert_eq!(verdict_from_additive_identity(&token, &pos, &output), Ap002Verdict::Fail);
    }

    // AP-003 (max position bound)
    #[test] fn ap003_pass_canonical() {
        // Positions [0, 1, 2, ..., 511] with max_position=512.
        let positions: Vec<u64> = (0..512).collect();
        assert_eq!(verdict_from_position_bound(&positions, 512), Ap003Verdict::Pass);
    }
    #[test] fn ap003_fail_at_max() {
        // t == max_position is OOB (strict <).
        let positions = vec![511_u64, 512];
        assert_eq!(verdict_from_position_bound(&positions, 512), Ap003Verdict::Fail);
    }
    #[test] fn ap003_fail_above_max() {
        let positions = vec![1024_u64];
        assert_eq!(verdict_from_position_bound(&positions, 512), Ap003Verdict::Fail);
    }
    #[test] fn ap003_fail_zero_max() {
        assert_eq!(verdict_from_position_bound(&[0_u64], 0), Ap003Verdict::Fail);
    }
    #[test] fn ap003_fail_empty() {
        assert_eq!(verdict_from_position_bound(&[], 512), Ap003Verdict::Fail);
    }

    // AP-004 (finite output)
    #[test] fn ap004_pass_canonical() {
        let token = vec![1.0_f32, 2.0, 3.0];
        let pos = vec![0.5_f32, 1.0, 1.5];
        let output = vec![1.5_f32, 3.0, 4.5];
        assert_eq!(verdict_from_finite_output(&token, &pos, &output), Ap004Verdict::Pass);
    }
    #[test] fn ap004_fail_nan_token() {
        // Precondition violated.
        let token = vec![f32::NAN];
        let pos = vec![1.0_f32];
        let output = vec![f32::NAN];
        assert_eq!(verdict_from_finite_output(&token, &pos, &output), Ap004Verdict::Fail);
    }
    #[test] fn ap004_fail_inf_output() {
        // Finite inputs producing non-finite output (the regression class).
        let token = vec![1.0_f32];
        let pos = vec![1.0_f32];
        let output = vec![f32::INFINITY];
        assert_eq!(verdict_from_finite_output(&token, &pos, &output), Ap004Verdict::Fail);
    }
    #[test] fn ap004_fail_length_mismatch() {
        let token = vec![1.0_f32];
        let pos = vec![1.0_f32, 2.0];
        let output = vec![1.0_f32];
        assert_eq!(verdict_from_finite_output(&token, &pos, &output), Ap004Verdict::Fail);
    }
    #[test] fn ap004_fail_empty() {
        assert_eq!(verdict_from_finite_output(&[], &[], &[]), Ap004Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_AP_002_TOLERANCE - 1e-6).abs() < 1e-12);
    }
}
