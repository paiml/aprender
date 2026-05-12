// SHIP-TWO-001 — `qwen3moe-e2e-verification-v1` algorithm-level PARTIAL
// discharge for FALSIFY-QM3E-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/qwen3moe-e2e-verification-v1.yaml`.
// Spec: Qwen3-235B-A22B (MoE) end-to-end verification.

// ===========================================================================
// QM3E-001 — Total params ≈ 235.1B for Qwen3-235B-A22B (within 1%)
// ===========================================================================

pub const AC_QM3E_001_TARGET_TOTAL_PARAMS: u64 = 235_100_000_000;
pub const AC_QM3E_001_TOLERANCE_FRAC: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3e001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_total_param_count(observed: u64) -> Qm3e001Verdict {
    if observed == 0 { return Qm3e001Verdict::Fail; }
    let target = AC_QM3E_001_TARGET_TOTAL_PARAMS as f64;
    let rel = (observed as f64 - target).abs() / target;
    if rel <= AC_QM3E_001_TOLERANCE_FRAC { Qm3e001Verdict::Pass } else { Qm3e001Verdict::Fail }
}

// ===========================================================================
// QM3E-002 — Active params ≈ 22.2B (top-8 of 128 experts)
// ===========================================================================

pub const AC_QM3E_002_TARGET_ACTIVE_PARAMS: u64 = 22_200_000_000;
pub const AC_QM3E_002_TOLERANCE_FRAC: f64 = 0.01;
pub const AC_QM3E_002_NUM_EXPERTS: u64 = 128;
pub const AC_QM3E_002_TOP_K: u64 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3e002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_active_param_count(observed: u64, top_k: u64, num_experts: u64) -> Qm3e002Verdict {
    if observed == 0 { return Qm3e002Verdict::Fail; }
    if top_k == 0 || top_k > num_experts { return Qm3e002Verdict::Fail; }
    if num_experts != AC_QM3E_002_NUM_EXPERTS { return Qm3e002Verdict::Fail; }
    if top_k != AC_QM3E_002_TOP_K { return Qm3e002Verdict::Fail; }
    let target = AC_QM3E_002_TARGET_ACTIVE_PARAMS as f64;
    let rel = (observed as f64 - target).abs() / target;
    if rel <= AC_QM3E_002_TOLERANCE_FRAC { Qm3e002Verdict::Pass } else { Qm3e002Verdict::Fail }
}

// ===========================================================================
// QM3E-003 — FLOPs ≈ 2 * A per forward token (active params)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3e003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_flops_per_token(active_params: u64, observed: u64) -> Qm3e003Verdict {
    if active_params == 0 { return Qm3e003Verdict::Fail; }
    if observed == 2 * active_params { Qm3e003Verdict::Pass } else { Qm3e003Verdict::Fail }
}

// ===========================================================================
// QM3E-004 — Memory ordering: Q4K < Q6K < F16 < F32
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3e004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_memory_ordering(q4k: u64, q6k: u64, f16: u64, f32_: u64) -> Qm3e004Verdict {
    if q4k == 0 || q6k == 0 || f16 == 0 || f32_ == 0 { return Qm3e004Verdict::Fail; }
    if q4k < q6k && q6k < f16 && f16 < f32_ { Qm3e004Verdict::Pass } else { Qm3e004Verdict::Fail }
}

// ===========================================================================
// QM3E-005 — Throughput roofline + bandwidth monotonicity
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3e005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_throughput_roofline(observed_tps: f64, bw_tps: f64, compute_tps: f64) -> Qm3e005Verdict {
    if !observed_tps.is_finite() || !bw_tps.is_finite() || !compute_tps.is_finite() {
        return Qm3e005Verdict::Fail;
    }
    if bw_tps <= 0.0 || compute_tps <= 0.0 || observed_tps < 0.0 { return Qm3e005Verdict::Fail; }
    if observed_tps <= bw_tps.min(compute_tps) { Qm3e005Verdict::Pass } else { Qm3e005Verdict::Fail }
}

#[must_use]
pub fn verdict_from_bandwidth_monotonicity(bw1: f64, tps1: f64, bw2: f64, tps2: f64) -> Qm3e005Verdict {
    if !bw1.is_finite() || !bw2.is_finite() || !tps1.is_finite() || !tps2.is_finite() {
        return Qm3e005Verdict::Fail;
    }
    if bw1 <= 0.0 || bw2 <= 0.0 || tps1 < 0.0 || tps2 < 0.0 { return Qm3e005Verdict::Fail; }
    let monotone = (bw1 < bw2 && tps1 <= tps2) || bw1 >= bw2;
    if monotone { Qm3e005Verdict::Pass } else { Qm3e005Verdict::Fail }
}

// ===========================================================================
// QM3E-006 — MoE block shape preservation: shape(block(x)) == shape(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3e006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_block_shape(input: &[u64], output: &[u64]) -> Qm3e006Verdict {
    if input.is_empty() || output.is_empty() { return Qm3e006Verdict::Fail; }
    if input == output { Qm3e006Verdict::Pass } else { Qm3e006Verdict::Fail }
}

// ===========================================================================
// QM3E-007 — E2E shape: tokens [seq] -> hidden [seq, 4096] -> logits [seq, 151936]
// ===========================================================================

pub const AC_QM3E_007_HIDDEN_DIM: u64 = 4096;
pub const AC_QM3E_007_VOCAB: u64 = 151_936;
pub const AC_QM3E_007_NUM_LAYERS: u64 = 94;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qm3e007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_e2e_shape(tokens: &[u64], hidden: &[u64], logits: &[u64]) -> Qm3e007Verdict {
    if tokens.len() != 1 || hidden.len() != 2 || logits.len() != 2 { return Qm3e007Verdict::Fail; }
    let seq = tokens[0];
    if seq == 0 { return Qm3e007Verdict::Fail; }
    if hidden[0] != seq || hidden[1] != AC_QM3E_007_HIDDEN_DIM { return Qm3e007Verdict::Fail; }
    if logits[0] != seq || logits[1] != AC_QM3E_007_VOCAB { return Qm3e007Verdict::Fail; }
    Qm3e007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // QM3E-001 (Total params ≈ 235.1B)
    #[test] fn qm3e001_pass() { assert_eq!(verdict_from_total_param_count(235_100_000_000), Qm3e001Verdict::Pass); }
    #[test] fn qm3e001_pass_within_tol() {
        let off = (235_100_000_000_u64 as f64 * 0.005) as u64;
        assert_eq!(verdict_from_total_param_count(235_100_000_000 + off), Qm3e001Verdict::Pass);
    }
    #[test] fn qm3e001_fail_above_tol() {
        let off = (235_100_000_000_u64 as f64 * 0.02) as u64;
        assert_eq!(verdict_from_total_param_count(235_100_000_000 + off), Qm3e001Verdict::Fail);
    }
    #[test] fn qm3e001_fail_zero() { assert_eq!(verdict_from_total_param_count(0), Qm3e001Verdict::Fail); }

    // QM3E-002 (Active params ≈ 22.2B)
    #[test] fn qm3e002_pass() {
        assert_eq!(verdict_from_active_param_count(22_200_000_000, 8, 128), Qm3e002Verdict::Pass);
    }
    #[test] fn qm3e002_fail_wrong_top_k() {
        assert_eq!(verdict_from_active_param_count(22_200_000_000, 4, 128), Qm3e002Verdict::Fail);
    }
    #[test] fn qm3e002_fail_wrong_n_experts() {
        assert_eq!(verdict_from_active_param_count(22_200_000_000, 8, 64), Qm3e002Verdict::Fail);
    }
    #[test] fn qm3e002_fail_top_k_zero() {
        assert_eq!(verdict_from_active_param_count(22_200_000_000, 0, 128), Qm3e002Verdict::Fail);
    }
    #[test] fn qm3e002_fail_top_k_above_n_experts() {
        assert_eq!(verdict_from_active_param_count(22_200_000_000, 200, 128), Qm3e002Verdict::Fail);
    }

    // QM3E-003 (FLOPs)
    #[test] fn qm3e003_pass() {
        assert_eq!(verdict_from_flops_per_token(22_000_000_000, 44_000_000_000), Qm3e003Verdict::Pass);
    }
    #[test] fn qm3e003_fail_wrong() {
        assert_eq!(verdict_from_flops_per_token(22_000_000_000, 22_000_000_000), Qm3e003Verdict::Fail);
    }
    #[test] fn qm3e003_fail_zero_a() {
        assert_eq!(verdict_from_flops_per_token(0, 0), Qm3e003Verdict::Fail);
    }

    // QM3E-004 (Memory ordering)
    #[test] fn qm3e004_pass() { assert_eq!(verdict_from_memory_ordering(500, 750, 2000, 4000), Qm3e004Verdict::Pass); }
    #[test] fn qm3e004_fail_q4k_above_q6k() {
        assert_eq!(verdict_from_memory_ordering(800, 750, 2000, 4000), Qm3e004Verdict::Fail);
    }
    #[test] fn qm3e004_fail_zero() {
        assert_eq!(verdict_from_memory_ordering(0, 750, 2000, 4000), Qm3e004Verdict::Fail);
    }

    // QM3E-005 (Throughput)
    #[test] fn qm3e005_pass_in_roofline() {
        assert_eq!(verdict_from_throughput_roofline(150.0, 1000.0, 200.0), Qm3e005Verdict::Pass);
    }
    #[test] fn qm3e005_fail_above() {
        assert_eq!(verdict_from_throughput_roofline(250.0, 1000.0, 200.0), Qm3e005Verdict::Fail);
    }
    #[test] fn qm3e005_fail_nan() {
        assert_eq!(verdict_from_throughput_roofline(f64::NAN, 1000.0, 200.0), Qm3e005Verdict::Fail);
    }
    #[test] fn qm3e005_pass_monotonic() {
        // bw1 < bw2 → tps1 ≤ tps2.
        assert_eq!(verdict_from_bandwidth_monotonicity(100.0, 50.0, 200.0, 100.0), Qm3e005Verdict::Pass);
    }
    #[test] fn qm3e005_fail_monotonic_violation() {
        // bw1 < bw2 but tps1 > tps2.
        assert_eq!(verdict_from_bandwidth_monotonicity(100.0, 100.0, 200.0, 50.0), Qm3e005Verdict::Fail);
    }
    #[test] fn qm3e005_pass_monotonic_decreasing_bw() {
        // bw1 ≥ bw2 — vacuously Pass.
        assert_eq!(verdict_from_bandwidth_monotonicity(200.0, 50.0, 100.0, 100.0), Qm3e005Verdict::Pass);
    }

    // QM3E-006 (Block shape preservation)
    #[test] fn qm3e006_pass_match() {
        let s = vec![16, 4096];
        assert_eq!(verdict_from_block_shape(&s, &s), Qm3e006Verdict::Pass);
    }
    #[test] fn qm3e006_fail_drift() {
        let a = vec![16, 4096];
        let b = vec![16, 4097];
        assert_eq!(verdict_from_block_shape(&a, &b), Qm3e006Verdict::Fail);
    }

    // QM3E-007 (E2E shape)
    #[test] fn qm3e007_pass_canonical() {
        assert_eq!(
            verdict_from_e2e_shape(&[16], &[16, 4096], &[16, 151_936]),
            Qm3e007Verdict::Pass
        );
    }
    #[test] fn qm3e007_fail_seq_mismatch() {
        assert_eq!(
            verdict_from_e2e_shape(&[16], &[15, 4096], &[16, 151_936]),
            Qm3e007Verdict::Fail
        );
    }
    #[test] fn qm3e007_fail_wrong_hidden() {
        assert_eq!(
            verdict_from_e2e_shape(&[16], &[16, 3584], &[16, 151_936]),
            Qm3e007Verdict::Fail
        );
    }
    #[test] fn qm3e007_fail_zero_seq() {
        assert_eq!(
            verdict_from_e2e_shape(&[0], &[0, 4096], &[0, 151_936]),
            Qm3e007Verdict::Fail
        );
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_QM3E_001_TARGET_TOTAL_PARAMS, 235_100_000_000);
        assert_eq!(AC_QM3E_002_TARGET_ACTIVE_PARAMS, 22_200_000_000);
        assert_eq!(AC_QM3E_002_NUM_EXPERTS, 128);
        assert_eq!(AC_QM3E_002_TOP_K, 8);
        assert_eq!(AC_QM3E_007_HIDDEN_DIM, 4096);
        assert_eq!(AC_QM3E_007_VOCAB, 151_936);
        assert_eq!(AC_QM3E_007_NUM_LAYERS, 94);
    }
}
