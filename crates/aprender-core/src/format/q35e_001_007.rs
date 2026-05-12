// SHIP-TWO-001 — `qwen35-e2e-verification-v1` algorithm-level PARTIAL
// discharge for FALSIFY-QE2E-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/qwen35-e2e-verification-v1.yaml`.
// Spec: Qwen3.5-9B end-to-end verification (hybrid GDN+attention).

// ===========================================================================
// QE2E-001 — Total params ≈ 9.05B for Qwen3.5-9B (within 1.5%)
// ===========================================================================

pub const AC_Q35E_001_TARGET_PARAMS: u64 = 9_050_000_000;
pub const AC_Q35E_001_TOLERANCE_FRAC: f64 = 0.015;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35e001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_param_count(observed: u64) -> Q35e001Verdict {
    if observed == 0 { return Q35e001Verdict::Fail; }
    let target = AC_Q35E_001_TARGET_PARAMS as f64;
    let rel = (observed as f64 - target).abs() / target;
    if rel <= AC_Q35E_001_TOLERANCE_FRAC { Q35e001Verdict::Pass } else { Q35e001Verdict::Fail }
}

// ===========================================================================
// QE2E-002 — FLOPs ≈ 2 * P per forward token (with O(seq*d*L) overhead)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35e002Verdict { Pass, Fail }

/// Bound: F <= 2 * P + overhead (overhead reflects O(seq * d * L) attention term).
#[must_use]
pub const fn verdict_from_flops_per_token(params: u64, observed: u64, overhead: u64) -> Q35e002Verdict {
    if params == 0 { return Q35e002Verdict::Fail; }
    let upper = 2u64.saturating_mul(params).saturating_add(overhead);
    if observed <= upper && observed >= 2 * params { Q35e002Verdict::Pass } else { Q35e002Verdict::Fail }
}

// ===========================================================================
// QE2E-003 — Memory ordering: Q4K < Q6K < F16 < F32
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35e003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_memory_ordering(q4k: u64, q6k: u64, f16: u64, f32_: u64) -> Q35e003Verdict {
    if q4k == 0 || q6k == 0 || f16 == 0 || f32_ == 0 { return Q35e003Verdict::Fail; }
    if q4k < q6k && q6k < f16 && f16 < f32_ { Q35e003Verdict::Pass } else { Q35e003Verdict::Fail }
}

// ===========================================================================
// QE2E-004 — Throughput roofline + bandwidth monotonicity
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35e004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_throughput_roofline(observed_tps: f64, bw_tps: f64, compute_tps: f64) -> Q35e004Verdict {
    if !observed_tps.is_finite() || !bw_tps.is_finite() || !compute_tps.is_finite() {
        return Q35e004Verdict::Fail;
    }
    if bw_tps <= 0.0 || compute_tps <= 0.0 || observed_tps < 0.0 { return Q35e004Verdict::Fail; }
    if observed_tps <= bw_tps.min(compute_tps) { Q35e004Verdict::Pass } else { Q35e004Verdict::Fail }
}

#[must_use]
pub fn verdict_from_bandwidth_monotonicity(bw1: f64, tps1: f64, bw2: f64, tps2: f64) -> Q35e004Verdict {
    if !bw1.is_finite() || !bw2.is_finite() || !tps1.is_finite() || !tps2.is_finite() {
        return Q35e004Verdict::Fail;
    }
    if bw1 <= 0.0 || bw2 <= 0.0 || tps1 < 0.0 || tps2 < 0.0 { return Q35e004Verdict::Fail; }
    // Pass if bandwidth didn't decrease while keeping tps within bound,
    // or if bandwidth stayed the same / went up regardless of tps direction.
    let monotone = (bw1 < bw2 && tps1 <= tps2) || bw1 >= bw2;
    if monotone { Q35e004Verdict::Pass } else { Q35e004Verdict::Fail }
}

// ===========================================================================
// QE2E-005 — Coverage completeness: every obligation has evidence (= 1.0)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35e005Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_obligation_coverage(total: u64, with_evidence: u64) -> Q35e005Verdict {
    if total == 0 { return Q35e005Verdict::Fail; }
    if with_evidence > total { return Q35e005Verdict::Fail; }
    if with_evidence == total { Q35e005Verdict::Pass } else { Q35e005Verdict::Fail }
}

// ===========================================================================
// QE2E-006 — Block shape preservation: shape(block_l(x)) = shape(x) ∀l
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35e006Verdict { Pass, Fail }

/// All `outputs[l]` must equal `input` (compositional invariant across all L blocks).
#[must_use]
pub fn verdict_from_block_shape_chain(input: &[u64], outputs: &[Vec<u64>]) -> Q35e006Verdict {
    if input.is_empty() || outputs.is_empty() { return Q35e006Verdict::Fail; }
    for out in outputs {
        if out.as_slice() != input { return Q35e006Verdict::Fail; }
    }
    Q35e006Verdict::Pass
}

// ===========================================================================
// QE2E-007 — E2E shape: tokens [seq] -> hidden [seq, 4096] -> logits [seq, 151936]
// ===========================================================================

pub const AC_Q35E_007_HIDDEN_DIM: u64 = 4096;
pub const AC_Q35E_007_VOCAB: u64 = 151_936;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q35e007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_e2e_shape(tokens: &[u64], hidden: &[u64], logits: &[u64]) -> Q35e007Verdict {
    if tokens.len() != 1 || hidden.len() != 2 || logits.len() != 2 { return Q35e007Verdict::Fail; }
    let seq = tokens[0];
    if seq == 0 { return Q35e007Verdict::Fail; }
    if hidden[0] != seq || hidden[1] != AC_Q35E_007_HIDDEN_DIM { return Q35e007Verdict::Fail; }
    if logits[0] != seq || logits[1] != AC_Q35E_007_VOCAB { return Q35e007Verdict::Fail; }
    Q35e007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // QE2E-001
    #[test] fn q35e001_pass() { assert_eq!(verdict_from_param_count(9_050_000_000), Q35e001Verdict::Pass); }
    #[test] fn q35e001_pass_within_tol() {
        let off = (9_050_000_000_u64 as f64 * 0.01) as u64;
        assert_eq!(verdict_from_param_count(9_050_000_000 + off), Q35e001Verdict::Pass);
    }
    #[test] fn q35e001_fail_above_tol() {
        let off = (9_050_000_000_u64 as f64 * 0.05) as u64;
        assert_eq!(verdict_from_param_count(9_050_000_000 + off), Q35e001Verdict::Fail);
    }
    #[test] fn q35e001_fail_zero() { assert_eq!(verdict_from_param_count(0), Q35e001Verdict::Fail); }

    // QE2E-002
    #[test] fn q35e002_pass_strict_2p() {
        assert_eq!(verdict_from_flops_per_token(9_000_000_000, 18_000_000_000, 0), Q35e002Verdict::Pass);
    }
    #[test] fn q35e002_pass_with_overhead() {
        // 2P + 1B overhead.
        assert_eq!(
            verdict_from_flops_per_token(9_000_000_000, 19_000_000_000, 1_000_000_000),
            Q35e002Verdict::Pass
        );
    }
    #[test] fn q35e002_fail_below_2p() {
        assert_eq!(verdict_from_flops_per_token(9_000_000_000, 17_000_000_000, 1_000_000_000), Q35e002Verdict::Fail);
    }
    #[test] fn q35e002_fail_above_overhead_band() {
        assert_eq!(
            verdict_from_flops_per_token(9_000_000_000, 30_000_000_000, 1_000_000_000),
            Q35e002Verdict::Fail
        );
    }
    #[test] fn q35e002_fail_zero_p() {
        assert_eq!(verdict_from_flops_per_token(0, 0, 0), Q35e002Verdict::Fail);
    }

    // QE2E-003
    #[test] fn q35e003_pass() { assert_eq!(verdict_from_memory_ordering(500, 750, 2000, 4000), Q35e003Verdict::Pass); }
    #[test] fn q35e003_fail_q4k_above_q6k() {
        assert_eq!(verdict_from_memory_ordering(800, 750, 2000, 4000), Q35e003Verdict::Fail);
    }
    #[test] fn q35e003_fail_zero() {
        assert_eq!(verdict_from_memory_ordering(0, 750, 2000, 4000), Q35e003Verdict::Fail);
    }

    // QE2E-004
    #[test] fn q35e004_pass_in_roofline() {
        assert_eq!(verdict_from_throughput_roofline(150.0, 1000.0, 200.0), Q35e004Verdict::Pass);
    }
    #[test] fn q35e004_fail_above() {
        assert_eq!(verdict_from_throughput_roofline(250.0, 1000.0, 200.0), Q35e004Verdict::Fail);
    }
    #[test] fn q35e004_fail_nan() {
        assert_eq!(verdict_from_throughput_roofline(f64::NAN, 1000.0, 200.0), Q35e004Verdict::Fail);
    }
    #[test] fn q35e004_pass_monotonic() {
        assert_eq!(verdict_from_bandwidth_monotonicity(100.0, 50.0, 200.0, 100.0), Q35e004Verdict::Pass);
    }
    #[test] fn q35e004_fail_monotonic_violation() {
        assert_eq!(verdict_from_bandwidth_monotonicity(100.0, 100.0, 200.0, 50.0), Q35e004Verdict::Fail);
    }

    // QE2E-005
    #[test] fn q35e005_pass_full() { assert_eq!(verdict_from_obligation_coverage(10, 10), Q35e005Verdict::Pass); }
    #[test] fn q35e005_fail_partial() { assert_eq!(verdict_from_obligation_coverage(10, 9), Q35e005Verdict::Fail); }
    #[test] fn q35e005_fail_zero_total() { assert_eq!(verdict_from_obligation_coverage(0, 0), Q35e005Verdict::Fail); }
    #[test] fn q35e005_fail_inverted() { assert_eq!(verdict_from_obligation_coverage(5, 10), Q35e005Verdict::Fail); }

    // QE2E-006 — chain over all L blocks
    #[test] fn q35e006_pass_all_blocks() {
        let s = vec![16, 4096];
        let chain: Vec<Vec<u64>> = (0..40).map(|_| s.clone()).collect();
        assert_eq!(verdict_from_block_shape_chain(&s, &chain), Q35e006Verdict::Pass);
    }
    #[test] fn q35e006_fail_one_block_drift() {
        let s = vec![16, 4096];
        let mut chain: Vec<Vec<u64>> = (0..40).map(|_| s.clone()).collect();
        chain[15] = vec![16, 4097]; // off-by-one in one layer breaks composition
        assert_eq!(verdict_from_block_shape_chain(&s, &chain), Q35e006Verdict::Fail);
    }
    #[test] fn q35e006_fail_empty() {
        let s = vec![16, 4096];
        let chain: Vec<Vec<u64>> = vec![];
        assert_eq!(verdict_from_block_shape_chain(&s, &chain), Q35e006Verdict::Fail);
    }

    // QE2E-007
    #[test] fn q35e007_pass_canonical() {
        assert_eq!(
            verdict_from_e2e_shape(&[16], &[16, 4096], &[16, 151_936]),
            Q35e007Verdict::Pass
        );
    }
    #[test] fn q35e007_fail_seq_mismatch() {
        assert_eq!(
            verdict_from_e2e_shape(&[16], &[15, 4096], &[16, 151_936]),
            Q35e007Verdict::Fail
        );
    }
    #[test] fn q35e007_fail_wrong_hidden() {
        assert_eq!(
            verdict_from_e2e_shape(&[16], &[16, 3584], &[16, 151_936]),
            Q35e007Verdict::Fail
        );
    }
    #[test] fn q35e007_fail_zero_seq() {
        assert_eq!(
            verdict_from_e2e_shape(&[0], &[0, 4096], &[0, 151_936]),
            Q35e007Verdict::Fail
        );
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_Q35E_001_TARGET_PARAMS, 9_050_000_000);
        assert_eq!(AC_Q35E_007_HIDDEN_DIM, 4096);
        assert_eq!(AC_Q35E_007_VOCAB, 151_936);
    }
}
