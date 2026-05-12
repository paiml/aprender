// SHIP-TWO-001 — `qwen3-e2e-verification-v1` algorithm-level PARTIAL
// discharge for FALSIFY-QW3E-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/qwen3-e2e-verification-v1.yaml`.
// Spec: Qwen3-8B end-to-end verification.

// ===========================================================================
// QW3E-001 — Param count ≈ 8.19B for Qwen3-8B (within 0.5%)
// ===========================================================================

pub const AC_QW3E_001_TARGET_PARAMS: u64 = 8_190_000_000;
pub const AC_QW3E_001_TOLERANCE_FRAC: f64 = 0.005;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3e001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_param_count(observed: u64) -> Qw3e001Verdict {
    if observed == 0 { return Qw3e001Verdict::Fail; }
    let target = AC_QW3E_001_TARGET_PARAMS as f64;
    let rel = (observed as f64 - target).abs() / target;
    if rel <= AC_QW3E_001_TOLERANCE_FRAC { Qw3e001Verdict::Pass } else { Qw3e001Verdict::Fail }
}

// ===========================================================================
// QW3E-002 — FLOPs == 2 * P per forward token (strict)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3e002Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_flops_per_token(params: u64, observed: u64) -> Qw3e002Verdict {
    if params == 0 { return Qw3e002Verdict::Fail; }
    if observed == 2 * params { Qw3e002Verdict::Pass } else { Qw3e002Verdict::Fail }
}

// ===========================================================================
// QW3E-003 — Memory ordering: Q4K < Q6K < F16 < F32
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3e003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_memory_ordering(q4k: u64, q6k: u64, f16: u64, f32_: u64) -> Qw3e003Verdict {
    if q4k == 0 || q6k == 0 || f16 == 0 || f32_ == 0 { return Qw3e003Verdict::Fail; }
    if q4k < q6k && q6k < f16 && f16 < f32_ { Qw3e003Verdict::Pass } else { Qw3e003Verdict::Fail }
}

// ===========================================================================
// QW3E-004 — Roofline: observed_tps <= min(bandwidth_tps, compute_tps)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3e004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_throughput_roofline(observed_tps: f64, bw_tps: f64, compute_tps: f64) -> Qw3e004Verdict {
    if !observed_tps.is_finite() || !bw_tps.is_finite() || !compute_tps.is_finite() {
        return Qw3e004Verdict::Fail;
    }
    if bw_tps <= 0.0 || compute_tps <= 0.0 || observed_tps < 0.0 { return Qw3e004Verdict::Fail; }
    if observed_tps <= bw_tps.min(compute_tps) { Qw3e004Verdict::Pass } else { Qw3e004Verdict::Fail }
}

// ===========================================================================
// QW3E-005 — Coverage: every obligation has evidence
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3e005Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_obligation_coverage(total: u64, with_evidence: u64) -> Qw3e005Verdict {
    if total == 0 { return Qw3e005Verdict::Fail; }
    if with_evidence > total { return Qw3e005Verdict::Fail; }
    if with_evidence == total { Qw3e005Verdict::Pass } else { Qw3e005Verdict::Fail }
}

// ===========================================================================
// QW3E-006 — Block shape preservation: shape(block(x)) == shape(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3e006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_block_shape(input: &[u64], output: &[u64]) -> Qw3e006Verdict {
    if input.is_empty() || output.is_empty() { return Qw3e006Verdict::Fail; }
    if input == output { Qw3e006Verdict::Pass } else { Qw3e006Verdict::Fail }
}

// ===========================================================================
// QW3E-007 — E2E shape: tokens [seq] -> hidden [seq, 4096] -> logits [seq, 151936]
// ===========================================================================

pub const AC_QW3E_007_HIDDEN_DIM: u64 = 4096;
pub const AC_QW3E_007_VOCAB: u64 = 151_936;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3e007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_e2e_shape(tokens: &[u64], hidden: &[u64], logits: &[u64]) -> Qw3e007Verdict {
    if tokens.len() != 1 || hidden.len() != 2 || logits.len() != 2 { return Qw3e007Verdict::Fail; }
    let seq = tokens[0];
    if seq == 0 { return Qw3e007Verdict::Fail; }
    if hidden[0] != seq || hidden[1] != AC_QW3E_007_HIDDEN_DIM { return Qw3e007Verdict::Fail; }
    if logits[0] != seq || logits[1] != AC_QW3E_007_VOCAB { return Qw3e007Verdict::Fail; }
    Qw3e007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // QW3E-001
    #[test] fn qw3e001_pass() { assert_eq!(verdict_from_param_count(8_190_000_000), Qw3e001Verdict::Pass); }
    #[test] fn qw3e001_pass_within_tol() {
        let off = (8_190_000_000_u64 as f64 * 0.004) as u64;
        assert_eq!(verdict_from_param_count(8_190_000_000 + off), Qw3e001Verdict::Pass);
    }
    #[test] fn qw3e001_fail_above_tol() {
        let off = (8_190_000_000_u64 as f64 * 0.01) as u64;
        assert_eq!(verdict_from_param_count(8_190_000_000 + off), Qw3e001Verdict::Fail);
    }
    #[test] fn qw3e001_fail_zero() { assert_eq!(verdict_from_param_count(0), Qw3e001Verdict::Fail); }

    // QW3E-002
    #[test] fn qw3e002_pass() {
        assert_eq!(verdict_from_flops_per_token(8_000_000_000, 16_000_000_000), Qw3e002Verdict::Pass);
    }
    #[test] fn qw3e002_fail_one_p() {
        assert_eq!(verdict_from_flops_per_token(8_000_000_000, 8_000_000_000), Qw3e002Verdict::Fail);
    }

    // QW3E-003
    #[test] fn qw3e003_pass() { assert_eq!(verdict_from_memory_ordering(500, 750, 2000, 4000), Qw3e003Verdict::Pass); }
    #[test] fn qw3e003_fail_q4k_above_q6k() {
        assert_eq!(verdict_from_memory_ordering(800, 750, 2000, 4000), Qw3e003Verdict::Fail);
    }
    #[test] fn qw3e003_fail_zero() {
        assert_eq!(verdict_from_memory_ordering(0, 750, 2000, 4000), Qw3e003Verdict::Fail);
    }

    // QW3E-004
    #[test] fn qw3e004_pass_in_roofline() {
        assert_eq!(verdict_from_throughput_roofline(150.0, 1000.0, 200.0), Qw3e004Verdict::Pass);
    }
    #[test] fn qw3e004_fail_above() {
        assert_eq!(verdict_from_throughput_roofline(250.0, 1000.0, 200.0), Qw3e004Verdict::Fail);
    }
    #[test] fn qw3e004_fail_nan() {
        assert_eq!(verdict_from_throughput_roofline(f64::NAN, 1000.0, 200.0), Qw3e004Verdict::Fail);
    }

    // QW3E-005
    #[test] fn qw3e005_pass_full() { assert_eq!(verdict_from_obligation_coverage(10, 10), Qw3e005Verdict::Pass); }
    #[test] fn qw3e005_fail_partial() { assert_eq!(verdict_from_obligation_coverage(10, 9), Qw3e005Verdict::Fail); }
    #[test] fn qw3e005_fail_zero_total() { assert_eq!(verdict_from_obligation_coverage(0, 0), Qw3e005Verdict::Fail); }
    #[test] fn qw3e005_fail_inverted() { assert_eq!(verdict_from_obligation_coverage(5, 10), Qw3e005Verdict::Fail); }

    // QW3E-006
    #[test] fn qw3e006_pass_match() {
        let s = vec![16, 4096];
        assert_eq!(verdict_from_block_shape(&s, &s), Qw3e006Verdict::Pass);
    }
    #[test] fn qw3e006_fail_drift() {
        let a = vec![16, 4096];
        let b = vec![16, 4097];
        assert_eq!(verdict_from_block_shape(&a, &b), Qw3e006Verdict::Fail);
    }

    // QW3E-007
    #[test] fn qw3e007_pass_canonical() {
        assert_eq!(
            verdict_from_e2e_shape(&[16], &[16, 4096], &[16, 151_936]),
            Qw3e007Verdict::Pass
        );
    }
    #[test] fn qw3e007_fail_seq_mismatch() {
        assert_eq!(
            verdict_from_e2e_shape(&[16], &[15, 4096], &[16, 151_936]),
            Qw3e007Verdict::Fail
        );
    }
    #[test] fn qw3e007_fail_wrong_hidden() {
        assert_eq!(
            verdict_from_e2e_shape(&[16], &[16, 3584], &[16, 151_936]),
            Qw3e007Verdict::Fail
        );
    }
    #[test] fn qw3e007_fail_zero_seq() {
        assert_eq!(
            verdict_from_e2e_shape(&[0], &[0, 4096], &[0, 151_936]),
            Qw3e007Verdict::Fail
        );
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_QW3E_001_TARGET_PARAMS, 8_190_000_000);
        assert_eq!(AC_QW3E_007_HIDDEN_DIM, 4096);
        assert_eq!(AC_QW3E_007_VOCAB, 151_936);
    }
}
