// SHIP-TWO-001 — `qwen2-e2e-verification-v1` algorithm-level PARTIAL
// discharge for FALSIFY-QW2E-001..007 (closes 7 unbound; the SHIP-*
// gates 001..024 are bound separately).
//
// Contract: `contracts/qwen2-e2e-verification-v1.yaml`.
// Spec: SHIP-TWO-001 §4 (MODEL-1 Qwen2.5-Coder-7B teacher).

// ===========================================================================
// QW2E-001 — Parameter count ≈ 7.62B for Qwen2.5-Coder-7B
// ===========================================================================
//
// Spec-pinned canonical count. Tolerance is 0.5% of 7.62B (~38M) to
// allow for embedding-tying and head-merge variants while still
// flagging architectural drift.

pub const AC_QW2E_001_TARGET_PARAMS: u64 = 7_615_616_512;
pub const AC_QW2E_001_TOLERANCE_FRAC: f64 = 0.005;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2e001Verdict { Pass, Fail }

/// Pass iff `|observed - target| / target <= 0.5%`.
#[must_use]
pub fn verdict_from_param_count(observed: u64) -> Qw2e001Verdict {
    if observed == 0 { return Qw2e001Verdict::Fail; }
    let target = AC_QW2E_001_TARGET_PARAMS as f64;
    let diff = (observed as f64 - target).abs();
    let rel = diff / target;
    if rel <= AC_QW2E_001_TOLERANCE_FRAC { Qw2e001Verdict::Pass } else { Qw2e001Verdict::Fail }
}

// ===========================================================================
// QW2E-002 — FLOPs estimate: 2P FLOPs per forward token
// ===========================================================================
//
// Standard transformer roofline: each forward token costs ≈ 2 × P
// FLOPs (1× for forward, 1× for matmul accumulator).

pub const AC_QW2E_002_FLOPS_PER_PARAM_PER_TOKEN: u64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2e002Verdict { Pass, Fail }

/// Pass iff `observed_flops_per_token == 2 * params` exactly.
/// Strict equality is intentional — this is the canonical roofline
/// formula and any deviation is a contract drift signal.
#[must_use]
pub const fn verdict_from_flops_per_token(params: u64, observed_flops: u64) -> Qw2e002Verdict {
    if params == 0 { return Qw2e002Verdict::Fail; }
    if observed_flops == AC_QW2E_002_FLOPS_PER_PARAM_PER_TOKEN * params {
        Qw2e002Verdict::Pass
    } else {
        Qw2e002Verdict::Fail
    }
}

// ===========================================================================
// QW2E-003 — Memory ordering: Q4K < Q6K < F16 < F32
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2e003Verdict { Pass, Fail }

/// Pass iff `q4k_bytes < q6k_bytes < f16_bytes < f32_bytes` strictly.
#[must_use]
pub const fn verdict_from_memory_ordering(
    q4k_bytes: u64,
    q6k_bytes: u64,
    f16_bytes: u64,
    f32_bytes: u64,
) -> Qw2e003Verdict {
    if q4k_bytes == 0 || q6k_bytes == 0 || f16_bytes == 0 || f32_bytes == 0 {
        return Qw2e003Verdict::Fail;
    }
    if q4k_bytes < q6k_bytes && q6k_bytes < f16_bytes && f16_bytes < f32_bytes {
        Qw2e003Verdict::Pass
    } else {
        Qw2e003Verdict::Fail
    }
}

// ===========================================================================
// QW2E-004 — Throughput roofline: tok/s ≤ min(bandwidth, compute)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2e004Verdict { Pass, Fail }

/// Pass iff `observed_tps <= min(bandwidth_tps, compute_tps)`.
/// A measured throughput exceeding the roofline indicates the
/// formula is wrong (or the hardware report is dishonest). Both
/// rooflines must be > 0 to avoid divide-by-zero ambiguity.
#[must_use]
pub fn verdict_from_throughput_roofline(
    observed_tps: f64,
    bandwidth_tps: f64,
    compute_tps: f64,
) -> Qw2e004Verdict {
    if !observed_tps.is_finite() || !bandwidth_tps.is_finite() || !compute_tps.is_finite() {
        return Qw2e004Verdict::Fail;
    }
    if bandwidth_tps <= 0.0 || compute_tps <= 0.0 || observed_tps < 0.0 {
        return Qw2e004Verdict::Fail;
    }
    let roofline = bandwidth_tps.min(compute_tps);
    if observed_tps <= roofline { Qw2e004Verdict::Pass } else { Qw2e004Verdict::Fail }
}

// ===========================================================================
// QW2E-005 — Coverage completeness: every obligation has test or proof
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2e005Verdict { Pass, Fail }

/// Pass iff `obligations_with_evidence == total_obligations` AND
/// `total_obligations > 0`. Strict equality — every obligation must
/// have at least one piece of evidence (test, proof, or runtime
/// falsifier).
#[must_use]
pub const fn verdict_from_obligation_coverage(
    total_obligations: u64,
    obligations_with_evidence: u64,
) -> Qw2e005Verdict {
    if total_obligations == 0 { return Qw2e005Verdict::Fail; }
    if obligations_with_evidence > total_obligations { return Qw2e005Verdict::Fail; }
    if obligations_with_evidence == total_obligations {
        Qw2e005Verdict::Pass
    } else {
        Qw2e005Verdict::Fail
    }
}

// ===========================================================================
// QW2E-006 — Compositional proof: shape(block_l(x)) == shape(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2e006Verdict { Pass, Fail }

/// Pass iff `output_shape == input_shape`. Each transformer block
/// is a shape-preserving map; any divergence indicates a malformed
/// residual stream / projection.
#[must_use]
pub fn verdict_from_block_shape_preservation(
    input_shape: &[u64],
    output_shape: &[u64],
) -> Qw2e006Verdict {
    if input_shape.is_empty() || output_shape.is_empty() {
        return Qw2e006Verdict::Fail;
    }
    if input_shape == output_shape { Qw2e006Verdict::Pass } else { Qw2e006Verdict::Fail }
}

// ===========================================================================
// QW2E-007 — End-to-end shape conservation
//
// tokens (`[seq]`) -> embedding (`[seq, hidden]`) -> ... ->
// lm_head logits (`[seq, vocab]`).
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw2e007Verdict { Pass, Fail }

/// Pass iff:
///   - `token_shape == [seq]` (rank-1, len > 0)
///   - `hidden_shape == [seq, hidden]` (rank-2, dim 0 matches tokens, hidden > 0)
///   - `logits_shape == [seq, vocab]` (rank-2, dim 0 matches tokens, vocab > 0)
#[must_use]
pub fn verdict_from_e2e_shape_conservation(
    token_shape: &[u64],
    hidden_shape: &[u64],
    logits_shape: &[u64],
) -> Qw2e007Verdict {
    if token_shape.len() != 1 { return Qw2e007Verdict::Fail; }
    if hidden_shape.len() != 2 { return Qw2e007Verdict::Fail; }
    if logits_shape.len() != 2 { return Qw2e007Verdict::Fail; }
    let seq = token_shape[0];
    if seq == 0 { return Qw2e007Verdict::Fail; }
    if hidden_shape[0] != seq || hidden_shape[1] == 0 { return Qw2e007Verdict::Fail; }
    if logits_shape[0] != seq || logits_shape[1] == 0 { return Qw2e007Verdict::Fail; }
    Qw2e007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- QW2E-001 ----------------------------------------------------------

    #[test]
    fn qw2e001_pass_exact_target() {
        assert_eq!(verdict_from_param_count(AC_QW2E_001_TARGET_PARAMS), Qw2e001Verdict::Pass);
    }

    #[test]
    fn qw2e001_pass_within_tolerance() {
        // 0.4% off — inside the 0.5% band.
        let off = (AC_QW2E_001_TARGET_PARAMS as f64 * 0.004) as u64;
        assert_eq!(
            verdict_from_param_count(AC_QW2E_001_TARGET_PARAMS - off),
            Qw2e001Verdict::Pass
        );
        assert_eq!(
            verdict_from_param_count(AC_QW2E_001_TARGET_PARAMS + off),
            Qw2e001Verdict::Pass
        );
    }

    #[test]
    fn qw2e001_fail_above_tolerance() {
        // 1% off — outside the 0.5% band.
        let off = (AC_QW2E_001_TARGET_PARAMS as f64 * 0.01) as u64;
        assert_eq!(
            verdict_from_param_count(AC_QW2E_001_TARGET_PARAMS + off),
            Qw2e001Verdict::Fail
        );
    }

    #[test]
    fn qw2e001_fail_zero() {
        assert_eq!(verdict_from_param_count(0), Qw2e001Verdict::Fail);
    }

    #[test]
    fn qw2e001_provenance() {
        assert_eq!(AC_QW2E_001_TARGET_PARAMS, 7_615_616_512);
        assert!((AC_QW2E_001_TOLERANCE_FRAC - 0.005).abs() < 1e-12);
    }

    // ----- QW2E-002 ----------------------------------------------------------

    #[test]
    fn qw2e002_pass_canonical() {
        assert_eq!(
            verdict_from_flops_per_token(7_000_000_000, 14_000_000_000),
            Qw2e002Verdict::Pass
        );
    }

    #[test]
    fn qw2e002_fail_off_by_one() {
        assert_eq!(
            verdict_from_flops_per_token(7_000_000_000, 14_000_000_001),
            Qw2e002Verdict::Fail
        );
    }

    #[test]
    fn qw2e002_fail_one_p() {
        assert_eq!(
            verdict_from_flops_per_token(7_000_000_000, 7_000_000_000),
            Qw2e002Verdict::Fail
        );
    }

    #[test]
    fn qw2e002_fail_zero_params() {
        assert_eq!(verdict_from_flops_per_token(0, 0), Qw2e002Verdict::Fail);
    }

    // ----- QW2E-003 ----------------------------------------------------------

    #[test]
    fn qw2e003_pass_canonical_ordering() {
        // 1B params:
        //   Q4K ≈ 0.5 GB, Q6K ≈ 0.75 GB, F16 = 2 GB, F32 = 4 GB.
        let q4k = 500_000_000;
        let q6k = 750_000_000;
        let f16 = 2_000_000_000;
        let f32 = 4_000_000_000;
        assert_eq!(
            verdict_from_memory_ordering(q4k, q6k, f16, f32),
            Qw2e003Verdict::Pass
        );
    }

    #[test]
    fn qw2e003_fail_q4k_above_q6k() {
        assert_eq!(
            verdict_from_memory_ordering(800, 750, 2_000, 4_000),
            Qw2e003Verdict::Fail
        );
    }

    #[test]
    fn qw2e003_fail_f16_above_f32() {
        assert_eq!(
            verdict_from_memory_ordering(500, 750, 4_000, 2_000),
            Qw2e003Verdict::Fail
        );
    }

    #[test]
    fn qw2e003_fail_equal_q4k_q6k() {
        // Strict ordering — equality is Fail.
        assert_eq!(
            verdict_from_memory_ordering(500, 500, 2_000, 4_000),
            Qw2e003Verdict::Fail
        );
    }

    #[test]
    fn qw2e003_fail_zero_arg() {
        assert_eq!(
            verdict_from_memory_ordering(0, 750, 2_000, 4_000),
            Qw2e003Verdict::Fail
        );
    }

    // ----- QW2E-004 ----------------------------------------------------------

    #[test]
    fn qw2e004_pass_compute_bound() {
        // Bandwidth high (1000), compute low (200), observed below
        // compute roofline.
        assert_eq!(
            verdict_from_throughput_roofline(150.0, 1000.0, 200.0),
            Qw2e004Verdict::Pass
        );
    }

    #[test]
    fn qw2e004_pass_at_min_roofline() {
        assert_eq!(
            verdict_from_throughput_roofline(200.0, 1000.0, 200.0),
            Qw2e004Verdict::Pass
        );
    }

    #[test]
    fn qw2e004_fail_above_roofline() {
        // observed > min(1000, 200) = 200.
        assert_eq!(
            verdict_from_throughput_roofline(250.0, 1000.0, 200.0),
            Qw2e004Verdict::Fail
        );
    }

    #[test]
    fn qw2e004_fail_negative_observed() {
        assert_eq!(
            verdict_from_throughput_roofline(-1.0, 1000.0, 200.0),
            Qw2e004Verdict::Fail
        );
    }

    #[test]
    fn qw2e004_fail_zero_roofline() {
        assert_eq!(
            verdict_from_throughput_roofline(50.0, 0.0, 200.0),
            Qw2e004Verdict::Fail
        );
    }

    #[test]
    fn qw2e004_fail_nan() {
        assert_eq!(
            verdict_from_throughput_roofline(f64::NAN, 1000.0, 200.0),
            Qw2e004Verdict::Fail
        );
    }

    // ----- QW2E-005 ----------------------------------------------------------

    #[test]
    fn qw2e005_pass_full_coverage() {
        assert_eq!(verdict_from_obligation_coverage(10, 10), Qw2e005Verdict::Pass);
    }

    #[test]
    fn qw2e005_fail_partial_coverage() {
        assert_eq!(verdict_from_obligation_coverage(10, 9), Qw2e005Verdict::Fail);
    }

    #[test]
    fn qw2e005_fail_no_coverage() {
        assert_eq!(verdict_from_obligation_coverage(10, 0), Qw2e005Verdict::Fail);
    }

    #[test]
    fn qw2e005_fail_zero_total() {
        // Vacuous truth not accepted — zero obligations means the
        // contract isn't doing its job.
        assert_eq!(verdict_from_obligation_coverage(0, 0), Qw2e005Verdict::Fail);
    }

    #[test]
    fn qw2e005_fail_more_evidence_than_obligations() {
        // Counter inversion (defensive Fail).
        assert_eq!(verdict_from_obligation_coverage(5, 10), Qw2e005Verdict::Fail);
    }

    // ----- QW2E-006 ----------------------------------------------------------

    #[test]
    fn qw2e006_pass_2d_match() {
        let s = vec![16, 3584];
        assert_eq!(verdict_from_block_shape_preservation(&s, &s), Qw2e006Verdict::Pass);
    }

    #[test]
    fn qw2e006_pass_3d_match() {
        let s = vec![1, 16, 3584];
        assert_eq!(verdict_from_block_shape_preservation(&s, &s), Qw2e006Verdict::Pass);
    }

    #[test]
    fn qw2e006_fail_dim_drift() {
        let a = vec![16, 3584];
        let b = vec![16, 3585];
        assert_eq!(verdict_from_block_shape_preservation(&a, &b), Qw2e006Verdict::Fail);
    }

    #[test]
    fn qw2e006_fail_rank_drift() {
        let a = vec![16, 3584];
        let b = vec![16, 3584, 1];
        assert_eq!(verdict_from_block_shape_preservation(&a, &b), Qw2e006Verdict::Fail);
    }

    #[test]
    fn qw2e006_fail_empty() {
        let empty: Vec<u64> = vec![];
        let s = vec![16, 3584];
        assert_eq!(verdict_from_block_shape_preservation(&empty, &s), Qw2e006Verdict::Fail);
        assert_eq!(verdict_from_block_shape_preservation(&s, &empty), Qw2e006Verdict::Fail);
    }

    // ----- QW2E-007 ----------------------------------------------------------

    #[test]
    fn qw2e007_pass_qwen_canonical() {
        let tokens = vec![16];
        let hidden = vec![16, 3584];
        let logits = vec![16, 152_064];
        assert_eq!(
            verdict_from_e2e_shape_conservation(&tokens, &hidden, &logits),
            Qw2e007Verdict::Pass
        );
    }

    #[test]
    fn qw2e007_fail_token_rank_drift() {
        let tokens = vec![1, 16]; // rank-2 instead of rank-1
        let hidden = vec![16, 3584];
        let logits = vec![16, 152_064];
        assert_eq!(
            verdict_from_e2e_shape_conservation(&tokens, &hidden, &logits),
            Qw2e007Verdict::Fail
        );
    }

    #[test]
    fn qw2e007_fail_seq_mismatch_hidden() {
        let tokens = vec![16];
        let hidden = vec![15, 3584]; // wrong seq dim
        let logits = vec![16, 152_064];
        assert_eq!(
            verdict_from_e2e_shape_conservation(&tokens, &hidden, &logits),
            Qw2e007Verdict::Fail
        );
    }

    #[test]
    fn qw2e007_fail_seq_mismatch_logits() {
        let tokens = vec![16];
        let hidden = vec![16, 3584];
        let logits = vec![15, 152_064];
        assert_eq!(
            verdict_from_e2e_shape_conservation(&tokens, &hidden, &logits),
            Qw2e007Verdict::Fail
        );
    }

    #[test]
    fn qw2e007_fail_zero_seq() {
        let tokens = vec![0];
        let hidden = vec![0, 3584];
        let logits = vec![0, 152_064];
        assert_eq!(
            verdict_from_e2e_shape_conservation(&tokens, &hidden, &logits),
            Qw2e007Verdict::Fail
        );
    }

    #[test]
    fn qw2e007_fail_zero_vocab() {
        let tokens = vec![16];
        let hidden = vec![16, 3584];
        let logits = vec![16, 0];
        assert_eq!(
            verdict_from_e2e_shape_conservation(&tokens, &hidden, &logits),
            Qw2e007Verdict::Fail
        );
    }

    #[test]
    fn qw2e007_fail_zero_hidden() {
        let tokens = vec![16];
        let hidden = vec![16, 0];
        let logits = vec![16, 152_064];
        assert_eq!(
            verdict_from_e2e_shape_conservation(&tokens, &hidden, &logits),
            Qw2e007Verdict::Fail
        );
    }
}
