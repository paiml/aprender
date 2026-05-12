// SHIP-TWO-001 — `continuous-batching-v1` algorithm-level PARTIAL
// discharge for FALSIFY-CB-001..009 (closes 9/9 sweep).
//
// Contract: `contracts/continuous-batching-v1.yaml`.
//
// Each verdict is a pure decision rule isolated from the live
// scheduler so the algorithm-level rule is testable offline.

// ===========================================================================
// CB-001 — Token budget: scheduled tokens never exceed max_batch_tokens
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_token_budget(scheduled_tokens: u64, max_batch_tokens: u64) -> Cb001Verdict {
    if max_batch_tokens == 0 { return Cb001Verdict::Fail; }
    if scheduled_tokens <= max_batch_tokens { Cb001Verdict::Pass } else { Cb001Verdict::Fail }
}

// ===========================================================================
// CB-002 — Computed tokens monotonic across steps
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_computed_monotonic(history: &[u64]) -> Cb002Verdict {
    if history.is_empty() { return Cb002Verdict::Fail; }
    for w in history.windows(2) {
        if w[1] < w[0] { return Cb002Verdict::Fail; }
    }
    Cb002Verdict::Pass
}

// ===========================================================================
// CB-003 — Chunked prefill equivalence: same KV cache as full prefill
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb003Verdict { Pass, Fail }

/// Pass iff `chunked_kv_cache_hash == full_kv_cache_hash` (byte-identical).
#[must_use]
pub fn verdict_from_chunked_prefill_equivalence(
    chunked_hash: &[u8],
    full_hash: &[u8],
) -> Cb003Verdict {
    if chunked_hash.is_empty() || full_hash.is_empty() { return Cb003Verdict::Fail; }
    if chunked_hash == full_hash { Cb003Verdict::Pass } else { Cb003Verdict::Fail }
}

// ===========================================================================
// CB-004 — Decode degradation bounded: c=4 per-request >= 50% of c=1
// ===========================================================================

pub const AC_CB_004_MIN_DEGRADATION_RATIO: f64 = 0.50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_decode_degradation(c1_tps: f64, c4_per_request_tps: f64) -> Cb004Verdict {
    if !c1_tps.is_finite() || !c4_per_request_tps.is_finite() { return Cb004Verdict::Fail; }
    if c1_tps <= 0.0 || c4_per_request_tps < 0.0 { return Cb004Verdict::Fail; }
    let ratio = c4_per_request_tps / c1_tps;
    if ratio >= AC_CB_004_MIN_DEGRADATION_RATIO { Cb004Verdict::Pass } else { Cb004Verdict::Fail }
}

// ===========================================================================
// CB-005 — No starvation: max wait time ≤ bound
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_no_starvation(wait_times_ms: &[u64], max_wait_bound_ms: u64) -> Cb005Verdict {
    if wait_times_ms.is_empty() { return Cb005Verdict::Fail; }
    if max_wait_bound_ms == 0 { return Cb005Verdict::Fail; }
    for &w in wait_times_ms {
        if w > max_wait_bound_ms { return Cb005Verdict::Fail; }
    }
    Cb005Verdict::Pass
}

// ===========================================================================
// CB-006 — Correctness under batching: c=1 token output == c=4 token output
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_batching_correctness(
    c1_tokens: &[u32],
    c4_tokens: &[u32],
) -> Cb006Verdict {
    if c1_tokens.is_empty() || c4_tokens.is_empty() { return Cb006Verdict::Fail; }
    if c1_tokens == c4_tokens { Cb006Verdict::Pass } else { Cb006Verdict::Fail }
}

// ===========================================================================
// CB-007 — No empty outputs: every completed request has ≥1 output token
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_no_empty_outputs(per_request_token_counts: &[u64]) -> Cb007Verdict {
    if per_request_token_counts.is_empty() { return Cb007Verdict::Fail; }
    for &n in per_request_token_counts {
        if n == 0 { return Cb007Verdict::Fail; }
    }
    Cb007Verdict::Pass
}

// ===========================================================================
// CB-008 — No frozen slots: tokens change between consecutive steps for all M
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb008Verdict { Pass, Fail }

/// `step_tokens[step][slot]` is the token emitted at that (step, slot).
/// Pass iff for every slot at least one consecutive (step, step+1) pair
/// produced different tokens — i.e., no slot is frozen at one constant
/// token across all steps.
#[must_use]
pub fn verdict_from_no_frozen_slots(step_tokens: &[Vec<u32>]) -> Cb008Verdict {
    if step_tokens.len() < 2 { return Cb008Verdict::Fail; }
    let m = step_tokens[0].len();
    if m == 0 { return Cb008Verdict::Fail; }
    if step_tokens.iter().any(|row| row.len() != m) { return Cb008Verdict::Fail; }
    for slot in 0..m {
        let mut changed = false;
        for w in step_tokens.windows(2) {
            if w[0][slot] != w[1][slot] { changed = true; break; }
        }
        if !changed { return Cb008Verdict::Fail; }
    }
    Cb008Verdict::Pass
}

// ===========================================================================
// CB-009 — KV cache populated for all M slots after prefill
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cb009Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_kv_lengths_uniform(
    batched_kv_lengths: &[u64],
    expected_prefill_len: u64,
) -> Cb009Verdict {
    if batched_kv_lengths.is_empty() { return Cb009Verdict::Fail; }
    if expected_prefill_len == 0 { return Cb009Verdict::Fail; }
    for &len in batched_kv_lengths {
        if len != expected_prefill_len { return Cb009Verdict::Fail; }
    }
    Cb009Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // CB-001
    #[test] fn cb001_pass_under_budget() { assert_eq!(verdict_from_token_budget(100, 256), Cb001Verdict::Pass); }
    #[test] fn cb001_pass_at_budget() { assert_eq!(verdict_from_token_budget(256, 256), Cb001Verdict::Pass); }
    #[test] fn cb001_fail_over_budget() { assert_eq!(verdict_from_token_budget(257, 256), Cb001Verdict::Fail); }
    #[test] fn cb001_fail_zero_max() { assert_eq!(verdict_from_token_budget(0, 0), Cb001Verdict::Fail); }

    // CB-002
    #[test] fn cb002_pass_increasing() {
        assert_eq!(verdict_from_computed_monotonic(&[10, 20, 30, 40]), Cb002Verdict::Pass);
    }
    #[test] fn cb002_pass_constant() {
        assert_eq!(verdict_from_computed_monotonic(&[10, 10, 10]), Cb002Verdict::Pass);
    }
    #[test] fn cb002_fail_decrease() {
        assert_eq!(verdict_from_computed_monotonic(&[10, 20, 15]), Cb002Verdict::Fail);
    }
    #[test] fn cb002_fail_empty() {
        assert_eq!(verdict_from_computed_monotonic(&[]), Cb002Verdict::Fail);
    }

    // CB-003
    #[test] fn cb003_pass_match() {
        assert_eq!(verdict_from_chunked_prefill_equivalence(b"abc", b"abc"), Cb003Verdict::Pass);
    }
    #[test] fn cb003_fail_drift() {
        assert_eq!(verdict_from_chunked_prefill_equivalence(b"abc", b"abd"), Cb003Verdict::Fail);
    }
    #[test] fn cb003_fail_empty() {
        assert_eq!(verdict_from_chunked_prefill_equivalence(b"", b"abc"), Cb003Verdict::Fail);
    }

    // CB-004
    #[test] fn cb004_pass_at_50pct() {
        assert_eq!(verdict_from_decode_degradation(100.0, 50.0), Cb004Verdict::Pass);
    }
    #[test] fn cb004_pass_above_50pct() {
        assert_eq!(verdict_from_decode_degradation(100.0, 75.0), Cb004Verdict::Pass);
    }
    #[test] fn cb004_fail_below_50pct() {
        assert_eq!(verdict_from_decode_degradation(100.0, 40.0), Cb004Verdict::Fail);
    }
    #[test] fn cb004_fail_zero_baseline() {
        assert_eq!(verdict_from_decode_degradation(0.0, 50.0), Cb004Verdict::Fail);
    }

    // CB-005
    #[test] fn cb005_pass_within_bound() {
        assert_eq!(verdict_from_no_starvation(&[10, 20, 30, 100], 200), Cb005Verdict::Pass);
    }
    #[test] fn cb005_fail_above_bound() {
        assert_eq!(verdict_from_no_starvation(&[10, 250, 30], 200), Cb005Verdict::Fail);
    }
    #[test] fn cb005_fail_empty() {
        assert_eq!(verdict_from_no_starvation(&[], 200), Cb005Verdict::Fail);
    }

    // CB-006
    #[test] fn cb006_pass_match() {
        assert_eq!(verdict_from_batching_correctness(&[1, 2, 3], &[1, 2, 3]), Cb006Verdict::Pass);
    }
    #[test] fn cb006_fail_drift() {
        assert_eq!(verdict_from_batching_correctness(&[1, 2, 3], &[1, 2, 4]), Cb006Verdict::Fail);
    }
    #[test] fn cb006_fail_length_drift() {
        assert_eq!(verdict_from_batching_correctness(&[1, 2], &[1, 2, 3]), Cb006Verdict::Fail);
    }

    // CB-007
    #[test] fn cb007_pass_all_nonempty() {
        assert_eq!(verdict_from_no_empty_outputs(&[10, 5, 20, 1]), Cb007Verdict::Pass);
    }
    #[test] fn cb007_fail_zero_request() {
        assert_eq!(verdict_from_no_empty_outputs(&[10, 0, 20]), Cb007Verdict::Fail);
    }
    #[test] fn cb007_fail_empty_list() {
        assert_eq!(verdict_from_no_empty_outputs(&[]), Cb007Verdict::Fail);
    }

    // CB-008
    #[test] fn cb008_pass_changing_slots() {
        let steps = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
        ];
        assert_eq!(verdict_from_no_frozen_slots(&steps), Cb008Verdict::Pass);
    }
    #[test] fn cb008_fail_frozen_slot_2() {
        let steps = vec![
            vec![1, 2, 99, 4],
            vec![5, 6, 99, 8], // slot 2 stuck at 99
            vec![9, 10, 99, 12],
        ];
        assert_eq!(verdict_from_no_frozen_slots(&steps), Cb008Verdict::Fail);
    }
    #[test] fn cb008_fail_only_one_step() {
        let steps = vec![vec![1, 2, 3]];
        assert_eq!(verdict_from_no_frozen_slots(&steps), Cb008Verdict::Fail);
    }
    #[test] fn cb008_fail_inconsistent_widths() {
        let steps = vec![vec![1, 2, 3], vec![1, 2]];
        assert_eq!(verdict_from_no_frozen_slots(&steps), Cb008Verdict::Fail);
    }

    // CB-009
    #[test] fn cb009_pass_all_match() {
        assert_eq!(verdict_from_kv_lengths_uniform(&[128, 128, 128, 128], 128), Cb009Verdict::Pass);
    }
    #[test] fn cb009_fail_one_short() {
        assert_eq!(verdict_from_kv_lengths_uniform(&[128, 128, 0, 128], 128), Cb009Verdict::Fail);
    }
    #[test] fn cb009_fail_empty() {
        assert_eq!(verdict_from_kv_lengths_uniform(&[], 128), Cb009Verdict::Fail);
    }
    #[test] fn cb009_fail_zero_expected() {
        assert_eq!(verdict_from_kv_lengths_uniform(&[0, 0, 0], 0), Cb009Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_min_ratio() {
        assert!((AC_CB_004_MIN_DEGRADATION_RATIO - 0.50).abs() < 1e-12);
    }
}
