// Bundles two sister contracts in one verdict module:
//
//   `tied-embeddings-v1` (FALSIFY-TE-001..004)
//   `work-dbc-v1` (FALSIFY-DBC-001..005)
//
// TE-001: tied LM head output shape == (seq_len, vocab_size)
// TE-002: tied head output bit-exact equal to explicit matmul(x, W^T)
// TE-003: tied head adds zero learnable parameters
// TE-004: all logits finite when inputs are finite
// DBC-001: only forward state transitions allowed
// DBC-002: require clauses block InProgress when preconditions fail
// DBC-003: ensure clauses block Completed when postconditions fail
// DBC-004: falsify is non-destructive — state unchanged
// DBC-005: rescue attempts bounded (≤ 3 retries before "limit reached")

/// DBC-005: max rescue attempts before "limit reached".
pub const AC_DBC_MAX_RESCUE_ATTEMPTS: u32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeDbcVerdict {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkState {
    Pending,
    InProgress,
    Completed,
    Failed,
}

// ----------------------------------------------------------------
// TE-001
// ----------------------------------------------------------------

/// TE-001: tied LM head output shape == (seq_len, vocab_size).
#[must_use]
pub fn verdict_from_tied_output_shape(
    seq_len: usize,
    vocab_size: usize,
    actual_rows: usize,
    actual_cols: usize,
) -> TeDbcVerdict {
    if seq_len == 0 || vocab_size == 0 {
        return TeDbcVerdict::Fail;
    }
    if actual_rows == seq_len && actual_cols == vocab_size {
        TeDbcVerdict::Pass
    } else {
        TeDbcVerdict::Fail
    }
}

// ----------------------------------------------------------------
// TE-002
// ----------------------------------------------------------------

/// TE-002: tied output bit-exact equal to explicit matmul(x, W^T).
#[must_use]
pub fn verdict_from_tied_matmul_equivalence(
    tied_output: &[f32],
    explicit_output: &[f32],
) -> TeDbcVerdict {
    if tied_output.is_empty() || tied_output.len() != explicit_output.len() {
        return TeDbcVerdict::Fail;
    }
    for (a, b) in tied_output.iter().zip(explicit_output.iter()) {
        if a.to_bits() != b.to_bits() {
            return TeDbcVerdict::Fail;
        }
    }
    TeDbcVerdict::Pass
}

// ----------------------------------------------------------------
// TE-003
// ----------------------------------------------------------------

/// TE-003: tied head adds zero learnable params.
#[must_use]
pub fn verdict_from_tied_no_extra_params(
    params_before: u64,
    params_after: u64,
) -> TeDbcVerdict {
    if params_after == params_before {
        TeDbcVerdict::Pass
    } else {
        TeDbcVerdict::Fail
    }
}

// ----------------------------------------------------------------
// TE-004
// ----------------------------------------------------------------

/// TE-004: every output logit is finite given finite inputs.
#[must_use]
pub fn verdict_from_tied_finite_output(logits: &[f32]) -> TeDbcVerdict {
    if logits.is_empty() {
        return TeDbcVerdict::Fail;
    }
    if logits.iter().all(|x| x.is_finite()) {
        TeDbcVerdict::Pass
    } else {
        TeDbcVerdict::Fail
    }
}

// ----------------------------------------------------------------
// DBC-001
// ----------------------------------------------------------------

/// DBC-001: only forward transitions allowed.
///
/// Forward transitions:
///   Pending → InProgress
///   InProgress → Completed
///   InProgress → Failed
///   Failed → InProgress (rescue retry — counts as forward in this lifecycle)
/// All other transitions are backward and must be rejected.
#[must_use]
pub fn verdict_from_only_forward_transition(
    from: WorkState,
    to: WorkState,
    transition_was_rejected: bool,
) -> TeDbcVerdict {
    let is_forward = matches!(
        (from, to),
        (WorkState::Pending | WorkState::Failed, WorkState::InProgress)
            | (WorkState::InProgress, WorkState::Completed | WorkState::Failed)
    );
    if is_forward != transition_was_rejected {
        TeDbcVerdict::Pass
    } else {
        TeDbcVerdict::Fail
    }
}

// ----------------------------------------------------------------
// DBC-002 + DBC-003 — require / ensure
// ----------------------------------------------------------------

/// DBC-002: missing precondition (e.g. Cargo.toml absent) blocks InProgress.
#[must_use]
pub fn verdict_from_require_blocks(
    preconditions_satisfied: bool,
    transition_was_rejected: bool,
) -> TeDbcVerdict {
    if !preconditions_satisfied && !transition_was_rejected {
        return TeDbcVerdict::Fail;
    }
    if preconditions_satisfied && transition_was_rejected {
        // Bug: rejected even though preconditions OK.
        return TeDbcVerdict::Fail;
    }
    TeDbcVerdict::Pass
}

/// DBC-003: missing postcondition (e.g. uncommitted changes) blocks Completed.
#[must_use]
pub fn verdict_from_ensure_blocks(
    postconditions_satisfied: bool,
    completion_was_rejected: bool,
) -> TeDbcVerdict {
    if !postconditions_satisfied && !completion_was_rejected {
        return TeDbcVerdict::Fail;
    }
    if postconditions_satisfied && completion_was_rejected {
        return TeDbcVerdict::Fail;
    }
    TeDbcVerdict::Pass
}

// ----------------------------------------------------------------
// DBC-004
// ----------------------------------------------------------------

/// DBC-004: falsify does not modify state.
#[must_use]
pub fn verdict_from_falsify_nondestructive(
    state_before: WorkState,
    state_after: WorkState,
) -> TeDbcVerdict {
    if state_before == state_after {
        TeDbcVerdict::Pass
    } else {
        TeDbcVerdict::Fail
    }
}

// ----------------------------------------------------------------
// DBC-005
// ----------------------------------------------------------------

/// DBC-005: rescue attempts bounded — ≤ 3 retries before "limit reached".
#[must_use]
pub fn verdict_from_rescue_bounded(
    attempt_index: u32,
    error_contains_limit_reached: bool,
) -> TeDbcVerdict {
    if attempt_index < AC_DBC_MAX_RESCUE_ATTEMPTS {
        // First 3 attempts must NOT yield "limit reached"
        if !error_contains_limit_reached {
            TeDbcVerdict::Pass
        } else {
            TeDbcVerdict::Fail
        }
    } else if error_contains_limit_reached {
        // 4th+ attempts MUST yield "limit reached"
        TeDbcVerdict::Pass
    } else {
        TeDbcVerdict::Fail
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
        assert_eq!(AC_DBC_MAX_RESCUE_ATTEMPTS, 3);
    }

    // -----------------------------------------------------------------
    // Section 2: TE-001..004 tied embeddings.
    // -----------------------------------------------------------------
    #[test]
    fn fte001_pass_canonical_shape() {
        let v = verdict_from_tied_output_shape(128, 32_000, 128, 32_000);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fte001_fail_dim_mismatch() {
        let v = verdict_from_tied_output_shape(128, 32_000, 128, 256);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fte001_fail_zero_seq_len() {
        let v = verdict_from_tied_output_shape(0, 32_000, 0, 32_000);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fte002_pass_bit_exact() {
        let tied = vec![1.0_f32, 2.5, -3.0];
        let explicit = vec![1.0_f32, 2.5, -3.0];
        let v = verdict_from_tied_matmul_equivalence(&tied, &explicit);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fte002_fail_one_ulp() {
        let bumped = f32::from_bits(2.5_f32.to_bits() + 1);
        let v = verdict_from_tied_matmul_equivalence(&[1.0, 2.5], &[1.0, bumped]);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fte002_fail_length_mismatch() {
        let v = verdict_from_tied_matmul_equivalence(&[1.0, 2.5], &[1.0, 2.5, 3.0]);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fte003_pass_no_extra_params() {
        let v = verdict_from_tied_no_extra_params(1_000_000, 1_000_000);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fte003_fail_extra_projection_weight() {
        // The regression class — a separate projection weight allocated.
        let v = verdict_from_tied_no_extra_params(1_000_000, 1_896_000);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fte004_pass_finite_logits() {
        let v = verdict_from_tied_finite_output(&[1.0, -2.0, 3.0]);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fte004_fail_nan() {
        let v = verdict_from_tied_finite_output(&[1.0, f32::NAN]);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fte004_fail_infinity() {
        let v = verdict_from_tied_finite_output(&[1.0, f32::INFINITY]);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: DBC-001 forward transitions.
    // -----------------------------------------------------------------
    #[test]
    fn fdbc001_pass_pending_to_inprogress() {
        let v = verdict_from_only_forward_transition(
            WorkState::Pending,
            WorkState::InProgress,
            false,
        );
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fdbc001_pass_inprogress_to_completed() {
        let v = verdict_from_only_forward_transition(
            WorkState::InProgress,
            WorkState::Completed,
            false,
        );
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fdbc001_pass_completed_to_inprogress_rejected() {
        // Backward transition — must be rejected.
        let v = verdict_from_only_forward_transition(
            WorkState::Completed,
            WorkState::InProgress,
            true,
        );
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fdbc001_fail_completed_to_inprogress_accepted() {
        // The exact regression class — restart from terminal state.
        let v = verdict_from_only_forward_transition(
            WorkState::Completed,
            WorkState::InProgress,
            false,
        );
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fdbc001_fail_forward_rejected() {
        // Forward transition wrongly rejected.
        let v = verdict_from_only_forward_transition(
            WorkState::Pending,
            WorkState::InProgress,
            true,
        );
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: DBC-002 + 003 require/ensure.
    // -----------------------------------------------------------------
    #[test]
    fn fdbc002_pass_missing_cargo_rejected() {
        let v = verdict_from_require_blocks(false, true);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fdbc002_pass_satisfied_accepted() {
        let v = verdict_from_require_blocks(true, false);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fdbc002_fail_missing_cargo_accepted() {
        let v = verdict_from_require_blocks(false, false);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fdbc003_pass_dirty_git_rejected() {
        let v = verdict_from_ensure_blocks(false, true);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fdbc003_fail_dirty_git_accepted() {
        let v = verdict_from_ensure_blocks(false, false);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: DBC-004 + 005.
    // -----------------------------------------------------------------
    #[test]
    fn fdbc004_pass_state_unchanged() {
        let v = verdict_from_falsify_nondestructive(WorkState::InProgress, WorkState::InProgress);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fdbc004_fail_state_changed() {
        let v = verdict_from_falsify_nondestructive(WorkState::InProgress, WorkState::Failed);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fdbc005_pass_attempt_2_no_limit() {
        let v = verdict_from_rescue_bounded(2, false);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fdbc005_pass_attempt_3_yields_limit() {
        // index=3 is the 4th attempt (0-indexed) → must yield "limit reached"
        let v = verdict_from_rescue_bounded(3, true);
        assert_eq!(v, TeDbcVerdict::Pass);
    }

    #[test]
    fn fdbc005_fail_attempt_5_no_limit() {
        // The unbounded-retry regression class.
        let v = verdict_from_rescue_bounded(5, false);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    #[test]
    fn fdbc005_fail_attempt_1_premature_limit() {
        let v = verdict_from_rescue_bounded(1, true);
        assert_eq!(v, TeDbcVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: Mutation surveys.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_dbc001_transition_table() {
        // Every (from, to) pair, with rejection matching forward-ness.
        let states = [
            WorkState::Pending,
            WorkState::InProgress,
            WorkState::Completed,
            WorkState::Failed,
        ];
        for &f in &states {
            for &t in &states {
                if f == t {
                    continue;
                }
                let is_forward = matches!(
                    (f, t),
                    (WorkState::Pending, WorkState::InProgress)
                        | (WorkState::InProgress, WorkState::Completed)
                        | (WorkState::InProgress, WorkState::Failed)
                        | (WorkState::Failed, WorkState::InProgress)
                );
                // Correct rejection
                let v = verdict_from_only_forward_transition(f, t, !is_forward);
                assert_eq!(v, TeDbcVerdict::Pass, "({f:?} → {t:?})");
            }
        }
    }

    #[test]
    fn mutation_survey_dbc005_attempt_band() {
        for i in 0_u32..6 {
            let yields_limit = i >= 3;
            let v = verdict_from_rescue_bounded(i, yields_limit);
            assert_eq!(v, TeDbcVerdict::Pass, "attempt={i}");
        }
    }

    // -----------------------------------------------------------------
    // Section 7: Realistic.
    // -----------------------------------------------------------------
    #[test]
    fn realistic_healthy_passes_all_9() {
        let v1 = verdict_from_tied_output_shape(128, 32_000, 128, 32_000);
        let v2 = verdict_from_tied_matmul_equivalence(&[1.0, 2.5], &[1.0, 2.5]);
        let v3 = verdict_from_tied_no_extra_params(1_000_000, 1_000_000);
        let v4 = verdict_from_tied_finite_output(&[1.0, -2.0, 3.0]);
        let v5 = verdict_from_only_forward_transition(
            WorkState::Pending,
            WorkState::InProgress,
            false,
        );
        let v6 = verdict_from_require_blocks(false, true);
        let v7 = verdict_from_ensure_blocks(false, true);
        let v8 = verdict_from_falsify_nondestructive(WorkState::InProgress, WorkState::InProgress);
        let v9 = verdict_from_rescue_bounded(2, false);
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9] {
            assert_eq!(v, TeDbcVerdict::Pass);
        }
    }

    #[test]
    fn realistic_pre_fix_all_9_failures() {
        // 9 simultaneous regressions:
        //   1: tied head produced wrong shape
        //   2: tied head bit-different from explicit
        //   3: tied head allocated extra projection
        //   4: NaN propagation in matmul
        //   5: terminal state restart accepted
        //   6: missing Cargo.toml didn't block InProgress
        //   7: dirty git tree accepted Completed
        //   8: falsify modified state
        //   9: 5th rescue attempt didn't yield "limit reached"
        let v1 = verdict_from_tied_output_shape(128, 32_000, 128, 256);
        let v2 = verdict_from_tied_matmul_equivalence(&[1.0, 2.5], &[1.0, 99.0]);
        let v3 = verdict_from_tied_no_extra_params(1_000_000, 2_000_000);
        let v4 = verdict_from_tied_finite_output(&[1.0, f32::NAN]);
        let v5 = verdict_from_only_forward_transition(
            WorkState::Completed,
            WorkState::InProgress,
            false,
        );
        let v6 = verdict_from_require_blocks(false, false);
        let v7 = verdict_from_ensure_blocks(false, false);
        let v8 = verdict_from_falsify_nondestructive(WorkState::InProgress, WorkState::Failed);
        let v9 = verdict_from_rescue_bounded(5, false);
        for v in [v1, v2, v3, v4, v5, v6, v7, v8, v9] {
            assert_eq!(v, TeDbcVerdict::Fail);
        }
    }
}
