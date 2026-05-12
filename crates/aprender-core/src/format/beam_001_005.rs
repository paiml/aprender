// `batched-beam-search-v1` algorithm-level PARTIAL discharge for the 5
// batched-beam-search falsifiers (output equivalence with sequential,
// top-K consistency, beam=1 == greedy, termination, edge-case property).
//
// Contract: `contracts/batched-beam-search-v1.yaml`.
// Refs: Freitag & Al-Onaizan (2017) Beam Search Strategies, Graves
// (2012) Sequence Transduction §3.1, Radford et al. (2023) Whisper.

/// Tolerance for batched-vs-sequential matmul output comparison.
pub const AC_BEAM_OUTPUT_TOLERANCE: f32 = 1.0e-5;

/// Whisper vocab size (used in canonical N×V test layouts).
pub const AC_BEAM_WHISPER_VOCAB: usize = 51_864;

// =============================================================================
// FALSIFY-BATCH-001 — output equivalence with sequential beam projection
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamOutputEquivalenceVerdict {
    /// max_b max_i |batched[b][i] - sequential[b][i]| < 1e-5.
    Pass,
    /// At least one element exceeds tolerance — FP non-associativity bug.
    Fail,
}

/// Inputs are flattened (n_beams * d_out) row-major output tensors.
#[must_use]
pub fn verdict_from_beam_output_equivalence(
    batched: &[f32],
    sequential: &[f32],
) -> BeamOutputEquivalenceVerdict {
    if batched.len() != sequential.len() {
        return BeamOutputEquivalenceVerdict::Fail;
    }
    if batched.is_empty() {
        return BeamOutputEquivalenceVerdict::Fail;
    }
    for (a, b) in batched.iter().zip(sequential.iter()) {
        if (a - b).abs() >= AC_BEAM_OUTPUT_TOLERANCE {
            return BeamOutputEquivalenceVerdict::Fail;
        }
    }
    BeamOutputEquivalenceVerdict::Pass
}

// =============================================================================
// FALSIFY-BATCH-002 — top-K beam selection matches sequential
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamTopKConsistencyVerdict {
    /// Set of (parent, token) pairs from batched == set from sequential.
    Pass,
    /// At least one pair differs — flattened-index decomposition bug.
    Fail,
}

/// `(parent_beam, token_id)` tuples from each implementation, in any order.
#[must_use]
pub fn verdict_from_beam_topk_consistency(
    batched: &[(u32, u32)],
    sequential: &[(u32, u32)],
) -> BeamTopKConsistencyVerdict {
    use std::collections::HashSet;
    if batched.len() != sequential.len() {
        return BeamTopKConsistencyVerdict::Fail;
    }
    if batched.is_empty() {
        return BeamTopKConsistencyVerdict::Fail;
    }
    let set_a: HashSet<&(u32, u32)> = batched.iter().collect();
    let set_b: HashSet<&(u32, u32)> = sequential.iter().collect();
    if set_a == set_b {
        BeamTopKConsistencyVerdict::Pass
    } else {
        BeamTopKConsistencyVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-BATCH-003 — beam=1 produces same sequence as greedy decode
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamGreedyParityVerdict {
    /// beam_search(input, K=1) token sequence == greedy_decode(input).
    Pass,
    /// Sequences differ — beam-bookkeeping corrupts K=1 degenerate case.
    Fail,
}

#[must_use]
pub fn verdict_from_beam_greedy_parity(
    beam_tokens: &[u32],
    greedy_tokens: &[u32],
) -> BeamGreedyParityVerdict {
    if beam_tokens == greedy_tokens {
        BeamGreedyParityVerdict::Pass
    } else {
        BeamGreedyParityVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-BATCH-004 — termination within max_len
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamTerminationVerdict {
    /// All returned beams have length ≤ max_len AND step counter ≤ max_len.
    Pass,
    /// At least one beam exceeded max_len — termination check broken.
    Fail,
}

#[must_use]
pub fn verdict_from_beam_termination(
    final_step_count: usize,
    max_len: usize,
    returned_beam_lengths: &[usize],
) -> BeamTerminationVerdict {
    if max_len == 0 {
        return BeamTerminationVerdict::Fail;
    }
    if final_step_count > max_len {
        return BeamTerminationVerdict::Fail;
    }
    if returned_beam_lengths.is_empty() {
        // Fallback guarantees at least one beam.
        return BeamTerminationVerdict::Fail;
    }
    for &len in returned_beam_lengths {
        if len > max_len {
            return BeamTerminationVerdict::Fail;
        }
    }
    BeamTerminationVerdict::Pass
}

// =============================================================================
// FALSIFY-BATCH-005 — boundary-condition termination property
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamBoundaryVerdict {
    /// Adversarial boundary inputs (max_len ∈ {1, 10, 100, 448}) all
    /// terminate cleanly with ≥1 returned beam.
    Pass,
    /// Boundary case violated termination invariant.
    Fail,
}

/// Each entry: (max_len_used, did_terminate_cleanly, returned_beams_count).
#[must_use]
pub fn verdict_from_beam_boundary(boundary_runs: &[(usize, bool, usize)]) -> BeamBoundaryVerdict {
    if boundary_runs.is_empty() {
        return BeamBoundaryVerdict::Fail;
    }
    for (_max_len, terminated_clean, beams_returned) in boundary_runs {
        if !*terminated_clean {
            return BeamBoundaryVerdict::Fail;
        }
        if *beams_returned == 0 {
            return BeamBoundaryVerdict::Fail;
        }
    }
    BeamBoundaryVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_output_tolerance_1e_neg5() {
        assert!((AC_BEAM_OUTPUT_TOLERANCE - 1.0e-5).abs() < f32::EPSILON);
    }

    #[test]
    fn provenance_whisper_vocab_51864() {
        assert_eq!(AC_BEAM_WHISPER_VOCAB, 51_864);
    }

    // -------------------------------------------------------------------------
    // Section 2: BATCH-001 output equivalence.
    // -------------------------------------------------------------------------
    #[test]
    fn fb001_pass_exact_match() {
        let a = vec![1.0; 32];
        assert_eq!(
            verdict_from_beam_output_equivalence(&a, &a),
            BeamOutputEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fb001_pass_within_tolerance() {
        let batched = vec![1.0 + 5e-6, 2.0 + 5e-6];
        let sequential = vec![1.0, 2.0];
        assert_eq!(
            verdict_from_beam_output_equivalence(&batched, &sequential),
            BeamOutputEquivalenceVerdict::Pass
        );
    }

    #[test]
    fn fb001_fail_above_tolerance() {
        let batched = vec![1.5];
        let sequential = vec![1.0];
        assert_eq!(
            verdict_from_beam_output_equivalence(&batched, &sequential),
            BeamOutputEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fb001_fail_length_mismatch() {
        assert_eq!(
            verdict_from_beam_output_equivalence(&[1.0], &[1.0, 2.0]),
            BeamOutputEquivalenceVerdict::Fail
        );
    }

    #[test]
    fn fb001_fail_empty() {
        assert_eq!(
            verdict_from_beam_output_equivalence(&[], &[]),
            BeamOutputEquivalenceVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: BATCH-002 top-K consistency.
    // -------------------------------------------------------------------------
    #[test]
    fn fb002_pass_same_set_same_order() {
        let a = [(0u32, 100u32), (1, 200), (2, 300)];
        assert_eq!(
            verdict_from_beam_topk_consistency(&a, &a),
            BeamTopKConsistencyVerdict::Pass
        );
    }

    #[test]
    fn fb002_pass_same_set_different_order() {
        // Set semantics: order doesn't matter.
        let a = [(0u32, 100), (1, 200)];
        let b = [(1u32, 200), (0, 100)];
        assert_eq!(
            verdict_from_beam_topk_consistency(&a, &b),
            BeamTopKConsistencyVerdict::Pass
        );
    }

    #[test]
    fn fb002_fail_different_pairs() {
        let a = [(0u32, 100)];
        let b = [(0u32, 101)]; // off-by-one in token_id
        assert_eq!(
            verdict_from_beam_topk_consistency(&a, &b),
            BeamTopKConsistencyVerdict::Fail
        );
    }

    #[test]
    fn fb002_fail_length_mismatch() {
        let a = [(0u32, 100), (1, 200)];
        let b = [(0u32, 100)];
        assert_eq!(
            verdict_from_beam_topk_consistency(&a, &b),
            BeamTopKConsistencyVerdict::Fail
        );
    }

    #[test]
    fn fb002_fail_empty() {
        let empty: [(u32, u32); 0] = [];
        assert_eq!(
            verdict_from_beam_topk_consistency(&empty, &empty),
            BeamTopKConsistencyVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: BATCH-003 beam=1 == greedy.
    // -------------------------------------------------------------------------
    #[test]
    fn fb003_pass_identical_sequences() {
        let beam = [1u32, 2, 3, 4];
        let greedy = [1u32, 2, 3, 4];
        assert_eq!(
            verdict_from_beam_greedy_parity(&beam, &greedy),
            BeamGreedyParityVerdict::Pass
        );
    }

    #[test]
    fn fb003_fail_different_first_token() {
        let beam = [2u32, 3, 4];
        let greedy = [1u32, 3, 4];
        assert_eq!(
            verdict_from_beam_greedy_parity(&beam, &greedy),
            BeamGreedyParityVerdict::Fail
        );
    }

    #[test]
    fn fb003_fail_different_length() {
        let beam = [1u32, 2, 3];
        let greedy = [1u32, 2, 3, 4];
        assert_eq!(
            verdict_from_beam_greedy_parity(&beam, &greedy),
            BeamGreedyParityVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: BATCH-004 termination.
    // -------------------------------------------------------------------------
    #[test]
    fn fb004_pass_terminated_in_time() {
        assert_eq!(
            verdict_from_beam_termination(50, 100, &[40, 50, 30]),
            BeamTerminationVerdict::Pass
        );
    }

    #[test]
    fn fb004_pass_at_max_len() {
        assert_eq!(
            verdict_from_beam_termination(100, 100, &[100]),
            BeamTerminationVerdict::Pass
        );
    }

    #[test]
    fn fb004_fail_step_count_overrun() {
        assert_eq!(
            verdict_from_beam_termination(101, 100, &[100]),
            BeamTerminationVerdict::Fail
        );
    }

    #[test]
    fn fb004_fail_beam_overrun() {
        assert_eq!(
            verdict_from_beam_termination(100, 100, &[100, 101]),
            BeamTerminationVerdict::Fail
        );
    }

    #[test]
    fn fb004_fail_no_beams_returned() {
        // Fallback should guarantee ≥1 beam.
        assert_eq!(
            verdict_from_beam_termination(50, 100, &[]),
            BeamTerminationVerdict::Fail
        );
    }

    #[test]
    fn fb004_fail_zero_max_len() {
        assert_eq!(
            verdict_from_beam_termination(0, 0, &[0]),
            BeamTerminationVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: BATCH-005 boundary conditions.
    // -------------------------------------------------------------------------
    #[test]
    fn fb005_pass_canonical_boundaries() {
        // max_len ∈ {1, 10, 100, 448} per contract test.
        let runs = [
            (1usize, true, 1usize),
            (10, true, 4),
            (100, true, 4),
            (448, true, 4),
        ];
        assert_eq!(
            verdict_from_beam_boundary(&runs),
            BeamBoundaryVerdict::Pass
        );
    }

    #[test]
    fn fb005_fail_one_boundary_no_terminate() {
        let runs = [(1usize, true, 1usize), (448, false, 0)];
        assert_eq!(verdict_from_beam_boundary(&runs), BeamBoundaryVerdict::Fail);
    }

    #[test]
    fn fb005_fail_one_boundary_no_beams() {
        let runs = [(10usize, true, 0usize)];
        assert_eq!(verdict_from_beam_boundary(&runs), BeamBoundaryVerdict::Fail);
    }

    #[test]
    fn fb005_fail_empty_run_list() {
        let runs: [(usize, bool, usize); 0] = [];
        assert_eq!(verdict_from_beam_boundary(&runs), BeamBoundaryVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy beam search passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_beam_search_passes_all_5() {
        let batched = vec![1.0; 64];
        assert_eq!(
            verdict_from_beam_output_equivalence(&batched, &batched),
            BeamOutputEquivalenceVerdict::Pass
        );
        let topk = [(0u32, 100), (1, 200), (2, 300), (3, 400)];
        assert_eq!(
            verdict_from_beam_topk_consistency(&topk, &topk),
            BeamTopKConsistencyVerdict::Pass
        );
        let seq = [1u32, 2, 3];
        assert_eq!(
            verdict_from_beam_greedy_parity(&seq, &seq),
            BeamGreedyParityVerdict::Pass
        );
        assert_eq!(
            verdict_from_beam_termination(50, 100, &[40, 50, 30, 35]),
            BeamTerminationVerdict::Pass
        );
        let runs = [(1usize, true, 1), (10, true, 4), (100, true, 4), (448, true, 4)];
        assert_eq!(
            verdict_from_beam_boundary(&runs),
            BeamBoundaryVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 001: GEMM accumulation order diverged.
        let batched = vec![1.5];
        let sequential = vec![1.0];
        assert_eq!(
            verdict_from_beam_output_equivalence(&batched, &sequential),
            BeamOutputEquivalenceVerdict::Fail
        );
        // 002: index decomposition off-by-one.
        let a = [(0u32, 100)];
        let b = [(0u32, 101)];
        assert_eq!(
            verdict_from_beam_topk_consistency(&a, &b),
            BeamTopKConsistencyVerdict::Fail
        );
        // 003: K=1 bookkeeping corrupted output.
        let beam = [99u32];
        let greedy = [100u32];
        assert_eq!(
            verdict_from_beam_greedy_parity(&beam, &greedy),
            BeamGreedyParityVerdict::Fail
        );
        // 004: step counter overshot.
        assert_eq!(
            verdict_from_beam_termination(150, 100, &[150]),
            BeamTerminationVerdict::Fail
        );
        // 005: max_len=1 boundary failed to terminate.
        let runs = [(1usize, false, 0)];
        assert_eq!(verdict_from_beam_boundary(&runs), BeamBoundaryVerdict::Fail);
    }
}
