// SHIP-TWO-001 — `moe-expert-dispatch-v1` algorithm-level PARTIAL
// discharge for FALSIFY-MOE_EXPERT_DISPATCH_V1_001..002.
//
// Contract: `contracts/moe-expert-dispatch-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Two dispatch gates from Fedus et al. (2022) Switch Transformers:
//
// - MOE-DISPATCH-001 (expert isolation): every expert e only processes
//   tokens routed to e. Concretely: `dispatched[e]` is a subset of
//   `selected[t]` ⇒ {t : e ∈ selected[t]}.
// - MOE-DISPATCH-002 (weighted aggregation):
//   output[t] = Σ_{e ∈ selected[t]} weight[t,e] * expert_output[t,e]
//   matches a reference scalar implementation within tolerance.
//
// In-module reference: `expert_dispatch` returns per-expert token
// lists; `weighted_aggregate` does the post-FFN combine.

/// ULP tolerance for weighted aggregation parity.
pub const AC_MOE_DISPATCH_002_AGG_TOLERANCE: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeDispatchVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference dispatch.
// -----------------------------------------------------------------------------

/// Dispatch tokens to experts. Given `selected[t]` = the list of expert
/// indices selected for token t, returns `dispatched[e]` = list of
/// (token_idx, weight_idx) tuples for expert e, where weight_idx is
/// the position of e within selected[t] (so the caller can find the
/// weight in selected_weights[t][weight_idx]).
#[must_use]
pub fn expert_dispatch(
    selected: &[Vec<usize>],
    n_experts: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut dispatched: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_experts];
    for (t, sel) in selected.iter().enumerate() {
        for (k_idx, &e) in sel.iter().enumerate() {
            if e < n_experts {
                dispatched[e].push((t, k_idx));
            }
        }
    }
    dispatched
}

/// Reference weighted aggregation:
/// output[t][d] = Σ_{e ∈ selected[t]} weight[t][e] * expert_output[t][e][d]
///
/// `expert_outputs[t][k][d]` is the d-th hidden dim of the FFN output
/// of the k-th expert chosen for token t. `selected_weights[t][k]` is
/// the renormalized weight for that expert.
///
/// Returns flat row-major `[n_tokens, hidden_dim]`.
#[must_use]
pub fn weighted_aggregate(
    expert_outputs: &[Vec<Vec<f32>>],
    selected_weights: &[Vec<f32>],
    hidden_dim: usize,
) -> Vec<f32> {
    let n_tokens = expert_outputs.len();
    let mut out = vec![0.0_f32; n_tokens * hidden_dim];
    for t in 0..n_tokens {
        let k = expert_outputs[t].len();
        for k_idx in 0..k {
            let w = selected_weights[t][k_idx];
            for d in 0..hidden_dim {
                out[t * hidden_dim + d] += w * expert_outputs[t][k_idx][d];
            }
        }
    }
    out
}

// -----------------------------------------------------------------------------
// Verdict 1: MOE-DISPATCH-001 — expert isolation.
// -----------------------------------------------------------------------------

/// Pass iff every (token, k_idx) in `dispatched[e]` corresponds to
/// `selected[token][k_idx] == e`.
///
/// In other words: each expert sees exactly the tokens routed to it,
/// nothing more, nothing less.
#[must_use]
pub fn verdict_from_expert_isolation(
    dispatched: &[Vec<(usize, usize)>],
    selected: &[Vec<usize>],
) -> MoeDispatchVerdict {
    let n_experts = dispatched.len();
    if n_experts == 0 {
        return MoeDispatchVerdict::Fail;
    }

    // Build the "expected" multiset from `selected`.
    let mut expected: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_experts];
    for (t, sel) in selected.iter().enumerate() {
        for (k_idx, &e) in sel.iter().enumerate() {
            if e >= n_experts {
                return MoeDispatchVerdict::Fail;
            }
            expected[e].push((t, k_idx));
        }
    }

    // Compare per-expert sorted lists for set-equality (allowing
    // any internal order in dispatched).
    for e in 0..n_experts {
        let mut a = dispatched[e].clone();
        let mut b = expected[e].clone();
        a.sort_unstable();
        b.sort_unstable();
        if a != b {
            return MoeDispatchVerdict::Fail;
        }
    }
    MoeDispatchVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: MOE-DISPATCH-002 — weighted aggregation.
// -----------------------------------------------------------------------------

/// Pass iff `actual` and `expected` agree elementwise within
/// `AC_MOE_DISPATCH_002_AGG_TOLERANCE`.
#[must_use]
pub fn verdict_from_weighted_aggregation(actual: &[f32], expected: &[f32]) -> MoeDispatchVerdict {
    if actual.len() != expected.len() {
        return MoeDispatchVerdict::Fail;
    }
    if actual.is_empty() {
        return MoeDispatchVerdict::Fail;
    }
    for (a, e) in actual.iter().zip(expected.iter()) {
        if !a.is_finite() || !e.is_finite() {
            return MoeDispatchVerdict::Fail;
        }
        if (a - e).abs() >= AC_MOE_DISPATCH_002_AGG_TOLERANCE {
            return MoeDispatchVerdict::Fail;
        }
    }
    MoeDispatchVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_agg_tolerance_1e_5() {
        assert_eq!(AC_MOE_DISPATCH_002_AGG_TOLERANCE, 1e-5);
    }

    // -------------------------------------------------------------------------
    // Section 2: MOE-DISPATCH-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn moed001_pass_simple_dispatch() {
        // 3 tokens, 4 experts, top-2.
        // token 0 → experts {1, 0}
        // token 1 → experts {2, 3}
        // token 2 → experts {1, 3}
        let selected = vec![vec![1_usize, 0], vec![2, 3], vec![1, 3]];
        let dispatched = expert_dispatch(&selected, 4);
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Pass
        );
    }

    #[test]
    fn moed001_pass_single_token_single_expert() {
        let selected = vec![vec![0_usize]];
        let dispatched = expert_dispatch(&selected, 4);
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Pass
        );
    }

    #[test]
    fn moed001_pass_unused_expert() {
        // Expert 3 never selected; its dispatched list must be empty.
        let selected = vec![vec![0_usize, 1], vec![0, 2]];
        let dispatched = expert_dispatch(&selected, 4);
        assert!(dispatched[3].is_empty());
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: MOE-DISPATCH-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn moed001_fail_extra_token_routed_to_expert() {
        let selected = vec![vec![1_usize, 0]];
        let mut dispatched = expert_dispatch(&selected, 4);
        // Inject an extra (token=99, k_idx=0) into expert 2 — token 99
        // never selected expert 2.
        dispatched[2].push((99, 0));
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn moed001_fail_missing_token_from_expert() {
        let selected = vec![vec![1_usize, 0], vec![1, 2]];
        let mut dispatched = expert_dispatch(&selected, 4);
        // Drop one token from expert 1 — it should have 2 entries.
        dispatched[1].pop();
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn moed001_fail_expert_index_out_of_range() {
        // Selected has expert id 99 but we declared n_experts=4.
        let selected = vec![vec![99_usize, 0]];
        let dispatched = expert_dispatch(&selected, 4);
        // expert_dispatch silently drops e >= n_experts; isolation
        // verdict catches this via the comparison loop.
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn moed001_fail_empty_dispatched() {
        // n_experts=0 is illegal.
        let selected: Vec<Vec<usize>> = vec![vec![0_usize]];
        let dispatched: Vec<Vec<(usize, usize)>> = vec![];
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn moed001_fail_wrong_token_index() {
        let selected = vec![vec![0_usize], vec![0]];
        let mut dispatched = expert_dispatch(&selected, 2);
        // expert 0 should have entries [(0, 0), (1, 0)]; corrupt one:
        dispatched[0][0].0 = 5; // wrong token id
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: MOE-DISPATCH-002 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn moed002_pass_identical() {
        let actual = vec![1.0_f32, 2.0, 3.0];
        let expected = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Pass
        );
    }

    #[test]
    fn moed002_pass_within_tolerance() {
        let actual = vec![1.0_f32, 2.000005];
        let expected = vec![1.0_f32, 2.0];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Pass
        );
    }

    #[test]
    fn moed002_pass_realistic_aggregation() {
        // 1 token, 2 experts, hidden_dim=3.
        // selected[0] = [0, 1], weights = [0.6, 0.4]
        // expert_outputs[0][0] = [1, 2, 3], [0][1] = [4, 5, 6]
        // expected output[0] = 0.6*[1,2,3] + 0.4*[4,5,6] = [2.2, 3.2, 4.2]
        let expert_outputs = vec![vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]]];
        let selected_weights = vec![vec![0.6_f32, 0.4]];
        let actual = weighted_aggregate(&expert_outputs, &selected_weights, 3);

        let expected = vec![2.2_f32, 3.2, 4.2];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: MOE-DISPATCH-002 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn moed002_fail_above_tolerance() {
        let actual = vec![1.0_f32, 2.001];
        let expected = vec![1.0_f32, 2.0];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn moed002_fail_length_mismatch() {
        let actual = vec![1.0_f32, 2.0];
        let expected = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn moed002_fail_empty_both() {
        let v: Vec<f32> = vec![];
        assert_eq!(
            verdict_from_weighted_aggregation(&v, &v),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn moed002_fail_nan_actual() {
        let actual = vec![f32::NAN];
        let expected = vec![1.0_f32];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn moed002_fail_inf_expected() {
        let actual = vec![1.0_f32];
        let expected = vec![f32::INFINITY];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Domain — reference functions.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_expert_dispatch_correct_tuples() {
        let selected = vec![vec![1_usize, 0], vec![1, 2]];
        let dispatched = expert_dispatch(&selected, 3);
        // Expert 0: token 0 picked it at k_idx=1.
        assert_eq!(dispatched[0], vec![(0, 1)]);
        // Expert 1: tokens 0, 1 each picked it at k_idx=0.
        assert_eq!(dispatched[1], vec![(0, 0), (1, 0)]);
        // Expert 2: token 1 picked it at k_idx=1.
        assert_eq!(dispatched[2], vec![(1, 1)]);
    }

    #[test]
    fn domain_weighted_aggregate_zero_weights_zero_output() {
        let expert_outputs = vec![vec![vec![1.0_f32], vec![2.0]]];
        let selected_weights = vec![vec![0.0_f32, 0.0]];
        let out = weighted_aggregate(&expert_outputs, &selected_weights, 1);
        assert_eq!(out, vec![0.0_f32]);
    }

    #[test]
    fn domain_weighted_aggregate_single_expert_unweighted() {
        // weight=1.0 selects single expert ⇒ output = expert_output.
        let expert_outputs = vec![vec![vec![5.0_f32, 6.0]]];
        let selected_weights = vec![vec![1.0_f32]];
        let out = weighted_aggregate(&expert_outputs, &selected_weights, 2);
        assert_eq!(out, vec![5.0_f32, 6.0]);
    }

    #[test]
    fn domain_dispatch_then_aggregate_round_trip() {
        // 2 tokens, 3 experts, top-2; verify dispatch + aggregate
        // commute cleanly.
        let selected = vec![vec![0_usize, 1], vec![1, 2]];
        let n_experts = 3;
        let dispatched = expert_dispatch(&selected, n_experts);
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Pass
        );

        // Build expert_outputs/selected_weights from the dispatch:
        // (synthetic — each expert outputs constant-ish vectors)
        let weights = vec![vec![0.5_f32, 0.5], vec![0.6, 0.4]];
        let outputs = vec![
            vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0_f32, 1.0], vec![2.0, 2.0]],
        ];
        let actual = weighted_aggregate(&outputs, &weights, 2);

        // Manually compute expected:
        // token 0: 0.5*[1,0] + 0.5*[0,1] = [0.5, 0.5]
        // token 1: 0.6*[1,1] + 0.4*[2,2] = [1.4, 1.4]
        let expected = vec![0.5_f32, 0.5, 1.4, 1.4];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Sweep — n_experts and top-k combinations.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_isolation_passes_for_various_topologies() {
        // Try (n_tokens, n_experts, k) combinations.
        let test_cases = [(3_usize, 4, 1), (3, 4, 2), (5, 8, 2), (10, 16, 4)];
        for (nt, ne, k) in test_cases {
            let mut selected = Vec::new();
            for t in 0..nt {
                let row: Vec<usize> = (0..k).map(|i| (t + i) % ne).collect();
                selected.push(row);
            }
            let dispatched = expert_dispatch(&selected, ne);
            assert_eq!(
                verdict_from_expert_isolation(&dispatched, &selected),
                MoeDispatchVerdict::Pass,
                "nt={nt} ne={ne} k={k}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 8: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_isolation_violation_caught() {
        // Bug regression: an expert receives a token that wasn't routed
        // to it. Catch via verdict.
        let selected = vec![vec![0_usize, 1]];
        let mut dispatched = expert_dispatch(&selected, 4);
        dispatched[2].push((0, 0)); // Bug: token 0 wasn't routed to expert 2.
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn realistic_aggregation_off_by_factor() {
        // Bug regression: weights applied incorrectly (e.g., sum uses
        // unrenormalized weights instead of renormalized).
        let actual = vec![1.5_f32]; // bug: 1.5x correct
        let expected = vec![1.0_f32];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Fail
        );
    }

    #[test]
    fn realistic_full_dispatch_pipeline() {
        // 2 tokens, 4 experts, top-2 — full pipeline.
        let selected = vec![vec![1_usize, 0], vec![2, 3]];
        let weights = vec![vec![0.7_f32, 0.3], vec![0.5, 0.5]];

        // Synthetic expert outputs per chosen-expert per token, dim=2.
        let expert_outputs = vec![
            // token 0: expert 1 output, then expert 0 output
            vec![vec![10.0_f32, 0.0], vec![0.0, 10.0]],
            // token 1: expert 2 output, then expert 3 output
            vec![vec![1.0_f32, 1.0], vec![2.0, 2.0]],
        ];

        let dispatched = expert_dispatch(&selected, 4);
        assert_eq!(
            verdict_from_expert_isolation(&dispatched, &selected),
            MoeDispatchVerdict::Pass
        );

        let actual = weighted_aggregate(&expert_outputs, &weights, 2);
        // token 0: 0.7*[10,0] + 0.3*[0,10] = [7, 3]
        // token 1: 0.5*[1,1] + 0.5*[2,2] = [1.5, 1.5]
        let expected = vec![7.0_f32, 3.0, 1.5, 1.5];
        assert_eq!(
            verdict_from_weighted_aggregation(&actual, &expected),
            MoeDispatchVerdict::Pass
        );
    }
}
