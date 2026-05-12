// `context-generation-v1` algorithm-level PARTIAL discharge for the 2
// RAG context-generation falsifiers (relevance ordering, context budget).
//
// Contract: `contracts/context-generation-v1.yaml`.

// =============================================================================
// FALSIFY-CONTEXT_GENERATION_V1_001 — relevance descending order
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtxRelevanceOrderingVerdict {
    /// retrieved_docs.scores sorted in descending order.
    Pass,
    /// At least one adjacent pair is out of order.
    Fail,
}

#[must_use]
pub fn verdict_from_ctx_relevance_ordering(scores: &[f32]) -> CtxRelevanceOrderingVerdict {
    if scores.is_empty() {
        return CtxRelevanceOrderingVerdict::Fail;
    }
    for window in scores.windows(2) {
        let (a, b) = (window[0], window[1]);
        if !a.is_finite() || !b.is_finite() {
            return CtxRelevanceOrderingVerdict::Fail;
        }
        if a < b {
            return CtxRelevanceOrderingVerdict::Fail;
        }
    }
    CtxRelevanceOrderingVerdict::Pass
}

// =============================================================================
// FALSIFY-CONTEXT_GENERATION_V1_002 — context budget
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CtxBudgetVerdict {
    /// Σ tokens(doc_i) ≤ max_context_length.
    Pass,
    /// Tokens exceed budget — context window overflow.
    Fail,
}

#[must_use]
pub fn verdict_from_ctx_budget(per_doc_tokens: &[u32], max_context_length: u32) -> CtxBudgetVerdict {
    if max_context_length == 0 {
        return CtxBudgetVerdict::Fail;
    }
    let total: u64 = per_doc_tokens.iter().map(|&t| t as u64).sum();
    if total <= max_context_length as u64 {
        CtxBudgetVerdict::Pass
    } else {
        CtxBudgetVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: CTX-001 relevance ordering.
    // -------------------------------------------------------------------------
    #[test]
    fn fctx001_pass_descending() {
        let s = vec![0.99_f32, 0.85, 0.42, 0.10];
        assert_eq!(
            verdict_from_ctx_relevance_ordering(&s),
            CtxRelevanceOrderingVerdict::Pass
        );
    }

    #[test]
    fn fctx001_pass_equal_adjacent() {
        // Ties are descending-or-equal; PASS per "sorted by descending".
        let s = vec![0.5_f32, 0.5, 0.3];
        assert_eq!(
            verdict_from_ctx_relevance_ordering(&s),
            CtxRelevanceOrderingVerdict::Pass
        );
    }

    #[test]
    fn fctx001_pass_single() {
        let s = vec![0.7_f32];
        assert_eq!(
            verdict_from_ctx_relevance_ordering(&s),
            CtxRelevanceOrderingVerdict::Pass
        );
    }

    #[test]
    fn fctx001_fail_ascending_pair() {
        let s = vec![0.5_f32, 0.7];
        assert_eq!(
            verdict_from_ctx_relevance_ordering(&s),
            CtxRelevanceOrderingVerdict::Fail
        );
    }

    #[test]
    fn fctx001_fail_one_inversion() {
        let s = vec![0.9_f32, 0.8, 0.85, 0.5];
        assert_eq!(
            verdict_from_ctx_relevance_ordering(&s),
            CtxRelevanceOrderingVerdict::Fail
        );
    }

    #[test]
    fn fctx001_fail_nan() {
        let s = vec![0.9_f32, f32::NAN];
        assert_eq!(
            verdict_from_ctx_relevance_ordering(&s),
            CtxRelevanceOrderingVerdict::Fail
        );
    }

    #[test]
    fn fctx001_fail_empty() {
        assert_eq!(
            verdict_from_ctx_relevance_ordering(&[]),
            CtxRelevanceOrderingVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: CTX-002 context budget.
    // -------------------------------------------------------------------------
    #[test]
    fn fctx002_pass_under_budget() {
        let t = vec![100u32, 200, 300, 400];
        assert_eq!(verdict_from_ctx_budget(&t, 4096), CtxBudgetVerdict::Pass);
    }

    #[test]
    fn fctx002_pass_at_budget() {
        let t = vec![1024u32, 1024, 1024, 1024];
        assert_eq!(verdict_from_ctx_budget(&t, 4096), CtxBudgetVerdict::Pass);
    }

    #[test]
    fn fctx002_pass_empty_zero_total() {
        let t: Vec<u32> = vec![];
        assert_eq!(verdict_from_ctx_budget(&t, 4096), CtxBudgetVerdict::Pass);
    }

    #[test]
    fn fctx002_fail_over_budget() {
        let t = vec![2048u32, 2048, 1];
        assert_eq!(verdict_from_ctx_budget(&t, 4096), CtxBudgetVerdict::Fail);
    }

    #[test]
    fn fctx002_fail_zero_max() {
        let t = vec![1u32];
        assert_eq!(verdict_from_ctx_budget(&t, 0), CtxBudgetVerdict::Fail);
    }

    #[test]
    fn fctx002_pass_large_budget_qwen3_30b() {
        // Qwen3-Coder-30B max_context = 256k, single 100k doc.
        let t = vec![100_000u32];
        assert_eq!(verdict_from_ctx_budget(&t, 262_144), CtxBudgetVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Realistic — full healthy RAG passes both.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_rag_passes_both() {
        // 5 docs, monotonically descending relevance, total 4000 tokens.
        let scores = vec![0.95_f32, 0.87, 0.62, 0.41, 0.18];
        assert_eq!(
            verdict_from_ctx_relevance_ordering(&scores),
            CtxRelevanceOrderingVerdict::Pass
        );
        let tokens = vec![800u32, 1000, 800, 600, 800];
        assert_eq!(verdict_from_ctx_budget(&tokens, 4096), CtxBudgetVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_both_failures() {
        // Bug class 001: ranking pipeline emitted in ascending order.
        let bad_order = vec![0.1_f32, 0.5, 0.95];
        assert_eq!(
            verdict_from_ctx_relevance_ordering(&bad_order),
            CtxRelevanceOrderingVerdict::Fail
        );
        // Bug class 002: token-budget guard skipped → 8192 tokens emitted into
        // a 4096-token context window.
        let bad_budget = vec![4096u32, 4096];
        assert_eq!(verdict_from_ctx_budget(&bad_budget, 4096), CtxBudgetVerdict::Fail);
    }
}
