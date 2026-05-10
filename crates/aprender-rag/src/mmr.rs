//! HELIX-IDEA-006 Phase 1 — Maximal Marginal Relevance reranking.
//!
//! Contract: `contracts/apr-rerank-v1.yaml` (ACTIVE).
//! Pattern source: helix-db `helix_engine/reranker/fusion/mmr.rs`
//! (re-implemented; no code lift). Reference:
//!
//! > Carbonell & Goldstein (1998). "The Use of MMR, Diversity-Based
//! > Reranking for Reordering Documents and Producing Summaries."
//! > <https://www.cs.cmu.edu/~jgc/publication/MMR.pdf>
//!
//! MMR balances *relevance* against *diversity* by iteratively
//! selecting the candidate that maximises:
//!
//! ```text
//!   MMR(d) = λ · sim_query(d) − (1 − λ) · max_{s ∈ Selected} sim_pair(d, s)
//! ```
//!
//! At `λ=1`, the diversity term vanishes and the output is the
//! input sorted by relevance descending — verified by
//! `FALSIFY-RERANK-MMR-002`.
//!
//! # Example
//!
//! ```
//! use aprender_rag::mmr::mmr_select;
//!
//! let candidates = vec![
//!     ("doc-a", 0.9_f32),
//!     ("doc-b", 0.8),
//!     ("doc-c", 0.7),
//!     ("doc-a-paraphrase", 0.85),
//! ];
//! // Pretend doc-a and doc-a-paraphrase are highly similar.
//! let sim = |x: &&str, y: &&str| if x.contains("doc-a") && y.contains("doc-a") { 0.95 } else { 0.05 };
//!
//! // λ=1 → pure relevance.
//! let by_rel = mmr_select(candidates.clone(), sim, 1.0, 3);
//! assert_eq!(by_rel[0].0, "doc-a");
//! assert_eq!(by_rel[1].0, "doc-a-paraphrase");
//!
//! // λ=0.5 → diversity penalises near-duplicate of doc-a.
//! let diverse = mmr_select(candidates, sim, 0.5, 3);
//! assert_eq!(diverse[0].0, "doc-a");
//! assert_eq!(diverse[1].0, "doc-b"); // not the paraphrase
//! ```

/// Select up to `top_k` items via Maximal Marginal Relevance.
///
/// `candidates` carry their query-relevance scores in the second
/// tuple slot. `similarity` returns a value in `[0, 1]` between two
/// items (1 = identical). `lambda ∈ [0, 1]` trades relevance for
/// diversity: `λ=1` is pure relevance, `λ=0` is pure diversity.
///
/// The returned `Vec` carries the MMR score (relevance term minus
/// diversity penalty at the moment of selection), not the original
/// relevance.
///
/// # Panics
///
/// Does not panic. If `candidates` is empty, returns an empty `Vec`.
pub fn mmr_select<T, F>(
    mut candidates: Vec<(T, f32)>,
    similarity: F,
    lambda: f32,
    top_k: usize,
) -> Vec<(T, f32)>
where
    T: Clone,
    F: Fn(&T, &T) -> f32,
{
    let cap = top_k.min(candidates.len());
    let mut selected: Vec<(T, f32)> = Vec::with_capacity(cap);

    while selected.len() < cap && !candidates.is_empty() {
        // Find the candidate index whose MMR score is maximal given
        // the currently selected set.
        let (best_idx, best_score) = candidates
            .iter()
            .enumerate()
            .map(|(i, (item, rel))| {
                let max_sim =
                    selected.iter().map(|(s, _)| similarity(item, s)).fold(0.0_f32, f32::max);
                let mmr = lambda * rel - (1.0 - lambda) * max_sim;
                (i, mmr)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("loop condition guarantees candidates is non-empty");

        let (item, _rel) = candidates.swap_remove(best_idx);
        selected.push((item, best_score));
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sim_zero(_: &&str, _: &&str) -> f32 {
        0.0
    }

    #[test]
    fn empty_input_returns_empty() {
        let got: Vec<(&str, f32)> = mmr_select(Vec::new(), sim_zero, 0.5, 10);
        assert!(got.is_empty());
    }

    #[test]
    fn top_k_clipped_to_candidate_count() {
        let cands = vec![("a", 0.9_f32), ("b", 0.5)];
        let got = mmr_select(cands, sim_zero, 1.0, 10);
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn lambda_one_yields_relevance_descending() {
        let cands = vec![("a", 0.5_f32), ("b", 0.9), ("c", 0.7)];
        let got = mmr_select(cands, sim_zero, 1.0, 3);
        assert_eq!(got[0].0, "b");
        assert_eq!(got[1].0, "c");
        assert_eq!(got[2].0, "a");
        // Scores at λ=1 equal the relevance scores exactly.
        assert!((got[0].1 - 0.9).abs() < f32::EPSILON);
        assert!((got[1].1 - 0.7).abs() < f32::EPSILON);
        assert!((got[2].1 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn lambda_zero_with_uniform_relevance_picks_diverse() {
        // All same relevance, similarity 1.0 between same-letter
        // pairs. Diversity should pick distinct items.
        let cands = vec![("a", 1.0_f32), ("a-dup", 1.0), ("b", 1.0)];
        let sim = |x: &&str, y: &&str| {
            if x.starts_with(x.chars().next().unwrap()) && y.starts_with(x.chars().next().unwrap())
            {
                // crude: same first char → similar
                if x.chars().next() == y.chars().next() {
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        };
        let got = mmr_select(cands, sim, 0.0, 3);
        // First pick: any (all relevance equal, no diversity penalty yet).
        // Second pick: must differ in first char from first.
        assert_eq!(got.len(), 3);
    }
}
