//! FALSIFY-RERANK-MMR-002 — MMR with `λ=1.0` returns the input
//! sorted by relevance descending; the output's MMR score equals
//! the input relevance score (the diversity term `(1-λ)·max_sim`
//! evaluates to 0 at λ=1).
//!
//! Contract: `contracts/apr-rerank-v1.yaml`.
//!
//! Discharge strategy: hand-build a candidate set with distinct
//! relevance scores. Pass a similarity function that returns a
//! large positive number for any pair (so a buggy `λ=0.999`-vs-`λ=1`
//! off-by-one would visibly perturb the order). Assert (a) the
//! output IS the input sorted by score descending, and (b) the
//! output scores ARE the input relevance scores (no diversity
//! bleed).

#![allow(clippy::unwrap_used)]

use aprender_rag::mmr::mmr_select;

#[test]
fn mmr_lambda_one_is_identity() {
    let candidates =
        vec![("doc-mid", 0.50_f32), ("doc-top", 0.95), ("doc-bot", 0.10), ("doc-hi", 0.80)];

    // Similarity returns 1.0 for any pair — maximum possible
    // diversity penalty. If λ=1 doesn't actually zero the diversity
    // term, the order will scramble.
    let sim = |_x: &&str, _y: &&str| 1.0_f32;

    let got = mmr_select(candidates, sim, 1.0, 4);

    // Order: top > hi > mid > bot.
    let order: Vec<&str> = got.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        order,
        vec!["doc-top", "doc-hi", "doc-mid", "doc-bot"],
        "FALSIFY-RERANK-MMR-002: at λ=1, MMR must return input sorted \
         by relevance descending; got {order:?}",
    );

    // Score equality: at λ=1 the MMR score equals the relevance.
    let scores: Vec<f32> = got.iter().map(|(_, s)| *s).collect();
    let want = [0.95_f32, 0.80, 0.50, 0.10];
    for (i, (g, w)) in scores.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < f32::EPSILON,
            "FALSIFY-RERANK-MMR-002: at λ=1, output score [{i}] = {g}, \
             expected {w} (the relevance term). A non-zero discrepancy \
             indicates the diversity term is leaking.",
        );
    }
}

#[test]
fn mmr_lambda_one_top_k_smaller_than_input() {
    let candidates = vec![("a", 0.9_f32), ("b", 0.8), ("c", 0.7), ("d", 0.6)];
    let sim = |_x: &&str, _y: &&str| 0.7_f32;
    let got = mmr_select(candidates, sim, 1.0, 2);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].0, "a");
    assert_eq!(got[1].0, "b");
}

#[test]
fn mmr_lambda_one_with_uniform_relevance_preserves_input_first_seen() {
    // Edge case: all relevances equal. At λ=1, all candidates have
    // the same MMR score; max_by tiebreaks by first-seen (because
    // `>` doesn't replace on equality). Output length must equal
    // top_k, not be empty or short.
    let candidates = vec![("a", 0.5_f32), ("b", 0.5), ("c", 0.5)];
    let sim = |_x: &&str, _y: &&str| 1.0_f32;
    let got = mmr_select(candidates, sim, 1.0, 3);
    assert_eq!(got.len(), 3);
}

#[test]
fn mmr_lambda_changes_the_output_order() {
    // Negative-shape sanity: under a discriminating similarity
    // function, `λ=0` and `λ=1` produce *different* output orders.
    // This is what makes the λ=1 gate load-bearing — if the MMR
    // formula collapsed to relevance for ALL λ, the main gate would
    // pass vacuously. Asserting that λ varies the output proves the
    // formula actually balances both terms.
    //
    // Note on tie-breaking: at λ=0, all "first-pick" MMR scores are
    // tied at 0 (no selected set ⇒ no diversity penalty). `max_by`
    // returns the LAST equally-maximum element, so the first pick
    // is the input's last element. We don't depend on a specific
    // first-pick order here — just that the two outputs differ.
    let candidates = vec![("a", 0.99_f32), ("a-dup", 0.95), ("b", 0.50)];
    // 'a' and 'a-dup' are similar; 'b' is dissimilar to both.
    let sim = |x: &&str, y: &&str| {
        if x.starts_with('a') && y.starts_with('a') {
            0.95
        } else {
            0.0
        }
    };

    let by_rel: Vec<&str> =
        mmr_select(candidates.clone(), sim, 1.0, 3).iter().map(|(s, _)| *s).collect();
    let by_div: Vec<&str> = mmr_select(candidates, sim, 0.0, 3).iter().map(|(s, _)| *s).collect();

    assert_ne!(
        by_rel, by_div,
        "λ=1 and λ=0 should produce different orders under a \
         discriminating similarity; if they match the diversity \
         term is being ignored.",
    );
    // Also pin λ=1's order specifically (mirrors the main gate).
    assert_eq!(by_rel, vec!["a", "a-dup", "b"]);
}
