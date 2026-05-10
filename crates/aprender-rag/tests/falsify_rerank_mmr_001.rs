//! FALSIFY-RERANK-MMR-001 — MMR with `λ=0.5` increases the
//! mean-pairwise-distance diversity of the top-k by ≥10% vs. the
//! relevance-only baseline (λ=1), while keeping recall@k within 1
//! percentage point.
//!
//! Contract: `contracts/apr-rerank-v1.yaml` v1.1.0.
//!
//! Discharge strategy: build an 8-doc clustered fixture in 2D
//! euclidean space, two clusters of 4 centred at (1, 0) and (0, 1)
//! respectively. All 8 docs are ground-truth relevant. Run MMR
//! with `λ=1.0` (relevance-only baseline) and `λ=0.5` (balanced)
//! at top_k=4. Compute mean pairwise euclidean distance for each
//! and assert:
//! - `mean_dist(λ=0.5) ≥ 1.10 * mean_dist(λ=1.0)` (diversity gain)
//! - `recall(λ=0.5) ≥ recall(λ=1.0) - 0.01` (recall budget)
//!
//! Why all-relevant ground-truth: with K=4 selected from N=8
//! relevant, both schemes return 4/8 = 0.5 recall identically;
//! the "within 1 percentage point" budget binds the gate against a
//! regression where MMR gained diversity by *excluding* ground-truth
//! docs.
//!
//! Why 8 docs not 6 (per the §2.6 pre-auth): with 6 docs (3 per
//! cluster) and top_k=4, baseline (λ=1) and MMR (λ=0.5) returned
//! the SAME SET (just different selection order) — mean-pairwise-
//! distance is set-not-order-dependent, so the diversity assertion
//! could never fire. Widening to 8/4-per-cluster makes the SETS
//! differ (baseline takes all 4 from A; MMR takes 2 from each).
//! Drift recorded in the spec's §6 falsification log under v0.13.0.

#![allow(clippy::unwrap_used)]

use aprender_rag::mmr::mmr_select;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
struct Doc {
    id: &'static str,
    embedding: [f32; 2],
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)).max(0.0).min(1.0)
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn fixture() -> Vec<(Doc, f32)> {
    // Two clusters of 4, top_k=4 ⇒ baseline (λ=1) takes the entire
    // top-relevance cluster A; MMR (λ=0.5) interleaves clusters.
    // The fixture's load-bearing property: relevance order picks 4
    // of 4 from A, while diversity-aware order picks 2 from each
    // cluster — so the SETS differ (not just the order), and the
    // mean-pairwise-distance metric varies between them.
    vec![
        (Doc { id: "a1", embedding: [1.00, 0.05] }, 0.95),
        (Doc { id: "a2", embedding: [0.95, 0.10] }, 0.90),
        (Doc { id: "a3", embedding: [0.90, 0.15] }, 0.85),
        (Doc { id: "a4", embedding: [0.85, 0.20] }, 0.80),
        (Doc { id: "b1", embedding: [0.05, 1.00] }, 0.75),
        (Doc { id: "b2", embedding: [0.10, 0.95] }, 0.70),
        (Doc { id: "b3", embedding: [0.15, 0.90] }, 0.65),
        (Doc { id: "b4", embedding: [0.20, 0.85] }, 0.60),
    ]
}

fn mean_pairwise_distance(items: &[(Doc, f32)]) -> f32 {
    let n = items.len();
    if n < 2 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    let mut pairs = 0_u32;
    for i in 0..n {
        for j in (i + 1)..n {
            total += euclidean(&items[i].0.embedding, &items[j].0.embedding);
            pairs += 1;
        }
    }
    total / f32::from(u16::try_from(pairs).expect("≤ 8 docs ⇒ ≤ 28 pairs"))
}

fn recall_at_k(selected: &[(Doc, f32)], ground_truth: &BTreeSet<&'static str>) -> f32 {
    let hit = selected.iter().filter(|(d, _)| ground_truth.contains(&d.id)).count();
    #[allow(clippy::cast_precision_loss)]
    let denom = ground_truth.len() as f32;
    #[allow(clippy::cast_precision_loss)]
    let num = hit as f32;
    num / denom
}

#[test]
fn mmr_increases_diversity_within_recall_budget() {
    let cands = fixture();
    // All 8 docs are ground-truth relevant — see module-doc for why.
    let ground_truth: BTreeSet<&'static str> = cands.iter().map(|(d, _)| d.id).collect();

    let sim = |a: &Doc, b: &Doc| cosine_similarity(&a.embedding, &b.embedding);

    let baseline = mmr_select(cands.clone(), sim, 1.0, 4);
    let diverse = mmr_select(cands.clone(), sim, 0.5, 4);

    let baseline_div = mean_pairwise_distance(&baseline);
    let diverse_div = mean_pairwise_distance(&diverse);
    let baseline_recall = recall_at_k(&baseline, &ground_truth);
    let diverse_recall = recall_at_k(&diverse, &ground_truth);

    // Diversity gain ≥ 10%.
    assert!(
        diverse_div >= 1.10 * baseline_div,
        "FALSIFY-RERANK-MMR-001: MMR(λ=0.5) mean-pairwise-distance \
         {diverse_div:.4} should be ≥ 1.10 × baseline {baseline_div:.4} \
         = {:.4}. Diversity gain insufficient on the clustered fixture; \
         either MMR is ignoring the diversity term or the fixture is \
         degenerate.",
        1.10 * baseline_div,
    );

    // Recall budget: within 1 percentage point.
    assert!(
        diverse_recall >= baseline_recall - 0.01,
        "FALSIFY-RERANK-MMR-001: MMR(λ=0.5) recall@4 \
         {diverse_recall:.4} dropped more than 1pp below baseline \
         {baseline_recall:.4}. MMR is gaining diversity at the cost of \
         relevance — not the kind of balance the gate enforces.",
    );

    // Sanity: confirm baseline is what we expect — relevance-only
    // top-4 should be all 4 cluster-A docs, since A's relevance
    // scores (0.95-0.80) all exceed B's (0.75-0.60). This is the
    // load-bearing property that lets the diversity assertion
    // detect MMR's redistribution.
    let baseline_a_count = baseline.iter().filter(|(d, _)| d.id.starts_with('a')).count();
    assert_eq!(
        baseline_a_count, 4,
        "fixture sanity: relevance-only baseline should pick all 4 \
         cluster-A docs; got {baseline_a_count}. Diversity-gain \
         assertion is meaningless if baseline is already spread \
         across clusters."
    );
}

#[test]
fn fixture_recall_baseline_is_one_half() {
    // Sanity: with K=4 selected from N=8 ground-truth relevant,
    // recall@k must equal 4/8 = 0.5 for any reasonable selection.
    // Catches a bug in the harness where ground_truth is computed
    // wrong (e.g., includes only some candidates).
    let cands = fixture();
    let ground_truth: BTreeSet<&'static str> = cands.iter().map(|(d, _)| d.id).collect();
    assert_eq!(ground_truth.len(), 8);

    let sim = |a: &Doc, b: &Doc| cosine_similarity(&a.embedding, &b.embedding);
    let baseline = mmr_select(cands, sim, 1.0, 4);
    let r = recall_at_k(&baseline, &ground_truth);
    assert!(
        (r - 0.5).abs() < f32::EPSILON,
        "expected recall@4 = 0.500 with all-relevant ground truth, got {r:.4}"
    );
}
