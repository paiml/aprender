//! FALSIFY-RERANK-RRF-002 — RRF is input-order invariant on a
//! tie-free rotational fixture.
//!
//! Contract: `contracts/apr-rerank-v1.yaml`.
//!
//! Discharge strategy: construct a 3-document rotation
//! (`a = [A, B, C]`, `b = [B, C, A]`) so that every output combined
//! score is distinct (no HashMap iteration-order ambiguity in the
//! tie-breaker sort). RRF math: `score(d) = sum_i 1/(k + rank_i(d) + 1)`
//! where `rank_i` is d's 0-indexed position in list i. Combined
//! scores for the rotation:
//!
//! - A: 1/(60+1) + 1/(60+3) = 1/61 + 1/63 ≈ 0.03226
//! - B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.03252
//! - C: 1/(60+3) + 1/(60+2) = 1/63 + 1/62 ≈ 0.03200
//!
//! All three scores distinct → sort is fully determined → output is
//! byte-stable across input swap.

#![allow(clippy::unwrap_used)]

use aprender_rag::fusion::FusionStrategy;
use aprender_rag::ChunkId;
use uuid::Uuid;

fn ids() -> (ChunkId, ChunkId, ChunkId) {
    // Deterministic UUIDs so the fixture is reproducible across runs.
    (
        ChunkId(Uuid::from_u128(0x0000_0000_0000_0000_0000_0000_0000_000A)),
        ChunkId(Uuid::from_u128(0x0000_0000_0000_0000_0000_0000_0000_000B)),
        ChunkId(Uuid::from_u128(0x0000_0000_0000_0000_0000_0000_0000_000C)),
    )
}

#[test]
fn rrf_is_input_order_invariant() {
    let (a, b, c) = ids();
    let list_p = vec![(a, 0.9_f32), (b, 0.8), (c, 0.7)];
    let list_q = vec![(b, 0.9_f32), (c, 0.8), (a, 0.7)];

    let strategy = FusionStrategy::RRF { k: 60.0 };

    let pq = strategy.fuse(&list_p, &list_q);
    let qp = strategy.fuse(&list_q, &list_p);

    assert_eq!(
        pq, qp,
        "FALSIFY-RERANK-RRF-002: rrf(p, q) and rrf(q, p) must produce \
         byte-identical output on a tie-free fixture. \
         pq={pq:?}, qp={qp:?}",
    );
}

#[test]
fn rrf_distinct_scores_on_rotational_fixture() {
    // Sanity for the harness itself: the three output scores must
    // actually be distinct, otherwise the main gate's "byte-for-byte"
    // claim could pass via HashMap reorder coincidence.
    let (a, b, c) = ids();
    let list_p = vec![(a, 0.9_f32), (b, 0.8), (c, 0.7)];
    let list_q = vec![(b, 0.9_f32), (c, 0.8), (a, 0.7)];

    let strategy = FusionStrategy::RRF { k: 60.0 };
    let fused = strategy.fuse(&list_p, &list_q);

    let scores: Vec<f32> = fused.iter().map(|(_, s)| *s).collect();
    let mut sorted = scores.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    sorted.dedup_by(|x, y| (*x - *y).abs() < f32::EPSILON);
    assert_eq!(
        sorted.len(),
        scores.len(),
        "fixture should produce distinct scores; got {scores:?}"
    );
}

#[test]
fn rrf_three_way_input_swap_consistency() {
    // Stronger invariance: pairwise commutativity holds for every
    // ordering of two lists. Trivial corollary of the math, but
    // catches the case where the impl special-cases "first list is
    // dense, second is sparse" with asymmetric tiebreaking.
    let (a, b, c) = ids();
    let list_p = vec![(a, 0.9_f32), (b, 0.8), (c, 0.7)];
    let list_q = vec![(b, 0.9_f32), (c, 0.8), (a, 0.7)];

    let strategy = FusionStrategy::RRF { k: 60.0 };
    let pq = strategy.fuse(&list_p, &list_q);
    let qp = strategy.fuse(&list_q, &list_p);

    // Order should be B > A > C per the math above.
    assert_eq!(pq.len(), 3);
    assert_eq!(pq[0].0, b);
    assert_eq!(pq[1].0, a);
    assert_eq!(pq[2].0, c);
    assert_eq!(qp[0].0, b);
    assert_eq!(qp[1].0, a);
    assert_eq!(qp[2].0, c);
}
