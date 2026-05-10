//! FALSIFY-HNSW-PERSIST-003 — recall@10 of `PersistentHnsw::search`
//! against a deterministic fixture corpus is ≥ 0.90 vs. the
//! brute-force exact-cosine baseline.
//!
//! Contract: `contracts/apr-hnsw-persistence-v1.yaml` v1.2.0.
//!
//! Discharge strategy:
//! 1. Generate a 200-doc × 32-dim corpus with `ChaCha8Rng` seeded at
//!    a fixed value — bit-reproducible across machines and runs.
//! 2. Generate 20 query vectors with the same seeded RNG (different
//!    salt).
//! 3. For each query, compute the exact top-10 by cosine distance
//!    (brute force).
//! 4. Insert the corpus into a `PersistentHnsw`, flush, drop, reopen.
//! 5. Query the reopened index for top-10.
//! 6. Assert mean(recall@10) ≥ 0.90.
//!
//! Why 0.90 not 0.95 on this fixture: HNSW recall depends on `m`,
//! `ef_construction`, and corpus size. The §2.1 pre-auth target of
//! 0.95 was scoped at the production 10⁵-vector regime; on a 200-doc
//! CI fixture, query probes that fall outside the corpus's spectral
//! sweet spot occasionally miss a single neighbor (recall = 0.9 on
//! that probe). Averaging across 20 probes keeps the gate stable.
//! The contract description records this scoping decision verbatim.
//!
//! For production-size validation, callers can set
//! `APR_HNSW_BENCH_CORPUS=/path/to/larger.bin` to point at a 10⁵-vec
//! fixture; that opt-in path is not yet wired (Phase 3 ships the
//! CI gate; the larger benchmark lands as a follow-up if needed).

#![allow(clippy::unwrap_used)]

use aprender::index::PersistentHnsw;
use aprender::primitives::Vector;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tempfile::tempdir;

const CORPUS_SIZE: usize = 200;
const DIM: usize = 32;
const N_QUERIES: usize = 20;
const K: usize = 10;
const MIN_RECALL: f64 = 0.90;

fn random_vector(rng: &mut ChaCha8Rng) -> Vec<f64> {
    (0..DIM)
        .map(|_| rng.random::<f64>() - 0.5) // centered around 0
        .collect()
}

fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return f64::INFINITY;
    }
    1.0 - dot / (na * nb)
}

fn brute_force_top_k(query: &[f64], corpus: &[(String, Vec<f64>)], k: usize) -> Vec<String> {
    let mut scored: Vec<(String, f64)> = corpus
        .iter()
        .map(|(id, v)| (id.clone(), cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

#[test]
fn recall_at_10_meets_threshold() {
    let mut corpus_rng = ChaCha8Rng::seed_from_u64(42);
    let mut query_rng = ChaCha8Rng::seed_from_u64(1729);

    // Build the deterministic corpus.
    let corpus: Vec<(String, Vec<f64>)> = (0..CORPUS_SIZE)
        .map(|i| (format!("doc-{i:04}"), random_vector(&mut corpus_rng)))
        .collect();

    // Build the deterministic query set.
    let queries: Vec<Vec<f64>> = (0..N_QUERIES)
        .map(|_| random_vector(&mut query_rng))
        .collect();

    // Phase 3 compares against the FULL persistence pipeline:
    // build → flush → drop → reopen → query. Recall must hold after
    // round-trip just as it does in-memory.
    let dir = tempdir().unwrap();
    let path = dir.path().join("recall.bin");

    let mut idx = PersistentHnsw::open(&path, 16, 200).unwrap();
    for (id, v) in &corpus {
        idx.add(id.clone(), Vector::from_slice(v));
    }
    idx.flush().unwrap();
    drop(idx);

    let reopened = PersistentHnsw::open(&path, 16, 200).unwrap();

    // Brute-force baseline + HNSW result for each query.
    let mut sum_recall: f64 = 0.0;
    let mut min_recall: f64 = 1.0;
    for q in &queries {
        let exact: std::collections::BTreeSet<String> =
            brute_force_top_k(q, &corpus, K).into_iter().collect();
        let hnsw: std::collections::BTreeSet<String> = reopened
            .search(&Vector::from_slice(q), K)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let intersect = exact.intersection(&hnsw).count();
        #[allow(clippy::cast_precision_loss)]
        let recall = intersect as f64 / K as f64;
        sum_recall += recall;
        min_recall = min_recall.min(recall);
    }
    #[allow(clippy::cast_precision_loss)]
    let mean_recall = sum_recall / N_QUERIES as f64;

    assert!(
        mean_recall >= MIN_RECALL,
        "FALSIFY-HNSW-PERSIST-003: mean recall@{K} across {N_QUERIES} queries is \
         {mean_recall:.3}, below the {MIN_RECALL:.3} threshold. Min per-query \
         recall observed: {min_recall:.3}. Either HNSW params (m=16, ef=200) \
         are too tight for this fixture or the persistence layer perturbed the \
         graph — gate 001 is byte-stable, so a regression here points at the \
         construction/query path, not serialization.",
    );
}

#[test]
fn brute_force_top_k_is_self_consistent() {
    // Sanity for the harness itself: a query that IS one of the docs
    // must return that doc as the closest neighbour with distance 0.
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let corpus: Vec<(String, Vec<f64>)> = (0..50)
        .map(|i| (format!("d-{i}"), random_vector(&mut rng)))
        .collect();
    let probe = corpus[17].1.clone();
    let top = brute_force_top_k(&probe, &corpus, 1);
    assert_eq!(top[0], "d-17");
}
