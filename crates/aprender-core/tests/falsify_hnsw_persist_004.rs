//! FALSIFY-HNSW-PERSIST-004 — cold-open + first-query latency on the
//! CI fixture stays under the budget (default 500 ms; tunable via
//! `APR_HNSW_OPEN_BUDGET_MS`). Falsifies "open() rebuilds the graph
//! eagerly" or "first query hits a cold cache that takes seconds".
//!
//! Contract: `contracts/apr-hnsw-persistence-v1.yaml` v1.3.0.
//!
//! Discharge strategy:
//! 1. Build a deterministic 200-doc × 32-dim fixture (same shape as
//!    gate 003 so the workload is comparable).
//! 2. Flush, drop the handle.
//! 3. Start a `Instant` clock, `open()` the snapshot, run one
//!    `search()`, stop the clock.
//! 4. Assert elapsed < budget.
//!
//! The 500 ms budget is *comfortably loose* on a 200-doc fixture —
//! Phase 1 implementation typically completes in 1-10 ms. The gate
//! exists to catch order-of-magnitude regressions (e.g., a future
//! refactor accidentally re-running `add()` for every persisted
//! vector on open), not to chase tens of ms. Operators on
//! latency-sensitive paths can tighten via the env var.

#![allow(clippy::unwrap_used)]

use aprender::index::PersistentHnsw;
use aprender::primitives::Vector;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const CORPUS_SIZE: usize = 200;
const DIM: usize = 32;

fn budget() -> Duration {
    let ms = std::env::var("APR_HNSW_OPEN_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(500);
    Duration::from_millis(ms)
}

fn random_vector(rng: &mut ChaCha8Rng) -> Vec<f64> {
    (0..DIM).map(|_| rng.random::<f64>() - 0.5).collect()
}

#[test]
fn cold_open_first_query_within_budget() {
    let mut rng = ChaCha8Rng::seed_from_u64(2025);
    let corpus: Vec<(String, Vec<f64>)> = (0..CORPUS_SIZE)
        .map(|i| (format!("doc-{i:04}"), random_vector(&mut rng)))
        .collect();
    let probe = random_vector(&mut rng);

    // Set up the snapshot.
    let dir = tempdir().unwrap();
    let path = dir.path().join("cold.bin");
    let mut idx = PersistentHnsw::open(&path, 16, 200).unwrap();
    for (id, v) in &corpus {
        idx.add(id.clone(), Vector::from_slice(v));
    }
    idx.flush().unwrap();
    drop(idx);

    // Cold open + first query measurement.
    let probe_vec = Vector::from_slice(&probe);
    let start = Instant::now();
    let reopened = PersistentHnsw::open(&path, 16, 200).unwrap();
    let hits = reopened.search(&probe_vec, 10);
    let elapsed = start.elapsed();

    let max = budget();
    assert!(
        elapsed < max,
        "FALSIFY-HNSW-PERSIST-004: cold-open + first-query took \
         {}ms, exceeding budget {}ms. Likely cause: open() is doing \
         work that should be lazy (e.g., re-running add() per vector \
         instead of reading the serialized graph) or first-search \
         is rebuilding state. Tunable via APR_HNSW_OPEN_BUDGET_MS.",
        elapsed.as_millis(),
        max.as_millis(),
    );
    // Sanity: the search returned actual results (not zero from a
    // degenerate empty index).
    assert!(
        !hits.is_empty(),
        "first search after cold open returned no hits — fixture or open() is broken"
    );
}

#[test]
fn open_alone_is_well_under_budget() {
    // Cleanly separates the contributions: just open(), no search.
    // If THIS exceeds the budget, the rebuild path is misbehaving;
    // the main gate's failure would otherwise look ambiguous.
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let corpus: Vec<(String, Vec<f64>)> = (0..CORPUS_SIZE)
        .map(|i| (format!("doc-{i:04}"), random_vector(&mut rng)))
        .collect();
    let dir = tempdir().unwrap();
    let path = dir.path().join("open.bin");
    let mut idx = PersistentHnsw::open(&path, 16, 200).unwrap();
    for (id, v) in &corpus {
        idx.add(id.clone(), Vector::from_slice(v));
    }
    idx.flush().unwrap();
    drop(idx);

    let start = Instant::now();
    let _ = PersistentHnsw::open(&path, 16, 200).unwrap();
    let elapsed = start.elapsed();
    let max = budget();
    assert!(
        elapsed < max,
        "open() alone took {}ms, exceeding {}ms — bincode \
         deserialize should be O(file size), not O(corpus_size × \
         add_cost).",
        elapsed.as_millis(),
        max.as_millis(),
    );
}
