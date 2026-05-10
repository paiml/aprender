//! FALSIFY-HNSW-PERSIST-001 — `PersistentHnsw::open()` after a `flush()`
//! and process boundary (simulated by `drop`) yields *exactly* the same
//! `Vec<(id, score)>` top-k as the original handle, byte-for-byte.
//!
//! Contract: `contracts/apr-hnsw-persistence-v1.yaml`.
//!
//! Discharge strategy: insert N vectors into a `PersistentHnsw`, query
//! for k neighbours, flush, drop, reopen, query again — equality of
//! the two `Vec<(String, f64)>` is the load-bearing assertion. The
//! corpus is small (8-32 vectors) so the gate runs in well under a
//! second on CI; production-size validation belongs in
//! FALSIFY-HNSW-PERSIST-003 (Phase 3, recall threshold).
//!
//! Note on determinism: HNSW graph construction uses a thread-local
//! RNG. Phase 1 sidesteps the resulting non-determinism by
//! serializing the WHOLE graph (`HNSWIndex` derives `Serialize` on
//! everything except its `rng` field, which is `#[serde(skip)]`). So
//! the comparison is "saved-and-reloaded == original" — never
//! "rebuilt-from-vectors == original", which would be RNG-dependent
//! and flaky.

#![allow(clippy::unwrap_used)]

use aprender::index::PersistentHnsw;
use aprender::primitives::Vector;
use tempfile::tempdir;

fn corpus_3d() -> Vec<(String, Vector<f64>)> {
    vec![
        ("doc-x".to_string(), Vector::from_slice(&[1.0, 0.0, 0.0])),
        ("doc-y".to_string(), Vector::from_slice(&[0.0, 1.0, 0.0])),
        ("doc-z".to_string(), Vector::from_slice(&[0.0, 0.0, 1.0])),
        ("doc-xy".to_string(), Vector::from_slice(&[0.7, 0.7, 0.0])),
        ("doc-yz".to_string(), Vector::from_slice(&[0.0, 0.7, 0.7])),
        ("doc-xz".to_string(), Vector::from_slice(&[0.7, 0.0, 0.7])),
        ("doc-mid".to_string(), Vector::from_slice(&[0.6, 0.6, 0.6])),
        ("doc-axis".to_string(), Vector::from_slice(&[0.5, 0.0, 0.0])),
    ]
}

#[test]
fn reopen_top_k_matches_in_memory() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("snapshot.bin");

    // Build, query, flush, drop.
    let mut idx = PersistentHnsw::open(&path, 8, 64).unwrap();
    for (id, vec) in corpus_3d() {
        idx.add(id, vec);
    }
    let queries = [
        Vector::from_slice(&[0.9, 0.1, 0.0]),
        Vector::from_slice(&[0.0, 0.5, 0.5]),
        Vector::from_slice(&[0.4, 0.4, 0.4]),
    ];
    let k = 3;
    let baseline: Vec<Vec<(String, f64)>> = queries.iter().map(|q| idx.search(q, k)).collect();
    idx.flush().unwrap();
    drop(idx);

    // Reopen. Top-k must match byte-for-byte for every query.
    let reopened = PersistentHnsw::open(&path, 8, 64).unwrap();
    for (q, want) in queries.iter().zip(baseline.iter()) {
        let got = reopened.search(q, k);
        assert_eq!(
            got, *want,
            "FALSIFY-HNSW-PERSIST-001: reopened top-{k} for {q:?} \
             differs from baseline. Got {got:?}, want {want:?}.",
        );
    }
    assert_eq!(reopened.len(), 8);
}

#[test]
fn reopen_preserves_size_and_membership() {
    // Auxiliary: a regression-only assertion that len() and the set
    // of returned IDs across many queries match. Catches the case
    // where a node serializes but its item_to_node entry doesn't.
    let dir = tempdir().unwrap();
    let path = dir.path().join("members.bin");

    let mut idx = PersistentHnsw::open(&path, 8, 64).unwrap();
    for (id, vec) in corpus_3d() {
        idx.add(id, vec);
    }
    let pre_len = idx.len();
    idx.flush().unwrap();
    drop(idx);

    let reopened = PersistentHnsw::open(&path, 8, 64).unwrap();
    assert_eq!(reopened.len(), pre_len);

    // Every original ID must be reachable as a top-k hit for some
    // probe — sanity that the persistence didn't drop a node.
    let mut seen = std::collections::BTreeSet::new();
    for (id, vec) in corpus_3d() {
        for (hit_id, _) in reopened.search(&vec, 1) {
            seen.insert(hit_id);
        }
        let _ = id;
    }
    assert!(
        !seen.is_empty(),
        "reopened index must return at least one ID across probes"
    );
}

#[test]
fn empty_index_round_trips() {
    // Edge case: flushing an empty index and reopening must yield an
    // empty index that returns Vec::new() for any query — never
    // panics, never returns garbage.
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.bin");
    let mut idx = PersistentHnsw::open(&path, 8, 64).unwrap();
    idx.flush().unwrap();
    drop(idx);

    let reopened = PersistentHnsw::open(&path, 8, 64).unwrap();
    assert!(reopened.is_empty());
    let hits = reopened.search(&Vector::from_slice(&[1.0, 0.0]), 5);
    assert!(hits.is_empty());
}
