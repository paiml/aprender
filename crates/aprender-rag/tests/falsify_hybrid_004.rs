//! FALSIFY-HYBRID-004 — `BM25Index` batch ingest of a deterministic
//! 5k-doc fixture stays within a tunable budget. Extrapolates
//! linearly to the §2.5 production target of "<2 min for 1M docs".
//!
//! Contract: `contracts/apr-hybrid-retrieval-v1.yaml` v1.1.0.
//!
//! Discharge strategy:
//! 1. Generate 5k synthetic sentences with `ChaCha8Rng::seed_from_u64(2026)`,
//!    each sentence drawing 10 tokens from a 100-word vocabulary.
//!    Bit-reproducible across machines.
//! 2. Build chunks from those sentences (no embeddings — BM25 doesn't
//!    need them).
//! 3. Time `BM25Index::add_batch(&chunks)`.
//! 4. Assert elapsed < budget.
//!
//! Why 5k docs not 100k: full 100k would still be O(seconds) on
//! commodity hardware but blows up CI memory + wall-clock without
//! changing what the gate detects (super-linear regressions). 5k is
//! large enough that an O(N²) bug hits the budget; small enough to
//! run in well under a second on the happy path.
//!
//! The 10s budget is loose (≥16× over the linear-extrapolated
//! expectation of ~0.6s) so shared CI runners with cold caches
//! don't flake. Tighter operator-set budgets via
//! `APR_BM25_BUILD_BUDGET_MS`.

#![allow(clippy::unwrap_used)]

use aprender_rag::index::{BM25Index, SparseIndex};
use aprender_rag::{Chunk, DocumentId};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

const CORPUS_SIZE: usize = 5_000;
const WORDS_PER_DOC: usize = 10;
const VOCAB_SIZE: usize = 100;

fn budget() -> Duration {
    let ms = std::env::var("APR_BM25_BUILD_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10_000);
    Duration::from_millis(ms)
}

fn vocab() -> Vec<String> {
    (0..VOCAB_SIZE).map(|i| format!("word{i:03}")).collect()
}

fn synthetic_sentence(rng: &mut ChaCha8Rng, vocab: &[String]) -> String {
    (0..WORDS_PER_DOC)
        .map(|_| {
            let idx = rng.random_range(0..vocab.len());
            vocab[idx].as_str()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn build_corpus() -> Vec<Chunk> {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let v = vocab();
    let doc_id = DocumentId::new();
    // Note: `Chunk::new` generates a fresh `ChunkId` per call (UUID
    // v4). For this perf gate the IDs don't need to be reproducible —
    // only the CONTENT does (so vocabulary distribution and average
    // doc length are stable across runs).
    (0..CORPUS_SIZE)
        .map(|_| {
            let content = synthetic_sentence(&mut rng, &v);
            Chunk::new(doc_id, content.clone(), 0, content.len())
        })
        .collect()
}

#[test]
fn bm25_batch_index_within_budget() {
    let chunks = build_corpus();
    assert_eq!(chunks.len(), CORPUS_SIZE, "fixture corpus size should match constant");

    let mut index = BM25Index::new();
    let start = Instant::now();
    index.add_batch(&chunks);
    let elapsed = start.elapsed();

    let max = budget();
    assert!(
        elapsed < max,
        "FALSIFY-HYBRID-004: BM25Index::add_batch on a {CORPUS_SIZE}-doc \
         fixture took {}ms, exceeding budget {}ms. The §2.5 production \
         target extrapolates linearly to ~0.6s for 5k docs; this gate's \
         {}s ceiling is ≥16× headroom and exists to catch \
         super-linear-in-corpus regressions, not microbenchmark perf. \
         Tunable via APR_BM25_BUILD_BUDGET_MS.",
        elapsed.as_millis(),
        max.as_millis(),
        max.as_secs(),
    );

    // Sanity: index actually contains what we added (not just
    // returning early on no-op input).
    assert_eq!(index.len(), CORPUS_SIZE, "index size should match input after add_batch");
}

#[test]
fn bm25_search_after_batch_returns_results() {
    // Companion: an O(N²) regression in `add_batch` could "succeed"
    // while leaving the inverted index in a state where `search`
    // returns nothing. This catches that mode.
    let chunks = build_corpus();
    let mut index = BM25Index::new();
    index.add_batch(&chunks);

    // Pick a token that's drawn from the same vocab as the corpus —
    // it MUST appear in some doc.
    let probe = "word042";
    let hits = index.search(probe, 5);
    assert!(
        !hits.is_empty(),
        "search for known-vocab token {probe:?} returned 0 hits — \
         add_batch likely failed silently. Index size: {}",
        index.len(),
    );
}

#[test]
fn bm25_per_doc_cost_is_sub_millisecond_on_average() {
    // Companion: enforce per-doc latency-of-add stays in the
    // microsecond range. If add_batch is O(N²) but completes in
    // budget on the 5k fixture, this companion test still catches
    // the regression — per-doc cost would scale with N.
    let chunks = build_corpus();
    let mut index = BM25Index::new();
    let start = Instant::now();
    index.add_batch(&chunks);
    let elapsed = start.elapsed();

    let per_doc_us = elapsed.as_micros() / u128::from(u32::try_from(CORPUS_SIZE).unwrap());
    // Generous: 500us per doc is far above linear expectations.
    assert!(
        per_doc_us < 500,
        "average per-doc add cost {per_doc_us}μs exceeds 500μs budget; \
         super-linear regression suspected. Total elapsed: {}ms, \
         corpus: {CORPUS_SIZE} docs.",
        elapsed.as_millis(),
    );
}
