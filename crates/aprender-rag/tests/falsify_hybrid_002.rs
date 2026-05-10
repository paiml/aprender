//! FALSIFY-HYBRID-002 — `HybridRetriever::retrieve` is score-equivalent
//! to a manual `FusionStrategy::fuse(dense_search, sparse_search)`
//! callsite. The trait method does not silently re-normalize, drop
//! candidates, or change weighting compared to the documented
//! arithmetic.
//!
//! Contract: `contracts/apr-hybrid-retrieval-v1.yaml`.
//!
//! Discharge strategy: index a fixture corpus into a real
//! `HybridRetriever`, run `retrieve(query, k)`, then replay the same
//! pipeline manually using the public accessors
//! (`dense_store().search`, `sparse_index().search`, and the same
//! `FusionStrategy::fuse`). Assert the `(chunk_id, fused_score)`
//! pairs match.
//!
//! The test runs against multiple FusionStrategy variants (RRF,
//! Linear) so a regression in any strategy's wiring fails the gate.

#![allow(clippy::unwrap_used)]

use aprender_rag::embed::{Embedder, MockEmbedder};
use aprender_rag::fusion::FusionStrategy;
use aprender_rag::index::{BM25Index, SparseIndex, VectorStore};
use aprender_rag::retrieve::{HybridRetriever, HybridRetrieverConfig};
use aprender_rag::{Chunk, ChunkId, DocumentId};

const DIM: usize = 8;

/// Build a deterministic corpus where each chunk's embedding is
/// directly given (so the test doesn't depend on the embedder's
/// content-derivation algorithm).
fn corpus() -> Vec<(Chunk, Vec<f32>)> {
    let docs = [
        (
            "doc-1",
            "machine learning algorithms train models",
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "doc-2",
            "deep learning neural networks for vision",
            vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "doc-3",
            "natural language processing transformers",
            vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        ("doc-4", "search engines use BM25 ranking", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("doc-5", "rust programming systems memory", vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (
            "doc-6",
            "machine ranking and learning fusion",
            vec![0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ];
    docs.into_iter()
        .map(|(_label, content, emb)| {
            let mut chunk = Chunk::new(DocumentId::new(), content.to_string(), 0, content.len());
            chunk.set_embedding(emb.clone());
            (chunk, emb)
        })
        .collect()
}

fn build_retriever(
    strategy: FusionStrategy,
) -> (HybridRetriever<MockEmbedder>, MockEmbedder, HybridRetrieverConfig) {
    let embedder = MockEmbedder::new(DIM);
    let dense = VectorStore::with_dimension(DIM);
    let sparse = BM25Index::new();

    let config = HybridRetrieverConfig {
        candidates_per_source: 20,
        fusion: strategy,
        use_dense: true,
        use_sparse: true,
    };
    let mut retriever =
        HybridRetriever::new(dense, sparse, embedder.clone()).with_config(config.clone());
    for (chunk, _) in corpus() {
        retriever.index(chunk).unwrap();
    }
    (retriever, embedder, config)
}

/// Replay the trait method's pipeline by hand.
fn manual_fused_pairs(
    retriever: &HybridRetriever<MockEmbedder>,
    embedder: &MockEmbedder,
    config: &HybridRetrieverConfig,
    query: &str,
    k: usize,
) -> Vec<(ChunkId, f32)> {
    let q_emb = embedder.embed_query(query).unwrap();
    let dense_results =
        retriever.dense_store().search(&q_emb, config.candidates_per_source).unwrap();
    let sparse_results = retriever.sparse_index().search(query, config.candidates_per_source);
    config.fusion.fuse(&dense_results, &sparse_results).into_iter().take(k).collect()
}

fn trait_fused_pairs(
    retriever: &HybridRetriever<MockEmbedder>,
    query: &str,
    k: usize,
) -> Vec<(ChunkId, f32)> {
    retriever
        .retrieve(query, k)
        .unwrap()
        .into_iter()
        .map(|r| {
            let id = r.chunk.id;
            let score = r.fused_score.expect("retrieve() always populates fused_score");
            (id, score)
        })
        .collect()
}

#[test]
fn trait_method_matches_explicit_combine() {
    // Run the gate against multiple FusionStrategy variants — a
    // regression in any one strategy's wiring breaks the gate.
    let strategies =
        [FusionStrategy::RRF { k: 60.0 }, FusionStrategy::Linear { dense_weight: 0.7 }];

    for strategy in strategies {
        let (retriever, embedder, config) = build_retriever(strategy.clone());

        for (query, k) in [("learning algorithms", 3), ("ranking BM25", 4), ("rust", 5)] {
            let trait_side = trait_fused_pairs(&retriever, query, k);
            let manual_side = manual_fused_pairs(&retriever, &embedder, &config, query, k);
            assert_eq!(
                trait_side, manual_side,
                "FALSIFY-HYBRID-002: HybridRetriever::retrieve diverged from \
                 manual `FusionStrategy::fuse` for strategy={strategy:?}, \
                 query={query:?}, k={k}.\n  \
                 trait:  {trait_side:?}\n  \
                 manual: {manual_side:?}",
            );
        }
    }
}

#[test]
fn trait_method_respects_k_truncation() {
    // Sanity: requesting top-2 returns at most 2, never the full
    // candidates_per_source. Catches a regression where retrieve()
    // forgets to apply the .take(k).
    let (retriever, _, _) = build_retriever(FusionStrategy::RRF { k: 60.0 });
    let got = retriever.retrieve("machine learning", 2).unwrap();
    assert!(got.len() <= 2);
}

#[test]
fn trait_method_populates_per_leg_scores_when_present() {
    // Companion: trait method exposes the dense + sparse scores
    // separately on the result, even when fusion is RRF (rank-based,
    // not score-based). A future refactor that loses the per-leg
    // scores would silently break downstream rerankers that consult
    // `RetrievalResult::dense_score`/`sparse_score`.
    let (retriever, _, _) = build_retriever(FusionStrategy::RRF { k: 60.0 });
    let got = retriever.retrieve("machine learning", 3).unwrap();
    let any_dense = got.iter().any(|r| r.dense_score.is_some());
    let any_sparse = got.iter().any(|r| r.sparse_score.is_some());
    assert!(
        any_dense || any_sparse,
        "at least one leg's per-leg scores must populate \
         RetrievalResult fields; got {:?}",
        got.iter().map(|r| (r.dense_score, r.sparse_score, r.fused_score)).collect::<Vec<_>>(),
    );
}
