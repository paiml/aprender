//! FALSIFY-RERANK-RRF-001 — `FusionStrategy::RRF.fuse(dense, sparse)`
//! over the dense and sparse legs of the HYBRID-001 adversarial
//! fixture yields ≥3-point nDCG@k improvement vs. either single
//! retriever.
//!
//! Contract: `contracts/apr-rerank-v1.yaml` v1.2.0.
//!
//! Discharge strategy: reuse the same 5-doc fixture as
//! `falsify_hybrid_001.rs` (intentional duplication so each
//! falsifier file is self-contained per the project's
//! falsifier-first cascade pattern). Compute nDCG@3 for the
//! single-leg results and the RRF-fused result; assert RRF beats
//! max(dense_nDCG, sparse_nDCG) by ≥0.03.
//!
//! Expected on the fixture:
//! - Dense top-3 = {d1, d2, x1}: nDCG@3 ≈ 0.765 (relevant at
//!   positions 1+2, irrelevant at 3).
//! - Sparse top-3 = {d1, d3, x2}: nDCG@3 ≈ 0.765 (same shape).
//! - RRF top-3 = {d1, d2, d3}: nDCG@3 = 1.000 (all relevant).
//! - Improvement = 0.235, far above the 0.03 contractual
//!   threshold.

#![allow(clippy::unwrap_used)]

use aprender_rag::embed::Embedder;
use aprender_rag::error::Result;
use aprender_rag::fusion::FusionStrategy;
use aprender_rag::index::{BM25Index, SparseIndex, VectorStore};
use aprender_rag::{Chunk, ChunkId, DocumentId};
use std::collections::{HashMap, HashSet};

const DIM: usize = 4;
const TOP_K: usize = 3;
const QUERY: &str = "find alpha";

#[derive(Clone)]
struct FixedEmbedder {
    map: HashMap<String, [f32; DIM]>,
}

impl FixedEmbedder {
    fn new() -> Self {
        let mut map = HashMap::new();
        map.insert(QUERY.to_string(), [1.0, 0.0, 0.0, 0.0]);
        map.insert("alpha alpha alpha".to_string(), [1.0, 0.0, 0.0, 0.0]);
        map.insert("horse zebra giraffe".to_string(), [1.0, 0.1, 0.0, 0.0]);
        map.insert("alpha alpha".to_string(), [0.0, 0.0, 1.0, 0.0]);
        map.insert("monkey rabbit".to_string(), [1.0, 0.4, 0.0, 0.0]);
        map.insert("alpha noise filler text padding".to_string(), [0.0, 0.0, 1.0, 0.3]);
        Self { map }
    }
}

impl Embedder for FixedEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.map.get(text).copied().unwrap_or([0.0; DIM]).to_vec())
    }
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
    fn dimension(&self) -> usize {
        DIM
    }
    fn model_id(&self) -> &str {
        "fixed-embedder-test-only"
    }
}

struct DocSpec {
    id: &'static str,
    content: &'static str,
    relevant: bool,
}

fn corpus() -> Vec<DocSpec> {
    vec![
        DocSpec { id: "d1", content: "alpha alpha alpha", relevant: true },
        DocSpec { id: "d2", content: "horse zebra giraffe", relevant: true },
        DocSpec { id: "d3", content: "alpha alpha", relevant: true },
        DocSpec { id: "x1", content: "monkey rabbit", relevant: false },
        DocSpec { id: "x2", content: "alpha noise filler text padding", relevant: false },
    ]
}

/// Discounted Cumulative Gain at k. With binary relevance, each
/// position's gain is `rel(i) / log2(i + 1)` where positions are
/// 1-indexed.
fn dcg_at_k(ranked: &[ChunkId], ground_truth: &HashSet<ChunkId>, k: usize) -> f32 {
    ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let rel = if ground_truth.contains(id) { 1.0 } else { 0.0 };
            #[allow(clippy::cast_precision_loss)]
            let pos = (i + 2) as f32; // log2(i+1+1) where i is 0-indexed
            rel / pos.log2()
        })
        .sum()
}

/// Ideal DCG: DCG of the ranking that puts all ground-truth-relevant
/// items first. For binary relevance, this is just `min(k, |relevant|)`
/// items each contributing 1 / log2(pos + 1).
fn idcg_at_k(num_relevant: usize, k: usize) -> f32 {
    let n = num_relevant.min(k);
    (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let pos = (i + 2) as f32;
            1.0 / pos.log2()
        })
        .sum()
}

fn ndcg_at_k(ranked: &[ChunkId], ground_truth: &HashSet<ChunkId>, k: usize) -> f32 {
    let dcg = dcg_at_k(ranked, ground_truth, k);
    let idcg = idcg_at_k(ground_truth.len(), k);
    if idcg == 0.0 {
        return 0.0;
    }
    dcg / idcg
}

#[test]
fn rrf_beats_single_retriever_ndcg10() {
    let embedder = FixedEmbedder::new();
    let mut dense = VectorStore::with_dimension(DIM);
    let mut sparse = BM25Index::new();
    let mut ground_truth: HashSet<ChunkId> = HashSet::new();
    let mut labels: HashMap<ChunkId, &'static str> = HashMap::new();

    for spec in corpus() {
        let mut chunk =
            Chunk::new(DocumentId::new(), spec.content.to_string(), 0, spec.content.len());
        chunk.set_embedding(embedder.embed(spec.content).unwrap());
        let id = chunk.id;
        if spec.relevant {
            ground_truth.insert(id);
        }
        labels.insert(id, spec.id);
        sparse.add(&chunk);
        dense.insert(chunk).unwrap();
    }

    // Single-leg results.
    let q_emb = embedder.embed_query(QUERY).unwrap();
    let dense_results = dense.search(&q_emb, TOP_K).unwrap();
    let sparse_results = sparse.search(QUERY, TOP_K);

    // RRF-fused result.
    let strategy = FusionStrategy::RRF { k: 60.0 };
    let fused = strategy.fuse(&dense_results, &sparse_results);
    let fused_top: Vec<ChunkId> = fused.into_iter().take(TOP_K).map(|(id, _)| id).collect();

    let dense_top: Vec<ChunkId> = dense_results.iter().map(|(id, _)| *id).collect();
    let sparse_top: Vec<ChunkId> = sparse_results.iter().map(|(id, _)| *id).collect();

    let dense_ndcg = ndcg_at_k(&dense_top, &ground_truth, TOP_K);
    let sparse_ndcg = ndcg_at_k(&sparse_top, &ground_truth, TOP_K);
    let fused_ndcg = ndcg_at_k(&fused_top, &ground_truth, TOP_K);

    let max_leg = dense_ndcg.max(sparse_ndcg);

    assert!(
        fused_ndcg >= max_leg + 0.03,
        "FALSIFY-RERANK-RRF-001: RRF nDCG@{TOP_K} {fused_ndcg:.4} \
         did not beat max(dense={dense_ndcg:.4}, sparse={sparse_ndcg:.4}) = {max_leg:.4} \
         by ≥0.03 on the adversarial fixture. RRF is not adding signal beyond \
         what each leg provides alone.\n  \
         dense top: {dense_labels:?}\n  \
         sparse top: {sparse_labels:?}\n  \
         fused top: {fused_labels:?}",
        dense_labels = dense_top.iter().map(|id| labels[id]).collect::<Vec<_>>(),
        sparse_labels = sparse_top.iter().map(|id| labels[id]).collect::<Vec<_>>(),
        fused_labels = fused_top.iter().map(|id| labels[id]).collect::<Vec<_>>(),
    );
}

#[test]
fn ndcg_self_consistency() {
    // Sanity: nDCG of the ideal ordering is 1.0; nDCG of an
    // empty/zero-relevant ranking is 0.0. Catches a buggy harness
    // that silently passes the main gate.
    let mut gt = HashSet::new();
    let id_a = ChunkId(uuid::Uuid::from_u128(0xA));
    let id_b = ChunkId(uuid::Uuid::from_u128(0xB));
    let id_c = ChunkId(uuid::Uuid::from_u128(0xC));
    gt.insert(id_a);
    gt.insert(id_b);

    let ideal = vec![id_a, id_b, id_c];
    let nothing = vec![id_c];

    let n = ndcg_at_k(&ideal, &gt, 3);
    assert!((n - 1.0).abs() < 1e-4, "ideal ordering should give nDCG=1.0, got {n}");
    let z = ndcg_at_k(&nothing, &gt, 3);
    assert!((z - 0.0).abs() < f32::EPSILON, "zero-relevant top-k should give nDCG=0.0, got {z}");
}
