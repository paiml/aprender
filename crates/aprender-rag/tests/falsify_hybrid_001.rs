//! FALSIFY-HYBRID-001 — hybrid retrieval recall@k beats
//! max(dense recall@k, sparse recall@k) by ≥5 percentage points on
//! a hand-crafted adversarial fixture where the two legs cover
//! disjoint subsets of the ground-truth set.
//!
//! Contract: `contracts/apr-hybrid-retrieval-v1.yaml` v1.2.0.
//!
//! Discharge strategy: build a 5-doc corpus where ONE doc (d1) sits
//! at rank 1 in BOTH legs, and the remaining relevant docs split
//! across legs. The RRF math then puts d1 alone at the top, with
//! both unique-leg relevant docs (d2 from dense, d3 from sparse)
//! tied above the unique-leg irrelevant docs (x1 from dense, x2
//! from sparse). With `candidates_per_source=3`, the cos=0 docs
//! never enter the dense candidate list — avoiding the trap where
//! sparse-only relevant docs accidentally accumulate dense rank-N
//! contributions and tie with irrelevant ones.
//!
//! Concrete fixture:
//! - d1 (relevant): perfect cosine + highest BM25 tf=3 → rank 1
//!   dense AND rank 1 sparse → RRF = 2/61 ≈ 0.0328 (DOMINATES).
//! - d2 (relevant, semantic): cosine ≈ 0.995 → dense rank 2.
//!   Content has no query keyword → absent from sparse list.
//!   RRF = 1/62 ≈ 0.0161.
//! - d3 (relevant, lexical): cos=0 (orthogonal embedding, with
//!   `candidates_per_source=3` it never enters the dense top-3).
//!   Content "alpha alpha" tf=2 → sparse rank 2.
//!   RRF = 1/62 ≈ 0.0161.
//! - x1 (irrelevant): cosine ≈ 0.928 → dense rank 3. No keyword.
//!   RRF = 1/63 ≈ 0.0159.
//! - x2 (irrelevant): cos=0, content "alpha ..." with tf=1 in a
//!   longer doc → sparse rank 3. RRF = 1/63 ≈ 0.0159.
//!
//! Sort by RRF desc: d1 (0.0328) > {d2, d3} tied at 0.0161 >
//! {x1, x2} tied at 0.0159. Top-3 hybrid is {d1, d2, d3} regardless
//! of how the tie within {d2, d3} resolves (because both beat x1/x2
//! cleanly). Single-leg top-3: dense = {d1, d2, x1} (recall 2/3),
//! sparse = {d1, d3, x2} (recall 2/3). Hybrid recall = 3/3 = 1.000,
//! a +0.333 gain — well above the 0.05 contractual threshold.

#![allow(clippy::unwrap_used)]

use aprender_rag::embed::Embedder;
use aprender_rag::error::Result;
use aprender_rag::fusion::FusionStrategy;
use aprender_rag::index::{BM25Index, VectorStore};
use aprender_rag::retrieve::{HybridRetriever, HybridRetrieverConfig};
use aprender_rag::{Chunk, ChunkId, DocumentId};
use std::collections::{BTreeSet, HashMap, HashSet};

const DIM: usize = 4;
const TOP_K: usize = 3;
const CANDIDATES_PER_SOURCE: usize = 3;
// Two-word query: BM25 tokenizes to {"find", "alpha"}; "find" matches
// no doc, so sparse scoring is determined entirely by "alpha"
// occurrence. The full string differs from any doc content so the
// embedder map can distinguish it from d6's content "alpha" alone
// (which would otherwise overwrite the query embedding).
const QUERY: &str = "find alpha";

/// Hand-controlled embedder: maps known text → fixed vector.
/// Unknown text returns the zero vector (so accidentally querying
/// for missing strings produces clearly-wrong scores instead of
/// silently working).
#[derive(Clone)]
struct FixedEmbedder {
    map: HashMap<String, [f32; DIM]>,
}

impl FixedEmbedder {
    fn new() -> Self {
        let mut map = HashMap::new();
        // Query embedding: aligned with "x"-axis (1, 0, 0, 0).
        map.insert(QUERY.to_string(), [1.0, 0.0, 0.0, 0.0]);
        // d1: rank 1 in BOTH legs (perfect cosine + max BM25 tf).
        map.insert("alpha alpha alpha".to_string(), [1.0, 0.0, 0.0, 0.0]);
        // d2: dense-only relevant (semantic match, no keyword).
        map.insert("horse zebra giraffe".to_string(), [1.0, 0.1, 0.0, 0.0]);
        // d3: sparse-only relevant (orthogonal embedding, lexical match).
        map.insert("alpha alpha".to_string(), [0.0, 0.0, 1.0, 0.0]);
        // x1: dense-only irrelevant (less aligned, no keyword).
        map.insert("monkey rabbit".to_string(), [1.0, 0.4, 0.0, 0.0]);
        // x2: sparse-only irrelevant (long doc with one keyword).
        map.insert("alpha noise filler text padding".to_string(), [0.0, 0.0, 1.0, 0.3]);
        Self { map }
    }
}

impl Embedder for FixedEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let v = self.map.get(text).copied().unwrap_or([0.0; DIM]);
        Ok(v.to_vec())
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

/// Build retriever, return (retriever, ground_truth_chunk_ids,
/// label_lookup). The label_lookup maps ChunkId → human label
/// ("d1".."d8") so test failures print a recognisable trace.
fn build_retriever(
) -> (HybridRetriever<FixedEmbedder>, HashSet<ChunkId>, HashMap<ChunkId, &'static str>) {
    let embedder = FixedEmbedder::new();
    let dense = VectorStore::with_dimension(DIM);
    let sparse = BM25Index::new();

    let config = HybridRetrieverConfig {
        // Critical: candidates_per_source = 3 (= top_k) so each leg
        // returns ONLY its top-3 candidates. With a larger value,
        // dense returns cos=0 docs at low ranks, which adds RRF
        // contributions to sparse-only items and breaks the
        // tie-structure we rely on.
        candidates_per_source: CANDIDATES_PER_SOURCE,
        fusion: FusionStrategy::RRF { k: 60.0 },
        use_dense: true,
        use_sparse: true,
    };
    let mut retriever = HybridRetriever::new(dense, sparse, embedder.clone()).with_config(config);

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
        retriever.index(chunk).unwrap();
    }

    (retriever, ground_truth, labels)
}

fn recall_at_k(hits: &[ChunkId], ground_truth: &HashSet<ChunkId>) -> f32 {
    let denom = ground_truth.len();
    assert!(denom > 0);
    let num = hits.iter().filter(|id| ground_truth.contains(id)).count();
    #[allow(clippy::cast_precision_loss)]
    let n = num as f32;
    #[allow(clippy::cast_precision_loss)]
    let d = denom as f32;
    n / d
}

fn label_set(hits: &[ChunkId], labels: &HashMap<ChunkId, &'static str>) -> Vec<&'static str> {
    hits.iter().map(|id| labels.get(id).copied().unwrap_or("???")).collect()
}

#[test]
fn hybrid_beats_max_of_legs_by_5pts() {
    let (retriever, ground_truth, labels) = build_retriever();

    let dense_results = retriever.retrieve_dense(QUERY, TOP_K).unwrap();
    let sparse_results = retriever.retrieve_sparse(QUERY, TOP_K).unwrap();
    let hybrid_results = retriever.retrieve(QUERY, TOP_K).unwrap();

    let dense_ids: Vec<ChunkId> = dense_results.iter().map(|r| r.chunk.id).collect();
    let sparse_ids: Vec<ChunkId> = sparse_results.iter().map(|r| r.chunk.id).collect();
    let hybrid_ids: Vec<ChunkId> = hybrid_results.iter().map(|r| r.chunk.id).collect();

    let dense_recall = recall_at_k(&dense_ids, &ground_truth);
    let sparse_recall = recall_at_k(&sparse_ids, &ground_truth);
    let hybrid_recall = recall_at_k(&hybrid_ids, &ground_truth);

    let max_leg = dense_recall.max(sparse_recall);

    assert!(
        hybrid_recall >= max_leg + 0.05,
        "FALSIFY-HYBRID-001: hybrid recall@{TOP_K} {hybrid_recall:.4} \
         did not beat max(dense={dense_recall:.4}, sparse={sparse_recall:.4}) = {max_leg:.4} \
         by ≥0.05. Hybrid is statistically equivalent to one of the legs \
         on this fixture, so the fusion is not adding signal.\n  \
         dense top-{TOP_K}:  {dense_labels:?}\n  \
         sparse top-{TOP_K}: {sparse_labels:?}\n  \
         hybrid top-{TOP_K}: {hybrid_labels:?}\n  \
         ground truth (relevant): {gt_count}",
        dense_labels = label_set(&dense_ids, &labels),
        sparse_labels = label_set(&sparse_ids, &labels),
        hybrid_labels = label_set(&hybrid_ids, &labels),
        gt_count = ground_truth.len(),
    );
}

#[test]
fn fixture_legs_cover_overlapping_but_distinct_subsets() {
    // Sanity: confirm the fixture actually behaves as designed —
    // d1 must be rank 1 in BOTH legs; dense top-3 = {d1, d2, x1};
    // sparse top-3 = {d1, d3, x2}. If this drifts, the main gate's
    // tie-structure assumption breaks silently.
    let (retriever, _, labels) = build_retriever();

    let dense_top3: BTreeSet<&'static str> =
        retriever.retrieve_dense(QUERY, 3).unwrap().iter().map(|r| labels[&r.chunk.id]).collect();
    let sparse_top3: BTreeSet<&'static str> =
        retriever.retrieve_sparse(QUERY, 3).unwrap().iter().map(|r| labels[&r.chunk.id]).collect();

    let want_dense: BTreeSet<&'static str> = ["d1", "d2", "x1"].into_iter().collect();
    let want_sparse: BTreeSet<&'static str> = ["d1", "d3", "x2"].into_iter().collect();

    assert_eq!(
        dense_top3, want_dense,
        "fixture sanity: dense top-3 should be {{d1,d2,x1}}, got {dense_top3:?}",
    );
    assert_eq!(
        sparse_top3, want_sparse,
        "fixture sanity: sparse top-3 should be {{d1,d3,x2}}, got {sparse_top3:?}",
    );
}
