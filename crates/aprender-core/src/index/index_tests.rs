pub(crate) use super::*;

#[test]
fn test_cross_encoder_score() {
    let ce = default_cross_encoder();
    let query = vec![1.0, 0.0];
    let doc1 = vec![1.0, 0.0];
    let doc2 = vec![0.0, 1.0];

    let score1 = ce.score(&query, &doc1);
    let score2 = ce.score(&query, &doc2);
    assert!(score1 > score2);
}

#[test]
fn test_cross_encoder_rerank() {
    let ce = default_cross_encoder();
    let query = vec![1.0, 0.0];
    let candidates = vec![
        ("doc1", vec![0.0, 1.0]),
        ("doc2", vec![1.0, 0.0]),
        ("doc3", vec![0.5, 0.5]),
    ];

    let reranked = ce.rerank(&query, &candidates, 2);
    assert_eq!(reranked.len(), 2);
    assert_eq!(*reranked[0].0, "doc2");
}

#[test]
fn test_hybrid_search_fuse() {
    let hs = HybridSearch::new(0.6, 0.4);
    let dense = vec![("a".to_string(), 0.9), ("b".to_string(), 0.5)];
    let sparse = vec![("b".to_string(), 1.0), ("c".to_string(), 0.7)];

    let fused = hs.fuse_scores(&dense, &sparse, 3);
    assert!(fused.len() <= 3);
}

#[test]
fn test_hybrid_search_rrf() {
    let hs = HybridSearch::default();
    let rankings = vec![
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec!["b".to_string(), "a".to_string(), "d".to_string()],
    ];

    let fused = hs.rrf_fuse(&rankings, 60.0, 3);
    assert_eq!(fused.len(), 3);
}

#[test]
fn test_hybrid_search_default() {
    let hs = HybridSearch::default();
    assert!((hs.dense_weight() - 0.7).abs() < 1e-6);
    assert!((hs.sparse_weight() - 0.3).abs() < 1e-6);
}

// BiEncoder Tests

#[test]
fn test_bi_encoder_cosine() {
    let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::Cosine);
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0];
    let c = vec![0.0, 1.0];

    let sim_ab = encoder.similarity(&a, &b);
    let sim_ac = encoder.similarity(&a, &c);

    assert!((sim_ab - 1.0).abs() < 1e-6); // Same direction
    assert!(sim_ac.abs() < 1e-6); // Orthogonal
}

#[test]
fn test_bi_encoder_dot_product() {
    let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::DotProduct);
    let a = vec![2.0, 3.0];
    let b = vec![1.0, 2.0];

    let sim = encoder.similarity(&a, &b);
    assert!((sim - 8.0).abs() < 1e-6); // 2*1 + 3*2 = 8
}

#[test]
fn test_bi_encoder_euclidean() {
    let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::Euclidean);
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];

    let sim = encoder.similarity(&a, &b);
    assert!((sim - (-5.0)).abs() < 1e-6); // -sqrt(9+16) = -5
}

#[test]
fn test_bi_encoder_encode() {
    let encoder = BiEncoder::new(
        |x: &[f32]| x.iter().map(|&v| v * 2.0).collect(),
        SimilarityMetric::Cosine,
    );

    let input = vec![1.0, 2.0, 3.0];
    let encoded = encoder.encode(&input);

    assert_eq!(encoded, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_bi_encoder_encode_batch() {
    let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::Cosine);
    let inputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let encoded = encoder.encode_batch(&inputs);
    assert_eq!(encoded.len(), 2);
}

#[test]
fn test_bi_encoder_retrieve() {
    let encoder = BiEncoder::new(|x: &[f32]| x.to_vec(), SimilarityMetric::Cosine);
    let corpus = vec![
        ("doc1", vec![1.0, 0.0]),
        ("doc2", vec![0.0, 1.0]),
        ("doc3", vec![0.707, 0.707]),
    ];

    let query = vec![1.0, 0.0];
    let results = encoder.retrieve(&query, &corpus, 2);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "doc1"); // Exact match
}

// ColBERT Tests

#[test]
fn test_colbert_creation() {
    let colbert = ColBERT::new(128);
    assert_eq!(colbert.embedding_dim(), 128);
}

#[test]
fn test_colbert_maxsim_identical() {
    let colbert = ColBERT::new(4);
    let query = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    let doc = query.clone();

    let score = colbert.maxsim(&query, &doc);
    assert!((score - 2.0).abs() < 1e-5); // 1.0 + 1.0
}

#[test]
fn test_colbert_maxsim_different() {
    let colbert = ColBERT::new(4);
    let query = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let doc = vec![vec![0.0, 1.0, 0.0, 0.0]];

    let score = colbert.maxsim(&query, &doc);
    assert!(score.abs() < 1e-5); // Orthogonal
}

#[test]
fn test_colbert_maxsim_empty() {
    let colbert = ColBERT::new(4);
    let empty: Vec<Vec<f32>> = vec![];
    let doc = vec![vec![1.0, 0.0, 0.0, 0.0]];

    assert_eq!(colbert.maxsim(&empty, &doc), 0.0);
    assert_eq!(colbert.maxsim(&doc, &empty), 0.0);
}

#[test]
fn test_colbert_score_documents() {
    let colbert = ColBERT::new(2);
    let query = vec![vec![1.0, 0.0]];
    let docs = vec![vec![vec![1.0, 0.0]], vec![vec![0.0, 1.0]]];

    let scores = colbert.score_documents(&query, &docs);
    assert_eq!(scores.len(), 2);
    assert!(scores[0] > scores[1]); // First doc matches better
}

#[test]
fn test_colbert_retrieve() {
    let colbert = ColBERT::new(2);
    let query = vec![vec![1.0, 0.0], vec![0.707, 0.707]];
    let corpus = vec![
        ("doc1", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
        ("doc2", vec![vec![0.0, 1.0]]),
    ];

    let results = colbert.retrieve(&query, &corpus, 2);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_similarity_metric_equality() {
    assert_eq!(SimilarityMetric::Cosine, SimilarityMetric::Cosine);
    assert_ne!(SimilarityMetric::Cosine, SimilarityMetric::DotProduct);
}
