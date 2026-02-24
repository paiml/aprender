// =========================================================================
// FALSIFY-EMB: embedding-algebra-v1.yaml contract (code CodeEmbedding)
//
// Five-Whys (PMAT-354):
//   Why 1: code had CodeEmbedding tests but zero FALSIFY-EMB-* tests
//   Why 2: unit tests verify API shapes, not algebraic invariants
//   Why 3: no mapping from embedding-algebra-v1.yaml to code test names
//   Why 4: code/embedding predates the provable-contracts YAML convention
//   Why 5: cosine similarity was "obviously correct" (textbook dot/norm)
//
// References:
//   - provable-contracts/contracts/embedding-algebra-v1.yaml
//   - Alon et al. (2019) "code2vec: Learning Distributed Representations of Code"
// =========================================================================

use super::*;
use crate::code::ast::{Token, TokenType};
use crate::code::path::AstPath;

/// FALSIFY-EMB-001: Cosine self-similarity = 1.0
///
/// cos(x, x) = 1.0 for any non-zero embedding.
#[test]
fn falsify_emb_001_cosine_self_similarity() {
    let test_vecs = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.5, -0.3, 0.8, 0.1],
        vec![100.0, -200.0, 300.0],
    ];

    for (idx, v) in test_vecs.iter().enumerate() {
        let emb = CodeEmbedding::new(Vector::from_vec(v.clone()));
        let sim = emb.cosine_similarity(&emb);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "FALSIFIED EMB-001: cosine_similarity(x, x) = {sim} != 1.0 for test case {idx}"
        );
    }
}

/// FALSIFY-EMB-002: Dimension mismatch returns 0.0
///
/// When embeddings have different dimensions, cosine_similarity must return 0.0.
#[test]
fn falsify_emb_002_dimension_mismatch() {
    let a = CodeEmbedding::new(Vector::from_vec(vec![1.0, 2.0, 3.0]));
    let b = CodeEmbedding::new(Vector::from_vec(vec![1.0, 2.0]));

    let sim = a.cosine_similarity(&b);
    assert!(
        sim.abs() < 1e-10,
        "FALSIFIED EMB-002: dimension mismatch should return 0.0, got {sim}"
    );
}

/// FALSIFY-EMB-003: Attention weights sum to 1.0 (softmax invariant)
///
/// aggregate_paths must produce attention weights that sum to 1.0.
#[test]
fn falsify_emb_003_attention_weights_sum_to_one() {
    let encoder = Code2VecEncoder::new(32).with_seed(42);

    let paths = vec![
        AstPath::new(
            Token::new(TokenType::Identifier, "x"),
            vec![AstNodeType::Parameter, AstNodeType::Function],
            Token::new(TokenType::Identifier, "y"),
        ),
        AstPath::new(
            Token::new(TokenType::Identifier, "a"),
            vec![AstNodeType::Return],
            Token::new(TokenType::Identifier, "b"),
        ),
        AstPath::new(
            Token::new(TokenType::Identifier, "foo"),
            vec![
                AstNodeType::Function,
                AstNodeType::Block,
                AstNodeType::Return,
            ],
            Token::new(TokenType::Identifier, "bar"),
        ),
    ];

    let embedding = encoder.aggregate_paths(&paths);
    let weights = embedding
        .attention_weights()
        .expect("FALSIFIED EMB-003: no attention weights returned");

    assert_eq!(
        weights.len(),
        paths.len(),
        "FALSIFIED EMB-003: attention weights len {} != paths len {}",
        weights.len(),
        paths.len()
    );

    let sum: f64 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "FALSIFIED EMB-003: attention weights sum = {sum} != 1.0"
    );

    // All weights must be non-negative (softmax output)
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            w >= 0.0,
            "FALSIFIED EMB-003: attention weight[{i}] = {w} < 0"
        );
    }
}

/// FALSIFY-EMB-004: Deterministic encoding — same seed + same path = same embedding
#[test]
fn falsify_emb_004_deterministic_encoding() {
    let path = AstPath::new(
        Token::new(TokenType::Identifier, "alpha"),
        vec![AstNodeType::Function, AstNodeType::Return],
        Token::new(TokenType::Identifier, "beta"),
    );

    let enc1 = Code2VecEncoder::new(64).with_seed(777);
    let enc2 = Code2VecEncoder::new(64).with_seed(777);

    let emb1 = enc1.encode_path(&path);
    let emb2 = enc2.encode_path(&path);

    for (i, (&a, &b)) in emb1.iter().zip(emb2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "FALSIFIED EMB-004: encode_path differs at dim {i}: {a} vs {b}"
        );
    }
}

/// FALSIFY-EMB-005: aggregate_paths output dimension matches encoder dim
#[test]
fn falsify_emb_005_aggregate_output_dimension() {
    let dims = [4, 16, 64, 128];
    for &dim in &dims {
        let encoder = Code2VecEncoder::new(dim);
        let paths = vec![AstPath::new(
            Token::new(TokenType::Identifier, "v"),
            vec![AstNodeType::Variable],
            Token::new(TokenType::Identifier, "w"),
        )];

        let embedding = encoder.aggregate_paths(&paths);
        assert_eq!(
            embedding.dim(),
            dim,
            "FALSIFIED EMB-005: aggregate dim {} != encoder dim {dim}",
            embedding.dim()
        );
    }
}

/// FALSIFY-EMB-006: Empty paths → zero vector
///
/// aggregate_paths([]) must return a zero vector of the correct dimension.
#[test]
fn falsify_emb_006_empty_paths_zero_vector() {
    let encoder = Code2VecEncoder::new(32);
    let embedding = encoder.aggregate_paths(&[]);

    assert_eq!(
        embedding.dim(),
        32,
        "FALSIFIED EMB-006: empty aggregate dim {} != 32",
        embedding.dim()
    );

    for (i, &val) in embedding.vector().as_slice().iter().enumerate() {
        assert!(
            val.abs() < 1e-15,
            "FALSIFIED EMB-006: empty aggregate [{i}] = {val} != 0.0"
        );
    }
}

/// FALSIFY-EMB-007: Zero vector cosine similarity = 0.0
///
/// Degenerate case: cos(0, x) = 0.0 (guarded by epsilon check).
#[test]
fn falsify_emb_007_zero_vector_cosine() {
    let zero = CodeEmbedding::new(Vector::from_vec(vec![0.0, 0.0, 0.0]));
    let nonzero = CodeEmbedding::new(Vector::from_vec(vec![1.0, 2.0, 3.0]));

    let sim = zero.cosine_similarity(&nonzero);
    assert!(
        sim.abs() < 1e-10,
        "FALSIFIED EMB-007: cos(0, x) = {sim} != 0.0"
    );

    let sim2 = zero.cosine_similarity(&zero);
    assert!(
        sim2.abs() < 1e-10,
        "FALSIFIED EMB-007: cos(0, 0) = {sim2} != 0.0"
    );
}
