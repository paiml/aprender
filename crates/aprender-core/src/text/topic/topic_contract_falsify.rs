//! Topic Modeling (LDA) Contract Falsification Tests
//!
//! Popperian falsification of NLP spec §2.1.3 claims:
//!   - LDA produces valid probability distributions
//!   - Document-topic rows sum to ~1 (simplex constraint)
//!   - Topic-word rows sum to ~1 (simplex constraint)
//!   - Output shapes match (n_docs, n_topics) and (n_topics, n_terms)
//!   - Non-negative values only (probabilities)
//!   - Deterministic with fixed seed
//!
//! Five-Whys (PMAT-351):
//!   Why #1: topic module has unit tests but zero FALSIFY-LDA-* tests
//!   Why #2: unit tests check API, not probabilistic invariants
//!   Why #3: no provable-contract for LDA probability distributions
//!   Why #4: topic module was built before DbC methodology
//!   Why #5: Dirichlet simplex constraints not formally verified
//!
//! References:
//!   - Blei, D.M., et al. (2003). Latent Dirichlet Allocation. JMLR.
//!   - docs/specifications/nlp-models-techniques-spec.md §2.1.3
//!   - src/text/topic/mod.rs

use super::*;
use crate::primitives::Matrix;

fn make_dtm() -> Matrix<f64> {
    // 3 docs × 5 terms
    Matrix::from_vec(
        3,
        5,
        vec![
            2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0,
        ],
    )
    .expect("dtm")
}

// ============================================================================
// FALSIFY-LDA-001: Output shapes
// Contract: doc-topic is (n_docs, n_topics), topic-word is (n_topics, n_terms)
// ============================================================================

#[test]
fn falsify_lda_001_output_shapes() {
    let dtm = make_dtm();
    let n_topics = 2;
    let mut lda = LatentDirichletAllocation::new(n_topics).with_random_seed(42);

    lda.fit(&dtm, 20).expect("fit");

    let doc_topic = lda.document_topics().expect("doc_topic");
    assert_eq!(
        doc_topic.n_rows(),
        3,
        "FALSIFIED LDA-001: doc_topic rows {} != n_docs 3",
        doc_topic.n_rows()
    );
    assert_eq!(
        doc_topic.n_cols(),
        n_topics,
        "FALSIFIED LDA-001: doc_topic cols {} != n_topics {}",
        doc_topic.n_cols(),
        n_topics
    );

    let topic_word = lda.topic_words().expect("topic_word");
    assert_eq!(
        topic_word.n_rows(),
        n_topics,
        "FALSIFIED LDA-001: topic_word rows {} != n_topics {}",
        topic_word.n_rows(),
        n_topics
    );
    assert_eq!(
        topic_word.n_cols(),
        5,
        "FALSIFIED LDA-001: topic_word cols {} != n_terms 5",
        topic_word.n_cols()
    );
}

// ============================================================================
// FALSIFY-LDA-002: Non-negative values (probability constraint)
// Contract: all values in doc-topic and topic-word matrices are ≥ 0
// ============================================================================

#[test]
fn falsify_lda_002_non_negative_values() {
    let dtm = make_dtm();
    let mut lda = LatentDirichletAllocation::new(2).with_random_seed(42);
    lda.fit(&dtm, 20).expect("fit");

    let doc_topic = lda.document_topics().expect("doc_topic");
    for r in 0..doc_topic.n_rows() {
        for c in 0..doc_topic.n_cols() {
            let val = doc_topic.get(r, c);
            assert!(
                val >= 0.0,
                "FALSIFIED LDA-002: doc_topic[{r},{c}] = {val} is negative"
            );
        }
    }

    let topic_word = lda.topic_words().expect("topic_word");
    for r in 0..topic_word.n_rows() {
        for c in 0..topic_word.n_cols() {
            let val = topic_word.get(r, c);
            assert!(
                val >= 0.0,
                "FALSIFIED LDA-002: topic_word[{r},{c}] = {val} is negative"
            );
        }
    }
}

// ============================================================================
// FALSIFY-LDA-003: Simplex constraint (rows sum to ~1)
// Contract: each row of doc-topic and topic-word sums approximately to 1
// ============================================================================

#[test]
fn falsify_lda_003_doc_topic_simplex() {
    let dtm = make_dtm();
    let mut lda = LatentDirichletAllocation::new(2).with_random_seed(42);
    lda.fit(&dtm, 50).expect("fit");

    let doc_topic = lda.document_topics().expect("doc_topic");
    for r in 0..doc_topic.n_rows() {
        let row_sum: f64 = (0..doc_topic.n_cols()).map(|c| doc_topic.get(r, c)).sum();
        assert!(
            (row_sum - 1.0).abs() < 0.1,
            "FALSIFIED LDA-003: doc_topic row {} sums to {}, expected ~1.0",
            r,
            row_sum
        );
    }
}

#[test]
fn falsify_lda_003_topic_word_simplex() {
    let dtm = make_dtm();
    let mut lda = LatentDirichletAllocation::new(2).with_random_seed(42);
    lda.fit(&dtm, 50).expect("fit");

    let topic_word = lda.topic_words().expect("topic_word");
    for r in 0..topic_word.n_rows() {
        let row_sum: f64 = (0..topic_word.n_cols()).map(|c| topic_word.get(r, c)).sum();
        assert!(
            (row_sum - 1.0).abs() < 0.1,
            "FALSIFIED LDA-003: topic_word row {} sums to {}, expected ~1.0",
            r,
            row_sum
        );
    }
}

// ============================================================================
// FALSIFY-LDA-004: Determinism with fixed seed
// Contract: same DTM + same seed + same iterations → same result
// ============================================================================

#[test]
fn falsify_lda_004_determinism_fixed_seed() {
    let dtm = make_dtm();

    let mut lda1 = LatentDirichletAllocation::new(2).with_random_seed(42);
    lda1.fit(&dtm, 20).expect("fit1");
    let dt1 = lda1.document_topics().expect("dt1");

    let mut lda2 = LatentDirichletAllocation::new(2).with_random_seed(42);
    lda2.fit(&dtm, 20).expect("fit2");
    let dt2 = lda2.document_topics().expect("dt2");

    for r in 0..dt1.n_rows() {
        for c in 0..dt1.n_cols() {
            assert!(
                (dt1.get(r, c) - dt2.get(r, c)).abs() < 1e-10,
                "FALSIFIED LDA-004: non-deterministic at [{r},{c}]: {} vs {}",
                dt1.get(r, c),
                dt2.get(r, c)
            );
        }
    }
}

// ============================================================================
// FALSIFY-LDA-005: Finite values
// Contract: no NaN or infinity in output matrices
// ============================================================================

#[test]
fn falsify_lda_005_finite_values() {
    let dtm = make_dtm();
    let mut lda = LatentDirichletAllocation::new(2).with_random_seed(42);
    lda.fit(&dtm, 20).expect("fit");

    let doc_topic = lda.document_topics().expect("doc_topic");
    for r in 0..doc_topic.n_rows() {
        for c in 0..doc_topic.n_cols() {
            let val = doc_topic.get(r, c);
            assert!(
                val.is_finite(),
                "FALSIFIED LDA-005: doc_topic[{r},{c}] = {val} is not finite"
            );
        }
    }

    let topic_word = lda.topic_words().expect("topic_word");
    for r in 0..topic_word.n_rows() {
        for c in 0..topic_word.n_cols() {
            let val = topic_word.get(r, c);
            assert!(
                val.is_finite(),
                "FALSIFIED LDA-005: topic_word[{r},{c}] = {val} is not finite"
            );
        }
    }
}
