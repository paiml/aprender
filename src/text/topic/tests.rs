pub(crate) use super::*;

#[test]
fn test_lda_fit() {
    let dtm = Matrix::from_vec(
        3,
        5,
        vec![
            2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0,
        ],
    )
    .expect("matrix should succeed");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 10).expect("fit should succeed");

    let doc_topics = lda.document_topics().expect("should have doc topics");
    assert_eq!(doc_topics.n_rows(), 3);
    assert_eq!(doc_topics.n_cols(), 2);
}

#[test]
fn test_lda_top_words() {
    let dtm =
        Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0]).expect("matrix should succeed");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 10).expect("fit should succeed");

    let vocab = vec![
        "word1".to_string(),
        "word2".to_string(),
        "word3".to_string(),
    ];
    let top_words = lda.top_words(&vocab, 2).expect("top words should succeed");

    assert_eq!(top_words.len(), 2); // 2 topics
    assert_eq!(top_words[0].len(), 2); // 2 words per topic
}

// =========================================================================
// Extended coverage tests
// =========================================================================

#[test]
fn test_lda_with_random_seed() {
    let lda = LatentDirichletAllocation::new(3).with_random_seed(12345);
    assert_eq!(lda.random_seed, 12345);
    assert_eq!(lda.n_topics, 3);
}

#[test]
fn test_lda_empty_matrix_error() {
    let dtm = Matrix::from_vec(0, 0, vec![]).expect("empty matrix");
    let mut lda = LatentDirichletAllocation::new(2);
    let result = lda.fit(&dtm, 10);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("empty"));
}

#[test]
fn test_document_topics_not_fitted() {
    let lda = LatentDirichletAllocation::new(2);
    let result = lda.document_topics();
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not fitted"));
}

#[test]
fn test_topic_words_not_fitted() {
    let lda = LatentDirichletAllocation::new(2);
    let result = lda.topic_words();
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("not fitted"));
}

#[test]
fn test_top_words_vocab_mismatch() {
    let dtm =
        Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0]).expect("matrix should succeed");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 5).expect("fit should succeed");

    // Vocabulary with wrong size (should be 3, providing 2)
    let vocab = vec!["word1".to_string(), "word2".to_string()];
    let result = lda.top_words(&vocab, 2);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Vocabulary size"));
}

#[test]
fn test_lda_debug() {
    let lda = LatentDirichletAllocation::new(5);
    let debug_str = format!("{:?}", lda);
    assert!(debug_str.contains("LatentDirichletAllocation"));
    assert!(debug_str.contains("n_topics"));
}

#[test]
fn test_lda_topic_words_access() {
    let dtm =
        Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0]).expect("matrix should succeed");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 5).expect("fit should succeed");

    let topic_words = lda.topic_words().expect("should have topic words");
    assert_eq!(topic_words.n_rows(), 2); // 2 topics
    assert_eq!(topic_words.n_cols(), 3); // 3 terms

    // Each row should sum to approximately 1 (normalized)
    for row in 0..topic_words.n_rows() {
        let mut sum = 0.0;
        for col in 0..topic_words.n_cols() {
            sum += topic_words.get(row, col);
        }
        assert!((sum - 1.0).abs() < 0.01, "Row {} sum: {}", row, sum);
    }
}

#[test]
fn test_lda_single_topic() {
    let dtm =
        Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0]).expect("matrix should succeed");

    let mut lda = LatentDirichletAllocation::new(1);
    lda.fit(&dtm, 5).expect("fit should succeed");

    let doc_topics = lda.document_topics().expect("should have doc topics");
    assert_eq!(doc_topics.n_cols(), 1);

    // With one topic, all docs should have 1.0 weight for that topic
    for row in 0..doc_topics.n_rows() {
        assert!((doc_topics.get(row, 0) - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_lda_many_topics() {
    let dtm = Matrix::from_vec(
        3,
        4,
        vec![2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0],
    )
    .expect("matrix should succeed");

    let mut lda = LatentDirichletAllocation::new(3);
    lda.fit(&dtm, 5).expect("fit should succeed");

    let doc_topics = lda.document_topics().expect("should have doc topics");
    assert_eq!(doc_topics.n_rows(), 3);
    assert_eq!(doc_topics.n_cols(), 3);
}

#[test]
fn test_lda_sparse_matrix() {
    // Matrix with many zeros
    let dtm = Matrix::from_vec(
        3,
        5,
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .expect("matrix should succeed");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 5).expect("fit should succeed");

    // Should still work with sparse data
    let doc_topics = lda.document_topics().expect("should have doc topics");
    assert_eq!(doc_topics.n_rows(), 3);
}

#[test]
fn test_lda_top_words_all() {
    let dtm = Matrix::from_vec(2, 4, vec![2.0, 1.0, 0.0, 0.5, 0.0, 1.0, 2.0, 0.5])
        .expect("matrix should succeed");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 10).expect("fit should succeed");

    let vocab = vec![
        "alpha".to_string(),
        "beta".to_string(),
        "gamma".to_string(),
        "delta".to_string(),
    ];

    // Request all words
    let top_words = lda.top_words(&vocab, 4).expect("top words should succeed");
    assert_eq!(top_words.len(), 2);
    assert_eq!(top_words[0].len(), 4);
    assert_eq!(top_words[1].len(), 4);
}

#[test]
fn test_lda_reproducibility() {
    let dtm =
        Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0]).expect("matrix should succeed");

    // Fit with same seed twice
    let mut lda1 = LatentDirichletAllocation::new(2).with_random_seed(42);
    lda1.fit(&dtm, 5).expect("fit should succeed");

    let mut lda2 = LatentDirichletAllocation::new(2).with_random_seed(42);
    lda2.fit(&dtm, 5).expect("fit should succeed");

    let topics1 = lda1.document_topics().expect("topics");
    let topics2 = lda2.document_topics().expect("topics");

    // Same seed should give same results
    for row in 0..topics1.n_rows() {
        for col in 0..topics1.n_cols() {
            let diff = (topics1.get(row, col) - topics2.get(row, col)).abs();
            assert!(diff < 1e-10, "Results differ at ({}, {})", row, col);
        }
    }
}

#[test]
fn test_normalize_rows_zero_sum() {
    let mut data = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    LatentDirichletAllocation::normalize_rows(&mut data, 2, 3);
    // Zero rows should stay zero
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_pseudo_random_range() {
    let lda = LatentDirichletAllocation::new(2).with_random_seed(123);
    for idx in 0..100 {
        let val = lda.pseudo_random(idx);
        assert!(
            val >= 0.0 && val < 1.0,
            "Value {} out of range: {}",
            idx,
            val
        );
    }
}

#[test]
fn test_lda_default_random_seed() {
    let lda = LatentDirichletAllocation::new(3);
    assert_eq!(lda.random_seed, 42); // Default seed
}

#[test]
fn test_lda_fit_with_zero_counts() {
    // Matrix where some documents have zero counts in certain terms
    let dtm = Matrix::from_vec(
        3,
        4,
        vec![
            0.0, 0.0, 1.0, 2.0, // Doc 0: no words in first two terms
            1.0, 2.0, 0.0, 0.0, // Doc 1: no words in last two terms
            0.0, 1.0, 1.0, 0.0, // Doc 2: sparse
        ],
    )
    .expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 10).expect("fit should succeed");

    let doc_topics = lda.document_topics().expect("should have topics");
    // Each row should sum to 1
    for row in 0..doc_topics.n_rows() {
        let mut sum = 0.0;
        for col in 0..doc_topics.n_cols() {
            sum += doc_topics.get(row, col);
        }
        assert!((sum - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_lda_fit_single_iteration() {
    let dtm = Matrix::from_vec(2, 3, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 1).expect("fit should succeed");

    // Even with 1 iteration, we should have valid distributions
    let topics = lda.document_topics().expect("topics");
    assert_eq!(topics.n_rows(), 2);
}

#[test]
fn test_lda_fit_zero_iterations() {
    let dtm = Matrix::from_vec(2, 3, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    // With zero iterations, only initialization happens
    lda.fit(&dtm, 0).expect("fit should succeed");

    let topics = lda.document_topics().expect("topics");
    assert_eq!(topics.n_rows(), 2);
}

#[test]
fn test_lda_fit_many_iterations() {
    let dtm = Matrix::from_vec(
        3,
        4,
        vec![3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
    )
    .expect("matrix");

    let mut lda = LatentDirichletAllocation::new(3);
    // More iterations should converge
    lda.fit(&dtm, 50).expect("fit should succeed");

    let topics = lda.topic_words().expect("topic words");
    assert_eq!(topics.n_rows(), 3);
    assert_eq!(topics.n_cols(), 4);
}

#[test]
fn test_lda_large_document_term_matrix() {
    // Larger matrix to test performance
    let n_docs = 10;
    let n_terms = 20;
    let mut data = vec![0.0; n_docs * n_terms];
    for i in 0..n_docs {
        for j in 0..n_terms {
            // Create some pattern
            data[i * n_terms + j] = if (i + j) % 3 == 0 { 2.0 } else { 0.0 };
        }
    }

    let dtm = Matrix::from_vec(n_docs, n_terms, data).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(4);
    lda.fit(&dtm, 5).expect("fit should succeed");

    let doc_topics = lda.document_topics().expect("doc topics");
    assert_eq!(doc_topics.n_rows(), n_docs);
    assert_eq!(doc_topics.n_cols(), 4);
}

#[test]
fn test_top_words_more_than_available() {
    let dtm = Matrix::from_vec(2, 3, vec![1.0, 2.0, 0.0, 0.0, 1.0, 2.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 5).expect("fit should succeed");

    let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    // Request more words than available
    let top_words = lda.top_words(&vocab, 10).expect("top words");
    assert_eq!(top_words.len(), 2);
    // Should truncate to available count
    assert_eq!(top_words[0].len(), 3);
    assert_eq!(top_words[1].len(), 3);
}

#[test]
fn test_top_words_request_one() {
    let dtm = Matrix::from_vec(2, 3, vec![2.0, 1.0, 0.0, 0.0, 1.0, 2.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 5).expect("fit should succeed");

    let vocab = vec![
        "first".to_string(),
        "second".to_string(),
        "third".to_string(),
    ];

    let top_words = lda.top_words(&vocab, 1).expect("top words");
    assert_eq!(top_words.len(), 2);
    assert_eq!(top_words[0].len(), 1);
    assert_eq!(top_words[1].len(), 1);
}

#[test]
fn test_top_words_request_zero() {
    let dtm = Matrix::from_vec(2, 3, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 5).expect("fit should succeed");

    let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    let top_words = lda.top_words(&vocab, 0).expect("top words");
    assert_eq!(top_words.len(), 2);
    assert_eq!(top_words[0].len(), 0);
    assert_eq!(top_words[1].len(), 0);
}

#[test]
fn test_lda_different_seeds_initialization() {
    // Test that different seeds initialize differently (before convergence)
    let lda1 = LatentDirichletAllocation::new(2).with_random_seed(1);
    let lda2 = LatentDirichletAllocation::new(2).with_random_seed(999);

    // Different seeds should give different pseudo-random values
    let mut any_different = false;
    for idx in 0..10 {
        if (lda1.pseudo_random(idx) - lda2.pseudo_random(idx)).abs() > 1e-10 {
            any_different = true;
            break;
        }
    }
    assert!(
        any_different,
        "Different seeds should produce different initializations"
    );
}

#[test]
fn test_normalize_rows_single_row() {
    let mut data = vec![1.0, 2.0, 3.0];
    LatentDirichletAllocation::normalize_rows(&mut data, 1, 3);
    let sum: f64 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_normalize_rows_multiple_rows() {
    let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    LatentDirichletAllocation::normalize_rows(&mut data, 2, 3);

    // First row: 1+2+3=6, so 1/6, 2/6, 3/6
    assert!((data[0] - 1.0 / 6.0).abs() < 1e-10);
    assert!((data[1] - 2.0 / 6.0).abs() < 1e-10);
    assert!((data[2] - 3.0 / 6.0).abs() < 1e-10);

    // Second row: 4+5+6=15, so 4/15, 5/15, 6/15
    assert!((data[3] - 4.0 / 15.0).abs() < 1e-10);
    assert!((data[4] - 5.0 / 15.0).abs() < 1e-10);
    assert!((data[5] - 6.0 / 15.0).abs() < 1e-10);
}

#[test]
fn test_normalize_rows_very_small_sum() {
    let mut data = vec![1e-15, 1e-15, 1e-15];
    LatentDirichletAllocation::normalize_rows(&mut data, 1, 3);
    // Sum is 3e-15 which is < 1e-10, so values should remain unchanged
    assert!(data[0] < 1e-14);
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
