
#[test]
fn test_pseudo_random_deterministic() {
    let lda1 = LatentDirichletAllocation::new(2).with_random_seed(42);
    let lda2 = LatentDirichletAllocation::new(2).with_random_seed(42);

    for idx in 0..50 {
        assert_eq!(lda1.pseudo_random(idx), lda2.pseudo_random(idx));
    }
}

#[test]
fn test_pseudo_random_different_seeds() {
    let lda1 = LatentDirichletAllocation::new(2).with_random_seed(1);
    let lda2 = LatentDirichletAllocation::new(2).with_random_seed(2);

    // At least one value should differ
    let mut any_different = false;
    for idx in 0..10 {
        if (lda1.pseudo_random(idx) - lda2.pseudo_random(idx)).abs() > 1e-10 {
            any_different = true;
            break;
        }
    }
    assert!(any_different);
}

#[test]
fn test_lda_fit_all_zero_document() {
    // One document has all zeros
    let dtm = Matrix::from_vec(
        3,
        3,
        vec![
            1.0, 2.0, 0.0, 0.0, 0.0, 0.0, // All zeros
            0.0, 1.0, 2.0,
        ],
    )
    .expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 10).expect("fit should succeed");

    let doc_topics = lda.document_topics().expect("topics");
    assert_eq!(doc_topics.n_rows(), 3);
}

#[test]
fn test_lda_fit_uniform_distribution() {
    // All documents have same word distribution
    let dtm = Matrix::from_vec(
        3,
        4,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 10).expect("fit should succeed");

    let topic_words = lda.topic_words().expect("topic words");
    // Each topic should have similar word distributions
    for topic in 0..2 {
        let mut sum = 0.0;
        for word in 0..4 {
            sum += topic_words.get(topic, word);
        }
        assert!((sum - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_lda_top_words_with_ties() {
    // Create a situation where multiple words have same probability
    let dtm = Matrix::from_vec(2, 4, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(1);
    lda.fit(&dtm, 10).expect("fit");

    let vocab = vec![
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
    ];
    let top_words = lda.top_words(&vocab, 2).expect("top words");

    assert_eq!(top_words.len(), 1);
    assert_eq!(top_words[0].len(), 2);
}

#[test]
fn test_lda_single_document() {
    let dtm = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 5).expect("fit");

    let doc_topics = lda.document_topics().expect("topics");
    assert_eq!(doc_topics.n_rows(), 1);
    assert_eq!(doc_topics.n_cols(), 2);
}

#[test]
fn test_lda_single_term() {
    let dtm = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 5).expect("fit");

    let topic_words = lda.topic_words().expect("topic words");
    assert_eq!(topic_words.n_cols(), 1);
    // Each topic should have weight 1.0 for the only word
    for topic in 0..2 {
        assert!((topic_words.get(topic, 0) - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_lda_topics_equals_terms() {
    // Edge case: number of topics equals number of terms
    let dtm =
        Matrix::from_vec(3, 3, vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(3);
    lda.fit(&dtm, 20).expect("fit");

    let doc_topics = lda.document_topics().expect("topics");
    assert_eq!(doc_topics.n_cols(), 3);
}

#[test]
fn test_lda_topics_exceeds_terms() {
    // More topics than terms
    let dtm = Matrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 1.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(5);
    lda.fit(&dtm, 5).expect("fit");

    let doc_topics = lda.document_topics().expect("topics");
    assert_eq!(doc_topics.n_cols(), 5);
}

#[test]
fn test_lda_high_count_values() {
    // Test with larger count values
    let dtm = Matrix::from_vec(2, 3, vec![100.0, 50.0, 10.0, 10.0, 50.0, 100.0]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 10).expect("fit");

    let doc_topics = lda.document_topics().expect("topics");
    for row in 0..2 {
        let mut sum = 0.0;
        for col in 0..2 {
            let val = doc_topics.get(row, col);
            assert!(val >= 0.0 && val <= 1.0);
            sum += val;
        }
        assert!((sum - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_lda_small_count_values() {
    // Test with very small count values
    let dtm =
        Matrix::from_vec(2, 3, vec![0.001, 0.002, 0.001, 0.002, 0.001, 0.002]).expect("matrix");

    let mut lda = LatentDirichletAllocation::new(2);
    lda.fit(&dtm, 10).expect("fit");

    let topic_words = lda.topic_words().expect("topic words");
    assert_eq!(topic_words.n_rows(), 2);
}

#[test]
fn test_empty_rows_error() {
    // Matrix with 0 rows but some columns
    let dtm = Matrix::from_vec(0, 3, vec![]).expect("matrix");
    let mut lda = LatentDirichletAllocation::new(2);
    let result = lda.fit(&dtm, 5);
    assert!(result.is_err());
}

#[test]
fn test_empty_cols_error() {
    // Matrix with some rows but 0 columns
    let dtm = Matrix::from_vec(3, 0, vec![]).expect("matrix");
    let mut lda = LatentDirichletAllocation::new(2);
    let result = lda.fit(&dtm, 5);
    assert!(result.is_err());
}

#[test]
fn test_lda_n_topics_getter() {
    let lda = LatentDirichletAllocation::new(7);
    assert_eq!(lda.n_topics, 7);
}
