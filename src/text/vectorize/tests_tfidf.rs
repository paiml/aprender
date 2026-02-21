use super::*;

#[test]
fn test_tfidf_with_max_df() {
    let docs = vec!["the cat", "the dog", "the bird"];

    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_max_df(0.5);

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    assert!(!vocab.contains_key("the"));
}

#[test]
fn test_tfidf_ngram_range() {
    let docs = vec!["hello world"];

    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_ngram_range(1, 2);

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    assert!(vocab.contains_key("hello"));
    assert!(vocab.contains_key("world"));
    assert!(vocab.contains_key("hello_world"));
}

#[test]
fn test_tfidf_max_features() {
    let docs = vec!["a b c d e f g"];

    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_max_features(3);

    vectorizer.fit(&docs).expect("fit should succeed");

    assert_eq!(vectorizer.vocabulary_size(), 3);
}

#[test]
fn test_tfidf_lowercase() {
    let docs = vec!["Hello WORLD"];

    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_lowercase(true);

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    assert!(vocab.contains_key("hello"));
    assert!(vocab.contains_key("world"));
    assert!(!vocab.contains_key("Hello"));
}

#[test]
fn test_hash_term_deterministic() {
    let hash1 = hash_term("test", 1000);
    let hash2 = hash_term("test", 1000);
    assert_eq!(hash1, hash2);

    // Different terms should (usually) hash differently
    let hash3 = hash_term("other", 1000);
    assert_ne!(hash1, hash3);
}

#[test]
fn test_hash_term_within_range() {
    for term in &["a", "test", "hello world", "12345"] {
        let hash = hash_term(term, 100);
        assert!(hash < 100);
    }
}

#[test]
fn test_hashing_vectorizer_with_ngrams() {
    let docs = vec!["a b c"];

    let vectorizer = HashingVectorizer::new(1000)
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_ngram_range(1, 2);

    let matrix = vectorizer
        .transform(&docs)
        .expect("transform should succeed");

    // Should have features for unigrams and bigrams
    let row_sum: f64 = (0..1000).map(|i| matrix.get(0, i)).sum();
    assert!(row_sum > 3.0); // at least 3 unigrams + 2 bigrams
}

#[test]
fn test_count_vectorizer_transform_unseen_words() {
    let train_docs = vec!["cat dog"];
    let test_docs = vec!["cat elephant"]; // elephant not in vocab

    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    vectorizer.fit(&train_docs).expect("fit should succeed");

    let matrix = vectorizer
        .transform(&test_docs)
        .expect("transform should succeed");

    // Should only count "cat", not "elephant"
    let row_sum: f64 = (0..matrix.n_cols()).map(|i| matrix.get(0, i)).sum();
    assert!((row_sum - 1.0).abs() < 1e-6); // Only cat is counted
}

// ========================================================================
// Additional coverage tests
// ========================================================================

#[test]
fn test_count_vectorizer_transform_no_tokenizer_error() {
    let docs = vec!["hello"];
    let vectorizer = CountVectorizer::new();
    // Manually set vocabulary to bypass fit() check
    let mut v = vectorizer;
    v.vocabulary.insert("hello".to_string(), 0);
    let result = v.transform(&docs);
    assert!(result.is_err());
}

#[test]
fn test_tfidf_vectorizer_with_strip_accents_integration() {
    let docs = vec!["café naïve résumé", "cafe naive resume"];

    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_strip_accents(true);

    vectorizer.fit(&docs).expect("fit should succeed");

    // After stripping accents, accented and non-accented versions should be same
    let vocab = vectorizer.vocabulary();
    assert!(vocab.contains_key("cafe"));
    assert!(!vocab.contains_key("café"));
}

#[test]
fn test_count_vectorizer_lowercase_false() {
    let docs = vec!["Hello WORLD hello"];

    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_lowercase(false);

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    // With lowercase=false, "Hello" and "hello" are different
    assert!(vocab.contains_key("Hello"));
    assert!(vocab.contains_key("hello"));
    assert!(vocab.contains_key("WORLD"));
}

#[test]
fn test_tfidf_lowercase_false() {
    let docs = vec!["Hello WORLD"];

    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_lowercase(false);

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    assert!(vocab.contains_key("Hello"));
    assert!(vocab.contains_key("WORLD"));
}

#[test]
fn test_hashing_vectorizer_lowercase_true() {
    let docs = vec!["Hello WORLD"];

    let vectorizer = HashingVectorizer::new(100)
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_lowercase(true);

    let matrix = vectorizer
        .transform(&docs)
        .expect("transform should succeed");

    // Just verify it works with lowercase
    assert_eq!(matrix.n_rows(), 1);
}

#[test]
fn test_count_vectorizer_vocabulary_size() {
    let docs = vec!["a b c d e"];

    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    vectorizer.fit(&docs).expect("fit should succeed");
    assert_eq!(vectorizer.vocabulary_size(), 5);
}

#[test]
fn test_tfidf_transform_multiple_docs() {
    let train_docs = vec!["hello world", "rust programming"];
    let test_docs = vec!["hello rust", "world programming"];

    let mut vectorizer =
        TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    vectorizer.fit(&train_docs).expect("fit should succeed");

    let matrix = vectorizer
        .transform(&test_docs)
        .expect("transform should succeed");

    assert_eq!(matrix.n_rows(), 2);
    assert_eq!(matrix.n_cols(), 4); // 4 unique words
}

#[test]
fn test_count_vectorizer_ngram_range_min_equals_max() {
    let docs = vec!["a b c"];

    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_ngram_range(2, 2); // Only bigrams

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    // Should only have bigrams
    assert!(vocab.contains_key("a_b"));
    assert!(vocab.contains_key("b_c"));
    assert!(!vocab.contains_key("a"));
}

#[test]
fn test_strip_accents_passthrough_non_accented() {
    // Ensure non-accented characters pass through unchanged
    assert_eq!(strip_accents_unicode("abc123!@#"), "abc123!@#");
    assert_eq!(strip_accents_unicode(""), "");
    assert_eq!(strip_accents_unicode("   "), "   ");
}

#[test]
fn test_max_df_clamping() {
    // Test that max_df is properly clamped to 0.0-1.0
    let vectorizer = CountVectorizer::new().with_max_df(1.5);
    assert!((vectorizer.max_df - 1.0).abs() < 1e-6);

    let vectorizer2 = CountVectorizer::new().with_max_df(-0.5);
    assert!((vectorizer2.max_df - 0.0).abs() < 1e-6);
}

#[test]
fn test_ngram_range_min_zero_correction() {
    // Test that min_n=0 gets corrected to 1
    let vectorizer = CountVectorizer::new().with_ngram_range(0, 2);
    assert_eq!(vectorizer.ngram_range, (1, 2));

    let vectorizer2 = HashingVectorizer::new(100).with_ngram_range(0, 0);
    assert_eq!(vectorizer2.ngram_range, (1, 1));
}

#[test]
fn test_transform_stop_words_filtering_in_transform() {
    let train_docs = vec!["cat the dog"];
    let test_docs = vec!["the cat the dog the"]; // "the" repeated

    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_stop_words_english();

    vectorizer.fit(&train_docs).expect("fit should succeed");

    let matrix = vectorizer
        .transform(&test_docs)
        .expect("transform should succeed");

    // "the" should be filtered out, only "cat" and "dog" counted
    let row_sum: f64 = (0..matrix.n_cols()).map(|i| matrix.get(0, i)).sum();
    assert!((row_sum - 2.0).abs() < 1e-6);
}

#[test]
fn test_hash_term_empty_string() {
    // Empty string should still hash to a valid index
    let hash = hash_term("", 100);
    assert!(hash < 100);
}

#[test]
fn test_hash_term_unicode() {
    // Unicode strings should hash correctly
    let hash = hash_term("日本語", 1000);
    assert!(hash < 1000);

    let hash2 = hash_term("émoji🎉", 1000);
    assert!(hash2 < 1000);
}

#[test]
fn test_count_vectorizer_fit_transform_consistency() {
    let docs = vec!["a b c", "b c d", "c d e"];

    let mut v1 = CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    // fit_transform should give same result as fit then transform
    let matrix1 = v1
        .fit_transform(&docs)
        .expect("fit_transform should succeed");

    let mut v2 = CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    v2.fit(&docs).expect("fit should succeed");
    let matrix2 = v2.transform(&docs).expect("transform should succeed");

    // Matrices should be identical
    assert_eq!(matrix1.n_rows(), matrix2.n_rows());
    assert_eq!(matrix1.n_cols(), matrix2.n_cols());
    for i in 0..matrix1.n_rows() {
        for j in 0..matrix1.n_cols() {
            assert!((matrix1.get(i, j) - matrix2.get(i, j)).abs() < 1e-10);
        }
    }
}

#[test]
fn test_tfidf_vectorizer_fit_transform_consistency() {
    let docs = vec!["a b", "b c", "c a"];

    let mut v1 = TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let matrix1 = v1
        .fit_transform(&docs)
        .expect("fit_transform should succeed");

    let mut v2 = TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    v2.fit(&docs).expect("fit should succeed");
    let matrix2 = v2.transform(&docs).expect("transform should succeed");

    assert_eq!(matrix1.n_rows(), matrix2.n_rows());
    assert_eq!(matrix1.n_cols(), matrix2.n_cols());
    for i in 0..matrix1.n_rows() {
        for j in 0..matrix1.n_cols() {
            assert!((matrix1.get(i, j) - matrix2.get(i, j)).abs() < 1e-10);
        }
    }
}
