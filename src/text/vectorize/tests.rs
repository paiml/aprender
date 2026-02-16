pub(crate) use super::*;
pub(crate) use crate::text::tokenize::WhitespaceTokenizer;

#[test]
fn test_count_vectorizer_basic() {
    let docs = vec!["cat dog", "dog bird", "cat bird bird"];

    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let matrix = vectorizer
        .fit_transform(&docs)
        .expect("fit_transform should succeed");

    assert_eq!(matrix.n_rows(), 3);
    assert_eq!(matrix.n_cols(), 3); // 3 unique words
}

#[test]
fn test_count_vectorizer_vocabulary() {
    let docs = vec!["hello world", "hello rust"];

    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    assert_eq!(vocab.len(), 3);
    assert!(vocab.contains_key("hello"));
    assert!(vocab.contains_key("world"));
    assert!(vocab.contains_key("rust"));
}

#[test]
fn test_tfidf_vectorizer_basic() {
    let docs = vec!["hello world", "hello rust", "world programming"];

    let mut vectorizer =
        TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let matrix = vectorizer
        .fit_transform(&docs)
        .expect("fit_transform should succeed");

    assert_eq!(matrix.n_rows(), 3);
    assert_eq!(vectorizer.vocabulary_size(), 4);
}

#[test]
fn test_tfidf_idf_values() {
    let docs = vec!["hello world", "hello rust"];

    let mut vectorizer =
        TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    vectorizer.fit(&docs).expect("fit should succeed");

    let idf = vectorizer.idf_values();
    assert_eq!(idf.len(), 3);
    // All IDF values should be positive
    for &value in idf {
        assert!(value > 0.0);
    }
}

#[test]
fn test_ngram_extraction() {
    let docs = vec!["the quick brown fox"];

    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_ngram_range(1, 2); // unigrams and bigrams

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    // Should have 4 unigrams + 3 bigrams = 7 terms
    assert_eq!(vocab.len(), 7);
    assert!(vocab.contains_key("the"));
    assert!(vocab.contains_key("the_quick")); // bigram
    assert!(vocab.contains_key("brown_fox")); // bigram
}

#[test]
fn test_min_df_filtering() {
    let docs = vec!["cat dog", "cat bird", "fish"]; // cat appears in 2 docs

    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_min_df(2); // require term in at least 2 docs

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    // Only "cat" appears in 2+ docs
    assert_eq!(vocab.len(), 1);
    assert!(vocab.contains_key("cat"));
}

#[test]
fn test_max_df_filtering() {
    let docs = vec!["the cat", "the dog", "the bird"]; // "the" in 100% of docs

    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_max_df(0.5); // exclude terms in >50% of docs

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    // "the" should be excluded (appears in 100% of docs)
    assert!(!vocab.contains_key("the"));
    assert_eq!(vocab.len(), 3); // cat, dog, bird
}

#[test]
fn test_sublinear_tf() {
    let docs = vec!["word word word word"]; // word appears 4 times

    let mut vectorizer_normal =
        TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let mut vectorizer_sublinear = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_sublinear_tf(true);

    let matrix_normal = vectorizer_normal
        .fit_transform(&docs)
        .expect("fit should succeed");
    let matrix_sublinear = vectorizer_sublinear
        .fit_transform(&docs)
        .expect("fit should succeed");

    // With sublinear TF, the score should be lower (1 + ln(4) ≈ 2.39 vs 4)
    assert!(matrix_sublinear.get(0, 0) < matrix_normal.get(0, 0));
}

#[test]
fn test_tfidf_full_pipeline() {
    let docs = vec![
        "machine learning is great",
        "deep learning is powerful",
        "machine learning and deep learning",
    ];

    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_ngram_range(1, 2)
        .with_sublinear_tf(true);

    let matrix = vectorizer.fit_transform(&docs).expect("fit should succeed");
    assert_eq!(matrix.n_rows(), 3);
    assert!(vectorizer.vocabulary_size() > 0);
}

#[test]
fn test_count_vectorizer_stop_words_english() {
    let docs = vec!["the cat and dog", "a bird is flying"];
    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_stop_words_english();

    let _matrix = vectorizer.fit_transform(&docs).expect("fit should succeed");
    // "the", "and", "a", "is" should be filtered out
    let vocab = vectorizer.vocabulary();
    assert!(!vocab.contains_key("the"));
    assert!(!vocab.contains_key("and"));
    assert!(vocab.contains_key("cat") || vocab.contains_key("dog"));
}

#[test]
fn test_count_vectorizer_custom_stop_words() {
    let docs = vec!["hello world hello", "world test"];
    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_stop_words(&["hello"]);

    let _matrix = vectorizer.fit_transform(&docs).expect("fit should succeed");
    let vocab = vectorizer.vocabulary();
    assert!(!vocab.contains_key("hello"));
    assert!(vocab.contains_key("world"));
}

#[test]
fn test_count_vectorizer_strip_accents() {
    let vectorizer = CountVectorizer::new().with_strip_accents(true);
    assert!(vectorizer.strip_accents);
}

#[test]
fn test_tfidf_stop_words_english() {
    let docs = vec!["the quick brown fox", "a lazy dog"];
    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_stop_words_english();

    let _matrix = vectorizer.fit_transform(&docs).expect("fit should succeed");
    let vocab = vectorizer.vocabulary();
    assert!(!vocab.contains_key("the"));
    assert!(!vocab.contains_key("a"));
}

#[test]
fn test_tfidf_custom_stop_words() {
    let docs = vec!["foo bar baz", "bar qux"];
    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_custom_stop_words(&["foo", "baz"]);

    vectorizer.fit(&docs).expect("fit should succeed");
    let vocab = vectorizer.vocabulary();
    assert!(!vocab.contains_key("foo"));
    assert!(!vocab.contains_key("baz"));
    assert!(vocab.contains_key("bar"));
}

#[test]
fn test_tfidf_strip_accents_builder() {
    let _vectorizer = TfidfVectorizer::new().with_strip_accents(true);
    // Just verify it compiles and doesn't panic
}

#[test]
fn test_hashing_vectorizer_n_features() {
    let vectorizer =
        HashingVectorizer::new(1024).with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    assert_eq!(vectorizer.n_features, 1024);
}

#[test]
fn test_hashing_vectorizer_ngram_range() {
    let vectorizer = HashingVectorizer::new(2048).with_ngram_range(1, 3);
    assert_eq!(vectorizer.ngram_range, (1, 3));
}

#[test]
fn test_hashing_vectorizer_transform() {
    let docs = vec!["hello world", "world hello hello"];

    let vectorizer =
        HashingVectorizer::new(100).with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let matrix = vectorizer
        .transform(&docs)
        .expect("transform should succeed");

    assert_eq!(matrix.n_rows(), 2);
    assert_eq!(matrix.n_cols(), 100);
}

#[test]
fn test_hashing_vectorizer_with_lowercase() {
    let vectorizer = HashingVectorizer::new(100).with_lowercase(false);
    assert!(!vectorizer.lowercase);
}

#[test]
fn test_hashing_vectorizer_with_stop_words() {
    let docs = vec!["the cat and dog", "a bird"];

    let vectorizer = HashingVectorizer::new(100)
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_stop_words_english();

    let matrix = vectorizer
        .transform(&docs)
        .expect("transform should succeed");
    assert_eq!(matrix.n_rows(), 2);
}

#[test]
fn test_hashing_vectorizer_empty_docs_error() {
    let docs: Vec<&str> = vec![];

    let vectorizer =
        HashingVectorizer::new(100).with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let result = vectorizer.transform(&docs);
    assert!(result.is_err());
}

#[test]
fn test_hashing_vectorizer_no_tokenizer_error() {
    let docs = vec!["hello"];

    let vectorizer = HashingVectorizer::new(100);

    let result = vectorizer.transform(&docs);
    assert!(result.is_err());
}

#[test]
fn test_count_vectorizer_empty_docs_error() {
    let docs: Vec<&str> = vec![];

    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let result = vectorizer.fit(&docs);
    assert!(result.is_err());
}

#[test]
fn test_count_vectorizer_no_tokenizer_error() {
    let docs = vec!["hello"];

    let mut vectorizer = CountVectorizer::new();

    let result = vectorizer.fit(&docs);
    assert!(result.is_err());
}

#[test]
fn test_count_vectorizer_transform_empty_vocab_error() {
    let docs = vec!["hello"];

    let vectorizer = CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let result = vectorizer.transform(&docs);
    assert!(result.is_err());
}

#[test]
fn test_count_vectorizer_transform_empty_docs_error() {
    let docs = vec!["hello"];
    let empty_docs: Vec<&str> = vec![];

    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    vectorizer.fit(&docs).expect("fit should succeed");

    let result = vectorizer.transform(&empty_docs);
    assert!(result.is_err());
}

#[test]
fn test_tfidf_transform_without_fit_error() {
    let docs = vec!["hello"];

    let vectorizer = TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let result = vectorizer.transform(&docs);
    assert!(result.is_err());
}

#[test]
fn test_strip_accents_unicode() {
    assert_eq!(strip_accents_unicode("café"), "cafe");
    assert_eq!(strip_accents_unicode("naïve"), "naive");
    assert_eq!(strip_accents_unicode("résumé"), "resume");
    assert_eq!(strip_accents_unicode("CAFÉ"), "CAFE");
    assert_eq!(strip_accents_unicode("señor"), "senor");
    assert_eq!(strip_accents_unicode("façade"), "facade");
    assert_eq!(strip_accents_unicode("über"), "uber");
    assert_eq!(strip_accents_unicode("hello"), "hello");
}

#[test]
fn test_strip_accents_all_characters() {
    // Test all accent mappings
    assert_eq!(strip_accents_unicode("àáâäãå"), "aaaaaa");
    assert_eq!(strip_accents_unicode("èéêë"), "eeee");
    assert_eq!(strip_accents_unicode("ìíîï"), "iiii");
    assert_eq!(strip_accents_unicode("òóôöõ"), "ooooo");
    assert_eq!(strip_accents_unicode("ùúûü"), "uuuu");
    assert_eq!(strip_accents_unicode("ýÿ"), "yy");
    assert_eq!(strip_accents_unicode("ñ"), "n");
    assert_eq!(strip_accents_unicode("ç"), "c");
    // Uppercase
    assert_eq!(strip_accents_unicode("ÀÁÂÄÃÅ"), "AAAAAA");
    assert_eq!(strip_accents_unicode("ÈÉÊË"), "EEEE");
    assert_eq!(strip_accents_unicode("ÌÍÎÏ"), "IIII");
    assert_eq!(strip_accents_unicode("ÒÓÔÖÕ"), "OOOOO");
    assert_eq!(strip_accents_unicode("ÙÚÛÜ"), "UUUU");
    assert_eq!(strip_accents_unicode("Ý"), "Y");
    assert_eq!(strip_accents_unicode("Ñ"), "N");
    assert_eq!(strip_accents_unicode("Ç"), "C");
}

#[test]
fn test_count_vectorizer_with_strip_accents_integration() {
    let docs = vec!["café résumé", "cafe resume"];

    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_strip_accents(true);

    vectorizer.fit(&docs).expect("fit should succeed");

    // After stripping accents, "café" and "cafe" should be the same
    let vocab = vectorizer.vocabulary();
    assert!(vocab.contains_key("cafe"));
    assert!(vocab.contains_key("resume"));
    // Should not have the accented versions as separate entries
    assert!(!vocab.contains_key("café"));
}

#[test]
fn test_count_vectorizer_max_features() {
    let docs = vec!["a b c d e f g h i j"];

    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_max_features(5);

    vectorizer.fit(&docs).expect("fit should succeed");

    // Should be limited to 5 features
    assert_eq!(vectorizer.vocabulary_size(), 5);
}

#[test]
fn test_count_vectorizer_default() {
    let vectorizer = CountVectorizer::default();
    assert!(vectorizer.lowercase);
    assert_eq!(vectorizer.ngram_range, (1, 1));
}

#[test]
fn test_tfidf_vectorizer_default() {
    let vectorizer = TfidfVectorizer::default();
    assert!(!vectorizer.sublinear_tf);
}

#[test]
fn test_tfidf_with_min_df() {
    let docs = vec!["cat", "cat dog", "dog bird"]; // cat in 2, dog in 2, bird in 1

    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_min_df(2);

    vectorizer.fit(&docs).expect("fit should succeed");

    let vocab = vectorizer.vocabulary();
    assert!(vocab.contains_key("cat"));
    assert!(vocab.contains_key("dog"));
    assert!(!vocab.contains_key("bird"));
}

#[path = "tests_part_02.rs"]

mod tests_part_02;
