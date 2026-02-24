//! Vectorization Contract Falsification Tests
//!
//! Popperian falsification of NLP spec §2.1.2 claims:
//!   - BoW output shape is (n_docs, vocab_size)
//!   - BoW counts are non-negative integers
//!   - TF-IDF values are non-negative reals
//!   - TF-IDF: rare terms get higher weight than common terms
//!   - fit_transform ≡ fit + transform (composition equivalence)
//!   - Vectorization is deterministic
//!
//! Five-Whys (PMAT-348):
//!   Why #1: vectorize module has 800 lines of unit tests but zero FALSIFY-VEC-* tests
//!   Why #2: unit tests check API behavior, not mathematical contract properties
//!   Why #3: no provable-contract YAML for text vectorization
//!   Why #4: vectorize module was built before DbC methodology
//!   Why #5: no systematic audit of NLP spec §2.1.2 claims vs test coverage
//!
//! References:
//!   - docs/specifications/nlp-models-techniques-spec.md §2.1.2
//!   - src/text/vectorize/tfidf_vectorizer.rs (CountVectorizer)
//!   - src/text/vectorize/hashing_vectorizer.rs (TfidfVectorizer)

pub(crate) use super::*;
pub(crate) use crate::text::tokenize::WhitespaceTokenizer;

// ============================================================================
// FALSIFY-VEC-001: BoW output shape
// Contract: fit_transform(docs) returns Matrix(n_docs, vocab_size)
// ============================================================================

#[test]
fn falsify_vec_001_bow_output_shape() {
    let docs = vec!["cat dog", "dog bird", "cat bird bird"];

    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let matrix = vectorizer.fit_transform(&docs).expect("fit_transform");

    assert_eq!(
        matrix.n_rows(),
        3,
        "FALSIFIED VEC-001: BoW rows {} != n_docs 3",
        matrix.n_rows()
    );
    assert_eq!(
        matrix.n_cols(),
        vectorizer.vocabulary_size(),
        "FALSIFIED VEC-001: BoW cols {} != vocab_size {}",
        matrix.n_cols(),
        vectorizer.vocabulary_size()
    );
}

#[test]
fn falsify_vec_001_single_doc_shape() {
    let docs = vec!["hello world"];
    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let matrix = vectorizer.fit_transform(&docs).expect("fit_transform");

    assert_eq!(
        matrix.n_rows(),
        1,
        "FALSIFIED VEC-001: single doc shape wrong"
    );
    assert_eq!(
        matrix.n_cols(),
        2,
        "FALSIFIED VEC-001: single doc vocab size wrong"
    );
}

// ============================================================================
// FALSIFY-VEC-002: BoW counts are non-negative integers
// Contract: every element in BoW matrix is ≥ 0 and is a whole number
// ============================================================================

#[test]
fn falsify_vec_002_bow_counts_non_negative() {
    let docs = vec![
        "the cat sat on the mat",
        "the dog sat on the log",
        "hello world",
    ];

    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let matrix = vectorizer.fit_transform(&docs).expect("fit_transform");

    for row in 0..matrix.n_rows() {
        for col in 0..matrix.n_cols() {
            let val = matrix.get(row, col);
            assert!(
                val >= 0.0,
                "FALSIFIED VEC-002: BoW[{row},{col}] = {val} is negative"
            );
            assert!(
                (val - val.round()).abs() < f64::EPSILON,
                "FALSIFIED VEC-002: BoW[{row},{col}] = {val} is not an integer"
            );
        }
    }
}

#[test]
fn falsify_vec_002_bow_repeated_word_count() {
    let docs = vec!["cat cat cat"];
    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let matrix = vectorizer.fit_transform(&docs).expect("fit_transform");

    // "cat" appears 3 times — must be reflected in count
    let cat_idx = vectorizer.vocabulary().get("cat").expect("cat in vocab");
    let count = matrix.get(0, *cat_idx);
    assert!(
        (count - 3.0).abs() < f64::EPSILON,
        "FALSIFIED VEC-002: 'cat' repeated 3 times but count = {count}"
    );
}

// ============================================================================
// FALSIFY-VEC-003: TF-IDF values are non-negative
// Contract: TF-IDF weights are ≥ 0 (TF ≥ 0, IDF ≥ 0, product ≥ 0)
// ============================================================================

#[test]
fn falsify_vec_003_tfidf_non_negative() {
    let docs = vec![
        "the cat sat on the mat",
        "the dog sat on the log",
        "hello world programming rust",
    ];

    let mut vectorizer =
        TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let matrix = vectorizer.fit_transform(&docs).expect("fit_transform");

    for row in 0..matrix.n_rows() {
        for col in 0..matrix.n_cols() {
            let val = matrix.get(row, col);
            assert!(
                val >= 0.0,
                "FALSIFIED VEC-003: TF-IDF[{row},{col}] = {val} is negative"
            );
            assert!(
                val.is_finite(),
                "FALSIFIED VEC-003: TF-IDF[{row},{col}] = {val} is not finite"
            );
        }
    }
}

// ============================================================================
// FALSIFY-VEC-004: TF-IDF: rare terms > common terms
// Contract: a term appearing in fewer documents gets higher IDF weight
// ============================================================================

#[test]
fn falsify_vec_004_tfidf_rare_terms_higher_weight() {
    // "the" appears in all 3 docs, "rust" in only 1
    let docs = vec![
        "the cat the dog",
        "the bird the fish",
        "the rust programming",
    ];

    let mut vectorizer =
        TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    vectorizer.fit(&docs).expect("fit");

    let idf = vectorizer.idf_values();
    let vocab = vectorizer.vocabulary();

    // "the" appears in all docs → lowest IDF
    // "rust" appears in 1 doc → highest IDF
    if let (Some(&the_idx), Some(&rust_idx)) = (vocab.get("the"), vocab.get("rust")) {
        let the_idf = idf[the_idx];
        let rust_idf = idf[rust_idx];

        assert!(
            rust_idf > the_idf,
            "FALSIFIED VEC-004: rare 'rust' IDF ({rust_idf}) should be > common 'the' IDF ({the_idf})"
        );
    }
}

// ============================================================================
// FALSIFY-VEC-005: fit_transform ≡ fit + transform
// Contract: fit_transform(docs) produces same result as fit(docs) + transform(docs)
// ============================================================================

#[test]
fn falsify_vec_005_fit_transform_equivalence_bow() {
    let docs = vec!["hello world", "hello rust", "world programming"];

    // Method 1: fit_transform
    let mut v1 = CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let m1 = v1.fit_transform(&docs).expect("fit_transform");

    // Method 2: fit + transform
    let mut v2 = CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    v2.fit(&docs).expect("fit");
    let m2 = v2.transform(&docs).expect("transform");

    assert_eq!(
        m1.n_rows(),
        m2.n_rows(),
        "FALSIFIED VEC-005: BoW fit_transform rows != fit+transform rows"
    );
    assert_eq!(
        m1.n_cols(),
        m2.n_cols(),
        "FALSIFIED VEC-005: BoW fit_transform cols != fit+transform cols"
    );

    for row in 0..m1.n_rows() {
        for col in 0..m1.n_cols() {
            assert!(
                (m1.get(row, col) - m2.get(row, col)).abs() < f64::EPSILON,
                "FALSIFIED VEC-005: BoW[{row},{col}] diverges: fit_transform={}, fit+transform={}",
                m1.get(row, col),
                m2.get(row, col)
            );
        }
    }
}

#[test]
fn falsify_vec_005_fit_transform_equivalence_tfidf() {
    let docs = vec!["hello world", "hello rust", "world programming"];

    let mut v1 = TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let m1 = v1.fit_transform(&docs).expect("fit_transform");

    let mut v2 = TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    v2.fit(&docs).expect("fit");
    let m2 = v2.transform(&docs).expect("transform");

    for row in 0..m1.n_rows() {
        for col in 0..m1.n_cols() {
            assert!(
                (m1.get(row, col) - m2.get(row, col)).abs() < 1e-12,
                "FALSIFIED VEC-005: TF-IDF[{row},{col}] diverges: fit_transform={}, fit+transform={}",
                m1.get(row, col),
                m2.get(row, col)
            );
        }
    }
}

// ============================================================================
// FALSIFY-VEC-006: Vectorization determinism
// Contract: same documents always produce same matrix
// ============================================================================

#[test]
fn falsify_vec_006_determinism() {
    let docs = vec!["cat dog bird", "rust programming language"];

    let mut v1 = CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let m1 = v1.fit_transform(&docs).expect("first");

    let mut v2 = CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let m2 = v2.fit_transform(&docs).expect("second");

    assert_eq!(m1.n_rows(), m2.n_rows(), "FALSIFIED VEC-006: rows differ");
    assert_eq!(m1.n_cols(), m2.n_cols(), "FALSIFIED VEC-006: cols differ");

    for row in 0..m1.n_rows() {
        for col in 0..m1.n_cols() {
            assert!(
                (m1.get(row, col) - m2.get(row, col)).abs() < f64::EPSILON,
                "FALSIFIED VEC-006: non-deterministic at [{row},{col}]"
            );
        }
    }
}

// ============================================================================
// FALSIFY-VEC-007: TF-IDF shape matches BoW shape
// Contract: TfidfVectorizer produces same shape as CountVectorizer for same docs
// ============================================================================

#[test]
fn falsify_vec_007_tfidf_bow_shape_parity() {
    let docs = vec!["cat dog", "dog bird", "cat bird bird"];

    let mut bow = CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let m_bow = bow.fit_transform(&docs).expect("bow");

    let mut tfidf = TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let m_tfidf = tfidf.fit_transform(&docs).expect("tfidf");

    assert_eq!(
        m_bow.n_rows(),
        m_tfidf.n_rows(),
        "FALSIFIED VEC-007: TF-IDF rows {} != BoW rows {}",
        m_tfidf.n_rows(),
        m_bow.n_rows()
    );
    assert_eq!(
        m_bow.n_cols(),
        m_tfidf.n_cols(),
        "FALSIFIED VEC-007: TF-IDF cols {} != BoW cols {}",
        m_tfidf.n_cols(),
        m_bow.n_cols()
    );
}
