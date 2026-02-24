//! Text Preprocessing Contract Falsification Tests
//!
//! Popperian falsification of NLP spec §2.1.1 claims:
//!   - Tokenization splits text into tokens deterministically
//!   - Stop word removal preserves domain-specific terms
//!   - Stemming reduces words to root forms
//!   - Pipeline composition is order-independent for commutative ops
//!
//! Five-Whys (PMAT-346):
//!   Why #1: NLP spec §2.1.1 defines text preprocessing pipeline, zero FALSIFY tests exist
//!   Why #2: text module has unit tests but no contract-level falsification
//!   Why #3: no provable-contract YAML for text preprocessing
//!   Why #4: text preprocessing was added before DbC methodology
//!   Why #5: no systematic contract audit of pre-DbC modules
//!
//! References:
//!   - docs/specifications/nlp-models-techniques-spec.md §2.1.1
//!   - src/text/tokenize/mod.rs (WhitespaceTokenizer)
//!   - src/text/stopwords.rs (StopWordsFilter)
//!   - src/text/stem.rs (PorterStemmer)

use crate::text::stem::{PorterStemmer, Stemmer};
use crate::text::stopwords::StopWordsFilter;
use crate::text::tokenize::WhitespaceTokenizer;
use crate::text::Tokenizer;

// ============================================================================
// FALSIFY-PP-001: Tokenizer determinism
// Contract: same input always produces same output
// ============================================================================

#[test]
fn falsify_pp_001_tokenizer_determinism() {
    let tokenizer = WhitespaceTokenizer::new();
    let input = "fix: null pointer dereference in parse_expr()";

    let t1 = tokenizer.tokenize(input).expect("tokenize should succeed");
    let t2 = tokenizer.tokenize(input).expect("tokenize should succeed");

    assert_eq!(t1, t2, "FALSIFIED PP-001: tokenizer is non-deterministic");
}

#[test]
fn falsify_pp_001_tokenizer_empty_input() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer.tokenize("").expect("tokenize empty");
    assert!(
        tokens.is_empty(),
        "FALSIFIED PP-001: empty input should produce empty tokens, got {:?}",
        tokens
    );
}

#[test]
fn falsify_pp_001_tokenizer_whitespace_only() {
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer
        .tokenize("   \t\n  ")
        .expect("tokenize whitespace");
    assert!(
        tokens.is_empty(),
        "FALSIFIED PP-001: whitespace-only input should produce empty tokens, got {:?}",
        tokens
    );
}

// ============================================================================
// FALSIFY-PP-002: Tokenizer preserves all content
// Contract: join(tokenize(text), " ") reconstructs the non-whitespace content
// ============================================================================

#[test]
fn falsify_pp_002_tokenizer_preserves_content() {
    let tokenizer = WhitespaceTokenizer::new();
    let input = "Hello world foo bar";
    let tokens = tokenizer.tokenize(input).expect("tokenize");

    assert_eq!(
        tokens,
        vec!["Hello", "world", "foo", "bar"],
        "FALSIFIED PP-002: tokens don't match expected split"
    );
}

#[test]
fn falsify_pp_002_tokenizer_preserves_punctuation() {
    let tokenizer = WhitespaceTokenizer::new();
    // NLP spec §2.1.1: "fix: null pointer dereference in parse_expr()"
    let tokens = tokenizer
        .tokenize("fix: null pointer dereference in parse_expr()")
        .expect("tokenize");

    // Whitespace tokenizer keeps punctuation attached to words
    assert!(
        tokens.contains(&"fix:".to_string()),
        "FALSIFIED PP-002: 'fix:' should be a single token (punctuation preserved)"
    );
    assert!(
        tokens.contains(&"parse_expr()".to_string()),
        "FALSIFIED PP-002: 'parse_expr()' should be a single token (code token preserved)"
    );
}

// ============================================================================
// FALSIFY-PP-003: Stop word filter preserves domain terms
// Contract: domain-specific terms (fix, null, pointer) are NOT stop words
// ============================================================================

#[test]
fn falsify_pp_003_stop_words_preserve_domain_terms() {
    let filter = StopWordsFilter::english();

    // These domain terms must NOT be stop words
    let domain_terms = [
        "fix",
        "null",
        "pointer",
        "dereference",
        "error",
        "bug",
        "race",
    ];
    for term in &domain_terms {
        assert!(
            !filter.is_stop_word(term),
            "FALSIFIED PP-003: domain term '{}' incorrectly classified as stop word",
            term
        );
    }
}

#[test]
fn falsify_pp_003_stop_words_filter_common() {
    let filter = StopWordsFilter::english();

    // These common words MUST be stop words
    let common = ["the", "a", "an", "and", "or", "is", "in", "to", "of"];
    for word in &common {
        assert!(
            filter.is_stop_word(word),
            "FALSIFIED PP-003: common word '{}' not in stop word list",
            word
        );
    }
}

#[test]
fn falsify_pp_003_stop_words_case_insensitive() {
    let filter = StopWordsFilter::english();

    assert!(
        filter.is_stop_word("The"),
        "FALSIFIED PP-003: 'The' should match stop word 'the' (case-insensitive)"
    );
    assert!(
        filter.is_stop_word("AND"),
        "FALSIFIED PP-003: 'AND' should match stop word 'and' (case-insensitive)"
    );
}

#[test]
fn falsify_pp_003_stop_words_filter_operation() {
    let filter = StopWordsFilter::english();
    let tokens = vec![
        "fix".to_string(),
        "the".to_string(),
        "null".to_string(),
        "pointer".to_string(),
        "in".to_string(),
        "parse_expr".to_string(),
    ];

    let filtered = filter.filter_owned(tokens).expect("filter");

    // "the" and "in" should be removed, rest preserved
    assert!(
        filtered.contains(&"fix".to_string()),
        "FALSIFIED PP-003: 'fix' should survive stop word filter"
    );
    assert!(
        filtered.contains(&"null".to_string()),
        "FALSIFIED PP-003: 'null' should survive stop word filter"
    );
    assert!(
        !filtered.contains(&"the".to_string()),
        "FALSIFIED PP-003: 'the' should be removed by stop word filter"
    );
    assert!(
        !filtered.contains(&"in".to_string()),
        "FALSIFIED PP-003: 'in' should be removed by stop word filter"
    );
}

// ============================================================================
// FALSIFY-PP-004: Stemmer correctness
// Contract: stemmer reduces words to valid root forms
// ============================================================================

#[test]
fn falsify_pp_004_stemmer_known_words() {
    let stemmer = PorterStemmer::new();

    // NLP spec §2.1.1 examples
    let cases = [("running", "run"), ("flies", "fli"), ("easily", "easili")];

    for (input, expected) in &cases {
        let result = stemmer.stem(input).expect("stem should succeed");
        assert_eq!(
            &result, expected,
            "FALSIFIED PP-004: stem('{}') = '{}', expected '{}'",
            input, result, expected
        );
    }
}

#[test]
fn falsify_pp_004_stemmer_determinism() {
    let stemmer = PorterStemmer::new();
    let word = "programming";

    let s1 = stemmer.stem(word).expect("stem");
    let s2 = stemmer.stem(word).expect("stem");

    assert_eq!(
        s1, s2,
        "FALSIFIED PP-004: stemmer is non-deterministic for '{word}'"
    );
}

#[test]
fn falsify_pp_004_stemmer_idempotent() {
    let stemmer = PorterStemmer::new();

    // Stemming an already-stemmed word should produce the same result
    let words = ["run", "fix", "bug", "test"];
    for &w in &words {
        let once = stemmer.stem(w).expect("stem once");
        let twice = stemmer.stem(&once).expect("stem twice");
        assert_eq!(
            once, twice,
            "FALSIFIED PP-004: stem is not idempotent for '{w}': '{once}' -> '{twice}'"
        );
    }
}

#[test]
fn falsify_pp_004_stemmer_empty_input() {
    let stemmer = PorterStemmer::new();
    let result = stemmer.stem("").expect("stem empty");
    assert_eq!(
        result, "",
        "FALSIFIED PP-004: stem('') should return '', got '{result}'"
    );
}

#[test]
fn falsify_pp_004_stemmer_batch() {
    let stemmer = PorterStemmer::new();
    let words = vec!["running", "flies", "easily"];
    let stemmed = stemmer.stem_tokens(&words).expect("stem_tokens");

    assert_eq!(stemmed.len(), 3);
    assert_eq!(stemmed[0], "run");
}

// ============================================================================
// FALSIFY-PP-005: Pipeline composition
// Contract: tokenize → filter → stem produces valid output
// ============================================================================

#[test]
fn falsify_pp_005_full_pipeline() {
    let tokenizer = WhitespaceTokenizer::new();
    let filter = StopWordsFilter::english();
    let stemmer = PorterStemmer::new();

    // NLP spec §2.1.1 example
    let text = "fix the null pointer dereference in parse_expr()";

    // Step 1: Tokenize
    let tokens = tokenizer.tokenize(text).expect("tokenize");
    assert!(
        !tokens.is_empty(),
        "FALSIFIED PP-005: tokenization produced empty"
    );

    // Step 2: Filter stop words
    let filtered = filter.filter_owned(tokens).expect("filter");
    assert!(
        !filtered.is_empty(),
        "FALSIFIED PP-005: stop word filter removed all tokens"
    );

    // Step 3: Stem
    let stemmed = stemmer.stem_tokens(&filtered).expect("stem");
    assert_eq!(
        stemmed.len(),
        filtered.len(),
        "FALSIFIED PP-005: stemming changed token count"
    );

    // All results should be non-empty strings
    for (i, s) in stemmed.iter().enumerate() {
        assert!(
            !s.is_empty(),
            "FALSIFIED PP-005: stemmed token {i} is empty"
        );
    }
}

#[test]
fn falsify_pp_005_pipeline_preserves_code_tokens() {
    let tokenizer = WhitespaceTokenizer::new();
    let filter = StopWordsFilter::english();

    let text = "parse_expr() into_iter() Option<T>";
    let tokens = tokenizer.tokenize(text).expect("tokenize");
    let filtered = filter.filter_owned(tokens).expect("filter");

    // Code tokens should survive stop word filtering
    assert!(
        filtered.iter().any(|t: &String| t.contains("parse_expr")),
        "FALSIFIED PP-005: code token 'parse_expr()' was removed by pipeline"
    );
    assert!(
        filtered.iter().any(|t: &String| t.contains("into_iter")),
        "FALSIFIED PP-005: code token 'into_iter()' was removed by pipeline"
    );
}

// ============================================================================
// FALSIFY-PP-006: English stop words count
// Contract: NLTK/sklearn English stop words list has ~170 words
// ============================================================================

#[test]
fn falsify_pp_006_english_stop_words_count() {
    let filter = StopWordsFilter::english();
    let count = filter.len();

    // NLP spec references NLTK/sklearn stop words (typically 170-180)
    assert!(
        count >= 100,
        "FALSIFIED PP-006: English stop words has only {count} words (expected >= 100)"
    );
    assert!(
        count <= 300,
        "FALSIFIED PP-006: English stop words has {count} words (expected <= 300, too many)"
    );
}
