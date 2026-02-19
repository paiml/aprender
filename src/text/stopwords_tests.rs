pub(crate) use super::*;

// ========== StopWordsFilter Tests ==========

/// Helper: filter tokens through English stop words and assert expected result.
fn assert_english_filter(tokens: &[&str], expected: &[&str]) {
    let filter = StopWordsFilter::english();
    let filtered = filter.filter(tokens).expect("filter should succeed");
    let expected_strings: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
    assert_eq!(filtered, expected_strings, "input: {tokens:?}");
}

#[test]
fn test_english_filter_basic_cases() {
    // Data-driven: (input_tokens, expected_output)
    let cases: &[(&[&str], &[&str])] = &[
        // Basic filtering
        (&["the", "quick", "brown", "fox"], &["quick", "brown", "fox"]),
        // Case-insensitive
        (&["The", "Cat", "IS", "happy"], &["Cat", "happy"]),
        // Preserves original case
        (&["Machine", "learning", "the", "FUTURE"], &["Machine", "learning", "FUTURE"]),
        // Empty input
        (&[], &[]),
        // All stop words -> empty output
        (&["the", "and", "is", "a"], &[]),
        // No stop words -> all pass through
        (&["machine", "learning", "neural", "network"], &["machine", "learning", "neural", "network"]),
    ];
    for (tokens, expected) in cases {
        assert_english_filter(tokens, expected);
    }
}

#[test]
fn test_custom_stop_words() {
    let filter = StopWordsFilter::new(vec!["foo", "bar", "baz"]);
    let tokens = vec!["foo", "test", "bar", "data", "baz"];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(filtered, vec!["test", "data"]);
}

#[test]
fn test_filter_owned() {
    let filter = StopWordsFilter::english();
    let tokens = vec![
        "the".to_string(),
        "cat".to_string(),
        "is".to_string(),
        "happy".to_string(),
    ];
    let filtered = filter.filter_owned(tokens).expect("filter should succeed");
    assert_eq!(filtered, vec!["cat", "happy"]);
}

#[test]
fn test_is_stop_word() {
    let filter = StopWordsFilter::english();
    assert!(filter.is_stop_word("the"));
    assert!(filter.is_stop_word("THE"));
    assert!(filter.is_stop_word("is"));
    assert!(!filter.is_stop_word("machine"));
    assert!(!filter.is_stop_word("learning"));
}

#[test]
fn test_filter_len() {
    let english = StopWordsFilter::english();
    assert_eq!(english.len(), 171);

    let custom = StopWordsFilter::new(vec!["foo", "bar"]);
    assert_eq!(custom.len(), 2);

    let empty = StopWordsFilter::new(Vec::<String>::new());
    assert_eq!(empty.len(), 0);
}

#[test]
fn test_filter_is_empty() {
    let english = StopWordsFilter::english();
    assert!(!english.is_empty());

    let empty = StopWordsFilter::new(Vec::<String>::new());
    assert!(empty.is_empty());
}

#[test]
fn test_unicode_tokens() {
    let filter = StopWordsFilter::english();
    let tokens = vec!["the", "世界", "is", "beautiful"];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(filtered, vec!["世界", "beautiful"]);
}

#[test]
fn test_punctuation_handling() {
    let filter = StopWordsFilter::english();
    // Stop words with punctuation attached are NOT filtered
    // (this is expected - tokenization should separate punctuation)
    let tokens = vec!["the.", "cat,", "is!", "happy"];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    // "the.", "cat,", "is!" are not exact matches for stop words
    assert_eq!(filtered, vec!["the.", "cat,", "is!", "happy"]);
}

#[test]
fn test_real_world_sentences() {
    // Data-driven: filter real-world token sequences
    let cases: &[(&[&str], &[&str])] = &[
        (
            &["I", "love", "machine", "learning", "and", "neural", "networks", "because", "they", "are", "powerful"],
            &["love", "machine", "learning", "neural", "networks", "powerful"],
        ),
        (
            &["Natural", "language", "processing", "is", "a", "subfield", "of", "artificial", "intelligence"],
            &["Natural", "language", "processing", "subfield", "artificial", "intelligence"],
        ),
    ];
    for (tokens, expected) in cases {
        assert_english_filter(tokens, expected);
    }
}

// ========== ENGLISH_STOP_WORDS Tests ==========

#[test]
fn test_stop_words_list_has_expected_count() {
    // Verify exact count of English stop words (NLTK/sklearn-based list)
    assert_eq!(ENGLISH_STOP_WORDS.len(), 171);
}

/// Data-driven: expected stop words by category, consolidated from
/// separate per-category tests to eliminate DataTransformation repetition.
const EXPECTED_STOP_WORDS_BY_CATEGORY: &[(&str, &[&str])] = &[
    ("articles", &["a", "an", "the"]),
    ("pronouns", &["i", "you", "he", "she", "it", "we", "they"]),
    ("prepositions", &["in", "on", "at", "by", "for", "with"]),
    ("conjunctions", &["and", "or", "but", "if", "because"]),
];

#[test]
fn test_stop_words_contains_expected_categories() {
    for (category, words) in EXPECTED_STOP_WORDS_BY_CATEGORY {
        for word in *words {
            assert!(
                ENGLISH_STOP_WORDS.contains(word),
                "{category}: expected {word:?} in ENGLISH_STOP_WORDS"
            );
        }
    }
}

#[test]
fn test_stop_words_lowercase_only() {
    // All stop words should be lowercase
    for word in ENGLISH_STOP_WORDS {
        assert_eq!(*word, word.to_lowercase());
    }
}
