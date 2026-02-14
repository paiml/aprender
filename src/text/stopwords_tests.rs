use super::*;

// ========== StopWordsFilter Tests ==========

#[test]
fn test_english_filter_basic() {
    let filter = StopWordsFilter::english();
    let tokens = vec!["the", "quick", "brown", "fox"];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(filtered, vec!["quick", "brown", "fox"]);
}

#[test]
fn test_english_filter_case_insensitive() {
    let filter = StopWordsFilter::english();
    let tokens = vec!["The", "Cat", "IS", "happy"];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(filtered, vec!["Cat", "happy"]);
}

#[test]
fn test_english_filter_preserves_case() {
    let filter = StopWordsFilter::english();
    let tokens = vec!["Machine", "learning", "the", "FUTURE"];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(filtered, vec!["Machine", "learning", "FUTURE"]);
}

#[test]
fn test_custom_stop_words() {
    let filter = StopWordsFilter::new(vec!["foo", "bar", "baz"]);
    let tokens = vec!["foo", "test", "bar", "data", "baz"];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(filtered, vec!["test", "data"]);
}

#[test]
fn test_empty_tokens() {
    let filter = StopWordsFilter::english();
    let tokens: Vec<&str> = vec![];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(filtered, Vec::<String>::new());
}

#[test]
fn test_all_stop_words() {
    let filter = StopWordsFilter::english();
    let tokens = vec!["the", "and", "is", "a"];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(filtered, Vec::<String>::new());
}

#[test]
fn test_no_stop_words() {
    let filter = StopWordsFilter::english();
    let tokens = vec!["machine", "learning", "neural", "network"];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(filtered, vec!["machine", "learning", "neural", "network"]);
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
fn test_mixed_content() {
    let filter = StopWordsFilter::english();
    let tokens = vec![
        "I", "love", "machine", "learning", "and", "neural", "networks", "because", "they",
        "are", "powerful",
    ];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(
        filtered,
        vec!["love", "machine", "learning", "neural", "networks", "powerful"]
    );
}

#[test]
fn test_real_world_sentence() {
    let filter = StopWordsFilter::english();
    let tokens = vec![
        "Natural",
        "language",
        "processing",
        "is",
        "a",
        "subfield",
        "of",
        "artificial",
        "intelligence",
    ];
    let filtered = filter.filter(&tokens).expect("filter should succeed");
    assert_eq!(
        filtered,
        vec![
            "Natural",
            "language",
            "processing",
            "subfield",
            "artificial",
            "intelligence"
        ]
    );
}

// ========== ENGLISH_STOP_WORDS Tests ==========

#[test]
fn test_stop_words_list_has_expected_count() {
    // Verify exact count of English stop words (NLTK/sklearn-based list)
    assert_eq!(ENGLISH_STOP_WORDS.len(), 171);
}

#[test]
fn test_stop_words_contains_articles() {
    assert!(ENGLISH_STOP_WORDS.contains(&"a"));
    assert!(ENGLISH_STOP_WORDS.contains(&"an"));
    assert!(ENGLISH_STOP_WORDS.contains(&"the"));
}

#[test]
fn test_stop_words_contains_pronouns() {
    assert!(ENGLISH_STOP_WORDS.contains(&"i"));
    assert!(ENGLISH_STOP_WORDS.contains(&"you"));
    assert!(ENGLISH_STOP_WORDS.contains(&"he"));
    assert!(ENGLISH_STOP_WORDS.contains(&"she"));
    assert!(ENGLISH_STOP_WORDS.contains(&"it"));
    assert!(ENGLISH_STOP_WORDS.contains(&"we"));
    assert!(ENGLISH_STOP_WORDS.contains(&"they"));
}

#[test]
fn test_stop_words_contains_prepositions() {
    assert!(ENGLISH_STOP_WORDS.contains(&"in"));
    assert!(ENGLISH_STOP_WORDS.contains(&"on"));
    assert!(ENGLISH_STOP_WORDS.contains(&"at"));
    assert!(ENGLISH_STOP_WORDS.contains(&"by"));
    assert!(ENGLISH_STOP_WORDS.contains(&"for"));
    assert!(ENGLISH_STOP_WORDS.contains(&"with"));
}

#[test]
fn test_stop_words_contains_conjunctions() {
    assert!(ENGLISH_STOP_WORDS.contains(&"and"));
    assert!(ENGLISH_STOP_WORDS.contains(&"or"));
    assert!(ENGLISH_STOP_WORDS.contains(&"but"));
    assert!(ENGLISH_STOP_WORDS.contains(&"if"));
    assert!(ENGLISH_STOP_WORDS.contains(&"because"));
}

#[test]
fn test_stop_words_lowercase_only() {
    // All stop words should be lowercase
    for word in ENGLISH_STOP_WORDS {
        assert_eq!(*word, word.to_lowercase());
    }
}
