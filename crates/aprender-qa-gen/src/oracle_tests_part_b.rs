/// Verify exactly three repetitions triggers detection
#[test]
fn test_char_ngram_boundary_exactly_three_reps() {
    // Exactly 3 repetitions, coverage 100% — should trigger
    assert!(check_substring_repetition("abcabcabc"));
}

/// Verify exactly two repetitions stays below detection threshold
#[test]
fn test_char_ngram_boundary_exactly_two_reps() {
    // Exactly 2 repetitions — below threshold of 3
    assert!(!check_substring_repetition("abcabc"));
}

// Mutation-killing tests for thresholds

/// Verify minimum repetition count threshold of 3
#[test]
fn test_char_ngram_min_reps_threshold() {
    // 3 reps at 100% coverage → true
    assert!(check_substring_repetition("xyzxyzxyz"));
    // 2 reps at 100% coverage → false (must be >= 3)
    assert!(!check_substring_repetition("xyzxyz"));
}

/// Verify 70% coverage threshold for character n-gram detection
#[test]
fn test_char_ngram_coverage_threshold() {
    // 3 reps of "ab" in "abababXXXX" = 6/10 = 60% < 70% → false
    assert!(!check_substring_repetition("abababXXXX"));
    // 3 reps of "ab" in "ababab" = 6/6 = 100% → true
    assert!(check_substring_repetition("ababab"));
}

/// Verify minimum period of 2 for character n-gram detection
#[test]
fn test_char_ngram_min_period_is_two() {
    // Period 1 is not checked — single char "aaa" with len < 6 is skipped
    assert!(!check_substring_repetition("aaa"));
    // But period 2 "aa" in a long string works
    assert!(check_substring_repetition("aaaaaaaaaaaa"));
}

/// Verify per-word n-gram detection requires minimum word length of 6
#[test]
fn test_char_ngram_word_len_threshold() {
    // Words shorter than 6 chars are not individually checked
    assert!(!has_char_ngram_repetition("aaaa bbbb"));
    // Word with 6+ chars that is repetitive gets caught
    assert!(has_char_ngram_repetition("normal ababababab text"));
}

/// Verify strings shorter than 6 bytes always return false for substring repetition
#[test]
fn test_char_ngram_too_short_string() {
    // Strings shorter than 6 bytes always return false
    assert!(!check_substring_repetition("abab"));
    assert!(!check_substring_repetition("aa"));
    assert!(!check_substring_repetition(""));
}

/// Verify GarbageOracle catches VILLE character n-gram repetition pattern
#[test]
fn test_garbage_oracle_catches_ville() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "VILLEVILLEVILLEVILLE");
    assert!(result.is_falsified());
}

/// Verify GarbageOracle catches VILLE pattern embedded in normal text
#[test]
fn test_garbage_oracle_catches_embedded_repetition() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "Result: VILLEVILLEVILLEVILLE done");
    assert!(result.is_falsified());
}

/// Verify word-level repetition detection still works after char n-gram addition
#[test]
fn test_word_level_repetition_still_works() {
    assert!(is_repetitive("foo foo foo foo foo foo"));
    assert!(is_repetitive("bar baz bar baz bar baz bar baz"));
    assert!(!is_repetitive(
        "The quick brown fox jumps over the lazy dog"
    ));
}

/// Verify short word sequences without char-ngram patterns pass through
#[test]
fn test_word_level_short_still_skipped() {
    // Short word sequences without char-ngram patterns pass through
    assert!(!is_repetitive("hello world"));
    assert!(!is_repetitive("one two three"));
}

// --- Additional mutation-killing tests ---

/// Verify ArithmeticOracle handles division with nonzero denominator correctly
#[test]
fn test_arithmetic_division_nonzero_denominator() {
    let oracle = ArithmeticOracle::new();
    // 20/4 should correctly evaluate to 5
    let result = oracle.evaluate("20/4=", "5");
    assert!(result.is_corroborated());
    // 20/4 should NOT match 6 (wrong answer)
    let wrong = oracle.evaluate("20/4=", "6");
    assert!(wrong.is_falsified());
}

/// Verify CodeSyntaxOracle corroborates short output without code patterns
#[test]
fn test_code_syntax_short_output_without_patterns() {
    let oracle = CodeSyntaxOracle::new();
    // Short output (< 20 chars) without code patterns should still corroborate
    let result = oracle.evaluate("test", "short text here");
    assert!(result.is_corroborated());
}

/// Verify CodeSyntaxOracle corroborates long output containing code patterns
#[test]
fn test_code_syntax_long_output_with_patterns() {
    let oracle = CodeSyntaxOracle::new();
    // Long output (>= 20 chars) with code patterns should corroborate
    let result = oracle.evaluate("test", "def foo(): return 42 end");
    assert!(result.is_corroborated());
}

/// Verify is_repetitive correctly identifies all-same-word sequences
#[test]
fn test_is_repetitive_same_word_repeated() {
    // All same words should be flagged as repetitive
    assert!(is_repetitive("word word word word word word"));
    // Truly different words should NOT be flagged (12 unique words to avoid 2-word pattern)
    assert!(!is_repetitive(
        "one two three four five six seven eight nine ten eleven twelve"
    ));
}

/// Verify is_repetitive distinguishes uniform from diverse word sequences
#[test]
fn test_is_repetitive_mixed_words() {
    // If first word matches all others, it's repetitive
    assert!(is_repetitive("x x x x x"));
    // Completely different words should not trigger any pattern
    assert!(!is_repetitive("a b c d e f g h i j k l"));
}
