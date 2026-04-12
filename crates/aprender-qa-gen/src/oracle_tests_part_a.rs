/// Verify arithmetic oracle corroborates correct answer
#[test]
fn test_arithmetic_oracle_correct() {
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("2+2=", "The answer is 4.");
    assert!(result.is_corroborated());
}

/// Verify arithmetic oracle falsifies incorrect answer
#[test]
fn test_arithmetic_oracle_incorrect() {
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("2+2=", "The answer is 5.");
    assert!(result.is_falsified());
}

/// Verify arithmetic oracle skips non-arithmetic prompts
#[test]
fn test_arithmetic_oracle_non_arithmetic() {
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("What is your name?", "I am an AI.");
    assert!(result.is_corroborated()); // Skipped
}

/// Verify garbage oracle falsifies empty output
#[test]
fn test_garbage_oracle_empty() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "");
    assert!(result.is_falsified());
}

/// Verify garbage oracle corroborates valid text output
#[test]
fn test_garbage_oracle_valid() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "This is a valid response.");
    assert!(result.is_corroborated());
}

/// Verify garbage oracle falsifies output containing NaN
#[test]
fn test_garbage_oracle_nan() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "The value is NaN");
    assert!(result.is_falsified());
}

/// Verify garbage oracle falsifies repetitive output
#[test]
fn test_garbage_oracle_repetitive() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "ak ak ak ak ak ak ak ak");
    assert!(result.is_falsified());
}

/// Verify select_oracle returns arithmetic oracle for math prompts
#[test]
fn test_select_oracle_arithmetic() {
    let oracle = select_oracle("What is 2+2?");
    assert_eq!(oracle.name(), "arithmetic");
}

/// Verify select_oracle returns code_syntax oracle for code prompts
#[test]
fn test_select_oracle_code() {
    let oracle = select_oracle("def fibonacci(n):");
    assert_eq!(oracle.name(), "code_syntax");
}

/// Verify select_oracle falls back to garbage oracle for unknown prompts
#[test]
fn test_select_oracle_default() {
    let oracle = select_oracle("Tell me a joke");
    assert_eq!(oracle.name(), "garbage");
}

/// Verify is_repetitive detects repeating word patterns
#[test]
fn test_is_repetitive() {
    assert!(is_repetitive("foo foo foo foo foo foo"));
    assert!(is_repetitive("bar baz bar baz bar baz bar baz"));
    assert!(!is_repetitive(
        "The quick brown fox jumps over the lazy dog"
    ));
}

/// Verify is_repetitive returns false for short non-repetitive inputs
#[test]
fn test_is_repetitive_short() {
    assert!(!is_repetitive("a b c"));
    assert!(!is_repetitive(""));
}

/// Verify OracleResult::Corroborated reports as corroborated
#[test]
fn test_oracle_result_is_corroborated() {
    let result = OracleResult::Corroborated {
        evidence: "test".to_string(),
    };
    assert!(result.is_corroborated());
    assert!(!result.is_falsified());
}

/// Verify OracleResult::Falsified reports as falsified
#[test]
fn test_oracle_result_is_falsified() {
    let result = OracleResult::Falsified {
        reason: "bad".to_string(),
        evidence: "test".to_string(),
    };
    assert!(!result.is_corroborated());
    assert!(result.is_falsified());
}

/// Verify garbage oracle falsifies output containing control characters
#[test]
fn test_garbage_oracle_control_chars() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "Hello\x00World");
    assert!(result.is_falsified());
}

/// Verify garbage oracle falsifies output containing Unicode replacement character
#[test]
fn test_garbage_oracle_replacement_char() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "Hello\u{FFFD}World");
    assert!(result.is_falsified());
}

/// Verify garbage oracle falsifies output containing Inf values
#[test]
fn test_garbage_oracle_inf() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "The value is Inf");
    assert!(result.is_falsified());

    let result2 = oracle.evaluate("test", "The value is inf");
    assert!(result2.is_falsified());
}

/// Verify garbage oracle falsifies whitespace-only output
#[test]
fn test_garbage_oracle_whitespace_only() {
    let oracle = GarbageOracle::new();
    let result = oracle.evaluate("test", "   \n\t  ");
    assert!(result.is_falsified());
}

/// Verify code syntax oracle corroborates valid code output
#[test]
fn test_code_syntax_oracle_valid() {
    let oracle = CodeSyntaxOracle::new();
    let result = oracle.evaluate("def foo():", "    return 42");
    assert!(result.is_corroborated());
}

/// Verify code syntax oracle corroborates output with code patterns
#[test]
fn test_code_syntax_oracle_with_patterns() {
    let oracle = CodeSyntaxOracle::new();
    let result = oracle.evaluate("test", "fn main() { let x = 5; }");
    assert!(result.is_corroborated());
}

/// Verify code syntax oracle falsifies long prose without code patterns
#[test]
fn test_code_syntax_oracle_long_prose() {
    let oracle = CodeSyntaxOracle::new();
    let result = oracle.evaluate(
        "test",
        "This is a long description that doesn't contain any code patterns whatsoever.",
    );
    assert!(result.is_falsified()); // Popperian: no code patterns → falsified
}

/// Verify code syntax oracle falsifies empty output
#[test]
fn test_code_syntax_oracle_garbage() {
    let oracle = CodeSyntaxOracle::new();
    let result = oracle.evaluate("test", "");
    assert!(result.is_falsified());
}

/// Verify composite oracle corroborates when all sub-oracles pass
#[test]
fn test_composite_oracle_all_pass() {
    let mut composite = CompositeOracle::new("test");
    composite.add(GarbageOracle::new());
    let result = composite.evaluate("test", "Valid output here");
    assert!(result.is_corroborated());
}

/// Verify composite oracle falsifies when any sub-oracle fails
#[test]
fn test_composite_oracle_one_fails() {
    let mut composite = CompositeOracle::new("test");
    composite.add(GarbageOracle::new());
    let result = composite.evaluate("test", "");
    assert!(result.is_falsified());
}

/// Verify CompositeOracle Debug format contains struct name and label
#[test]
fn test_composite_oracle_debug() {
    let composite = CompositeOracle::new("test");
    let debug_str = format!("{composite:?}");
    assert!(debug_str.contains("CompositeOracle"));
    assert!(debug_str.contains("test"));
}

/// Verify arithmetic oracle evaluates subtraction correctly
#[test]
fn test_arithmetic_eval_subtraction() {
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("10-3=", "7");
    assert!(result.is_corroborated());
}

/// Verify arithmetic oracle evaluates multiplication correctly
#[test]
fn test_arithmetic_eval_multiplication() {
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("5*6=", "30");
    assert!(result.is_corroborated());
}

/// Verify arithmetic oracle evaluates division correctly
#[test]
fn test_arithmetic_eval_division() {
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("20/4=", "5");
    assert!(result.is_corroborated());
}

/// Verify arithmetic oracle falsifies division by zero (Popperian: unevaluable ≠ pass)
#[test]
fn test_arithmetic_division_by_zero() {
    let oracle = ArithmeticOracle::new();
    // Division by zero is arithmetic but unevaluable → Falsified
    // (absence of test ≠ pass; Popperian falsification)
    let result = oracle.evaluate("5/0=", "undefined");
    assert!(result.is_falsified());
}

/// Verify natural language arithmetic prompts are correctly evaluated
#[test]
fn test_arithmetic_natural_language_extraction() {
    let oracle = ArithmeticOracle::new();
    // "What is 2+2?" → extracts "2+2" → expected 4
    let result = oracle.evaluate("What is 2+2?", "The answer is 4.");
    assert!(result.is_corroborated());
    // "Calculate 7*8" → extracts "7*8" → expected 56
    let result = oracle.evaluate("Calculate 7*8", "56");
    assert!(result.is_corroborated());
    // Wrong answer for natural language prompt → Falsified
    let result = oracle.evaluate("What is 15-7?", "The answer is 10.");
    assert!(result.is_falsified());
}

/// Verify is_arithmetic_prompt detects math expressions
#[test]
fn test_is_arithmetic_prompt() {
    assert!(is_arithmetic_prompt("2+2="));
    assert!(is_arithmetic_prompt("What is 3*4?"));
    assert!(!is_arithmetic_prompt("Hello world"));
}

/// Verify is_raw_arithmetic_expr distinguishes raw expressions from natural language
#[test]
fn test_is_raw_arithmetic_expr() {
    assert!(is_raw_arithmetic_expr("2+2="));
    assert!(is_raw_arithmetic_expr("5/0="));
    assert!(is_raw_arithmetic_expr("100*3?"));
    assert!(is_raw_arithmetic_expr("-5+3"));
    assert!(!is_raw_arithmetic_expr("What is 2+2?"));
    assert!(!is_raw_arithmetic_expr("Calculate 5*3"));
    assert!(!is_raw_arithmetic_expr("Hello world"));
    assert!(!is_raw_arithmetic_expr(""));
}

/// Verify is_code_prompt detects code syntax patterns
#[test]
fn test_is_code_prompt() {
    assert!(is_code_prompt("def foo():"));
    assert!(is_code_prompt("fn main() {"));
    assert!(is_code_prompt("function test() {"));
    assert!(is_code_prompt("class Foo:"));
    assert!(is_code_prompt("async function bar() {"));
    assert!(is_code_prompt("```python\nx=1\n```"));
    assert!(!is_code_prompt("Hello world"));
}

/// Verify truncate shortens strings and appends ellipsis when needed
#[test]
fn test_truncate() {
    assert_eq!(truncate("hello", 10), "hello");
    assert_eq!(truncate("hello world", 5), "hello...");
}

/// Verify each oracle returns its expected name string
#[test]
fn test_oracle_names() {
    assert_eq!(ArithmeticOracle::new().name(), "arithmetic");
    assert_eq!(GarbageOracle::new().name(), "garbage");
    assert_eq!(CodeSyntaxOracle::new().name(), "code_syntax");
}

/// Verify OracleResult clone preserves corroborated status
#[test]
fn test_oracle_result_clone() {
    let result = OracleResult::Corroborated {
        evidence: "test".to_string(),
    };
    let cloned = result.clone();
    assert!(cloned.is_corroborated());
}

/// Verify OracleResult serializes to JSON with variant name
#[test]
fn test_oracle_result_serialize() {
    let result = OracleResult::Falsified {
        reason: "bad".to_string(),
        evidence: "test".to_string(),
    };
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(json.contains("Falsified"));
}

// Mutation-killing tests for arithmetic operations
/// Verify addition is not confused with multiplication (mutation kill)
#[test]
fn test_arithmetic_addition_not_multiplication() {
    // If + were replaced with *, 2+3 would give 6, not 5
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("2+3=", "5");
    assert!(result.is_corroborated());
    let wrong = oracle.evaluate("2+3=", "6");
    assert!(wrong.is_falsified());
}

/// Verify subtraction is not confused with addition (mutation kill)
#[test]
fn test_arithmetic_subtraction_not_other() {
    // If - were replaced with +, 10-3 would give 13, not 7
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("10-3=", "7");
    assert!(result.is_corroborated());
    let wrong = oracle.evaluate("10-3=", "13");
    assert!(wrong.is_falsified());
}

/// Verify multiplication is not confused with addition (mutation kill)
#[test]
fn test_arithmetic_multiplication_not_addition() {
    // If * were replaced with +, 3*4 would give 7, not 12
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("3*4=", "12");
    assert!(result.is_corroborated());
    let wrong = oracle.evaluate("3*4=", "7");
    assert!(wrong.is_falsified());
}

// Mutation-killing tests for is_arithmetic_prompt
/// Verify is_arithmetic_prompt requires both operator AND digit
#[test]
fn test_is_arithmetic_requires_both_operator_and_digit() {
    // Must have BOTH operator AND digit (tests && vs ||)
    assert!(!is_arithmetic_prompt("hello + world")); // has + but no digit
    assert!(!is_arithmetic_prompt("123")); // has digit but no operator
    assert!(is_arithmetic_prompt("1+2")); // has both
}

/// Verify all four arithmetic operators are recognized
#[test]
fn test_is_arithmetic_all_operators() {
    assert!(is_arithmetic_prompt("1+2"));
    assert!(is_arithmetic_prompt("5-3"));
    assert!(is_arithmetic_prompt("4*6"));
    assert!(is_arithmetic_prompt("8/2"));
}

// Mutation-killing tests for is_repetitive
/// Verify repetition detection requires minimum word count
#[test]
fn test_is_repetitive_needs_minimum_words() {
    // < 5 distinct words without char-level repetition → false
    assert!(!is_repetitive("one two three four"));
    // "a a a a" now correctly detected by char-level ngram check
    assert!(is_repetitive("a a a a"));
    assert!(is_repetitive("a a a a a")); // 5 words, all same
}

/// Verify two-word repeating pattern detection
#[test]
fn test_is_repetitive_two_word_pattern() {
    // Test 2-word pattern detection
    assert!(is_repetitive("foo bar foo bar foo bar"));
    // 6 words, threshold = 6/2/2 = 1, and first pair matches, so returns true
    // To test non-repetitive, need more words with fewer matches
    assert!(!is_repetitive("a b c d e f g h i j k l m n o p"));
}

/// Verify partial matches do not exceed repetition threshold
#[test]
fn test_is_repetitive_match_count_threshold() {
    // Partial matches shouldn't trigger
    assert!(!is_repetitive("a b c d e f g h i j"));
    assert!(is_repetitive("x y x y x y x y x y"));
}

// Mutation-killing tests for GarbageOracle conditions
/// Verify garbage oracle detects NaN, Inf, and inf patterns
#[test]
fn test_garbage_detects_different_nan_cases() {
    let oracle = GarbageOracle::new();
    // Checks for "NaN", "Inf", "inf" (case-sensitive)
    assert!(oracle.evaluate("test", "result: NaN").is_falsified());
    assert!(oracle.evaluate("test", "Inf value").is_falsified());
    assert!(oracle.evaluate("test", "inf error").is_falsified());
}

/// Verify garbage oracle distinguishes empty, whitespace, and real content
#[test]
fn test_garbage_non_empty_non_whitespace() {
    let oracle = GarbageOracle::new();
    // Empty is falsified
    assert!(oracle.evaluate("test", "").is_falsified());
    // Whitespace only is falsified
    assert!(oracle.evaluate("test", "   ").is_falsified());
    // Real content is corroborated
    assert!(oracle.evaluate("test", "x").is_corroborated());
}

// Mutation-killing tests for CodeSyntaxOracle
/// Verify code syntax oracle detects return, def, and fn patterns
#[test]
fn test_code_syntax_detects_patterns() {
    let oracle = CodeSyntaxOracle::new();
    // Should find code patterns
    assert!(oracle.evaluate("code", "return x;").is_corroborated());
    assert!(oracle.evaluate("code", "def foo(): pass").is_corroborated());
    assert!(oracle.evaluate("code", "fn bar() {}").is_corroborated());
}

// Test oracle name returns are not empty
/// Verify all oracle name() methods return non-empty strings
#[test]
fn test_oracle_names_not_empty() {
    assert!(!ArithmeticOracle::new().name().is_empty());
    assert!(!GarbageOracle::new().name().is_empty());
    assert!(!CodeSyntaxOracle::new().name().is_empty());
    let composite = CompositeOracle::new("test");
    assert!(!composite.name().is_empty());
}

// Test OracleWrapper name delegation
/// Verify OracleWrapper delegates name() to wrapped oracle
#[test]
fn test_oracle_wrapper_name() {
    let wrapper = OracleWrapper(ArithmeticOracle::new());
    assert_eq!(wrapper.name(), "arithmetic");
}

// --- Character-level n-gram repetition tests ---

/// Verify VILLE-style character repetition is detected
#[test]
fn test_char_ngram_ville_pattern() {
    // The motivating case from aprender#189
    assert!(check_substring_repetition("VILLEVILLEVILLEVILLE"));
    assert!(is_repetitive("VILLEVILLEVILLEVILLE"));
}

/// Verify short repeating character patterns are detected
#[test]
fn test_char_ngram_short_patterns() {
    assert!(check_substring_repetition("abcabcabc"));
    assert!(check_substring_repetition("xyxyxyxy"));
}

/// Verify longer repeating substrings are detected
#[test]
fn test_char_ngram_longer_patterns() {
    assert!(check_substring_repetition("helloWorldhelloWorldhelloWorld"));
}

/// Verify normal prose does not trigger n-gram repetition detection
#[test]
fn test_char_ngram_not_triggered_on_normal_text() {
    assert!(!check_substring_repetition("The quick brown fox"));
    assert!(!check_substring_repetition("Hello, world!"));
    assert!(!check_substring_repetition(
        "Rust is a systems programming language"
    ));
    assert!(!has_char_ngram_repetition(
        "The quick brown fox jumps over the lazy dog"
    ));
}

/// Verify per-word n-gram detection catches garbage words in normal text
#[test]
fn test_char_ngram_per_word_detection() {
    // Garbage word embedded in normal sentence
    assert!(has_char_ngram_repetition("output VILLEVILLEVILLEVILLE end"));
}

/// Verify single character repeated many times is detected
#[test]
fn test_char_ngram_single_char_repeat() {
    // "aaaaaaaaaaaa" — period 2 "aa" repeats 6 times, coverage 100%
    assert!(check_substring_repetition("aaaaaaaaaaaa"));
}

/// Verify partial coverage below 70% threshold is not flagged
#[test]
fn test_char_ngram_partial_coverage_not_flagged() {
    // "abcabcXYZ" — 2 reps of "abc" = 6/9 = 66%, below 70% threshold
    assert!(!check_substring_repetition("abcabcXYZ"));
}

// --- NaN/Inf word-boundary tests ---

/// Verify NaN/Inf detection does not false-positive on common English words
#[test]
fn test_garbage_oracle_no_false_positive_on_inf_words() {
    let oracle = GarbageOracle::new();
    // "information" contains "inf" but should NOT be flagged
    assert!(oracle.evaluate("test", "For more information, visit our website").is_corroborated());
    // "Infinity" contains "Inf" but is a different token
    assert!(oracle.evaluate("test", "Infinity stones are fictional").is_corroborated());
    // "Infrastructure" contains "Inf" substring
    assert!(oracle.evaluate("test", "Cloud infrastructure is important").is_corroborated());
}

/// Verify NaN/Inf detection catches standalone tokens
#[test]
fn test_garbage_oracle_catches_standalone_nan_inf() {
    let oracle = GarbageOracle::new();
    assert!(oracle.evaluate("test", "result: NaN").is_falsified());
    assert!(oracle.evaluate("test", "result: nan").is_falsified());
    assert!(oracle.evaluate("test", "loss: inf").is_falsified());
    assert!(oracle.evaluate("test", "[NaN, Inf, -Inf]").is_falsified());
}

/// Verify NaN/Inf detection in JSON, equals, angle brackets, and pipes
#[test]
fn test_nan_detection_additional_delimiters() {
    let oracle = GarbageOracle::new();
    // JSON embedded NaN
    assert!(oracle.evaluate("test", r#"{"value":NaN}"#).is_falsified());
    // Equals-delimited
    assert!(oracle.evaluate("test", "value=NaN").is_falsified());
    // Angle brackets
    assert!(oracle.evaluate("test", "<NaN>").is_falsified());
    // Quoted
    assert!(oracle.evaluate("test", r#""NaN""#).is_falsified());
    // Pipe-delimited
    assert!(oracle.evaluate("test", "NaN|Inf").is_falsified());
    // Still no false positives on words containing "nan"/"inf"
    assert!(oracle.evaluate("test", "information about nanotech").is_corroborated());
}

/// Verify two-word repetition detected at mid-output offset
#[test]
fn test_mid_output_two_word_repetition() {
    // Repetition starts after a coherent prefix — old code missed this
    assert!(is_repetitive("Normal start foo bar foo bar foo bar foo bar"));
    // Still detects repetition from the beginning
    assert!(is_repetitive("x y x y x y x y x y"));
}

