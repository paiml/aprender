// ── Oracle edge cases for additional coverage ──────────────────────────────

/// Verify OracleWrapper delegates evaluate() to wrapped oracle
#[test]
fn test_oracle_wrapper_evaluate_delegates() {
    let wrapper = OracleWrapper(ArithmeticOracle::new());
    // Arithmetic prompt should corroborate correct answer
    let result = wrapper.evaluate("3+4=", "The result is 7.");
    assert!(result.is_corroborated());
    // Incorrect answer should falsify
    let result = wrapper.evaluate("3+4=", "The result is 9.");
    assert!(result.is_falsified());
}

/// Verify OracleWrapper with GarbageOracle detects empty output
#[test]
fn test_oracle_wrapper_garbage_evaluate() {
    let wrapper = OracleWrapper(GarbageOracle::new());
    let result = wrapper.evaluate("test", "");
    assert!(result.is_falsified());
    let result = wrapper.evaluate("test", "Valid text here");
    assert!(result.is_corroborated());
}

/// Verify eval_arithmetic handles expression with trailing question mark
#[test]
fn test_arithmetic_trailing_question_mark() {
    let oracle = ArithmeticOracle::new();
    // "3+5?" should parse as 3+5=8
    let result = oracle.evaluate("3+5?", "8");
    assert!(result.is_corroborated());
}

/// Verify eval_arithmetic handles expression with leading/trailing whitespace
#[test]
fn test_arithmetic_whitespace_handling() {
    let oracle = ArithmeticOracle::new();
    let result = oracle.evaluate("  7 + 3 = ", "10");
    assert!(result.is_corroborated());
}

/// Verify composite oracle with no sub-oracles returns corroborated with count 0
#[test]
fn test_composite_oracle_empty() {
    let composite = CompositeOracle::new("empty");
    let result = composite.evaluate("test", "output");
    assert!(result.is_corroborated());
    if let OracleResult::Corroborated { evidence } = result {
        assert!(evidence.contains("0 oracles"));
    }
}

/// Verify composite oracle with multiple sub-oracles counts correctly
#[test]
fn test_composite_oracle_multiple_all_pass() {
    let mut composite = CompositeOracle::new("multi");
    composite.add(GarbageOracle::new());
    composite.add(CodeSyntaxOracle::new());
    // Use a longer diverse string to avoid repetition detection false positives
    let result = composite.evaluate(
        "test",
        "fn calculate_sum(a: i32, b: i32) -> i32 { let result = a + b; return result; }",
    );
    assert!(result.is_corroborated());
    if let OracleResult::Corroborated { evidence } = result {
        assert!(evidence.contains("2 oracles"));
    }
}

/// Verify select_oracle recognizes async function prompts as code
#[test]
fn test_select_oracle_async_code() {
    let oracle = select_oracle("async function fetchData() {");
    assert_eq!(oracle.name(), "code_syntax");
}

/// Verify select_oracle recognizes class prompts as code
#[test]
fn test_select_oracle_class_prompt() {
    let oracle = select_oracle("class MyModel:");
    assert_eq!(oracle.name(), "code_syntax");
}

/// Verify is_repetitive with exactly 5 identical words triggers all_words_identical
#[test]
fn test_all_words_identical_exactly_five() {
    assert!(is_repetitive("same same same same same"));
}

/// Verify has_two_word_repetition with exactly 6 words at threshold
#[test]
fn test_two_word_repetition_boundary_six_words() {
    // 6 words, pairs: (foo bar), (foo bar), (foo bar)
    // matches = 3, threshold = 6/2/2 = 1, so 3 >= 1 → true
    assert!(is_repetitive("foo bar foo bar foo bar"));
}

/// Verify check_substring_repetition with max_period boundary
#[test]
fn test_char_ngram_max_period_boundary() {
    // Period 20 is the max — string must be at least 60 chars for period 20
    // 20-char pattern repeated 3 times = 60 chars
    let pattern = "abcdefghijklmnopqrst"; // 20 chars
    let repeated = pattern.repeat(3);
    assert!(check_substring_repetition(&repeated));
}
