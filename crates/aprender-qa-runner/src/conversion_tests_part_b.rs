#[test]
fn test_conversion_execution_result_debug() {
    let result = ConversionExecutionResult {
        passed: 5,
        failed: 1,
        total: 6,
        evidence: vec![],
        results: vec![],
        duration_ms: 500,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("ConversionExecutionResult"));
}

#[test]
fn test_all_backends_content() {
    let backends = all_backends();
    assert!(backends.contains(&Backend::Cpu));
    assert!(backends.contains(&Backend::Gpu));
}

#[test]
fn test_gate_id_all_combinations() {
    // Test all source/target combinations
    let combos = [
        (Format::Gguf, Format::Apr, "F-CONV-G-A"),
        (Format::Apr, Format::Gguf, "F-CONV-A-G"),
        (Format::Gguf, Format::SafeTensors, "F-CONV-G-S"),
        (Format::SafeTensors, Format::Gguf, "F-CONV-S-G"),
        (Format::Apr, Format::SafeTensors, "F-CONV-A-S"),
        (Format::SafeTensors, Format::Apr, "F-CONV-S-A"),
    ];

    for (source, target, expected) in combos {
        let test = ConversionTest::new(source, target, Backend::Cpu, ModelId::new("t", "m"));
        assert_eq!(test.gate_id(), expected);
    }
}

#[test]
fn test_compute_diff_partially_matching() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // "hello" vs "hallo" - 1 char different out of 5
    let diff = test.compute_diff("hello", "hallo");
    assert!(diff > 0.0);
    assert!(diff < 1.0);
}

#[test]
fn test_find_diff_indices_longer_second() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // "ab" vs "abc" - trailing difference at index 2
    let indices = test.find_diff_indices("ab", "abc");
    assert_eq!(indices, vec![2]); // position 2 differs (shorter string ends)
}

#[test]
fn test_conversion_execution_result_all_passed() {
    let result = ConversionExecutionResult {
        passed: 10,
        failed: 0,
        total: 10,
        evidence: vec![],
        results: vec![],
        duration_ms: 1000,
    };
    assert!(result.all_passed());
}

#[test]
fn test_conversion_execution_result_not_all_passed() {
    let result = ConversionExecutionResult {
        passed: 8,
        failed: 2,
        total: 10,
        evidence: vec![],
        results: vec![],
        duration_ms: 1000,
    };
    assert!(!result.all_passed());
}

#[test]
fn test_conversion_execution_result_pass_rate() {
    let result = ConversionExecutionResult {
        passed: 8,
        failed: 2,
        total: 10,
        evidence: vec![],
        results: vec![],
        duration_ms: 1000,
    };
    let rate = result.pass_rate();
    assert!((rate - 80.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_execution_result_pass_rate_zero_total() {
    let result = ConversionExecutionResult {
        passed: 0,
        failed: 0,
        total: 0,
        evidence: vec![],
        results: vec![],
        duration_ms: 0,
    };
    let rate = result.pass_rate();
    // Popperian: 0 tests = 0% pass rate (untested ≠ passed)
    assert!(rate.abs() < f64::EPSILON);
}

#[test]
fn test_conversion_execution_result_pass_rate_all_passed() {
    let result = ConversionExecutionResult {
        passed: 5,
        failed: 0,
        total: 5,
        evidence: vec![],
        results: vec![],
        duration_ms: 500,
    };
    let rate = result.pass_rate();
    assert!((rate - 100.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_execution_result_pass_rate_none_passed() {
    let result = ConversionExecutionResult {
        passed: 0,
        failed: 5,
        total: 5,
        evidence: vec![],
        results: vec![],
        duration_ms: 500,
    };
    let rate = result.pass_rate();
    assert!((rate - 0.0).abs() < f64::EPSILON);
}

// Tests for ConversionBugType (GH-187)

#[test]
fn test_bug_type_gate_ids() {
    assert_eq!(
        ConversionBugType::EmbeddingTransposition.gate_id(),
        "F-CONV-EMBED-001"
    );
    assert_eq!(
        ConversionBugType::TokenizerMissing.gate_id(),
        "F-CONV-TOK-001"
    );
    assert_eq!(
        ConversionBugType::WeightCorruption.gate_id(),
        "F-CONV-WEIGHT-001"
    );
    assert_eq!(
        ConversionBugType::ShapeMismatch.gate_id(),
        "F-CONV-SHAPE-001"
    );
    assert_eq!(
        ConversionBugType::SemanticDrift.gate_id(),
        "F-CONV-SEMANTIC-001"
    );
    assert_eq!(ConversionBugType::Unknown.gate_id(), "F-CONV-UNKNOWN-001");
}

#[test]
fn test_bug_type_descriptions() {
    assert!(
        ConversionBugType::EmbeddingTransposition
            .description()
            .contains("transposition")
    );
    assert!(
        ConversionBugType::TokenizerMissing
            .description()
            .contains("tokenizer")
    );
    assert!(
        ConversionBugType::WeightCorruption
            .description()
            .contains("corruption")
    );
}

#[test]
fn test_bug_type_clone() {
    let bug = ConversionBugType::EmbeddingTransposition;
    let cloned = bug;
    assert_eq!(bug, cloned);
}

#[test]
fn test_bug_type_debug() {
    let debug_str = format!("{:?}", ConversionBugType::TokenizerMissing);
    assert!(debug_str.contains("TokenizerMissing"));
}

#[test]
fn test_semantic_test_new() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    assert_eq!(test.source_format, Format::Gguf);
    assert_eq!(test.target_format, Format::Apr);
}

#[test]
fn test_semantic_test_clone() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let cloned = test.clone();
    assert_eq!(test.source_format, cloned.source_format);
}

#[test]
fn test_semantic_test_debug() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let debug_str = format!("{test:?}");
    assert!(debug_str.contains("SemanticConversionTest"));
}

#[test]
fn test_semantic_result_is_pass() {
    let pass = SemanticTestResult::Corroborated {
        source_output: "4".to_string(),
        target_output: "4".to_string(),
    };
    assert!(pass.is_pass());

    let fail = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::EmbeddingTransposition,
        source_output: "4".to_string(),
        target_output: "garbage".to_string(),
        stderr: String::new(),
    };
    assert!(!fail.is_pass());
}

#[test]
fn test_semantic_result_bug_type() {
    let pass = SemanticTestResult::Corroborated {
        source_output: "4".to_string(),
        target_output: "4".to_string(),
    };
    assert!(pass.bug_type().is_none());

    let fail = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::TokenizerMissing,
        source_output: "4".to_string(),
        target_output: "garbage".to_string(),
        stderr: String::new(),
    };
    assert_eq!(fail.bug_type(), Some(ConversionBugType::TokenizerMissing));
}

#[test]
fn test_garbage_patterns_detection() {
    // These patterns should trigger embedding transposition detection
    let garbage_outputs = [
        "1. What is the difference between",
        "<pad><pad><pad>",
        "PAD PAD PAD",
        "token 151935 151935",
    ];

    for output in garbage_outputs {
        let has_garbage = GARBAGE_PATTERNS.iter().any(|p| output.contains(p));
        assert!(has_garbage, "Should detect garbage in: {output}");
    }
}

#[test]
fn test_arithmetic_expected_detection() {
    // These patterns should be recognized as correct answers
    let correct_outputs = [
        "The answer is 4",
        "2+2=4",
        "equals 4.",
        "It's four",
        "Four is the answer",
    ];

    for output in correct_outputs {
        let has_expected = ARITHMETIC_EXPECTED.iter().any(|p| output.contains(p));
        assert!(has_expected, "Should detect correct answer in: {output}");
    }
}

#[test]
fn test_semantic_result_clone() {
    let result = SemanticTestResult::Corroborated {
        source_output: "test".to_string(),
        target_output: "test".to_string(),
    };
    let cloned = result.clone();
    assert!(cloned.is_pass());
}

#[test]
fn test_semantic_result_debug() {
    let result = SemanticTestResult::Falsified {
        bug_type: ConversionBugType::Unknown,
        source_output: "a".to_string(),
        target_output: "b".to_string(),
        stderr: String::new(),
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("Falsified"));
}

// Tests for classify_bug logic
#[test]
fn test_classify_bug_tokenizer_missing() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let bug = test.classify_bug("The answer is 4", "The answer is 4", true);
    assert_eq!(bug, Some(ConversionBugType::TokenizerMissing));
}

#[test]
fn test_classify_bug_embedding_transposition() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source has correct answer, target has garbage
    let bug = test.classify_bug("The answer is 4", "PAD PAD PAD garbage", false);
    assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
}

#[test]
fn test_classify_bug_semantic_drift() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source has correct answer, target has wrong but not garbage answer
    let bug = test.classify_bug("The answer is 4", "The answer is 7", false);
    assert_eq!(bug, Some(ConversionBugType::SemanticDrift));
}

#[test]
fn test_classify_bug_weight_corruption() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source has output (but not expected arithmetic answer), target is empty
    // WeightCorruption is only detected when target is empty/whitespace
    let bug = test.classify_bug("Hello world, here is some text", "   ", false);
    assert_eq!(bug, Some(ConversionBugType::WeightCorruption));
}

#[test]
fn test_classify_bug_no_bug() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Both outputs are identical
    let bug = test.classify_bug("The answer is 4", "The answer is 4", false);
    assert!(bug.is_none());
}

#[test]
fn test_classify_bug_unknown() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Source has no expected answer, outputs differ
    let bug = test.classify_bug("random text", "different text", false);
    assert_eq!(bug, Some(ConversionBugType::Unknown));
}

#[test]
fn test_classify_bug_with_endoftext_pattern() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let bug = test.classify_bug(
        "The answer is 4",
        "Output: <|endoftext|><|endoftext|>",
        false,
    );
    assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
}

#[test]
fn test_classify_bug_with_null_chars() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let bug = test.classify_bug("The answer is 4", "text\u{0000}with\u{0000}nulls", false);
    assert_eq!(bug, Some(ConversionBugType::EmbeddingTransposition));
}

#[test]
fn test_classify_bug_whitespace_trimming() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Same content but different whitespace - should match
    let bug = test.classify_bug("  The answer is 4  ", "The answer is 4", false);
    assert!(bug.is_none());
}
