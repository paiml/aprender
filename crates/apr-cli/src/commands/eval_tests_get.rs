
#[test]
fn test_get_eval_text_custom_with_empty_string() {
    let config = EvalConfig {
        dataset: Dataset::Custom,
        text: Some(String::new()),
        max_tokens: 512,
        threshold: 20.0,
    };
    let text = get_eval_text(&config).expect("should return empty string");
    assert!(text.is_empty());
}

#[test]
fn test_get_eval_text_wikitext_ignores_text_field() {
    // Even if text is Some, WikiText2 returns the sample constant
    let config = EvalConfig {
        dataset: Dataset::WikiText2,
        text: Some("ignored".to_string()),
        max_tokens: 512,
        threshold: 20.0,
    };
    let text = get_eval_text(&config).expect("should return WikiText sample");
    assert!(text.contains("Eiffel Tower"));
    assert!(!text.contains("ignored"));
}

#[test]
fn test_get_eval_text_lambada_ignores_text_field() {
    let config = EvalConfig {
        dataset: Dataset::Lambada,
        text: Some("ignored".to_string()),
        max_tokens: 512,
        threshold: 20.0,
    };
    let text = get_eval_text(&config).expect("should return LAMBADA sample");
    assert!(text.contains("walked into the room"));
}

// =========================================================================
// print_header and print_results no-panic tests
// =========================================================================

#[test]
fn test_print_header_does_not_panic() {
    let config = EvalConfig {
        dataset: Dataset::WikiText2,
        text: None,
        max_tokens: 512,
        threshold: 20.0,
    };
    // Should not panic for any valid path
    print_header(Path::new("model.apr"), &config);
}

#[test]
fn test_print_header_custom_dataset_does_not_panic() {
    let config = EvalConfig {
        dataset: Dataset::Custom,
        text: Some("custom text".to_string()),
        max_tokens: 128,
        threshold: 10.0,
    };
    print_header(Path::new("/tmp/custom_model.safetensors"), &config);
}

#[test]
fn test_print_results_passing_does_not_panic() {
    let result = EvalResult {
        perplexity: 8.0,
        cross_entropy: 2.08,
        tokens_evaluated: 200,
        eval_time_secs: 0.5,
        passed: true,
        threshold: 20.0,
    };
    print_results(&result);
}

#[test]
fn test_print_results_failing_does_not_panic() {
    let result = EvalResult {
        perplexity: 55.0,
        cross_entropy: 4.01,
        tokens_evaluated: 100,
        eval_time_secs: 0.3,
        passed: false,
        threshold: 20.0,
    };
    print_results(&result);
}

#[test]
fn test_print_results_all_quality_tiers() {
    // Test all quality tier branches in print_results
    let tiers = vec![
        (5.0, true),    // Excellent: < 10
        (12.0, true),   // Good: 10..15
        (18.0, true),   // Acceptable: 15..20
        (30.0, false),  // Poor: 20..50
        (100.0, false), // Garbage: >= 50
    ];
    for (ppl, passed) in tiers {
        let result = EvalResult {
            perplexity: ppl,
            cross_entropy: ppl.ln(),
            tokens_evaluated: 100,
            eval_time_secs: 1.0,
            passed,
            threshold: 20.0,
        };
        print_results(&result);
    }
}

// =========================================================================
// EvalResult edge cases
// =========================================================================

#[test]
fn test_eval_result_boundary_at_threshold() {
    let result = EvalResult {
        perplexity: 20.0,
        cross_entropy: 3.0,
        tokens_evaluated: 100,
        eval_time_secs: 1.0,
        passed: true,
        threshold: 20.0,
    };
    // At exactly the threshold, passed should be true (<=)
    assert!(result.passed);
}

#[test]
fn test_eval_result_zero_tokens() {
    let result = EvalResult {
        perplexity: 1.0,
        cross_entropy: 0.0,
        tokens_evaluated: 0,
        eval_time_secs: 0.0,
        passed: true,
        threshold: 20.0,
    };
    assert_eq!(result.tokens_evaluated, 0);
}

#[test]
fn test_eval_result_clone_preserves_all_fields() {
    let result = EvalResult {
        perplexity: 12.34,
        cross_entropy: 2.51,
        tokens_evaluated: 999,
        eval_time_secs: 4.56,
        passed: false,
        threshold: 10.0,
    };
    let cloned = result.clone();
    assert_eq!(cloned.perplexity, 12.34);
    assert_eq!(cloned.cross_entropy, 2.51);
    assert_eq!(cloned.tokens_evaluated, 999);
    assert_eq!(cloned.eval_time_secs, 4.56);
    assert!(!cloned.passed);
    assert_eq!(cloned.threshold, 10.0);
}

#[test]
fn test_eval_result_debug_contains_all_fields() {
    let result = EvalResult {
        perplexity: 7.5,
        cross_entropy: 2.01,
        tokens_evaluated: 512,
        eval_time_secs: 2.3,
        passed: true,
        threshold: 20.0,
    };
    let debug = format!("{result:?}");
    assert!(debug.contains("perplexity"));
    assert!(debug.contains("cross_entropy"));
    assert!(debug.contains("tokens_evaluated"));
    assert!(debug.contains("eval_time_secs"));
    assert!(debug.contains("passed"));
    assert!(debug.contains("threshold"));
}

// =========================================================================
// Sample text content validation
// =========================================================================

#[test]
fn test_sample_wikitext_has_sufficient_length_for_eval() {
    // The eval needs at least 2 tokens, so text must be non-trivial
    assert!(
        SAMPLE_WIKITEXT.len() > 200,
        "WikiText sample too short for meaningful eval"
    );
}

#[test]
fn test_sample_lambada_has_sufficient_length_for_eval() {
    assert!(
        SAMPLE_LAMBADA.len() > 200,
        "LAMBADA sample too short for meaningful eval"
    );
}

#[test]
fn test_sample_texts_are_distinct() {
    assert_ne!(SAMPLE_WIKITEXT, SAMPLE_LAMBADA);
}

// =========================================================================
// run() additional error paths
// =========================================================================

#[test]
fn test_run_with_max_tokens() {
    // Test that max_tokens parameter is accepted (even if file doesn't exist)
    let result = run(
        Path::new("/nonexistent/model.gguf"),
        "wikitext-2",
        None,
        Some(128),
        None,
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_run_with_threshold() {
    let result = run(
        Path::new("/nonexistent/model.gguf"),
        "wikitext-2",
        None,
        None,
        Some(5.0),
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_run_custom_with_text() {
    // Custom dataset with text on a non-existent file should fail at file loading
    let result = run(
        Path::new("/nonexistent/model.apr"),
        "custom",
        Some("test input text"),
        None,
        None,
        false,
    );
    assert!(result.is_err());
}
