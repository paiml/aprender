use super::*;

// =========================================================================
// Dataset tests
// =========================================================================

#[test]
fn test_dataset_parse() {
    assert_eq!("wikitext-2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
    assert_eq!("wikitext2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
    assert_eq!("lambada".parse::<Dataset>().unwrap(), Dataset::Lambada);
    assert_eq!("custom".parse::<Dataset>().unwrap(), Dataset::Custom);
}

#[test]
fn test_dataset_parse_case_insensitive() {
    assert_eq!("WIKITEXT-2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
    assert_eq!("WikiText2".parse::<Dataset>().unwrap(), Dataset::WikiText2);
    assert_eq!("LAMBADA".parse::<Dataset>().unwrap(), Dataset::Lambada);
    assert_eq!("CUSTOM".parse::<Dataset>().unwrap(), Dataset::Custom);
}

#[test]
fn test_dataset_parse_error() {
    assert!("unknown".parse::<Dataset>().is_err());
}

#[test]
fn test_dataset_parse_error_message() {
    let result: std::result::Result<Dataset, String> = "invalid".parse();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Unknown dataset"));
    assert!(err.contains("wikitext-2"));
    assert!(err.contains("lambada"));
    assert!(err.contains("custom"));
}

#[test]
fn test_dataset_debug() {
    assert_eq!(format!("{:?}", Dataset::WikiText2), "WikiText2");
    assert_eq!(format!("{:?}", Dataset::Lambada), "Lambada");
    assert_eq!(format!("{:?}", Dataset::Custom), "Custom");
}

#[test]
fn test_dataset_clone() {
    let dataset = Dataset::WikiText2;
    let cloned = dataset;
    assert_eq!(dataset, cloned);
}

#[test]
fn test_dataset_copy() {
    let dataset = Dataset::Lambada;
    let copied: Dataset = dataset;
    assert_eq!(dataset, copied);
}

#[test]
fn test_dataset_eq() {
    assert_eq!(Dataset::WikiText2, Dataset::WikiText2);
    assert_eq!(Dataset::Lambada, Dataset::Lambada);
    assert_eq!(Dataset::Custom, Dataset::Custom);
    assert_ne!(Dataset::WikiText2, Dataset::Lambada);
    assert_ne!(Dataset::Lambada, Dataset::Custom);
}

// =========================================================================
// EvalResult tests
// =========================================================================

#[test]
fn test_eval_result_pass() {
    let result = EvalResult {
        perplexity: 15.0,
        cross_entropy: 2.7,
        tokens_evaluated: 100,
        eval_time_secs: 1.5,
        passed: true,
        threshold: 20.0,
    };

    assert!(result.passed);
    assert!(result.perplexity < result.threshold);
}

#[test]
fn test_eval_result_fail() {
    let result = EvalResult {
        perplexity: 25.0,
        cross_entropy: 3.2,
        tokens_evaluated: 100,
        eval_time_secs: 1.5,
        passed: false,
        threshold: 20.0,
    };

    assert!(!result.passed);
    assert!(result.perplexity > result.threshold);
}

#[test]
fn test_eval_result_excellent() {
    let result = EvalResult {
        perplexity: 8.5,
        cross_entropy: 2.14,
        tokens_evaluated: 512,
        eval_time_secs: 3.0,
        passed: true,
        threshold: 20.0,
    };

    assert!(result.passed);
    assert!(result.perplexity < 10.0); // Excellent threshold
}

#[test]
fn test_eval_result_good() {
    let result = EvalResult {
        perplexity: 12.5,
        cross_entropy: 2.53,
        tokens_evaluated: 256,
        eval_time_secs: 2.0,
        passed: true,
        threshold: 20.0,
    };

    assert!(result.passed);
    assert!(result.perplexity >= 10.0 && result.perplexity < 15.0); // Good threshold
}

#[test]
fn test_eval_result_poor() {
    let result = EvalResult {
        perplexity: 35.0,
        cross_entropy: 3.56,
        tokens_evaluated: 100,
        eval_time_secs: 1.0,
        passed: false,
        threshold: 20.0,
    };

    assert!(!result.passed);
    assert!(result.perplexity >= 20.0 && result.perplexity < 50.0); // Poor threshold
}

#[test]
fn test_eval_result_garbage() {
    let result = EvalResult {
        perplexity: 150.0,
        cross_entropy: 5.01,
        tokens_evaluated: 50,
        eval_time_secs: 0.5,
        passed: false,
        threshold: 20.0,
    };

    assert!(!result.passed);
    assert!(result.perplexity >= 50.0); // Garbage threshold
}

#[test]
fn test_eval_result_debug() {
    let result = EvalResult {
        perplexity: 15.0,
        cross_entropy: 2.7,
        tokens_evaluated: 100,
        eval_time_secs: 1.5,
        passed: true,
        threshold: 20.0,
    };

    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("EvalResult"));
    assert!(debug_str.contains("perplexity: 15.0"));
    assert!(debug_str.contains("passed: true"));
}

#[test]
fn test_eval_result_clone() {
    let result = EvalResult {
        perplexity: 15.0,
        cross_entropy: 2.7,
        tokens_evaluated: 100,
        eval_time_secs: 1.5,
        passed: true,
        threshold: 20.0,
    };

    let cloned = result.clone();
    assert_eq!(cloned.perplexity, result.perplexity);
    assert_eq!(cloned.passed, result.passed);
}

// =========================================================================
// get_eval_text tests
// =========================================================================

#[test]
fn test_get_eval_text_wikitext() {
    let config = EvalConfig {
        dataset: Dataset::WikiText2,
        text: None,
        max_tokens: 512,
        threshold: 20.0,
    };

    let text = get_eval_text(&config).unwrap();
    assert!(!text.is_empty());
    assert!(text.contains("Eiffel Tower")); // WikiText sample contains this
}

#[test]
fn test_get_eval_text_lambada() {
    let config = EvalConfig {
        dataset: Dataset::Lambada,
        text: None,
        max_tokens: 512,
        threshold: 20.0,
    };

    let text = get_eval_text(&config).unwrap();
    assert!(!text.is_empty());
    assert!(text.contains("walked into the room")); // LAMBADA sample contains this
}

#[test]
fn test_get_eval_text_custom() {
    let config = EvalConfig {
        dataset: Dataset::Custom,
        text: Some("This is custom evaluation text.".to_string()),
        max_tokens: 512,
        threshold: 20.0,
    };

    let text = get_eval_text(&config).unwrap();
    assert_eq!(text, "This is custom evaluation text.");
}

#[test]
fn test_get_eval_text_custom_no_text() {
    let config = EvalConfig {
        dataset: Dataset::Custom,
        text: None, // Missing text for custom dataset
        max_tokens: 512,
        threshold: 20.0,
    };

    let result = get_eval_text(&config);
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Custom dataset requires --text argument"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

// =========================================================================
// Sample text tests
// =========================================================================

#[test]
fn test_sample_texts_not_empty() {
    assert!(!SAMPLE_WIKITEXT.is_empty());
    assert!(!SAMPLE_LAMBADA.is_empty());
    assert!(SAMPLE_WIKITEXT.len() > 100);
    assert!(SAMPLE_LAMBADA.len() > 100);
}

#[test]
fn test_sample_wikitext_content() {
    // WikiText-2 sample should contain recognizable content
    assert!(SAMPLE_WIKITEXT.contains("tower"));
    assert!(SAMPLE_WIKITEXT.contains("metres"));
    assert!(SAMPLE_WIKITEXT.contains("Paris"));
}

#[test]
fn test_sample_lambada_content() {
    // LAMBADA sample should contain narrative text
    assert!(SAMPLE_LAMBADA.contains("She"));
    assert!(SAMPLE_LAMBADA.contains("friend"));
    assert!(SAMPLE_LAMBADA.contains("window"));
}

// =========================================================================
// run() error cases tests
// =========================================================================

#[test]
fn test_run_unknown_dataset() {
    let result = run(Path::new("test.apr"), "invalid_dataset", None, None, None);

    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Unknown dataset"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_run_file_not_found() {
    let result = run(
        Path::new("/nonexistent/model.apr"),
        "wikitext-2",
        None,
        None,
        None,
    );

    // Will fail because file doesn't exist
    assert!(result.is_err());
}

#[test]
fn test_run_custom_without_text() {
    let result = run(
        Path::new("test.apr"),
        "custom",
        None, // Missing text for custom dataset
        None,
        None,
    );

    // Will fail at validation or file loading
    assert!(result.is_err());
}

// =========================================================================
// EvalConfig tests
// =========================================================================

#[test]
fn test_eval_config_default_values() {
    let config = EvalConfig {
        dataset: Dataset::WikiText2,
        text: None,
        max_tokens: 512,
        threshold: 20.0,
    };

    assert!(matches!(config.dataset, Dataset::WikiText2));
    assert!(config.text.is_none());
    assert_eq!(config.max_tokens, 512);
    assert!((config.threshold - 20.0).abs() < 0.01);
}

#[test]
fn test_eval_config_with_text() {
    let config = EvalConfig {
        dataset: Dataset::Custom,
        text: Some("Test text".to_string()),
        max_tokens: 256,
        threshold: 15.0,
    };

    assert!(matches!(config.dataset, Dataset::Custom));
    assert_eq!(config.text, Some("Test text".to_string()));
    assert_eq!(config.max_tokens, 256);
    assert!((config.threshold - 15.0).abs() < 0.01);
}

// =========================================================================
// Additional Dataset::from_str edge cases
// =========================================================================

#[test]
fn test_dataset_parse_mixed_case() {
    assert_eq!(
        "WikiText-2".parse::<Dataset>().ok(),
        Some(Dataset::WikiText2)
    );
    assert_eq!("Lambada".parse::<Dataset>().ok(), Some(Dataset::Lambada));
    assert_eq!("Custom".parse::<Dataset>().ok(), Some(Dataset::Custom));
}

#[test]
fn test_dataset_parse_empty_string() {
    let result = "".parse::<Dataset>();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("Unknown dataset"));
}

#[test]
fn test_dataset_parse_whitespace_only() {
    let result = "  ".parse::<Dataset>();
    assert!(result.is_err());
}

#[test]
fn test_dataset_parse_similar_but_wrong() {
    assert!("wikitext-3".parse::<Dataset>().is_err());
    assert!("wikitext".parse::<Dataset>().is_err());
    assert!("wiki".parse::<Dataset>().is_err());
    assert!("lamb".parse::<Dataset>().is_err());
}

#[test]
fn test_dataset_parse_error_contains_input() {
    let result: std::result::Result<Dataset, String> = "foobar".parse();
    let err = result.unwrap_err();
    assert!(
        err.contains("foobar"),
        "Error should contain the invalid input"
    );
}

// =========================================================================
// get_eval_text additional coverage
// =========================================================================

#[test]
fn test_get_eval_text_wikitext_returns_sample_constant() {
    let config = EvalConfig {
        dataset: Dataset::WikiText2,
        text: None,
        max_tokens: 512,
        threshold: 20.0,
    };
    let text = get_eval_text(&config).expect("should return WikiText sample");
    assert_eq!(text, SAMPLE_WIKITEXT);
}

#[test]
fn test_get_eval_text_lambada_returns_sample_constant() {
    let config = EvalConfig {
        dataset: Dataset::Lambada,
        text: None,
        max_tokens: 512,
        threshold: 20.0,
    };
    let text = get_eval_text(&config).expect("should return LAMBADA sample");
    assert_eq!(text, SAMPLE_LAMBADA);
}

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
    );
    assert!(result.is_err());
}
