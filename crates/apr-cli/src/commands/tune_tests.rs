use super::*;
use std::fs;

// =========================================================================
// TuneMethod tests
// =========================================================================

#[test]
fn test_tune_method_parse() {
    assert!(matches!(
        "lora".parse::<TuneMethod>().unwrap(),
        TuneMethod::LoRA
    ));
    assert!(matches!(
        "qlora".parse::<TuneMethod>().unwrap(),
        TuneMethod::QLoRA
    ));
    assert!(matches!(
        "auto".parse::<TuneMethod>().unwrap(),
        TuneMethod::Auto
    ));
    assert!(matches!(
        "full".parse::<TuneMethod>().unwrap(),
        TuneMethod::Full
    ));
}

#[test]
fn test_tune_method_parse_case_insensitive() {
    assert!(matches!(
        "LORA".parse::<TuneMethod>().unwrap(),
        TuneMethod::LoRA
    ));
    assert!(matches!(
        "LoRa".parse::<TuneMethod>().unwrap(),
        TuneMethod::LoRA
    ));
    assert!(matches!(
        "QLORA".parse::<TuneMethod>().unwrap(),
        TuneMethod::QLoRA
    ));
}

#[test]
fn test_tune_method_parse_invalid() {
    let result: Result<TuneMethod, _> = "invalid".parse();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown method"));
}

#[test]
fn test_tune_method_default() {
    let method = TuneMethod::default();
    assert!(matches!(method, TuneMethod::Auto));
}

#[test]
fn test_tune_method_debug() {
    assert_eq!(format!("{:?}", TuneMethod::Auto), "Auto");
    assert_eq!(format!("{:?}", TuneMethod::Full), "Full");
    assert_eq!(format!("{:?}", TuneMethod::LoRA), "LoRA");
    assert_eq!(format!("{:?}", TuneMethod::QLoRA), "QLoRA");
}

#[test]
fn test_tune_method_clone() {
    let method = TuneMethod::LoRA;
    let cloned = method;
    assert!(matches!(cloned, TuneMethod::LoRA));
}

#[test]
fn test_tune_method_copy() {
    let method = TuneMethod::QLoRA;
    let copied: TuneMethod = method;
    assert!(matches!(method, TuneMethod::QLoRA));
    assert!(matches!(copied, TuneMethod::QLoRA));
}

#[test]
fn test_tune_method_into_entrenar_method() {
    let auto: Method = TuneMethod::Auto.into();
    assert!(matches!(auto, Method::Auto));

    let full: Method = TuneMethod::Full.into();
    assert!(matches!(full, Method::Full));

    let lora: Method = TuneMethod::LoRA.into();
    assert!(matches!(lora, Method::LoRA));

    let qlora: Method = TuneMethod::QLoRA.into();
    assert!(matches!(qlora, Method::QLoRA));
}

// =========================================================================
// parse_model_size tests
// =========================================================================

#[test]
fn test_parse_model_size() {
    assert_eq!(parse_model_size("7B").unwrap(), 7_000_000_000);
    assert_eq!(parse_model_size("1.5B").unwrap(), 1_500_000_000);
    assert_eq!(parse_model_size("70B").unwrap(), 70_000_000_000);
    assert_eq!(parse_model_size("500M").unwrap(), 500_000_000);
}

#[test]
fn test_parse_model_size_case_insensitive() {
    assert_eq!(parse_model_size("7b").unwrap(), 7_000_000_000);
    assert_eq!(parse_model_size("1.5b").unwrap(), 1_500_000_000);
}

#[test]
fn test_parse_model_size_invalid() {
    assert!(parse_model_size("7").is_err());
    assert!(parse_model_size("7GB").is_err());
    assert!(parse_model_size("abc").is_err());
}

#[test]
fn test_parse_model_size_decimal() {
    assert_eq!(parse_model_size("0.5B").unwrap(), 500_000_000);
    assert_eq!(parse_model_size("2.7B").unwrap(), 2_700_000_000);
    assert_eq!(parse_model_size("13.5B").unwrap(), 13_500_000_000);
}

#[test]
fn test_parse_model_size_millions() {
    assert_eq!(parse_model_size("125M").unwrap(), 125_000_000);
    assert_eq!(parse_model_size("350M").unwrap(), 350_000_000);
    assert_eq!(parse_model_size("1000M").unwrap(), 1_000_000_000);
}

#[test]
fn test_parse_model_size_large() {
    assert_eq!(parse_model_size("180B").unwrap(), 180_000_000_000);
    assert_eq!(parse_model_size("405B").unwrap(), 405_000_000_000);
}

#[test]
fn test_parse_model_size_invalid_number() {
    let result = parse_model_size("abcB");
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Invalid number"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

// =========================================================================
// format_params tests
// =========================================================================

#[test]
fn test_format_params() {
    assert_eq!(format_params(7_000_000_000), "7.0B");
    assert_eq!(format_params(1_500_000_000), "1.5B");
    assert_eq!(format_params(500_000_000), "500.0M");
}

#[test]
fn test_format_params_small() {
    assert_eq!(format_params(100_000), "100000");
    assert_eq!(format_params(999_999), "999999");
}

#[test]
fn test_format_params_millions() {
    assert_eq!(format_params(1_000_000), "1.0M");
    assert_eq!(format_params(125_000_000), "125.0M");
    assert_eq!(format_params(999_999_999), "1000.0M");
}

#[test]
fn test_format_params_billions() {
    assert_eq!(format_params(1_000_000_000), "1.0B");
    assert_eq!(format_params(70_000_000_000), "70.0B");
    assert_eq!(format_params(405_000_000_000), "405.0B");
}

// =========================================================================
// estimate_params_from_file tests
// =========================================================================

#[test]
fn test_estimate_params_from_file() {
    let temp_dir = std::env::temp_dir().join("apr_tune_test");
    let _ = fs::create_dir_all(&temp_dir);
    let data = vec![0u8; 1_000_000];

    // GH-484: GGUF files use Q4 estimate (size * 2)
    let gguf_file = temp_dir.join("test_model.gguf");
    let _ = fs::write(&gguf_file, &data);
    let params = estimate_params_from_file(&gguf_file).unwrap();
    assert_eq!(params, 2_000_000, "GGUF: 1MB * 2 = 2M params (Q4 estimate)");

    // GH-484: Non-GGUF files use fp16 estimate (size / 2)
    let st_file = temp_dir.join("test_model.safetensors");
    let _ = fs::write(&st_file, &data);
    let params = estimate_params_from_file(&st_file).unwrap();
    assert_eq!(params, 500_000, "SafeTensors: 1MB / 2 = 500K params (fp16)");

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_estimate_params_from_file_not_found() {
    let result = estimate_params_from_file(Path::new("/nonexistent/model.bin"));
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Cannot read model file"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

// =========================================================================
// run() error cases tests
// =========================================================================

#[test]
fn test_run_no_model_or_size() {
    let result = run(
        None, // No model path
        TuneMethod::Auto,
        None,
        16.0,
        true,
        None, // No model size
        false,
        None,
        false,
    );

    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Either --model or model path required"));
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_run_with_model_size() {
    let result = run(
        None,
        TuneMethod::LoRA,
        Some(8),
        24.0,
        true,
        Some("7B"),
        false,
        None,
        false,
    );

    assert!(result.is_ok());
}

#[test]
fn test_run_with_model_size_json_output() {
    let result = run(
        None,
        TuneMethod::QLoRA,
        Some(16),
        16.0,
        true,
        Some("1.5B"),
        false,
        None,
        true, // JSON output
    );

    assert!(result.is_ok());
}

#[test]
fn test_run_plan_only() {
    let result = run(
        None,
        TuneMethod::Auto,
        None,
        8.0,
        true, // plan_only
        Some("3B"),
        false,
        None,
        false,
    );

    assert!(result.is_ok());
}

#[test]
fn test_run_with_rank() {
    let result = run(
        None,
        TuneMethod::LoRA,
        Some(4), // rank
        16.0,
        true,
        Some("7B"),
        false,
        None,
        false,
    );

    assert!(result.is_ok());
}

#[test]
fn test_run_with_model_file() {
    let temp_dir = std::env::temp_dir().join("apr_tune_run_test");
    let _ = fs::create_dir_all(&temp_dir);

    // Create a test model file (small for fast tests)
    let test_file = temp_dir.join("test_model.gguf");
    let data = vec![0u8; 100_000]; // 100KB
    let _ = fs::write(&test_file, &data);

    let result = run(
        Some(&test_file),
        TuneMethod::QLoRA,
        None,
        8.0,
        true,
        None,
        false,
        None,
        false,
    );

    assert!(result.is_ok());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_run_model_file_not_found() {
    let result = run(
        Some(Path::new("/nonexistent/model.gguf")),
        TuneMethod::Auto,
        None,
        16.0,
        true,
        None,
        false,
        None,
        false,
    );

    assert!(result.is_err());
}

#[test]
fn test_run_invalid_model_size() {
    let result = run(
        None,
        TuneMethod::Auto,
        None,
        16.0,
        true,
        Some("invalid"), // Invalid size format
        false,
        None,
        false,
    );

    assert!(result.is_err());
}

// =========================================================================
// classify tune tests (SPEC-TUNE-2026-001)
// =========================================================================

#[test]
fn test_classify_tune_json_output() {
    let result = run_classify_tune(
        None, 3,      // budget
        "tpe",  // strategy
        "asha", // scheduler
        true,   // scout
        None,   // no data (dry run)
        5,      // num_classes
        None,   // model_size
        None,   // from_scout
        20,     // max_epochs
        None,   // time_limit
        true,   // json output
    );
    assert!(result.is_ok(), "JSON classify tune should succeed");
}

#[test]
fn test_classify_tune_human_output() {
    let result = run_classify_tune(
        None, 5,        // budget
        "random", // strategy
        "none",   // scheduler
        false,    // full mode
        None,     // no data
        3,        // num_classes
        None,     // model_size
        None,     // from_scout
        10,       // max_epochs
        None,     // time_limit
        false,    // human output
    );
    assert!(result.is_ok(), "Human classify tune should succeed");
}

#[test]
fn test_classify_tune_invalid_strategy() {
    let result = run_classify_tune(
        None,
        5,
        "invalid_strategy",
        "asha",
        true,
        None,
        5,
        None,
        None,
        20,
        None,
        false,
    );
    assert!(result.is_err(), "Invalid strategy should fail");
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("Unknown strategy"),
                "Error should mention unknown strategy, got: {msg}"
            );
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_classify_tune_budget_zero() {
    let result = run_classify_tune(
        None, 0, // budget=0
        "tpe", "asha", true, None, 5, None, None, 20, None, false,
    );
    assert!(result.is_err(), "Budget=0 should fail");
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("FALSIFY-TUNE-001"),
                "Error should contain FALSIFY-TUNE-001, got: {msg}"
            );
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_classify_tune_missing_data() {
    let result = run_classify_tune(
        None,
        3,
        "tpe",
        "asha",
        true,
        Some(Path::new("/nonexistent/corpus.jsonl")),
        5,
        None,
        None,
        20,
        None,
        false,
    );
    assert!(result.is_err(), "Missing data file should fail");
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("FALSIFY-TUNE-003"),
                "Error should contain FALSIFY-TUNE-003, got: {msg}"
            );
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

// ── Additional falsification tests (SPEC-TUNE-2026-001 §7) ────

#[test]
fn test_classify_tune_grid_strategy_json() {
    let result = run_classify_tune(
        None, 5, "grid", "median", false, None, 3, None, None, 10, None, true, // JSON output
    );
    assert!(
        result.is_ok(),
        "Grid strategy with JSON output should succeed"
    );
}

#[test]
fn test_classify_tune_random_strategy() {
    let result = run_classify_tune(
        None, 3, "random", "none", true, None, 5, None, None, 1, None, false,
    );
    assert!(result.is_ok(), "Random strategy should succeed");
}

#[test]
fn test_classify_tune_invalid_scheduler() {
    let result = run_classify_tune(
        None,
        5,
        "tpe",
        "hyperband_v99", // invalid scheduler
        true,
        None,
        5,
        None,
        None,
        20,
        None,
        false,
    );
    assert!(result.is_err(), "Invalid scheduler should fail");
}

#[test]
fn test_classify_tune_num_classes_zero() {
    let result = run_classify_tune(
        None, 3, "tpe", "asha", true, None, 0, // num_classes=0
        None, None, 20, None, false,
    );
    assert!(result.is_err(), "num_classes=0 should fail");
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("FALSIFY-TUNE-004"),
                "Error should contain FALSIFY-TUNE-004, got: {msg}"
            );
        }
        other => panic!("Expected ValidationFailed, got {:?}", other),
    }
}

#[test]
fn test_classify_tune_scout_mode_caps_epochs() {
    // Scout mode should succeed with budget=1 (minimal)
    let result = run_classify_tune(
        None, 1, "tpe", "asha", true, // scout mode
        None, 5, None, None, 100, None, true, // JSON for easy verification
    );
    assert!(result.is_ok(), "Scout mode with budget=1 should succeed");
}

#[test]
fn test_classify_tune_large_budget_json() {
    let result = run_classify_tune(
        None, 100, "tpe", "asha", false, None, 10, None, None, 20, None, true, // JSON output
    );
    // Should succeed — budget=100 with no data just shows sample configs
    assert!(result.is_ok(), "Large budget with JSON should succeed");
}
