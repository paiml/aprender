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

    // Create a 1MB file
    let test_file = temp_dir.join("test_model.bin");
    let data = vec![0u8; 1_000_000];
    let _ = fs::write(&test_file, &data);

    let params = estimate_params_from_file(&test_file).unwrap();
    // 1MB file * 2 (Q4 estimate) = 2M params
    assert_eq!(params, 2_000_000);

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
