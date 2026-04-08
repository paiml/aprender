use super::*;
use std::io::Write;
use tempfile::NamedTempFile;

// ========================================================================
// CanaryData and TensorCanary Tests
// ========================================================================

#[test]
fn test_canary_data_serialize_deserialize() {
    let canary = CanaryData {
        model_name: "test-model.safetensors".to_string(),
        tensor_count: 1,
        tensors: BTreeMap::new(),
        created_at: "2024-01-01T00:00:00Z".to_string(),
    };
    let json = serde_json::to_string(&canary).expect("serialize");
    let parsed: CanaryData = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.model_name, "test-model.safetensors");
    assert_eq!(parsed.tensor_count, 1);
}

#[test]
fn test_tensor_canary_serialize_deserialize() {
    let tensor = TensorCanary {
        shape: vec![768, 768],
        count: 589824,
        mean: 0.0,
        std: 0.02,
        min: -0.1,
        max: 0.1,
    };
    let json = serde_json::to_string(&tensor).expect("serialize");
    let parsed: TensorCanary = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.shape, vec![768, 768]);
    assert_eq!(parsed.count, 589824);
}

#[test]
fn test_canary_data_with_tensors() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "encoder.weight".to_string(),
        TensorCanary {
            shape: vec![768, 768],
            count: 589824,
            mean: 0.0,
            std: 0.02,
            min: -0.1,
            max: 0.1,
        },
    );
    let canary = CanaryData {
        model_name: "test.safetensors".to_string(),
        tensor_count: 1,
        tensors,
        created_at: "2024-01-01T00:00:00Z".to_string(),
    };
    let json = serde_json::to_string_pretty(&canary).expect("serialize");
    assert!(json.contains("encoder.weight"));
    assert!(json.contains("768"));
}

#[test]
fn test_canary_data_clone() {
    let canary = CanaryData {
        model_name: "test.safetensors".to_string(),
        tensor_count: 0,
        tensors: BTreeMap::new(),
        created_at: "2024-01-01T00:00:00Z".to_string(),
    };
    let cloned = canary.clone();
    assert_eq!(cloned.model_name, canary.model_name);
}

#[test]
fn test_tensor_canary_clone() {
    let tensor = TensorCanary {
        shape: vec![768],
        count: 768,
        mean: 0.5,
        std: 0.1,
        min: 0.0,
        max: 1.0,
    };
    let cloned = tensor.clone();
    assert_eq!(cloned.mean, tensor.mean);
}

// ========================================================================
// CanaryCheckResult Tests
// ========================================================================

#[test]
fn test_canary_check_result_passed() {
    let result = CanaryCheckResult {
        tensor_name: "weight".to_string(),
        passed: true,
        mean_drift: 0.01,
        std_drift: 0.02,
        shape_match: true,
        message: None,
    };
    assert!(result.passed);
    assert!(result.message.is_none());
}

#[test]
fn test_canary_check_result_failed() {
    let result = CanaryCheckResult {
        tensor_name: "weight".to_string(),
        passed: false,
        mean_drift: 0.15,
        std_drift: 0.02,
        shape_match: true,
        message: Some("Mean drift exceeded".to_string()),
    };
    assert!(!result.passed);
    assert!(result.message.is_some());
}

#[test]
fn test_canary_check_result_debug() {
    let result = CanaryCheckResult {
        tensor_name: "test".to_string(),
        passed: true,
        mean_drift: 0.0,
        std_drift: 0.0,
        shape_match: true,
        message: None,
    };
    let debug = format!("{result:?}");
    assert!(debug.contains("CanaryCheckResult"));
}

// ========================================================================
// compute_relative_drift Tests
// ========================================================================

#[test]
fn test_compute_relative_drift_normal() {
    // 10% drift: actual=1.1, expected=1.0
    let drift = compute_relative_drift(1.1, 1.0);
    assert!((drift - 0.1).abs() < 0.001);
}

#[test]
fn test_compute_relative_drift_negative() {
    // -10% drift: actual=0.9, expected=1.0
    let drift = compute_relative_drift(0.9, 1.0);
    assert!((drift - 0.1).abs() < 0.001);
}

#[test]
fn test_compute_relative_drift_zero_expected() {
    // When expected is near zero, use absolute difference
    let drift = compute_relative_drift(0.001, 0.0);
    assert!((drift - 0.001).abs() < 0.0001);
}

#[test]
fn test_compute_relative_drift_same_value() {
    let drift = compute_relative_drift(1.0, 1.0);
    assert_eq!(drift, 0.0);
}

#[test]
fn test_compute_relative_drift_large_values() {
    // 50% drift: actual=150, expected=100
    let drift = compute_relative_drift(150.0, 100.0);
    assert!((drift - 0.5).abs() < 0.001);
}

// ========================================================================
// missing_tensor_result Tests
// ========================================================================

#[test]
fn test_missing_tensor_result() {
    let result = missing_tensor_result("missing_weight");
    assert_eq!(result.tensor_name, "missing_weight");
    assert!(!result.passed);
    assert_eq!(result.mean_drift, f32::MAX);
    assert_eq!(result.std_drift, f32::MAX);
    assert!(!result.shape_match);
    assert!(result.message.is_some());
    assert!(result.message.expect("message").contains("not found"));
}

// ========================================================================
// build_failure_message Tests
// ========================================================================

#[test]
fn test_build_failure_message_passed() {
    // Create a minimal TensorCanary for reference
    let _expected = TensorCanary {
        shape: vec![768],
        count: 768,
        mean: 0.0,
        std: 0.02,
        min: -0.1,
        max: 0.1,
    };
    // Since we can't easily create TensorMetadata, we use a helper
    // to test the passed case where message should be None
    let msg = build_failure_message_test_helper(true, true, 0.01, 0.01);
    assert!(msg.is_none());
}

// Helper for testing build_failure_message without needing TensorMetadata
fn build_failure_message_test_helper(
    passed: bool,
    shape_match: bool,
    mean_drift: f32,
    std_drift: f32,
) -> Option<String> {
    if passed {
        return None;
    }
    Some(if !shape_match {
        "Shape mismatch".to_string()
    } else if mean_drift > MEAN_THRESHOLD {
        format!("Mean drift {:.1}% exceeds threshold", mean_drift * 100.0)
    } else {
        format!("Std drift {:.1}% exceeds threshold", std_drift * 100.0)
    })
}

#[test]
fn test_build_failure_message_shape_mismatch() {
    let msg = build_failure_message_test_helper(false, false, 0.01, 0.01);
    assert!(msg.is_some());
    assert!(msg.expect("value").contains("Shape mismatch"));
}

#[test]
fn test_build_failure_message_mean_drift() {
    let msg = build_failure_message_test_helper(false, true, 0.15, 0.01);
    assert!(msg.is_some());
    assert!(msg.expect("value").contains("Mean drift"));
}

#[test]
fn test_build_failure_message_std_drift() {
    let msg = build_failure_message_test_helper(false, true, 0.05, 0.25);
    assert!(msg.is_some());
    assert!(msg.expect("value").contains("Std drift"));
}

// ========================================================================
// CanaryCommands Tests
// ========================================================================

#[test]
fn test_canary_commands_create() {
    let cmd = CanaryCommands::Create {
        file: PathBuf::from("model.safetensors"),
        input: PathBuf::from("input.wav"),
        output: PathBuf::from("canary.json"),
    };
    match cmd {
        CanaryCommands::Create {
            file,
            input,
            output,
        } => {
            assert_eq!(file.to_string_lossy(), "model.safetensors");
            assert_eq!(input.to_string_lossy(), "input.wav");
            assert_eq!(output.to_string_lossy(), "canary.json");
        }
        _ => panic!("Wrong command variant"),
    }
}

#[test]
fn test_canary_commands_check() {
    let cmd = CanaryCommands::Check {
        file: PathBuf::from("model.safetensors"),
        canary: PathBuf::from("canary.json"),
    };
    match cmd {
        CanaryCommands::Check { file, canary } => {
            assert_eq!(file.to_string_lossy(), "model.safetensors");
            assert_eq!(canary.to_string_lossy(), "canary.json");
        }
        _ => panic!("Wrong command variant"),
    }
}

#[test]
fn test_canary_commands_clone() {
    let cmd = CanaryCommands::Create {
        file: PathBuf::from("model.safetensors"),
        input: PathBuf::from("input.wav"),
        output: PathBuf::from("canary.json"),
    };
    let cloned = cmd.clone();
    match cloned {
        CanaryCommands::Create { file, .. } => {
            assert_eq!(file.to_string_lossy(), "model.safetensors");
        }
        _ => panic!("Wrong command variant"),
    }
}

#[test]
fn test_canary_commands_debug() {
    let cmd = CanaryCommands::Check {
        file: PathBuf::from("model.safetensors"),
        canary: PathBuf::from("canary.json"),
    };
    let debug = format!("{cmd:?}");
    assert!(debug.contains("Check"));
}

// ========================================================================
// run Command Tests
// ========================================================================

#[test]
fn test_run_create_model_not_found() {
    let output = NamedTempFile::with_suffix(".json").expect("create output");
    let input = NamedTempFile::with_suffix(".wav").expect("create input");
    let cmd = CanaryCommands::Create {
        file: PathBuf::from("/nonexistent/model.safetensors"),
        input: input.path().to_path_buf(),
        output: output.path().to_path_buf(),
    };
    let result = run(cmd);
    assert!(result.is_err());
}

#[test]
fn test_run_create_invalid_model() {
    let mut model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    model
        .write_all(b"not a valid safetensors file")
        .expect("write");
    let output = NamedTempFile::with_suffix(".json").expect("create output");
    let input = NamedTempFile::with_suffix(".wav").expect("create input");

    let cmd = CanaryCommands::Create {
        file: model.path().to_path_buf(),
        input: input.path().to_path_buf(),
        output: output.path().to_path_buf(),
    };
    let result = run(cmd);
    assert!(result.is_err());
}

#[test]
fn test_run_check_model_not_found() {
    let mut canary = NamedTempFile::with_suffix(".json").expect("create canary");
    canary
        .write_all(br#"{"model_name": "test", "tensor_count": 0, "tensors": {}, "created_at": ""}"#)
        .expect("write");

    let cmd = CanaryCommands::Check {
        file: PathBuf::from("/nonexistent/model.safetensors"),
        canary: canary.path().to_path_buf(),
    };
    let result = run(cmd);
    assert!(result.is_err());
}

#[test]
fn test_run_check_canary_not_found() {
    let mut model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    model.write_all(b"fake model").expect("write");

    let cmd = CanaryCommands::Check {
        file: model.path().to_path_buf(),
        canary: PathBuf::from("/nonexistent/canary.json"),
    };
    let result = run(cmd);
    assert!(result.is_err());
}

#[test]
fn test_run_check_invalid_canary() {
    let mut model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    model.write_all(b"fake model").expect("write");
    let mut canary = NamedTempFile::with_suffix(".json").expect("create canary");
    canary.write_all(b"not valid json").expect("write");

    let cmd = CanaryCommands::Check {
        file: model.path().to_path_buf(),
        canary: canary.path().to_path_buf(),
    };
    let result = run(cmd);
    assert!(result.is_err());
}

// ========================================================================
// validate_paths_exist Tests
// ========================================================================

#[test]
fn test_validate_paths_exist_model_missing() {
    let canary = NamedTempFile::with_suffix(".json").expect("create canary");
    let result = validate_paths_exist(Path::new("/nonexistent/model.safetensors"), canary.path());
    assert!(result.is_err());
}

#[test]
fn test_validate_paths_exist_canary_missing() {
    let model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    let result = validate_paths_exist(model.path(), Path::new("/nonexistent/canary.json"));
    assert!(result.is_err());
}

#[test]
fn test_validate_paths_exist_both_exist() {
    let model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    let canary = NamedTempFile::with_suffix(".json").expect("create canary");
    let result = validate_paths_exist(model.path(), canary.path());
    assert!(result.is_ok());
}

// ========================================================================
// load_canary_data Tests
// ========================================================================

#[test]
fn test_load_canary_data_valid() {
    let mut canary = NamedTempFile::with_suffix(".json").expect("create canary");
    canary.write_all(br#"{"model_name": "test.safetensors", "tensor_count": 0, "tensors": {}, "created_at": "2024-01-01"}"#).expect("write");

    let result = load_canary_data(canary.path());
    assert!(result.is_ok());
    assert_eq!(result.expect("value").model_name, "test.safetensors");
}

#[test]
fn test_load_canary_data_invalid_json() {
    let mut canary = NamedTempFile::with_suffix(".json").expect("create canary");
    canary.write_all(b"not valid json").expect("write");

    let result = load_canary_data(canary.path());
    assert!(result.is_err());
}

#[test]
fn test_load_canary_data_file_not_found() {
    let result = load_canary_data(Path::new("/nonexistent/canary.json"));
    assert!(result.is_err());
}

include!("canary_tests_mean_threshold_std.rs");

// ========================================================================
// build_failure_message_generic (actual function) Tests
// ========================================================================

#[test]
fn test_build_failure_message_generic_passed_returns_none() {
    let expected = TensorCanary {
        shape: vec![768],
        count: 768,
        mean: 0.5,
        std: 0.1,
        min: 0.0,
        max: 1.0,
    };
    let result = build_failure_message_generic(true, true, 0.01, 0.01, &expected, &[768]);
    assert!(result.is_none());
}

#[test]
fn test_build_failure_message_generic_shape_mismatch() {
    let expected = TensorCanary {
        shape: vec![768, 768],
        count: 589824,
        mean: 0.0,
        std: 0.02,
        min: -0.1,
        max: 0.1,
    };
    let result = build_failure_message_generic(false, false, 0.01, 0.01, &expected, &[512, 512]);
    assert!(result.is_some());
    let msg = result.expect("should have message");
    assert!(msg.contains("Shape mismatch"));
    assert!(msg.contains("[768, 768]"));
    assert!(msg.contains("[512, 512]"));
}

#[test]
fn test_build_failure_message_generic_mean_drift_exceeded() {
    let expected = TensorCanary {
        shape: vec![768],
        count: 768,
        mean: 0.5,
        std: 0.1,
        min: 0.0,
        max: 1.0,
    };
    let result = build_failure_message_generic(false, true, 0.15, 0.01, &expected, &[768]);
    assert!(result.is_some());
    let msg = result.expect("should have message");
    assert!(msg.contains("Mean drift"));
    assert!(msg.contains("15.0%"));
    assert!(msg.contains("10.0%"));
}

#[test]
fn test_build_failure_message_generic_std_drift_exceeded() {
    let expected = TensorCanary {
        shape: vec![768],
        count: 768,
        mean: 0.5,
        std: 0.1,
        min: 0.0,
        max: 1.0,
    };
    // mean_drift within threshold, std_drift exceeds
    let result = build_failure_message_generic(false, true, 0.05, 0.25, &expected, &[768]);
    assert!(result.is_some());
    let msg = result.expect("should have message");
    assert!(msg.contains("Std drift"));
    assert!(msg.contains("25.0%"));
    assert!(msg.contains("20.0%"));
}

// ========================================================================
// compare_single_tensor_generic Tests
// ========================================================================

#[test]
fn test_compare_single_tensor_generic_pass() {
    let expected = TensorCanary {
        shape: vec![4],
        count: 4,
        mean: 2.5,
        std: 1.118_033_9,
        min: 1.0,
        max: 4.0,
    };
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let shape = [4usize];
    let result = compare_single_tensor_generic("test.weight", &expected, &data, &shape);
    assert!(result.passed, "should pass with matching data");
    assert!(result.shape_match);
    assert!(result.message.is_none());
    assert_eq!(result.tensor_name, "test.weight");
}

#[test]
fn test_compare_single_tensor_generic_shape_mismatch() {
    let expected = TensorCanary {
        shape: vec![2, 2],
        count: 4,
        mean: 2.5,
        std: 1.118_033_9,
        min: 1.0,
        max: 4.0,
    };
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let shape = [4usize]; // different shape
    let result = compare_single_tensor_generic("test.weight", &expected, &data, &shape);
    assert!(!result.passed, "should fail on shape mismatch");
    assert!(!result.shape_match);
    assert!(result.message.is_some());
}

#[test]
fn test_compare_single_tensor_generic_mean_drift() {
    // expected mean ~2.5, actual data has much higher mean
    let expected = TensorCanary {
        shape: vec![4],
        count: 4,
        mean: 2.5,
        std: 1.118_033_9,
        min: 1.0,
        max: 4.0,
    };
    let data = [10.0f32, 20.0, 30.0, 40.0]; // mean = 25.0, huge drift from 2.5
    let shape = [4usize];
    let result = compare_single_tensor_generic("test.weight", &expected, &data, &shape);
    assert!(!result.passed, "should fail on mean drift");
    assert!(result.shape_match);
    assert!(result.mean_drift > MEAN_THRESHOLD);
}

#[test]
fn test_compare_single_tensor_generic_std_drift() {
    // Same mean but very different std
    let expected = TensorCanary {
        shape: vec![4],
        count: 4,
        mean: 0.0,
        std: 0.1,
        min: -0.1,
        max: 0.1,
    };
    // Data with same-ish mean (0.0) but much higher std
    let data = [-100.0f32, 100.0, -100.0, 100.0]; // mean ~0, std ~100
    let shape = [4usize];
    let result = compare_single_tensor_generic("test.weight", &expected, &data, &shape);
    assert!(!result.passed, "should fail on std drift");
    assert!(result.std_drift > STD_THRESHOLD);
}

// ========================================================================
// compare_all_tensors_generic Tests
// ========================================================================

#[test]
fn test_compare_all_tensors_generic_all_match() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "weight".to_string(),
        TensorCanary {
            shape: vec![4],
            count: 4,
            mean: 2.5,
            std: 1.118_033_9,
            min: 1.0,
            max: 4.0,
        },
    );
    let canary = CanaryData {
        model_name: "test".to_string(),
        tensor_count: 1,
        tensors,
        created_at: "2024-01-01".to_string(),
    };

    let mut tensor_data = BTreeMap::new();
    tensor_data.insert("weight".to_string(), (vec![1.0f32, 2.0, 3.0, 4.0], vec![4]));

    let results = compare_all_tensors_generic(&canary, &tensor_data);
    assert_eq!(results.len(), 1);
    assert!(results[0].passed);
}

#[test]
fn test_compare_all_tensors_generic_missing_tensor() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "weight".to_string(),
        TensorCanary {
            shape: vec![4],
            count: 4,
            mean: 2.5,
            std: 1.118_033_9,
            min: 1.0,
            max: 4.0,
        },
    );
    let canary = CanaryData {
        model_name: "test".to_string(),
        tensor_count: 1,
        tensors,
        created_at: "2024-01-01".to_string(),
    };

    // Empty tensor data — tensor is "missing"
    let tensor_data: TensorDataMap = BTreeMap::new();

    let results = compare_all_tensors_generic(&canary, &tensor_data);
    assert_eq!(results.len(), 1);
    assert!(!results[0].passed);
    assert_eq!(results[0].mean_drift, f32::MAX);
    assert!(results[0]
        .message
        .as_ref()
        .expect("msg")
        .contains("not found"));
}

#[test]
fn test_compare_all_tensors_generic_multiple_tensors_mixed() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "layer1.weight".to_string(),
        TensorCanary {
            shape: vec![4],
            count: 4,
            mean: 2.5,
            std: 1.118_033_9,
            min: 1.0,
            max: 4.0,
        },
    );
    tensors.insert(
        "layer2.weight".to_string(),
        TensorCanary {
            shape: vec![4],
            count: 4,
            mean: 2.5,
            std: 1.118_033_9,
            min: 1.0,
            max: 4.0,
        },
    );
    let canary = CanaryData {
        model_name: "test".to_string(),
        tensor_count: 2,
        tensors,
        created_at: "2024-01-01".to_string(),
    };

    let mut tensor_data = BTreeMap::new();
    // layer1 matches
    tensor_data.insert(
        "layer1.weight".to_string(),
        (vec![1.0f32, 2.0, 3.0, 4.0], vec![4]),
    );
    // layer2 missing
    let results = compare_all_tensors_generic(&canary, &tensor_data);
    assert_eq!(results.len(), 2);

    let layer1 = results.iter().find(|r| r.tensor_name == "layer1.weight");
    let layer2 = results.iter().find(|r| r.tensor_name == "layer2.weight");
    assert!(layer1.expect("layer1").passed);
    assert!(!layer2.expect("layer2").passed);
}

#[test]
fn test_compare_all_tensors_generic_empty_canary() {
    let canary = CanaryData {
        model_name: "test".to_string(),
        tensor_count: 0,
        tensors: BTreeMap::new(),
        created_at: "2024-01-01".to_string(),
    };
    let tensor_data: TensorDataMap = BTreeMap::new();
    let results = compare_all_tensors_generic(&canary, &tensor_data);
    assert!(results.is_empty());
}

// ========================================================================
// display_canary_results Tests
// ========================================================================

#[test]
fn test_display_canary_results_all_passed() {
    let results = vec![
        CanaryCheckResult {
            tensor_name: "weight".to_string(),
            passed: true,
            mean_drift: 0.01,
            std_drift: 0.02,
            shape_match: true,
            message: None,
        },
        CanaryCheckResult {
            tensor_name: "bias".to_string(),
            passed: true,
            mean_drift: 0.005,
            std_drift: 0.003,
            shape_match: true,
            message: None,
        },
    ];
    let result = display_canary_results(&results, 2);
    assert!(result.is_ok(), "all passed should return Ok");
}

#[test]
fn test_display_canary_results_some_failed() {
    let results = vec![
        CanaryCheckResult {
            tensor_name: "weight".to_string(),
            passed: true,
            mean_drift: 0.01,
            std_drift: 0.02,
            shape_match: true,
            message: None,
        },
        CanaryCheckResult {
            tensor_name: "bias".to_string(),
            passed: false,
            mean_drift: 0.5,
            std_drift: 0.02,
            shape_match: true,
            message: Some("Mean drift exceeded".to_string()),
        },
    ];
    let result = display_canary_results(&results, 2);
    assert!(result.is_err(), "some failures should return Err");
    let err_msg = format!("{}", result.expect_err("error"));
    assert!(err_msg.contains("1 of 2"));
}

#[test]
fn test_display_canary_results_all_failed() {
    let results = vec![
        CanaryCheckResult {
            tensor_name: "weight".to_string(),
            passed: false,
            mean_drift: 0.5,
            std_drift: 0.3,
            shape_match: false,
            message: Some("Shape mismatch".to_string()),
        },
        CanaryCheckResult {
            tensor_name: "bias".to_string(),
            passed: false,
            mean_drift: 0.8,
            std_drift: 0.9,
            shape_match: true,
            message: Some("Mean drift exceeded".to_string()),
        },
    ];
    let result = display_canary_results(&results, 2);
    assert!(result.is_err());
    let err_msg = format!("{}", result.expect_err("error"));
    assert!(err_msg.contains("2 of 2"));
}

#[test]
fn test_display_canary_results_empty() {
    let results: Vec<CanaryCheckResult> = vec![];
    let result = display_canary_results(&results, 0);
    assert!(result.is_ok(), "empty results with 0 count should pass");
}

#[test]
fn test_display_canary_results_failed_no_message() {
    // A failed result with message = None (edge case)
    let results = vec![CanaryCheckResult {
        tensor_name: "orphan".to_string(),
        passed: false,
        mean_drift: 0.99,
        std_drift: 0.99,
        shape_match: false,
        message: None,
    }];
    let result = display_canary_results(&results, 1);
    assert!(result.is_err());
}

// ========================================================================
// print_canary_check_header Tests
// ========================================================================

#[test]
fn test_print_canary_check_header_runs() {
    // Verifies no panic on normal paths
    print_canary_check_header(Path::new("model.safetensors"), Path::new("canary.json"));
}

#[test]
fn test_print_canary_check_header_unicode_paths() {
    print_canary_check_header(
        Path::new("/tmp/модель.safetensors"),
        Path::new("/tmp/канарейка.json"),
    );
}

// ========================================================================
// compute_relative_drift edge cases
// ========================================================================

#[test]
fn test_compute_relative_drift_near_zero_boundary() {
    // Expected exactly at the 1e-6 boundary
    let drift = compute_relative_drift(0.001, 1e-6);
    // expected > 1e-6, so relative drift = |0.001 - 1e-6| / 1e-6
    assert!(drift > 0.0);
}

#[test]
fn test_compute_relative_drift_both_zero() {
    let drift = compute_relative_drift(0.0, 0.0);
    assert_eq!(drift, 0.0);
}

#[test]
fn test_compute_relative_drift_negative_expected() {
    // actual=-0.9, expected=-1.0 => relative drift = |(-0.9 - -1.0) / -1.0| = |0.1/-1.0| = 0.1
    let drift = compute_relative_drift(-0.9, -1.0);
    assert!((drift - 0.1).abs() < 0.001);
}

#[test]
fn test_compute_relative_drift_below_zero_threshold() {
    // expected = 0.5e-6 (below 1e-6 threshold), so absolute diff is used
    let drift = compute_relative_drift(0.002, 0.5e-6);
    assert!((drift - 0.002).abs() < 0.001);
}

// ========================================================================
// missing_tensor_result edge cases
// ========================================================================

#[test]
fn test_missing_tensor_result_empty_name() {
    let result = missing_tensor_result("");
    assert_eq!(result.tensor_name, "");
    assert!(!result.passed);
}

#[test]
fn test_missing_tensor_result_long_name() {
    let name = "model.transformer.blocks.42.attention.query_key_value.weight";
    let result = missing_tensor_result(name);
    assert_eq!(result.tensor_name, name);
    assert_eq!(result.mean_drift, f32::MAX);
    assert_eq!(result.std_drift, f32::MAX);
}

// ========================================================================
// load_tensor_data error paths
// ========================================================================

#[test]
fn test_load_tensor_data_unknown_format() {
    let mut file = NamedTempFile::with_suffix(".xyz").expect("create temp file");
    file.write_all(b"random bytes that are not any known format")
        .expect("write");
    let result = load_tensor_data(file.path());
    assert!(result.is_err(), "unknown format should error");
}

#[test]
fn test_load_tensor_data_empty_file() {
    let file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    let result = load_tensor_data(file.path());
    assert!(result.is_err(), "empty file should error");
}

#[test]
fn test_load_tensor_data_gguf_corrupt() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    // Write GGUF magic but truncated
    file.write_all(b"GGUF\x03\x00\x00\x00").expect("write");
    let result = load_tensor_data(file.path());
    assert!(result.is_err(), "corrupt GGUF should error");
}

// ========================================================================
// load_tensor_data_apr Tests
// ========================================================================

#[test]
fn test_load_tensor_data_apr_format() {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};

    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);

    let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    writer.add_tensor("test.weight", TensorDType::F32, vec![2, 2], data);

    let mut apr_bytes = Vec::new();
    writer.write_to(&mut apr_bytes).expect("write APR");

    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(&apr_bytes).expect("write");

    let result = load_tensor_data(file.path());
    assert!(result.is_ok(), "APR format should load: {result:?}");

    let tensor_map = result.expect("value");
    assert!(tensor_map.contains_key("test.weight"));
    let (values, shape) = &tensor_map["test.weight"];
    assert_eq!(shape, &[2, 2]);
    assert_eq!(values.len(), 4);
    assert!((values[0] - 1.0).abs() < f32::EPSILON);
    assert!((values[3] - 4.0).abs() < f32::EPSILON);
}

#[test]
fn test_load_tensor_data_apr_multiple_tensors() {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};

    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);

    let data1: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let data2: Vec<u8> = [5.0f32, 6.0].iter().flat_map(|f| f.to_le_bytes()).collect();

    writer.add_tensor("weight", TensorDType::F32, vec![2, 2], data1);
    writer.add_tensor("bias", TensorDType::F32, vec![2], data2);

    let mut apr_bytes = Vec::new();
    writer.write_to(&mut apr_bytes).expect("write APR");

    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(&apr_bytes).expect("write");

    let result = load_tensor_data(file.path());
    assert!(result.is_ok());
    let tensor_map = result.expect("value");
    assert_eq!(tensor_map.len(), 2);
    assert!(tensor_map.contains_key("weight"));
    assert!(tensor_map.contains_key("bias"));
}

#[test]
fn test_load_tensor_data_apr_invalid() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    // APR magic + garbage
    file.write_all(b"APR\0garbage").expect("write");
    let result = load_tensor_data(file.path());
    assert!(result.is_err(), "invalid APR should error");
}

// ========================================================================
// validate_paths_exist edge cases
// ========================================================================

#[test]
fn test_validate_paths_exist_error_is_file_not_found() {
    let canary = NamedTempFile::with_suffix(".json").expect("create canary");
    let result = validate_paths_exist(Path::new("/no/such/model.safetensors"), canary.path());
    let err = result.expect_err("should error");
    let msg = format!("{err}");
    assert!(msg.contains("File not found"));
}

#[test]
fn test_validate_paths_exist_canary_error_message() {
    let model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    let result = validate_paths_exist(model.path(), Path::new("/no/such/canary.json"));
    let err = result.expect_err("should error");
    let msg = format!("{err}");
    assert!(msg.contains("File not found"));
}

// ========================================================================
// load_canary_data edge cases
// ========================================================================

#[test]
fn test_load_canary_data_missing_fields() {
    let mut file = NamedTempFile::with_suffix(".json").expect("create file");
    // Missing required fields
    file.write_all(br#"{"model_name": "test"}"#).expect("write");
    let result = load_canary_data(file.path());
    assert!(result.is_err(), "missing fields should error");
}

#[test]
fn test_load_canary_data_with_tensors() {
    let mut file = NamedTempFile::with_suffix(".json").expect("create file");
    let json = serde_json::json!({
        "model_name": "test.apr",
        "tensor_count": 1,
        "created_at": "2024-01-01T00:00:00Z",
        "tensors": {
            "encoder.weight": {
                "shape": [768, 768],
                "count": 589824,
                "mean": 0.001,
                "std": 0.02,
                "min": -0.1,
                "max": 0.1
            }
        }
    });
    file.write_all(json.to_string().as_bytes()).expect("write");

    let result = load_canary_data(file.path());
    assert!(result.is_ok());
    let data = result.expect("canary data");
    assert_eq!(data.tensor_count, 1);
    assert!(data.tensors.contains_key("encoder.weight"));
    let t = &data.tensors["encoder.weight"];
    assert_eq!(t.shape, vec![768, 768]);
    assert_eq!(t.count, 589824);
}

#[test]
fn test_load_canary_data_empty_json_object() {
    let mut file = NamedTempFile::with_suffix(".json").expect("create file");
    file.write_all(b"{}").expect("write");
    let result = load_canary_data(file.path());
    assert!(result.is_err(), "empty object missing required fields");
}

// ========================================================================
// CanaryData round-trip Tests
// ========================================================================

#[test]
fn test_canary_data_round_trip_with_tensors() {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "attn.q.weight".to_string(),
        TensorCanary {
            shape: vec![768, 768],
            count: 589824,
            mean: 0.001,
            std: 0.02,
            min: -0.15,
            max: 0.15,
        },
    );
    tensors.insert(
        "attn.k.weight".to_string(),
        TensorCanary {
            shape: vec![768, 768],
            count: 589824,
            mean: -0.002,
            std: 0.019,
            min: -0.12,
            max: 0.13,
        },
    );
    let canary = CanaryData {
        model_name: "test-model.apr".to_string(),
        tensor_count: 2,
        tensors,
        created_at: "2024-06-15T10:30:00Z".to_string(),
    };

    let json = serde_json::to_string_pretty(&canary).expect("serialize");
    let parsed: CanaryData = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(parsed.model_name, "test-model.apr");
    assert_eq!(parsed.tensor_count, 2);
    assert_eq!(parsed.tensors.len(), 2);
    assert_eq!(parsed.created_at, "2024-06-15T10:30:00Z");

    let q = &parsed.tensors["attn.q.weight"];
    assert_eq!(q.shape, vec![768, 768]);
    assert!((q.mean - 0.001).abs() < f32::EPSILON);

    let k = &parsed.tensors["attn.k.weight"];
    assert!((k.mean - (-0.002)).abs() < f32::EPSILON);
}

#[test]
fn test_canary_data_debug_format() {
    let canary = CanaryData {
        model_name: "dbg-test".to_string(),
        tensor_count: 0,
        tensors: BTreeMap::new(),
        created_at: "2024-01-01".to_string(),
    };
    let debug = format!("{canary:?}");
    assert!(debug.contains("CanaryData"));
    assert!(debug.contains("dbg-test"));
}

#[test]
fn test_tensor_canary_debug_format() {
    let t = TensorCanary {
        shape: vec![3],
        count: 3,
        mean: 1.0,
        std: 0.5,
        min: 0.0,
        max: 2.0,
    };
    let debug = format!("{t:?}");
    assert!(debug.contains("TensorCanary"));
    assert!(debug.contains("1.0"));
}

// ========================================================================
// run() dispatch Tests
// ========================================================================

#[test]
fn test_run_create_input_not_found() {
    let mut model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    model.write_all(b"fake model data").expect("write");
    let output = NamedTempFile::with_suffix(".json").expect("create output");

    let cmd = CanaryCommands::Create {
        file: model.path().to_path_buf(),
        input: PathBuf::from("/nonexistent/input.wav"),
        output: output.path().to_path_buf(),
    };
    let result = run(cmd);
    assert!(result.is_err(), "missing input file should error");
}

#[test]
fn test_run_create_empty_input_path() {
    // Empty input path should skip input validation
    let mut model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    model.write_all(b"fake model data").expect("write");
    let output = NamedTempFile::with_suffix(".json").expect("create output");

    let cmd = CanaryCommands::Create {
        file: model.path().to_path_buf(),
        input: PathBuf::from(""),
        output: output.path().to_path_buf(),
    };
    // Will fail on format detection, not input validation
    let result = run(cmd);
    assert!(result.is_err());
    let err_msg = format!("{}", result.expect_err("error"));
    // Should NOT be FileNotFound for input — should be a format/model error
    assert!(
        !err_msg.contains("input"),
        "error should not be about input file"
    );
}

// ========================================================================
// create_canary with valid model (end-to-end via GGUF)
// ========================================================================

#[test]
fn test_create_canary_success_gguf() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Build valid GGUF
    let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: floats,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];
    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let mut model_file = NamedTempFile::with_suffix(".gguf").expect("create model");
    model_file.write_all(&gguf_bytes).expect("write");

    let input_file = NamedTempFile::with_suffix(".wav").expect("create input");
    let output_file = NamedTempFile::with_suffix(".json").expect("create output");

    let result = create_canary(model_file.path(), input_file.path(), output_file.path());
    assert!(result.is_ok(), "create_canary should succeed: {result:?}");

    // Verify the output JSON was written and is valid
    let json_content = fs::read_to_string(output_file.path()).expect("read output");
    let canary: CanaryData = serde_json::from_str(&json_content).expect("parse output");
    assert_eq!(canary.tensor_count, 1);
    assert!(canary.tensors.contains_key("model.weight"));
    let t = &canary.tensors["model.weight"];
    assert_eq!(t.shape, vec![2, 2]);
    assert_eq!(t.count, 4);
}

#[test]
fn test_create_canary_success_safetensors() {
    // Build valid SafeTensors
    let floats: Vec<u8> = [0.5f32, -0.5, 1.0, -1.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let header_json = serde_json::json!({
        "layer.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).expect("serialize header");
    let header_len = header_bytes.len() as u64;

    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&header_len.to_le_bytes());
    st_bytes.extend_from_slice(&header_bytes);
    st_bytes.extend_from_slice(&floats);

    let mut model_file = NamedTempFile::with_suffix(".safetensors").expect("create model");
    model_file.write_all(&st_bytes).expect("write");

    let input_file = NamedTempFile::with_suffix(".wav").expect("create input");
    let output_file = NamedTempFile::with_suffix(".json").expect("create output");

    let result = create_canary(model_file.path(), input_file.path(), output_file.path());
    assert!(result.is_ok(), "create_canary should succeed: {result:?}");

    let json_content = fs::read_to_string(output_file.path()).expect("read output");
    let canary: CanaryData = serde_json::from_str(&json_content).expect("parse output");
    assert_eq!(canary.tensor_count, 1);
    assert!(canary.tensors.contains_key("layer.weight"));
}

// ========================================================================
// check_canary end-to-end Tests
// ========================================================================

#[test]
fn test_check_canary_pass_gguf() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Build GGUF model
    let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensor = GgufTensor {
        name: "weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: floats,
    };
    let metadata_kv = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];
    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata_kv).expect("export GGUF");

    let mut model_file = NamedTempFile::with_suffix(".gguf").expect("create model");
    model_file.write_all(&gguf_bytes).expect("write");

    // Compute expected stats from the same data
    use aprender::format::TensorStats;
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let stats = TensorStats::compute("weight", &data);

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "weight".to_string(),
        TensorCanary {
            shape: vec![2, 2],
            count: 4,
            mean: stats.mean,
            std: stats.std,
            min: stats.min,
            max: stats.max,
        },
    );
    let canary = CanaryData {
        model_name: "test".to_string(),
        tensor_count: 1,
        tensors,
        created_at: "2024-01-01".to_string(),
    };

    let mut canary_file = NamedTempFile::with_suffix(".json").expect("create canary");
    let json = serde_json::to_string_pretty(&canary).expect("serialize");
    canary_file.write_all(json.as_bytes()).expect("write");

    let result = check_canary(model_file.path(), canary_file.path());
    assert!(result.is_ok(), "canary check should pass: {result:?}");
}

#[test]
fn test_check_canary_fail_drift() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Build GGUF model
    let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensor = GgufTensor {
        name: "weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: floats,
    };
    let metadata_kv = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];
    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata_kv).expect("export GGUF");

    let mut model_file = NamedTempFile::with_suffix(".gguf").expect("create model");
    model_file.write_all(&gguf_bytes).expect("write");

    // Canary with very different stats — should fail
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "weight".to_string(),
        TensorCanary {
            shape: vec![2, 2],
            count: 4,
            mean: 100.0, // actual mean ~2.5, huge drift
            std: 0.001,
            min: 99.0,
            max: 101.0,
        },
    );
    let canary = CanaryData {
        model_name: "test".to_string(),
        tensor_count: 1,
        tensors,
        created_at: "2024-01-01".to_string(),
    };

    let mut canary_file = NamedTempFile::with_suffix(".json").expect("create canary");
    let json = serde_json::to_string_pretty(&canary).expect("serialize");
    canary_file.write_all(json.as_bytes()).expect("write");

    let result = check_canary(model_file.path(), canary_file.path());
    assert!(result.is_err(), "canary check should fail on drift");
}

#[test]
fn test_check_canary_model_not_found() {
    let mut canary_file = NamedTempFile::with_suffix(".json").expect("create canary");
    canary_file
        .write_all(br#"{"model_name":"t","tensor_count":0,"tensors":{},"created_at":"x"}"#)
        .expect("write");

    let result = check_canary(Path::new("/no/model.safetensors"), canary_file.path());
    assert!(result.is_err());
}

#[test]
fn test_check_canary_canary_not_found() {
    let model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    let result = check_canary(model.path(), Path::new("/no/canary.json"));
    assert!(result.is_err());
}

#[test]
fn test_check_canary_invalid_canary_json() {
    let model = NamedTempFile::with_suffix(".safetensors").expect("create model");
    let mut canary_file = NamedTempFile::with_suffix(".json").expect("create canary");
    canary_file.write_all(b"NOT JSON").expect("write");
    let result = check_canary(model.path(), canary_file.path());
    assert!(result.is_err());
}

// ========================================================================
// create_canary + check_canary round-trip
// ========================================================================

#[test]
fn test_create_then_check_canary_round_trip() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Build GGUF model with two tensors
    let floats1: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let floats2: Vec<u8> = [0.1f32, 0.2, 0.3, 0.4]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let tensors = vec![
        GgufTensor {
            name: "layer.weight".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: floats1,
        },
        GgufTensor {
            name: "layer.bias".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: floats2,
        },
    ];
    let metadata_kv = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];
    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata_kv).expect("export GGUF");

    let mut model_file = NamedTempFile::with_suffix(".gguf").expect("create model");
    model_file.write_all(&gguf_bytes).expect("write");

    let input_file = NamedTempFile::with_suffix(".wav").expect("create input");
    let canary_file = NamedTempFile::with_suffix(".json").expect("create canary output");

    // Step 1: create canary
    let create_result = create_canary(model_file.path(), input_file.path(), canary_file.path());
    assert!(
        create_result.is_ok(),
        "create should succeed: {create_result:?}"
    );

    // Step 2: check same model against its own canary (should pass)
    let check_result = check_canary(model_file.path(), canary_file.path());
    assert!(
        check_result.is_ok(),
        "self-check should pass: {check_result:?}"
    );
}

// ========================================================================
// TensorCanary field-level edge cases
// ========================================================================

#[test]
fn test_tensor_canary_zero_count() {
    let t = TensorCanary {
        shape: vec![],
        count: 0,
        mean: 0.0,
        std: 0.0,
        min: 0.0,
        max: 0.0,
    };
    let json = serde_json::to_string(&t).expect("serialize");
    let parsed: TensorCanary = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.count, 0);
    assert!(parsed.shape.is_empty());
}

#[test]
fn test_tensor_canary_negative_values() {
    let t = TensorCanary {
        shape: vec![10],
        count: 10,
        mean: -0.5,
        std: 0.3,
        min: -1.0,
        max: 0.0,
    };
    let json = serde_json::to_string(&t).expect("serialize");
    let parsed: TensorCanary = serde_json::from_str(&json).expect("deserialize");
    assert!((parsed.mean - (-0.5)).abs() < f32::EPSILON);
    assert!((parsed.min - (-1.0)).abs() < f32::EPSILON);
}

#[test]
fn test_tensor_canary_large_shape() {
    let t = TensorCanary {
        shape: vec![4096, 4096, 32],
        count: 536_870_912,
        mean: 0.0,
        std: 0.02,
        min: -0.1,
        max: 0.1,
    };
    let json = serde_json::to_string(&t).expect("serialize");
    let parsed: TensorCanary = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.shape, vec![4096, 4096, 32]);
    assert_eq!(parsed.count, 536_870_912);
}

// ========================================================================
// CanaryCheckResult edge cases
// ========================================================================

#[test]
fn test_canary_check_result_clone() {
    let result = CanaryCheckResult {
        tensor_name: "test".to_string(),
        passed: false,
        mean_drift: 0.15,
        std_drift: 0.05,
        shape_match: true,
        message: Some("drift exceeded".to_string()),
    };
    let cloned = result.clone();
    assert_eq!(cloned.tensor_name, "test");
    assert!(!cloned.passed);
    assert_eq!(cloned.message, Some("drift exceeded".to_string()));
}

#[test]
fn test_canary_check_result_max_drift_values() {
    let result = CanaryCheckResult {
        tensor_name: "extreme".to_string(),
        passed: false,
        mean_drift: f32::MAX,
        std_drift: f32::MAX,
        shape_match: false,
        message: Some("everything wrong".to_string()),
    };
    assert_eq!(result.mean_drift, f32::MAX);
    assert_eq!(result.std_drift, f32::MAX);
}
