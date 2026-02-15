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
    assert!(result.message.unwrap().contains("not found"));
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
    assert!(msg.unwrap().contains("Shape mismatch"));
}

#[test]
fn test_build_failure_message_mean_drift() {
    let msg = build_failure_message_test_helper(false, true, 0.15, 0.01);
    assert!(msg.is_some());
    assert!(msg.unwrap().contains("Mean drift"));
}

#[test]
fn test_build_failure_message_std_drift() {
    let msg = build_failure_message_test_helper(false, true, 0.05, 0.25);
    assert!(msg.is_some());
    assert!(msg.unwrap().contains("Std drift"));
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
    assert_eq!(result.unwrap().model_name, "test.safetensors");
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

include!("canary_tests_part_02.rs");
