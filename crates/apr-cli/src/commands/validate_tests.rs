use super::*;
use std::collections::HashMap;
use std::io::Write;
use tempfile::{tempdir, NamedTempFile};

// ========================================================================
// Path Validation Tests
// ========================================================================

#[test]
fn test_validate_path_not_found() {
    let result = validate_path(Path::new("/nonexistent/model.apr"));
    assert!(result.is_err());
    match result {
        Err(CliError::FileNotFound(_)) => {}
        _ => panic!("Expected FileNotFound error"),
    }
}

#[test]
fn test_validate_path_is_directory() {
    let dir = tempdir().expect("create temp dir");
    let result = validate_path(dir.path());
    assert!(result.is_err());
    match result {
        Err(CliError::NotAFile(_)) => {}
        _ => panic!("Expected NotAFile error"),
    }
}

#[test]
fn test_validate_path_valid_file() {
    let file = NamedTempFile::new().expect("create temp file");
    let result = validate_path(file.path());
    assert!(result.is_ok());
}

// ========================================================================
// Run Command Tests
// ========================================================================

#[test]
fn test_run_file_not_found() {
    let result = run(
        Path::new("/nonexistent/model.apr"),
        false,
        false,
        None,
        false,
    );
    assert!(result.is_err());
    match result {
        Err(CliError::FileNotFound(_)) => {}
        _ => panic!("Expected FileNotFound error"),
    }
}

#[test]
fn test_run_is_directory() {
    let dir = tempdir().expect("create temp dir");
    let result = run(dir.path(), false, false, None, false);
    assert!(result.is_err());
    match result {
        Err(CliError::NotAFile(_)) => {}
        _ => panic!("Expected NotAFile error"),
    }
}

#[test]
fn test_run_invalid_file() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not a valid APR file").expect("write");

    let result = run(file.path(), false, false, None, false);
    // Should fail validation because file is not valid APR
    assert!(result.is_err());
}

#[test]
fn test_run_with_quality_flag() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"invalid data").expect("write");

    let result = run(file.path(), true, false, None, false);
    // Should fail but quality flag is handled
    assert!(result.is_err());
}

#[test]
fn test_run_with_min_score() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"invalid data").expect("write");

    let result = run(file.path(), false, false, Some(100), false);
    // Should fail before min_score check because file is invalid
    assert!(result.is_err());
}

#[test]
fn test_run_with_strict_flag() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"test data").expect("write");

    let result = run(file.path(), false, true, None, false);
    // Should fail with strict mode
    assert!(result.is_err());
}

#[test]
fn test_run_with_all_flags() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"test data").expect("write");

    let result = run(file.path(), true, true, Some(50), false);
    // Should fail with all flags enabled
    assert!(result.is_err());
}

#[test]
fn test_run_empty_file() {
    let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    // Empty file - no write

    let result = run(file.path(), false, false, None, false);
    // Empty file should fail validation
    assert!(result.is_err());
}

// ========================================================================
// Category Score Tests (using mocked reports via AprValidator)
// ========================================================================

#[test]
fn test_quality_assessment_display() {
    let mut category_scores = HashMap::new();
    category_scores.insert(Category::Structure, 25);
    category_scores.insert(Category::Physics, 20);
    category_scores.insert(Category::Tooling, 15);
    category_scores.insert(Category::Conversion, 10);

    let report = ValidationReport {
        checks: Vec::new(),
        total_score: 70,
        category_scores,
    };

    // Should not panic
    print_quality_assessment(&report);
}

#[test]
fn test_quality_assessment_missing_categories() {
    let report = ValidationReport {
        checks: Vec::new(),
        total_score: 0,
        category_scores: HashMap::new(),
    };

    // Should handle missing categories gracefully (default to 0)
    print_quality_assessment(&report);
}

#[test]
fn test_quality_assessment_all_score_ranges() {
    // High scores
    let mut high_scores = HashMap::new();
    high_scores.insert(Category::Structure, 25);
    high_scores.insert(Category::Physics, 25);
    high_scores.insert(Category::Tooling, 25);
    high_scores.insert(Category::Conversion, 25);

    let high_report = ValidationReport {
        checks: Vec::new(),
        total_score: 100,
        category_scores: high_scores,
    };

    // Low scores
    let mut low_scores = HashMap::new();
    low_scores.insert(Category::Structure, 5);

    let low_report = ValidationReport {
        checks: Vec::new(),
        total_score: 5,
        category_scores: low_scores,
    };

    // All should display without panic
    print_quality_assessment(&high_report);
    print_quality_assessment(&low_report);
}

// ========================================================================
// Print Summary Tests
// ========================================================================

#[test]
fn test_print_summary_valid_report() {
    let report = ValidationReport {
        checks: Vec::new(), // No failed checks
        total_score: 100,
        category_scores: HashMap::new(),
    };

    let result = print_summary(&report, false);
    assert!(result.is_ok());
}

#[test]
fn test_print_quality_assessment_empty() {
    let report = ValidationReport {
        checks: Vec::new(),
        total_score: 0,
        category_scores: HashMap::new(),
    };

    // Should not panic even with empty report
    print_quality_assessment(&report);
}

// ========================================================================
// Multi-Format Dispatch Tests (GGUF, SafeTensors)
// ========================================================================

#[test]
fn test_run_gguf_format_dispatch() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create valid GGUF file with non-zero tensor data
    let floats: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![4, 4],
        dtype: GgmlType::F32,
        data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    file.write_all(&gguf_bytes).expect("write GGUF");

    // Should dispatch to GGUF validation path (RosettaStone::validate)
    let result = run(file.path(), false, false, None, false);
    // GGUF validation should succeed (physics constraints pass)
    assert!(result.is_ok(), "GGUF format dispatch should work");
}

#[test]
fn test_run_safetensors_format_dispatch() {
    // Create valid SafeTensors file manually
    let header_json = serde_json::json!({
        "test.weight": {
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
    // Add valid tensor data (4 floats = 16 bytes)
    let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    for f in floats {
        st_bytes.extend_from_slice(&f.to_le_bytes());
    }

    let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    file.write_all(&st_bytes).expect("write SafeTensors");

    // Should dispatch to SafeTensors validation path (RosettaStone::validate)
    let result = run(file.path(), false, false, None, false);
    // SafeTensors validation should succeed
    assert!(result.is_ok(), "SafeTensors format dispatch should work");
}

#[test]
fn test_run_gguf_format_detection_by_magic() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create GGUF with .bin extension (magic detection, not extension)
    // Use valid non-zero tensor data
    let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let tensor_data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: tensor_data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
    file.write_all(&gguf_bytes).expect("write GGUF");

    // Should detect GGUF by magic bytes, not extension
    let result = run(file.path(), false, false, None, false);
    assert!(result.is_ok(), "Should detect GGUF by magic bytes");
}

#[test]
fn test_run_gguf_with_physics_violations() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create GGUF with NaN values (physics violation)
    let nan_f32 = f32::NAN.to_le_bytes();
    let mut tensor_data = Vec::new();
    for _ in 0..4 {
        tensor_data.extend_from_slice(&nan_f32);
    }

    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: tensor_data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    file.write_all(&gguf_bytes).expect("write GGUF");

    // Should fail due to NaN physics violation
    let result = run(file.path(), false, false, None, false);
    assert!(result.is_err(), "Should fail with NaN tensors");
}
