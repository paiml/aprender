use super::*;

#[test]
fn tcov_tensor_validation_none_of_the_above() {
    let tv = TensorValidation {
        name: "healthy.weight".to_string(),
        is_valid: true,
        nan_count: 0,
        inf_count: 0,
        zero_count: 10,
        element_count: 100,
        min: -1.0,
        max: 1.0,
        mean: 0.01,
        std: 0.5,
        failures: vec![],
    };
    assert!(!tv.has_nan());
    assert!(!tv.has_inf());
    assert!(!tv.is_all_zeros());
}

// ========================================================================
// Section 18: ValidationReport Methods & Display (T-COV-95)
// ========================================================================

#[test]
fn tcov_validation_report_passed_true() {
    let report = ValidationReport {
        format: FormatType::Apr,
        file_path: "test.apr".to_string(),
        is_valid: true,
        tensor_count: 5,
        failed_tensor_count: 0,
        total_nan_count: 0,
        total_inf_count: 0,
        all_zero_tensors: vec![],
        tensors: vec![],
        duration_ms: 50,
    };
    assert!(report.passed());
    let summary = report.summary();
    assert!(summary.contains("VALID"));
    assert!(summary.contains("5 tensors"));
    assert!(summary.contains("0 contract violations"));
}

#[test]
fn tcov_validation_report_passed_false() {
    let report = ValidationReport {
        format: FormatType::Gguf,
        file_path: "test.gguf".to_string(),
        is_valid: false,
        tensor_count: 10,
        failed_tensor_count: 3,
        total_nan_count: 15,
        total_inf_count: 2,
        all_zero_tensors: vec!["layer.5.weight".to_string()],
        tensors: vec![
            TensorValidation {
                name: "layer.0.weight".to_string(),
                is_valid: false,
                nan_count: 10,
                inf_count: 0,
                zero_count: 0,
                element_count: 1000,
                min: -1.0,
                max: 1.0,
                mean: 0.0,
                std: 0.5,
                failures: vec!["[F-DATA-QUALITY-002] 10 NaN values detected".to_string()],
            },
            TensorValidation {
                name: "layer.1.weight".to_string(),
                is_valid: false,
                nan_count: 5,
                inf_count: 2,
                zero_count: 0,
                element_count: 500,
                min: -2.0,
                max: 2.0,
                mean: 0.1,
                std: 0.8,
                failures: vec![
                    "[F-DATA-QUALITY-002] 5 NaN values detected".to_string(),
                    "[F-DATA-QUALITY-002] 2 Inf values detected".to_string(),
                ],
            },
        ],
        duration_ms: 100,
    };
    assert!(!report.passed());
    let summary = report.summary();
    assert!(summary.contains("INVALID"));
    assert!(summary.contains("15 NaN"));
    assert!(summary.contains("2 Inf"));
    assert!(summary.contains("1 all-zeros"));
}

#[test]
fn tcov_validation_report_display_invalid() {
    let report = ValidationReport {
        format: FormatType::SafeTensors,
        file_path: "broken.safetensors".to_string(),
        is_valid: false,
        tensor_count: 3,
        failed_tensor_count: 1,
        total_nan_count: 7,
        total_inf_count: 0,
        all_zero_tensors: vec![],
        tensors: vec![
            TensorValidation {
                name: "bad.weight".to_string(),
                is_valid: false,
                nan_count: 7,
                inf_count: 0,
                zero_count: 0,
                element_count: 100,
                min: -1.0,
                max: 1.0,
                mean: 0.0,
                std: 0.5,
                failures: vec!["[F-DATA-QUALITY-002] 7 NaN values detected".to_string()],
            },
            TensorValidation {
                name: "good.weight".to_string(),
                is_valid: true,
                nan_count: 0,
                inf_count: 0,
                zero_count: 5,
                element_count: 100,
                min: -1.0,
                max: 1.0,
                mean: 0.01,
                std: 0.5,
                failures: vec![],
            },
        ],
        duration_ms: 42,
    };
    let display = format!("{report}");
    assert!(display.contains("INVALID"));
    assert!(display.contains("SafeTensors"));
    assert!(display.contains("broken.safetensors"));
    assert!(display.contains("Failed Tensors"));
    assert!(display.contains("bad.weight"));
    assert!(display.contains("7 NaN"));
    assert!(display.contains("F-DATA-QUALITY-002"));
    // good.weight should NOT appear in failed tensors section
    assert!(!display.contains("good.weight: "));
}

#[test]
fn tcov_validation_report_display_valid() {
    let report = ValidationReport {
        format: FormatType::Apr,
        file_path: "good.apr".to_string(),
        is_valid: true,
        tensor_count: 5,
        failed_tensor_count: 0,
        total_nan_count: 0,
        total_inf_count: 0,
        all_zero_tensors: vec![],
        tensors: vec![],
        duration_ms: 10,
    };
    let display = format!("{report}");
    assert!(display.contains("VALID"));
    assert!(!display.contains("Failed Tensors"));
}

// ========================================================================
// Section 19: InspectionReport Display Edge Cases (T-COV-95)
// ========================================================================

#[test]
fn tcov_inspection_display_with_architecture_and_quantization() {
    let report = InspectionReport {
        format: FormatType::Gguf,
        file_size: 5_000_000_000,
        metadata: BTreeMap::new(),
        tensors: vec![TensorInfo {
            name: "embed.weight".to_string(),
            dtype: "Q4_K_M".to_string(),
            shape: vec![32000, 4096],
            size_bytes: 32000 * 4096 / 2,
            stats: None,
        }],
        total_params: 7_000_000_000,
        quantization: Some("Q4_K_M".to_string()),
        architecture: Some("llama".to_string()),
    };
    let display = format!("{report}");
    assert!(display.contains("Architecture: llama"));
    assert!(display.contains("Quantization: Q4_K_M"));
    assert!(display.contains("GGUF"));
}

#[test]
fn tcov_inspection_display_truncates_many_tensors() {
    // Create 20 tensors - Display should truncate middle ones
    let tensors: Vec<TensorInfo> = (0..20)
        .map(|i| TensorInfo {
            name: format!("layer.{i}.weight"),
            dtype: "F32".to_string(),
            shape: vec![256, 256],
            size_bytes: 256 * 256 * 4,
            stats: None,
        })
        .collect();

    let report = InspectionReport {
        format: FormatType::Apr,
        file_size: 20 * 256 * 256 * 4,
        metadata: BTreeMap::new(),
        tensors,
        total_params: 20 * 256 * 256,
        quantization: None,
        architecture: None,
    };
    let display = format!("{report}");
    // Should show first 10, then "... (N more tensors) ...", then last 2
    assert!(display.contains("layer.0.weight"));
    assert!(display.contains("layer.9.weight"));
    assert!(display.contains("more tensors"));
    assert!(display.contains("layer.18.weight"));
    assert!(display.contains("layer.19.weight"));
    // Middle tensors should NOT appear
    assert!(!display.contains("layer.11.weight"));
}

#[test]
fn tcov_inspection_display_metadata_truncation() {
    let mut metadata = BTreeMap::new();
    let long_value = "a".repeat(100);
    metadata.insert("long_key".to_string(), long_value);
    metadata.insert("short_key".to_string(), "short".to_string());

    let report = InspectionReport {
        format: FormatType::SafeTensors,
        file_size: 100,
        metadata,
        tensors: vec![],
        total_params: 0,
        quantization: None,
        architecture: None,
    };
    let display = format!("{report}");
    assert!(display.contains("long_key: aaaaaa")); // starts with 'a's
    assert!(display.contains("...")); // truncated
    assert!(display.contains("short_key: short")); // short value not truncated
}

// ========================================================================
// Section 20: Validate with NaN/Inf/Zeros Fixtures (T-COV-95)
// Exercises compute_tensor_validation through the public validate() API
// ========================================================================

#[test]
fn tcov_validate_safetensors_with_nan() {
    use std::io::Write;

    let path = unique_temp_path("test_nan", "safetensors");
    let mut file = std::fs::File::create(&path).expect("Create temp file");

    let header = r#"{"test.bias":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("Write header len");
    file.write_all(header.as_bytes()).expect("Write header");

    // Include a NaN value
    let data: [f32; 4] = [0.1, f32::NAN, 0.3, -0.1];
    for val in &data {
        file.write_all(&val.to_le_bytes()).expect("Write tensor");
    }
    drop(file);

    let rosetta = RosettaStone::new();
    let report = rosetta.validate(&path).expect("validate");
    assert!(!report.is_valid);
    assert!(report.total_nan_count > 0);
    assert!(report.tensors.iter().any(|t| t.has_nan()));

    let _ = std::fs::remove_file(path);
}

#[test]
fn tcov_validate_safetensors_with_inf() {
    use std::io::Write;

    let path = unique_temp_path("test_inf", "safetensors");
    let mut file = std::fs::File::create(&path).expect("Create temp file");

    let header = r#"{"test.bias":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("Write header len");
    file.write_all(header.as_bytes()).expect("Write header");

    let data: [f32; 4] = [0.1, f32::INFINITY, 0.3, f32::NEG_INFINITY];
    for val in &data {
        file.write_all(&val.to_le_bytes()).expect("Write tensor");
    }
    drop(file);

    let rosetta = RosettaStone::new();
    let report = rosetta.validate(&path).expect("validate");
    assert!(!report.is_valid);
    assert!(report.total_inf_count > 0);
    assert!(report.tensors.iter().any(|t| t.has_inf()));

    let _ = std::fs::remove_file(path);
}

#[test]
fn tcov_validate_safetensors_all_zeros() {
    use std::io::Write;

    let path = unique_temp_path("test_zeros", "safetensors");
    let mut file = std::fs::File::create(&path).expect("Create temp file");

    let header = r#"{"test.weight":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("Write header len");
    file.write_all(header.as_bytes()).expect("Write header");

    let data: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    for val in &data {
        file.write_all(&val.to_le_bytes()).expect("Write tensor");
    }
    drop(file);

    let rosetta = RosettaStone::new();
    let report = rosetta.validate(&path).expect("validate");
    assert!(!report.is_valid);
    assert!(!report.all_zero_tensors.is_empty());

    let _ = std::fs::remove_file(path);
}

// ========================================================================
// Section 21: load_tensor_f32 Tests (T-COV-95)
// ========================================================================

#[test]
fn tcov_load_tensor_f32_safetensors() {
    let path = create_safetensors_fixture();
    let rosetta = RosettaStone::new();
    let data = rosetta
        .load_tensor_f32(&path, "test.bias")
        .expect("load tensor");
    assert_eq!(data.len(), 4);
    assert!((data[0] - 0.01).abs() < 1e-6);
    let _ = std::fs::remove_file(path);
}

#[test]
fn tcov_load_tensor_f32_apr() {
    let path = create_apr_fixture();
    let rosetta = RosettaStone::new();
    let data = rosetta
        .load_tensor_f32(&path, "test.bias")
        .expect("load tensor");
    assert_eq!(data.len(), 4);
    assert!((data[0] - 0.01).abs() < 1e-5);
    let _ = std::fs::remove_file(path);
}

#[test]
fn tcov_load_tensor_f32_not_found() {
    let path = create_apr_fixture();
    let rosetta = RosettaStone::new();
    let result = rosetta.load_tensor_f32(&path, "nonexistent.tensor");
    assert!(result.is_err());
    let _ = std::fs::remove_file(path);
}

#[test]
fn tcov_load_tensor_f32_safetensors_not_found() {
    let path = create_safetensors_fixture();
    let rosetta = RosettaStone::new();
    let result = rosetta.load_tensor_f32(&path, "nonexistent.tensor");
    assert!(result.is_err());
    let _ = std::fs::remove_file(path);
}

// ========================================================================
// Section 22: Chain & Verify Roundtrip Tests (T-COV-95)
// ========================================================================

#[test]
fn tcov_chain_too_short() {
    let path = create_safetensors_fixture();
    let rosetta = RosettaStone::new();
    let result = rosetta.chain(&path, &[FormatType::SafeTensors], Path::new("/tmp"));
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("at least 2"));
    let _ = std::fs::remove_file(path);
}

#[test]
fn tcov_chain_with_cycle_detection() {
    let path = create_safetensors_fixture();
    let rosetta = RosettaStone::new();
    // Chain: SafeTensors → APR → SafeTensors → APR → SafeTensors has APR twice
    let result = rosetta.chain(
        &path,
        &[
            FormatType::SafeTensors,
            FormatType::Apr,
            FormatType::SafeTensors,
            FormatType::Apr,
            FormatType::SafeTensors,
        ],
        Path::new("/tmp"),
    );
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("cycle"));
    let _ = std::fs::remove_file(path);
}

#[test]
fn tcov_chain_safetensors_to_apr() {
    let source = create_safetensors_fixture();
    let work_dir = std::env::temp_dir().join("rosetta_chain_test");
    std::fs::create_dir_all(&work_dir).expect("Create work dir");

    let rosetta = RosettaStone::new();
    let reports = rosetta
        .chain(
            &source,
            &[FormatType::SafeTensors, FormatType::Apr],
            &work_dir,
        )
        .expect("chain conversion");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].path.source, FormatType::SafeTensors);
    assert_eq!(reports[0].path.target, FormatType::Apr);

    let _ = std::fs::remove_file(source);
    let _ = std::fs::remove_dir_all(work_dir);
}
