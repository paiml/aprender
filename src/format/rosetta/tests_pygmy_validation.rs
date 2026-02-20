use super::*;

/// P070i: compute_tensor_validation embedding below density threshold (49% zeros)
/// Covers: density gate where zero_pct <= threshold (embedding at 49%, under 50%)
#[test]
fn p070i_compute_validation_embedding_below_density_threshold() {
    let rosetta = RosettaStone::new();
    let mut data = vec![0.0_f32; 49];
    data.extend(vec![0.1_f32; 51]);
    let tv = rosetta.compute_tensor_validation("model.embed_tokens.weight", &data);
    assert!(
        !tv.failures.iter().any(|f| f.contains("DENSITY")),
        "49% zeros in embedding should not trigger 50% density gate: {:?}",
        tv.failures
    );
}

// ========================================================================
// P080-P089: Struct construction, VerificationReport, ConversionOptions
// ========================================================================

/// P080: VerificationReport construction with tensor_diffs and changed_metadata
/// Covers: VerificationReport fields used in compare_files
#[test]
fn p080_verification_report_full_construction() {
    let mut tensor_diffs = BTreeMap::new();
    tensor_diffs.insert("layer.0.weight".to_string(), 1e-5_f32);
    tensor_diffs.insert("layer.1.weight".to_string(), 2e-5_f32);

    let report = VerificationReport {
        is_equivalent: true,
        max_diff: 2e-5,
        mean_diff: 1.5e-5,
        tensor_diffs: tensor_diffs.clone(),
        changed_metadata: vec!["conversion_tool".to_string()],
        failed_tensors: Vec::new(),
    };
    assert!(report.is_equivalent);
    assert!(report.passes_with_tolerance(1e-4));
    assert!(!report.passes_with_tolerance(1e-6));
    assert_eq!(report.tensor_diffs.len(), 2);
    assert_eq!(report.changed_metadata.len(), 1);
}

/// P081b: VerificationReport with max_diff exactly at tolerance boundary
#[test]
fn p081b_verification_tolerance_boundary() {
    let report = VerificationReport {
        is_equivalent: true,
        max_diff: 1e-4,
        mean_diff: 1e-5,
        tensor_diffs: BTreeMap::new(),
        changed_metadata: Vec::new(),
        failed_tensors: Vec::new(),
    };
    // max_diff == epsilon should pass (<=)
    assert!(
        report.passes_with_tolerance(1e-4),
        "max_diff == epsilon should pass"
    );
    // max_diff > epsilon should fail
    assert!(
        !report.passes_with_tolerance(9.99e-5),
        "max_diff > epsilon should fail"
    );
}

/// P082b: TensorStats debug and clone
#[test]
fn p082b_tensor_stats_debug_clone() {
    let stats = TensorStats {
        min: -3.14,
        max: 2.71,
        mean: -0.21,
        std: 1.41,
    };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("TensorStats"));
    let stats2 = stats; // Copy trait
    assert_eq!(stats2.min, -3.14);
}

/// P083b: ConversionOptions quantization mapping (q4_k, int8, fp16, etc.)
/// Tests the quantization string mapping in convert_internal indirectly
/// by verifying the options struct accepts all expected values
#[test]
fn p083b_conversion_options_quantization_variants() {
    let variants = [
        "q4_k", "q4_k_m", "int4", "q6_k", "int8", "q8_0", "fp16", "f16",
    ];
    for variant in &variants {
        let opts = ConversionOptions {
            quantization: Some(variant.to_string()),
            ..Default::default()
        };
        assert_eq!(
            opts.quantization.as_deref(),
            Some(*variant),
            "Should accept quantization variant: {}",
            variant
        );
    }
}

/// P084b: ConversionOptions with unknown quantization (maps to None in convert_internal)
#[test]
fn p084b_conversion_options_unknown_quantization() {
    let opts = ConversionOptions {
        quantization: Some("unknown_quant".to_string()),
        ..Default::default()
    };
    // The ConversionOptions struct itself just stores the string
    // The mapping happens in convert_internal - "unknown_quant" would map to None
    assert_eq!(opts.quantization, Some("unknown_quant".to_string()));
}

// ========================================================================
// P085-P089: InspectionReport and ValidationReport edge cases
// ========================================================================

/// P085: InspectionReport display with exactly 12 tensors (boundary for truncation)
/// In Display impl: first 10 shown, then "... (N more) ...", then last 2
/// With 12 tensors: first 10 + "... (0 more) ..." + last 2 = all shown
#[test]
fn p085_inspection_display_exactly_12_tensors() {
    let tensors: Vec<TensorInfo> = (0..12)
        .map(|i| TensorInfo {
            name: format!("layer.{i}.weight"),
            dtype: "F32".to_string(),
            shape: vec![64, 64],
            size_bytes: 64 * 64 * 4,
            stats: None,
        })
        .collect();
    let report = InspectionReport {
        format: FormatType::Apr,
        file_size: 12 * 64 * 64 * 4,
        metadata: BTreeMap::new(),
        tensors,
        total_params: 12 * 64 * 64,
        quantization: None,
        architecture: None,
    };
    let display = format!("{report}");
    // With 12 tensors: indices 0-9 shown, index 10 shows "... (0 more) ...", indices 10-11 shown
    assert!(display.contains("layer.0.weight"));
    assert!(display.contains("layer.9.weight"));
    assert!(display.contains("layer.10.weight"));
    assert!(display.contains("layer.11.weight"));
}

/// P086: InspectionReport display with 11 tensors (just over truncation boundary)
#[test]
fn p086_inspection_display_11_tensors() {
    let tensors: Vec<TensorInfo> = (0..11)
        .map(|i| TensorInfo {
            name: format!("t.{i}"),
            dtype: "F32".to_string(),
            shape: vec![4],
            size_bytes: 16,
            stats: None,
        })
        .collect();
    let report = InspectionReport {
        format: FormatType::Apr,
        file_size: 0,
        metadata: BTreeMap::new(),
        tensors,
        total_params: 0,
        quantization: None,
        architecture: None,
    };
    let display = format!("{report}");
    // With 11 tensors, the middle section shows at i==10: "... (N more) ..."
    // i < 10 -> shown, i == 10 -> "...", i >= 11-2=9 -> shown
    // So t.0 through t.9 shown, t.10 shows "...", t.9 and t.10 shown as last 2
    assert!(display.contains("t.0"));
    assert!(display.contains("t.9"));
    assert!(display.contains("t.10"));
}

/// P087: ValidationReport summary with contract failures counted
/// Covers: lines 553-554 (contract_failures summed from tensor failures)
#[test]
fn p087_validation_summary_contract_failure_count() {
    let report = ValidationReport {
        format: FormatType::Gguf,
        file_path: "test.gguf".to_string(),
        is_valid: false,
        tensor_count: 3,
        failed_tensor_count: 2,
        total_nan_count: 5,
        total_inf_count: 3,
        all_zero_tensors: vec!["dead.weight".to_string()],
        tensors: vec![
            TensorValidation {
                name: "t1.weight".to_string(),
                is_valid: false,
                nan_count: 5,
                inf_count: 0,
                zero_count: 0,
                element_count: 100,
                min: -1.0,
                max: 1.0,
                mean: 0.0,
                std: 0.5,
                failures: vec![
                    "[F-DATA-QUALITY-002] 5 NaN detected".to_string(),
                    "[F-DATA-QUALITY-003] variation issue".to_string(),
                ],
            },
            TensorValidation {
                name: "t2.weight".to_string(),
                is_valid: false,
                nan_count: 0,
                inf_count: 3,
                zero_count: 0,
                element_count: 50,
                min: -2.0,
                max: 2.0,
                mean: 0.0,
                std: 1.0,
                failures: vec!["[F-DATA-QUALITY-002] 3 Inf detected".to_string()],
            },
        ],
        duration_ms: 50,
    };
    let summary = report.summary();
    // contract_failures = 2 + 1 = 3
    assert!(
        summary.contains("3 contract violations"),
        "Summary should count 3 contract violations: {}",
        summary
    );
    assert!(summary.contains("INVALID"));
    assert!(summary.contains("5 NaN"));
    assert!(summary.contains("3 Inf"));
    assert!(summary.contains("1 all-zeros"));
}

/// P088: ValidationReport display for invalid report shows failure details
#[test]
fn p088_validation_display_failure_details() {
    let report = ValidationReport {
        format: FormatType::Apr,
        file_path: "bad_model.apr".to_string(),
        is_valid: false,
        tensor_count: 2,
        failed_tensor_count: 1,
        total_nan_count: 0,
        total_inf_count: 0,
        all_zero_tensors: vec![],
        tensors: vec![
            TensorValidation {
                name: "good.bias".to_string(),
                is_valid: true,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
                element_count: 10,
                min: -0.1,
                max: 0.1,
                mean: 0.0,
                std: 0.05,
                failures: vec![],
            },
            TensorValidation {
                name: "bad.weight".to_string(),
                is_valid: false,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
                element_count: 100,
                min: 0.5,
                max: 0.5,
                mean: 0.5,
                std: 0.0,
                failures: vec![
                    "[F-DATA-QUALITY-003] All values identical: tensor is constant".to_string(),
                ],
            },
        ],
        duration_ms: 25,
    };
    let display = format!("{report}");
    assert!(display.contains("INVALID"));
    assert!(display.contains("Failed Tensors"));
    assert!(display.contains("bad.weight"));
    assert!(display.contains("F-DATA-QUALITY-003"));
    // good.bias should NOT appear in failed section
    assert!(
        !display.contains("good.bias:"),
        "Valid tensor should not appear in failed section"
    );
}

/// P089: compute_tensor_validation with healthy normal data
/// Covers: the happy path where all gates pass
#[test]
fn p089_compute_validation_healthy_data() {
    let rosetta = RosettaStone::new();
    // Normal random-like weight values
    let data: Vec<f32> = (0..1000)
        .map(|i| ((i as f32) * 0.001 - 0.5) * 2.0)
        .collect();
    let tv = rosetta.compute_tensor_validation("model.layers.0.weight", &data);
    assert!(
        tv.is_valid,
        "Healthy data should be valid: {:?}",
        tv.failures
    );
    assert_eq!(tv.nan_count, 0);
    assert_eq!(tv.inf_count, 0);
    assert_eq!(tv.element_count, 1000);
    assert!(tv.min < tv.max);
    assert!(tv.std > 0.0);
    assert!(tv.failures.is_empty());
}

// ========================================================================
// Bug 212: Sharded SafeTensors detection
// ========================================================================

#[test]
fn test_bug_212_is_sharded_index_positive() {
    use super::is_sharded_index;
    use std::path::Path;

    assert!(is_sharded_index(Path::new("model.safetensors.index.json")));
    assert!(is_sharded_index(Path::new(
        "/path/to/model.safetensors.index.json"
    )));
    assert!(is_sharded_index(Path::new("some.other.index.json")));
}

#[test]
fn test_bug_212_is_sharded_index_negative() {
    use super::is_sharded_index;
    use std::path::Path;

    assert!(!is_sharded_index(Path::new("model.safetensors")));
    assert!(!is_sharded_index(Path::new("model.gguf")));
    assert!(!is_sharded_index(Path::new("config.json")));
    assert!(!is_sharded_index(Path::new("tokenizer.json")));
}

#[test]
fn test_bug_212_inspect_sharded_nonexistent() {
    let rosetta = RosettaStone::new();
    let result = rosetta.inspect("/tmp/nonexistent_sharded_model.safetensors.index.json");
    // Should fail because the file doesn't exist
    assert!(result.is_err());
}

#[test]
fn test_bug_212_convert_sharded_nonexistent() {
    let rosetta = RosettaStone::new();
    let result = rosetta.convert(
        "/tmp/nonexistent_sharded_model.safetensors.index.json",
        "/tmp/output.apr",
        None,
    );
    // Should fail because the source doesn't exist
    assert!(result.is_err());
}
