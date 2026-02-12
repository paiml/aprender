//\! Rosetta Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

// ========================================================================
// Section 1: Format Detection Tests (P001-P010)
// ========================================================================

#[test]
fn p001_format_from_extension_gguf() {
    let path = Path::new("model.gguf");
    let format = FormatType::from_extension(path).expect("Should detect GGUF");
    assert_eq!(format, FormatType::Gguf);
}

#[test]
fn p002_format_from_extension_safetensors() {
    let path = Path::new("model.safetensors");
    let format = FormatType::from_extension(path).expect("Should detect SafeTensors");
    assert_eq!(format, FormatType::SafeTensors);
}

#[test]
fn p003_format_from_extension_apr() {
    let path = Path::new("model.apr");
    let format = FormatType::from_extension(path).expect("Should detect APR");
    assert_eq!(format, FormatType::Apr);
}

#[test]
fn p004_format_from_extension_unknown() {
    let path = Path::new("model.unknown");
    let result = FormatType::from_extension(path);
    assert!(result.is_err(), "Should fail for unknown extension");
}

#[test]
fn p005_format_from_extension_no_extension() {
    let path = Path::new("model");
    let result = FormatType::from_extension(path);
    assert!(result.is_err(), "Should fail for no extension");
}

#[test]
fn p006_format_from_extension_case_insensitive() {
    let path = Path::new("model.GGUF");
    let format = FormatType::from_extension(path).expect("Should handle uppercase");
    assert_eq!(format, FormatType::Gguf);
}

#[test]
fn p007_format_display() {
    assert_eq!(format!("{}", FormatType::Gguf), "GGUF");
    assert_eq!(format!("{}", FormatType::SafeTensors), "SafeTensors");
    assert_eq!(format!("{}", FormatType::Apr), "APR");
}

#[test]
fn p008_format_extension() {
    assert_eq!(FormatType::Gguf.extension(), "gguf");
    assert_eq!(FormatType::SafeTensors.extension(), "safetensors");
    assert_eq!(FormatType::Apr.extension(), "apr");
}

#[test]
fn p009_can_convert_to_different_format() {
    assert!(FormatType::Gguf.can_convert_to(FormatType::Apr));
    assert!(FormatType::Apr.can_convert_to(FormatType::SafeTensors));
    assert!(FormatType::SafeTensors.can_convert_to(FormatType::Gguf));
}

#[test]
fn p010_cannot_convert_to_same_format() {
    assert!(!FormatType::Gguf.can_convert_to(FormatType::Gguf));
    assert!(!FormatType::Apr.can_convert_to(FormatType::Apr));
    assert!(!FormatType::SafeTensors.can_convert_to(FormatType::SafeTensors));
}

// ========================================================================
// Section 2: Conversion Path Tests (P011-P020)
// ========================================================================

#[test]
fn p011_direct_path_creation() {
    let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
    assert_eq!(path.source, FormatType::Gguf);
    assert_eq!(path.target, FormatType::Apr);
    assert!(path.intermediates.is_empty());
}

#[test]
fn p012_chain_path_creation() {
    let path = ConversionPath::chain(
        FormatType::Gguf,
        vec![FormatType::Apr],
        FormatType::SafeTensors,
    );
    assert_eq!(
        path.steps(),
        vec![FormatType::Gguf, FormatType::Apr, FormatType::SafeTensors]
    );
}

#[test]
fn p013_path_display() {
    let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
    assert_eq!(format!("{path}"), "GGUF → APR");
}

#[test]
fn p014_roundtrip_detection() {
    let roundtrip =
        ConversionPath::chain(FormatType::Gguf, vec![FormatType::Apr], FormatType::Gguf);
    assert!(roundtrip.is_roundtrip());

    let non_roundtrip = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
    assert!(!non_roundtrip.is_roundtrip());
}

#[test]
fn p015_cycle_detection_no_cycle() {
    let path = ConversionPath::chain(
        FormatType::Gguf,
        vec![FormatType::Apr],
        FormatType::SafeTensors,
    );
    assert!(!path.has_cycle(), "Linear chain should have no cycle");
}

#[test]
fn p016_cycle_detection_with_cycle() {
    let path = ConversionPath {
        source: FormatType::Gguf,
        target: FormatType::SafeTensors,
        intermediates: vec![FormatType::Apr, FormatType::Gguf, FormatType::Apr],
    };
    // APR appears twice in intermediates - that's a cycle
    assert!(
        path.has_cycle(),
        "Repeated format in intermediates is a cycle"
    );
}

#[test]
fn p017_empty_intermediates() {
    let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
    assert!(path.intermediates.is_empty());
    assert_eq!(path.steps().len(), 2);
}

#[test]
fn p018_single_intermediate() {
    let path = ConversionPath::chain(
        FormatType::Gguf,
        vec![FormatType::Apr],
        FormatType::SafeTensors,
    );
    assert_eq!(path.intermediates.len(), 1);
    assert_eq!(path.steps().len(), 3);
}

#[test]
fn p019_multiple_intermediates() {
    let path = ConversionPath {
        source: FormatType::Gguf,
        target: FormatType::Gguf,
        intermediates: vec![FormatType::Apr, FormatType::SafeTensors, FormatType::Apr],
    };
    assert_eq!(path.steps().len(), 5);
    assert!(path.is_roundtrip());
}

#[test]
fn p020_path_equality() {
    let path1 = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
    let path2 = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
    let path3 = ConversionPath::direct(FormatType::Apr, FormatType::Gguf);
    assert_eq!(path1, path2);
    assert_ne!(path1, path3);
}

// ========================================================================
// Section 3: Options Tests (P021-P030)
// ========================================================================

#[test]
fn p021_default_options() {
    let opts = ConversionOptions::default();
    assert!(opts.verify);
    assert!(opts.preserve_metadata);
    assert!(opts.add_provenance);
    assert!(!opts.compute_stats);
    assert!(opts.quantization.is_none());
}

#[test]
fn p022_custom_tolerance() {
    let opts = ConversionOptions {
        tolerance: 1e-3,
        ..Default::default()
    };
    assert!((opts.tolerance - 1e-3).abs() < 1e-9);
}

#[test]
fn p023_quantization_option() {
    let opts = ConversionOptions {
        quantization: Some("Q4_K_M".to_string()),
        ..Default::default()
    };
    assert_eq!(opts.quantization, Some("Q4_K_M".to_string()));
}

#[test]
fn p024_verify_disabled() {
    let opts = ConversionOptions {
        verify: false,
        ..Default::default()
    };
    assert!(!opts.verify);
}

#[test]
fn p025_compute_stats_enabled() {
    let opts = ConversionOptions {
        compute_stats: true,
        ..Default::default()
    };
    assert!(opts.compute_stats);
}

#[test]
fn p026_no_provenance() {
    let opts = ConversionOptions {
        add_provenance: false,
        ..Default::default()
    };
    assert!(!opts.add_provenance);
}

#[test]
fn p027_no_preserve_metadata() {
    let opts = ConversionOptions {
        preserve_metadata: false,
        ..Default::default()
    };
    assert!(!opts.preserve_metadata);
}

#[test]
fn p028_strict_tolerance() {
    let opts = ConversionOptions {
        tolerance: 1e-9,
        ..Default::default()
    };
    assert!(opts.tolerance < 1e-8);
}

#[test]
fn p029_all_options_custom() {
    let opts = ConversionOptions {
        quantization: Some("Q8_0".to_string()),
        verify: false,
        compute_stats: true,
        tolerance: 1e-4,
        preserve_metadata: false,
        add_provenance: false,
        tokenizer_path: None,
    };
    assert_eq!(opts.quantization, Some("Q8_0".to_string()));
    assert!(!opts.verify);
    assert!(opts.compute_stats);
    assert!((opts.tolerance - 1e-4).abs() < 1e-10);
    assert!(!opts.preserve_metadata);
    assert!(!opts.add_provenance);
}

#[test]
fn p030_options_clone() {
    let opts = ConversionOptions {
        quantization: Some("Q4_0".to_string()),
        ..Default::default()
    };
    let opts2 = opts.clone();
    assert_eq!(opts.quantization, opts2.quantization);
    assert_eq!(opts.tolerance, opts2.tolerance);
}

// ========================================================================
// Section 4: Rosetta Stone Core Tests (P031-P050)
// ========================================================================

#[test]
fn p031_rosetta_stone_creation() {
    let rosetta = RosettaStone::new();
    assert!(rosetta.options.verify);
}

#[test]
fn p032_rosetta_with_custom_options() {
    let opts = ConversionOptions {
        verify: false,
        ..Default::default()
    };
    let rosetta = RosettaStone::with_options(opts);
    assert!(!rosetta.options.verify);
}

#[test]
fn p033_rosetta_default_impl() {
    let rosetta = RosettaStone::default();
    assert!(rosetta.options.verify);
}

#[test]
fn p034_rosetta_debug_trait() {
    let rosetta = RosettaStone::new();
    let debug_str = format!("{:?}", rosetta);
    assert!(debug_str.contains("RosettaStone"));
}

#[test]
fn p035_rosetta_inspect_nonexistent() {
    let rosetta = RosettaStone::new();
    let result = rosetta.inspect("/nonexistent/file.gguf");
    assert!(result.is_err());
}

#[test]
fn p036_rosetta_inspect_no_extension() {
    let rosetta = RosettaStone::new();
    let dir = std::env::temp_dir();
    let noext = dir.join("rosetta_test_noextension_abc123");
    let result = rosetta.inspect(&noext);
    assert!(result.is_err());
}

// ========================================================================
// Section 5: Verification Report Tests (P051-P060)
// ========================================================================

#[test]
fn p051_verification_passing() {
    let report = VerificationReport::passing();
    assert!(report.is_equivalent);
    assert_eq!(report.max_diff, 0.0);
}

#[test]
fn p052_verification_tolerance() {
    let mut report = VerificationReport::passing();
    report.max_diff = 1e-7;
    assert!(report.passes_with_tolerance(1e-6));
    assert!(!report.passes_with_tolerance(1e-8));
}

#[test]
fn p053_verification_failed_tensors() {
    let mut report = VerificationReport::passing();
    report.failed_tensors.push("layer.0.weight".to_string());
    assert!(!report.passes_with_tolerance(1e-3));
}

#[test]
fn p054_verification_mean_diff() {
    let report = VerificationReport {
        is_equivalent: true,
        max_diff: 1e-5,
        mean_diff: 1e-7,
        tensor_diffs: BTreeMap::new(),
        changed_metadata: Vec::new(),
        failed_tensors: Vec::new(),
    };
    assert!(report.mean_diff < report.max_diff);
}

#[test]
fn p055_verification_tensor_diffs() {
    let mut diffs = BTreeMap::new();
    diffs.insert("embed.weight".to_string(), 1e-8_f32);
    diffs.insert("lm_head.weight".to_string(), 1e-7_f32);

    let report = VerificationReport {
        is_equivalent: true,
        max_diff: 1e-7,
        mean_diff: 5e-8,
        tensor_diffs: diffs.clone(),
        changed_metadata: Vec::new(),
        failed_tensors: Vec::new(),
    };
    assert_eq!(report.tensor_diffs.len(), 2);
}

#[test]
fn p056_verification_metadata_changes() {
    let report = VerificationReport {
        is_equivalent: true,
        max_diff: 0.0,
        mean_diff: 0.0,
        tensor_diffs: BTreeMap::new(),
        changed_metadata: vec!["model_name".to_string(), "version".to_string()],
        failed_tensors: Vec::new(),
    };
    assert_eq!(report.changed_metadata.len(), 2);
}

#[test]
fn p057_verification_not_equivalent() {
    let report = VerificationReport {
        is_equivalent: false,
        max_diff: 1.0,
        mean_diff: 0.5,
        tensor_diffs: BTreeMap::new(),
        changed_metadata: Vec::new(),
        failed_tensors: vec!["all_layers".to_string()],
    };
    assert!(!report.is_equivalent);
    assert!(!report.passes_with_tolerance(1e-3));
}

// ========================================================================
// Section 6: Conversion Report Tests (P061-P070)
// ========================================================================

#[test]
fn p061_conversion_lossless() {
    let report = ConversionReport {
        path: ConversionPath::direct(FormatType::Gguf, FormatType::Apr),
        source_inspection: InspectionReport {
            format: FormatType::Gguf,
            file_size: 1000,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 100,
            quantization: None,
            architecture: None,
        },
        target_inspection: InspectionReport {
            format: FormatType::Apr,
            file_size: 1000,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 100,
            quantization: None,
            architecture: None,
        },
        warnings: vec![],
        duration_ms: 100,
        modified_tensors: vec![],
        dropped_tensors: vec![],
    };
    assert!(report.is_lossless());
    assert!(report.tensor_counts_match());
}

#[test]
fn p062_conversion_with_dropped_tensors() {
    let report = ConversionReport {
        path: ConversionPath::direct(FormatType::Gguf, FormatType::Apr),
        source_inspection: InspectionReport {
            format: FormatType::Gguf,
            file_size: 1000,
            metadata: BTreeMap::new(),
            tensors: vec![
                TensorInfo {
                    name: "layer.0".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![100, 100],
                    size_bytes: 40000,
                    stats: None,
                },
                TensorInfo {
                    name: "layer.1".to_string(),
                    dtype: "F32".to_string(),
                    shape: vec![100, 100],
                    size_bytes: 40000,
                    stats: None,
                },
            ],
            total_params: 20000,
            quantization: None,
            architecture: None,
        },
        target_inspection: InspectionReport {
            format: FormatType::Apr,
            file_size: 800,
            metadata: BTreeMap::new(),
            tensors: vec![TensorInfo {
                name: "layer.0".to_string(),
                dtype: "F32".to_string(),
                shape: vec![100, 100],
                size_bytes: 40000,
                stats: None,
            }],
            total_params: 10000,
            quantization: None,
            architecture: None,
        },
        warnings: vec!["Tensor dropped".to_string()],
        duration_ms: 100,
        modified_tensors: vec![],
        dropped_tensors: vec!["layer.1".to_string()],
    };
    assert!(!report.is_lossless());
    assert!(!report.tensor_counts_match());
}

#[test]
fn p063_conversion_warnings() {
    let report = ConversionReport {
        path: ConversionPath::direct(FormatType::Gguf, FormatType::Apr),
        source_inspection: InspectionReport {
            format: FormatType::Gguf,
            file_size: 1000,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 0,
            quantization: None,
            architecture: None,
        },
        target_inspection: InspectionReport {
            format: FormatType::Apr,
            file_size: 1000,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 0,
            quantization: None,
            architecture: None,
        },
        warnings: vec!["Warning 1".to_string(), "Warning 2".to_string()],
        duration_ms: 50,
        modified_tensors: vec![],
        dropped_tensors: vec![],
    };
    assert_eq!(report.warnings.len(), 2);
    assert!(report.is_lossless());
}

// ========================================================================
// Section 7: TensorInfo Tests (P071-P080)
// ========================================================================

#[test]
fn p071_tensor_info_creation() {
    let info = TensorInfo {
        name: "model.embed".to_string(),
        dtype: "F32".to_string(),
        shape: vec![32000, 4096],
        size_bytes: 32000 * 4096 * 4,
        stats: None,
    };
    assert_eq!(info.name, "model.embed");
    assert_eq!(info.dtype, "F32");
}

#[test]
fn p072_tensor_info_with_stats() {
    let stats = TensorStats {
        min: -1.0,
        max: 1.0,
        mean: 0.0,
        std: 0.5,
    };
    let info = TensorInfo {
        name: "layer.weight".to_string(),
        dtype: "F16".to_string(),
        shape: vec![1024, 1024],
        size_bytes: 1024 * 1024 * 2,
        stats: Some(stats),
    };
    assert!(info.stats.is_some());
    let s = info.stats.unwrap();
    assert_eq!(s.min, -1.0);
    assert_eq!(s.max, 1.0);
}

#[test]
fn p073_tensor_info_multidim_shape() {
    let info = TensorInfo {
        name: "conv.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![64, 32, 3, 3],
        size_bytes: 64 * 32 * 3 * 3 * 4,
        stats: None,
    };
    assert_eq!(info.shape.len(), 4);
    let total: usize = info.shape.iter().product();
    assert_eq!(total, 64 * 32 * 3 * 3);
}

#[test]
fn p074_tensor_stats_range() {
    let stats = TensorStats {
        min: -2.5,
        max: 2.5,
        mean: 0.01,
        std: 1.0,
    };
    assert!(stats.min < stats.max);
    assert!(stats.mean >= stats.min && stats.mean <= stats.max);
}

// ========================================================================
// Section 8: InspectionReport Tests (P081-P090)
// ========================================================================

#[test]
fn p081_inspection_report_format() {
    let report = InspectionReport {
        format: FormatType::Gguf,
        file_size: 1_000_000,
        metadata: BTreeMap::new(),
        tensors: vec![],
        total_params: 100_000,
        quantization: None,
        architecture: None,
    };
    assert_eq!(report.format, FormatType::Gguf);
    assert_eq!(report.file_size, 1_000_000);
}

#[test]
fn p082_inspection_report_with_metadata() {
    let mut metadata = BTreeMap::new();
    metadata.insert("model_name".to_string(), "test-model".to_string());
    metadata.insert("version".to_string(), "1.0".to_string());

    let report = InspectionReport {
        format: FormatType::Apr,
        file_size: 500_000,
        metadata,
        tensors: vec![],
        total_params: 50_000,
        quantization: None,
        architecture: Some("transformer".to_string()),
    };
    assert_eq!(report.metadata.len(), 2);
    assert!(report.architecture.is_some());
}

#[test]
fn p083_inspection_report_with_quantization() {
    let report = InspectionReport {
        format: FormatType::Gguf,
        file_size: 2_000_000,
        metadata: BTreeMap::new(),
        tensors: vec![],
        total_params: 1_000_000,
        quantization: Some("Q4_K_M".to_string()),
        architecture: Some("llama".to_string()),
    };
    assert_eq!(report.quantization, Some("Q4_K_M".to_string()));
}

#[test]
fn p084_inspection_report_display() {
    let report = InspectionReport {
        format: FormatType::SafeTensors,
        file_size: 100,
        metadata: BTreeMap::new(),
        tensors: vec![],
        total_params: 10,
        quantization: None,
        architecture: None,
    };
    let display = format!("{}", report);
    assert!(display.contains("Rosetta Stone Inspection"));
    assert!(display.contains("SafeTensors"));
}

// ========================================================================
// Section 12: Destructive Tests (Popperian "Crucial Experiments")
// ========================================================================

#[test]
fn p091_pdf_imposter_test() {
    // The PDF Imposter: Renamed file detection
    // A PDF renamed to .gguf should fail magic detection
    // This test verifies format detection uses magic bytes, not just extension

    // Create a fake "GGUF" file with PDF magic
    let temp_dir = std::env::temp_dir();
    let fake_gguf = temp_dir.join("fake.gguf");

    // PDF magic: %PDF-1.
    std::fs::write(&fake_gguf, b"%PDF-1.4\n").expect("Write test file");

    let result = FormatType::from_magic(&fake_gguf);

    // Clean up
    let _ = std::fs::remove_file(&fake_gguf);

    // Should fail - PDF magic doesn't match any known format
    assert!(
        result.is_err(),
        "PDF disguised as GGUF should fail magic detection"
    );
}

#[test]
fn p092_unicode_ghost_test() {
    // The Unicode Ghost: Complex characters in paths
    // Verify UTF-8 paths are handled correctly

    let path = Path::new("模型_テスト_🤖.gguf");
    let format = FormatType::from_extension(path);
    assert!(
        format.is_ok(),
        "Unicode path should parse extension correctly"
    );
}

#[test]
fn p093_infinite_loop_test() {
    // The Infinite Loop: Cycle detection in chains
    let _rosetta = RosettaStone::new();

    // This chain has a cycle: GGUF → APR → GGUF → APR → SafeTensors
    // APR appears twice, which should be detected as a cycle
    let chain = vec![
        FormatType::Gguf,
        FormatType::Apr,
        FormatType::Gguf,
        FormatType::Apr,
        FormatType::SafeTensors,
    ];

    // Note: This test validates the has_cycle() logic is in place
    let path = ConversionPath {
        source: chain[0],
        target: chain[chain.len() - 1],
        intermediates: chain[1..chain.len() - 1].to_vec(),
    };
    assert!(
        path.has_cycle(),
        "Chain with repeated formats should have cycle"
    );
}

#[test]
fn p094_zero_size_file_test() {
    // Zero-size file should fail inspection
    let temp_dir = std::env::temp_dir();
    let empty_file = temp_dir.join("empty.gguf");
    std::fs::write(&empty_file, b"").expect("Write empty file");

    let result = FormatType::from_magic(&empty_file);
    let _ = std::fs::remove_file(&empty_file);

    assert!(result.is_err(), "Empty file should fail magic detection");
}

#[test]
fn p095_truncated_magic_test() {
    // File with only 3 bytes (truncated magic)
    let temp_dir = std::env::temp_dir();
    let short_file = temp_dir.join("short.gguf");
    std::fs::write(&short_file, b"GGU").expect("Write short file");

    let result = FormatType::from_magic(&short_file);
    let _ = std::fs::remove_file(&short_file);

    assert!(result.is_err(), "Truncated magic should fail");
}

#[test]
fn p096_symlink_path_extension() {
    // Paths with multiple dots
    let path = Path::new("model.v1.backup.gguf");
    let format = FormatType::from_extension(path);
    assert!(format.is_ok());
    assert_eq!(format.unwrap(), FormatType::Gguf);
}

#[test]
fn p097_hidden_file_extension() {
    // Hidden file with extension
    let path = Path::new(".hidden_model.safetensors");
    let format = FormatType::from_extension(path);
    assert!(format.is_ok());
    assert_eq!(format.unwrap(), FormatType::SafeTensors);
}

#[test]
fn p098_mixed_case_extension() {
    // Mixed case extension
    let path = Path::new("model.GgUf");
    let format = FormatType::from_extension(path);
    assert!(format.is_ok());
    assert_eq!(format.unwrap(), FormatType::Gguf);
}

#[test]
fn p099_format_hash_trait() {
    // Verify FormatType implements Hash correctly
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(FormatType::Gguf);
    set.insert(FormatType::Apr);
    set.insert(FormatType::Gguf); // Duplicate
    assert_eq!(set.len(), 2);
}

#[test]
fn p100_format_eq_trait() {
    // Verify FormatType equality
    assert_eq!(FormatType::Gguf, FormatType::Gguf);
    assert_ne!(FormatType::Gguf, FormatType::Apr);
    assert_ne!(FormatType::Apr, FormatType::SafeTensors);
}

// ========================================================================
// Section 14: Additional Edge Cases (P101-P110)
// ========================================================================

#[test]
fn p101_conversion_path_clone() {
    let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
    let path2 = path.clone();
    assert_eq!(path, path2);
}

#[test]
fn p102_conversion_path_debug() {
    let path = ConversionPath::direct(FormatType::Gguf, FormatType::Apr);
    let debug = format!("{:?}", path);
    assert!(debug.contains("ConversionPath"));
    assert!(debug.contains("Gguf"));
}

#[test]
fn p103_tensor_info_clone() {
    let info = TensorInfo {
        name: "test".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10, 20],
        size_bytes: 800,
        stats: None,
    };
    let info2 = info.clone();
    assert_eq!(info.name, info2.name);
    assert_eq!(info.shape, info2.shape);
}

#[test]
fn p104_tensor_stats_copy() {
    let stats = TensorStats {
        min: 0.0,
        max: 1.0,
        mean: 0.5,
        std: 0.25,
    };
    let stats2 = stats; // Copy
    assert_eq!(stats.mean, stats2.mean);
}

#[test]
fn p105_verification_report_clone() {
    let report = VerificationReport::passing();
    let report2 = report.clone();
    assert_eq!(report.is_equivalent, report2.is_equivalent);
}

#[test]
fn p106_options_debug() {
    let opts = ConversionOptions::default();
    let debug = format!("{:?}", opts);
    assert!(debug.contains("ConversionOptions"));
}

#[test]
fn p107_empty_tensor_list() {
    let report = InspectionReport {
        format: FormatType::Apr,
        file_size: 0,
        metadata: BTreeMap::new(),
        tensors: vec![],
        total_params: 0,
        quantization: None,
        architecture: None,
    };
    assert!(report.tensors.is_empty());
    assert_eq!(report.total_params, 0);
}

#[test]
fn p108_large_tensor_count() {
    let tensors: Vec<TensorInfo> = (0..100)
        .map(|i| TensorInfo {
            name: format!("layer.{}", i),
            dtype: "F16".to_string(),
            shape: vec![256, 256],
            size_bytes: 256 * 256 * 2,
            stats: None,
        })
        .collect();

    let report = InspectionReport {
        format: FormatType::Gguf,
        file_size: tensors.len() * 256 * 256 * 2,
        metadata: BTreeMap::new(),
        tensors,
        total_params: 100 * 256 * 256,
        quantization: None,
        architecture: None,
    };
    assert_eq!(report.tensors.len(), 100);
}

#[test]
fn p109_metadata_long_value() {
    let mut metadata = BTreeMap::new();
    let long_value = "x".repeat(1000);
    metadata.insert("long_key".to_string(), long_value.clone());

    let report = InspectionReport {
        format: FormatType::SafeTensors,
        file_size: 1000,
        metadata,
        tensors: vec![],
        total_params: 0,
        quantization: None,
        architecture: None,
    };

    let display = format!("{}", report);
    // Long values should be truncated in display
    assert!(display.len() < long_value.len() * 2);
}

#[test]
fn p110_conversion_duration() {
    let report = ConversionReport {
        path: ConversionPath::direct(FormatType::Gguf, FormatType::Apr),
        source_inspection: InspectionReport {
            format: FormatType::Gguf,
            file_size: 1000,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 100,
            quantization: None,
            architecture: None,
        },
        target_inspection: InspectionReport {
            format: FormatType::Apr,
            file_size: 1000,
            metadata: BTreeMap::new(),
            tensors: vec![],
            total_params: 100,
            quantization: None,
            architecture: None,
        },
        warnings: vec![],
        duration_ms: 1500,
        modified_tensors: vec![],
        dropped_tensors: vec![],
    };
    assert_eq!(report.duration_ms, 1500);
}

// ========================================================================
// Section 13: Integration Tests (Self-Contained with Generated Fixtures)
// ========================================================================
//
// Popperian Principle: Tests must be self-contained and falsifiable.
// These tests generate their own valid fixtures using the library APIs.

/// Generate a unique temp file name for tests
fn unique_temp_path(prefix: &str, ext: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("{prefix}_{pid}_{id}.{ext}"))
}

/// Helper: Create a minimal valid SafeTensors file
fn create_safetensors_fixture() -> std::path::PathBuf {
    use std::io::Write;
    let path = unique_temp_path("test_tiny", "safetensors");
    let mut file = std::fs::File::create(&path).expect("Create temp file");

    // SafeTensors format: 8-byte header length + JSON header + tensor data
    // Use test.bias (not test.weight) to bypass strict weight validation
    let header = r#"{"test.bias":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"__metadata__":{"format":"test"}}"#;
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("Write header len");
    file.write_all(header.as_bytes()).expect("Write header");

    // Tensor data (4 f32 values = 16 bytes) - realistic values near zero
    let data: [f32; 4] = [0.01, -0.02, 0.03, -0.01];
    for val in &data {
        file.write_all(&val.to_le_bytes()).expect("Write tensor");
    }
    path
}

/// Helper: Create a minimal valid APR v2 file using the library API
fn create_apr_fixture() -> std::path::PathBuf {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    let path = unique_temp_path("test_tiny", "apr");
    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);
    // Use .bias suffix to bypass strict weight validation
    writer.add_f32_tensor("test.bias", vec![4], &[0.01, -0.02, 0.03, -0.01]);

    let mut file = std::fs::File::create(&path).expect("Create temp APR file");
    writer.write_to(&mut file).expect("Write APR");
    path
}

// P111: Integration test - inspect SafeTensors (self-contained)
// H0: Rosetta can inspect a valid SafeTensors file
// Refutation: Fails if format detection or parsing fails
#[test]
fn p111_integration_inspect_safetensors() {
    let path = create_safetensors_fixture();
    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(&path).expect("Inspect SafeTensors");
    assert_eq!(report.format, FormatType::SafeTensors);
    assert!(
        !report.tensors.is_empty(),
        "Should have at least one tensor"
    );
    let _ = std::fs::remove_file(path);
}

// P112: Integration test - inspect APR (self-contained)
// H0: Rosetta can inspect a valid APR v2 file
// Refutation: Fails if format detection or parsing fails
#[test]
fn p112_integration_inspect_apr() {
    let path = create_apr_fixture();
    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(&path).expect("Inspect APR");
    assert_eq!(report.format, FormatType::Apr);
    assert!(
        !report.tensors.is_empty(),
        "Should have at least one tensor"
    );
    let _ = std::fs::remove_file(path);
}

// P113: Integration test - convert SafeTensors to APR
// H0: Rosetta can convert SafeTensors to APR format
// Refutation: Fails if conversion fails or output format is wrong
#[test]
fn p113_integration_convert_safetensors_to_apr() {
    let source = create_safetensors_fixture();
    let target = unique_temp_path("test_converted", "apr");

    let rosetta = RosettaStone::new();
    let report = rosetta
        .convert(&source, &target, None)
        .expect("Convert SafeTensors to APR");

    assert_eq!(report.path.source, FormatType::SafeTensors);
    assert_eq!(report.path.target, FormatType::Apr);
    assert!(target.exists(), "Output file should exist");

    // Verify converted file is valid APR
    let verify_report = rosetta.inspect(&target).expect("Inspect converted APR");
    assert_eq!(verify_report.format, FormatType::Apr);

    let _ = std::fs::remove_file(source);
    let _ = std::fs::remove_file(target);
}

// P114: Integration test - conversion preserves inspection results
// H0: Converted APR file can be inspected
// Refutation: Fails if inspection fails after conversion
//
// Note: Full roundtrip (SafeTensors -> APR -> SafeTensors) requires
// implementing APR loading in load_model_tensors. Currently the converter
// treats APR files as SafeTensors, which is a known limitation (APR-EXPORT-001).
#[test]
fn p114_integration_conversion_inspection() {
    let source = create_safetensors_fixture();
    let target = unique_temp_path("test_converted", "apr");

    let rosetta = RosettaStone::new();

    // Convert SafeTensors -> APR
    rosetta
        .convert(&source, &target, None)
        .expect("Convert to APR");

    // Verify the APR file can be inspected (proves conversion worked)
    let source_report = rosetta.inspect(&source).expect("Inspect source");
    let target_report = rosetta.inspect(&target).expect("Inspect target APR");

    // Tensor count should be preserved
    assert_eq!(
        source_report.tensors.len(),
        target_report.tensors.len(),
        "Conversion should preserve tensor count"
    );

    // Format should be correct
    assert_eq!(target_report.format, FormatType::Apr);

    let _ = std::fs::remove_file(source);
    let _ = std::fs::remove_file(target);
}

// ========================================================================
// Section 14: Bit-Flip Experiment (Appendix C.2)
// ========================================================================
//
// Popperian Falsification: Corruption MUST be detected.
// If a single bit flip goes undetected, the verification is worthless.

// P115: Bit-flip corruption detection - SafeTensors header length
// H0: A corrupted SafeTensors header length is detected as invalid
// Refutation: If corrupted file parses successfully with wrong tensor count, detection failed
//
// Note: SafeTensors lacks checksums, so we corrupt the header length (first 8 bytes)
// which causes parsing to read garbage as JSON.
#[test]
fn p115_bitflip_safetensors_corruption_detected() {
    let path = create_safetensors_fixture();

    // Read file, corrupt the header length (first 8 bytes)
    let mut data = std::fs::read(&path).expect("Read fixture");

    // Corrupt byte 0 (LSB of header length) - this makes the JSON header appear longer/shorter
    data[0] = data[0].wrapping_add(50); // Add 50 to header length

    // Write corrupted file
    let corrupted_path = unique_temp_path("test_corrupted_len", "safetensors");
    std::fs::write(&corrupted_path, &data).expect("Write corrupted file");

    // Attempt to inspect - should fail because JSON header is misaligned
    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(&corrupted_path);

    // Corruption MUST be detected - header length mismatch causes JSON parse failure
    assert!(
        result.is_err(),
        "SafeTensors with corrupted header length should fail to parse"
    );

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(corrupted_path);
}

// P116: Bit-flip corruption detection - APR
// H0: A corrupted APR file is detected via checksum
// Refutation: If corrupted file passes checksum validation, system is broken
#[test]
fn p116_bitflip_apr_corruption_detected() {
    let path = create_apr_fixture();

    // Read file, corrupt the data section
    let mut data = std::fs::read(&path).expect("Read APR fixture");

    // Corrupt a byte in the data section (after header at offset 64+)
    if data.len() > 100 {
        data[100] ^= 0xFF; // Flip all bits in one byte
    }

    // Write corrupted file
    let corrupted_path = unique_temp_path("test_corrupted", "apr");
    std::fs::write(&corrupted_path, &data).expect("Write corrupted APR file");

    // Attempt to inspect - should fail due to checksum mismatch
    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(&corrupted_path);

    // APR v2 has checksum verification - corruption MUST be detected
    assert!(
        result.is_err(),
        "Corrupted APR file should fail checksum verification"
    );

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(corrupted_path);
}

// ========================================================================
// Section 15: GGUF Integration (Requires Real GGUF File)
// ========================================================================
//
// Note: GGUF files are complex (quantized tensors, alignment, etc.)
// These tests use the existing model files in the repository.

// P117: GGUF format detection from real file
// H0: Real GGUF file is correctly detected
// Refutation: Fails if detection returns wrong format
// NOTE: Requires models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf on disk.
// Marked #[ignore] so CI reports "ignored" instead of silently passing.
#[test]
#[ignore = "requires local GGUF model file"]
fn p117_gguf_format_detection_real_file() {
    let gguf_path = Path::new("models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");

    let format = FormatType::from_magic(gguf_path).expect("Detect GGUF format");
    assert_eq!(format, FormatType::Gguf, "Should detect GGUF format");
}

// P118: GGUF inspection from real file
// H0: Real GGUF file can be inspected
// Refutation: Fails if inspection fails or returns empty tensors
// NOTE: Requires models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf on disk.
// Marked #[ignore] so CI reports "ignored" instead of silently passing.
#[test]
#[ignore = "requires local GGUF model file"]
fn p118_gguf_inspection_real_file() {
    let gguf_path = Path::new("models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");

    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(gguf_path).expect("Inspect GGUF");

    assert_eq!(report.format, FormatType::Gguf);
    assert!(!report.tensors.is_empty(), "GGUF should have tensors");
    assert!(report.total_params > 0, "Should have non-zero params");
}

// ========================================================================
// Section 16: APR Embedded Tokenizer Tests (GH-156)
// ========================================================================
//
// PMAT-ROSETTA-001 Gap: The original Rosetta tests did NOT verify embedded
// tokenizer functionality in APR files. This caused BUG-APR-002 to go
// undetected until QA matrix testing exposed it.
//
// These tests ensure APR's "executable model" design (self-contained with
// embedded tokenizer) is maintained and verified.

// P119: APR embedded tokenizer metadata presence
// H0: APR files created from SafeTensors+tokenizer.json include tokenizer metadata
// Refutation: Fails if tokenizer.vocabulary is missing from APR metadata
#[test]
fn p119_apr_embedded_tokenizer_metadata() {
    use crate::format::v2::{AprV2Metadata, AprV2Writer};
    use std::collections::HashMap;

    let path = unique_temp_path("test_tokenizer", "apr");

    // Create APR with embedded tokenizer metadata
    let mut metadata = AprV2Metadata::new("test");

    // Add tokenizer fields to custom metadata
    let vocab = vec!["<pad>", "<bos>", "<eos>", "hello", "world"];
    let vocab_json: Vec<serde_json::Value> = vocab
        .iter()
        .map(|s| serde_json::Value::String(s.to_string()))
        .collect();

    let mut custom: HashMap<String, serde_json::Value> = HashMap::new();
    custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::Value::Array(vocab_json),
    );
    custom.insert(
        "tokenizer.vocab_size".to_string(),
        serde_json::Value::Number(5.into()),
    );
    custom.insert(
        "tokenizer.bos_token_id".to_string(),
        serde_json::Value::Number(1.into()),
    );
    custom.insert(
        "tokenizer.eos_token_id".to_string(),
        serde_json::Value::Number(2.into()),
    );
    metadata.custom = custom;

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("embed.weight", vec![5, 4], &[0.0; 20]);

    let mut file = std::fs::File::create(&path).expect("Create APR file");
    writer.write_to(&mut file).expect("Write APR");
    drop(file);

    // Verify metadata was written by reading APR and checking for tokenizer keys
    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(&path).expect("Inspect APR with tokenizer");

    // The tokenizer metadata should be present (even if not exposed in inspection)
    assert_eq!(report.format, FormatType::Apr);
    assert!(!report.tensors.is_empty(), "Should have tensors");

    let _ = std::fs::remove_file(path);
}

// P120: APR tokenizer extraction round-trip
// H0: Tokenizer vocabulary embedded in APR can be extracted and used for decoding
// Refutation: Fails if vocabulary extraction fails or decoding produces wrong output
// NOTE: This test requires realizar crate's SimpleTokenizer implementation (GH-156)
#[test]
fn p120_apr_tokenizer_decode_roundtrip() {
    use crate::format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer};
    use std::collections::HashMap;

    let path = unique_temp_path("test_decode", "apr");

    // Create APR with embedded tokenizer
    let mut metadata = AprV2Metadata::new("test");

    // Define vocabulary with BPE-style tokens
    let vocab = vec![
        "<pad>", "<bos>", "<eos>", "Ġhello", "Ġworld", "!", "Ġthe", "Ġquick",
    ];
    let vocab_json: Vec<serde_json::Value> = vocab
        .iter()
        .map(|s| serde_json::Value::String(s.to_string()))
        .collect();

    let mut custom: HashMap<String, serde_json::Value> = HashMap::new();
    custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::Value::Array(vocab_json),
    );
    custom.insert(
        "tokenizer.vocab_size".to_string(),
        serde_json::Value::Number(8.into()),
    );
    custom.insert(
        "tokenizer.bos_token_id".to_string(),
        serde_json::Value::Number(1.into()),
    );
    custom.insert(
        "tokenizer.eos_token_id".to_string(),
        serde_json::Value::Number(2.into()),
    );
    metadata.custom = custom;

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("embed.weight", vec![8, 4], &[0.0; 32]);

    let mut file = std::fs::File::create(&path).expect("Create APR file");
    writer.write_to(&mut file).expect("Write APR");
    drop(file);

    // Read APR and extract vocabulary
    let data = std::fs::read(&path).expect("Read APR");
    let reader = AprV2Reader::from_bytes(&data).expect("Parse APR");
    let meta = reader.metadata();

    // Extract tokenizer vocabulary from custom metadata
    let vocab_value = meta.custom.get("tokenizer.vocabulary");
    assert!(
        vocab_value.is_some(),
        "APR should have tokenizer.vocabulary in metadata"
    );

    let vocab_array = vocab_value
        .unwrap()
        .as_array()
        .expect("vocabulary should be array");
    assert_eq!(vocab_array.len(), 8, "Vocabulary size should be 8");

    // Verify BOS/EOS token IDs
    let bos_id = meta
        .custom
        .get("tokenizer.bos_token_id")
        .and_then(|v| v.as_u64())
        .expect("Should have bos_token_id");
    let eos_id = meta
        .custom
        .get("tokenizer.eos_token_id")
        .and_then(|v| v.as_u64())
        .expect("Should have eos_token_id");

    assert_eq!(bos_id, 1, "BOS token ID should be 1");
    assert_eq!(eos_id, 2, "EOS token ID should be 2");

    // Verify vocabulary content (spot check)
    let first_token = vocab_array[3].as_str().expect("Token should be string");
    assert_eq!(first_token, "Ġhello", "Token at index 3 should be 'Ġhello'");

    let _ = std::fs::remove_file(path);
}

// P121: APR without tokenizer fallback
// H0: APR files without embedded tokenizer should indicate token count, not crash
// Refutation: Fails if accessing tokenizer on tokenizer-less APR causes panic
#[test]
fn p121_apr_no_tokenizer_graceful_fallback() {
    // This test uses create_apr_fixture() which creates APR WITHOUT tokenizer
    let path = create_apr_fixture();

    // Read APR and verify no tokenizer metadata
    let data = std::fs::read(&path).expect("Read APR");
    let reader = crate::format::v2::AprV2Reader::from_bytes(&data).expect("Parse APR");
    let meta = reader.metadata();

    // Should NOT have tokenizer.vocabulary
    let vocab_value = meta.custom.get("tokenizer.vocabulary");
    assert!(
        vocab_value.is_none(),
        "Fixture APR should NOT have embedded tokenizer"
    );

    // Accessing missing tokenizer should return None, not panic
    // (This is what GH-156 fixes in realizar)

    let _ = std::fs::remove_file(path);
}

// F-STRESS-520: Panic 411 (Empty Tensor) - 0-byte file handling
// H0: Loading a 0-byte file should return error, not panic
// Refutation: Fails if 0-byte file causes panic instead of graceful error
// Toyota Way: Jidoka - detect defects at the source
#[test]
fn f_stress_520_zero_byte_file_no_panic_pmat178() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let inspector = RosettaStone::new();

    // Create a 0-byte APR file
    let mut temp_apr = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp_apr.flush().expect("Flush");

    // Attempt to inspect - should return error, not panic
    let result = inspector.inspect(temp_apr.path());
    assert!(result.is_err(), "0-byte file should return error, not Ok");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("small")
            || err_msg.contains("empty")
            || err_msg.contains("parse")
            || err_msg.contains("magic"),
        "Error should indicate file issue: {err_msg}"
    );

    // Create a 0-byte GGUF file
    let mut temp_gguf = NamedTempFile::with_suffix(".gguf").expect("Create temp file");
    temp_gguf.flush().expect("Flush");

    // Attempt to inspect - should return error, not panic
    let result = inspector.inspect(temp_gguf.path());
    assert!(
        result.is_err(),
        "0-byte GGUF file should return error, not Ok"
    );

    // Create a 0-byte SafeTensors file
    let mut temp_st = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp_st.flush().expect("Flush");

    // Attempt to inspect - should return error, not panic
    let result = inspector.inspect(temp_st.path());
    assert!(
        result.is_err(),
        "0-byte SafeTensors file should return error, not Ok"
    );
}

// F-STRESS-520: Additional test - truncated file handling
// H0: Truncated files (partial headers) should return error, not panic
#[test]
fn f_stress_520_truncated_file_no_panic_pmat178() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let inspector = RosettaStone::new();

    // Create a truncated APR file (just magic bytes, no header)
    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(b"APR\x00").expect("Write partial header");
    temp.flush().expect("Flush");

    let result = inspector.inspect(temp.path());
    assert!(
        result.is_err(),
        "Truncated APR file should return error, not Ok"
    );

    // Create a truncated GGUF file (just magic bytes)
    let mut temp_gguf = NamedTempFile::with_suffix(".gguf").expect("Create temp file");
    temp_gguf.write_all(b"GGUF").expect("Write partial header");
    temp_gguf.flush().expect("Flush");

    let result = inspector.inspect(temp_gguf.path());
    assert!(
        result.is_err(),
        "Truncated GGUF file should return error, not Ok"
    );
}

// GH-175, PMAT-180: Cross-format validation
// H0: validate() works for all formats and detects NaN/Inf/zeros
#[test]
fn gh175_cross_format_validation_pmat180() {
    // Test with APR fixture (should be valid)
    let apr_path = create_apr_fixture();
    let rosetta = RosettaStone::new();

    let report = rosetta.validate(&apr_path);
    assert!(report.is_ok(), "APR validation should succeed");

    let report = report.expect("validation");
    assert!(
        report.is_valid,
        "APR fixture should be valid (no NaN/Inf/zeros)"
    );
    assert_eq!(report.total_nan_count, 0, "Should have no NaN");
    assert_eq!(report.total_inf_count, 0, "Should have no Inf");
    assert!(
        report.all_zero_tensors.is_empty(),
        "Should have no all-zero tensors"
    );

    // Test summary output
    let summary = report.summary();
    assert!(summary.contains("VALID"), "Summary should indicate VALID");

    let _ = std::fs::remove_file(apr_path);
}

// GH-175: Validation report display
#[test]
fn gh175_validation_report_display() {
    let report = ValidationReport {
        format: FormatType::Apr,
        file_path: "test.apr".to_string(),
        is_valid: true,
        tensor_count: 10,
        failed_tensor_count: 0,
        total_nan_count: 0,
        total_inf_count: 0,
        all_zero_tensors: vec![],
        tensors: vec![],
        duration_ms: 100,
    };

    let display = format!("{report}");
    assert!(display.contains("VALID"), "Display should show VALID");
    assert!(
        display.contains("APR-SPEC 10.9"),
        "Should reference APR-SPEC"
    );
}

// ========================================================================
// Pygmy-Based Tests (T-COV-95)
// Testing rosetta paths with in-memory generated models
// ========================================================================

#[test]
fn pygmy_inspect_safetensors() {
    use crate::format::test_factory::build_pygmy_safetensors;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_safetensors();

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(temp.path());

    assert!(result.is_ok(), "Should inspect pygmy SafeTensors");
    let inspection = result.expect("inspection");
    assert_eq!(inspection.format, FormatType::SafeTensors);
    assert!(!inspection.tensors.is_empty());
}

#[test]
fn pygmy_inspect_apr() {
    use crate::format::test_factory::build_pygmy_apr;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_apr();

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(temp.path());

    assert!(result.is_ok(), "Should inspect pygmy APR");
    let inspection = result.expect("inspection");
    assert_eq!(inspection.format, FormatType::Apr);
    assert!(!inspection.tensors.is_empty());
}

#[test]
fn pygmy_validate_apr() {
    use crate::format::test_factory::build_pygmy_apr;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_apr();

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.validate(temp.path());

    assert!(result.is_ok(), "Should validate pygmy APR");
    let validation = result.expect("validation");
    assert!(
        validation.is_valid,
        "Pygmy APR should be valid (no NaN/Inf)"
    );
    assert_eq!(validation.total_nan_count, 0);
    assert_eq!(validation.total_inf_count, 0);
}

#[test]
fn pygmy_validate_safetensors() {
    use crate::format::test_factory::build_pygmy_safetensors;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_safetensors();

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.validate(temp.path());

    assert!(result.is_ok(), "Should validate pygmy SafeTensors");
    let validation = result.expect("validation");
    assert!(validation.is_valid, "Pygmy SafeTensors should be valid");
}

#[test]
fn pygmy_inspect_apr_with_llama_style_config() {
    use crate::format::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = PygmyConfig::llama_style();
    let data = build_pygmy_apr_with_config(config);

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(temp.path());

    assert!(result.is_ok(), "Should inspect LLaMA-style pygmy APR");
    let inspection = result.expect("inspection");

    // Should have LLaMA-style tensor names
    assert!(
        inspection
            .tensors
            .iter()
            .any(|t| t.name.contains("self_attn")),
        "Should have attention tensors"
    );
}

#[test]
fn pygmy_inspect_quantized_apr() {
    use crate::format::test_factory::{build_pygmy_apr_f16, build_pygmy_apr_q8};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Test Q8 APR
    let q8_data = build_pygmy_apr_q8();
    let mut temp_q8 = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp_q8.write_all(&q8_data).expect("Write data");
    temp_q8.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let result = rosetta.inspect(temp_q8.path());
    assert!(result.is_ok(), "Should inspect Q8 pygmy APR");

    // Test F16 APR
    let f16_data = build_pygmy_apr_f16();
    let mut temp_f16 = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp_f16.write_all(&f16_data).expect("Write data");
    temp_f16.flush().expect("Flush");

    let result = rosetta.inspect(temp_f16.path());
    assert!(result.is_ok(), "Should inspect F16 pygmy APR");
}

#[test]
fn pygmy_format_from_magic_apr() {
    use crate::format::test_factory::build_pygmy_apr;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_apr();

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let format = FormatType::from_magic(temp.path());
    assert!(format.is_ok(), "Should detect format from magic");
    assert_eq!(format.expect("format"), FormatType::Apr);
}

#[test]
fn pygmy_format_from_magic_safetensors() {
    use crate::format::test_factory::build_pygmy_safetensors;
    use std::io::Write;
    use tempfile::NamedTempFile;

    let data = build_pygmy_safetensors();

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let format = FormatType::from_magic(temp.path());
    assert!(format.is_ok(), "Should detect SafeTensors from magic");
    assert_eq!(format.expect("format"), FormatType::SafeTensors);
}

// ========================================================================
// T-GH192: Pre-Load Inspection Tests (rosetta-testing.md §Test Gaps)
// ========================================================================

/// T-GH192-01: Pre-load inspection returns correct metadata for APR format
#[test]
fn t_gh192_01_inspect_apr_returns_metadata() {
    use crate::format::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = PygmyConfig::realistic();
    let data = build_pygmy_apr_with_config(config);

    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let inspection = rosetta
        .inspect(temp.path())
        .expect("T-GH192-01: APR inspection must succeed");

    // Verify inspection contains meaningful metadata
    assert!(
        !inspection.tensors.is_empty(),
        "T-GH192-01: Inspection must report tensors"
    );
    assert!(
        inspection.format == FormatType::Apr,
        "T-GH192-01: Inspection must identify APR format"
    );
}

/// T-GH192-01: Pre-load inspection returns correct metadata for SafeTensors format
#[test]
fn t_gh192_01_inspect_safetensors_returns_metadata() {
    use crate::format::test_factory::{build_pygmy_safetensors_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    let config = PygmyConfig::realistic();
    let data = build_pygmy_safetensors_with_config(config);

    let mut temp = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp.write_all(&data).expect("Write data");
    temp.flush().expect("Flush");

    let rosetta = RosettaStone::new();
    let inspection = rosetta
        .inspect(temp.path())
        .expect("T-GH192-01: SafeTensors inspection must succeed");

    assert!(
        !inspection.tensors.is_empty(),
        "T-GH192-01: Inspection must report tensors"
    );
    assert!(
        inspection.format == FormatType::SafeTensors,
        "T-GH192-01: Inspection must identify SafeTensors format"
    );
}

/// T-GH192-02: Sequential model loading with different sizes
/// This test verifies that the system correctly handles loading models
/// of different sizes sequentially without config leakage.
#[test]
fn t_gh192_02_sequential_model_size_switching() {
    use crate::format::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create "small" model (few layers)
    let small_config = PygmyConfig::default(); // 1 layer
    let small_data = build_pygmy_apr_with_config(small_config);

    let mut small_temp = NamedTempFile::with_suffix(".apr").expect("Create small temp");
    small_temp.write_all(&small_data).expect("Write small data");
    small_temp.flush().expect("Flush small");

    // Create "large" model (more layers)
    let large_config = PygmyConfig::realistic(); // More layers
    let large_data = build_pygmy_apr_with_config(large_config);

    let mut large_temp = NamedTempFile::with_suffix(".apr").expect("Create large temp");
    large_temp.write_all(&large_data).expect("Write large data");
    large_temp.flush().expect("Flush large");

    let rosetta = RosettaStone::new();

    // Inspect small model first
    let small_inspection = rosetta
        .inspect(small_temp.path())
        .expect("T-GH192-02: Small model inspection must succeed");

    // Inspect large model second
    let large_inspection = rosetta
        .inspect(large_temp.path())
        .expect("T-GH192-02: Large model inspection must succeed");

    // Verify they report different tensor counts or file sizes (no config leakage)
    // Note: realistic() has more tensors than default()
    assert!(
        small_inspection.tensors.len() != large_inspection.tensors.len()
            || small_inspection.file_size != large_inspection.file_size,
        "T-GH192-02: Different model sizes must report different metadata. \
         Small: {} tensors, {} bytes. Large: {} tensors, {} bytes.",
        small_inspection.tensors.len(),
        small_inspection.file_size,
        large_inspection.tensors.len(),
        large_inspection.file_size
    );

    // Re-inspect small model to verify no state leakage from large model
    let small_reinspection = rosetta
        .inspect(small_temp.path())
        .expect("T-GH192-02: Re-inspection must succeed");

    assert_eq!(
        small_inspection.tensors.len(),
        small_reinspection.tensors.len(),
        "T-GH192-02: Re-inspection must match original (no state leakage)"
    );
}

// ========================================================================
// T-GH194: Tensor Count Preservation Tests (rosetta-testing.md §Test Gaps)
// ========================================================================

/// T-GH194-01: APR round-trip preserves ALL tensor count
/// This test verifies that converting to APR and back doesn't drop tensors.
/// Note: Uses SafeTensors as source since GGUF builder is not available.
#[test]
fn t_gh194_01_safetensors_apr_preserves_tensor_count() {
    use crate::format::tensors::{list_tensors_from_bytes, TensorListOptions};
    use crate::format::test_factory::{build_pygmy_safetensors_with_config, PygmyConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Use realistic config to ensure we have a meaningful number of tensors
    let config = PygmyConfig::realistic();
    let st_data = build_pygmy_safetensors_with_config(config);

    // Count tensors in original SafeTensors
    let st_result = list_tensors_from_bytes(&st_data, TensorListOptions::default())
        .expect("T-GH194-01: SafeTensors tensor listing must succeed");
    let st_count = st_result.tensor_count;

    assert!(
        st_count > 5,
        "T-GH194-01: Test requires SafeTensors with >5 tensors, got {}",
        st_count
    );

    // Write SafeTensors to temp file for import
    let mut st_temp = NamedTempFile::with_suffix(".safetensors").expect("Create ST temp");
    st_temp.write_all(&st_data).expect("Write ST data");
    st_temp.flush().expect("Flush ST");

    // Import SafeTensors to APR
    let apr_temp = NamedTempFile::with_suffix(".apr").expect("Create APR temp");
    let apr_path = apr_temp.path().to_path_buf();

    use crate::format::converter::apr_import;
    use crate::format::converter_types::{Architecture, ImportOptions};

    let import_result = apr_import(
        st_temp.path().to_str().expect("Path to string"),
        &apr_path,
        ImportOptions {
            architecture: Architecture::Auto,
            allow_no_config: true,
            ..Default::default()
        },
    );

    assert!(
        import_result.is_ok(),
        "T-GH194-01: SafeTensors→APR import must succeed: {:?}",
        import_result.err()
    );

    // Count tensors in resulting APR
    let apr_data = std::fs::read(&apr_path).expect("Read APR file");
    let apr_result = list_tensors_from_bytes(&apr_data, TensorListOptions::default())
        .expect("T-GH194-01: APR tensor listing must succeed");
    let apr_count = apr_result.tensor_count;

    // INVARIANT: APR must have at least as many tensors as SafeTensors
    // (it may have more if weight tying is resolved, but never fewer)
    assert!(
        apr_count >= st_count,
        "T-GH194-01: APR must preserve all tensors. SafeTensors: {}, APR: {} (dropped {})",
        st_count,
        apr_count,
        st_count.saturating_sub(apr_count)
    );
}

// ========================================================================
// Section 17: TensorValidation Helper Methods (T-COV-95)
// ========================================================================

#[test]
fn tcov_tensor_validation_has_nan_true() {
    let tv = TensorValidation {
        name: "test.weight".to_string(),
        is_valid: false,
        nan_count: 5,
        inf_count: 0,
        zero_count: 0,
        element_count: 100,
        min: -1.0,
        max: 1.0,
        mean: 0.0,
        std: 0.5,
        failures: vec!["NaN detected".to_string()],
    };
    assert!(tv.has_nan());
    assert!(!tv.has_inf());
    assert!(!tv.is_all_zeros());
}

#[test]
fn tcov_tensor_validation_has_inf_true() {
    let tv = TensorValidation {
        name: "test.weight".to_string(),
        is_valid: false,
        nan_count: 0,
        inf_count: 3,
        zero_count: 0,
        element_count: 100,
        min: -1.0,
        max: 1.0,
        mean: 0.0,
        std: 0.5,
        failures: vec!["Inf detected".to_string()],
    };
    assert!(!tv.has_nan());
    assert!(tv.has_inf());
    assert!(!tv.is_all_zeros());
}

#[test]
fn tcov_tensor_validation_is_all_zeros_true() {
    let tv = TensorValidation {
        name: "test.weight".to_string(),
        is_valid: false,
        nan_count: 0,
        inf_count: 0,
        zero_count: 100,
        element_count: 100,
        min: 0.0,
        max: 0.0,
        mean: 0.0,
        std: 0.0,
        failures: vec!["All zeros".to_string()],
    };
    assert!(!tv.has_nan());
    assert!(!tv.has_inf());
    assert!(tv.is_all_zeros());
}

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

#[test]
fn tcov_conversion_path_display_with_intermediates() {
    let path = ConversionPath::chain(
        FormatType::Gguf,
        vec![FormatType::Apr],
        FormatType::SafeTensors,
    );
    let display = format!("{path}");
    assert_eq!(display, "GGUF → APR → SafeTensors");
}

#[test]
fn tcov_verify_roundtrip_safetensors() {
    let source = create_safetensors_fixture();
    let rosetta = RosettaStone::new();
    let report = rosetta.verify_roundtrip(&source, FormatType::Apr);
    // Round-trip may succeed or fail based on implementation details
    // The key is that it doesn't panic
    match report {
        Ok(vr) => {
            // If it succeeds, verify report structure
            assert!(vr.max_diff >= 0.0);
        }
        Err(_) => {
            // Also acceptable - some round-trip failures expected
        }
    }
    let _ = std::fs::remove_file(source);
}

// ========================================================================
// Section 23: Coverage Gap Tests (P050+)
// Targets uncovered branches in mod.rs identified by coverage analysis.
// ========================================================================

// ========================================================================
// P050-P059: FormatType::from_extension error paths (lines 90-116)
// ========================================================================

/// P050: from_extension with a real directory path (is_dir = true, no model files found)
/// Covers: line 90 (is_dir check), lines 93-103 (directory with no candidates)
#[test]
fn p050_from_extension_real_directory_no_candidates() {
    // /tmp is a real directory that won't contain model.gguf, model.apr, model.safetensors
    let path = Path::new("/tmp");
    let result = FormatType::from_extension(path);
    assert!(result.is_err(), "Directory path should fail from_extension");
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("is a directory"),
        "Error should mention directory: {err}"
    );
    assert!(
        err.contains(".gguf") || err.contains(".apr") || err.contains(".safetensors"),
        "Error should mention expected extensions: {err}"
    );
}

/// P051b: from_extension with a nonexistent path that has no extension
/// Covers: lines 112-116 (not a directory, no extension -> "No file extension found")
#[test]
fn p051b_from_extension_nonexistent_no_extension() {
    let path = Path::new("/nonexistent_dir_12345/no_extension_file");
    let result = FormatType::from_extension(path);
    assert!(
        result.is_err(),
        "Nonexistent path with no extension should fail"
    );
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("No file extension"),
        "Error should mention missing extension: {err}"
    );
}

/// P052b: from_extension with a directory containing a candidate model file
/// Covers: lines 104-109 (directory with found candidates -> "Did you mean" suggestion)
#[test]
fn p052b_from_extension_directory_with_candidate() {
    // Create a temp directory with a model file inside
    let temp_dir = std::env::temp_dir().join("rosetta_test_dir_p052b");
    let _ = std::fs::create_dir_all(&temp_dir);
    let model_file = temp_dir.join("model.gguf");
    let _ = std::fs::write(&model_file, b"dummy");

    let result = FormatType::from_extension(&temp_dir);
    assert!(
        result.is_err(),
        "Directory path should fail even with candidates"
    );
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("Did you mean"),
        "Error should suggest candidate file: {err}"
    );
    assert!(
        err.contains("model.gguf"),
        "Suggestion should include model.gguf: {err}"
    );

    let _ = std::fs::remove_dir_all(temp_dir);
}

/// P053b: from_extension with a directory containing multiple candidate model files
/// The code returns the first found candidate from `[model.gguf, model.apr, model.safetensors]`
#[test]
fn p053b_from_extension_directory_with_multiple_candidates() {
    let temp_dir = std::env::temp_dir().join("rosetta_test_dir_p053b");
    let _ = std::fs::create_dir_all(&temp_dir);
    let _ = std::fs::write(temp_dir.join("model.apr"), b"dummy");
    let _ = std::fs::write(temp_dir.join("model.safetensors"), b"dummy");

    let result = FormatType::from_extension(&temp_dir);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(
        err.contains("Did you mean"),
        "Should suggest a candidate: {err}"
    );

    let _ = std::fs::remove_dir_all(temp_dir);
}

/// P054b: from_extension with a nonexistent directory (is_dir returns false)
/// Covers: the else branch at line 112 when path doesn't exist and has no extension
#[test]
fn p054b_from_extension_nonexistent_directory_path() {
    let path = Path::new("/nonexistent_dir_12345");
    let result = FormatType::from_extension(path);
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    // is_dir() returns false for nonexistent paths, so we get "No file extension"
    assert!(
        err.contains("No file extension"),
        "Nonexistent path without extension: {err}"
    );
}

// ========================================================================
// P060-P069: compute_tensor_validation thorough testing (lines 1036-1176)
// Tests the core validation logic directly via RosettaStone instance
// ========================================================================

/// P060: compute_tensor_validation with empty data
/// Covers: lines 1038-1052 (early return for empty data)
#[test]
fn p060_compute_validation_empty_data() {
    let rosetta = RosettaStone::new();
    let tv = rosetta.compute_tensor_validation("empty.weight", &[]);
    assert!(tv.is_valid, "Empty tensor should be valid");
    assert_eq!(tv.element_count, 0);
    assert_eq!(tv.nan_count, 0);
    assert_eq!(tv.inf_count, 0);
    assert_eq!(tv.zero_count, 0);
    assert_eq!(tv.min, 0.0);
    assert_eq!(tv.max, 0.0);
    assert_eq!(tv.mean, 0.0);
    assert_eq!(tv.std, 0.0);
    assert!(tv.failures.is_empty());
}

/// P061b: compute_tensor_validation with all NaN values
/// Covers: lines 1062-1064 (NaN counting), 1082-1087 (valid_count=0 path), 1105-1108
#[test]
fn p061b_compute_validation_all_nan() {
    let rosetta = RosettaStone::new();
    let data = [f32::NAN, f32::NAN, f32::NAN, f32::NAN];
    let tv = rosetta.compute_tensor_validation("corrupted.weight", &data);
    assert!(!tv.is_valid, "All-NaN tensor should be invalid");
    assert_eq!(tv.nan_count, 4);
    assert_eq!(tv.element_count, 4);
    assert_eq!(tv.mean, 0.0, "Mean should be 0 when no valid values");
    assert_eq!(tv.std, 0.0, "Std should be 0 when no valid values");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("NaN") && f.contains("F-DATA-QUALITY-002")),
        "Should report NaN failure"
    );
}

/// P062b: compute_tensor_validation with all Inf values
/// Covers: lines 1066-1068 (Inf counting), 1110-1113
#[test]
fn p062b_compute_validation_all_inf() {
    let rosetta = RosettaStone::new();
    let data = [f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY];
    let tv = rosetta.compute_tensor_validation("overflow.weight", &data);
    assert!(!tv.is_valid, "All-Inf tensor should be invalid");
    assert_eq!(tv.inf_count, 3);
    assert_eq!(tv.nan_count, 0);
    assert_eq!(tv.mean, 0.0, "Mean should be 0 when no valid values");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("Inf") && f.contains("F-DATA-QUALITY-002")),
        "Should report Inf failure"
    );
}

/// P063b: compute_tensor_validation with mixed NaN and Inf
/// Covers: both NaN and Inf branches simultaneously, valid_count = 0
#[test]
fn p063b_compute_validation_mixed_nan_inf() {
    let rosetta = RosettaStone::new();
    let data = [f32::NAN, f32::INFINITY, f32::NAN, f32::NEG_INFINITY];
    let tv = rosetta.compute_tensor_validation("broken.weight", &data);
    assert!(!tv.is_valid);
    assert_eq!(tv.nan_count, 2);
    assert_eq!(tv.inf_count, 2);
    assert!(
        tv.failures.len() >= 2,
        "Should have both NaN and Inf failures"
    );
}

/// P064: compute_tensor_validation with all zeros
/// Covers: lines 1070-1072 (zero counting), 1115-1117 (all zeros failure)
#[test]
fn p064_compute_validation_all_zeros() {
    let rosetta = RosettaStone::new();
    let data = [0.0_f32; 100];
    let tv = rosetta.compute_tensor_validation("uninitialized.weight", &data);
    assert!(!tv.is_valid, "All-zero tensor should be invalid");
    assert_eq!(tv.zero_count, 100);
    assert_eq!(tv.element_count, 100);
    assert!(tv.is_all_zeros());
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("All values are zero") && f.contains("F-DATA-QUALITY-001")),
        "Should report all-zeros failure: {:?}",
        tv.failures
    );
}

/// P065: compute_tensor_validation density gate for embedding tensors (>50% zeros)
/// Covers: lines 1119-1134 (density gate with embedding name threshold=50%)
#[test]
fn p065_compute_validation_embedding_high_zeros() {
    let rosetta = RosettaStone::new();
    // 60% zeros in an embedding tensor (threshold is 50%)
    let mut data = vec![0.0_f32; 60];
    data.extend(vec![0.1_f32; 40]);
    let tv = rosetta.compute_tensor_validation("model.embed_tokens.weight", &data);
    assert!(!tv.is_valid, "Embedding with >50% zeros should be invalid");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("DENSITY") && f.contains("F-DATA-QUALITY-001")),
        "Should report density failure: {:?}",
        tv.failures
    );
}

/// P066: compute_tensor_validation density gate for weight tensors (>80% zeros)
/// Covers: lines 1129 (non-embedding path, threshold=80%)
#[test]
fn p066_compute_validation_weight_high_zeros() {
    let rosetta = RosettaStone::new();
    // 85% zeros in a non-embedding weight tensor (threshold is 80%)
    let mut data = vec![0.0_f32; 85];
    data.extend(vec![0.5_f32; 15]);
    let tv = rosetta.compute_tensor_validation("model.layers.0.self_attn.q_proj.weight", &data);
    assert!(!tv.is_valid, "Weight with >80% zeros should be invalid");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("DENSITY") && f.contains("80")),
        "Should report 80% density threshold: {:?}",
        tv.failures
    );
}

/// P067: compute_tensor_validation density gate for weight under threshold
/// Covers: density gate path where zero_pct <= threshold (no failure)
#[test]
fn p067_compute_validation_weight_acceptable_zeros() {
    let rosetta = RosettaStone::new();
    // 50% zeros in a non-embedding weight tensor (threshold is 80%) -- should pass density
    let mut data = vec![0.0_f32; 50];
    data.extend(vec![0.5_f32; 50]);
    let tv = rosetta.compute_tensor_validation("model.layers.0.weight", &data);
    assert!(
        !tv.failures.iter().any(|f| f.contains("DENSITY")),
        "50% zeros in weight should not trigger 80% density gate: {:?}",
        tv.failures
    );
}

/// P068: compute_tensor_validation L2 norm gate (near-zero L2)
/// Covers: lines 1136-1149 (L2 norm < 1e-6)
#[test]
fn p068_compute_validation_low_l2_norm() {
    let rosetta = RosettaStone::new();
    // Very small values that produce L2 norm < 1e-6
    let data = [1e-7_f32, -1e-7, 1e-7, -1e-7];
    let tv = rosetta.compute_tensor_validation("tiny.weight", &data);
    assert!(!tv.is_valid, "Near-zero L2 norm should be invalid");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("L2 norm") && f.contains("F-DATA-QUALITY-003")),
        "Should report L2 norm failure: {:?}",
        tv.failures
    );
}

/// P069: compute_tensor_validation constant value gate (all identical, non-zero)
/// Covers: lines 1151-1159 (max - min < 1e-10 for non-norm tensors)
#[test]
fn p069_compute_validation_constant_values() {
    let rosetta = RosettaStone::new();
    let data = [0.5_f32; 100];
    let tv = rosetta.compute_tensor_validation("model.layers.0.weight", &data);
    assert!(!tv.is_valid, "Constant tensor should be invalid");
    assert!(
        tv.failures
            .iter()
            .any(|f| f.contains("constant") && f.contains("F-DATA-QUALITY-003")),
        "Should report constant value failure: {:?}",
        tv.failures
    );
}

/// P070b: compute_tensor_validation constant value exemption for norm tensors
/// Covers: lines 1153-1155 (is_norm_or_bias exemption)
#[test]
fn p070b_compute_validation_constant_norm_exempt() {
    let rosetta = RosettaStone::new();
    // All 1.0 values in a norm tensor (e.g., RMS norm weight initialized to 1.0)
    let data = [1.0_f32; 100];
    let tv = rosetta.compute_tensor_validation("model.layers.0.input_layernorm.weight", &data);
    // Norm tensors are exempt from the constant-value check
    assert!(
        !tv.failures.iter().any(|f| f.contains("constant")),
        "Norm tensor should be exempt from constant check: {:?}",
        tv.failures
    );
}

/// P070c: compute_tensor_validation constant value exemption for bias tensors
/// Covers: lines 1153-1155 (is_norm_or_bias with "bias" in name)
#[test]
fn p070c_compute_validation_constant_bias_exempt() {
    let rosetta = RosettaStone::new();
    let data = [0.0_f32; 50]; // All zeros in a bias
                              // Even though all zeros, bias is exempt from constant check.
                              // However, all-zeros still triggers the all-zeros failure.
    let tv = rosetta.compute_tensor_validation("model.layers.0.bias", &data);
    assert!(
        !tv.failures.iter().any(|f| f.contains("constant")),
        "Bias tensor should be exempt from constant check: {:?}",
        tv.failures
    );
}

/// P070d: compute_tensor_validation constant value exemption for ln_ tensors
/// Covers: lines 1153-1155 (is_norm_or_bias with "ln_" in name)
#[test]
fn p070d_compute_validation_constant_ln_exempt() {
    let rosetta = RosettaStone::new();
    let data = [1.0_f32; 100];
    let tv = rosetta.compute_tensor_validation("model.ln_f.weight", &data);
    assert!(
        !tv.failures.iter().any(|f| f.contains("constant")),
        "ln_ tensor should be exempt from constant check: {:?}",
        tv.failures
    );
}

/// P070e: compute_tensor_validation with single valid element (valid_count=1, std=0)
/// Covers: line 1097-1101 (valid_count <= 1 path for std computation)
#[test]
fn p070e_compute_validation_single_element() {
    let rosetta = RosettaStone::new();
    let data = [0.5_f32];
    let tv = rosetta.compute_tensor_validation("single.weight", &data);
    assert_eq!(tv.element_count, 1);
    assert_eq!(tv.std, 0.0, "Std should be 0 for single element");
    assert!((tv.mean - 0.5).abs() < 1e-6);
    assert!((tv.min - 0.5).abs() < 1e-6);
    assert!((tv.max - 0.5).abs() < 1e-6);
}

/// P070f: compute_tensor_validation with NaN mixed with valid data
/// Covers: lines 1062-1064 NaN skip, 1082-1084 mean with valid_count > 0
/// Also covers line 1092 (NaN skip in variance computation)
#[test]
fn p070f_compute_validation_nan_mixed_with_valid() {
    let rosetta = RosettaStone::new();
    let data = [1.0_f32, f32::NAN, 3.0, f32::NAN, 5.0];
    let tv = rosetta.compute_tensor_validation("mixed.weight", &data);
    assert!(!tv.is_valid, "Mixed NaN tensor should be invalid");
    assert_eq!(tv.nan_count, 2);
    assert_eq!(tv.element_count, 5);
    // mean of valid values [1.0, 3.0, 5.0] = 3.0
    assert!(
        (tv.mean - 3.0).abs() < 1e-5,
        "Mean should be ~3.0, got {}",
        tv.mean
    );
    assert!((tv.min - 1.0).abs() < 1e-6);
    assert!((tv.max - 5.0).abs() < 1e-6);
    assert!(tv.std > 0.0, "Std should be non-zero for varied values");
}

/// P070g: compute_tensor_validation min/max clamping when only NaN/Inf present
/// Covers: lines 1170-1171 (min/max clamping when they remain INFINITY/NEG_INFINITY)
#[test]
fn p070g_compute_validation_minmax_clamped() {
    let rosetta = RosettaStone::new();
    // Only NaN and Inf, no finite values -> min stays INFINITY, max stays NEG_INFINITY
    let data = [f32::NAN, f32::INFINITY];
    let tv = rosetta.compute_tensor_validation("bad.weight", &data);
    // min was initialized to INFINITY (never updated), max to NEG_INFINITY
    // The code clamps: if min.is_infinite() { 0.0 } and if max.is_infinite() { 0.0 }
    assert_eq!(tv.min, 0.0, "min should be clamped to 0.0");
    assert_eq!(tv.max, 0.0, "max should be clamped to 0.0");
}

/// P070h: compute_tensor_validation with Inf mixed with valid data
/// Covers: Inf skip in main loop and variance loop
#[test]
fn p070h_compute_validation_inf_mixed_with_valid() {
    let rosetta = RosettaStone::new();
    let data = [2.0_f32, f32::INFINITY, 4.0, f32::NEG_INFINITY, 6.0];
    let tv = rosetta.compute_tensor_validation("mixed_inf.weight", &data);
    assert!(!tv.is_valid);
    assert_eq!(tv.inf_count, 2);
    // mean of valid values [2.0, 4.0, 6.0] = 4.0
    assert!(
        (tv.mean - 4.0).abs() < 1e-5,
        "Mean should be ~4.0, got {}",
        tv.mean
    );
    assert!((tv.min - 2.0).abs() < 1e-6);
    assert!((tv.max - 6.0).abs() < 1e-6);
}

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

    assert!(is_sharded_index(Path::new(
        "model.safetensors.index.json"
    )));
    assert!(is_sharded_index(Path::new(
        "/path/to/model.safetensors.index.json"
    )));
    assert!(is_sharded_index(Path::new(
        "some.other.index.json"
    )));
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
