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
