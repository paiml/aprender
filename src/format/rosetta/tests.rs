//\! Rosetta Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

pub(crate) use super::*;
pub(crate) use tests_part_03::{create_apr_fixture, create_safetensors_fixture, unique_temp_path};

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

#[path = "tests_part_02.rs"]
mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
#[path = "tests_part_04.rs"]
mod tests_part_04;
