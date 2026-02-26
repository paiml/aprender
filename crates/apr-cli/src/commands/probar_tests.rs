use super::*;

#[test]
fn test_export_format_parse() {
    assert!(matches!(
        "json".parse::<ExportFormat>(),
        Ok(ExportFormat::Json)
    ));
    assert!(matches!(
        "png".parse::<ExportFormat>(),
        Ok(ExportFormat::Png)
    ));
    assert!(matches!(
        "both".parse::<ExportFormat>(),
        Ok(ExportFormat::Both)
    ));
    assert!(matches!(
        "all".parse::<ExportFormat>(),
        Ok(ExportFormat::Both)
    ));
    assert!("invalid".parse::<ExportFormat>().is_err());
}

#[test]
fn test_layer_snapshot_serialize() {
    let snapshot = LayerSnapshot {
        name: "test".to_string(),
        index: 0,
        histogram: vec![1, 2, 3],
        mean: 0.5,
        std: 1.0,
        min: -1.0,
        max: 2.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    };

    let json = serde_json::to_string(&snapshot).expect("serialize");
    assert!(json.contains("\"name\":\"test\""));
}

// ========================================================================
// ExportFormat Tests
// ========================================================================

#[test]
fn test_export_format_parse_uppercase() {
    assert!(matches!(
        "JSON".parse::<ExportFormat>(),
        Ok(ExportFormat::Json)
    ));
    assert!(matches!(
        "PNG".parse::<ExportFormat>(),
        Ok(ExportFormat::Png)
    ));
}

#[test]
fn test_export_format_debug() {
    let format = ExportFormat::Json;
    let debug = format!("{format:?}");
    assert!(debug.contains("Json"));
}

#[test]
fn test_export_format_clone() {
    let format = ExportFormat::Png;
    let cloned = format;
    assert!(matches!(cloned, ExportFormat::Png));
}

// ========================================================================
// LayerSnapshot Tests
// ========================================================================

#[test]
fn test_layer_snapshot_with_heatmap() {
    let snapshot = LayerSnapshot {
        name: "attn".to_string(),
        index: 1,
        histogram: vec![0; 256],
        mean: 0.0,
        std: 1.0,
        min: -3.0,
        max: 3.0,
        heatmap: Some(vec![1.0, 2.0, 3.0, 4.0]),
        heatmap_width: Some(2),
        heatmap_height: Some(2),
    };
    assert!(snapshot.heatmap.is_some());
    assert_eq!(snapshot.heatmap_width, Some(2));
}

#[test]
fn test_layer_snapshot_clone() {
    let snapshot = LayerSnapshot {
        name: "test".to_string(),
        index: 0,
        histogram: vec![1, 2, 3],
        mean: 0.5,
        std: 1.0,
        min: -1.0,
        max: 2.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    };
    let cloned = snapshot.clone();
    assert_eq!(cloned.name, snapshot.name);
    assert_eq!(cloned.index, snapshot.index);
}

#[test]
fn test_layer_snapshot_deserialize() {
    let json = r#"{"name":"test","index":0,"histogram":[1,2,3],"mean":0.5,"std":1.0,"min":-1.0,"max":2.0}"#;
    let snapshot: LayerSnapshot = serde_json::from_str(json).expect("deserialize");
    assert_eq!(snapshot.name, "test");
    assert_eq!(snapshot.index, 0);
}

#[test]
fn test_layer_snapshot_histogram() {
    let snapshot = LayerSnapshot {
        name: "hist".to_string(),
        index: 0,
        histogram: vec![10, 20, 30, 40],
        mean: 0.0,
        std: 1.0,
        min: -2.0,
        max: 2.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    };
    assert_eq!(snapshot.histogram.len(), 4);
    assert_eq!(snapshot.histogram[0], 10);
}

// ========================================================================
// run Command Tests
// ========================================================================

use std::io::Write;
use tempfile::{tempdir, NamedTempFile};

#[test]
fn test_run_file_not_found() {
    let output_dir = tempdir().expect("create output dir");
    let result = run(
        Path::new("/nonexistent/model.apr"),
        output_dir.path(),
        ExportFormat::Json,
        None,
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_run_invalid_apr() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not a valid apr file").expect("write");
    let output_dir = tempdir().expect("create output dir");

    let result = run(
        file.path(),
        output_dir.path(),
        ExportFormat::Json,
        None,
        None,
    );
    // Should fail (invalid APR)
    assert!(result.is_err());
}

#[test]
fn test_run_with_png_format() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");
    let output_dir = tempdir().expect("create output dir");

    let result = run(
        file.path(),
        output_dir.path(),
        ExportFormat::Png,
        None,
        None,
    );
    // Should fail (invalid file)
    assert!(result.is_err());
}

#[test]
fn test_run_with_both_format() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");
    let output_dir = tempdir().expect("create output dir");

    let result = run(
        file.path(),
        output_dir.path(),
        ExportFormat::Both,
        None,
        None,
    );
    // Should fail (invalid file)
    assert!(result.is_err());
}

#[test]
fn test_run_with_golden() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");
    let mut golden = NamedTempFile::with_suffix(".json").expect("create golden file");
    golden.write_all(b"{}").expect("write");
    let output_dir = tempdir().expect("create output dir");

    let result = run(
        file.path(),
        output_dir.path(),
        ExportFormat::Json,
        Some(golden.path()),
        None,
    );
    // Should fail (invalid file)
    assert!(result.is_err());
}

#[test]
fn test_run_with_layer_filter() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not valid").expect("write");
    let output_dir = tempdir().expect("create output dir");

    let result = run(
        file.path(),
        output_dir.path(),
        ExportFormat::Json,
        None,
        Some("encoder"),
    );
    // Should fail (invalid file)
    assert!(result.is_err());
}

#[test]
fn test_run_is_directory() {
    let dir = tempdir().expect("create input dir");
    let output_dir = tempdir().expect("create output dir");

    let result = run(
        dir.path(),
        output_dir.path(),
        ExportFormat::Json,
        None,
        None,
    );
    // Should fail (is a directory)
    assert!(result.is_err());
}

#[test]
fn test_run_gguf_format() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    file.write_all(b"not valid gguf").expect("write");
    let output_dir = tempdir().expect("create output dir");

    let result = run(
        file.path(),
        output_dir.path(),
        ExportFormat::Json,
        None,
        None,
    );
    // Should fail (invalid GGUF)
    assert!(result.is_err());
}

#[test]
fn test_run_safetensors_format() {
    let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    file.write_all(b"not valid safetensors").expect("write");
    let output_dir = tempdir().expect("create output dir");

    let result = run(
        file.path(),
        output_dir.path(),
        ExportFormat::Json,
        None,
        None,
    );
    // Should fail (invalid SafeTensors)
    assert!(result.is_err());
}

// ========================================================================
// ExportFormat Error Messages
// ========================================================================

#[test]
fn test_export_format_error_contains_input() {
    let err = "foobar".parse::<ExportFormat>().expect_err("should fail");
    assert!(
        err.contains("foobar"),
        "error message should contain the invalid input"
    );
}

#[test]
fn test_export_format_error_suggests_valid_options() {
    let err = "xyz".parse::<ExportFormat>().expect_err("should fail");
    assert!(err.contains("json"), "error should mention 'json'");
    assert!(err.contains("png"), "error should mention 'png'");
    assert!(err.contains("both"), "error should mention 'both'");
}

#[test]
fn test_export_format_case_insensitive_mixed() {
    assert!(matches!(
        "Json".parse::<ExportFormat>(),
        Ok(ExportFormat::Json)
    ));
    assert!(matches!(
        "pNg".parse::<ExportFormat>(),
        Ok(ExportFormat::Png)
    ));
    assert!(matches!(
        "BOTH".parse::<ExportFormat>(),
        Ok(ExportFormat::Both)
    ));
    assert!(matches!(
        "ALL".parse::<ExportFormat>(),
        Ok(ExportFormat::Both)
    ));
}

#[test]
fn test_export_format_copy_semantics() {
    let a = ExportFormat::Both;
    let b = a; // Copy
               // Both a and b are valid after copy
    assert!(matches!(a, ExportFormat::Both));
    assert!(matches!(b, ExportFormat::Both));
}

#[test]
fn test_export_format_debug_all_variants() {
    assert_eq!(format!("{:?}", ExportFormat::Json), "Json");
    assert_eq!(format!("{:?}", ExportFormat::Png), "Png");
    assert_eq!(format!("{:?}", ExportFormat::Both), "Both");
}

// ========================================================================
// generate_snapshots Tests
// ========================================================================

#[test]
fn test_generate_snapshots_zero_layers_returns_fallback() {
    let snapshots = generate_snapshots(None, 0, None);
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].name, "fallback");
    assert_eq!(snapshots[0].index, 0);
    assert_eq!(snapshots[0].histogram.len(), 256);
    assert!(snapshots[0].heatmap.is_none());
}

#[test]
fn test_generate_snapshots_with_layer_count() {
    let snapshots = generate_snapshots(None, 3, None);

    assert_eq!(snapshots.len(), 3);
    for (i, snap) in snapshots.iter().enumerate() {
        assert_eq!(snap.name, format!("block_{i}"));
        assert_eq!(snap.index, i);
        assert_eq!(snap.histogram.len(), 256);
        // No model path → all zeros (no tensor data)
        assert!(snap.histogram.iter().all(|&v| v == 0));
        assert_eq!(snap.mean, 0.0);
        assert_eq!(snap.std, 0.0);
        assert!(snap.heatmap.is_none());
    }
}

#[test]
fn test_generate_snapshots_two_layers() {
    let snapshots = generate_snapshots(None, 2, None);

    assert_eq!(snapshots.len(), 2);
    assert_eq!(snapshots[0].name, "block_0");
    assert_eq!(snapshots[1].name, "block_1");
}

include!("probar_tests_generate_snapshots.rs");
include!("probar_tests_export_png.rs");
include!("probar_tests_generate_diff.rs");
