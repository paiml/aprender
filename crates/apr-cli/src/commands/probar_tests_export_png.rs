
#[test]
fn test_export_png_histogram_normalization() {
    let output_dir = tempdir().expect("create output dir");
    // Histogram with one spike: bin 128 has max value, rest are 0
    let mut histogram = vec![0u32; 256];
    histogram[128] = 1000;

    let layers = vec![LayerSnapshot {
        name: "spike".to_string(),
        index: 0,
        histogram,
        mean: 0.0,
        std: 0.01,
        min: 0.0,
        max: 0.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];

    export_png(&layers, output_dir.path()).expect("export png");

    // Read the PGM and verify pixel data
    let content = fs::read(output_dir.path().join("layer_000_spike.pgm")).expect("read pgm");
    // Find start of pixel data (after 3rd newline)
    let header_end = content
        .windows(1)
        .enumerate()
        .filter(|(_, w)| w[0] == b'\n')
        .nth(2)
        .map(|(i, _)| i + 1)
        .expect("find header end");

    let pixels = &content[header_end..];
    // Column 128 should have a black bar (value 0), other columns should be white (255)
    // Check bottom pixel of column 0 (should be white - no bar)
    let bottom_row = 99; // height - 1
    assert_eq!(
        pixels[bottom_row * 256 + 0],
        255,
        "column 0 bottom should be white"
    );
    // Column 128 bottom should be black (full bar)
    assert_eq!(
        pixels[bottom_row * 256 + 128],
        0,
        "column 128 bottom should be black"
    );
}

// ========================================================================
// export_by_format Tests
// ========================================================================

#[test]
fn test_export_by_format_json_creates_manifest_only() {
    let output_dir = tempdir().expect("create output dir");
    let manifest = ProbarManifest {
        source_model: "m.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![LayerSnapshot {
            name: "l".to_string(),
            index: 0,
            histogram: vec![1; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }],
        golden_reference: None,
    };

    export_by_format(
        ExportFormat::Json,
        &manifest,
        &manifest.layers,
        output_dir.path(),
    )
    .expect("export");

    assert!(output_dir.path().join("manifest.json").exists());
    // PNG/PGM should NOT exist
    assert!(!output_dir.path().join("layer_000_l.pgm").exists());
}

#[test]
fn test_export_by_format_png_creates_pgm_only() {
    let output_dir = tempdir().expect("create output dir");
    let layers = vec![LayerSnapshot {
        name: "x".to_string(),
        index: 0,
        histogram: vec![1; 256],
        mean: 0.0,
        std: 1.0,
        min: -1.0,
        max: 1.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];
    let manifest = ProbarManifest {
        source_model: "m.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: layers.clone(),
        golden_reference: None,
    };

    export_by_format(ExportFormat::Png, &manifest, &layers, output_dir.path()).expect("export");

    assert!(!output_dir.path().join("manifest.json").exists());
    assert!(output_dir.path().join("layer_000_x.pgm").exists());
}

#[test]
fn test_export_by_format_both_creates_all() {
    let output_dir = tempdir().expect("create output dir");
    let layers = vec![LayerSnapshot {
        name: "y".to_string(),
        index: 0,
        histogram: vec![1; 256],
        mean: 0.0,
        std: 1.0,
        min: -1.0,
        max: 1.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];
    let manifest = ProbarManifest {
        source_model: "m.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: layers.clone(),
        golden_reference: None,
    };

    export_by_format(ExportFormat::Both, &manifest, &layers, output_dir.path()).expect("export");

    assert!(output_dir.path().join("manifest.json").exists());
    assert!(output_dir.path().join("layer_000_y.pgm").exists());
}

// ========================================================================
// generate_diff Tests
// ========================================================================

#[test]
fn test_generate_diff_identical_models_produces_zero_diffs() {
    let golden_dir = tempdir().expect("golden dir");
    let output_dir = tempdir().expect("output dir");

    let layers = vec![LayerSnapshot {
        name: "block_0".to_string(),
        index: 0,
        histogram: vec![100; 256],
        mean: 0.5,
        std: 1.0,
        min: -2.0,
        max: 2.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];

    // Write golden manifest
    let golden_manifest = ProbarManifest {
        source_model: "golden.apr".to_string(),
        timestamp: "t1".to_string(),
        format: "APR".to_string(),
        layers: layers.clone(),
        golden_reference: None,
    };
    let golden_json = serde_json::to_string_pretty(&golden_manifest).expect("serialize golden");
    fs::write(golden_dir.path().join("manifest.json"), &golden_json).expect("write golden");

    // Current manifest with identical stats
    let current = ProbarManifest {
        source_model: "current.apr".to_string(),
        timestamp: "t2".to_string(),
        format: "APR".to_string(),
        layers,
        golden_reference: None,
    };

    generate_diff(golden_dir.path(), &current, output_dir.path()).expect("generate diff");

    let diff_content =
        fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read diff");
    let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse diff");

    assert_eq!(diff["total_diffs"], 0);
    assert!(diff["diffs"].as_array().expect("diffs array").is_empty());
}

#[test]
fn test_generate_diff_detects_name_mismatch() {
    let golden_dir = tempdir().expect("golden dir");
    let output_dir = tempdir().expect("output dir");

    let golden_manifest = ProbarManifest {
        source_model: "golden.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![LayerSnapshot {
            name: "layer_a".to_string(),
            index: 0,
            histogram: vec![0; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }],
        golden_reference: None,
    };
    fs::write(
        golden_dir.path().join("manifest.json"),
        serde_json::to_string(&golden_manifest).expect("ser"),
    )
    .expect("write");

    let current = ProbarManifest {
        source_model: "current.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![LayerSnapshot {
            name: "layer_b".to_string(),
            index: 0,
            histogram: vec![0; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }],
        golden_reference: None,
    };

    generate_diff(golden_dir.path(), &current, output_dir.path()).expect("diff");

    let diff_content =
        fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
    let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

    assert!(diff["total_diffs"].as_u64().expect("total") >= 1);
    let diffs = diff["diffs"].as_array().expect("diffs array");
    assert!(diffs.iter().any(|d| d["type"] == "name_mismatch"));
}

#[test]
fn test_generate_diff_detects_stats_divergence() {
    let golden_dir = tempdir().expect("golden dir");
    let output_dir = tempdir().expect("output dir");

    let golden_manifest = ProbarManifest {
        source_model: "golden.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![LayerSnapshot {
            name: "block_0".to_string(),
            index: 0,
            histogram: vec![0; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }],
        golden_reference: None,
    };
    fs::write(
        golden_dir.path().join("manifest.json"),
        serde_json::to_string(&golden_manifest).expect("ser"),
    )
    .expect("write");

    let current = ProbarManifest {
        source_model: "current.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![LayerSnapshot {
            name: "block_0".to_string(),
            index: 0,
            histogram: vec![0; 256],
            mean: 0.5, // diverged by 0.5 (> 0.01 threshold)
            std: 2.0,  // diverged by 1.0 (> 0.01 threshold)
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }],
        golden_reference: None,
    };

    generate_diff(golden_dir.path(), &current, output_dir.path()).expect("diff");

    let diff_content =
        fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
    let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

    assert!(diff["total_diffs"].as_u64().expect("total") >= 1);
    let diffs = diff["diffs"].as_array().expect("diffs array");
    assert!(diffs.iter().any(|d| d["type"] == "stats_divergence"));
}

#[test]
fn test_generate_diff_within_tolerance_no_divergence() {
    let golden_dir = tempdir().expect("golden dir");
    let output_dir = tempdir().expect("output dir");

    let golden_manifest = ProbarManifest {
        source_model: "golden.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![LayerSnapshot {
            name: "block_0".to_string(),
            index: 0,
            histogram: vec![0; 256],
            mean: 1.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }],
        golden_reference: None,
    };
    fs::write(
        golden_dir.path().join("manifest.json"),
        serde_json::to_string(&golden_manifest).expect("ser"),
    )
    .expect("write");

    let current = ProbarManifest {
        source_model: "current.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![LayerSnapshot {
            name: "block_0".to_string(),
            index: 0,
            histogram: vec![0; 256],
            mean: 1.005, // diff = 0.005, within 0.01 tolerance
            std: 1.009,  // diff = 0.009, within 0.01 tolerance
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }],
        golden_reference: None,
    };

    generate_diff(golden_dir.path(), &current, output_dir.path()).expect("diff");

    let diff_content =
        fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
    let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

    assert_eq!(diff["total_diffs"], 0);
}

#[test]
fn test_generate_diff_missing_golden_manifest() {
    let golden_dir = tempdir().expect("golden dir");
    let output_dir = tempdir().expect("output dir");
    // Don't create manifest.json in golden dir

    let current = ProbarManifest {
        source_model: "c.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![],
        golden_reference: None,
    };

    let result = generate_diff(golden_dir.path(), &current, output_dir.path());
    assert!(result.is_err(), "missing golden manifest should fail");
}

#[test]
fn test_generate_diff_invalid_golden_json() {
    let golden_dir = tempdir().expect("golden dir");
    let output_dir = tempdir().expect("output dir");

    fs::write(golden_dir.path().join("manifest.json"), "not valid json").expect("write bad json");

    let current = ProbarManifest {
        source_model: "c.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![],
        golden_reference: None,
    };

    let result = generate_diff(golden_dir.path(), &current, output_dir.path());
    assert!(result.is_err(), "invalid golden JSON should fail");
}
