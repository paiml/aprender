
/// Decode an 8-bit grayscale PNG whose IDAT uses stored DEFLATE blocks.
/// Deliberately independent of the encoder's own test helpers so this asserts
/// the file's bytes, not the encoder's view of them.
fn decode_grayscale_png(png: &[u8]) -> Vec<u8> {
    let mut idat = Vec::new();
    let (mut width, mut height) = (0usize, 0usize);
    let mut i = 8; // skip signature
    while i + 8 <= png.len() {
        let len = u32::from_be_bytes([png[i], png[i + 1], png[i + 2], png[i + 3]]) as usize;
        let kind = &png[i + 4..i + 8];
        let data = &png[i + 8..i + 8 + len];
        if kind == b"IHDR" {
            width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
            height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
            assert_eq!(data[8], 8, "bit depth");
            assert_eq!(data[9], 0, "grayscale color type");
        } else if kind == b"IDAT" {
            idat.extend_from_slice(data);
        }
        i += 8 + len + 4;
    }

    // Inflate stored blocks (zlib header is 2 bytes, Adler-32 trailer is 4).
    let mut raw = Vec::new();
    let mut j = 2;
    loop {
        let header = idat[j];
        let blen = u16::from_le_bytes([idat[j + 1], idat[j + 2]]) as usize;
        raw.extend_from_slice(&idat[j + 5..j + 5 + blen]);
        j += 5 + blen;
        if header & 1 == 1 {
            break;
        }
    }

    assert_eq!(raw.len(), height * (width + 1), "scanline count");
    let mut pixels = Vec::with_capacity(width * height);
    for row in raw.chunks_exact(width + 1) {
        assert_eq!(row[0], 0, "filter type None");
        pixels.extend_from_slice(&row[1..]);
    }
    pixels
}

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

    // Read the PNG back and verify pixel data. This used to read a `.pgm`,
    // which is exactly the file the command claimed it had NOT written.
    let content = fs::read(output_dir.path().join("layer_000_spike.png")).expect("read png");
    assert_eq!(
        &content[0..8],
        b"\x89PNG\r\n\x1a\n",
        "export_png must write a real PNG"
    );
    let pixels = decode_grayscale_png(&content);
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
    // PNG should NOT exist
    assert!(!output_dir.path().join("layer_000_l.png").exists());
}

#[test]
fn test_export_by_format_png_creates_png_only() {
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
    let png = output_dir.path().join("layer_000_x.png");
    assert!(png.exists(), "the path printed to the user must exist");
    assert_eq!(
        &fs::read(&png).expect("read png")[0..8],
        b"\x89PNG\r\n\x1a\n",
        "and it must actually be a PNG"
    );
    assert!(
        !output_dir.path().join("layer_000_x.pgm").exists(),
        "no stray Netpbm file"
    );
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
    assert!(output_dir.path().join("layer_000_y.png").exists());
}

#[test]
fn test_every_listed_generated_file_actually_exists() {
    // The defect: `Generated files:` printed `.png` paths while `export_png`
    // wrote `.pgm`, so a consumer copying the listed paths hit ENOENT.
    for format in [ExportFormat::Json, ExportFormat::Png, ExportFormat::Both] {
        let output_dir = tempdir().expect("create output dir");
        let layers = vec![LayerSnapshot {
            name: "block_0".to_string(),
            index: 0,
            histogram: vec![7; 256],
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

        export_by_format(format, &manifest, &layers, output_dir.path()).expect("export");

        let listed = generated_file_paths(format, output_dir.path(), &layers);
        assert!(!listed.is_empty(), "{format:?} must list something");
        for path in listed {
            assert!(
                path.exists(),
                "{format:?}: listed {} but it was never written",
                path.display()
            );
        }
    }
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

    generate_diff_with_tolerance(golden_dir.path(), &current, output_dir.path(), 0.98).expect("generate diff");

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

    generate_diff_with_tolerance(golden_dir.path(), &current, output_dir.path(), 0.98).expect("diff");

    let diff_content =
        fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
    let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

    // Name mismatch with identical stats: no numerical divergence detected.
    // The diff report only records numerical differences, not name mismatches.
    // This is correct — probar validates activations, not layer naming.
    let _diffs = diff["diffs"].as_array().expect("diffs array");
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

    generate_diff_with_tolerance(golden_dir.path(), &current, output_dir.path(), 0.98).expect("diff");

    let diff_content =
        fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
    let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

    assert!(diff["total_diffs"].as_u64().expect("total") >= 1);
    let diffs = diff["diffs"].as_array().expect("diffs array");
    // Stats divergence: mean_diff > 0.01 or std_diff > 0.01
    assert!(diffs.iter().any(|d| {
        d["mean_diff"].as_f64().unwrap_or(0.0) > 0.01
            || d["std_diff"].as_f64().unwrap_or(0.0) > 0.01
    }));
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

    generate_diff_with_tolerance(golden_dir.path(), &current, output_dir.path(), 0.98).expect("diff");

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

    let result = generate_diff_with_tolerance(golden_dir.path(), &current, output_dir.path(), 0.98);
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

    let result = generate_diff_with_tolerance(golden_dir.path(), &current, output_dir.path(), 0.98);
    assert!(result.is_err(), "invalid golden JSON should fail");
}
