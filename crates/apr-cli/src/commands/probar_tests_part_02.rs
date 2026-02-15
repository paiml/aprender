
#[test]
fn test_generate_snapshots_filter_matches_subset() {
    let mut hp = serde_json::Map::new();
    hp.insert("n_layer".to_string(), serde_json::json!(5));

    let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

    let bytes = rmp_serde::to_vec(&metadata).expect("encode");

    // Filter for "block_3" - should match only block_3
    let snapshots = generate_snapshots(&bytes, Some("block_3"));
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].name, "block_3");
    assert_eq!(snapshots[0].index, 3);
}

#[test]
fn test_generate_snapshots_filter_matches_none_returns_placeholder() {
    let mut hp = serde_json::Map::new();
    hp.insert("n_layer".to_string(), serde_json::json!(3));

    let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

    let bytes = rmp_serde::to_vec(&metadata).expect("encode");

    // Filter for something that doesn't match any layer
    let snapshots = generate_snapshots(&bytes, Some("nonexistent"));
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].name, "placeholder");
}

#[test]
fn test_generate_snapshots_filter_partial_match() {
    let mut hp = serde_json::Map::new();
    hp.insert("n_layer".to_string(), serde_json::json!(10));

    let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

    let bytes = rmp_serde::to_vec(&metadata).expect("encode");

    // "block_" matches all layers
    let snapshots = generate_snapshots(&bytes, Some("block_"));
    assert_eq!(snapshots.len(), 10);
}

#[test]
fn test_generate_snapshots_hyperparameters_not_object() {
    // hyperparameters is a string instead of an object
    let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    metadata.insert(
        "hyperparameters".to_string(),
        serde_json::json!("not an object"),
    );

    let bytes = rmp_serde::to_vec(&metadata).expect("encode");
    let snapshots = generate_snapshots(&bytes, None);

    // Falls through to placeholder since as_object() returns None
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].name, "placeholder");
}

#[test]
fn test_generate_snapshots_zero_layers() {
    let mut hp = serde_json::Map::new();
    hp.insert("n_layer".to_string(), serde_json::json!(0));

    let mut metadata: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    metadata.insert("hyperparameters".to_string(), serde_json::Value::Object(hp));

    let bytes = rmp_serde::to_vec(&metadata).expect("encode");
    let snapshots = generate_snapshots(&bytes, None);

    // 0 layers => empty => placeholder
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].name, "placeholder");
}

#[test]
fn test_generate_snapshots_placeholder_stats() {
    let snapshots = generate_snapshots(&[], None);
    let placeholder = &snapshots[0];
    assert_eq!(placeholder.mean, 0.0);
    assert_eq!(placeholder.std, 1.0);
    assert_eq!(placeholder.min, -1.0);
    assert_eq!(placeholder.max, 1.0);
    assert!(placeholder.heatmap.is_none());
    assert!(placeholder.heatmap_width.is_none());
    assert!(placeholder.heatmap_height.is_none());
}

// ========================================================================
// create_manifest Tests
// ========================================================================

#[test]
fn test_create_manifest_basic_fields() {
    let layers = vec![LayerSnapshot {
        name: "block_0".to_string(),
        index: 0,
        histogram: vec![100; 256],
        mean: 0.0,
        std: 1.0,
        min: -3.0,
        max: 3.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];

    let manifest = create_manifest(
        Path::new("/tmp/model.apr"),
        "APRN (aprender v1)",
        &layers,
        None,
    );

    assert_eq!(manifest.source_model, "/tmp/model.apr");
    assert_eq!(manifest.format, "APRN (aprender v1)");
    assert_eq!(manifest.layers.len(), 1);
    assert_eq!(manifest.layers[0].name, "block_0");
    assert!(manifest.golden_reference.is_none());
    // Timestamp should be non-empty RFC3339
    assert!(!manifest.timestamp.is_empty());
    assert!(manifest.timestamp.contains('T'));
}

#[test]
fn test_create_manifest_with_golden_reference() {
    let manifest = create_manifest(
        Path::new("/model.apr"),
        "APR v2",
        &[],
        Some(Path::new("/golden/reference")),
    );
    assert_eq!(
        manifest.golden_reference,
        Some("/golden/reference".to_string())
    );
}

#[test]
fn test_create_manifest_without_golden_reference() {
    let manifest = create_manifest(Path::new("/model.apr"), "APR v2", &[], None);
    assert!(manifest.golden_reference.is_none());
}

#[test]
fn test_create_manifest_preserves_layer_order() {
    let layers: Vec<LayerSnapshot> = (0..5)
        .map(|i| LayerSnapshot {
            name: format!("layer_{i}"),
            index: i,
            histogram: vec![0; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        })
        .collect();

    let manifest = create_manifest(Path::new("/m.apr"), "APR", &layers, None);
    for (i, layer) in manifest.layers.iter().enumerate() {
        assert_eq!(layer.name, format!("layer_{i}"));
        assert_eq!(layer.index, i);
    }
}

// ========================================================================
// validate_path Tests (direct)
// ========================================================================

#[test]
fn test_validate_path_nonexistent_returns_file_not_found() {
    let result = validate_path(Path::new("/absolutely/nonexistent/path.apr"));
    assert!(result.is_err());
    let err = result.expect_err("should be error");
    assert!(
        matches!(err, CliError::FileNotFound(_)),
        "expected FileNotFound, got {err:?}"
    );
}

#[test]
fn test_validate_path_directory_returns_not_a_file() {
    let dir = tempdir().expect("create temp dir");
    let result = validate_path(dir.path());
    assert!(result.is_err());
    let err = result.expect_err("should be error");
    assert!(
        matches!(err, CliError::NotAFile(_)),
        "expected NotAFile, got {err:?}"
    );
}

#[test]
fn test_validate_path_valid_file_succeeds() {
    let file = NamedTempFile::new().expect("create temp file");
    let result = validate_path(file.path());
    assert!(result.is_ok());
}

// ========================================================================
// export_json Tests
// ========================================================================

#[test]
fn test_export_json_creates_manifest_file() {
    let output_dir = tempdir().expect("create output dir");
    let manifest = ProbarManifest {
        source_model: "/test/model.apr".to_string(),
        timestamp: "2026-01-01T00:00:00Z".to_string(),
        format: "APRN".to_string(),
        layers: vec![],
        golden_reference: None,
    };

    export_json(&manifest, output_dir.path()).expect("export json");

    let json_path = output_dir.path().join("manifest.json");
    assert!(json_path.exists(), "manifest.json should be created");

    let content = fs::read_to_string(&json_path).expect("read manifest");
    assert!(content.contains("\"source_model\""));
    assert!(content.contains("/test/model.apr"));
    assert!(content.contains("\"format\""));
    assert!(content.contains("APRN"));
}

#[test]
fn test_export_json_contains_layer_data() {
    let output_dir = tempdir().expect("create output dir");
    let manifest = ProbarManifest {
        source_model: "m.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![LayerSnapshot {
            name: "test_layer".to_string(),
            index: 42,
            histogram: vec![1, 2, 3],
            mean: 0.5,
            std: 0.25,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        }],
        golden_reference: None,
    };

    export_json(&manifest, output_dir.path()).expect("export json");

    let content = fs::read_to_string(output_dir.path().join("manifest.json")).expect("read");
    assert!(content.contains("test_layer"));
    assert!(content.contains("42"));
}

#[test]
fn test_export_json_roundtrip() {
    let output_dir = tempdir().expect("create output dir");
    let manifest = ProbarManifest {
        source_model: "model.apr".to_string(),
        timestamp: "2026-02-06T12:00:00Z".to_string(),
        format: "GGUF".to_string(),
        layers: vec![LayerSnapshot {
            name: "block_0".to_string(),
            index: 0,
            histogram: vec![50; 256],
            mean: -0.1,
            std: 0.9,
            min: -4.0,
            max: 4.0,
            heatmap: Some(vec![1.0, 2.0]),
            heatmap_width: Some(2),
            heatmap_height: Some(1),
        }],
        golden_reference: Some("/golden".to_string()),
    };

    export_json(&manifest, output_dir.path()).expect("export");

    let content = fs::read_to_string(output_dir.path().join("manifest.json")).expect("read");
    let loaded: ProbarManifest = serde_json::from_str(&content).expect("deserialize");

    assert_eq!(loaded.source_model, "model.apr");
    assert_eq!(loaded.format, "GGUF");
    assert_eq!(loaded.layers.len(), 1);
    assert_eq!(loaded.layers[0].histogram.len(), 256);
    assert_eq!(loaded.golden_reference, Some("/golden".to_string()));
}

// ========================================================================
// export_png Tests
// ========================================================================

#[test]
fn test_export_png_creates_pgm_and_meta_files() {
    let output_dir = tempdir().expect("create output dir");
    let layers = vec![LayerSnapshot {
        name: "attn".to_string(),
        index: 0,
        histogram: vec![100; 256],
        mean: 0.0,
        std: 1.0,
        min: -3.0,
        max: 3.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];

    export_png(&layers, output_dir.path()).expect("export png");

    let pgm_path = output_dir.path().join("layer_000_attn.pgm");
    let meta_path = output_dir.path().join("layer_000_attn.meta.json");

    assert!(pgm_path.exists(), "PGM file should be created");
    assert!(meta_path.exists(), "meta.json sidecar should be created");
}

#[test]
fn test_export_png_pgm_header_format() {
    let output_dir = tempdir().expect("create output dir");
    let layers = vec![LayerSnapshot {
        name: "test".to_string(),
        index: 5,
        histogram: vec![50; 256],
        mean: 0.0,
        std: 1.0,
        min: -1.0,
        max: 1.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];

    export_png(&layers, output_dir.path()).expect("export png");

    let content = fs::read(output_dir.path().join("layer_005_test.pgm")).expect("read pgm");
    // PGM header: "P5\n256 100\n255\n" followed by pixel data
    let header_end = content
        .windows(1)
        .enumerate()
        .filter(|(_, w)| w[0] == b'\n')
        .nth(2)
        .map(|(i, _)| i + 1)
        .expect("find header end");

    let header = std::str::from_utf8(&content[..header_end]).expect("valid utf8 header");
    assert!(header.starts_with("P5\n"));
    assert!(header.contains("256 100"));
    assert!(header.contains("255"));

    // Pixel data should be 256 * 100 bytes
    assert_eq!(content.len() - header_end, 256 * 100);
}

#[test]
fn test_export_png_meta_json_contents() {
    let output_dir = tempdir().expect("create output dir");
    let layers = vec![LayerSnapshot {
        name: "ffn".to_string(),
        index: 7,
        histogram: vec![0; 256],
        mean: 0.5,
        std: 2.0,
        min: -5.0,
        max: 5.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];

    export_png(&layers, output_dir.path()).expect("export png");

    let meta_content =
        fs::read_to_string(output_dir.path().join("layer_007_ffn.meta.json")).expect("read meta");
    let meta: serde_json::Value = serde_json::from_str(&meta_content).expect("parse meta json");

    assert_eq!(meta["name"], "ffn");
    assert_eq!(meta["index"], 7);
    assert_eq!(meta["mean"], 0.5);
    assert_eq!(meta["std"], 2.0);
    assert_eq!(meta["min"], -5.0);
    assert_eq!(meta["max"], 5.0);
    assert_eq!(meta["histogram_bins"], 256);
    assert_eq!(meta["image_width"], 256);
    assert_eq!(meta["image_height"], 100);
}

#[test]
fn test_export_png_multiple_layers() {
    let output_dir = tempdir().expect("create output dir");
    let layers: Vec<LayerSnapshot> = (0..3)
        .map(|i| LayerSnapshot {
            name: format!("block_{i}"),
            index: i,
            histogram: vec![100; 256],
            mean: 0.0,
            std: 1.0,
            min: -1.0,
            max: 1.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        })
        .collect();

    export_png(&layers, output_dir.path()).expect("export png");

    for i in 0..3 {
        let pgm = output_dir
            .path()
            .join(format!("layer_{i:03}_block_{i}.pgm"));
        let meta = output_dir
            .path()
            .join(format!("layer_{i:03}_block_{i}.meta.json"));
        assert!(pgm.exists(), "PGM for layer {i} should exist");
        assert!(meta.exists(), "meta for layer {i} should exist");
    }
}

#[test]
fn test_export_png_empty_layers() {
    let output_dir = tempdir().expect("create output dir");
    let result = export_png(&[], output_dir.path());
    assert!(
        result.is_ok(),
        "empty layers should succeed (no files created)"
    );
}
