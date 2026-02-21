
#[test]
fn test_generate_diff_unequal_layer_counts_zips_shortest() {
    let golden_dir = tempdir().expect("golden dir");
    let output_dir = tempdir().expect("output dir");

    let mk_layer = |name: &str| LayerSnapshot {
        name: name.to_string(),
        index: 0,
        histogram: vec![0; 256],
        mean: 0.0,
        std: 1.0,
        min: -1.0,
        max: 1.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    };

    let golden_manifest = ProbarManifest {
        source_model: "g.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![mk_layer("a"), mk_layer("b"), mk_layer("c")],
        golden_reference: None,
    };
    fs::write(
        golden_dir.path().join("manifest.json"),
        serde_json::to_string(&golden_manifest).expect("ser"),
    )
    .expect("write");

    // Current has only 1 layer
    let current = ProbarManifest {
        source_model: "c.apr".to_string(),
        timestamp: "t".to_string(),
        format: "APR".to_string(),
        layers: vec![mk_layer("a")],
        golden_reference: None,
    };

    generate_diff(golden_dir.path(), &current, output_dir.path()).expect("diff");

    let diff_content =
        fs::read_to_string(output_dir.path().join("diff_report.json")).expect("read");
    let diff: serde_json::Value = serde_json::from_str(&diff_content).expect("parse");

    // Only 1 layer compared (zip stops at shortest), so 0 diffs
    assert_eq!(diff["total_diffs"], 0);
}

// ========================================================================
// ProbarManifest Serialization Tests
// ========================================================================

#[test]
fn test_probar_manifest_serialize_deserialize_roundtrip() {
    let manifest = ProbarManifest {
        source_model: "roundtrip.apr".to_string(),
        timestamp: "2026-02-06T00:00:00Z".to_string(),
        format: "GGUF (llama.cpp)".to_string(),
        layers: vec![
            LayerSnapshot {
                name: "block_0".to_string(),
                index: 0,
                histogram: vec![10, 20, 30],
                mean: -0.1,
                std: 0.9,
                min: -5.0,
                max: 5.0,
                heatmap: Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                heatmap_width: Some(3),
                heatmap_height: Some(2),
            },
            LayerSnapshot {
                name: "block_1".to_string(),
                index: 1,
                histogram: vec![],
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                heatmap: None,
                heatmap_width: None,
                heatmap_height: None,
            },
        ],
        golden_reference: Some("/golden/ref".to_string()),
    };

    let json = serde_json::to_string(&manifest).expect("serialize");
    let loaded: ProbarManifest = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(loaded.source_model, manifest.source_model);
    assert_eq!(loaded.timestamp, manifest.timestamp);
    assert_eq!(loaded.format, manifest.format);
    assert_eq!(loaded.layers.len(), 2);
    assert_eq!(loaded.layers[0].name, "block_0");
    assert_eq!(
        loaded.layers[0].heatmap,
        Some(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    );
    assert_eq!(loaded.layers[1].histogram.len(), 0);
    assert_eq!(loaded.golden_reference, Some("/golden/ref".to_string()));
}

// ========================================================================
// LayerSnapshot Full Round-Trip with Heatmap
// ========================================================================

#[test]
fn test_layer_snapshot_full_roundtrip_with_heatmap() {
    let original = LayerSnapshot {
        name: "embed".to_string(),
        index: 99,
        histogram: (0..256).map(|i| i as u32 * 2).collect(),
        mean: -0.001,
        std: 0.999,
        min: -10.0,
        max: 10.0,
        heatmap: Some(vec![f32::MIN, 0.0, f32::MAX]),
        heatmap_width: Some(3),
        heatmap_height: Some(1),
    };

    let json = serde_json::to_string(&original).expect("serialize");
    let restored: LayerSnapshot = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(restored.name, "embed");
    assert_eq!(restored.index, 99);
    assert_eq!(restored.histogram.len(), 256);
    assert_eq!(restored.histogram[0], 0);
    assert_eq!(restored.histogram[255], 510);
    assert_eq!(restored.heatmap_width, Some(3));
    assert_eq!(restored.heatmap_height, Some(1));
}

#[test]
fn test_layer_snapshot_deserialize_with_null_optionals() {
    let json = r#"{
        "name": "null_test",
        "index": 0,
        "histogram": [],
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "heatmap": null,
        "heatmap_width": null,
        "heatmap_height": null
    }"#;
    let snapshot: LayerSnapshot = serde_json::from_str(json).expect("deserialize");
    assert_eq!(snapshot.name, "null_test");
    assert!(snapshot.heatmap.is_none());
    assert!(snapshot.heatmap_width.is_none());
    assert!(snapshot.heatmap_height.is_none());
}

// ========================================================================
// print_* No-Panic Tests
// ========================================================================

#[test]
fn test_print_summary_does_not_panic() {
    let layers = vec![LayerSnapshot {
        name: "l".to_string(),
        index: 0,
        histogram: vec![],
        mean: 0.0,
        std: 0.0,
        min: 0.0,
        max: 0.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];
    // Without golden
    print_summary(Path::new("/m.apr"), Path::new("/out"), "APR", &layers, None);
    // With golden
    print_summary(
        Path::new("/m.apr"),
        Path::new("/out"),
        "APR",
        &layers,
        Some(Path::new("/golden")),
    );
}

#[test]
fn test_print_summary_empty_layers_does_not_panic() {
    print_summary(Path::new("/m.apr"), Path::new("/out"), "GGUF", &[], None);
}

#[test]
fn test_print_generated_files_json_does_not_panic() {
    print_generated_files(ExportFormat::Json, Path::new("/out"), &[]);
}

#[test]
fn test_print_generated_files_png_does_not_panic() {
    let layers = vec![
        LayerSnapshot {
            name: "a".to_string(),
            index: 0,
            histogram: vec![],
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        },
        LayerSnapshot {
            name: "b".to_string(),
            index: 1,
            histogram: vec![],
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            heatmap: None,
            heatmap_width: None,
            heatmap_height: None,
        },
    ];
    print_generated_files(ExportFormat::Png, Path::new("/out"), &layers);
}

#[test]
fn test_print_generated_files_both_does_not_panic() {
    let layers = vec![LayerSnapshot {
        name: "c".to_string(),
        index: 2,
        histogram: vec![],
        mean: 0.0,
        std: 0.0,
        min: 0.0,
        max: 0.0,
        heatmap: None,
        heatmap_width: None,
        heatmap_height: None,
    }];
    print_generated_files(ExportFormat::Both, Path::new("/output"), &layers);
}

#[test]
fn test_print_integration_guide_does_not_panic() {
    print_integration_guide();
}
