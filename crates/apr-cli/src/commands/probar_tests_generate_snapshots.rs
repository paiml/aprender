
#[test]
fn test_generate_snapshots_filter_matches_subset() {
    // Filter for "block_3" out of 5 layers
    let snapshots = generate_snapshots(None, 5, Some("block_3"));
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].name, "block_3");
    assert_eq!(snapshots[0].index, 3);
}

#[test]
fn test_generate_snapshots_filter_matches_none_returns_fallback() {
    // Filter for something that doesn't match any layer
    let snapshots = generate_snapshots(None, 3, Some("nonexistent"));
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].name, "fallback");
}

#[test]
fn test_generate_snapshots_filter_partial_match() {
    // "block_" matches all layers
    let snapshots = generate_snapshots(None, 10, Some("block_"));
    assert_eq!(snapshots.len(), 10);
}

#[test]
fn test_generate_snapshots_zero_n_layers() {
    let snapshots = generate_snapshots(None, 0, None);

    // 0 layers => empty => fallback
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].name, "fallback");
}

#[test]
fn test_generate_snapshots_fallback_stats() {
    let snapshots = generate_snapshots(None, 0, None);
    let fallback = &snapshots[0];
    assert_eq!(fallback.mean, 0.0);
    assert_eq!(fallback.std, 0.0);
    assert_eq!(fallback.min, 0.0);
    assert_eq!(fallback.max, 0.0);
    assert!(fallback.heatmap.is_none());
    assert!(fallback.heatmap_width.is_none());
    assert!(fallback.heatmap_height.is_none());
}

// ========================================================================
// detect_layer_count Tests
// ========================================================================

#[test]
fn test_detect_layer_count_empty_report() {
    use aprender::format::rosetta::{FormatType, InspectionReport};
    let report = InspectionReport {
        format: FormatType::Gguf,
        file_size: 0,
        total_params: 0,
        quantization: None,
        architecture: None,
        tensors: vec![],
        metadata: std::collections::BTreeMap::new(),
    };
    assert_eq!(detect_layer_count(&report), 0);
}

#[test]
fn test_detect_layer_count_gguf_naming() {
    use aprender::format::rosetta::{FormatType, InspectionReport, TensorInfo};
    let tensors = vec![
        TensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            dtype: "Q4_K".to_string(),
            shape: vec![],
            size_bytes: 0,
            stats: None,
        },
        TensorInfo {
            name: "blk.1.attn_q.weight".to_string(),
            dtype: "Q4_K".to_string(),
            shape: vec![],
            size_bytes: 0,
            stats: None,
        },
        TensorInfo {
            name: "blk.23.ffn_gate.weight".to_string(),
            dtype: "Q4_K".to_string(),
            shape: vec![],
            size_bytes: 0,
            stats: None,
        },
    ];
    let report = InspectionReport {
        format: FormatType::Gguf,
        file_size: 0,
        total_params: 0,
        quantization: None,
        architecture: None,
        tensors,
        metadata: std::collections::BTreeMap::new(),
    };
    assert_eq!(detect_layer_count(&report), 24); // 0..23 inclusive = 24 layers
}

#[test]
fn test_detect_layer_count_safetensors_naming() {
    use aprender::format::rosetta::{FormatType, InspectionReport, TensorInfo};
    let tensors = vec![
        TensorInfo {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dtype: "F16".to_string(),
            shape: vec![],
            size_bytes: 0,
            stats: None,
        },
        TensorInfo {
            name: "model.layers.27.mlp.up_proj.weight".to_string(),
            dtype: "F16".to_string(),
            shape: vec![],
            size_bytes: 0,
            stats: None,
        },
    ];
    let report = InspectionReport {
        format: FormatType::SafeTensors,
        file_size: 0,
        total_params: 0,
        quantization: None,
        architecture: None,
        tensors,
        metadata: std::collections::BTreeMap::new(),
    };
    assert_eq!(detect_layer_count(&report), 28); // 0..27 inclusive = 28 layers
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
