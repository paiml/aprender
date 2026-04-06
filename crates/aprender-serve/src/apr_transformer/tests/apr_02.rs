
#[test]
fn test_from_apr_bytes_metadata_beyond_file() {
    use crate::apr_transformer::AprTransformer;
    let mut data = vec![0u8; 128];
    // Set magic to "APR\0"
    data[0] = b'A';
    data[1] = b'P';
    data[2] = b'R';
    data[3] = 0;
    // Set metadata_offset to 0 (start of file is fine)
    // Set metadata_size to something huge (beyond file)
    data[20..24].copy_from_slice(&10000u32.to_le_bytes());
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("extends beyond") || err.contains("Metadata"),
        "Expected metadata error, got: {}",
        err
    );
}

#[test]
fn test_from_apr_bytes_valid_minimal_header() {
    use crate::apr_transformer::AprTransformer;
    // Build minimal valid APR v2 with empty metadata and no tensors
    let mut data = vec![0u8; 256];
    // Magic "APR2"
    data[0] = b'A';
    data[1] = b'P';
    data[2] = b'R';
    data[3] = b'2';
    // tensor_count = 0
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    // metadata_offset = 64
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    // metadata_size = 2 (empty JSON: "{}")
    data[20..24].copy_from_slice(&2u32.to_le_bytes());
    // Write empty JSON at offset 64
    data[64] = b'{';
    data[65] = b'}';
    // tensor_index_offset = 66
    data[24..32].copy_from_slice(&66u64.to_le_bytes());
    // data_offset = 128
    data[32..40].copy_from_slice(&128u64.to_le_bytes());

    // GH-478: Empty metadata defaults to zero dimensions, which the contract gate
    // now correctly rejects before weight loading. Previously fell through to
    // "missing embedding" error.
    let result = AprTransformer::from_apr_bytes(&data);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("embedding") || err.contains("FATAL")
            || err.contains("not found") || err.contains("contract_gate"),
        "Expected missing tensor or contract gate error, got: {}",
        err
    );
}

// ============================================================================
// LayerActivation and ForwardTrace with multiple layers
// ============================================================================

#[test]
fn test_forward_trace_multiple_layers() {
    let stats = ActivationStats::from_slice(&[1.0, 2.0, 3.0]);
    let layers: Vec<LayerActivation> = (0..4)
        .map(|i| LayerActivation {
            layer_idx: i,
            attn_norm_stats: stats.clone(),
            qkv_stats: stats.clone(),
            attn_out_stats: stats.clone(),
            ffn_norm_stats: stats.clone(),
            ffn_out_stats: stats.clone(),
            output_stats: stats.clone(),
        })
        .collect();

    let trace = ForwardTrace {
        input_tokens: vec![1, 2, 3, 4, 5],
        embed_stats: stats.clone(),
        layer_activations: layers,
        final_norm_stats: stats.clone(),
        logits_stats: stats,
        logits: vec![0.1; 100],
    };

    assert_eq!(trace.layer_activations.len(), 4);
    assert_eq!(trace.layer_activations[3].layer_idx, 3);
    assert_eq!(trace.logits.len(), 100);
}

#[test]
fn test_layer_activation_with_different_stats() {
    let attn_stats = ActivationStats::from_slice(&[1.0, 2.0, 3.0]);
    let ffn_stats = ActivationStats::from_slice(&[10.0, 20.0, 30.0]);

    let layer = LayerActivation {
        layer_idx: 0,
        attn_norm_stats: attn_stats.clone(),
        qkv_stats: attn_stats.clone(),
        attn_out_stats: attn_stats.clone(),
        ffn_norm_stats: ffn_stats.clone(),
        ffn_out_stats: ffn_stats.clone(),
        output_stats: ffn_stats,
    };

    assert!((layer.attn_norm_stats.mean - 2.0).abs() < 0.01);
    assert!((layer.ffn_norm_stats.mean - 20.0).abs() < 0.01);
    assert!((layer.output_stats.mean - 20.0).abs() < 0.01);
}
