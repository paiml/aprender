
// ============================================================================
// AprMetadata serialization roundtrip
// ============================================================================

#[test]
fn test_apr_metadata_serialization_roundtrip() {
    let m = AprMetadata {
        model_type: Some("transformer_lm".to_string()),
        architecture: Some("qwen2".to_string()),
        hidden_size: Some(2048),
        num_layers: Some(24),
        num_heads: Some(16),
        num_kv_heads: Some(4),
        vocab_size: Some(152064),
        intermediate_size: Some(8192),
        max_position_embeddings: Some(32768),
        rope_theta: Some(1000000.0),
        rope_type: Some(2),
        rms_norm_eps: Some(1e-6),
        ..Default::default()
    };

    let json = serde_json::to_string(&m).expect("serialize");
    let restored: AprMetadata = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.hidden_size, Some(2048));
    assert_eq!(restored.architecture, Some("qwen2".to_string()));
    assert!(restored.is_transformer());
}

#[test]
fn test_apr_metadata_debug() {
    let m = AprMetadata::default();
    let debug = format!("{:?}", m);
    assert!(debug.contains("AprMetadata"));
}

#[test]
fn test_apr_metadata_clone() {
    let m = AprMetadata {
        hidden_size: Some(512),
        ..Default::default()
    };
    let cloned = m.clone();
    assert_eq!(cloned.hidden_size, Some(512));
}
