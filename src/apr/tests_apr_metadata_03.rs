
// ============================================================================
// AprMetadata extra fields
// ============================================================================

#[test]
fn test_apr_metadata_extra_fields_preserved() {
    let json = r#"{"hidden_size": 512, "custom_field": "hello", "custom_int": 42}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.hidden_size, Some(512));
    assert_eq!(
        m.extra.get("custom_field").and_then(|v| v.as_str()),
        Some("hello")
    );
    assert_eq!(m.extra.get("custom_int").and_then(|v| v.as_i64()), Some(42));
}

#[test]
fn test_apr_metadata_n_embd_alias() {
    let json = r#"{"n_embd": 768}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.hidden_size, Some(768));
}

#[test]
fn test_apr_metadata_n_layers_alias() {
    let json = r#"{"n_layers": 12}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.num_layers, Some(12));
}

#[test]
fn test_apr_metadata_n_heads_alias() {
    let json = r#"{"n_heads": 16}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.num_heads, Some(16));
}

#[test]
fn test_apr_metadata_n_head_alias() {
    let json = r#"{"n_head": 8}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.num_heads, Some(8));
}

#[test]
fn test_apr_metadata_n_layer_alias() {
    let json = r#"{"n_layer": 6}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.num_layers, Some(6));
}

#[test]
fn test_apr_metadata_ffn_dim_alias() {
    let json = r#"{"ffn_dim": 4096}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.intermediate_size, Some(4096));
}

#[test]
fn test_apr_metadata_n_inner_alias() {
    let json = r#"{"n_inner": 3072}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.intermediate_size, Some(3072));
}

#[test]
fn test_apr_metadata_max_seq_len_alias() {
    let json = r#"{"max_seq_len": 2048}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.max_position_embeddings, Some(2048));
}

#[test]
fn test_apr_metadata_n_ctx_alias() {
    let json = r#"{"n_ctx": 8192}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.max_position_embeddings, Some(8192));
}

#[test]
fn test_apr_metadata_layer_norm_eps_alias() {
    let json = r#"{"layer_norm_eps": 0.00001}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert!(m.rms_norm_eps.is_some());
}

#[test]
fn test_apr_metadata_n_kv_heads_alias() {
    let json = r#"{"n_kv_heads": 4}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.num_kv_heads, Some(4));
}

// ============================================================================
// dequant re-exports
// ============================================================================

#[test]
fn test_f16_to_f32_re_export() {
    // Test that re-exported f16_to_f32 works
    assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6); // 1.0 in F16
    assert!((f16_to_f32(0x0000) - 0.0).abs() < 1e-6); // 0.0 in F16
    assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-6); // 2.0 in F16
}

#[test]
fn test_dtype_to_ggml_qtype_re_export() {
    assert_eq!(dtype_to_ggml_qtype("Q4_K"), Some(12));
    assert_eq!(dtype_to_ggml_qtype("Q6_K"), Some(14));
    assert_eq!(dtype_to_ggml_qtype("Q8_0"), Some(8));
    assert_eq!(dtype_to_ggml_qtype("F32"), None);
    assert_eq!(dtype_to_ggml_qtype("F16"), None);
}

#[test]
fn test_is_quantized_dtype_re_export() {
    assert!(is_quantized_dtype("Q4_K"));
    assert!(is_quantized_dtype("Q6_K"));
    assert!(is_quantized_dtype("Q8_0"));
    assert!(!is_quantized_dtype("F32"));
    assert!(!is_quantized_dtype("F16"));
    assert!(!is_quantized_dtype("BF16"));
}
