
// ============================================================================
// Layer weight presence tests
// ============================================================================

#[test]
fn test_apr_layer_has_attn_weights() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    assert!(!apr.layers.is_empty());
    let layer = &apr.layers[0];

    // Attention weights should exist
    assert!(!layer.attn_norm_weight.is_empty());
    assert!(!layer.qkv_weight.is_empty());
    assert!(!layer.attn_output_weight.is_empty());
}

#[test]
fn test_apr_layer_has_ffn_weights() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    assert!(!apr.layers.is_empty());
    let layer = &apr.layers[0];

    // FFN weights should exist
    assert!(
        layer
            .ffn_norm_weight
            .as_ref()
            .is_some_and(|v| !v.is_empty())
            || layer.ffn_norm_weight.is_none()
    );
    assert!(!layer.ffn_up_weight.is_empty());
    assert!(!layer.ffn_down_weight.is_empty());
}

#[test]
fn test_apr_layer_llama_has_gate() {
    let gguf_data = build_minimal_llama_gguf(32, 64, 128, 4, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    assert!(!apr.layers.is_empty());
    let layer = &apr.layers[0];

    // LLaMA architecture should have gate weights (SwiGLU)
    assert!(layer.ffn_gate_weight.is_some());
}

// ============================================================================
// Phi2 specific tests
// ============================================================================

#[test]
fn test_apr_phi2_architecture() {
    let gguf_data = build_minimal_phi2_gguf(32, 64, 128, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    assert_eq!(apr.config.architecture, "phi2");
}

#[test]
fn test_apr_phi2_has_layers() {
    let gguf_data = build_minimal_phi2_gguf(32, 64, 128, 4);
    let apr = GgufToAprConverter::convert(&gguf_data).expect("test value should be present");

    assert!(!apr.layers.is_empty());
}
