
// ====================================================================
// T-QKV-03: SafeTensors->APR->GGUF round-trip test
// ====================================================================

/// T-QKV-03: Export to GGUF preserves tensor count and separate Q/K/V
#[test]
fn test_t_qkv_03_safetensors_apr_gguf() {
    use crate::format::gguf::GgufReader;

    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::qwen2_gqa())
        .import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        })
        .export_to_gguf();

    let gguf_path = h.output_path().expect("GGUF output exists");
    let gguf_data = std::fs::read(gguf_path).expect("read GGUF");
    let reader = GgufReader::from_bytes(gguf_data).expect("T-QKV-03: parse GGUF");

    // Verify GGUF is valid and has tensors
    assert!(
        !reader.tensors.is_empty(),
        "T-QKV-03: GGUF must contain tensors"
    );

    // T-QKV-03 key check: Q/K/V must be SEPARATE (not fused as attn_qkv)
    let names: Vec<&str> = reader.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(
        !names.iter().any(|n| n.contains("attn_qkv")),
        "T-QKV-03: Must NOT have fused attn_qkv tensor. Found: {names:?}"
    );

    // Should have separate attn_q, attn_k, attn_v
    let has_q = names.iter().any(|n| n.contains("attn_q."));
    let has_k = names.iter().any(|n| n.contains("attn_k."));
    let has_v = names.iter().any(|n| n.contains("attn_v."));
    assert!(
        has_q && has_k && has_v,
        "T-QKV-03: Must have separate attn_q, attn_k, attn_v. Found: {names:?}"
    );
}

// ====================================================================
// T-QKV-04: Multi-hop chain test (ST->APR->GGUF->APR->ST)
// ====================================================================

/// T-QKV-04: Full multi-hop chain preserves tensor data
///
/// Bug 210: Uses qwen2_gqa() (with attention biases) instead of llama_style().
/// llama_style() lacks q_proj.bias, so architecture is correctly detected as "llama",
/// but the ST→GGUF→ST name round-trip only works for architectures with full name
/// mapping support. Qwen2 is the primary tested architecture.
#[test]
fn test_t_qkv_04_multihop_st_apr_gguf_apr_st() {
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::qwen2_gqa())
        .import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        })                                           // ST -> APR
        .export_to_gguf()                            // APR -> GGUF
        .reimport_to_apr()                           // GGUF -> APR
        .export_to_safetensors(); // APR -> ST

    // Verify final SafeTensors matches original input
    let result = h.verify_safetensors();
    assert!(
        result.passed(),
        "T-QKV-04: Multi-hop ST->APR->GGUF->APR->ST must preserve data. \
         Mismatches: {:?}",
        result.mismatches
    );
}
