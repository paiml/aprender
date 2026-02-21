use super::*;

#[test]
fn test_normalize_dtype_string_lowercase() {
    assert_eq!(normalize_dtype("f32"), "F32");
    assert_eq!(normalize_dtype("f16"), "F16");
    assert_eq!(normalize_dtype("q4_0"), "Q4_0");
    assert_eq!(normalize_dtype("q4_1"), "Q4_1");
    assert_eq!(normalize_dtype("q5_0"), "Q5_0");
    assert_eq!(normalize_dtype("q5_1"), "Q5_1");
    assert_eq!(normalize_dtype("q8_0"), "Q8_0");
    assert_eq!(normalize_dtype("q8_1"), "Q8_1");
    assert_eq!(normalize_dtype("q2_k"), "Q2_K");
    assert_eq!(normalize_dtype("q3_k"), "Q3_K");
    assert_eq!(normalize_dtype("q4_k"), "Q4_K");
    assert_eq!(normalize_dtype("q5_k"), "Q5_K");
    assert_eq!(normalize_dtype("q6_k"), "Q6_K");
    assert_eq!(normalize_dtype("q8_k"), "Q8_K");
    assert_eq!(normalize_dtype("iq2_xxs"), "IQ2_XXS");
    assert_eq!(normalize_dtype("iq2_xs"), "IQ2_XS");
    assert_eq!(normalize_dtype("iq3_xxs"), "IQ3_XXS");
    assert_eq!(normalize_dtype("iq1_s"), "IQ1_S");
}

#[test]
fn test_normalize_dtype_string_uppercase() {
    assert_eq!(normalize_dtype("F32"), "F32");
    assert_eq!(normalize_dtype("F16"), "F16");
    assert_eq!(normalize_dtype("Q4_0"), "Q4_0");
    assert_eq!(normalize_dtype("Q4_1"), "Q4_1");
    assert_eq!(normalize_dtype("Q5_0"), "Q5_0");
    assert_eq!(normalize_dtype("Q5_1"), "Q5_1");
    assert_eq!(normalize_dtype("Q8_0"), "Q8_0");
    assert_eq!(normalize_dtype("Q8_1"), "Q8_1");
    assert_eq!(normalize_dtype("Q2_K"), "Q2_K");
    assert_eq!(normalize_dtype("Q3_K"), "Q3_K");
    assert_eq!(normalize_dtype("Q4_K"), "Q4_K");
    assert_eq!(normalize_dtype("Q5_K"), "Q5_K");
    assert_eq!(normalize_dtype("Q6_K"), "Q6_K");
    assert_eq!(normalize_dtype("Q8_K"), "Q8_K");
    assert_eq!(normalize_dtype("IQ2_XXS"), "IQ2_XXS");
    assert_eq!(normalize_dtype("IQ2_XS"), "IQ2_XS");
    assert_eq!(normalize_dtype("IQ3_XXS"), "IQ3_XXS");
    assert_eq!(normalize_dtype("IQ1_S"), "IQ1_S");
}

#[test]
fn test_normalize_dtype_short_aliases() {
    // Short aliases without underscore (q2k, Q2K, etc.)
    assert_eq!(normalize_dtype("q2k"), "Q2_K");
    assert_eq!(normalize_dtype("Q2K"), "Q2_K");
    assert_eq!(normalize_dtype("q3k"), "Q3_K");
    assert_eq!(normalize_dtype("Q3K"), "Q3_K");
    assert_eq!(normalize_dtype("q4k"), "Q4_K");
    assert_eq!(normalize_dtype("Q4K"), "Q4_K");
    assert_eq!(normalize_dtype("q5k"), "Q5_K");
    assert_eq!(normalize_dtype("Q5K"), "Q5_K");
    assert_eq!(normalize_dtype("q6k"), "Q6_K");
    assert_eq!(normalize_dtype("Q6K"), "Q6_K");
    assert_eq!(normalize_dtype("q8k"), "Q8_K");
    assert_eq!(normalize_dtype("Q8K"), "Q8_K");
}

#[test]
fn test_normalize_dtype_bf16() {
    assert_eq!(normalize_dtype("bf16"), "BF16");
    assert_eq!(normalize_dtype("BF16"), "BF16");
}

#[test]
fn test_normalize_dtype_unknown_fallback() {
    // Catch-all: unknown types get uppercased
    assert_eq!(normalize_dtype("custom_type"), "CUSTOM_TYPE");
    assert_eq!(normalize_dtype("fp8"), "FP8");
    assert_eq!(normalize_dtype("int4"), "INT4");
    assert_eq!(normalize_dtype("26"), "26"); // Numeric code not in map
}

// ====================================================================
// Coverage: is_compatible_quant exhaustive branch tests
// ====================================================================

#[test]
fn test_is_compatible_quant_same_after_normalization() {
    // Same type after normalization returns true
    assert!(is_compatible_quant("f32", "F32"));
    assert!(is_compatible_quant("0", "F32"));
    assert!(is_compatible_quant("q4_k", "Q4_K"));
    assert!(is_compatible_quant("12", "Q4_K"));
    assert!(is_compatible_quant("q8_0", "Q8_0"));
    assert!(is_compatible_quant("8", "Q8_0"));
}

#[test]
fn test_is_compatible_quant_q5_q6_compatible() {
    // Q5 <-> Q6 compatibility (common import path)
    assert!(is_compatible_quant("Q5_0", "Q6_K"));
    assert!(is_compatible_quant("Q6_K", "Q5_0"));
    assert!(is_compatible_quant("Q5_K", "Q6_K"));
    assert!(is_compatible_quant("Q6_K", "Q5_K"));
    assert!(is_compatible_quant("Q5_1", "Q6_K"));
    assert!(is_compatible_quant("Q6_K", "Q5_1"));
}

#[test]
fn test_is_compatible_quant_q4_variants() {
    // Q4 <-> Q4 variants compatible
    assert!(is_compatible_quant("Q4_0", "Q4_K"));
    assert!(is_compatible_quant("Q4_K", "Q4_0"));
    assert!(is_compatible_quant("Q4_1", "Q4_K"));
    assert!(is_compatible_quant("Q4_K", "Q4_1"));
    assert!(is_compatible_quant("Q4_0", "Q4_1"));
}

#[test]
fn test_is_compatible_quant_q8_q6_compatible() {
    // Q8 <-> Q6 compatibility (downgrade)
    assert!(is_compatible_quant("Q8_0", "Q6_K"));
    assert!(is_compatible_quant("Q6_K", "Q8_0"));
    assert!(is_compatible_quant("Q8_K", "Q6_K"));
    assert!(is_compatible_quant("Q6_K", "Q8_K"));
}

#[test]
fn test_is_compatible_quant_incompatible_pairs() {
    // Truly incompatible pairs
    assert!(!is_compatible_quant("F32", "Q4_K"));
    assert!(!is_compatible_quant("Q4_K", "F32"));
    assert!(!is_compatible_quant("Q2_K", "Q8_0"));
    assert!(!is_compatible_quant("Q8_0", "Q2_K"));
    assert!(!is_compatible_quant("F16", "Q4_0"));
    assert!(!is_compatible_quant("BF16", "Q6_K"));
    assert!(!is_compatible_quant("Q3_K", "Q8_0"));
    assert!(!is_compatible_quant("F32", "F16"));
    assert!(!is_compatible_quant("IQ2_XXS", "Q4_K"));
}

#[test]
fn test_is_compatible_quant_with_numeric_codes() {
    // Using numeric GGUF codes should also work through normalization
    assert!(is_compatible_quant("12", "Q4_0")); // 12 = Q4_K, compatible with Q4_0
    assert!(is_compatible_quant("6", "14")); // Q5_0 <-> Q6_K compatible
    assert!(!is_compatible_quant("0", "12")); // F32 <-> Q4_K incompatible
}

// ====================================================================
// Coverage: map_gguf_to_apr_name all branches
// ====================================================================

#[test]
fn test_map_gguf_to_apr_name_attn_q_weight() {
    let (name, mapped) = map_gguf_to_apr_name("blk.0.attn_q.weight");
    assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_attn_q_bias() {
    let (name, mapped) = map_gguf_to_apr_name("blk.3.attn_q.bias");
    assert_eq!(name, "model.layers.3.self_attn.q_proj.bias");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_attn_k_weight() {
    let (name, mapped) = map_gguf_to_apr_name("blk.1.attn_k.weight");
    assert_eq!(name, "model.layers.1.self_attn.k_proj.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_attn_k_bias() {
    let (name, mapped) = map_gguf_to_apr_name("blk.7.attn_k.bias");
    assert_eq!(name, "model.layers.7.self_attn.k_proj.bias");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_attn_v_weight() {
    let (name, mapped) = map_gguf_to_apr_name("blk.2.attn_v.weight");
    assert_eq!(name, "model.layers.2.self_attn.v_proj.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_attn_v_bias() {
    let (name, mapped) = map_gguf_to_apr_name("blk.5.attn_v.bias");
    assert_eq!(name, "model.layers.5.self_attn.v_proj.bias");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_attn_output_weight() {
    let (name, mapped) = map_gguf_to_apr_name("blk.4.attn_output.weight");
    assert_eq!(name, "model.layers.4.self_attn.o_proj.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_attn_output_bias() {
    let (name, mapped) = map_gguf_to_apr_name("blk.0.attn_output.bias");
    assert_eq!(name, "model.layers.0.self_attn.o_proj.bias");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_attn_norm() {
    let (name, mapped) = map_gguf_to_apr_name("blk.6.attn_norm.weight");
    assert_eq!(name, "model.layers.6.input_layernorm.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_ffn_gate() {
    let (name, mapped) = map_gguf_to_apr_name("blk.0.ffn_gate.weight");
    assert_eq!(name, "model.layers.0.mlp.gate_proj.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_ffn_up() {
    let (name, mapped) = map_gguf_to_apr_name("blk.10.ffn_up.weight");
    assert_eq!(name, "model.layers.10.mlp.up_proj.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_ffn_down() {
    let (name, mapped) = map_gguf_to_apr_name("blk.31.ffn_down.weight");
    assert_eq!(name, "model.layers.31.mlp.down_proj.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_ffn_norm() {
    let (name, mapped) = map_gguf_to_apr_name("blk.0.ffn_norm.weight");
    assert_eq!(name, "model.layers.0.post_attention_layernorm.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_unknown_layer_suffix() {
    // Unknown suffix within blk.N.* returns unchanged
    let (name, mapped) = map_gguf_to_apr_name("blk.0.unknown_suffix.weight");
    assert_eq!(name, "blk.0.unknown_suffix.weight");
    assert!(!mapped);
}

#[test]
fn test_map_gguf_to_apr_name_token_embd() {
    let (name, mapped) = map_gguf_to_apr_name("token_embd.weight");
    assert_eq!(name, "model.embed_tokens.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_output_weight() {
    let (name, mapped) = map_gguf_to_apr_name("output.weight");
    assert_eq!(name, "lm_head.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_output_norm() {
    let (name, mapped) = map_gguf_to_apr_name("output_norm.weight");
    assert_eq!(name, "model.norm.weight");
    assert!(mapped);
}

#[test]
fn test_map_gguf_to_apr_name_unknown_non_layer() {
    // Unknown non-layer tensor returns unchanged
    let (name, mapped) = map_gguf_to_apr_name("some_other_tensor");
    assert_eq!(name, "some_other_tensor");
    assert!(!mapped);
}

#[test]
fn test_map_gguf_to_apr_name_already_apr_style() {
    // APR-style names pass through unchanged
    let (name, mapped) = map_gguf_to_apr_name("model.layers.0.self_attn.q_proj.weight");
    assert_eq!(name, "model.layers.0.self_attn.q_proj.weight");
    assert!(!mapped);
}

// ====================================================================
// Coverage: build_cross_format_map tests
// ====================================================================

#[test]
fn test_build_cross_format_map_gguf_names() {
    use crate::format::rosetta::TensorInfo;

    let tensors = vec![
        TensorInfo {
            name: "blk.0.attn_q.weight".to_string(),
            dtype: "Q4_K".to_string(),
            shape: vec![4096, 4096],
            size_bytes: 1000,
            stats: None,
        },
        TensorInfo {
            name: "token_embd.weight".to_string(),
            dtype: "F16".to_string(),
            shape: vec![32000, 4096],
            size_bytes: 2000,
            stats: None,
        },
    ];

    let map = build_cross_format_map(&tensors);

    // Original GGUF names should be present
    assert!(map.contains_key("blk.0.attn_q.weight"));
    assert!(map.contains_key("token_embd.weight"));

    // Mapped APR names should also be present
    assert!(map.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(map.contains_key("model.embed_tokens.weight"));
}

#[test]
fn test_build_cross_format_map_hf_names() {
    use crate::format::rosetta::TensorInfo;

    // HF/APR names that won't be mapped (no blk. prefix, not in non-layer map)
    let tensors = vec![TensorInfo {
        name: "model.layers.0.self_attn.q_proj.weight".to_string(),
        dtype: "F16".to_string(),
        shape: vec![4096, 4096],
        size_bytes: 1000,
        stats: None,
    }];

    let map = build_cross_format_map(&tensors);

    // Original name present
    assert!(map.contains_key("model.layers.0.self_attn.q_proj.weight"));
    // Only one entry (no mapping was applied)
    assert_eq!(map.len(), 1);
}

#[test]
fn test_build_cross_format_map_empty() {
    let tensors: Vec<crate::format::rosetta::TensorInfo> = vec![];
    let map = build_cross_format_map(&tensors);
    assert!(map.is_empty());
}

#[test]
fn test_build_cross_format_map_unknown_suffix() {
    use crate::format::rosetta::TensorInfo;

    // Unknown suffix within blk.N.* -- no mapping added
    let tensors = vec![TensorInfo {
        name: "blk.0.custom_thing.weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![100],
        size_bytes: 400,
        stats: None,
    }];

    let map = build_cross_format_map(&tensors);

    // Only the original name is present (no mapping)
    assert!(map.contains_key("blk.0.custom_thing.weight"));
    assert_eq!(map.len(), 1);
}

// ====================================================================
// Coverage: diff_inspections public API tests
// ====================================================================

#[test]
fn test_diff_inspections_identical() {
    let r = make_report(FormatType::Apr, 1000, 100, Some("llama"), None);
    let report = diff_inspections(&r, &r, "model_a.apr", "model_b.apr", DiffOptions::default());
    assert!(report.is_identical());
    assert_eq!(report.path1, "model_a.apr");
    assert_eq!(report.path2, "model_b.apr");
    assert_eq!(report.format1, "APR");
    assert_eq!(report.format2, "APR");
    assert!(report.inspection1.is_none());
    assert!(report.inspection2.is_none());
}

#[test]
fn test_diff_inspections_different_formats() {
    let r1 = make_report(FormatType::Apr, 1000, 100, Some("llama"), Some("Q4_K"));
    let r2 = make_report(FormatType::Gguf, 2000, 200, Some("qwen2"), Some("Q8_0"));
    let report = diff_inspections(&r1, &r2, "a.apr", "b.gguf", DiffOptions::default());
    assert!(!report.is_identical());
    assert!(!report.same_format());
    assert!(report.differences.iter().any(|d| d.field == "format"));
    assert!(report.differences.iter().any(|d| d.field == "file_size"));
    assert!(report.differences.iter().any(|d| d.field == "total_params"));
    assert!(report.differences.iter().any(|d| d.field == "architecture"));
    assert!(report.differences.iter().any(|d| d.field == "quantization"));
}

#[test]
fn test_diff_inspections_with_tensors() {
    use crate::format::rosetta::TensorInfo;

    let mut r1 = make_report(FormatType::Apr, 1000, 100, None, None);
    r1.tensors = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![10, 20],
        size_bytes: 800,
        stats: None,
    }];

    let mut r2 = make_report(FormatType::Apr, 1000, 100, None, None);
    r2.tensors = vec![TensorInfo {
        name: "weight".to_string(),
        dtype: "F32".to_string(),
        shape: vec![20, 10], // Transposed -- compatible
        size_bytes: 800,
        stats: None,
    }];

    let report = diff_inspections(&r1, &r2, "a.apr", "b.apr", DiffOptions::default());
    // Transposed shapes are compatible, so no shape diff
    assert!(report.is_identical());
}
