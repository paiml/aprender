use super::*;

// ========================================================================
// GH-219 Coverage Gap: Architecture pure functions
// from_model_type, is_inference_verified, display_name,
// qwen2_map_name (GGUF tensors), gpt2_map_name,
// split_gpt2_fused_qkv, split_gpt2_fused_qkv_raw,
// Source::parse
// ========================================================================

// -------------------------------------------------------------------------
// Architecture::from_model_type
// -------------------------------------------------------------------------

#[test]
fn test_from_model_type_qwen2_variants_gh219() {
    assert_eq!(Architecture::from_model_type("qwen2"), Some(Architecture::Qwen2));
    assert_eq!(Architecture::from_model_type("qwen"), Some(Architecture::Qwen2));
    assert_eq!(Architecture::from_model_type("qwen2.5"), Some(Architecture::Qwen2));
}

#[test]
fn test_from_model_type_qwen3_gh219() {
    assert_eq!(Architecture::from_model_type("qwen3"), Some(Architecture::Qwen3));
}

#[test]
fn test_from_model_type_llama_variants_gh219() {
    assert_eq!(Architecture::from_model_type("llama"), Some(Architecture::Llama));
    assert_eq!(Architecture::from_model_type("llama2"), Some(Architecture::Llama));
    assert_eq!(Architecture::from_model_type("llama3"), Some(Architecture::Llama));
}

#[test]
fn test_from_model_type_whisper_gh219() {
    assert_eq!(Architecture::from_model_type("whisper"), Some(Architecture::Whisper));
}

#[test]
fn test_from_model_type_bert_gh219() {
    assert_eq!(Architecture::from_model_type("bert"), Some(Architecture::Bert));
}

#[test]
fn test_from_model_type_gpt2_gh219() {
    assert_eq!(Architecture::from_model_type("gpt2"), Some(Architecture::Gpt2));
}

#[test]
fn test_from_model_type_phi_variants_gh219() {
    assert_eq!(Architecture::from_model_type("phi"), Some(Architecture::Phi));
    assert_eq!(Architecture::from_model_type("phi3"), Some(Architecture::Phi));
    assert_eq!(Architecture::from_model_type("phi4"), Some(Architecture::Phi));
}

#[test]
fn test_from_model_type_llama_derivatives_gh219() {
    for name in &["smollm", "smollm2", "granite", "granite3", "nemotron", "mistral", "gemma", "gemma2", "gemma3"] {
        assert_eq!(
            Architecture::from_model_type(name),
            Some(Architecture::Llama),
            "Expected Llama for derivative: {name}"
        );
    }
}

#[test]
fn test_from_model_type_case_insensitive_gh219() {
    assert_eq!(Architecture::from_model_type("QWEN2"), Some(Architecture::Qwen2));
    assert_eq!(Architecture::from_model_type("Llama"), Some(Architecture::Llama));
    assert_eq!(Architecture::from_model_type("GPT2"), Some(Architecture::Gpt2));
}

#[test]
fn test_from_model_type_unknown_gh219() {
    assert_eq!(Architecture::from_model_type("unknown_arch"), None);
    assert_eq!(Architecture::from_model_type(""), None);
    assert_eq!(Architecture::from_model_type("falcon"), None);
}

// -------------------------------------------------------------------------
// Architecture::is_inference_verified
// -------------------------------------------------------------------------

#[test]
fn test_is_inference_verified_true_gh219() {
    assert!(Architecture::Qwen2.is_inference_verified());
    assert!(Architecture::Qwen3.is_inference_verified());
    assert!(Architecture::Llama.is_inference_verified());
    assert!(Architecture::Phi.is_inference_verified());
}

#[test]
fn test_is_inference_verified_false_gh219() {
    assert!(!Architecture::Auto.is_inference_verified());
    assert!(!Architecture::Whisper.is_inference_verified());
    assert!(!Architecture::Bert.is_inference_verified());
    assert!(!Architecture::Gpt2.is_inference_verified());
}

// -------------------------------------------------------------------------
// Architecture::display_name
// -------------------------------------------------------------------------

#[test]
fn test_display_name_all_architectures_gh219() {
    assert_eq!(Architecture::Auto.display_name(), "auto-detected");
    assert_eq!(Architecture::Whisper.display_name(), "Whisper");
    assert_eq!(Architecture::Llama.display_name(), "LLaMA");
    assert_eq!(Architecture::Bert.display_name(), "BERT");
    assert_eq!(Architecture::Qwen2.display_name(), "Qwen2");
    assert_eq!(Architecture::Qwen3.display_name(), "Qwen3");
    assert_eq!(Architecture::Gpt2.display_name(), "GPT-2");
    assert_eq!(Architecture::Phi.display_name(), "Phi");
}

// -------------------------------------------------------------------------
// Architecture::qwen2_map_name (GGUF blk.N.* → model.layers.N.*)
// -------------------------------------------------------------------------

#[test]
fn test_qwen2_map_name_attn_tensors_gh219() {
    let arch = Architecture::Qwen2;
    assert_eq!(arch.map_name("blk.0.attn_q.weight"), "model.layers.0.self_attn.q_proj.weight");
    assert_eq!(arch.map_name("blk.0.attn_q.bias"), "model.layers.0.self_attn.q_proj.bias");
    assert_eq!(arch.map_name("blk.3.attn_k.weight"), "model.layers.3.self_attn.k_proj.weight");
    assert_eq!(arch.map_name("blk.3.attn_k.bias"), "model.layers.3.self_attn.k_proj.bias");
    assert_eq!(arch.map_name("blk.5.attn_v.weight"), "model.layers.5.self_attn.v_proj.weight");
    assert_eq!(arch.map_name("blk.5.attn_v.bias"), "model.layers.5.self_attn.v_proj.bias");
    assert_eq!(arch.map_name("blk.2.attn_output.weight"), "model.layers.2.self_attn.o_proj.weight");
    assert_eq!(arch.map_name("blk.2.attn_output.bias"), "model.layers.2.self_attn.o_proj.bias");
}

#[test]
fn test_qwen2_map_name_norm_tensors_gh219() {
    let arch = Architecture::Qwen2;
    assert_eq!(arch.map_name("blk.0.attn_norm.weight"), "model.layers.0.input_layernorm.weight");
    assert_eq!(arch.map_name("blk.7.ffn_norm.weight"), "model.layers.7.post_attention_layernorm.weight");
}

#[test]
fn test_qwen2_map_name_ffn_tensors_gh219() {
    let arch = Architecture::Qwen2;
    assert_eq!(arch.map_name("blk.1.ffn_gate.weight"), "model.layers.1.mlp.gate_proj.weight");
    assert_eq!(arch.map_name("blk.1.ffn_up.weight"), "model.layers.1.mlp.up_proj.weight");
    assert_eq!(arch.map_name("blk.1.ffn_down.weight"), "model.layers.1.mlp.down_proj.weight");
}

#[test]
fn test_qwen2_map_name_global_tensors_gh219() {
    let arch = Architecture::Qwen2;
    assert_eq!(arch.map_name("token_embd.weight"), "model.embed_tokens.weight");
    assert_eq!(arch.map_name("output.weight"), "lm_head.weight");
    assert_eq!(arch.map_name("output_norm.weight"), "model.norm.weight");
}

#[test]
fn test_qwen2_map_name_unknown_suffix_gh219() {
    let arch = Architecture::Qwen2;
    // Unknown GGUF suffix should be preserved
    assert_eq!(arch.map_name("blk.0.custom_layer.weight"), "model.layers.0.custom_layer.weight");
    // Unknown global name should be preserved
    assert_eq!(arch.map_name("some_unknown_tensor"), "some_unknown_tensor");
}

#[test]
fn test_qwen3_uses_qwen2_mapping_gh219() {
    // Qwen3 should use the same mapping as Qwen2
    assert_eq!(
        Architecture::Qwen3.map_name("blk.0.attn_q.weight"),
        Architecture::Qwen2.map_name("blk.0.attn_q.weight")
    );
    assert_eq!(
        Architecture::Qwen3.map_name("token_embd.weight"),
        Architecture::Qwen2.map_name("token_embd.weight")
    );
}

// -------------------------------------------------------------------------
// Architecture::gpt2_map_name
// -------------------------------------------------------------------------

#[test]
fn test_gpt2_map_name_layer_tensors_gh219() {
    let arch = Architecture::Gpt2;
    assert_eq!(arch.map_name("transformer.h.0.ln_1.weight"), "model.layers.0.input_layernorm.weight");
    assert_eq!(arch.map_name("transformer.h.0.ln_1.bias"), "model.layers.0.input_layernorm.bias");
    assert_eq!(arch.map_name("transformer.h.2.ln_2.weight"), "model.layers.2.post_attention_layernorm.weight");
    assert_eq!(arch.map_name("transformer.h.2.ln_2.bias"), "model.layers.2.post_attention_layernorm.bias");
}

#[test]
fn test_gpt2_map_name_attn_tensors_gh219() {
    let arch = Architecture::Gpt2;
    assert_eq!(arch.map_name("transformer.h.0.attn.c_attn.weight"), "model.layers.0.self_attn.c_attn.weight");
    assert_eq!(arch.map_name("transformer.h.0.attn.c_attn.bias"), "model.layers.0.self_attn.c_attn.bias");
    assert_eq!(arch.map_name("transformer.h.1.attn.c_proj.weight"), "model.layers.1.self_attn.o_proj.weight");
    assert_eq!(arch.map_name("transformer.h.1.attn.c_proj.bias"), "model.layers.1.self_attn.o_proj.bias");
}

#[test]
fn test_gpt2_map_name_mlp_tensors_gh219() {
    let arch = Architecture::Gpt2;
    assert_eq!(arch.map_name("transformer.h.0.mlp.c_fc.weight"), "model.layers.0.mlp.up_proj.weight");
    assert_eq!(arch.map_name("transformer.h.0.mlp.c_fc.bias"), "model.layers.0.mlp.up_proj.bias");
    assert_eq!(arch.map_name("transformer.h.0.mlp.c_proj.weight"), "model.layers.0.mlp.down_proj.weight");
    assert_eq!(arch.map_name("transformer.h.0.mlp.c_proj.bias"), "model.layers.0.mlp.down_proj.bias");
}

#[test]
fn test_gpt2_map_name_global_tensors_gh219() {
    let arch = Architecture::Gpt2;
    assert_eq!(arch.map_name("transformer.wte.weight"), "model.embed_tokens.weight");
    assert_eq!(arch.map_name("transformer.wpe.weight"), "model.position_embedding.weight");
    assert_eq!(arch.map_name("transformer.ln_f.weight"), "model.norm.weight");
    assert_eq!(arch.map_name("transformer.ln_f.bias"), "model.norm.bias");
}

#[test]
fn test_gpt2_map_name_safetensors_prefix_gh219() {
    // GH-255: SafeTensors uses "h.N.*" without "transformer." prefix
    let arch = Architecture::Gpt2;
    assert_eq!(arch.map_name("h.0.ln_1.weight"), "model.layers.0.input_layernorm.weight");
    assert_eq!(arch.map_name("h.3.attn.c_attn.weight"), "model.layers.3.self_attn.c_attn.weight");
    assert_eq!(arch.map_name("h.0.mlp.c_fc.weight"), "model.layers.0.mlp.up_proj.weight");
}

#[test]
fn test_gpt2_map_name_global_without_transformer_prefix_gh219() {
    let arch = Architecture::Gpt2;
    // Without "transformer." prefix
    assert_eq!(arch.map_name("wte.weight"), "model.embed_tokens.weight");
    assert_eq!(arch.map_name("wpe.weight"), "model.position_embedding.weight");
    assert_eq!(arch.map_name("ln_f.weight"), "model.norm.weight");
    assert_eq!(arch.map_name("ln_f.bias"), "model.norm.bias");
}

#[test]
fn test_gpt2_map_name_unknown_gh219() {
    let arch = Architecture::Gpt2;
    assert_eq!(arch.map_name("unknown_tensor"), "unknown_tensor");
    assert_eq!(arch.map_name("transformer.h.0.custom.weight"), "model.layers.0.custom.weight");
}

// -------------------------------------------------------------------------
// Architecture::split_gpt2_fused_qkv (f32)
// -------------------------------------------------------------------------

#[test]
fn test_split_gpt2_fused_qkv_bias_gh219() {
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Bias: 1D tensor [3*hidden] = [6] for hidden=2
    let bias_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    tensors.insert(
        "model.layers.0.self_attn.c_attn.bias".to_string(),
        (bias_data, vec![6]),
    );

    Architecture::split_gpt2_fused_qkv(&mut tensors);

    assert!(!tensors.contains_key("model.layers.0.self_attn.c_attn.bias"));
    let (q, q_shape) = &tensors["model.layers.0.self_attn.q_proj.bias"];
    assert_eq!(q, &[1.0, 2.0]);
    assert_eq!(q_shape, &[2]);
    let (k, _) = &tensors["model.layers.0.self_attn.k_proj.bias"];
    assert_eq!(k, &[3.0, 4.0]);
    let (v, _) = &tensors["model.layers.0.self_attn.v_proj.bias"];
    assert_eq!(v, &[5.0, 6.0]);
}

#[test]
fn test_split_gpt2_fused_qkv_weight_row_major_gh219() {
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Weight: 2D [3*hidden, hidden] = [6, 2] for hidden=2 — GGUF row split
    let weight_data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    tensors.insert(
        "model.layers.0.self_attn.c_attn.weight".to_string(),
        (weight_data, vec![6, 2]),
    );

    Architecture::split_gpt2_fused_qkv(&mut tensors);

    assert!(!tensors.contains_key("model.layers.0.self_attn.c_attn.weight"));
    let (q, q_shape) = &tensors["model.layers.0.self_attn.q_proj.weight"];
    assert_eq!(q_shape, &[2, 2]);
    assert_eq!(q, &[1.0, 2.0, 3.0, 4.0]);
    let (k, k_shape) = &tensors["model.layers.0.self_attn.k_proj.weight"];
    assert_eq!(k_shape, &[2, 2]);
    assert_eq!(k, &[5.0, 6.0, 7.0, 8.0]);
    let (v, v_shape) = &tensors["model.layers.0.self_attn.v_proj.weight"];
    assert_eq!(v_shape, &[2, 2]);
    assert_eq!(v, &[9.0, 10.0, 11.0, 12.0]);
}

#[test]
fn test_split_gpt2_fused_qkv_weight_col_major_gh219() {
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // Weight: 2D [hidden, 3*hidden] = [2, 6] for hidden=2 — SafeTensors column split
    // Row 0: [q0, q1, k0, k1, v0, v1] = [1, 2, 3, 4, 5, 6]
    // Row 1: [q2, q3, k2, k3, v2, v3] = [7, 8, 9, 10, 11, 12]
    let weight_data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    tensors.insert(
        "model.layers.0.self_attn.c_attn.weight".to_string(),
        (weight_data, vec![2, 6]),
    );

    Architecture::split_gpt2_fused_qkv(&mut tensors);

    assert!(!tensors.contains_key("model.layers.0.self_attn.c_attn.weight"));
    let (q, q_shape) = &tensors["model.layers.0.self_attn.q_proj.weight"];
    assert_eq!(q_shape, &[2, 2]);
    assert_eq!(q, &[1.0, 2.0, 7.0, 8.0]);
    let (k, _) = &tensors["model.layers.0.self_attn.k_proj.weight"];
    assert_eq!(k, &[3.0, 4.0, 9.0, 10.0]);
    let (v, _) = &tensors["model.layers.0.self_attn.v_proj.weight"];
    assert_eq!(v, &[5.0, 6.0, 11.0, 12.0]);
}

#[test]
fn test_split_gpt2_fused_qkv_no_fused_keys_gh219() {
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert("model.layers.0.self_attn.q_proj.weight".to_string(), (vec![1.0], vec![1]));

    Architecture::split_gpt2_fused_qkv(&mut tensors);

    // Should be unchanged
    assert_eq!(tensors.len(), 1);
    assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
}

#[test]
fn test_split_gpt2_fused_qkv_bias_not_divisible_by_3_gh219() {
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 7 elements — not divisible by 3, should be put back
    tensors.insert(
        "model.layers.0.self_attn.c_attn.bias".to_string(),
        (vec![1.0; 7], vec![7]),
    );

    Architecture::split_gpt2_fused_qkv(&mut tensors);

    // Should be restored since it can't be split
    assert!(tensors.contains_key("model.layers.0.self_attn.c_attn.bias"));
}

#[test]
fn test_split_gpt2_fused_qkv_weight_not_2d_gh219() {
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 3D weight — should be put back unchanged
    tensors.insert(
        "model.layers.0.self_attn.c_attn.weight".to_string(),
        (vec![1.0; 24], vec![2, 3, 4]),
    );

    Architecture::split_gpt2_fused_qkv(&mut tensors);

    assert!(tensors.contains_key("model.layers.0.self_attn.c_attn.weight"));
}
// -------------------------------------------------------------------------
// Architecture::split_gpt2_fused_qkv_raw (raw bytes)
// -------------------------------------------------------------------------

#[test]
fn test_split_gpt2_fused_qkv_raw_bias_gh219() {
    use crate::format::gguf::GgufRawTensor;
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
    // 12 bytes, shape [6], divisible by 3
    tensors.insert(
        "model.layers.0.self_attn.c_attn.bias".to_string(),
        GgufRawTensor {
            data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            shape: vec![6],
            dtype: 0, // F32
        },
    );

    Architecture::split_gpt2_fused_qkv_raw(&mut tensors);

    assert!(!tensors.contains_key("model.layers.0.self_attn.c_attn.bias"));
    let q = &tensors["model.layers.0.self_attn.q_proj.bias"];
    assert_eq!(q.data, vec![1, 2, 3, 4]);
    assert_eq!(q.shape, vec![2]);
    let k = &tensors["model.layers.0.self_attn.k_proj.bias"];
    assert_eq!(k.data, vec![5, 6, 7, 8]);
    let v = &tensors["model.layers.0.self_attn.v_proj.bias"];
    assert_eq!(v.data, vec![9, 10, 11, 12]);
}

#[test]
fn test_split_gpt2_fused_qkv_raw_weight_gh219() {
    use crate::format::gguf::GgufRawTensor;
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
    // Weight: 2D [6, 2], data 12 bytes — split rows
    tensors.insert(
        "model.layers.0.self_attn.c_attn.weight".to_string(),
        GgufRawTensor {
            data: (1u8..=12).collect(),
            shape: vec![6, 2],
            dtype: 0,
        },
    );

    Architecture::split_gpt2_fused_qkv_raw(&mut tensors);

    assert!(!tensors.contains_key("model.layers.0.self_attn.c_attn.weight"));
    let q = &tensors["model.layers.0.self_attn.q_proj.weight"];
    assert_eq!(q.shape, vec![2, 2]);
    assert_eq!(q.data, vec![1, 2, 3, 4]);
    let k = &tensors["model.layers.0.self_attn.k_proj.weight"];
    assert_eq!(k.data, vec![5, 6, 7, 8]);
    let v = &tensors["model.layers.0.self_attn.v_proj.weight"];
    assert_eq!(v.data, vec![9, 10, 11, 12]);
}

#[test]
fn test_split_gpt2_fused_qkv_raw_bias_not_splittable_gh219() {
    use crate::format::gguf::GgufRawTensor;
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
    // 7 bytes, shape [7] — not divisible by 3
    tensors.insert(
        "model.layers.0.self_attn.c_attn.bias".to_string(),
        GgufRawTensor {
            data: vec![1; 7],
            shape: vec![7],
            dtype: 0,
        },
    );

    Architecture::split_gpt2_fused_qkv_raw(&mut tensors);

    // Should be restored
    assert!(tensors.contains_key("model.layers.0.self_attn.c_attn.bias"));
}


include!("coverage_types_tests_part_05_include_01.rs");
