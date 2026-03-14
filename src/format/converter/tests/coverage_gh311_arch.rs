// GH-311: Falsification tests for non-LLaMA architecture support
// (GPT-NeoX, OPT, StarCoder/GPT-2 reuse)

use std::collections::BTreeMap;

use crate::format::converter_types::Architecture;

// ============================================================================
// Architecture::from_model_type
// ============================================================================

#[test]
fn falsify_gh311_from_model_type_gpt_neox() {
    assert_eq!(
        Architecture::from_model_type("gpt-neox"),
        Some(Architecture::GptNeoX)
    );
    assert_eq!(
        Architecture::from_model_type("gpt_neox"),
        Some(Architecture::GptNeoX)
    );
    assert_eq!(
        Architecture::from_model_type("gptneox"),
        Some(Architecture::GptNeoX)
    );
    assert_eq!(
        Architecture::from_model_type("pythia"),
        Some(Architecture::GptNeoX)
    );
}

#[test]
fn falsify_gh311_from_model_type_opt() {
    assert_eq!(
        Architecture::from_model_type("opt"),
        Some(Architecture::Opt)
    );
    assert_eq!(
        Architecture::from_model_type("galactica"),
        Some(Architecture::Opt)
    );
}

#[test]
fn falsify_gh311_from_model_type_starcoder_maps_to_gpt2() {
    // StarCoder reuses GPT-2 tensor naming
    assert_eq!(
        Architecture::from_model_type("starcoder"),
        Some(Architecture::Gpt2)
    );
    assert_eq!(
        Architecture::from_model_type("starcoder2"),
        Some(Architecture::Gpt2)
    );
    assert_eq!(
        Architecture::from_model_type("bigcode"),
        Some(Architecture::Gpt2)
    );
}

// ============================================================================
// display_name / completeness_key
// ============================================================================

#[test]
fn falsify_gh311_display_names() {
    assert_eq!(Architecture::GptNeoX.display_name(), "GPT-NeoX");
    assert_eq!(Architecture::Opt.display_name(), "OPT");
}

#[test]
fn falsify_gh311_completeness_key_none_for_new_archs() {
    // GPT-NeoX and OPT don't have completeness checks (different naming)
    assert!(Architecture::GptNeoX.completeness_key().is_none());
    assert!(Architecture::Opt.completeness_key().is_none());
}

#[test]
fn falsify_gh311_not_inference_verified() {
    // New architectures are NOT inference-verified yet
    assert!(!Architecture::GptNeoX.is_inference_verified());
    assert!(!Architecture::Opt.is_inference_verified());
}

// ============================================================================
// GPT-NeoX tensor name mapping
// ============================================================================

#[test]
fn falsify_gh311_neox_map_layer_tensors() {
    let arch = Architecture::GptNeoX;

    // Attention
    assert_eq!(
        arch.map_name("gpt_neox.layers.0.attention.query_key_value.weight"),
        "model.layers.0.self_attn.query_key_value.weight"
    );
    assert_eq!(
        arch.map_name("gpt_neox.layers.3.attention.dense.weight"),
        "model.layers.3.self_attn.o_proj.weight"
    );

    // MLP
    assert_eq!(
        arch.map_name("gpt_neox.layers.0.mlp.dense_h_to_4h.weight"),
        "model.layers.0.mlp.up_proj.weight"
    );
    assert_eq!(
        arch.map_name("gpt_neox.layers.0.mlp.dense_4h_to_h.weight"),
        "model.layers.0.mlp.down_proj.weight"
    );

    // Norms
    assert_eq!(
        arch.map_name("gpt_neox.layers.0.input_layernorm.weight"),
        "model.layers.0.input_layernorm.weight"
    );
    assert_eq!(
        arch.map_name("gpt_neox.layers.0.post_attention_layernorm.weight"),
        "model.layers.0.post_attention_layernorm.weight"
    );
}

#[test]
fn falsify_gh311_neox_map_non_layer_tensors() {
    let arch = Architecture::GptNeoX;

    assert_eq!(
        arch.map_name("gpt_neox.embed_in.weight"),
        "model.embed_tokens.weight"
    );
    assert_eq!(
        arch.map_name("gpt_neox.final_layer_norm.weight"),
        "model.norm.weight"
    );
    assert_eq!(
        arch.map_name("gpt_neox.final_layer_norm.bias"),
        "model.norm.bias"
    );
    assert_eq!(arch.map_name("embed_out.weight"), "lm_head.weight");
}

// ============================================================================
// OPT tensor name mapping
// ============================================================================

#[test]
fn falsify_gh311_opt_map_layer_tensors() {
    let arch = Architecture::Opt;

    // Attention (OPT has separate Q/K/V, no fusion)
    assert_eq!(
        arch.map_name("model.decoder.layers.0.self_attn.q_proj.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        arch.map_name("model.decoder.layers.0.self_attn.k_proj.weight"),
        "model.layers.0.self_attn.k_proj.weight"
    );
    assert_eq!(
        arch.map_name("model.decoder.layers.0.self_attn.v_proj.weight"),
        "model.layers.0.self_attn.v_proj.weight"
    );
    // OPT: out_proj → o_proj
    assert_eq!(
        arch.map_name("model.decoder.layers.0.self_attn.out_proj.weight"),
        "model.layers.0.self_attn.o_proj.weight"
    );

    // MLP: fc1 → up_proj, fc2 → down_proj
    assert_eq!(
        arch.map_name("model.decoder.layers.0.fc1.weight"),
        "model.layers.0.mlp.up_proj.weight"
    );
    assert_eq!(
        arch.map_name("model.decoder.layers.0.fc2.weight"),
        "model.layers.0.mlp.down_proj.weight"
    );

    // Norms
    assert_eq!(
        arch.map_name("model.decoder.layers.0.self_attn_layer_norm.weight"),
        "model.layers.0.input_layernorm.weight"
    );
    assert_eq!(
        arch.map_name("model.decoder.layers.0.final_layer_norm.weight"),
        "model.layers.0.post_attention_layernorm.weight"
    );
}

#[test]
fn falsify_gh311_opt_map_non_layer_tensors() {
    let arch = Architecture::Opt;

    assert_eq!(
        arch.map_name("model.decoder.embed_tokens.weight"),
        "model.embed_tokens.weight"
    );
    assert_eq!(
        arch.map_name("model.decoder.embed_positions.weight"),
        "model.position_embedding.weight"
    );
    assert_eq!(
        arch.map_name("model.decoder.final_layer_norm.weight"),
        "model.norm.weight"
    );
    assert_eq!(arch.map_name("lm_head.weight"), "lm_head.weight");
}

// ============================================================================
// GPT-NeoX fused QKV splitting
// ============================================================================

#[test]
fn falsify_gh311_neox_split_qkv_weight() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    let _hidden = 4;
    // Shape: [3*hidden, hidden] = [12, 4]
    let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
    tensors.insert(
        "model.layers.0.self_attn.query_key_value.weight".to_string(),
        (data, vec![12, 4]),
    );

    Architecture::split_neox_fused_qkv(&mut tensors);

    assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.k_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.v_proj.weight"));
    assert!(!tensors.contains_key("model.layers.0.self_attn.query_key_value.weight"));

    let (q_data, q_shape) = &tensors["model.layers.0.self_attn.q_proj.weight"];
    assert_eq!(q_shape, &[4, 4]);
    assert_eq!(q_data.len(), 16);

    let (k_data, k_shape) = &tensors["model.layers.0.self_attn.k_proj.weight"];
    assert_eq!(k_shape, &[4, 4]);
    assert_eq!(k_data.len(), 16);
}

#[test]
fn falsify_gh311_neox_split_qkv_bias() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    tensors.insert(
        "model.layers.0.self_attn.query_key_value.bias".to_string(),
        (data, vec![12]),
    );

    Architecture::split_neox_fused_qkv(&mut tensors);

    assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.bias"));
    assert!(tensors.contains_key("model.layers.0.self_attn.k_proj.bias"));
    assert!(tensors.contains_key("model.layers.0.self_attn.v_proj.bias"));

    let (q_data, q_shape) = &tensors["model.layers.0.self_attn.q_proj.bias"];
    assert_eq!(q_shape, &[4]);
    assert_eq!(q_data, &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn falsify_gh311_neox_split_qkv_not_divisible_passthrough() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    // 7 is not divisible by 3
    let data: Vec<f32> = (0..7).map(|i| i as f32).collect();
    tensors.insert(
        "model.layers.0.self_attn.query_key_value.bias".to_string(),
        (data, vec![7]),
    );

    Architecture::split_neox_fused_qkv(&mut tensors);

    // Should remain untouched
    assert!(tensors.contains_key("model.layers.0.self_attn.query_key_value.bias"));
}

// ============================================================================
// GPT-NeoX fused QKV raw splitting
// ============================================================================

#[test]
fn falsify_gh311_neox_split_qkv_raw_weight() {
    use crate::format::gguf::GgufRawTensor;

    let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
    let data: Vec<u8> = (0..48).collect();
    tensors.insert(
        "model.layers.0.self_attn.query_key_value.weight".to_string(),
        GgufRawTensor {
            data,
            shape: vec![6, 2],
            dtype: 0,
        },
    );

    Architecture::split_neox_fused_qkv_raw(&mut tensors);

    assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.k_proj.weight"));
    assert!(tensors.contains_key("model.layers.0.self_attn.v_proj.weight"));
    assert_eq!(tensors["model.layers.0.self_attn.q_proj.weight"].shape, vec![2, 2]);
    assert_eq!(tensors["model.layers.0.self_attn.q_proj.weight"].data.len(), 16);
}

#[test]
fn falsify_gh311_neox_split_qkv_raw_bias() {
    use crate::format::gguf::GgufRawTensor;

    let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
    let data: Vec<u8> = (0..12).collect();
    tensors.insert(
        "model.layers.0.self_attn.query_key_value.bias".to_string(),
        GgufRawTensor {
            data,
            shape: vec![12],
            dtype: 0,
        },
    );

    Architecture::split_neox_fused_qkv_raw(&mut tensors);

    assert!(tensors.contains_key("model.layers.0.self_attn.q_proj.bias"));
    assert_eq!(tensors["model.layers.0.self_attn.q_proj.bias"].shape, vec![4]);
    assert_eq!(tensors["model.layers.0.self_attn.q_proj.bias"].data.len(), 4);
}
