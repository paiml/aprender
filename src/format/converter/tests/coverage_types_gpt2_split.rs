
#[test]
fn test_split_gpt2_fused_qkv_raw_weight_not_2d_gh219() {
    use crate::format::gguf::GgufRawTensor;
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
    // 1D weight — should be put back
    tensors.insert(
        "model.layers.0.self_attn.c_attn.weight".to_string(),
        GgufRawTensor {
            data: vec![1; 12],
            shape: vec![12],
            dtype: 0,
        },
    );

    Architecture::split_gpt2_fused_qkv_raw(&mut tensors);

    assert!(tensors.contains_key("model.layers.0.self_attn.c_attn.weight"));
}

#[test]
fn test_split_gpt2_fused_qkv_raw_no_fused_keys_gh219() {
    use crate::format::gguf::GgufRawTensor;
    use std::collections::BTreeMap;

    let mut tensors: BTreeMap<String, GgufRawTensor> = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        GgufRawTensor {
            data: vec![1; 4],
            shape: vec![2, 2],
            dtype: 0,
        },
    );

    Architecture::split_gpt2_fused_qkv_raw(&mut tensors);

    assert_eq!(tensors.len(), 1);
}

// -------------------------------------------------------------------------
// Source::parse
// -------------------------------------------------------------------------

#[test]
fn test_source_parse_hf_basic_gh219() {
    let source = Source::parse("hf://meta-llama/Llama-3.2-1B").unwrap();
    match source {
        Source::HuggingFace { org, repo, file } => {
            assert_eq!(org, "meta-llama");
            assert_eq!(repo, "Llama-3.2-1B");
            assert!(file.is_none());
        }
        _ => panic!("Expected HuggingFace source"),
    }
}

#[test]
fn test_source_parse_hf_with_file_gh219() {
    let source = Source::parse("hf://org/repo/model.safetensors").unwrap();
    match source {
        Source::HuggingFace { org, repo, file } => {
            assert_eq!(org, "org");
            assert_eq!(repo, "repo");
            assert_eq!(file, Some("model.safetensors".to_string()));
        }
        _ => panic!("Expected HuggingFace source"),
    }
}

#[test]
fn test_source_parse_hf_resolve_main_gh219() {
    // GH-221: Strip resolve/main/ from browser-copied URLs
    let source = Source::parse("hf://org/repo/resolve/main/model.safetensors").unwrap();
    match source {
        Source::HuggingFace { file, .. } => {
            assert_eq!(file, Some("model.safetensors".to_string()));
        }
        _ => panic!("Expected HuggingFace source"),
    }
}

#[test]
fn test_source_parse_hf_blob_main_gh219() {
    let source = Source::parse("hf://org/repo/blob/main/weights.bin").unwrap();
    match source {
        Source::HuggingFace { file, .. } => {
            assert_eq!(file, Some("weights.bin".to_string()));
        }
        _ => panic!("Expected HuggingFace source"),
    }
}

#[test]
fn test_source_parse_hf_bare_resolve_main_gh219() {
    // Bare "resolve/main" without file should give None
    let source = Source::parse("hf://org/repo/resolve/main").unwrap();
    match source {
        Source::HuggingFace { file, .. } => {
            assert!(file.is_none());
        }
        _ => panic!("Expected HuggingFace source"),
    }
}

#[test]
fn test_source_parse_hf_too_few_parts_gh219() {
    let result = Source::parse("hf://only_org");
    assert!(result.is_err());
}

#[test]
fn test_source_parse_local_gh219() {
    let source = Source::parse("/path/to/model.safetensors").unwrap();
    assert!(matches!(source, Source::Local(_)));
}

#[test]
fn test_source_parse_http_gh219() {
    let source = Source::parse("https://example.com/model.safetensors").unwrap();
    assert!(matches!(source, Source::Url(_)));
}

#[test]
fn test_source_parse_http_no_s_gh219() {
    let source = Source::parse("http://example.com/model.safetensors").unwrap();
    assert!(matches!(source, Source::Url(_)));
}

// -------------------------------------------------------------------------
// Architecture::map_name dispatch for Phi (uses llama_map_name)
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// GH-278: Qwen3.5 architecture support
// -------------------------------------------------------------------------

#[test]
fn test_from_model_type_qwen3_5_gh278() {
    assert_eq!(Architecture::from_model_type("qwen3_5"), Some(Architecture::Qwen3_5));
    assert_eq!(Architecture::from_model_type("qwen3.5"), Some(Architecture::Qwen3_5));
}

#[test]
fn test_qwen3_5_is_inference_verified_gh278() {
    assert!(Architecture::Qwen3_5.is_inference_verified());
}

#[test]
fn test_qwen3_5_display_name_gh278() {
    assert_eq!(Architecture::Qwen3_5.display_name(), "Qwen3.5");
}

#[test]
fn test_qwen3_5_uses_qwen2_mapping_gh278() {
    // Qwen3.5 should use the same tensor name mapping as Qwen2
    assert_eq!(
        Architecture::Qwen3_5.map_name("blk.0.attn_q.weight"),
        Architecture::Qwen2.map_name("blk.0.attn_q.weight")
    );
    assert_eq!(
        Architecture::Qwen3_5.map_name("token_embd.weight"),
        Architecture::Qwen2.map_name("token_embd.weight")
    );
    assert_eq!(
        Architecture::Qwen3_5.map_name("output.weight"),
        Architecture::Qwen2.map_name("output.weight")
    );
}

#[test]
fn test_phi_uses_llama_mapping_gh219() {
    // Phi uses same mapping as LLaMA
    assert_eq!(
        Architecture::Phi.map_name("model.layers.0.self_attn.q_proj.weight"),
        Architecture::Llama.map_name("model.layers.0.self_attn.q_proj.weight")
    );
}

// -------------------------------------------------------------------------
// GH-279: Qwen3 QK normalization tensor mappings
// -------------------------------------------------------------------------

#[test]
fn test_qwen3_map_name_qk_norm_tensors_gh279() {
    let arch = Architecture::Qwen3;
    assert_eq!(
        arch.map_name("blk.0.attn_q_norm.weight"),
        "model.layers.0.self_attn.q_norm.weight"
    );
    assert_eq!(
        arch.map_name("blk.0.attn_k_norm.weight"),
        "model.layers.0.self_attn.k_norm.weight"
    );
    assert_eq!(
        arch.map_name("blk.35.attn_q_norm.weight"),
        "model.layers.35.self_attn.q_norm.weight"
    );
    assert_eq!(
        arch.map_name("blk.35.attn_k_norm.weight"),
        "model.layers.35.self_attn.k_norm.weight"
    );
}

#[test]
fn test_qwen2_map_name_qk_norm_also_works_gh279() {
    // Qwen2 uses the same mapper, so QK norm tensors should map correctly
    // even though Qwen2 models don't have them
    let arch = Architecture::Qwen2;
    assert_eq!(
        arch.map_name("blk.0.attn_q_norm.weight"),
        "model.layers.0.self_attn.q_norm.weight"
    );
    assert_eq!(
        arch.map_name("blk.0.attn_k_norm.weight"),
        "model.layers.0.self_attn.k_norm.weight"
    );
}

#[test]
fn test_qwen3_safetensors_passthrough_qk_norm_gh279() {
    // SafeTensors names are already in HF format — should pass through unchanged
    let arch = Architecture::Qwen3;
    assert_eq!(
        arch.map_name("model.layers.0.self_attn.q_norm.weight"),
        "model.layers.0.self_attn.q_norm.weight"
    );
    assert_eq!(
        arch.map_name("model.layers.0.self_attn.k_norm.weight"),
        "model.layers.0.self_attn.k_norm.weight"
    );
}

#[test]
fn test_qk_norm_tensor_expectation_is_rmsnorm_gh279() {
    // QK norm tensors must be classified as RMSNORM_WEIGHT, not LINEAR_WEIGHT
    let q_norm = TensorExpectation::for_tensor("model.layers.0.self_attn.q_norm.weight");
    assert!(q_norm.is_some(), "q_norm should have an expectation");
    assert_eq!(
        q_norm.unwrap().description,
        TensorExpectation::RMSNORM_WEIGHT.description,
        "q_norm should be RMSNORM_WEIGHT"
    );

    let k_norm = TensorExpectation::for_tensor("model.layers.0.self_attn.k_norm.weight");
    assert!(k_norm.is_some(), "k_norm should have an expectation");
    assert_eq!(
        k_norm.unwrap().description,
        TensorExpectation::RMSNORM_WEIGHT.description,
        "k_norm should be RMSNORM_WEIGHT"
    );
}

#[test]
fn test_qk_norm_gguf_tensor_expectation_is_rmsnorm_gh279() {
    // GGUF-style names should also be classified as RMSNORM_WEIGHT
    let q_norm = TensorExpectation::for_tensor("blk.0.attn_q_norm.weight");
    assert!(q_norm.is_some(), "GGUF q_norm should have an expectation");
    assert_eq!(
        q_norm.unwrap().description,
        TensorExpectation::RMSNORM_WEIGHT.description,
        "GGUF q_norm should be RMSNORM_WEIGHT"
    );

    let k_norm = TensorExpectation::for_tensor("blk.0.attn_k_norm.weight");
    assert!(k_norm.is_some(), "GGUF k_norm should have an expectation");
    assert_eq!(
        k_norm.unwrap().description,
        TensorExpectation::RMSNORM_WEIGHT.description,
        "GGUF k_norm should be RMSNORM_WEIGHT"
    );
}
