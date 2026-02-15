
#[test]
fn test_reader_rope_theta_float32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.rope.freq_base".to_string(),
        GgufValue::Float32(10000.0),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.rope_theta(), Some(10000.0));
}

#[test]
fn test_reader_rope_theta_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.rope.freq_base".to_string(), GgufValue::Uint32(10000));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.rope_theta(), Some(10000.0));
}

#[test]
fn test_reader_rms_norm_eps_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.rms_norm_eps().is_none());
}

#[test]
fn test_reader_rms_norm_eps_float32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.layer_norm_rms_epsilon".to_string(),
        GgufValue::Float32(1e-6),
    );
    let reader = make_test_reader(metadata);
    assert!((reader.rms_norm_eps().unwrap() - 1e-6).abs() < 1e-12);
}

#[test]
fn test_reader_with_custom_architecture() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("qwen2".into()),
    );
    metadata.insert(
        "qwen2.embedding_length".to_string(),
        GgufValue::Uint32(3584),
    );
    metadata.insert("qwen2.block_count".to_string(), GgufValue::Uint32(28));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.architecture(), Some("qwen2".into()));
    assert_eq!(reader.hidden_size(), Some(3584));
    assert_eq!(reader.num_layers(), Some(28));
}

#[test]
fn test_gguf_tensor_meta_clone_debug() {
    let meta = GgufTensorMeta {
        name: "test.weight".to_string(),
        dims: vec![10, 20],
        dtype: 0, // F32
        offset: 0,
    };
    let cloned = meta.clone();
    assert_eq!(cloned.name, "test.weight");
    assert!(format!("{cloned:?}").contains("GgufTensorMeta"));
}

#[test]
fn test_gguf_tokenizer_has_vocabulary_false() {
    let tokenizer = GgufTokenizer {
        vocabulary: vec![],
        merges: vec![],
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        architecture: None,
        model_name: None,
        ..Default::default()
    };
    assert!(!tokenizer.has_vocabulary());
}

#[test]
fn test_gguf_tokenizer_has_vocabulary_true() {
    let tokenizer = GgufTokenizer {
        vocabulary: vec!["a".into()],
        merges: vec![],
        model_type: None,
        bos_token_id: None,
        eos_token_id: None,
        architecture: None,
        model_name: None,
        ..Default::default()
    };
    assert!(tokenizer.has_vocabulary());
}

#[test]
fn test_gguf_tokenizer_vocab_size() {
    let tokenizer = GgufTokenizer {
        vocabulary: vec!["a".into(), "b".into(), "c".into()],
        merges: vec![],
        model_type: Some("llama".into()),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        architecture: Some("llama".into()),
        model_name: Some("Test Model".into()),
        ..Default::default()
    };
    assert_eq!(tokenizer.vocab_size(), 3);
    assert!(format!("{tokenizer:?}").contains("GgufTokenizer"));
}

#[test]
fn test_gguf_model_config_debug() {
    let config = GgufModelConfig {
        architecture: Some("llama".into()),
        hidden_size: Some(4096),
        num_layers: Some(32),
        num_heads: Some(32),
        num_kv_heads: Some(8),
        vocab_size: Some(32000),
        intermediate_size: Some(11008),
        max_position_embeddings: Some(4096),
        rope_theta: Some(10000.0),
        rms_norm_eps: Some(1e-6),
        rope_type: Some(0), // NORM style for LLaMA
    };
    assert!(format!("{config:?}").contains("GgufModelConfig"));
}

#[test]
fn test_gguf_load_result_debug() {
    let result = GgufLoadResult {
        tensors: std::collections::BTreeMap::new(),
        tokenizer: GgufTokenizer {
            vocabulary: vec![],
            merges: vec![],
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            architecture: None,
            model_name: None,
            ..Default::default()
        },
        model_config: GgufModelConfig {
            architecture: None,
            hidden_size: None,
            num_layers: None,
            num_heads: None,
            num_kv_heads: None,
            vocab_size: None,
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            rope_type: None,
        },
    };
    assert!(format!("{result:?}").contains("GgufLoadResult"));
}

#[test]
fn test_gguf_raw_tensor_debug() {
    let tensor = GgufRawTensor {
        data: vec![0u8; 16],
        shape: vec![4],
        dtype: 0, // F32
    };
    assert!(format!("{tensor:?}").contains("GgufRawTensor"));
}

#[test]
fn test_gguf_raw_load_result_debug() {
    let result = GgufRawLoadResult {
        tensors: std::collections::BTreeMap::new(),
        tokenizer: GgufTokenizer {
            vocabulary: vec![],
            merges: vec![],
            model_type: None,
            bos_token_id: None,
            eos_token_id: None,
            architecture: None,
            model_name: None,
            ..Default::default()
        },
        model_config: GgufModelConfig {
            architecture: None,
            hidden_size: None,
            num_layers: None,
            num_heads: None,
            num_kv_heads: None,
            vocab_size: None,
            intermediate_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            rms_norm_eps: None,
            rope_type: None,
        },
    };
    assert!(format!("{result:?}").contains("GgufRawLoadResult"));
}
