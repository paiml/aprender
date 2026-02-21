
#[test]
fn test_gguf_tensor_byte_size_q8_0() {
    let tensor = GgufTensor {
        name: "quantized".to_string(),
        shape: vec![64],
        dtype: GgmlType::Q8_0,
        data: vec![0; 100],
    };
    let size = tensor.byte_size();
    assert!(size > 0);
}

#[test]
fn test_gguf_tensor_clone_debug() {
    let tensor = GgufTensor {
        name: "test".to_string(),
        shape: vec![10],
        dtype: GgmlType::F32,
        data: vec![1, 2, 3, 4],
    };
    let cloned = tensor.clone();
    assert_eq!(cloned.name, "test");
    assert!(format!("{cloned:?}").contains("GgufTensor"));
}

// ========================================================================
// export_tensors_to_gguf tests
// ========================================================================

#[test]
fn test_export_tensors_to_gguf_empty() {
    let mut buf = Vec::new();
    export_tensors_to_gguf(&mut buf, &[], &[]).expect("export should succeed");
    // Should have header at minimum
    assert!(buf.len() >= 24);
}

#[test]
fn test_export_tensors_to_gguf_with_metadata() {
    let mut buf = Vec::new();
    let metadata = vec![
        (
            "model.name".to_string(),
            GgufValue::String("test".to_string()),
        ),
        ("model.version".to_string(), GgufValue::Uint32(1)),
    ];
    export_tensors_to_gguf(&mut buf, &[], &metadata).expect("export should succeed");
    assert!(buf.len() > 24);
}

#[test]
fn test_export_tensors_to_gguf_with_tensors() {
    let mut buf = Vec::new();
    let tensors = vec![GgufTensor {
        name: "weights".to_string(),
        shape: vec![4],
        dtype: GgmlType::F32,
        data: vec![0; 16], // 4 * 4 bytes
    }];
    export_tensors_to_gguf(&mut buf, &tensors, &[]).expect("export should succeed");
    assert!(buf.len() > 24);
}

#[test]
fn test_export_tensors_to_gguf_full() {
    let mut buf = Vec::new();
    let tensors = vec![
        GgufTensor {
            name: "layer.0.weight".to_string(),
            shape: vec![10, 10],
            dtype: GgmlType::F32,
            data: vec![0; 400],
        },
        GgufTensor {
            name: "layer.0.bias".to_string(),
            shape: vec![10],
            dtype: GgmlType::F32,
            data: vec![0; 40],
        },
    ];
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("test".to_string()),
        ),
        (
            "general.quantization_version".to_string(),
            GgufValue::Uint32(2),
        ),
    ];
    export_tensors_to_gguf(&mut buf, &tensors, &metadata).expect("export should succeed");
    // Verify header magic
    assert_eq!(&buf[0..4], b"GGUF");
}

#[test]
fn test_gguf_tensor_info_clone_debug() {
    let info = GgufTensorInfo {
        name: "test".to_string(),
        n_dims: 2,
        dims: vec![10, 20],
        dtype: GgmlType::F32,
        offset: 0,
    };
    let cloned = info.clone();
    assert_eq!(cloned.name, "test");
    assert!(format!("{cloned:?}").contains("GgufTensorInfo"));
}

// ========================================================================
// GgufReader accessor tests
// ========================================================================

fn make_test_reader(metadata: std::collections::BTreeMap<String, GgufValue>) -> GgufReader {
    GgufReader {
        data: vec![],
        version: GGUF_VERSION,
        tensor_count: 0,
        tensors: vec![],
        data_offset: 0,
        metadata,
    }
}

#[test]
fn test_reader_vocabulary_empty() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.vocabulary().is_none());
}

#[test]
fn test_reader_vocabulary_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::ArrayString(vec!["hello".into(), "world".into()]),
    );
    let reader = make_test_reader(metadata);
    let vocab = reader.vocabulary().expect("should have vocab");
    assert_eq!(vocab, vec!["hello", "world"]);
}

#[test]
fn test_reader_vocabulary_empty_array() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::ArrayString(vec![]),
    );
    let reader = make_test_reader(metadata);
    assert!(reader.vocabulary().is_none());
}

#[test]
fn test_reader_tokenizer_model_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.tokenizer_model().is_none());
}

#[test]
fn test_reader_tokenizer_model_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.model".to_string(),
        GgufValue::String("llama".into()),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.tokenizer_model(), Some("llama".into()));
}

#[test]
fn test_reader_bos_token_id_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.bos_token_id().is_none());
}

#[test]
fn test_reader_bos_token_id_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.bos_token_id".to_string(),
        GgufValue::Uint32(1),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.bos_token_id(), Some(1));
}

#[test]
fn test_reader_eos_token_id_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.eos_token_id().is_none());
}

#[test]
fn test_reader_eos_token_id_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.eos_token_id".to_string(),
        GgufValue::Uint32(2),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.eos_token_id(), Some(2));
}

#[test]
fn test_reader_architecture_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.architecture().is_none());
}

#[test]
fn test_reader_architecture_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        GgufValue::String("qwen2".into()),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.architecture(), Some("qwen2".into()));
}

#[test]
fn test_reader_model_name_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.model_name().is_none());
}

#[test]
fn test_reader_model_name_present() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "general.name".to_string(),
        GgufValue::String("My Model".into()),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.model_name(), Some("My Model".into()));
}

#[test]
fn test_reader_hidden_size_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.hidden_size().is_none());
}

#[test]
fn test_reader_hidden_size_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Uint32(4096),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.hidden_size(), Some(4096));
}

#[test]
fn test_reader_hidden_size_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.embedding_length".to_string(),
        GgufValue::Uint64(4096),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.hidden_size(), Some(4096));
}

#[test]
fn test_reader_num_layers_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.num_layers().is_none());
}

#[test]
fn test_reader_num_layers_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.block_count".to_string(), GgufValue::Uint32(32));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_layers(), Some(32));
}

#[test]
fn test_reader_num_layers_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.block_count".to_string(), GgufValue::Uint64(32));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_layers(), Some(32));
}

#[test]
fn test_reader_num_heads_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.num_heads().is_none());
}

#[test]
fn test_reader_num_heads_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count".to_string(),
        GgufValue::Uint32(32),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_heads(), Some(32));
}

#[test]
fn test_reader_num_heads_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count".to_string(),
        GgufValue::Uint64(32),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_heads(), Some(32));
}

#[test]
fn test_reader_num_kv_heads_none_fallback_to_num_heads() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count".to_string(),
        GgufValue::Uint32(32),
    );
    let reader = make_test_reader(metadata);
    // Without head_count_kv, should fall back to num_heads
    assert_eq!(reader.num_kv_heads(), Some(32));
}

#[test]
fn test_reader_num_kv_heads_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count_kv".to_string(),
        GgufValue::Uint32(8),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_kv_heads(), Some(8));
}

#[test]
fn test_reader_num_kv_heads_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.attention.head_count_kv".to_string(),
        GgufValue::Uint64(8),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.num_kv_heads(), Some(8));
}

#[test]
fn test_reader_vocab_size_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.vocab_size().is_none());
}

#[test]
fn test_reader_vocab_size_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.vocab_size".to_string(), GgufValue::Uint32(32000));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.vocab_size(), Some(32000));
}

#[test]
fn test_reader_vocab_size_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.vocab_size".to_string(), GgufValue::Uint64(32000));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.vocab_size(), Some(32000));
}

#[test]
fn test_reader_vocab_size_from_vocabulary() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufValue::ArrayString(vec!["a".into(), "b".into(), "c".into()]),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.vocab_size(), Some(3));
}

#[test]
fn test_reader_intermediate_size_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.intermediate_size().is_none());
}

#[test]
fn test_reader_intermediate_size_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.feed_forward_length".to_string(),
        GgufValue::Uint32(11008),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.intermediate_size(), Some(11008));
}

#[test]
fn test_reader_intermediate_size_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert(
        "llama.feed_forward_length".to_string(),
        GgufValue::Uint64(11008),
    );
    let reader = make_test_reader(metadata);
    assert_eq!(reader.intermediate_size(), Some(11008));
}

#[test]
fn test_reader_context_length_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.context_length().is_none());
}

#[test]
fn test_reader_context_length_uint32() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.context_length".to_string(), GgufValue::Uint32(4096));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.context_length(), Some(4096));
}

#[test]
fn test_reader_context_length_uint64() {
    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("llama.context_length".to_string(), GgufValue::Uint64(4096));
    let reader = make_test_reader(metadata);
    assert_eq!(reader.context_length(), Some(4096));
}

#[test]
fn test_reader_rope_theta_none() {
    let reader = make_test_reader(std::collections::BTreeMap::new());
    assert!(reader.rope_theta().is_none());
}
