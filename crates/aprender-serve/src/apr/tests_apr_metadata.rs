
#[test]
fn test_apr_metadata_is_transformer_false_missing_layers() {
    let m = AprMetadata {
        hidden_size: Some(512),
        num_heads: Some(8),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(!m.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_false_missing_heads() {
    let m = AprMetadata {
        hidden_size: Some(512),
        num_layers: Some(6),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(!m.is_transformer());
}

#[test]
fn test_apr_metadata_is_transformer_false_missing_vocab() {
    let m = AprMetadata {
        hidden_size: Some(512),
        num_layers: Some(6),
        num_heads: Some(8),
        ..Default::default()
    };
    assert!(!m.is_transformer());
}

// ============================================================================
// AprMetadata embedded tokenizer tests
// ============================================================================

#[test]
fn test_get_embedded_vocabulary_present() {
    let mut extra = HashMap::new();
    extra.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!(["hello", "world", "test"]),
    );
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    let vocab = m.get_embedded_vocabulary();
    assert!(vocab.is_some());
    assert_eq!(vocab.unwrap(), vec!["hello", "world", "test"]);
}

#[test]
fn test_get_embedded_vocabulary_missing() {
    let m = AprMetadata::default();
    assert!(m.get_embedded_vocabulary().is_none());
}

#[test]
fn test_get_embedded_vocabulary_empty_array() {
    let mut extra = HashMap::new();
    extra.insert("tokenizer.vocabulary".to_string(), serde_json::json!([]));
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert!(m.get_embedded_vocabulary().is_none());
}

#[test]
fn test_get_embedded_vocabulary_not_array() {
    let mut extra = HashMap::new();
    extra.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::json!("not_array"),
    );
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert!(m.get_embedded_vocabulary().is_none());
}

#[test]
fn test_get_embedded_bos_token_id_present() {
    let mut extra = HashMap::new();
    extra.insert("tokenizer.bos_token_id".to_string(), serde_json::json!(1));
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert_eq!(m.get_embedded_bos_token_id(), Some(1));
}

#[test]
fn test_get_embedded_bos_token_id_missing() {
    let m = AprMetadata::default();
    assert!(m.get_embedded_bos_token_id().is_none());
}

#[test]
fn test_get_embedded_eos_token_id_present() {
    let mut extra = HashMap::new();
    extra.insert("tokenizer.eos_token_id".to_string(), serde_json::json!(2));
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert_eq!(m.get_embedded_eos_token_id(), Some(2));
}

#[test]
fn test_get_embedded_eos_token_id_missing() {
    let m = AprMetadata::default();
    assert!(m.get_embedded_eos_token_id().is_none());
}

// ============================================================================
// AprMetadata::get_embedded_merges
// ============================================================================

#[test]
fn test_get_embedded_merges_present() {
    let mut extra = HashMap::new();
    extra.insert(
        "tokenizer.merges".to_string(),
        serde_json::json!(["a b", "c d", "ef gh"]),
    );
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    let merges = m.get_embedded_merges();
    assert!(merges.is_some());
    let merges = merges.unwrap();
    assert_eq!(merges.len(), 3);
    assert_eq!(merges[0], ("a".to_string(), "b".to_string()));
    assert_eq!(merges[1], ("c".to_string(), "d".to_string()));
    assert_eq!(merges[2], ("ef".to_string(), "gh".to_string()));
}

#[test]
fn test_get_embedded_merges_missing() {
    let m = AprMetadata::default();
    assert!(m.get_embedded_merges().is_none());
}

#[test]
fn test_get_embedded_merges_empty() {
    let mut extra = HashMap::new();
    extra.insert("tokenizer.merges".to_string(), serde_json::json!([]));
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    assert!(m.get_embedded_merges().is_none());
}

#[test]
fn test_get_embedded_merges_invalid_format() {
    let mut extra = HashMap::new();
    // Single words (no space separator) should be skipped
    extra.insert(
        "tokenizer.merges".to_string(),
        serde_json::json!(["nospace", "a b"]),
    );
    let m = AprMetadata {
        extra,
        ..Default::default()
    };
    let merges = m.get_embedded_merges();
    assert!(merges.is_some());
    // Only "a b" should be parsed (nospace has no space)
    assert_eq!(merges.unwrap().len(), 1);
}

// ============================================================================
// AprMetadata serde aliases
// ============================================================================

#[test]
fn test_apr_metadata_hidden_dim_alias() {
    let json = r#"{"hidden_dim": 1024}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.hidden_size, Some(1024));
}

#[test]
fn test_apr_metadata_num_hidden_layers_alias() {
    let json = r#"{"num_hidden_layers": 12}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.num_layers, Some(12));
}

#[test]
fn test_apr_metadata_num_attention_heads_alias() {
    let json = r#"{"num_attention_heads": 16}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.num_heads, Some(16));
}

#[test]
fn test_apr_metadata_d_model_alias() {
    let json = r#"{"d_model": 768}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.hidden_size, Some(768));
}

#[test]
fn test_apr_metadata_n_vocab_alias() {
    let json = r#"{"n_vocab": 50257}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.vocab_size, Some(50257));
}

#[test]
fn test_apr_metadata_intermediate_dim_alias() {
    let json = r#"{"intermediate_dim": 2048}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.intermediate_size, Some(2048));
}

#[test]
fn test_apr_metadata_context_length_alias() {
    let json = r#"{"context_length": 4096}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert_eq!(m.max_position_embeddings, Some(4096));
}

#[test]
fn test_apr_metadata_norm_eps_alias() {
    let json = r#"{"norm_eps": 0.00001}"#;
    let m: AprMetadata = serde_json::from_str(json).expect("parse");
    assert!(m.rms_norm_eps.is_some());
}

// ============================================================================
// AprV2Model::from_bytes tests
// ============================================================================

#[test]
fn test_apr_v2_model_from_bytes_encrypted_rejected() {
    let mut data = vec![0u8; 256];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[6..8].copy_from_slice(&AprFlags::ENCRYPTED.to_le_bytes()); // flags: encrypted
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&0u32.to_le_bytes()); // metadata_size = 0
    data[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset

    let result = AprV2Model::from_bytes(data);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("ncrypt"));
}

#[test]
fn test_apr_v2_model_from_bytes_minimal_valid() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
    data[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata_offset
    data[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata_size = 2
    data[24..32].copy_from_slice(&128u64.to_le_bytes()); // tensor_index_offset
    data[32..40].copy_from_slice(&128u64.to_le_bytes()); // data_offset
    // Put valid JSON metadata at offset 64
    data[64] = b'{';
    data[65] = b'}';

    let model = AprV2Model::from_bytes(data).expect("should parse");
    assert_eq!(model.tensor_count(), 0);
    assert!(model.tensor_names().is_empty());
}

#[test]
fn test_apr_v2_model_from_bytes_with_metadata() {
    let metadata = serde_json::json!({
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "vocab_size": 32000,
        "architecture": "llama"
    });
    let meta_bytes = serde_json::to_vec(&metadata).unwrap();
    let meta_padded = ((meta_bytes.len() + 63) / 64) * 64;

    let total_size = 64 + meta_padded;
    let mut data = vec![0u8; total_size];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2;
    data[8..12].copy_from_slice(&0u32.to_le_bytes());
    data[12..20].copy_from_slice(&64u64.to_le_bytes());
    data[20..24].copy_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&(total_size as u64).to_le_bytes());
    data[32..40].copy_from_slice(&(total_size as u64).to_le_bytes());
    data[64..64 + meta_bytes.len()].copy_from_slice(&meta_bytes);

    let model = AprV2Model::from_bytes(data).expect("should parse");
    let meta = model.metadata();
    assert!(meta.is_transformer());
    assert_eq!(meta.hidden_size, Some(512));
    assert_eq!(meta.architecture, Some("llama".to_string()));
}

// ============================================================================
// decode_tokens
// ============================================================================

#[test]
fn test_decode_tokens_basic() {
    let vocab = vec!["hello".to_string(), "Ġworld".to_string(), "!".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert_eq!(result, "hello world!");
}

#[test]
fn test_decode_tokens_special_chars() {
    let vocab = vec![
        "Ċ".to_string(),      // \n
        "ĉ".to_string(),      // \t
        "Ġhello".to_string(), // space + hello
    ];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert_eq!(result, "\n\t hello");
}

#[test]
fn test_decode_tokens_out_of_vocab() {
    let vocab = vec!["hello".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 99]);
    assert_eq!(result, "hello[99]");
}

#[test]
fn test_decode_tokens_empty() {
    let vocab = vec!["hello".to_string()];
    let result = AprV2Model::decode_tokens(&vocab, &[]);
    assert_eq!(result, "");
}

#[test]
fn test_decode_tokens_empty_vocab() {
    let vocab: Vec<String> = vec![];
    let result = AprV2Model::decode_tokens(&vocab, &[0, 1, 2]);
    assert_eq!(result, "[0][1][2]");
}

// ============================================================================
// MappedAprModel::dtype_to_qtype
// ============================================================================

#[test]
fn test_dtype_to_qtype_all_variants() {
    assert_eq!(MappedAprModel::dtype_to_qtype("F32"), 0);
    assert_eq!(MappedAprModel::dtype_to_qtype("F16"), 1);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_0"), 2);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_1"), 3);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_0"), 6);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_1"), 7);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q8_0"), 8);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q8_1"), 9);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q2_K"), 10);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q3_K"), 11);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q4_K"), 12);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q5_K"), 13);
    assert_eq!(MappedAprModel::dtype_to_qtype("Q6_K"), 14);
    assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XXS"), 16);
    assert_eq!(MappedAprModel::dtype_to_qtype("IQ2_XS"), 17);
    assert_eq!(MappedAprModel::dtype_to_qtype("BF16"), 30);
    assert_eq!(MappedAprModel::dtype_to_qtype("UNKNOWN"), 0); // Default
}

// ============================================================================
// extract_special_tokens_from_vocab
// ============================================================================

#[test]
fn test_extract_special_tokens_known_patterns() {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<|im_start|>".to_string(), 151643);
    vocab.insert("<|im_end|>".to_string(), 151644);
    vocab.insert("<|endoftext|>".to_string(), 151645);
    vocab.insert("<s>".to_string(), 1);
    vocab.insert("</s>".to_string(), 2);
    vocab.insert("<pad>".to_string(), 0);
    vocab.insert("<unk>".to_string(), 3);
    vocab.insert("hello".to_string(), 100); // Not special

    let specials = extract_special_tokens_from_vocab(&vocab);
    assert!(specials.contains_key("<|im_start|>"));
    assert!(specials.contains_key("<|im_end|>"));
    assert!(specials.contains_key("<|endoftext|>"));
    assert!(specials.contains_key("<s>"));
    assert!(specials.contains_key("</s>"));
    assert!(specials.contains_key("<pad>"));
    assert!(specials.contains_key("<unk>"));
    assert!(!specials.contains_key("hello"));
}

#[test]
fn test_extract_special_tokens_custom_pattern() {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<|custom_token|>".to_string(), 999);
    vocab.insert("regular_token".to_string(), 100);

    let specials = extract_special_tokens_from_vocab(&vocab);
    // <|custom_token|> matches <|...|> pattern
    assert!(specials.contains_key("<|custom_token|>"));
    assert!(!specials.contains_key("regular_token"));
}

#[test]
fn test_extract_special_tokens_empty_vocab() {
    let vocab: HashMap<String, u32> = HashMap::new();
    let specials = extract_special_tokens_from_vocab(&vocab);
    assert!(specials.is_empty());
}

#[test]
fn test_extract_special_tokens_code_model() {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    vocab.insert("<|fim_prefix|>".to_string(), 100);
    vocab.insert("<|fim_middle|>".to_string(), 101);
    vocab.insert("<|fim_suffix|>".to_string(), 102);

    let specials = extract_special_tokens_from_vocab(&vocab);
    assert!(specials.contains_key("<|fim_prefix|>"));
    assert!(specials.contains_key("<|fim_middle|>"));
    assert!(specials.contains_key("<|fim_suffix|>"));
}
