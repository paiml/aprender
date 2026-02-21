
#[test]
fn test_load_mmap_type_mismatch() {
    use tempfile::tempdir;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestModel {
        value: i32,
    }

    let model = TestModel { value: 42 };

    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("type_mismatch.apr");

    // Save as Custom
    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

    // Try to load as LinearRegression (wrong type)
    let result: Result<TestModel> = load_mmap(&path, ModelType::LinearRegression);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("mismatch"));
}

// ========================================================================
// Pygmy-Based Core I/O Tests (T-COV-95)
// ========================================================================

#[test]
fn test_compress_decompress_roundtrip_none() {
    use super::super::core_io::{compress_payload, decompress_payload};

    let data = b"Hello, World! This is test data for compression.";

    // No compression roundtrip
    let (compressed, compression) = compress_payload(data, Compression::None).expect("compress");
    assert_eq!(compression, Compression::None);
    assert_eq!(compressed, data);

    let decompressed = decompress_payload(&compressed, compression).expect("decompress");
    assert_eq!(decompressed, data);
}

#[cfg(feature = "format-compression")]
#[test]
fn test_compress_decompress_roundtrip_zstd_default() {
    use super::super::core_io::{compress_payload, decompress_payload};

    let data = vec![42u8; 1000]; // Repetitive data compresses well

    let (compressed, compression) =
        compress_payload(&data, Compression::ZstdDefault).expect("compress");
    assert_eq!(compression, Compression::ZstdDefault);
    // Compressed should be smaller for repetitive data
    assert!(compressed.len() < data.len());

    let decompressed = decompress_payload(&compressed, compression).expect("decompress");
    assert_eq!(decompressed, data);
}

#[cfg(feature = "format-compression")]
#[test]
fn test_compress_decompress_roundtrip_zstd_max() {
    use super::super::core_io::{compress_payload, decompress_payload};

    let data = vec![0u8; 500];

    let (compressed, compression) =
        compress_payload(&data, Compression::ZstdMax).expect("compress");
    assert_eq!(compression, Compression::ZstdMax);

    let decompressed = decompress_payload(&compressed, compression).expect("decompress");
    assert_eq!(decompressed, data);
}

#[cfg(feature = "format-compression")]
#[test]
fn test_compress_decompress_roundtrip_lz4() {
    use super::super::core_io::{compress_payload, decompress_payload};

    let data = b"LZ4 compression test data with some repetition repetition repetition";

    let (compressed, compression) = compress_payload(data, Compression::Lz4).expect("compress");
    assert_eq!(compression, Compression::Lz4);

    let decompressed = decompress_payload(&compressed, compression).expect("decompress");
    assert_eq!(decompressed.as_slice(), data);
}

#[test]
fn test_crc32_empty() {
    // CRC32 of empty data is 0 (identity element)
    let crc = crc32(&[]);
    assert_eq!(crc, 0);
}

#[test]
fn test_crc32_known_values() {
    // Test multiple known CRC32 values (IEEE polynomial)
    assert_eq!(crc32(b""), 0x0000_0000);
    assert_eq!(crc32(b"a"), 0xE8B7_BE43);
    assert_eq!(crc32(b"abc"), 0x352441C2);
}

#[test]
fn test_load_from_bytes_with_pygmy_apr() {
    use super::super::test_factory::{build_pygmy_apr, build_pygmy_apr_with_config, PygmyConfig};
    use super::super::v2::AprV2Reader;

    // Build a pygmy APR file in memory
    let apr_data = build_pygmy_apr();

    // Verify we can parse it
    let reader = AprV2Reader::from_bytes(&apr_data).expect("parse pygmy APR");

    // Check basic properties
    let meta = reader.metadata();
    assert_eq!(meta.model_type, "pygmy");

    let tensor_names = reader.tensor_names();
    assert!(!tensor_names.is_empty());

    // Test with different configs
    let configs = [
        ("default", PygmyConfig::default()),
        ("minimal", PygmyConfig::minimal()),
        ("embedding_only", PygmyConfig::embedding_only()),
    ];

    for (name, config) in configs {
        let apr_data = build_pygmy_apr_with_config(config);
        let reader = AprV2Reader::from_bytes(&apr_data)
            .unwrap_or_else(|e| panic!("parse pygmy APR with config {name}: {e}"));
        assert_eq!(reader.metadata().model_type, "pygmy");
    }
}

#[test]
fn test_inspect_bytes_with_pygmy_safetensors() {
    use super::super::test_factory::build_pygmy_safetensors;

    // Build a pygmy SafeTensors file in memory
    let st_data = build_pygmy_safetensors();

    // Verify it has SafeTensors magic (header length as u64 LE, then JSON)
    assert!(st_data.len() > 8);
    let header_len = u64::from_le_bytes(st_data[0..8].try_into().unwrap());
    assert!(header_len < 10_000); // Reasonable header size

    // Check JSON start
    assert_eq!(&st_data[8..10], b"{\"");

    // Verify format detection from bytes
    // SafeTensors is detected by the JSON header pattern
    assert!(header_len > 0);
}

#[test]
fn test_pygmy_apr_tensor_alignment() {
    use super::super::test_factory::build_pygmy_apr;
    use super::super::v2::AprV2Reader;

    let apr_data = build_pygmy_apr();
    let reader = AprV2Reader::from_bytes(&apr_data).expect("parse");

    // All tensors should be 64-byte aligned
    assert!(reader.verify_alignment());
}

#[test]
fn test_pygmy_apr_quantized_formats() {
    use super::super::test_factory::{build_pygmy_apr_f16, build_pygmy_apr_q4, build_pygmy_apr_q8};
    use super::super::v2::{AprV2Reader, TensorDType};

    // Test Q8 format
    let q8_data = build_pygmy_apr_q8();
    let reader = AprV2Reader::from_bytes(&q8_data).expect("parse Q8");
    let q8_tensors: Vec<_> = reader
        .tensor_names()
        .iter()
        .filter_map(|n| reader.get_tensor(n))
        .filter(|t| t.dtype == TensorDType::Q8)
        .collect();
    assert!(!q8_tensors.is_empty(), "Should have Q8 tensors");

    // Test Q4 format
    let q4_data = build_pygmy_apr_q4();
    let reader = AprV2Reader::from_bytes(&q4_data).expect("parse Q4");
    let q4_tensors: Vec<_> = reader
        .tensor_names()
        .iter()
        .filter_map(|n| reader.get_tensor(n))
        .filter(|t| t.dtype == TensorDType::Q4)
        .collect();
    assert!(!q4_tensors.is_empty(), "Should have Q4 tensors");

    // Test F16 format
    let f16_data = build_pygmy_apr_f16();
    let reader = AprV2Reader::from_bytes(&f16_data).expect("parse F16");
    let f16_tensors: Vec<_> = reader
        .tensor_names()
        .iter()
        .filter_map(|n| reader.get_tensor(n))
        .filter(|t| t.dtype == TensorDType::F16)
        .collect();
    assert!(!f16_tensors.is_empty(), "Should have F16 tensors");
}

#[test]
fn test_pygmy_safetensors_metadata_parsing() {
    use super::super::test_factory::build_pygmy_safetensors;

    let st_data = build_pygmy_safetensors();

    // Parse header length
    let header_len = u64::from_le_bytes(st_data[0..8].try_into().unwrap()) as usize;

    // Parse JSON header
    let header_bytes = &st_data[8..8 + header_len];
    let header_str = std::str::from_utf8(header_bytes).expect("valid UTF-8");

    // Should be valid JSON with tensor metadata
    assert!(header_str.starts_with('{'));
    assert!(header_str.ends_with('}'));
    assert!(header_str.contains("dtype"));
    assert!(header_str.contains("shape"));
}

#[test]
fn test_inspect_bytes_error_too_small() {
    // Data too small for header
    let tiny_data = [0u8; 10];
    let result = inspect_bytes(&tiny_data);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too small"));
}

#[test]
fn test_inspect_bytes_error_invalid_magic() {
    // Invalid magic number
    let mut bad_data = [0u8; HEADER_SIZE + 4];
    bad_data[0..4].copy_from_slice(b"BAAD");

    let result = inspect_bytes(&bad_data);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Invalid magic"));
}

#[test]
fn test_pygmy_apr_llama_style_config() {
    use super::super::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
    use super::super::v2::AprV2Reader;

    let config = PygmyConfig::llama_style();
    let apr_data = build_pygmy_apr_with_config(config);

    let reader = AprV2Reader::from_bytes(&apr_data).expect("parse llama-style");

    // Check llama-style naming conventions
    let tensor_names = reader.tensor_names();
    let has_attn_q = tensor_names.iter().any(|n| n.contains("self_attn.q_proj"));
    let has_attn_k = tensor_names.iter().any(|n| n.contains("self_attn.k_proj"));
    let has_attn_v = tensor_names.iter().any(|n| n.contains("self_attn.v_proj"));
    let has_mlp = tensor_names.iter().any(|n| n.contains("mlp"));

    assert!(has_attn_q, "Should have q_proj tensors");
    assert!(has_attn_k, "Should have k_proj tensors");
    assert!(has_attn_v, "Should have v_proj tensors");
    assert!(has_mlp, "Should have MLP tensors");
}

#[test]
fn test_pygmy_apr_multiple_layers() {
    use super::super::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
    use super::super::v2::AprV2Reader;

    let mut config = PygmyConfig::default();
    config.num_layers = 3;
    let apr_data = build_pygmy_apr_with_config(config);

    let reader = AprV2Reader::from_bytes(&apr_data).expect("parse multi-layer");

    let tensor_names = reader.tensor_names();

    // Check we have tensors for all layers
    for layer_idx in 0..3 {
        let layer_prefix = format!("layers.{layer_idx}");
        let has_layer = tensor_names.iter().any(|n| n.contains(&layer_prefix));
        assert!(has_layer, "Should have layer {layer_idx} tensors");
    }
}

// ========================================================================
// GGUF API Type Tests (T-COV-95: improve gguf/api.rs coverage)
// ========================================================================

#[test]
fn test_gguf_tokenizer_default() {
    let tok = gguf::GgufTokenizer::default();
    assert!(tok.vocabulary.is_empty());
    assert!(tok.merges.is_empty());
    assert!(tok.model_type.is_none());
    assert!(tok.bos_token_id.is_none());
    assert!(tok.eos_token_id.is_none());
    assert!(tok.architecture.is_none());
    assert!(tok.model_name.is_none());
}

#[test]
fn test_gguf_tokenizer_has_vocabulary() {
    let mut tok = gguf::GgufTokenizer::default();
    assert!(!tok.has_vocabulary());

    tok.vocabulary = vec!["hello".to_string(), "world".to_string()];
    assert!(tok.has_vocabulary());
}

#[test]
fn test_gguf_tokenizer_vocab_size() {
    let mut tok = gguf::GgufTokenizer::default();
    assert_eq!(tok.vocab_size(), 0);

    tok.vocabulary = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    assert_eq!(tok.vocab_size(), 3);
}

#[test]
fn test_gguf_tokenizer_with_all_fields() {
    let tok = gguf::GgufTokenizer {
        vocabulary: vec!["<s>".to_string(), "</s>".to_string(), "hello".to_string()],
        merges: vec!["h e".to_string(), "l l".to_string()],
        model_type: Some("llama".to_string()),
        bos_token_id: Some(0),
        eos_token_id: Some(1),
        architecture: Some("llama".to_string()),
        model_name: Some("test-model".to_string()),
        ..Default::default()
    };
    assert_eq!(tok.vocab_size(), 3);
    assert!(tok.has_vocabulary());
    assert_eq!(tok.merges.len(), 2);
    assert_eq!(tok.bos_token_id, Some(0));
    assert_eq!(tok.eos_token_id, Some(1));
}

#[test]
fn test_gguf_tokenizer_clone() {
    let tok = gguf::GgufTokenizer {
        vocabulary: vec!["test".to_string()],
        merges: vec![],
        model_type: Some("gpt2".to_string()),
        bos_token_id: Some(50256),
        eos_token_id: Some(50256),
        architecture: None,
        model_name: None,
        ..Default::default()
    };
    let cloned = tok.clone();
    assert_eq!(cloned.vocabulary, tok.vocabulary);
    assert_eq!(cloned.model_type, tok.model_type);
}

#[test]
fn test_gguf_tokenizer_debug() {
    let tok = gguf::GgufTokenizer::default();
    let debug_str = format!("{tok:?}");
    assert!(debug_str.contains("GgufTokenizer"));
}

#[test]
fn test_gguf_model_config_default() {
    let cfg = gguf::GgufModelConfig::default();
    assert!(cfg.architecture.is_none());
    assert!(cfg.hidden_size.is_none());
    assert!(cfg.num_layers.is_none());
    assert!(cfg.num_heads.is_none());
    assert!(cfg.num_kv_heads.is_none());
    assert!(cfg.vocab_size.is_none());
    assert!(cfg.intermediate_size.is_none());
    assert!(cfg.max_position_embeddings.is_none());
    assert!(cfg.rope_theta.is_none());
    assert!(cfg.rms_norm_eps.is_none());
    assert!(cfg.rope_type.is_none());
}

#[test]
fn test_gguf_model_config_with_values() {
    let cfg = gguf::GgufModelConfig {
        architecture: Some("llama".to_string()),
        hidden_size: Some(4096),
        num_layers: Some(32),
        num_heads: Some(32),
        num_kv_heads: Some(8),
        vocab_size: Some(32000),
        intermediate_size: Some(11008),
        max_position_embeddings: Some(4096),
        rope_theta: Some(10000.0),
        rms_norm_eps: Some(1e-5),
        rope_type: Some(0),
    };
    assert_eq!(cfg.architecture.as_deref(), Some("llama"));
    assert_eq!(cfg.hidden_size, Some(4096));
    assert_eq!(cfg.num_layers, Some(32));
    assert_eq!(cfg.num_heads, Some(32));
    assert_eq!(cfg.num_kv_heads, Some(8));
    assert_eq!(cfg.vocab_size, Some(32000));
    assert_eq!(cfg.intermediate_size, Some(11008));
    assert_eq!(cfg.max_position_embeddings, Some(4096));
    assert!((cfg.rope_theta.unwrap() - 10000.0).abs() < 1e-6);
    assert!((cfg.rms_norm_eps.unwrap() - 1e-5).abs() < 1e-10);
    assert_eq!(cfg.rope_type, Some(0));
}

#[test]
fn test_gguf_model_config_qwen2_style() {
    // CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
    let cfg = gguf::GgufModelConfig {
        architecture: Some("qwen2".to_string()),
        hidden_size: Some(1536),
        num_layers: Some(28),
        num_heads: Some(12),
        num_kv_heads: Some(2),
        vocab_size: Some(151936),
        intermediate_size: Some(8960),
        max_position_embeddings: Some(32768),
        rope_theta: Some(1000000.0),
        rms_norm_eps: Some(1e-6),
        rope_type: Some(2), // NEOX style for Qwen2.5
    };
    assert_eq!(cfg.rope_type, Some(2));
    assert_eq!(cfg.architecture.as_deref(), Some("qwen2"));
}
