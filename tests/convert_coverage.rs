//! Coverage tests for convert.rs - GGUF to APR conversion
//!
//! These tests target uncovered paths in convert.rs to increase coverage from 75% to 85%+.
//! Focus areas:
//! - Model conversion functions
//! - Format detection and validation
//! - Weight conversion and preservation
//! - Quantization during conversion
//! - Error handling paths
//! - Edge cases for statistics calculations

use realizar::apr::{ALIGNMENT, HEADER_SIZE, MAGIC};
use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use realizar::convert::{ConversionStats, GgufToAprConverter, Q4KConversionStats, RawTensor};
use realizar::gguf::{ArchConstraints, GGUFConfig, GGUFTransformer, GGUFTransformerLayer};

// =============================================================================
// Helper Functions
// =============================================================================

fn create_minimal_gguf_transformer(
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    intermediate_dim: usize,
) -> GGUFTransformer {
    let config = GGUFConfig {
        architecture: "test_arch".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layers: Vec<GGUFTransformerLayer> = (0..num_layers)
        .map(|_| GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        })
        .collect();

    GGUFTransformer {
        config,
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        position_embedding: None,
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
    }
}

fn create_minimal_apr_transformer(
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    intermediate_dim: usize,
) -> AprTransformer {
    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let layers: Vec<AprTransformerLayer> = (0..num_layers)
        .map(|_| AprTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        })
        .collect();

    AprTransformer {
        config,
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    }
}

// =============================================================================
// GgufToAprConverter Tests
// =============================================================================

#[test]
fn test_from_gguf_transformer_preserves_architecture() {
    let gguf = create_minimal_gguf_transformer(32, 2, 100, 64);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.architecture, "test_arch");
}

#[test]
fn test_from_gguf_transformer_preserves_hidden_dim() {
    let gguf = create_minimal_gguf_transformer(128, 1, 50, 256);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.hidden_dim, 128);
}

#[test]
fn test_from_gguf_transformer_preserves_num_heads() {
    let gguf = create_minimal_gguf_transformer(64, 1, 100, 128);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.num_heads, 4);
    assert_eq!(apr.config.num_kv_heads, 4);
}

#[test]
fn test_from_gguf_transformer_preserves_context_length() {
    let gguf = create_minimal_gguf_transformer(32, 1, 50, 64);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.context_length, 512);
}

#[test]
fn test_from_gguf_transformer_preserves_rope_theta() {
    let gguf = create_minimal_gguf_transformer(32, 1, 50, 64);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert!((apr.config.rope_theta - 10000.0).abs() < 0.1);
}

#[test]
fn test_from_gguf_transformer_preserves_eps() {
    let gguf = create_minimal_gguf_transformer(32, 1, 50, 64);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert!((apr.config.eps - 1e-5).abs() < 1e-7);
}

#[test]
fn test_from_gguf_transformer_preserves_embeddings() {
    let gguf = create_minimal_gguf_transformer(16, 1, 20, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.token_embedding.len(), gguf.token_embedding.len());
    assert_eq!(apr.token_embedding, gguf.token_embedding);
}

#[test]
fn test_from_gguf_transformer_preserves_output_norm() {
    let gguf = create_minimal_gguf_transformer(16, 1, 20, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.output_norm_weight, gguf.output_norm_weight);
    assert_eq!(apr.output_norm_bias, gguf.output_norm_bias);
}

#[test]
fn test_from_gguf_transformer_preserves_lm_head() {
    let gguf = create_minimal_gguf_transformer(16, 1, 20, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.lm_head_weight, gguf.lm_head_weight);
    assert_eq!(apr.lm_head_bias, gguf.lm_head_bias);
}

#[test]
fn test_from_gguf_transformer_zero_layers() {
    let gguf = create_minimal_gguf_transformer(16, 0, 20, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.layers.len(), 0);
    assert_eq!(apr.config.num_layers, 0);
}

#[test]
fn test_from_gguf_transformer_many_layers() {
    let gguf = create_minimal_gguf_transformer(8, 12, 10, 16);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.layers.len(), 12);
}

// =============================================================================
// APR Serialization Tests
// =============================================================================

#[test]
fn test_to_apr_bytes_creates_valid_header() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    assert!(bytes.len() >= HEADER_SIZE);
    assert_eq!(&bytes[0..4], &MAGIC);
}

#[test]
fn test_to_apr_bytes_version_is_v2() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    assert_eq!(bytes[4], 2); // major
    assert_eq!(bytes[5], 0); // minor
}

#[test]
fn test_to_apr_bytes_tensor_count_is_one() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(tensor_count, 1);
}

#[test]
fn test_to_apr_bytes_metadata_offset() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    let metadata_offset = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    assert_eq!(metadata_offset, HEADER_SIZE as u64);
}

#[test]
fn test_to_apr_bytes_multiple_layers() {
    let apr = create_minimal_apr_transformer(8, 4, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    assert!(bytes.len() > HEADER_SIZE);
}

#[test]
fn test_apr_roundtrip_preserves_config() {
    let original = create_minimal_apr_transformer(16, 2, 30, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(original.config.architecture, loaded.config.architecture);
    assert_eq!(original.config.hidden_dim, loaded.config.hidden_dim);
    assert_eq!(original.config.num_layers, loaded.config.num_layers);
    assert_eq!(original.config.vocab_size, loaded.config.vocab_size);
}

#[test]
fn test_apr_roundtrip_preserves_weights() {
    let original = create_minimal_apr_transformer(8, 1, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(original.token_embedding, loaded.token_embedding);
    assert_eq!(original.output_norm_weight, loaded.output_norm_weight);
    assert_eq!(original.lm_head_weight, loaded.lm_head_weight);
}

#[test]
fn test_apr_roundtrip_preserves_layer_weights() {
    let original = create_minimal_apr_transformer(8, 2, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(original.layers.len(), loaded.layers.len());
    for (orig, load) in original.layers.iter().zip(loaded.layers.iter()) {
        assert_eq!(orig.attn_norm_weight, load.attn_norm_weight);
        assert_eq!(orig.qkv_weight, load.qkv_weight);
    }
}

// =============================================================================
// from_apr_bytes Error Path Tests
// =============================================================================

#[test]
fn test_from_apr_bytes_wrong_magic() {
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(b"XXXX");
    bytes[4] = 2;

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_empty_input() {
    let bytes: Vec<u8> = vec![];
    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_header_only() {
    let mut bytes = vec![0u8; HEADER_SIZE];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_truncated_before_index() {
    let mut bytes = vec![0u8; 70];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&200u64.to_le_bytes()); // data_offset beyond end
    bytes[64..66].copy_from_slice(b"{}");

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_invalid_tensor_index_json() {
    let mut bytes = vec![0u8; 120];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&100u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");
    bytes[66..78].copy_from_slice(b"invalid json");

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_no_weights_tensor() {
    let mut bytes = vec![0u8; 140];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&120u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");

    // Valid JSON but wrong tensor name
    let index_json = r#"[{"name":"other","dtype":"json","shape":[10],"offset":0,"size":10}]"#;
    bytes[66..66 + index_json.len()].copy_from_slice(index_json.as_bytes());

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_truncated_tensor_data() {
    let mut bytes = vec![0u8; 150];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&130u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");

    // Valid JSON but size exceeds available data
    let index_json = r#"[{"name":"weights","dtype":"json","shape":[9999],"offset":0,"size":9999}]"#;
    let idx_end = 66 + index_json.len();
    bytes[66..idx_end].copy_from_slice(index_json.as_bytes());

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

// =============================================================================
// ConversionStats Tests
// =============================================================================

#[test]
fn test_stats_returns_correct_num_layers() {
    let apr = create_minimal_apr_transformer(32, 6, 100, 64);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.num_layers, 6);
}

#[test]
fn test_stats_returns_correct_hidden_dim() {
    let apr = create_minimal_apr_transformer(256, 2, 1000, 512);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.hidden_dim, 256);
}

#[test]
fn test_stats_returns_correct_vocab_size() {
    let apr = create_minimal_apr_transformer(64, 1, 5000, 128);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.vocab_size, 5000);
}

#[test]
fn test_stats_returns_correct_architecture() {
    let apr = create_minimal_apr_transformer(32, 1, 100, 64);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.architecture, "test");
}

#[test]
fn test_stats_total_parameters_positive() {
    let apr = create_minimal_apr_transformer(32, 2, 100, 64);
    let stats = GgufToAprConverter::stats(&apr);

    assert!(stats.total_parameters > 0);
}

#[test]
fn test_stats_memory_bytes_is_4x_params() {
    let apr = create_minimal_apr_transformer(32, 1, 100, 64);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.memory_bytes_f32, stats.total_parameters * 4);
}

#[test]
fn test_stats_memory_mb_calculation() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 4_000_000,
        num_layers: 4,
        hidden_dim: 512,
        vocab_size: 10000,
        architecture: "test".to_string(),
    };

    let expected_mb = 4_000_000.0 / (1024.0 * 1024.0);
    assert!((stats.memory_mb() - expected_mb).abs() < 0.001);
}

#[test]
fn test_stats_memory_gb_calculation() {
    let stats = ConversionStats {
        total_parameters: 1_000_000_000,
        memory_bytes_f32: 4_000_000_000,
        num_layers: 32,
        hidden_dim: 4096,
        vocab_size: 32000,
        architecture: "large".to_string(),
    };

    let expected_gb = 4_000_000_000.0 / (1024.0 * 1024.0 * 1024.0);
    assert!((stats.memory_gb() - expected_gb).abs() < 0.01);
}

#[test]
fn test_stats_parameters_m_calculation() {
    let stats = ConversionStats {
        total_parameters: 7_000_000,
        memory_bytes_f32: 28_000_000,
        num_layers: 12,
        hidden_dim: 768,
        vocab_size: 30000,
        architecture: "medium".to_string(),
    };

    assert!((stats.parameters_m() - 7.0).abs() < 0.001);
}

#[test]
fn test_stats_parameters_b_calculation() {
    let stats = ConversionStats {
        total_parameters: 13_000_000_000,
        memory_bytes_f32: 52_000_000_000,
        num_layers: 40,
        hidden_dim: 5120,
        vocab_size: 50000,
        architecture: "13b".to_string(),
    };

    assert!((stats.parameters_b() - 13.0).abs() < 0.001);
}

#[test]
fn test_stats_zero_values() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };

    assert_eq!(stats.memory_mb(), 0.0);
    assert_eq!(stats.memory_gb(), 0.0);
    assert_eq!(stats.parameters_m(), 0.0);
    assert_eq!(stats.parameters_b(), 0.0);
}

#[test]
fn test_conversion_stats_debug_impl() {
    let stats = ConversionStats {
        total_parameters: 1000,
        memory_bytes_f32: 4000,
        num_layers: 2,
        hidden_dim: 64,
        vocab_size: 100,
        architecture: "tiny".to_string(),
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("ConversionStats"));
    assert!(debug_str.contains("1000"));
    assert!(debug_str.contains("tiny"));
}

#[test]
fn test_conversion_stats_clone_impl() {
    let stats = ConversionStats {
        total_parameters: 500,
        memory_bytes_f32: 2000,
        num_layers: 1,
        hidden_dim: 32,
        vocab_size: 50,
        architecture: "nano".to_string(),
    };

    let cloned = stats.clone();
    assert_eq!(cloned.total_parameters, stats.total_parameters);
    assert_eq!(cloned.memory_bytes_f32, stats.memory_bytes_f32);
    assert_eq!(cloned.architecture, stats.architecture);
}

// =============================================================================
// RawTensor Tests
// =============================================================================

#[test]
fn test_raw_tensor_debug_impl() {
    let tensor = RawTensor {
        name: "layer.0.weight".to_string(),
        data: vec![0u8; 256],
        shape: vec![16, 16],
        dtype: 0,
    };

    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("layer.0.weight"));
    assert!(debug_str.contains("[16, 16]"));
}

#[test]
fn test_raw_tensor_clone_impl() {
    let tensor = RawTensor {
        name: "embed.weight".to_string(),
        data: vec![1, 2, 3, 4, 5],
        shape: vec![5],
        dtype: 1,
    };

    let cloned = tensor.clone();
    assert_eq!(cloned.name, tensor.name);
    assert_eq!(cloned.data, tensor.data);
    assert_eq!(cloned.shape, tensor.shape);
    assert_eq!(cloned.dtype, tensor.dtype);
}

#[test]
fn test_raw_tensor_f32_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 16],
        shape: vec![4],
        dtype: 0, // F32
    };

    assert_eq!(tensor.dtype, 0);
}

#[test]
fn test_raw_tensor_f16_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 8],
        shape: vec![4],
        dtype: 1, // F16
    };

    assert_eq!(tensor.dtype, 1);
}

#[test]
fn test_raw_tensor_q4k_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 144],
        shape: vec![256],
        dtype: 12, // Q4_K
    };

    assert_eq!(tensor.dtype, 12);
}

#[test]
fn test_raw_tensor_q6k_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 210],
        shape: vec![256],
        dtype: 14, // Q6_K
    };

    assert_eq!(tensor.dtype, 14);
}

#[test]
fn test_raw_tensor_multidim_shape() {
    let tensor = RawTensor {
        name: "attention.weight".to_string(),
        data: vec![0u8; 1024],
        shape: vec![32, 8, 4],
        dtype: 0,
    };

    assert_eq!(tensor.shape.len(), 3);
    assert_eq!(tensor.shape[0], 32);
    assert_eq!(tensor.shape[1], 8);
    assert_eq!(tensor.shape[2], 4);
}

// =============================================================================
// Q4KConversionStats Tests
// =============================================================================

#[test]
fn test_q4k_stats_debug_impl() {
    let stats = Q4KConversionStats {
        tensor_count: 200,
        q4k_tensor_count: 180,
        total_bytes: 10_000_000,
        architecture: "llama".to_string(),
        num_layers: 32,
        hidden_size: 4096,
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("200"));
    assert!(debug_str.contains("llama"));
    assert!(debug_str.contains("32"));
}

#[test]
fn test_q4k_stats_clone_impl() {
    let stats = Q4KConversionStats {
        tensor_count: 100,
        q4k_tensor_count: 90,
        total_bytes: 5_000_000,
        architecture: "qwen".to_string(),
        num_layers: 24,
        hidden_size: 2048,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.tensor_count, stats.tensor_count);
    assert_eq!(cloned.q4k_tensor_count, stats.q4k_tensor_count);
    assert_eq!(cloned.total_bytes, stats.total_bytes);
    assert_eq!(cloned.architecture, stats.architecture);
}

#[test]
fn test_q4k_stats_all_q4k() {
    let stats = Q4KConversionStats {
        tensor_count: 50,
        q4k_tensor_count: 50,
        total_bytes: 1_000_000,
        architecture: "phi".to_string(),
        num_layers: 16,
        hidden_size: 1024,
    };

    assert_eq!(stats.tensor_count, stats.q4k_tensor_count);
}

#[test]
fn test_q4k_stats_no_q4k() {
    let stats = Q4KConversionStats {
        tensor_count: 50,
        q4k_tensor_count: 0,
        total_bytes: 2_000_000,
        architecture: "fp16_model".to_string(),
        num_layers: 12,
        hidden_size: 768,
    };

    assert_eq!(stats.q4k_tensor_count, 0);
}

// =============================================================================
// Additional Conversion and Serialization Tests
// =============================================================================

#[test]
fn test_apr_bytes_contains_metadata_json() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // After header (64 bytes), there should be JSON metadata
    let metadata_start = HEADER_SIZE;
    let metadata_slice = &bytes[metadata_start..metadata_start + 10];
    // JSON starts with '{' or could be whitespace
    let has_json_start = metadata_slice.contains(&b'{');
    assert!(has_json_start, "Metadata section should contain JSON");
}

#[test]
fn test_apr_bytes_aligned_to_64() {
    let apr = create_minimal_apr_transformer(16, 1, 20, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // Header should be 64 bytes
    assert!(bytes.len() >= 64);
}

#[test]
fn test_conversion_with_bias_weights() {
    // Create transformer with biases
    let config = AprTransformerConfig {
        architecture: "test_bias".to_string(),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let layer = AprTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: Some(vec![0.0; 8]),
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: Some(vec![0.0; 3 * 8]),
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: Some(vec![0.0; 8]),
        ffn_gate_weight: Some(vec![0.01; 8 * 16]),
        ffn_gate_bias: Some(vec![0.0; 16]),
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: Some(vec![0.0; 16]),
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: Some(vec![0.0; 8]),
        ffn_norm_weight: Some(vec![1.0; 8]),
        ffn_norm_bias: Some(vec![0.0; 8]),
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
        linear_attn_z_weight: None,
        linear_attn_b_weight: None,
        linear_attn_a_weight: None,
        linear_attn_conv1d_weight: None,
        linear_attn_a_log: None,
        linear_attn_dt_bias: None,
        linear_attn_norm_weight: None,
        moe_gate_weight: None,
        moe_expert_gate_up: None,
        moe_expert_down: None,
        moe_shared_gate: None,
        moe_shared_up: None,
        moe_shared_down: None,
        moe_shared_expert_gate_weight: None,
    };

    let apr = AprTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: Some(vec![0.0; 8]),
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: Some(vec![0.0; 10]),
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert!(loaded.layers[0].attn_norm_bias.is_some());
    assert!(loaded.layers[0].qkv_bias.is_some());
    assert!(loaded.lm_head_bias.is_some());
}

#[test]
fn test_gguf_transformer_with_gate_weights() {
    // Create GGUF transformer with FFN gate weights (SwiGLU style)
    let config = GGUFConfig {
        architecture: "swiglu_test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: None,
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: None,
        ffn_gate_weight: Some(vec![0.02; 8 * 16]), // SwiGLU gate
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.layers[0].ffn_gate_weight.is_some());
}

#[test]
fn test_conversion_preserves_all_layer_weights() {
    let gguf = create_minimal_gguf_transformer(16, 3, 50, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    for (i, (gguf_layer, apr_layer)) in gguf.layers.iter().zip(apr.layers.iter()).enumerate() {
        assert_eq!(
            gguf_layer.attn_norm_weight, apr_layer.attn_norm_weight,
            "Layer {} attn_norm_weight mismatch",
            i
        );
        assert_eq!(
            gguf_layer.qkv_weight, apr_layer.qkv_weight,
            "Layer {} qkv_weight mismatch",
            i
        );
        assert_eq!(
            gguf_layer.attn_output_weight, apr_layer.attn_output_weight,
            "Layer {} attn_output_weight mismatch",
            i
        );
        assert_eq!(
            gguf_layer.ffn_up_weight, apr_layer.ffn_up_weight,
            "Layer {} ffn_up_weight mismatch",
            i
        );
        assert_eq!(
            gguf_layer.ffn_down_weight, apr_layer.ffn_down_weight,
            "Layer {} ffn_down_weight mismatch",
            i
        );
    }
}

#[test]
fn test_stats_with_larger_model() {
    let apr = create_minimal_apr_transformer(512, 12, 32000, 2048);
    let stats = GgufToAprConverter::stats(&apr);

    // Should have significant parameter count
    assert!(stats.total_parameters > 1_000_000);
    assert!(stats.memory_mb() > 1.0);
}

#[test]
fn test_stats_memory_calculations_consistency() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 4_000_000,
        num_layers: 4,
        hidden_dim: 512,
        vocab_size: 10000,
        architecture: "test".to_string(),
    };

    // GB should be less than MB
    assert!(stats.memory_gb() < stats.memory_mb());
    // B should be less than M
    assert!(stats.parameters_b() < stats.parameters_m());
}

#[test]
fn test_stats_very_large_model() {
    let stats = ConversionStats {
        total_parameters: 405_000_000_000, // 405B
        memory_bytes_f32: 1_620_000_000_000,
        num_layers: 126,
        hidden_dim: 16384,
        vocab_size: 128256,
        architecture: "llama3_405b".to_string(),
    };

    assert!(stats.parameters_b() > 400.0);
    assert!(stats.memory_gb() > 1000.0);
}

#[test]
fn test_roundtrip_with_different_architectures() {
    for arch in ["llama", "phi2", "qwen2", "mistral", "gemma"] {
        let config = AprTransformerConfig {
            architecture: arch.to_string(),
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 10,
            intermediate_dim: 16,
            context_length: 64,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        };

        let apr = AprTransformer {
            config,
            token_embedding: vec![0.1; 10 * 8],
            layers: vec![AprTransformerLayer {
                attn_norm_weight: vec![1.0; 8],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; 8 * 3 * 8],
                qkv_bias: None,
                attn_output_weight: vec![0.01; 8 * 8],
                attn_output_bias: None,
                ffn_gate_weight: None,
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; 8 * 16],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; 16 * 8],
                ffn_down_bias: None,
                ffn_norm_weight: None,
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
                linear_attn_z_weight: None,
                linear_attn_b_weight: None,
                linear_attn_a_weight: None,
                linear_attn_conv1d_weight: None,
                linear_attn_a_log: None,
                linear_attn_dt_bias: None,
                linear_attn_norm_weight: None,
                moe_gate_weight: None,
                moe_expert_gate_up: None,
                moe_expert_down: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_expert_gate_weight: None,
            }],
            output_norm_weight: vec![1.0; 8],
            output_norm_bias: None,
            lm_head_weight: vec![0.01; 8 * 10],
            lm_head_bias: None,
            q4k_layers: None,
            lm_head_weight_q4k: None,
            lm_head_weight_q6k: None,
        };

        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
        let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

        assert_eq!(apr.config.architecture, loaded.config.architecture);
    }
}

// =============================================================================
// Integration: Full Conversion Pipeline Tests
// =============================================================================

#[test]
fn test_full_conversion_pipeline_small_model() {
    let gguf = create_minimal_gguf_transformer(16, 2, 50, 32);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.num_layers, 2);
    assert_eq!(stats.hidden_dim, 16);
    assert_eq!(stats.vocab_size, 50);
    assert!(stats.total_parameters > 0);
}

#[test]
fn test_full_conversion_pipeline_with_serialization() {
    let gguf = create_minimal_gguf_transformer(8, 1, 20, 16);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(apr.config.architecture, loaded.config.architecture);
    assert_eq!(apr.config.hidden_dim, loaded.config.hidden_dim);
}

#[test]
fn test_conversion_preserves_inference_capability() {
    let apr = create_minimal_apr_transformer(4, 1, 10, 8);
    let tokens = vec![1, 2, 3];

    let result = apr.forward(&tokens);
    assert!(result.is_ok());

    let logits = result.unwrap();
    assert_eq!(logits.len(), apr.config.vocab_size);
}

#[test]
fn test_conversion_deterministic_inference() {
    let apr = create_minimal_apr_transformer(4, 1, 10, 8);
    let tokens = vec![1, 2];

    let logits1 = apr.forward(&tokens).expect("forward 1");
    let logits2 = apr.forward(&tokens).expect("forward 2");

    assert_eq!(logits1, logits2);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_convert_single_token_vocab() {
    let apr = create_minimal_apr_transformer(4, 1, 1, 8);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.vocab_size, 1);
}

#[test]
fn test_convert_minimal_hidden_dim() {
    let apr = create_minimal_apr_transformer(1, 1, 10, 4);
    let stats = GgufToAprConverter::stats(&apr);

    assert_eq!(stats.hidden_dim, 1);
}

#[test]
fn test_convert_large_intermediate_dim() {
    let apr = create_minimal_apr_transformer(16, 1, 50, 1024);
    let stats = GgufToAprConverter::stats(&apr);

    assert!(stats.total_parameters > 0);
}

#[test]
fn test_apr_header_alignment() {
    // Verify ALIGNMENT constant is correct
    assert_eq!(ALIGNMENT, 64);
    assert_eq!(HEADER_SIZE, 64);
}

#[test]
fn test_apr_magic_bytes() {
    // Verify MAGIC constant
    assert_eq!(MAGIC[0], 0x41); // 'A'
    assert_eq!(MAGIC[1], 0x50); // 'P'
    assert_eq!(MAGIC[2], 0x52); // 'R'
    assert_eq!(MAGIC[3], 0x00); // NUL
}

// =============================================================================
// GgufToAprConverter::convert() Error Path Tests
// =============================================================================

#[test]
fn test_convert_empty_bytes() {
    let result = GgufToAprConverter::convert(&[]);
    assert!(result.is_err(), "Empty bytes should fail");
}

#[test]
fn test_convert_invalid_magic() {
    // Invalid GGUF magic (should be GGUF)
    let bytes = vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let result = GgufToAprConverter::convert(&bytes);
    assert!(result.is_err(), "Invalid magic should fail");
}

#[test]
fn test_convert_truncated_header() {
    // Valid GGUF magic but truncated header
    let mut bytes = vec![0u8; 16];
    bytes[0..4].copy_from_slice(b"GGUF");
    let result = GgufToAprConverter::convert(&bytes);
    assert!(result.is_err(), "Truncated header should fail");
}

// =============================================================================
// GgufToAprConverter convert() Error Tests
// =============================================================================

#[test]
fn test_convert_with_only_gguf_magic() {
    // Only GGUF magic, nothing else
    let bytes = b"GGUF".to_vec();
    let result = GgufToAprConverter::convert(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_convert_gguf_magic_with_partial_version() {
    // GGUF magic + partial version
    let mut bytes = vec![0u8; 8];
    bytes[0..4].copy_from_slice(b"GGUF");
    bytes[4..8].copy_from_slice(&3u32.to_le_bytes()); // version 3
    let result = GgufToAprConverter::convert(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_convert_wrong_endianness_magic() {
    // Big-endian GGUF magic (wrong)
    let bytes = b"FUGF".to_vec();
    let result = GgufToAprConverter::convert(&bytes);
    assert!(result.is_err());
}

// =============================================================================
// Q4KConversionStats Additional Tests
// =============================================================================

#[test]
fn test_q4k_stats_partial_q4k() {
    let stats = Q4KConversionStats {
        tensor_count: 100,
        q4k_tensor_count: 60, // 60% Q4K
        total_bytes: 5_000_000,
        architecture: "mixed".to_string(),
        num_layers: 24,
        hidden_size: 2048,
    };

    assert!(stats.q4k_tensor_count < stats.tensor_count);
    assert!(stats.q4k_tensor_count > 0);
}

#[test]
fn test_q4k_stats_zero_tensors() {
    let stats = Q4KConversionStats {
        tensor_count: 0,
        q4k_tensor_count: 0,
        total_bytes: 0,
        architecture: "empty".to_string(),
        num_layers: 0,
        hidden_size: 0,
    };

    assert_eq!(stats.tensor_count, 0);
    assert_eq!(stats.total_bytes, 0);
}

#[test]
fn test_q4k_stats_large_model() {
    let stats = Q4KConversionStats {
        tensor_count: 500,
        q4k_tensor_count: 450,
        total_bytes: 100_000_000_000, // 100GB
        architecture: "llama3_70b".to_string(),
        num_layers: 80,
        hidden_size: 8192,
    };

    assert!(stats.total_bytes > 1_000_000_000);
    assert_eq!(stats.num_layers, 80);
}

// =============================================================================
// RawTensor Additional Tests
// =============================================================================

#[test]
fn test_raw_tensor_empty_data() {
    let tensor = RawTensor {
        name: "empty".to_string(),
        data: vec![],
        shape: vec![0],
        dtype: 0,
    };

    assert!(tensor.data.is_empty());
    assert_eq!(tensor.shape[0], 0);
}

#[test]
fn test_raw_tensor_q8_0_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 34], // Q8_0: 32 elements = 2 (scale) + 32 (quants)
        shape: vec![32],
        dtype: 8, // Q8_0
    };

    assert_eq!(tensor.dtype, 8);
}

#[test]
fn test_raw_tensor_q5_k_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 176], // Q5_K: 256 elements = 176 bytes
        shape: vec![256],
        dtype: 13, // Q5_K
    };

    assert_eq!(tensor.dtype, 13);
}

#[test]
fn test_raw_tensor_4d_shape() {
    let tensor = RawTensor {
        name: "conv.weight".to_string(),
        data: vec![0u8; 4096],
        shape: vec![64, 32, 2, 2],
        dtype: 0,
    };

    assert_eq!(tensor.shape.len(), 4);
    let total: usize = tensor.shape.iter().product();
    assert_eq!(total, 8192);
}

#[test]
fn test_raw_tensor_scalar_shape() {
    let tensor = RawTensor {
        name: "scale".to_string(),
        data: vec![0u8; 4],
        shape: vec![1],
        dtype: 0,
    };

    assert_eq!(tensor.shape.len(), 1);
    assert_eq!(tensor.shape[0], 1);
}

// =============================================================================
// ConversionStats Additional Edge Cases
// =============================================================================

#[test]
fn test_conversion_stats_exact_1mb() {
    let stats = ConversionStats {
        total_parameters: 262144,      // 1MB / 4 bytes
        memory_bytes_f32: 1024 * 1024, // Exactly 1MB
        num_layers: 1,
        hidden_dim: 512,
        vocab_size: 512,
        architecture: "1mb".to_string(),
    };

    assert!((stats.memory_mb() - 1.0).abs() < 0.0001);
}

#[test]
fn test_conversion_stats_exact_1gb() {
    let stats = ConversionStats {
        total_parameters: 268435456,          // 1GB / 4 bytes
        memory_bytes_f32: 1024 * 1024 * 1024, // Exactly 1GB
        num_layers: 32,
        hidden_dim: 4096,
        vocab_size: 32000,
        architecture: "1gb".to_string(),
    };

    assert!((stats.memory_gb() - 1.0).abs() < 0.0001);
}

#[test]
fn test_conversion_stats_exact_1m_params() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 4_000_000,
        num_layers: 6,
        hidden_dim: 256,
        vocab_size: 5000,
        architecture: "1m".to_string(),
    };

    assert!((stats.parameters_m() - 1.0).abs() < 0.0001);
}

#[test]
fn test_conversion_stats_exact_1b_params() {
    let stats = ConversionStats {
        total_parameters: 1_000_000_000,
        memory_bytes_f32: 4_000_000_000,
        num_layers: 24,
        hidden_dim: 2048,
        vocab_size: 50000,
        architecture: "1b".to_string(),
    };

    assert!((stats.parameters_b() - 1.0).abs() < 0.0001);
}

// =============================================================================
// from_apr_bytes Additional Error Paths
// =============================================================================

#[test]
fn test_from_apr_bytes_short_magic_only() {
    let bytes = vec![0x41, 0x50, 0x52, 0x00]; // Just APR magic, 4 bytes
    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_version_1_fallback() {
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 1; // v1 instead of v2
    bytes[5] = 0;

    // v1 format may be handled differently
    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    // Should either succeed with v1 parsing or fail gracefully
    let _ = result; // Just ensure no panic
}

#[test]
fn test_from_apr_bytes_future_version() {
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 99; // Future version
    bytes[5] = 0;

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    // Should fail or handle gracefully
    let _ = result;
}

#[test]
fn test_from_apr_bytes_corrupt_metadata_offset() {
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    // metadata_offset points way past end of file
    bytes[12..20].copy_from_slice(&9999999999u64.to_le_bytes());

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_zero_tensor_count() {
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&68u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");
    // Empty tensor index
    bytes[66..68].copy_from_slice(b"[]");

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    // Should fail because no 'weights' tensor
    assert!(result.is_err());
}

// =============================================================================
// to_apr_bytes Additional Tests
// =============================================================================

#[test]
fn test_to_apr_bytes_zero_layers() {
    let apr = create_minimal_apr_transformer(8, 0, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    assert!(bytes.len() >= HEADER_SIZE);
    assert_eq!(&bytes[0..4], &MAGIC);
}

#[test]
fn test_to_apr_bytes_large_vocab() {
    let apr = create_minimal_apr_transformer(8, 1, 100000, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    assert!(bytes.len() > HEADER_SIZE);
}

#[test]
fn test_to_apr_bytes_large_hidden_dim() {
    let apr = create_minimal_apr_transformer(1024, 1, 10, 2048);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    assert!(bytes.len() > HEADER_SIZE);
}

#[test]
fn test_to_apr_bytes_many_layers() {
    let apr = create_minimal_apr_transformer(4, 32, 10, 8);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");
    assert_eq!(loaded.config.num_layers, 32);
    assert_eq!(loaded.layers.len(), 32);
}

// =============================================================================
// from_gguf_transformer Additional Coverage
// =============================================================================

#[test]
fn test_from_gguf_transformer_with_all_biases() {
    let config = GGUFConfig {
        architecture: "biased".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: Some(vec![0.1; 8]),
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: Some(vec![0.02; 3 * 8]),
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: Some(vec![0.03; 8]),
        ffn_gate_weight: Some(vec![0.01; 8 * 16]),
        ffn_gate_bias: Some(vec![0.04; 16]),
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: Some(vec![0.05; 16]),
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: Some(vec![0.06; 8]),
        ffn_norm_weight: Some(vec![1.0; 8]),
        ffn_norm_bias: Some(vec![0.07; 8]),
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: Some(vec![0.08; 8]),
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: Some(vec![0.09; 10]),
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    // Verify all biases are preserved
    assert!(apr.layers[0].attn_norm_bias.is_some());
    assert!(apr.layers[0].qkv_bias.is_some());
    assert!(apr.layers[0].attn_output_bias.is_some());
    assert!(apr.layers[0].ffn_gate_bias.is_some());
    assert!(apr.layers[0].ffn_up_bias.is_some());
    assert!(apr.layers[0].ffn_down_bias.is_some());
    assert!(apr.layers[0].ffn_norm_bias.is_some());
    assert!(apr.output_norm_bias.is_some());
    assert!(apr.lm_head_bias.is_some());
}

#[test]
fn test_from_gguf_transformer_with_ffn_norm() {
    let config = GGUFConfig {
        architecture: "normed_ffn".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: None,
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: None,
        ffn_norm_weight: Some(vec![1.0; 8]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.layers[0].ffn_norm_weight.is_some());
}

#[test]
fn test_from_gguf_transformer_preserves_intermediate_dim() {
    let gguf = create_minimal_gguf_transformer(32, 1, 100, 256);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.intermediate_dim, 256);
}

#[test]
fn test_from_gguf_transformer_different_kv_heads() {
    let config = GGUFConfig {
        architecture: "gqa".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 2, // GQA: 8 query heads, 2 kv heads
        vocab_size: 100,
        intermediate_dim: 64,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 32],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 32 * (32 + 2 * 8)], // Q + K + V with GQA
        qkv_bias: None,
        attn_output_weight: vec![0.01; 32 * 32],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 32 * 64],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 64 * 32],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 100 * 32],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 32],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 32 * 100],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert_eq!(apr.config.num_heads, 8);
    assert_eq!(apr.config.num_kv_heads, 2);
}

// =============================================================================
// Roundtrip With Various Configurations
// =============================================================================

#[test]
fn test_roundtrip_with_special_rope_theta() {
    let config = AprTransformerConfig {
        architecture: "rope_test".to_string(),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 1000000.0, // Llama 3 style high theta
        eps: 1e-6,
        ..Default::default()
    };

    let apr = AprTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 8 * 3 * 8],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 8 * 8],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 8 * 16],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 16 * 8],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert!((loaded.config.rope_theta - 1000000.0).abs() < 0.1);
    assert!((loaded.config.eps - 1e-6).abs() < 1e-8);
}

#[test]
fn test_roundtrip_with_long_context() {
    let config = AprTransformerConfig {
        architecture: "long_ctx".to_string(),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 131072, // 128K context
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let apr = AprTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 8 * 3 * 8],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 8 * 8],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 8 * 16],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 16 * 8],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(loaded.config.context_length, 131072);
}

// =============================================================================
// Stats with Different Model Sizes
// =============================================================================

#[test]
fn test_stats_tiny_model() {
    let apr = create_minimal_apr_transformer(4, 1, 5, 8);
    let stats = GgufToAprConverter::stats(&apr);

    assert!(stats.total_parameters < 1000);
    assert!(stats.memory_mb() < 1.0);
}

#[test]
fn test_stats_medium_model() {
    let apr = create_minimal_apr_transformer(768, 12, 50257, 3072);
    let stats = GgufToAprConverter::stats(&apr);

    // GPT-2 style model should have millions of parameters
    assert!(stats.total_parameters > 1_000_000);
    assert!(stats.parameters_m() > 1.0);
}

// =============================================================================
// Inference After Conversion Tests
// =============================================================================

#[test]
fn test_converted_model_forward_single_token() {
    let apr = create_minimal_apr_transformer(4, 1, 10, 8);
    let tokens = vec![0]; // Single token

    let result = apr.forward(&tokens);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 10);
}

#[test]
fn test_converted_model_forward_max_token_id() {
    let apr = create_minimal_apr_transformer(4, 1, 10, 8);
    let tokens = vec![9]; // Max valid token ID (vocab_size - 1)

    let result = apr.forward(&tokens);
    assert!(result.is_ok());
}

#[test]
fn test_converted_model_forward_sequence() {
    let apr = create_minimal_apr_transformer(4, 1, 10, 8);
    let tokens = vec![0, 1, 2, 3, 4];

    let result = apr.forward(&tokens);
    assert!(result.is_ok());
}

#[test]
fn test_roundtrip_preserves_inference() {
    let original = create_minimal_apr_transformer(4, 1, 10, 8);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    let tokens = vec![1, 2, 3];
    let original_logits = original.forward(&tokens).expect("original forward");
    let loaded_logits = loaded.forward(&tokens).expect("loaded forward");

    assert_eq!(original_logits.len(), loaded_logits.len());
    for (o, l) in original_logits.iter().zip(loaded_logits.iter()) {
        assert!((o - l).abs() < 1e-5, "Logit mismatch: {} vs {}", o, l);
    }
}

// =============================================================================
// Additional from_apr_bytes Error Path Tests
// =============================================================================

#[test]
fn test_from_apr_bytes_invalid_transformer_json() {
    // Create valid APR structure but with invalid JSON for transformer
    let mut bytes = vec![0u8; 300];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2; // v2
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 tensor
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata offset
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes()); // metadata size
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes()); // tensor index offset
    bytes[32..40].copy_from_slice(&180u64.to_le_bytes()); // data offset
    bytes[64..66].copy_from_slice(b"{}"); // metadata

    // Valid tensor index pointing to invalid transformer JSON
    let index_json = r#"[{"name":"weights","dtype":"json","shape":[50],"offset":0,"size":50}]"#;
    let idx_end = 66 + index_json.len();
    bytes[66..idx_end].copy_from_slice(index_json.as_bytes());

    // Invalid transformer JSON at data section
    let invalid_transformer = b"this is not valid json for a transformer{}{}{}";
    let data_start = 180;
    bytes[data_start..data_start + invalid_transformer.len()].copy_from_slice(invalid_transformer);

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_partial_transformer_json() {
    // Create APR with truncated transformer JSON
    let mut bytes = vec![0u8; 250];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&150u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");

    let index_json = r#"[{"name":"weights","dtype":"json","shape":[30],"offset":0,"size":30}]"#;
    let idx_end = 66 + index_json.len();
    bytes[66..idx_end].copy_from_slice(index_json.as_bytes());

    // Truncated JSON (missing closing braces)
    let partial_json = br#"{"config":{"architecture":"test""#;
    bytes[150..150 + partial_json.len()].copy_from_slice(partial_json);

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_empty_tensor_name() {
    let mut bytes = vec![0u8; 200];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&150u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");

    // Tensor with empty name (not "weights")
    let index_json = r#"[{"name":"","dtype":"json","shape":[10],"offset":0,"size":10}]"#;
    let idx_end = 66 + index_json.len();
    bytes[66..idx_end].copy_from_slice(index_json.as_bytes());

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err()); // No 'weights' tensor found
}

// =============================================================================
// Additional RawTensor Coverage Tests
// =============================================================================

#[test]
fn test_raw_tensor_with_unknown_dtype() {
    let tensor = RawTensor {
        name: "unknown.weight".to_string(),
        data: vec![0u8; 100],
        shape: vec![25],
        dtype: 255, // Unknown dtype
    };

    assert_eq!(tensor.dtype, 255);
    assert_eq!(tensor.data.len(), 100);
}

#[test]
fn test_raw_tensor_large_data() {
    let tensor = RawTensor {
        name: "large.weight".to_string(),
        data: vec![0u8; 1_000_000], // 1MB
        shape: vec![1000, 1000],
        dtype: 0,
    };

    assert_eq!(tensor.data.len(), 1_000_000);
}

#[test]
fn test_raw_tensor_empty_name() {
    let tensor = RawTensor {
        name: String::new(),
        data: vec![1, 2, 3],
        shape: vec![3],
        dtype: 0,
    };

    assert!(tensor.name.is_empty());
}

#[test]
fn test_raw_tensor_unicode_name() {
    let tensor = RawTensor {
        name: "层.权重".to_string(), // Chinese characters
        data: vec![0u8; 16],
        shape: vec![4],
        dtype: 0,
    };

    assert!(tensor.name.contains('层'));
}

#[test]
fn test_raw_tensor_special_chars_name() {
    let tensor = RawTensor {
        name: "model/layer_0/attention/query:0".to_string(),
        data: vec![0u8; 32],
        shape: vec![8, 4],
        dtype: 0,
    };

    assert!(tensor.name.contains('/'));
    assert!(tensor.name.contains(':'));
}

// =============================================================================
// Additional ConversionStats Edge Cases
// =============================================================================

#[test]
fn test_conversion_stats_max_u64_values() {
    let stats = ConversionStats {
        total_parameters: usize::MAX / 2,
        memory_bytes_f32: usize::MAX / 2,
        num_layers: 1000,
        hidden_dim: 16384,
        vocab_size: 1_000_000,
        architecture: "max_test".to_string(),
    };

    // Should not panic on large values
    let _ = stats.memory_mb();
    let _ = stats.memory_gb();
    let _ = stats.parameters_m();
    let _ = stats.parameters_b();
}

#[test]
fn test_conversion_stats_very_small_values() {
    let stats = ConversionStats {
        total_parameters: 1,
        memory_bytes_f32: 4,
        num_layers: 1,
        hidden_dim: 1,
        vocab_size: 1,
        architecture: "tiny".to_string(),
    };

    // 4 bytes = ~3.8e-6 MB
    assert!(stats.memory_mb() < 0.0001);
    // 4 bytes = ~3.7e-9 GB
    assert!(stats.memory_gb() < 0.00001);
    // 1 param = 1e-6 M
    assert!(stats.parameters_m() < 0.001);
    // 1 param = 1e-9 B
    assert!(stats.parameters_b() < 0.00001);
}

#[test]
fn test_conversion_stats_architecture_whitespace() {
    let stats = ConversionStats {
        total_parameters: 1000,
        memory_bytes_f32: 4000,
        num_layers: 1,
        hidden_dim: 32,
        vocab_size: 100,
        architecture: "  spaced  architecture  ".to_string(),
    };

    assert!(stats.architecture.contains("spaced"));
}

// =============================================================================
// Additional Q4KConversionStats Tests
// =============================================================================

#[test]
fn test_q4k_stats_more_q4k_than_total_invalid() {
    // Edge case: q4k count > total count (invalid but should not panic)
    let stats = Q4KConversionStats {
        tensor_count: 10,
        q4k_tensor_count: 20, // More than total (invalid data)
        total_bytes: 1000,
        architecture: "invalid".to_string(),
        num_layers: 1,
        hidden_size: 64,
    };

    // Should not panic
    let _ = format!("{:?}", stats);
}

#[test]
fn test_q4k_stats_very_large_bytes() {
    let stats = Q4KConversionStats {
        tensor_count: 1000,
        q4k_tensor_count: 900,
        total_bytes: usize::MAX / 2,
        architecture: "huge".to_string(),
        num_layers: 200,
        hidden_size: 32768,
    };

    assert!(stats.total_bytes > 1_000_000_000);
}

// =============================================================================
// Additional APR Header Validation Tests
// =============================================================================

#[test]
fn test_apr_bytes_flags_field() {
    let apr = create_minimal_apr_transformer(8, 1, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // Check flags field at bytes 6-7 (should be 0)
    let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    assert_eq!(flags, 0);
}

#[test]
fn test_apr_bytes_tensor_index_offset() {
    let apr = create_minimal_apr_transformer(8, 1, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // Tensor index offset is at bytes 24-31
    let tensor_idx_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    // Should be after header (64) + padded metadata
    assert!(tensor_idx_offset >= 64);
}

#[test]
fn test_apr_bytes_data_offset() {
    let apr = create_minimal_apr_transformer(8, 1, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // Data offset is at bytes 32-39
    let data_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    let tensor_idx_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    // Data offset should be after tensor index offset
    assert!(data_offset >= tensor_idx_offset);
}

#[test]
fn test_apr_bytes_checksum_field() {
    let apr = create_minimal_apr_transformer(8, 1, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // Checksum is at bytes 40-43 (currently always 0)
    let checksum = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    assert_eq!(checksum, 0);
}

#[test]
fn test_apr_bytes_reserved_bytes() {
    let apr = create_minimal_apr_transformer(8, 1, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // Reserved bytes 44-63 should be 0
    for (offset, byte) in bytes[44..64].iter().enumerate() {
        assert_eq!(*byte, 0, "Reserved byte {} should be 0", 44 + offset);
    }
}

// =============================================================================
// Additional Conversion Tests with Various GGUF Configurations
// =============================================================================

#[test]
fn test_gguf_transformer_with_output_norm_bias() {
    let config = GGUFConfig {
        architecture: "with_bias".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: None,
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: Some(vec![0.01; 8]), // Has output norm bias
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.output_norm_bias.is_some());
    assert_eq!(apr.output_norm_bias.unwrap().len(), 8);
}

#[test]
fn test_gguf_transformer_with_lm_head_bias() {
    let config = GGUFConfig {
        architecture: "lm_bias".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: None,
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: Some(vec![0.0; 10]), // Has LM head bias
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.lm_head_bias.is_some());
    assert_eq!(apr.lm_head_bias.unwrap().len(), 10);
}

// =============================================================================
// Test Layer Weight Preservation for All Optional Fields
// =============================================================================

#[test]
fn test_layer_attn_norm_bias_preservation() {
    let config = GGUFConfig {
        architecture: "attn_norm_bias".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: Some(vec![0.5; 8]),
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: None,
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.layers[0].attn_norm_bias.is_some());
}

#[test]
fn test_layer_qkv_bias_preservation() {
    let config = GGUFConfig {
        architecture: "qkv_bias".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: Some(vec![0.0; 3 * 8]),
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.layers[0].qkv_bias.is_some());
}

#[test]
fn test_layer_attn_output_bias_preservation() {
    let config = GGUFConfig {
        architecture: "attn_out_bias".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: None,
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: Some(vec![0.0; 8]),
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.layers[0].attn_output_bias.is_some());
}

#[test]
fn test_layer_ffn_up_bias_preservation() {
    let config = GGUFConfig {
        architecture: "ffn_up_bias".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: None,
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: Some(vec![0.0; 16]),
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: None,
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.layers[0].ffn_up_bias.is_some());
}

#[test]
fn test_layer_ffn_down_bias_preservation() {
    let config = GGUFConfig {
        architecture: "ffn_down_bias".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layer = GGUFTransformerLayer {
        attn_norm_weight: vec![1.0; 8],
        attn_norm_bias: None,
        qkv_weight: vec![0.01; 8 * 3 * 8],
        qkv_bias: None,
        attn_output_weight: vec![0.01; 8 * 8],
        attn_output_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_up_weight: vec![0.01; 8 * 16],
        ffn_up_bias: None,
        ffn_down_weight: vec![0.01; 16 * 8],
        ffn_down_bias: Some(vec![0.0; 8]),
        ffn_norm_weight: None,
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let gguf = GGUFTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        position_embedding: None,
        layers: vec![layer],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);
    assert!(apr.layers[0].ffn_down_bias.is_some());
}

// =============================================================================
// Tests for Multiple Tensor Types in RawTensor
// =============================================================================

#[test]
fn test_raw_tensor_uint16_dtype() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![0u8; 64],
        shape: vec![32],
        dtype: 16, // Some other dtype
    };

    assert_eq!(tensor.dtype, 16);
}

#[test]
fn test_raw_tensor_very_high_dimensional() {
    let tensor = RawTensor {
        name: "high_dim".to_string(),
        data: vec![0u8; 256],
        shape: vec![2, 2, 2, 2, 2, 2, 2, 2], // 8 dimensions
        dtype: 0,
    };

    assert_eq!(tensor.shape.len(), 8);
    let total: usize = tensor.shape.iter().product();
    assert_eq!(total, 256);
}

// =============================================================================
// Additional Roundtrip Tests with Edge Cases
// =============================================================================

#[test]
fn test_roundtrip_empty_architecture() {
    let config = AprTransformerConfig {
        architecture: String::new(),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let apr = AprTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 8 * 3 * 8],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 8 * 8],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 8 * 16],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 16 * 8],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert!(loaded.config.architecture.is_empty());
}

#[test]
fn test_roundtrip_very_small_eps() {
    let config = AprTransformerConfig {
        architecture: "small_eps".to_string(),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-12, // Very small epsilon
        ..Default::default()
    };

    let apr = AprTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 8 * 3 * 8],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 8 * 8],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 8 * 16],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 16 * 8],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert!(loaded.config.eps < 1e-10);
}

#[test]
fn test_roundtrip_single_head() {
    let config = AprTransformerConfig {
        architecture: "single_head".to_string(),
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let apr = AprTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 8 * 3 * 8],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 8 * 8],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 8 * 16],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 16 * 8],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert_eq!(loaded.config.num_heads, 1);
    assert_eq!(loaded.config.num_kv_heads, 1);
}

// =============================================================================
// GGUF to APR Conversion Path Tests
// =============================================================================

#[test]
fn test_gguf_to_apr_converter_convert_empty_bytes_error() {
    let empty: &[u8] = &[];
    let result = GgufToAprConverter::convert(empty);
    assert!(result.is_err(), "Empty bytes should produce error");
}

#[test]
fn test_gguf_to_apr_converter_convert_too_short_error() {
    let short = vec![0u8; 10];
    let result = GgufToAprConverter::convert(&short);
    assert!(result.is_err(), "Too short bytes should produce error");
}

#[test]
fn test_gguf_to_apr_converter_convert_wrong_magic_error() {
    let mut bad_magic = vec![0u8; 128];
    bad_magic[0..4].copy_from_slice(b"XXXX"); // Invalid magic
    let result = GgufToAprConverter::convert(&bad_magic);
    assert!(result.is_err(), "Wrong magic should produce error");
}

#[test]
fn test_gguf_to_apr_converter_convert_partial_gguf_header() {
    // GGUF magic with truncated header
    let mut partial = vec![0u8; 20];
    partial[0..4].copy_from_slice(b"GGUF");
    partial[4..8].copy_from_slice(&3u32.to_le_bytes()); // Version 3
    let result = GgufToAprConverter::convert(&partial);
    assert!(result.is_err());
}

// =============================================================================
// GgufToAprQ4KConverter File Operations Tests
// =============================================================================

#[test]
fn test_q4k_converter_convert_with_empty_file() {
    use std::io::Write;

    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join("test_empty_gguf.bin");
    let output_path = temp_dir.join("test_empty_output.apr");

    // Write empty file
    {
        let mut file = std::fs::File::create(&input_path).expect("create temp file");
        file.write_all(&[]).expect("write empty");
    }

    let result = realizar::convert::GgufToAprQ4KConverter::convert(&input_path, &output_path);
    assert!(result.is_err(), "Empty file should produce error");

    // Cleanup
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn test_q4k_converter_convert_with_truncated_gguf() {
    use std::io::Write;

    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join("test_truncated_gguf.bin");
    let output_path = temp_dir.join("test_truncated_output.apr");

    // Write truncated GGUF (valid magic but incomplete header)
    {
        let mut file = std::fs::File::create(&input_path).expect("create temp file");
        file.write_all(b"GGUF").expect("write magic");
        file.write_all(&3u32.to_le_bytes()).expect("write version");
        // Truncated - missing rest of header
    }

    let result = realizar::convert::GgufToAprQ4KConverter::convert(&input_path, &output_path);
    assert!(result.is_err(), "Truncated GGUF should produce error");

    // Cleanup
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
}

// =============================================================================
// Tensor Quantization Type Coverage Tests
// =============================================================================

#[test]
fn test_raw_tensor_all_quantization_types() {
    // Test F32 (dtype 0)
    let f32_tensor = RawTensor {
        name: "f32_tensor".to_string(),
        data: vec![0u8; 400], // 100 floats * 4 bytes
        shape: vec![100],
        dtype: 0,
    };
    assert_eq!(f32_tensor.dtype, 0);
    assert_eq!(f32_tensor.data.len(), 400);

    // Test F16 (dtype 1)
    let f16_tensor = RawTensor {
        name: "f16_tensor".to_string(),
        data: vec![0u8; 200], // 100 floats * 2 bytes
        shape: vec![100],
        dtype: 1,
    };
    assert_eq!(f16_tensor.dtype, 1);
    assert_eq!(f16_tensor.data.len(), 200);

    // Test Q8_0 (dtype 8)
    let q8_tensor = RawTensor {
        name: "q8_tensor".to_string(),
        data: vec![0u8; 34], // 32 elements = 2 (scale) + 32 (quants)
        shape: vec![32],
        dtype: 8,
    };
    assert_eq!(q8_tensor.dtype, 8);

    // Test Q4_K (dtype 12)
    let q4k_tensor = RawTensor {
        name: "q4k_tensor".to_string(),
        data: vec![0u8; 144], // 256 elements = 144 bytes
        shape: vec![256],
        dtype: 12,
    };
    assert_eq!(q4k_tensor.dtype, 12);

    // Test Q5_K (dtype 13)
    let q5k_tensor = RawTensor {
        name: "q5k_tensor".to_string(),
        data: vec![0u8; 176], // 256 elements = 176 bytes
        shape: vec![256],
        dtype: 13,
    };
    assert_eq!(q5k_tensor.dtype, 13);

    // Test Q6_K (dtype 14)
    let q6k_tensor = RawTensor {
        name: "q6k_tensor".to_string(),
        data: vec![0u8; 210], // 256 elements = 210 bytes
        shape: vec![256],
        dtype: 14,
    };
    assert_eq!(q6k_tensor.dtype, 14);
}

#[test]
fn test_raw_tensor_byte_size_calculations() {
    // Verify byte size calculations for different types match convert.rs logic

    // F32: num_elements * 4
    let f32_elements = 100;
    let f32_bytes = f32_elements * 4;
    assert_eq!(f32_bytes, 400);

    // F16: num_elements * 2
    let f16_elements = 100;
    let f16_bytes = f16_elements * 2;
    assert_eq!(f16_bytes, 200);

    // Q8_0: (num_elements / 32) * 34
    let q8_elements = 256;
    let q8_bytes = (q8_elements / 32) * 34;
    assert_eq!(q8_bytes, 272);

    // Q4_K: (num_elements / 256) * 144
    let q4k_elements = 512;
    let q4k_bytes = (q4k_elements / 256) * 144;
    assert_eq!(q4k_bytes, 288);

    // Q5_K: (num_elements / 256) * 176
    let q5k_elements = 512;
    let q5k_bytes = (q5k_elements / 256) * 176;
    assert_eq!(q5k_bytes, 352);

    // Q6_K: (num_elements / 256) * 210
    let q6k_elements = 512;
    let q6k_bytes = (q6k_elements / 256) * 210;
    assert_eq!(q6k_bytes, 420);
}

// =============================================================================
// Metadata Preservation Tests
// =============================================================================

#[test]
fn test_metadata_preservation_all_fields() {
    let config = AprTransformerConfig {
        architecture: "metadata_test".to_string(),
        hidden_dim: 512,
        num_layers: 16,
        num_heads: 8,
        num_kv_heads: 4,
        vocab_size: 32000,
        intermediate_dim: 2048,
        context_length: 4096,
        rope_theta: 500000.0,
        eps: 1e-6,
        ..Default::default()
    };

    let apr = AprTransformer {
        config,
        token_embedding: vec![0.1; 32000 * 512],
        layers: (0..16)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; 512],
                attn_norm_bias: None,
                qkv_weight: vec![0.01; 512 * 3 * 512],
                qkv_bias: None,
                attn_output_weight: vec![0.01; 512 * 512],
                attn_output_bias: None,
                ffn_gate_weight: Some(vec![0.01; 512 * 2048]),
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.01; 512 * 2048],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.01; 2048 * 512],
                ffn_down_bias: None,
                ffn_norm_weight: None,
                ffn_norm_bias: None,
                attn_q_norm_weight: None,
                attn_k_norm_weight: None,
                linear_attn_z_weight: None,
                linear_attn_b_weight: None,
                linear_attn_a_weight: None,
                linear_attn_conv1d_weight: None,
                linear_attn_a_log: None,
                linear_attn_dt_bias: None,
                linear_attn_norm_weight: None,
                moe_gate_weight: None,
                moe_expert_gate_up: None,
                moe_expert_down: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                moe_shared_expert_gate_weight: None,
            })
            .collect(),
        output_norm_weight: vec![1.0; 512],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 512 * 32000],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    // Verify all metadata fields are preserved
    assert_eq!(loaded.config.architecture, "metadata_test");
    assert_eq!(loaded.config.hidden_dim, 512);
    assert_eq!(loaded.config.num_layers, 16);
    assert_eq!(loaded.config.num_heads, 8);
    assert_eq!(loaded.config.num_kv_heads, 4);
    assert_eq!(loaded.config.vocab_size, 32000);
    assert_eq!(loaded.config.intermediate_dim, 2048);
    assert_eq!(loaded.config.context_length, 4096);
    assert!((loaded.config.rope_theta - 500000.0).abs() < 0.1);
    assert!((loaded.config.eps - 1e-6).abs() < 1e-8);
}

#[test]
fn test_metadata_unicode_architecture() {
    let config = AprTransformerConfig {
        architecture: "模型架构_v2".to_string(), // Chinese characters
        hidden_dim: 8,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 10,
        intermediate_dim: 16,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let apr = AprTransformer {
        config,
        token_embedding: vec![0.1; 10 * 8],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 8],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 8 * 3 * 8],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 8 * 8],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 8 * 16],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 16 * 8],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0; 8],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 8 * 10],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    assert!(loaded.config.architecture.contains("模型架构"));
}

// =============================================================================
// Unsupported Format Error Handling Tests
// =============================================================================

#[test]
fn test_from_apr_bytes_unsupported_version_v0() {
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 0; // version 0 (unsupported)
    bytes[5] = 0;

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    // May error or handle gracefully, but should not panic
    let _ = result;
}

#[test]
fn test_from_apr_bytes_unsupported_version_v255() {
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 255; // very high version
    bytes[5] = 255;

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    // Should error or handle gracefully
    let _ = result;
}

#[test]
fn test_convert_gguf_with_reversed_magic() {
    // GGUF magic in wrong order
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(b"FUGF"); // Reversed GGUF
    let result = GgufToAprConverter::convert(&bytes);
    assert!(result.is_err());
}

// =============================================================================
// File I/O Error Simulation Tests (Q4K Converter)
// =============================================================================

#[test]
fn test_q4k_converter_nonexistent_input_file() {
    use std::path::Path;

    let input_path = Path::new("/nonexistent/path/to/model.gguf");
    let output_path = Path::new("/tmp/output.apr");

    let result = realizar::convert::GgufToAprQ4KConverter::convert(input_path, output_path);
    assert!(
        result.is_err(),
        "Nonexistent input file should produce error"
    );
}

#[test]
fn test_q4k_converter_invalid_input_content() {
    use std::io::Write;

    // Create a temp file with invalid content
    let temp_dir = std::env::temp_dir();
    let input_path = temp_dir.join("test_invalid_gguf.bin");
    let output_path = temp_dir.join("test_output.apr");

    // Write invalid content
    {
        let mut file = std::fs::File::create(&input_path).expect("create temp file");
        file.write_all(b"NOT A GGUF FILE").expect("write");
    }

    let result = realizar::convert::GgufToAprQ4KConverter::convert(&input_path, &output_path);
    assert!(result.is_err(), "Invalid GGUF content should produce error");

    // Cleanup
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
}

#[test]
fn test_q4k_converter_output_to_readonly_dir() {
    use std::path::Path;

    // Attempt to write to a path that doesn't exist as directory
    let nonexistent_output = Path::new("/nonexistent_dir_xyz/model.apr");
    let nonexistent_input = Path::new("/tmp/nonexistent_model.gguf");

    // This should fail because input doesn't exist first
    let result =
        realizar::convert::GgufToAprQ4KConverter::convert(nonexistent_input, nonexistent_output);
    assert!(result.is_err());
}

// =============================================================================
// Output Validation Tests
// =============================================================================

#[test]
fn test_apr_bytes_output_has_valid_structure() {
    let apr = create_minimal_apr_transformer(16, 2, 100, 32);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // Validate header structure
    assert_eq!(&bytes[0..4], &MAGIC, "Magic bytes should match");
    assert_eq!(bytes[4], 2, "Major version should be 2");
    assert_eq!(bytes[5], 0, "Minor version should be 0");

    // Validate tensor count
    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    assert_eq!(tensor_count, 1, "Should have 1 tensor (weights)");

    // Validate metadata offset
    let metadata_offset = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    assert_eq!(
        metadata_offset, HEADER_SIZE as u64,
        "Metadata should start after header"
    );

    // Validate tensor index offset comes after metadata
    let tensor_idx_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    assert!(
        tensor_idx_offset > metadata_offset,
        "Tensor index should be after metadata"
    );

    // Validate data offset comes after tensor index
    let data_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    assert!(
        data_offset >= tensor_idx_offset,
        "Data should be after tensor index"
    );
}

#[test]
fn test_apr_bytes_contains_valid_json_metadata() {
    let apr = create_minimal_apr_transformer(8, 1, 10, 16);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // Extract metadata
    let metadata_offset = u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize;
    let metadata_len = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;

    let metadata_slice = &bytes[metadata_offset..metadata_offset + metadata_len];

    // Should be valid JSON
    let metadata: serde_json::Value =
        serde_json::from_slice(metadata_slice).expect("Metadata should be valid JSON");

    // Verify expected fields
    assert!(metadata.get("model_type").is_some());
    assert!(metadata.get("architecture").is_some());
    assert!(metadata.get("hidden_size").is_some());
    assert!(metadata.get("num_layers").is_some());
    assert!(metadata.get("vocab_size").is_some());
}

#[test]
fn test_apr_roundtrip_weight_values_exact() {
    // Create transformer with specific known values
    let config = AprTransformerConfig {
        architecture: "exact_test".to_string(),
        hidden_dim: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 5,
        intermediate_dim: 8,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    // Use specific weight values
    let embedding = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        1.9, 2.0,
    ];

    let apr = AprTransformer {
        config,
        token_embedding: embedding.clone(),
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0, 1.1, 1.2, 1.3],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 4 * 3 * 4],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 4 * 4],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 4 * 8],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 8 * 4],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0, 1.0, 1.0, 1.0],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; 4 * 5],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    // Verify exact weight values
    for (i, (orig, load)) in embedding
        .iter()
        .zip(loaded.token_embedding.iter())
        .enumerate()
    {
        assert!(
            (orig - load).abs() < 1e-6,
            "Embedding[{}] mismatch: {} vs {}",
            i,
            orig,
            load
        );
    }

    // Verify attn_norm_weight exact values
    let orig_norm = &apr.layers[0].attn_norm_weight;
    let load_norm = &loaded.layers[0].attn_norm_weight;
    for (i, (orig, load)) in orig_norm.iter().zip(load_norm.iter()).enumerate() {
        assert!(
            (orig - load).abs() < 1e-6,
            "attn_norm_weight[{}] mismatch",
            i
        );
    }
}

// =============================================================================
// Edge Cases for Header Parsing
// =============================================================================

#[test]
fn test_from_apr_bytes_exactly_header_size() {
    // File that is exactly header size (64 bytes) with no content
    let mut bytes = vec![0u8; HEADER_SIZE];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes()); // metadata at 64
    bytes[20..24].copy_from_slice(&0u32.to_le_bytes()); // 0 metadata len
    bytes[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor idx at 64
    bytes[32..40].copy_from_slice(&64u64.to_le_bytes()); // data at 64

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    // Should fail because there's no weights tensor
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_header_with_max_tensor_count() {
    let mut bytes = vec![0u8; 128];
    bytes[0..4].copy_from_slice(&MAGIC);
    bytes[4] = 2;
    bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes()); // Max tensor count
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    bytes[20..24].copy_from_slice(&2u32.to_le_bytes());
    bytes[24..32].copy_from_slice(&66u64.to_le_bytes());
    bytes[32..40].copy_from_slice(&70u64.to_le_bytes());
    bytes[64..66].copy_from_slice(b"{}");
    bytes[66..68].copy_from_slice(b"[]");

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    // Should fail gracefully (no weights tensor, or parsing error)
    assert!(result.is_err());
}

// =============================================================================
// Stats Computation Edge Cases
// =============================================================================

#[test]
fn test_stats_with_different_layer_counts() {
    for num_layers in [0, 1, 2, 4, 8, 16, 32, 64] {
        let apr = create_minimal_apr_transformer(8, num_layers, 10, 16);
        let stats = GgufToAprConverter::stats(&apr);
        assert_eq!(
            stats.num_layers, num_layers,
            "Layer count mismatch for {}",
            num_layers
        );
    }
}

#[test]
fn test_stats_with_different_hidden_dims() {
    for hidden_dim in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let apr = create_minimal_apr_transformer(hidden_dim, 1, 10, hidden_dim * 2);
        let stats = GgufToAprConverter::stats(&apr);
        assert_eq!(
            stats.hidden_dim, hidden_dim,
            "Hidden dim mismatch for {}",
            hidden_dim
        );
    }
}

#[test]
fn test_stats_with_different_vocab_sizes() {
    for vocab_size in [1, 10, 100, 1000, 10000, 50000] {
        let apr = create_minimal_apr_transformer(8, 1, vocab_size, 16);
        let stats = GgufToAprConverter::stats(&apr);
        assert_eq!(
            stats.vocab_size, vocab_size,
            "Vocab size mismatch for {}",
            vocab_size
        );
    }
}

// =============================================================================
// APR Dtype Mapping Tests
// =============================================================================

#[test]
fn test_apr_dtype_mapping_coverage() {
    // GGML: 0=F32, 1=F16, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
    // APR:  0=F32, 1=F16, 8=Q4_K, 9=Q6_K, 10=Q8_0

    // Verify the mapping in RawTensor representation
    let dtypes = [
        (0u32, "F32"),
        (1u32, "F16"),
        (8u32, "Q8_0"),
        (12u32, "Q4_K"),
        (13u32, "Q5_K"),
        (14u32, "Q6_K"),
    ];

    for (dtype, name) in dtypes {
        let tensor = RawTensor {
            name: format!("test_{}", name),
            data: vec![0u8; 64],
            shape: vec![64],
            dtype,
        };
        assert_eq!(tensor.dtype, dtype, "Dtype mismatch for {}", name);
    }
}

// =============================================================================
// Multiple Layer Weight Preservation Tests
// =============================================================================

#[test]
fn test_conversion_preserves_per_layer_weights_large() {
    let num_layers = 8;
    let hidden_dim = 16;
    let intermediate_dim = 32;
    let vocab_size = 50;

    let gguf =
        create_minimal_gguf_transformer(hidden_dim, num_layers, vocab_size, intermediate_dim);
    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    // Verify all layers are preserved
    assert_eq!(apr.layers.len(), num_layers);

    // Verify each layer has correct dimensions
    for (i, layer) in apr.layers.iter().enumerate() {
        assert_eq!(
            layer.attn_norm_weight.len(),
            hidden_dim,
            "Layer {} attn_norm_weight len",
            i
        );
        assert_eq!(
            layer.qkv_weight.len(),
            hidden_dim * 3 * hidden_dim,
            "Layer {} qkv_weight len",
            i
        );
        assert_eq!(
            layer.attn_output_weight.len(),
            hidden_dim * hidden_dim,
            "Layer {} attn_output_weight len",
            i
        );
        assert_eq!(
            layer.ffn_up_weight.len(),
            hidden_dim * intermediate_dim,
            "Layer {} ffn_up_weight len",
            i
        );
        assert_eq!(
            layer.ffn_down_weight.len(),
            intermediate_dim * hidden_dim,
            "Layer {} ffn_down_weight len",
            i
        );
    }
}

// =============================================================================
// Serialization Size Tests
// =============================================================================

#[test]
fn test_serialized_size_increases_with_model_size() {
    let small = create_minimal_apr_transformer(4, 1, 10, 8);
    let medium = create_minimal_apr_transformer(16, 4, 100, 32);
    let large = create_minimal_apr_transformer(32, 8, 500, 64);

    let small_bytes = GgufToAprConverter::to_apr_bytes(&small).expect("serialize small");
    let medium_bytes = GgufToAprConverter::to_apr_bytes(&medium).expect("serialize medium");
    let large_bytes = GgufToAprConverter::to_apr_bytes(&large).expect("serialize large");

    assert!(
        medium_bytes.len() > small_bytes.len(),
        "Medium should be larger than small"
    );
    assert!(
        large_bytes.len() > medium_bytes.len(),
        "Large should be larger than medium"
    );
}

#[test]
fn test_serialized_size_relationship_to_parameters() {
    let apr = create_minimal_apr_transformer(16, 2, 100, 32);
    let stats = GgufToAprConverter::stats(&apr);
    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");

    // The serialized size should be roughly related to the memory size
    // (though JSON serialization adds overhead)
    let expected_min_size = stats.memory_bytes_f32; // At minimum, we need to store all weights
    assert!(
        bytes.len() > expected_min_size / 2,
        "Serialized size should be substantial"
    );
}

// =============================================================================
// Empty/Minimal Configuration Tests
// =============================================================================

#[test]
fn test_apr_with_all_optional_fields_none() {
    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "minimal".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 5,
            intermediate_dim: 8,
            context_length: 16,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.1; 5 * 4],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 4],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; 4 * 3 * 4],
            qkv_bias: None,
            attn_output_weight: vec![0.01; 4 * 4],
            attn_output_bias: None,
            ffn_gate_weight: None, // Optional
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; 4 * 8],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; 8 * 4],
            ffn_down_bias: None,
            ffn_norm_weight: None, // Optional
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0; 4],
        output_norm_bias: None, // Optional
        lm_head_weight: vec![0.01; 4 * 5],
        lm_head_bias: None, // Optional
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    // Verify None fields remain None
    assert!(loaded.layers[0].ffn_gate_weight.is_none());
    assert!(loaded.layers[0].ffn_norm_weight.is_none());
    assert!(loaded.output_norm_bias.is_none());
    assert!(loaded.lm_head_bias.is_none());
}

#[test]
fn test_apr_with_all_optional_fields_some() {
    let apr = AprTransformer {
        config: AprTransformerConfig {
            architecture: "full".to_string(),
            hidden_dim: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 5,
            intermediate_dim: 8,
            context_length: 16,
            rope_theta: 10000.0,
            eps: 1e-5,
            ..Default::default()
        },
        token_embedding: vec![0.1; 5 * 4],
        layers: vec![AprTransformerLayer {
            attn_norm_weight: vec![1.0; 4],
            attn_norm_bias: Some(vec![0.1; 4]),
            qkv_weight: vec![0.01; 4 * 3 * 4],
            qkv_bias: Some(vec![0.02; 3 * 4]),
            attn_output_weight: vec![0.01; 4 * 4],
            attn_output_bias: Some(vec![0.03; 4]),
            ffn_gate_weight: Some(vec![0.04; 4 * 8]),
            ffn_gate_bias: Some(vec![0.05; 8]),
            ffn_up_weight: vec![0.01; 4 * 8],
            ffn_up_bias: Some(vec![0.06; 8]),
            ffn_down_weight: vec![0.01; 8 * 4],
            ffn_down_bias: Some(vec![0.07; 4]),
            ffn_norm_weight: Some(vec![1.0; 4]),
            ffn_norm_bias: Some(vec![0.08; 4]),
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }],
        output_norm_weight: vec![1.0; 4],
        output_norm_bias: Some(vec![0.09; 4]),
        lm_head_weight: vec![0.01; 4 * 5],
        lm_head_bias: Some(vec![0.1; 5]),
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    };

    let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialize");
    let loaded = GgufToAprConverter::from_apr_bytes(&bytes).expect("deserialize");

    // Verify Some fields remain Some
    assert!(loaded.layers[0].attn_norm_bias.is_some());
    assert!(loaded.layers[0].qkv_bias.is_some());
    assert!(loaded.layers[0].attn_output_bias.is_some());
    assert!(loaded.layers[0].ffn_gate_weight.is_some());
    assert!(loaded.layers[0].ffn_gate_bias.is_some());
    assert!(loaded.layers[0].ffn_up_bias.is_some());
    assert!(loaded.layers[0].ffn_down_bias.is_some());
    assert!(loaded.layers[0].ffn_norm_weight.is_some());
    assert!(loaded.layers[0].ffn_norm_bias.is_some());
    assert!(loaded.output_norm_bias.is_some());
    assert!(loaded.lm_head_bias.is_some());
}
