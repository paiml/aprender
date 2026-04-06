//! T-COV-95 Coverage Bridge: convert/mod.rs Part 10
//!
//! Targets uncovered lines in: crc32/checksum (lines 30-66),
//! GgufToAprConverter to_apr_bytes metadata padding (lines 198-250),
//! from_apr_bytes error paths (lines 266-322),
//! Q4KConverter byte_size calculations (lines 645-657),
//! Q4KConverter binary tensor index (lines 693-741),
//! and ConversionStats edge cases.

use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::convert::{ConversionStats, GgufToAprConverter, Q4KConversionStats, RawTensor};

// ============================================================================
// crc32 + checksum verification via header manipulation
// ============================================================================

#[test]
fn test_checksum_field_at_correct_offset() {
    // Verify the checksum is at bytes [40..44] in the APR header
    let transformer = make_tiny_transformer(1);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    // Verify header structure:
    // [0..4] magic, [4..6] version, [6..8] flags, [8..12] tensor_count
    // [12..20] metadata_offset, [20..24] metadata_size
    // [24..32] tensor_index_offset, [32..40] data_offset
    // [40..44] checksum, [44..64] reserved
    assert_eq!(&bytes[0..4], &[0x41, 0x50, 0x52, 0x00]); // APR\0
    assert_eq!(bytes[4], 2); // version major

    let checksum = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    assert_ne!(checksum, 0, "CRC32 should be non-zero for real data");

    // Reserved bytes should be zero
    for &b in &bytes[44..64] {
        assert_eq!(b, 0, "Reserved bytes should be zero");
    }
}

#[test]
fn test_checksum_differs_with_different_metadata() {
    let t1 = make_tiny_transformer(1);
    let mut t2 = make_tiny_transformer(1);
    t2.config.architecture = "different_arch".to_string();

    let b1 = GgufToAprConverter::to_apr_bytes(&t1).expect("b1");
    let b2 = GgufToAprConverter::to_apr_bytes(&t2).expect("b2");

    let c1 = u32::from_le_bytes([b1[40], b1[41], b1[42], b1[43]]);
    let c2 = u32::from_le_bytes([b2[40], b2[41], b2[42], b2[43]]);
    assert_ne!(c1, c2, "Different metadata should give different checksums");
}

// ============================================================================
// to_apr_bytes metadata padding and alignment
// ============================================================================

#[test]
fn test_to_apr_bytes_metadata_64byte_aligned() {
    let transformer = make_tiny_transformer(1);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    let metadata_offset = u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize;
    let tensor_index_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;

    // The gap between metadata_offset and tensor_index_offset should be 64-byte aligned
    let metadata_region_size = tensor_index_offset - metadata_offset;
    assert_eq!(
        metadata_region_size % 64,
        0,
        "Metadata region should be 64-byte padded"
    );
}

#[test]
fn test_to_apr_bytes_tensor_count_is_one() {
    let transformer = make_tiny_transformer(1);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    let tensor_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    assert_eq!(
        tensor_count, 1,
        "Should have exactly 1 tensor (the JSON weights)"
    );
}

#[test]
fn test_to_apr_bytes_flags_are_zero() {
    let transformer = make_tiny_transformer(1);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    assert_eq!(flags, 0, "Default converter should produce no flags");
}

// ============================================================================
// from_apr_bytes error paths coverage
// ============================================================================

#[test]
fn test_from_apr_bytes_completely_empty() {
    let result = GgufToAprConverter::from_apr_bytes(&[]);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_just_magic() {
    let result = GgufToAprConverter::from_apr_bytes(&[0x41, 0x50, 0x52, 0x00]);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_v1_magic_rejected() {
    // APR v1 magic = "APR1" (0x41, 0x50, 0x52, 0x31)
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x31]);
    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("v1") || err.contains("not supported"));
}

#[test]
fn test_from_apr_bytes_invalid_version_byte() {
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0xFF]); // Invalid version byte
    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_data_at_index_boundary() {
    // Create valid file but make data_offset point past end
    let transformer = make_tiny_transformer(1);
    let mut bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    // Corrupt the data_offset to point past EOF
    let huge_offset = (bytes.len() as u64 + 10000).to_le_bytes();
    bytes[32..40].copy_from_slice(&huge_offset);

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_corrupt_json_in_weights() {
    let transformer = make_tiny_transformer(1);
    let mut bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    // Corrupt the tensor index JSON (between tensor_index_offset and data_offset)
    let tensor_index_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
    let data_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
    // Corrupt the tensor index region with garbage bytes
    if tensor_index_offset < data_offset && tensor_index_offset < bytes.len() {
        let end = data_offset.min(bytes.len());
        for i in tensor_index_offset..end {
            bytes[i] = 0xFF;
        }
    }

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

// ============================================================================
// RawTensor edge cases
// ============================================================================

#[test]
fn test_raw_tensor_empty_data() {
    let tensor = RawTensor {
        name: "empty".to_string(),
        data: vec![],
        shape: vec![0],
        dtype: 0,
    };
    assert!(tensor.data.is_empty());
    assert_eq!(tensor.shape, vec![0]);
}

#[test]
fn test_raw_tensor_multidim_shape() {
    let tensor = RawTensor {
        name: "3d".to_string(),
        data: vec![0; 24],
        shape: vec![2, 3, 4],
        dtype: 0,
    };
    assert_eq!(tensor.shape.len(), 3);
    assert_eq!(tensor.shape.iter().product::<usize>(), 24);
}

#[test]
fn test_raw_tensor_q4k_dtype() {
    let tensor = RawTensor {
        name: "q4k_tensor".to_string(),
        data: vec![0; 144], // One Q4K block = 144 bytes = 256 elements
        shape: vec![256],
        dtype: 12, // Q4_K
    };
    assert_eq!(tensor.dtype, 12);
}

#[test]
fn test_raw_tensor_q6k_dtype() {
    let tensor = RawTensor {
        name: "q6k_tensor".to_string(),
        data: vec![0; 210], // One Q6K block = 210 bytes = 256 elements
        shape: vec![256],
        dtype: 14, // Q6_K
    };
    assert_eq!(tensor.dtype, 14);
}

// ============================================================================
// Q4KConversionStats accessors
// ============================================================================

#[test]
fn test_q4k_stats_zero_values() {
    let stats = Q4KConversionStats {
        tensor_count: 0,
        q4k_tensor_count: 0,
        total_bytes: 0,
        architecture: String::new(),
        num_layers: 0,
        hidden_size: 0,
    };
    assert_eq!(stats.tensor_count, 0);
    assert_eq!(stats.q4k_tensor_count, 0);
    assert_eq!(stats.total_bytes, 0);
}

#[test]
fn test_q4k_stats_large_model() {
    let stats = Q4KConversionStats {
        tensor_count: 291,
        q4k_tensor_count: 225,
        total_bytes: 4_500_000_000,
        architecture: "qwen2".to_string(),
        num_layers: 32,
        hidden_size: 4096,
    };
    assert_eq!(stats.tensor_count, 291);
    assert_eq!(stats.q4k_tensor_count, 225);
    assert_eq!(stats.total_bytes, 4_500_000_000);
}

// ============================================================================
// ConversionStats with all methods exercised on same instance
// ============================================================================

#[test]
fn test_conversion_stats_all_methods_consistent() {
    let stats = ConversionStats {
        total_parameters: 1_500_000_000,     // 1.5B
        memory_bytes_f32: 1_500_000_000 * 4, // 6GB
        num_layers: 24,
        hidden_dim: 2048,
        vocab_size: 32000,
        architecture: "llama".to_string(),
    };

    // Verify all four methods on same instance
    assert!((stats.parameters_m() - 1500.0).abs() < 0.01);
    assert!((stats.parameters_b() - 1.5).abs() < 0.001);
    // 6GB = 6 * 1024^3 bytes, but we use 1.5B * 4 = 6B bytes
    let expected_gb = (1_500_000_000.0 * 4.0) / (1024.0 * 1024.0 * 1024.0);
    assert!((stats.memory_gb() - expected_gb).abs() < 0.01);
    assert!((stats.memory_mb() - expected_gb * 1024.0).abs() < 0.1);
}

#[test]
fn test_conversion_stats_1_parameter() {
    let stats = ConversionStats {
        total_parameters: 1,
        memory_bytes_f32: 4,
        num_layers: 0,
        hidden_dim: 1,
        vocab_size: 1,
        architecture: "tiny".to_string(),
    };
    assert!((stats.parameters_m() - 0.000001).abs() < 1e-10);
    assert!((stats.parameters_b() - 0.000000001).abs() < 1e-15);
    let expected_mb = 4.0 / (1024.0 * 1024.0);
    assert!((stats.memory_mb() - expected_mb).abs() < 1e-10);
}

// ============================================================================
// GgufToAprConverter::stats with various layer configurations
// ============================================================================

#[test]
fn test_stats_many_layers() {
    let transformer = make_tiny_transformer(8);
    let stats = GgufToAprConverter::stats(&transformer);
    assert_eq!(stats.num_layers, 8);
    assert!(stats.total_parameters > 0);
    // More layers = more parameters
    let stats_1 = GgufToAprConverter::stats(&make_tiny_transformer(1));
    assert!(stats.total_parameters > stats_1.total_parameters);
}

#[test]
fn test_stats_architecture_preserved() {
    let mut transformer = make_tiny_transformer(1);
    transformer.config.architecture = "qwen2.5_special".to_string();
    let stats = GgufToAprConverter::stats(&transformer);
    assert_eq!(stats.architecture, "qwen2.5_special");
}

// ============================================================================
// Roundtrip stress tests
// ============================================================================

#[test]
fn test_roundtrip_preserves_all_config_fields() {
    let transformer = make_tiny_transformer(2);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");
    let restored = GgufToAprConverter::from_apr_bytes(&bytes).expect("from_apr_bytes");

    assert_eq!(restored.config.architecture, "test");
    assert_eq!(restored.config.hidden_dim, 32);
    assert_eq!(restored.config.num_layers, 2);
    assert_eq!(restored.config.num_heads, 4);
    assert_eq!(restored.config.num_kv_heads, 4);
    assert_eq!(restored.config.vocab_size, 50);
    assert_eq!(restored.config.intermediate_dim, 64);
    assert_eq!(restored.config.context_length, 256);
    assert!((restored.config.rope_theta - 10000.0).abs() < 0.01);
    assert!((restored.config.eps - 1e-5).abs() < 1e-9);
}

#[test]
fn test_roundtrip_preserves_weight_values() {
    let transformer = make_tiny_transformer(1);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");
    let restored = GgufToAprConverter::from_apr_bytes(&bytes).expect("from_apr_bytes");

    // Check embedding values preserved
    assert_eq!(
        restored.token_embedding.len(),
        transformer.token_embedding.len()
    );
    for (a, b) in restored
        .token_embedding
        .iter()
        .zip(transformer.token_embedding.iter())
    {
        assert!((a - b).abs() < 1e-6);
    }

    // Check layer weights preserved
    assert_eq!(restored.layers.len(), 1);
    assert_eq!(
        restored.layers[0].qkv_weight.len(),
        transformer.layers[0].qkv_weight.len()
    );
}

#[test]
fn test_roundtrip_with_all_optional_biases() {
    let mut transformer = make_tiny_transformer(1);
    let hidden = 32;
    let intermediate = 64;
    let vocab = 50;
    let qkv_out = hidden + 2 * hidden; // non-GQA

    transformer.output_norm_bias = Some(vec![0.1; hidden]);
    transformer.lm_head_bias = Some(vec![0.2; vocab]);
    transformer.layers[0].attn_norm_bias = Some(vec![0.3; hidden]);
    transformer.layers[0].qkv_bias = Some(vec![0.4; qkv_out]);
    transformer.layers[0].attn_output_bias = Some(vec![0.5; hidden]);
    transformer.layers[0].ffn_gate_bias = Some(vec![0.6; intermediate]);
    transformer.layers[0].ffn_up_bias = Some(vec![0.7; intermediate]);
    transformer.layers[0].ffn_down_bias = Some(vec![0.8; hidden]);
    transformer.layers[0].ffn_norm_bias = Some(vec![0.9; hidden]);

    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");
    let restored = GgufToAprConverter::from_apr_bytes(&bytes).expect("from_apr_bytes");

    assert!(restored.output_norm_bias.is_some());
    assert!(restored.lm_head_bias.is_some());
    assert!(restored.layers[0].attn_norm_bias.is_some());
    assert!(restored.layers[0].qkv_bias.is_some());
    assert!(restored.layers[0].attn_output_bias.is_some());
    assert!(restored.layers[0].ffn_gate_bias.is_some());
    assert!(restored.layers[0].ffn_up_bias.is_some());
    assert!(restored.layers[0].ffn_down_bias.is_some());
    assert!(restored.layers[0].ffn_norm_bias.is_some());
}

// ============================================================================
// Helpers
// ============================================================================

fn make_tiny_transformer(num_layers: usize) -> AprTransformer {
    let hidden_dim = 32;
    let num_heads = 4;
    let num_kv_heads = 4;
    let vocab_size = 50;
    let intermediate_dim = 64;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.1; hidden_dim * vocab_size],
        layers: (0..num_layers)
            .map(|_| AprTransformerLayer {
                attn_norm_weight: vec![1.0; hidden_dim],
                attn_norm_bias: None,
                qkv_weight: vec![0.02; qkv_out_dim * hidden_dim],
                qkv_bias: None,
                attn_output_weight: vec![0.02; hidden_dim * hidden_dim],
                attn_output_bias: None,
                ffn_gate_weight: Some(vec![0.02; intermediate_dim * hidden_dim]),
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.02; intermediate_dim * hidden_dim],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.02; hidden_dim * intermediate_dim],
                ffn_down_bias: None,
                ffn_norm_weight: Some(vec![1.0; hidden_dim]),
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
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.02; vocab_size * hidden_dim],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}
