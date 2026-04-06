//! T-COV-95 Coverage Bridge: convert/mod.rs
//!
//! Targets: ConversionStats methods, GgufToAprConverter::to_apr_bytes,
//! GgufToAprConverter::from_apr_bytes error paths.

use crate::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use crate::convert::{ConversionStats, GgufToAprConverter};
use crate::gguf::{GGUFConfig, GGUFTransformer, GGUFTransformerLayer};

// ============================================================================
// ConversionStats methods coverage
// ============================================================================

#[test]
fn test_conversion_stats_memory_mb() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 1024 * 1024 * 100, // 100 MB
        num_layers: 12,
        hidden_dim: 768,
        vocab_size: 32000,
        architecture: "llama".to_string(),
    };
    let mb = stats.memory_mb();
    assert!((mb - 100.0).abs() < 0.01);
}

#[test]
fn test_conversion_stats_memory_gb() {
    let stats = ConversionStats {
        total_parameters: 1_000_000_000,
        memory_bytes_f32: 1024 * 1024 * 1024 * 2, // 2 GB
        num_layers: 32,
        hidden_dim: 4096,
        vocab_size: 128000,
        architecture: "qwen2".to_string(),
    };
    let gb = stats.memory_gb();
    assert!((gb - 2.0).abs() < 0.01);
}

#[test]
fn test_conversion_stats_parameters_m() {
    let stats = ConversionStats {
        total_parameters: 7_000_000_000, // 7B
        memory_bytes_f32: 0,
        num_layers: 32,
        hidden_dim: 4096,
        vocab_size: 32000,
        architecture: "llama".to_string(),
    };
    let m = stats.parameters_m();
    assert!((m - 7000.0).abs() < 0.01);
}

#[test]
fn test_conversion_stats_parameters_b() {
    let stats = ConversionStats {
        total_parameters: 7_000_000_000, // 7B
        memory_bytes_f32: 0,
        num_layers: 32,
        hidden_dim: 4096,
        vocab_size: 32000,
        architecture: "llama".to_string(),
    };
    let b = stats.parameters_b();
    assert!((b - 7.0).abs() < 0.01);
}

#[test]
fn test_conversion_stats_small_model() {
    let stats = ConversionStats {
        total_parameters: 125_000_000, // 125M
        memory_bytes_f32: 125_000_000 * 4,
        num_layers: 12,
        hidden_dim: 768,
        vocab_size: 50257,
        architecture: "gpt2".to_string(),
    };
    assert!((stats.parameters_m() - 125.0).abs() < 0.01);
    assert!((stats.parameters_b() - 0.125).abs() < 0.001);
}

#[test]
fn test_conversion_stats_zero_values() {
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
fn test_conversion_stats_debug() {
    let stats = ConversionStats {
        total_parameters: 1000,
        memory_bytes_f32: 4000,
        num_layers: 2,
        hidden_dim: 64,
        vocab_size: 100,
        architecture: "test".to_string(),
    };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("ConversionStats"));
    assert!(debug.contains("1000"));
    assert!(debug.contains("test"));
}

#[test]
fn test_conversion_stats_clone() {
    let stats = ConversionStats {
        total_parameters: 500,
        memory_bytes_f32: 2000,
        num_layers: 4,
        hidden_dim: 128,
        vocab_size: 256,
        architecture: "mini".to_string(),
    };
    let cloned = stats.clone();
    assert_eq!(cloned.total_parameters, 500);
    assert_eq!(cloned.architecture, "mini");
}

// ============================================================================
// GgufToAprConverter::stats coverage
// ============================================================================

fn create_minimal_transformer() -> AprTransformer {
    AprTransformer {
        config: AprTransformerConfig {
            architecture: "test".to_string(),
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 100,
            intermediate_dim: 256,
            context_length: 512,
            rope_theta: 10000.0,
            eps: 1e-5,
        ..Default::default()
        },
        token_embedding: vec![0.0; 64 * 100],
        layers: vec![
            AprTransformerLayer {
                attn_norm_weight: vec![1.0; 64],
                attn_norm_bias: None,
                qkv_weight: vec![0.0; 64 * 192],
                qkv_bias: None,
                attn_output_weight: vec![0.0; 64 * 64],
                attn_output_bias: None,
                ffn_gate_weight: Some(vec![0.0; 256 * 64]),
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.0; 256 * 64],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.0; 64 * 256],
                ffn_down_bias: None,
                ffn_norm_weight: Some(vec![1.0; 64]),
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
            },
            AprTransformerLayer {
                attn_norm_weight: vec![1.0; 64],
                attn_norm_bias: None,
                qkv_weight: vec![0.0; 64 * 192],
                qkv_bias: None,
                attn_output_weight: vec![0.0; 64 * 64],
                attn_output_bias: None,
                ffn_gate_weight: Some(vec![0.0; 256 * 64]),
                ffn_gate_bias: None,
                ffn_up_weight: vec![0.0; 256 * 64],
                ffn_up_bias: None,
                ffn_down_weight: vec![0.0; 64 * 256],
                ffn_down_bias: None,
                ffn_norm_weight: Some(vec![1.0; 64]),
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
            },
        ],
        output_norm_weight: vec![1.0; 64],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 100 * 64],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

#[test]
fn test_gguf_to_apr_stats() {
    let transformer = create_minimal_transformer();
    let stats = GgufToAprConverter::stats(&transformer);

    assert_eq!(stats.num_layers, 2);
    assert_eq!(stats.hidden_dim, 64);
    assert_eq!(stats.vocab_size, 100);
    assert_eq!(stats.architecture, "test");
    assert!(stats.total_parameters > 0);
    assert!(stats.memory_bytes_f32 > 0);
}

// ============================================================================
// GgufToAprConverter::to_apr_bytes coverage
// ============================================================================

#[test]
fn test_to_apr_bytes_creates_valid_header() {
    let transformer = create_minimal_transformer();
    let result = GgufToAprConverter::to_apr_bytes(&transformer);
    assert!(result.is_ok());

    let bytes = result.unwrap();
    // Check APR magic
    assert_eq!(&bytes[0..4], &[0x41, 0x50, 0x52, 0x00]); // APR\0
    // Check version
    assert_eq!(bytes[4], 2); // major
    assert_eq!(bytes[5], 0); // minor
}

#[test]
fn test_to_apr_bytes_roundtrip() {
    let transformer = create_minimal_transformer();
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).unwrap();

    // Should be able to load back
    let restored = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(restored.is_ok());

    let loaded = restored.unwrap();
    assert_eq!(loaded.config.hidden_dim, 64);
    assert_eq!(loaded.config.num_layers, 2);
    assert_eq!(loaded.config.vocab_size, 100);
}

// ============================================================================
// GgufToAprConverter::from_apr_bytes error paths
// ============================================================================

#[test]
fn test_from_apr_bytes_truncated_header() {
    let bytes = vec![0x41, 0x50, 0x52, 0x00, 2, 0]; // Only 6 bytes
    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_wrong_magic() {
    let mut bytes = vec![0u8; 100];
    bytes[0..4].copy_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Wrong magic
    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_truncated_tensor_index() {
    // Create minimal header but truncate before tensor index
    let transformer = create_minimal_transformer();
    let full_bytes = GgufToAprConverter::to_apr_bytes(&transformer).unwrap();

    // Truncate to just header + partial metadata
    let truncated = full_bytes[..80].to_vec();
    let result = GgufToAprConverter::from_apr_bytes(&truncated);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_invalid_tensor_index_json() {
    // Create a valid APR header but with invalid JSON in tensor index region
    let mut bytes = vec![0u8; 256];

    // APR magic
    bytes[0..4].copy_from_slice(&[0x41, 0x50, 0x52, 0x00]);
    bytes[4] = 2; // version major
    bytes[5] = 0; // version minor

    // tensor_count = 1
    bytes[8..12].copy_from_slice(&1u32.to_le_bytes());

    // metadata_offset = 64
    bytes[12..20].copy_from_slice(&64u64.to_le_bytes());
    // metadata_size = 32
    bytes[20..24].copy_from_slice(&32u32.to_le_bytes());
    // tensor_index_offset = 96
    bytes[24..32].copy_from_slice(&96u64.to_le_bytes());
    // data_offset = 160
    bytes[32..40].copy_from_slice(&160u64.to_le_bytes());

    // Put invalid JSON at tensor index location
    bytes[96..100].copy_from_slice(b"{{{{");

    let result = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(result.is_err());
}

// ============================================================================
// GgufToAprConverter::from_gguf_transformer coverage
// ============================================================================

#[test]
fn test_from_gguf_transformer_preserves_config() {
    let gguf = GGUFTransformer {
        config: GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 512,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 2048,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-6,
            rope_type: 0,
            bos_token_id: None,
            eos_token_id: None,
        ..Default::default()
        },
        token_embedding: vec![0.0; 512 * 32000],
        position_embedding: None,
        layers: vec![GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; 512],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; 512 * 1536],
            qkv_bias: None,
            attn_output_weight: vec![0.0; 512 * 512],
            attn_output_bias: None,
            ffn_gate_weight: Some(vec![0.0; 2048 * 512]),
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; 2048 * 512],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; 512 * 2048],
            ffn_down_bias: None,
            ffn_norm_weight: Some(vec![1.0; 512]),
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        }],
        output_norm_weight: vec![1.0; 512],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 32000 * 512],
        lm_head_bias: None,
    };

    let apr = GgufToAprConverter::from_gguf_transformer(&gguf);

    assert_eq!(apr.config.architecture, "llama");
    assert_eq!(apr.config.hidden_dim, 512);
    assert_eq!(apr.config.num_layers, 4);
    assert_eq!(apr.config.num_heads, 8);
    assert_eq!(apr.config.vocab_size, 32000);
    assert_eq!(apr.config.intermediate_dim, 2048);
    assert_eq!(apr.config.context_length, 2048);
    assert!((apr.config.rope_theta - 10000.0).abs() < 0.001);
    assert!((apr.config.eps - 1e-6).abs() < 1e-10);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_conversion_stats_large_values() {
    let stats = ConversionStats {
        total_parameters: 405_000_000_000, // 405B
        memory_bytes_f32: 405_000_000_000 * 4,
        num_layers: 128,
        hidden_dim: 16384,
        vocab_size: 256000,
        architecture: "large_model".to_string(),
    };
    assert!((stats.parameters_b() - 405.0).abs() < 0.01);
    // ~1.5TB in GB
    let gb = stats.memory_gb();
    assert!(gb > 1000.0);
}

#[test]
fn test_to_apr_bytes_empty_layers() {
    let transformer = AprTransformer {
        config: AprTransformerConfig {
            architecture: "empty".to_string(),
            hidden_dim: 32,
            num_layers: 0,
            num_heads: 2,
            num_kv_heads: 2,
            vocab_size: 50,
            intermediate_dim: 64,
            context_length: 128,
            rope_theta: 10000.0,
            eps: 1e-5,
        ..Default::default()
        },
        token_embedding: vec![0.0; 32 * 50],
        layers: vec![], // Empty
        output_norm_weight: vec![1.0; 32],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 50 * 32],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let result = GgufToAprConverter::to_apr_bytes(&transformer);
    assert!(result.is_ok());

    // Roundtrip
    let bytes = result.unwrap();
    let restored = GgufToAprConverter::from_apr_bytes(&bytes);
    assert!(restored.is_ok());
    assert_eq!(restored.unwrap().config.num_layers, 0);
}
