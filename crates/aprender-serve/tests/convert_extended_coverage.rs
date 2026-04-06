//! Extended coverage tests for realizar/src/convert.rs
//!
//! Additional tests for conversion functions, stats helpers, and Q4K converter.

use realizar::apr_transformer::{AprTransformer, AprTransformerConfig};
use realizar::convert::{ConversionStats, GgufToAprConverter, Q4KConversionStats, RawTensor};

// ============================================================================
// ConversionStats helper methods
// ============================================================================

#[test]
fn test_conversion_stats_memory_mb_small() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 4_000_000, // 4MB
        num_layers: 12,
        hidden_dim: 768,
        vocab_size: 50000,
        architecture: "test".to_string(),
    };
    let mb = stats.memory_mb();
    assert!((mb - 3.814).abs() < 0.01, "Expected ~3.814 MB, got {}", mb);
}

#[test]
fn test_conversion_stats_memory_mb_zero() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert_eq!(stats.memory_mb(), 0.0);
}

#[test]
fn test_conversion_stats_memory_gb_large() {
    let stats = ConversionStats {
        total_parameters: 7_000_000_000,
        memory_bytes_f32: 28_000_000_000, // 28GB
        num_layers: 32,
        hidden_dim: 4096,
        vocab_size: 128000,
        architecture: "llama3".to_string(),
    };
    let gb = stats.memory_gb();
    assert!(
        (gb - 26.077).abs() < 0.01,
        "Expected ~26.077 GB, got {}",
        gb
    );
}

#[test]
fn test_conversion_stats_memory_gb_small() {
    let stats = ConversionStats {
        total_parameters: 100_000,
        memory_bytes_f32: 400_000,
        num_layers: 2,
        hidden_dim: 64,
        vocab_size: 1000,
        architecture: "tiny".to_string(),
    };
    let gb = stats.memory_gb();
    assert!(gb < 0.001, "Expected < 0.001 GB, got {}", gb);
}

#[test]
fn test_conversion_stats_parameters_m() {
    let stats = ConversionStats {
        total_parameters: 125_000_000, // 125M
        memory_bytes_f32: 500_000_000,
        num_layers: 12,
        hidden_dim: 768,
        vocab_size: 50000,
        architecture: "bert".to_string(),
    };
    let m = stats.parameters_m();
    assert!((m - 125.0).abs() < 0.01, "Expected 125.0 M, got {}", m);
}

#[test]
fn test_conversion_stats_parameters_m_small() {
    let stats = ConversionStats {
        total_parameters: 500_000,
        memory_bytes_f32: 2_000_000,
        num_layers: 4,
        hidden_dim: 128,
        vocab_size: 5000,
        architecture: "tiny".to_string(),
    };
    let m = stats.parameters_m();
    assert!((m - 0.5).abs() < 0.01, "Expected 0.5 M, got {}", m);
}

#[test]
fn test_conversion_stats_parameters_b() {
    let stats = ConversionStats {
        total_parameters: 13_000_000_000, // 13B
        memory_bytes_f32: 52_000_000_000,
        num_layers: 40,
        hidden_dim: 5120,
        vocab_size: 128000,
        architecture: "llama2-13b".to_string(),
    };
    let b = stats.parameters_b();
    assert!((b - 13.0).abs() < 0.01, "Expected 13.0 B, got {}", b);
}

#[test]
fn test_conversion_stats_parameters_b_small() {
    let stats = ConversionStats {
        total_parameters: 100_000_000,
        memory_bytes_f32: 400_000_000,
        num_layers: 6,
        hidden_dim: 512,
        vocab_size: 32000,
        architecture: "small".to_string(),
    };
    let b = stats.parameters_b();
    assert!((b - 0.1).abs() < 0.001, "Expected 0.1 B, got {}", b);
}

#[test]
fn test_conversion_stats_parameters_b_zero() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert_eq!(stats.parameters_b(), 0.0);
}

// ============================================================================
// RawTensor tests
// ============================================================================

#[test]
fn test_raw_tensor_creation_f32() {
    let tensor = RawTensor {
        name: "model.embed_tokens.weight".to_string(),
        data: vec![0u8; 4 * 128 * 768], // 128 vocab, 768 dim, f32
        dtype: 0,                       // F32
        shape: vec![128, 768],
    };
    assert_eq!(tensor.name, "model.embed_tokens.weight");
    assert_eq!(tensor.data.len(), 4 * 128 * 768);
    assert_eq!(tensor.dtype, 0);
    assert_eq!(tensor.shape, vec![128, 768]);
}

#[test]
fn test_raw_tensor_creation_q4k() {
    // Q4_K block: 144 bytes per 256 elements
    let num_elements = 1024;
    let num_blocks = num_elements / 256;
    let data_size = num_blocks * 144;

    let tensor = RawTensor {
        name: "layer.0.self_attn.q_proj.weight".to_string(),
        data: vec![0u8; data_size],
        dtype: 6, // Q4_K
        shape: vec![1024, 768],
    };
    assert_eq!(tensor.dtype, 6);
    assert_eq!(tensor.shape.len(), 2);
}

#[test]
fn test_raw_tensor_empty() {
    let tensor = RawTensor {
        name: String::new(),
        data: vec![],
        dtype: 0,
        shape: vec![],
    };
    assert!(tensor.name.is_empty());
    assert!(tensor.data.is_empty());
    assert!(tensor.shape.is_empty());
}

#[test]
fn test_raw_tensor_clone() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![1, 2, 3, 4],
        dtype: 1,
        shape: vec![2, 2],
    };
    let cloned = tensor.clone();
    assert_eq!(tensor.name, cloned.name);
    assert_eq!(tensor.data, cloned.data);
    assert_eq!(tensor.dtype, cloned.dtype);
    assert_eq!(tensor.shape, cloned.shape);
}

#[test]
fn test_raw_tensor_debug() {
    let tensor = RawTensor {
        name: "weights".to_string(),
        data: vec![0u8; 16],
        dtype: 2,
        shape: vec![4, 4],
    };
    let debug = format!("{:?}", tensor);
    assert!(debug.contains("RawTensor"));
    assert!(debug.contains("weights"));
    assert!(debug.contains("dtype: 2"));
}

#[test]
fn test_raw_tensor_various_dtypes() {
    let dtypes = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (6, "Q4_K"),
        (7, "Q5_K"),
        (8, "Q6_K"),
        (9, "Q8_0"),
    ];

    for (dtype, name) in dtypes {
        let tensor = RawTensor {
            name: format!("{}_tensor", name),
            data: vec![0u8; 32],
            dtype,
            shape: vec![32],
        };
        assert_eq!(tensor.dtype, dtype);
    }
}

// ============================================================================
// Q4KConversionStats tests
// ============================================================================

#[test]
fn test_q4k_conversion_stats_creation() {
    let stats = Q4KConversionStats {
        tensor_count: 200,
        q4k_tensor_count: 180,
        total_bytes: 10_000_000,
        architecture: "llama".to_string(),
        num_layers: 24,
        hidden_size: 2048,
    };
    assert_eq!(stats.tensor_count, 200);
    assert_eq!(stats.q4k_tensor_count, 180);
    assert_eq!(stats.total_bytes, 10_000_000);
}

#[test]
fn test_q4k_conversion_stats_debug() {
    let stats = Q4KConversionStats {
        tensor_count: 50,
        q4k_tensor_count: 45,
        total_bytes: 5_000_000,
        architecture: "phi".to_string(),
        num_layers: 12,
        hidden_size: 1024,
    };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("Q4KConversionStats"));
    assert!(debug.contains("50"));
    assert!(debug.contains("phi"));
}

#[test]
fn test_q4k_conversion_stats_clone() {
    let stats = Q4KConversionStats {
        tensor_count: 100,
        q4k_tensor_count: 90,
        total_bytes: 1_000_000,
        architecture: "qwen".to_string(),
        num_layers: 8,
        hidden_size: 512,
    };
    let cloned = stats.clone();
    assert_eq!(stats.tensor_count, cloned.tensor_count);
    assert_eq!(stats.q4k_tensor_count, cloned.q4k_tensor_count);
    assert_eq!(stats.architecture, cloned.architecture);
}

#[test]
fn test_q4k_conversion_stats_compression_ratio() {
    let stats = Q4KConversionStats {
        tensor_count: 100,
        q4k_tensor_count: 75,
        total_bytes: 500_000,
        architecture: "test".to_string(),
        num_layers: 6,
        hidden_size: 256,
    };
    // 75% of tensors are Q4K quantized
    let ratio = stats.q4k_tensor_count as f64 / stats.tensor_count as f64;
    assert!((ratio - 0.75).abs() < 0.001);
}

// ============================================================================
// GgufToAprConverter tests - stats function
// ============================================================================

#[test]
fn test_converter_stats_tiny_model() {
    let config = AprTransformerConfig {
        hidden_dim: 8,
        num_layers: 1,
        vocab_size: 10,
        intermediate_dim: 16,
        num_heads: 1,
        num_kv_heads: 1,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let stats = GgufToAprConverter::stats(&transformer);

    assert_eq!(stats.num_layers, 1);
    assert_eq!(stats.hidden_dim, 8);
    assert_eq!(stats.vocab_size, 10);
    assert!(stats.total_parameters > 0);
}

#[test]
fn test_converter_stats_medium_model() {
    let config = AprTransformerConfig {
        hidden_dim: 256,
        num_layers: 6,
        vocab_size: 10000,
        intermediate_dim: 1024,
        num_heads: 8,
        num_kv_heads: 8,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let stats = GgufToAprConverter::stats(&transformer);

    assert_eq!(stats.num_layers, 6);
    assert_eq!(stats.hidden_dim, 256);
    assert_eq!(stats.vocab_size, 10000);
    assert!(stats.total_parameters > 1_000_000);
}

#[test]
fn test_converter_stats_gqa_model() {
    let config = AprTransformerConfig {
        hidden_dim: 128,
        num_layers: 4,
        vocab_size: 5000,
        intermediate_dim: 512,
        num_heads: 8,
        num_kv_heads: 2, // GQA: 8 heads, 2 kv heads
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let stats = GgufToAprConverter::stats(&transformer);

    assert_eq!(stats.num_layers, 4);
    // GQA reduces KV parameters
    assert!(stats.total_parameters > 0);
}

#[test]
fn test_converter_stats_memory_relationship() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        vocab_size: 1000,
        intermediate_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let stats = GgufToAprConverter::stats(&transformer);

    // memory_bytes_f32 should be 4 * total_parameters (f32 is 4 bytes)
    assert_eq!(stats.memory_bytes_f32, stats.total_parameters * 4);
}

// ============================================================================
// GgufToAprConverter tests - roundtrip
// ============================================================================

#[test]
fn test_converter_roundtrip_tiny() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        ..Default::default()
    };
    let original = AprTransformer::new(config);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("to_apr_bytes");
    let restored = GgufToAprConverter::from_apr_bytes(&bytes).expect("from_apr_bytes");

    assert_eq!(original.config().hidden_dim, restored.config().hidden_dim);
    assert_eq!(original.config().num_layers, restored.config().num_layers);
    assert_eq!(original.config().vocab_size, restored.config().vocab_size);
    assert_eq!(
        original.config().intermediate_dim,
        restored.config().intermediate_dim
    );
    assert_eq!(original.config().num_heads, restored.config().num_heads);
    assert_eq!(
        original.config().num_kv_heads,
        restored.config().num_kv_heads
    );
}

#[test]
fn test_converter_roundtrip_with_all_params() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        vocab_size: 100,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 2,
        eps: 1e-6,
        rope_theta: 500000.0,
        context_length: 1024,
        ..Default::default()
    };
    let original = AprTransformer::new(config);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("to_apr_bytes");
    let restored = GgufToAprConverter::from_apr_bytes(&bytes).expect("from_apr_bytes");

    assert_eq!(original.config().eps, restored.config().eps);
    assert_eq!(original.config().rope_theta, restored.config().rope_theta);
    assert_eq!(
        original.config().context_length,
        restored.config().context_length
    );
}

// ============================================================================
// GgufToAprConverter tests - error cases
// ============================================================================

#[test]
fn test_converter_from_apr_bytes_too_small() {
    let data = vec![0u8; 4];
    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_converter_from_apr_bytes_wrong_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"XXXX"); // Wrong magic
    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_converter_from_apr_bytes_empty() {
    let result = GgufToAprConverter::from_apr_bytes(&[]);
    assert!(result.is_err());
}

#[test]
fn test_converter_from_apr_bytes_minimal_valid_header() {
    // APR header is "APR\0" (4 bytes)
    let mut data = vec![0u8; 8];
    data[0..4].copy_from_slice(b"APR\0");
    // Should fail because not enough data for config
    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err());
}

// ============================================================================
// GgufToAprConverter tests - to_apr_bytes
// ============================================================================

#[test]
fn test_converter_to_apr_bytes_magic() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 1,
        vocab_size: 20,
        intermediate_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    // Check APR magic bytes
    assert_eq!(&bytes[0..4], b"APR\0");
}

#[test]
fn test_converter_to_apr_bytes_version() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 1,
        vocab_size: 20,
        intermediate_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    // Version is at bytes 4-8
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert!(
        (1..=100).contains(&version),
        "Unexpected version: {}",
        version
    );
}

#[test]
fn test_converter_to_apr_bytes_not_empty() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 100,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    // Should have substantial data
    assert!(bytes.len() > 100, "APR bytes too small: {}", bytes.len());
}

// ============================================================================
// ConversionStats additional tests
// ============================================================================

#[test]
fn test_conversion_stats_all_architectures() {
    let architectures = [
        "llama", "llama2", "llama3", "phi", "phi2", "phi3", "qwen", "qwen2", "mistral", "gemma",
        "gemma2",
    ];

    for arch in architectures {
        let stats = ConversionStats {
            total_parameters: 1_000_000,
            memory_bytes_f32: 4_000_000,
            num_layers: 12,
            hidden_dim: 768,
            vocab_size: 50000,
            architecture: arch.to_string(),
        };
        assert_eq!(stats.architecture, arch);
    }
}

#[test]
fn test_conversion_stats_extreme_values() {
    let stats = ConversionStats {
        total_parameters: usize::MAX / 2,
        memory_bytes_f32: usize::MAX / 2,
        num_layers: 1000,
        hidden_dim: 100000,
        vocab_size: 10_000_000,
        architecture: "mega".to_string(),
    };

    // Should not panic on extreme values
    let _ = stats.memory_mb();
    let _ = stats.memory_gb();
    let _ = stats.parameters_m();
    let _ = stats.parameters_b();
}

// ============================================================================
// Property-based tests
// ============================================================================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_conversion_stats_memory_relationship(
            params in 1usize..1_000_000_000usize
        ) {
            let stats = ConversionStats {
                total_parameters: params,
                memory_bytes_f32: params * 4,
                num_layers: 12,
                hidden_dim: 768,
                vocab_size: 50000,
                architecture: "test".to_string(),
            };

            // memory_bytes should be 4x parameters for f32
            prop_assert_eq!(stats.memory_bytes_f32, stats.total_parameters * 4);

            // memory_mb should be non-negative
            prop_assert!(stats.memory_mb() >= 0.0);

            // memory_gb should be non-negative
            prop_assert!(stats.memory_gb() >= 0.0);

            // parameters_m should be params / 1e6
            let expected_m = params as f64 / 1_000_000.0;
            prop_assert!((stats.parameters_m() - expected_m).abs() < 0.001);

            // parameters_b should be params / 1e9
            let expected_b = params as f64 / 1_000_000_000.0;
            prop_assert!((stats.parameters_b() - expected_b).abs() < 0.000001);
        }

        #[test]
        fn prop_q4k_stats_ratio_bounded(
            total in 1usize..1000usize,
            q4k in 0usize..1000usize
        ) {
            // q4k can't exceed total
            let q4k_bounded = q4k.min(total);

            let stats = Q4KConversionStats {
                tensor_count: total,
                q4k_tensor_count: q4k_bounded,
                total_bytes: total * 1000,
                architecture: "test".to_string(),
                num_layers: 12,
                hidden_size: 768,
            };

            // Ratio should be in [0, 1]
            let ratio = stats.q4k_tensor_count as f64 / stats.tensor_count as f64;
            prop_assert!((0.0..=1.0).contains(&ratio));
        }

        #[test]
        fn prop_raw_tensor_clone_is_equal(
            name in "\\PC{1,50}",
            data_len in 0usize..1000usize,
            dtype in 0u32..10u32
        ) {
            let tensor = RawTensor {
                name,
                data: vec![0u8; data_len],
                dtype,
                shape: vec![data_len],
            };
            let cloned = tensor.clone();

            prop_assert_eq!(tensor.name, cloned.name);
            prop_assert_eq!(tensor.data.len(), cloned.data.len());
            prop_assert_eq!(tensor.dtype, cloned.dtype);
            prop_assert_eq!(tensor.shape, cloned.shape);
        }
    }
}
