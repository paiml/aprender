//! Deep coverage tests for realizar/src/convert.rs
//!
//! This module provides additional coverage for conversion functions
//! not covered by existing tests. Targets 95%+ coverage.

use realizar::apr_transformer::{AprTransformer, AprTransformerConfig};
use realizar::convert::{ConversionStats, GgufToAprConverter, Q4KConversionStats, RawTensor};

// ============================================================================
// Test 1-10: ConversionStats tests
// ============================================================================

#[test]
fn test_conversion_stats_zeroed() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert_eq!(stats.total_parameters, 0);
    assert_eq!(stats.memory_bytes_f32, 0);
    assert_eq!(stats.num_layers, 0);
    assert_eq!(stats.hidden_dim, 0);
    assert_eq!(stats.vocab_size, 0);
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
    let debug = format!("{stats:?}");
    assert!(debug.contains("ConversionStats"));
    assert!(debug.contains("1000"));
}

#[test]
fn test_conversion_stats_clone() {
    let stats = ConversionStats {
        total_parameters: 1000,
        memory_bytes_f32: 4000,
        num_layers: 2,
        hidden_dim: 64,
        vocab_size: 100,
        architecture: "llama".to_string(),
    };
    let cloned = stats.clone();
    assert_eq!(stats.total_parameters, cloned.total_parameters);
    assert_eq!(stats.memory_bytes_f32, cloned.memory_bytes_f32);
    assert_eq!(stats.architecture, cloned.architecture);
}

#[test]
fn test_conversion_stats_with_architecture() {
    let stats = ConversionStats {
        total_parameters: 7_000_000_000,
        memory_bytes_f32: 28_000_000_000,
        num_layers: 32,
        hidden_dim: 4096,
        vocab_size: 128000,
        architecture: "llama3".to_string(),
    };
    assert_eq!(stats.architecture, "llama3");
    assert_eq!(stats.num_layers, 32);
}

// ============================================================================
// Test 11-20: RawTensor tests
// ============================================================================

#[test]
fn test_raw_tensor_creation() {
    let tensor = RawTensor {
        name: "test_tensor".to_string(),
        data: vec![0u8; 1024],
        dtype: 0,
        shape: vec![32, 32],
    };
    assert_eq!(tensor.name, "test_tensor");
    assert_eq!(tensor.data.len(), 1024);
    assert_eq!(tensor.dtype, 0);
    assert_eq!(tensor.shape, vec![32, 32]);
}

#[test]
fn test_raw_tensor_debug() {
    let tensor = RawTensor {
        name: "weights".to_string(),
        data: vec![0u8; 100],
        dtype: 1,
        shape: vec![10, 10],
    };
    let debug = format!("{tensor:?}");
    assert!(debug.contains("RawTensor"));
    assert!(debug.contains("weights"));
}

#[test]
fn test_raw_tensor_clone() {
    let tensor = RawTensor {
        name: "original".to_string(),
        data: vec![1, 2, 3, 4],
        dtype: 0,
        shape: vec![2, 2],
    };
    let cloned = tensor.clone();
    assert_eq!(tensor.name, cloned.name);
    assert_eq!(tensor.data, cloned.data);
    assert_eq!(tensor.dtype, cloned.dtype);
    assert_eq!(tensor.shape, cloned.shape);
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
fn test_raw_tensor_large_shape() {
    let tensor = RawTensor {
        name: "large".to_string(),
        data: vec![0u8; 16],
        dtype: 0,
        shape: vec![4096, 4096, 32],
    };
    assert_eq!(tensor.shape.len(), 3);
    assert_eq!(tensor.shape[0], 4096);
}

#[test]
fn test_raw_tensor_single_dim() {
    let tensor = RawTensor {
        name: "1d".to_string(),
        data: vec![0u8; 100],
        dtype: 0,
        shape: vec![100],
    };
    assert_eq!(tensor.shape.len(), 1);
    assert_eq!(tensor.shape[0], 100);
}

// ============================================================================
// Test 21-30: Q4KConversionStats tests
// ============================================================================

#[test]
fn test_q4k_conversion_stats_zeroed() {
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
    assert_eq!(stats.num_layers, 0);
}

#[test]
fn test_q4k_conversion_stats_clone() {
    let stats = Q4KConversionStats {
        tensor_count: 100,
        q4k_tensor_count: 80,
        total_bytes: 1_000_000,
        architecture: "phi3".to_string(),
        num_layers: 12,
        hidden_size: 2048,
    };
    let cloned = stats.clone();
    assert_eq!(stats.tensor_count, cloned.tensor_count);
    assert_eq!(stats.q4k_tensor_count, cloned.q4k_tensor_count);
    assert_eq!(stats.total_bytes, cloned.total_bytes);
    assert_eq!(stats.architecture, cloned.architecture);
}

#[test]
fn test_q4k_conversion_stats_debug() {
    let stats = Q4KConversionStats {
        tensor_count: 50,
        q4k_tensor_count: 40,
        total_bytes: 500_000,
        architecture: "qwen".to_string(),
        num_layers: 8,
        hidden_size: 1024,
    };
    let debug = format!("{stats:?}");
    assert!(debug.contains("Q4KConversionStats"));
    assert!(debug.contains("50"));
}

#[test]
fn test_q4k_conversion_stats_with_architecture() {
    let stats = Q4KConversionStats {
        tensor_count: 200,
        q4k_tensor_count: 180,
        total_bytes: 10_000_000,
        architecture: "mistral".to_string(),
        num_layers: 24,
        hidden_size: 4096,
    };
    assert_eq!(stats.architecture, "mistral");
    assert_eq!(stats.num_layers, 24);
    assert_eq!(stats.hidden_size, 4096);
}

// ============================================================================
// Test 31-40: GgufToAprConverter tests
// ============================================================================

#[test]
fn test_converter_stats_small_model() {
    let config = AprTransformerConfig {
        hidden_dim: 16,
        num_layers: 1,
        vocab_size: 10,
        intermediate_dim: 32,
        num_heads: 2,
        num_kv_heads: 2,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let stats = GgufToAprConverter::stats(&transformer);

    assert_eq!(stats.num_layers, 1);
    assert_eq!(stats.hidden_dim, 16);
    assert_eq!(stats.vocab_size, 10);
    assert!(stats.total_parameters > 0);
}

#[test]
fn test_converter_stats_large_vocab() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 128000,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let stats = GgufToAprConverter::stats(&transformer);

    assert_eq!(stats.vocab_size, 128000);
    // Large vocab means most parameters are in embeddings
    assert!(stats.total_parameters > 128000 * 32);
}

#[test]
fn test_converter_to_apr_bytes_creates_valid_header() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    // Check magic
    assert_eq!(&bytes[0..4], b"APR\0");
}

#[test]
fn test_converter_from_apr_bytes_too_small() {
    let data = vec![0u8; 10];
    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_converter_from_apr_bytes_invalid_magic() {
    let mut data = vec![0u8; 128];
    data[0..4].copy_from_slice(b"XXXX");
    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_converter_apr_bytes_roundtrip() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let original = AprTransformer::new(config);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("to_apr_bytes");
    let restored = GgufToAprConverter::from_apr_bytes(&bytes).expect("from_apr_bytes");

    assert_eq!(original.config().hidden_dim, restored.config().hidden_dim);
    assert_eq!(original.config().num_layers, restored.config().num_layers);
    assert_eq!(original.config().vocab_size, restored.config().vocab_size);
}

// ============================================================================
// Test 41-50: More converter edge cases
// ============================================================================

#[test]
fn test_converter_preserves_gqa_config() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        vocab_size: 100,
        intermediate_dim: 128,
        num_heads: 8,
        num_kv_heads: 2, // GQA config
        ..Default::default()
    };
    let original = AprTransformer::new(config);
    let bytes = GgufToAprConverter::to_apr_bytes(&original).expect("to_apr_bytes");
    let restored = GgufToAprConverter::from_apr_bytes(&bytes).expect("from_apr_bytes");

    assert_eq!(restored.config().num_heads, 8);
    assert_eq!(restored.config().num_kv_heads, 2);
}

#[test]
fn test_transformer_with_biases_to_apr_bytes() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let mut transformer = AprTransformer::new(config);

    // Add optional biases
    transformer.layers[0].qkv_bias = Some(vec![0.1; 3 * 32]);
    transformer.layers[0].attn_output_bias = Some(vec![0.1; 32]);
    transformer.output_norm_bias = Some(vec![0.1; 32]);
    transformer.lm_head_bias = Some(vec![0.1; 50]);

    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");
    assert!(!bytes.is_empty());
    assert_eq!(&bytes[0..4], b"APR\0");
}

#[test]
fn test_converter_stats_multi_layer() {
    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 8,
        vocab_size: 32000,
        intermediate_dim: 256,
        num_heads: 8,
        num_kv_heads: 8,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let stats = GgufToAprConverter::stats(&transformer);

    assert_eq!(stats.num_layers, 8);
    // More layers means more parameters
    assert!(stats.total_parameters > 1_000_000);
}

#[test]
fn test_conversion_stats_memory_relationship() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 4_000_000, // Should be total_parameters * 4
        num_layers: 4,
        hidden_dim: 512,
        vocab_size: 32000,
        architecture: "test".to_string(),
    };
    // F32 is 4 bytes per parameter
    assert_eq!(stats.memory_bytes_f32, stats.total_parameters * 4);
}

#[test]
fn test_raw_tensor_dtype_values() {
    // Test different dtype values
    let f32_tensor = RawTensor {
        name: "f32".to_string(),
        data: vec![0u8; 128],
        dtype: 0, // F32
        shape: vec![32],
    };
    assert_eq!(f32_tensor.dtype, 0);

    let f16_tensor = RawTensor {
        name: "f16".to_string(),
        data: vec![0u8; 64],
        dtype: 1, // F16
        shape: vec![32],
    };
    assert_eq!(f16_tensor.dtype, 1);

    let q4k_tensor = RawTensor {
        name: "q4k".to_string(),
        data: vec![0u8; 144],
        dtype: 6, // Q4_K
        shape: vec![256],
    };
    assert_eq!(q4k_tensor.dtype, 6);
}

// ============================================================================
// Test 51-60: Additional edge cases
// ============================================================================

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
    assert_eq!(stats.total_parameters, 0);
    assert!(stats.architecture.is_empty());
}

#[test]
fn test_q4k_stats_compression_ratio() {
    let stats = Q4KConversionStats {
        tensor_count: 100,
        q4k_tensor_count: 90,
        total_bytes: 1_000_000,
        architecture: "phi".to_string(),
        num_layers: 24,
        hidden_size: 2048,
    };
    // 90% of tensors are Q4K quantized
    let q4k_ratio = stats.q4k_tensor_count as f64 / stats.tensor_count as f64;
    assert!((q4k_ratio - 0.9).abs() < 0.01);
}

#[test]
fn test_converter_apr_bytes_header_version() {
    let config = AprTransformerConfig {
        hidden_dim: 32,
        num_layers: 1,
        vocab_size: 50,
        intermediate_dim: 64,
        num_heads: 4,
        num_kv_heads: 4,
        ..Default::default()
    };
    let transformer = AprTransformer::new(config);
    let bytes = GgufToAprConverter::to_apr_bytes(&transformer).expect("to_apr_bytes");

    // Check version field (bytes 4-8)
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    // Version should be reasonable (1-10)
    assert!((1..=10).contains(&version));
}

#[test]
fn test_raw_tensor_high_dimensional() {
    let tensor = RawTensor {
        name: "4d".to_string(),
        data: vec![0u8; 16],
        dtype: 0,
        shape: vec![2, 2, 2, 2],
    };
    assert_eq!(tensor.shape.len(), 4);
    assert_eq!(tensor.shape.iter().product::<usize>(), 16);
}

#[test]
fn test_conversion_stats_architecture_variety() {
    let architectures = vec!["llama", "phi", "qwen", "mistral", "gemma"];
    for arch in architectures {
        let stats = ConversionStats {
            total_parameters: 1000,
            memory_bytes_f32: 4000,
            num_layers: 2,
            hidden_dim: 64,
            vocab_size: 100,
            architecture: arch.to_string(),
        };
        assert_eq!(stats.architecture, arch);
    }
}
