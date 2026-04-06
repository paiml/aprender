//! T-COV-95 Coverage Bridge (Part 11 - ConversionStats, RawTensor, Q4KConversionStats, crc32/checksum)
//!
//! Covers:
//! - ConversionStats::memory_mb, memory_gb, parameters_m, parameters_b
//! - ConversionStats struct construction and Debug/Clone
//! - RawTensor struct construction and Debug/Clone
//! - Q4KConversionStats struct construction and Debug/Clone
//! - crc32 via round-trip checksum validation
//! - compute_apr_header_checksum via round-trip validation
//! - GgufToAprConverter::stats with synthetic AprTransformer
//! - from_apr_bytes error paths (truncated, missing weights, bad json)

use super::*;

// ============================================================================
// ConversionStats
// ============================================================================

#[test]
fn test_conversion_stats_memory_mb() {
    let stats = ConversionStats {
        total_parameters: 1_000_000,
        memory_bytes_f32: 1024 * 1024, // 1 MB
        num_layers: 12,
        hidden_dim: 768,
        vocab_size: 32000,
        architecture: "qwen2".to_string(),
    };
    let mb = stats.memory_mb();
    assert!((mb - 1.0).abs() < 0.001, "Expected 1.0 MB, got {}", mb);
}

#[test]
fn test_conversion_stats_memory_gb() {
    let stats = ConversionStats {
        total_parameters: 1_000_000_000,
        memory_bytes_f32: 1024 * 1024 * 1024, // 1 GB
        num_layers: 32,
        hidden_dim: 4096,
        vocab_size: 128000,
        architecture: "llama".to_string(),
    };
    let gb = stats.memory_gb();
    assert!((gb - 1.0).abs() < 0.001, "Expected 1.0 GB, got {}", gb);
}

#[test]
fn test_conversion_stats_parameters_m() {
    let stats = ConversionStats {
        total_parameters: 7_000_000,
        memory_bytes_f32: 28_000_000,
        num_layers: 6,
        hidden_dim: 512,
        vocab_size: 10000,
        architecture: "phi2".to_string(),
    };
    let pm = stats.parameters_m();
    assert!((pm - 7.0).abs() < 0.001, "Expected 7.0M, got {}", pm);
}

#[test]
fn test_conversion_stats_parameters_b() {
    let stats = ConversionStats {
        total_parameters: 1_500_000_000,
        memory_bytes_f32: 6_000_000_000,
        num_layers: 48,
        hidden_dim: 8192,
        vocab_size: 128000,
        architecture: "llama".to_string(),
    };
    let pb = stats.parameters_b();
    assert!((pb - 1.5).abs() < 0.001, "Expected 1.5B, got {}", pb);
}

#[test]
fn test_conversion_stats_zero_bytes() {
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
fn test_conversion_stats_clone() {
    let stats = ConversionStats {
        total_parameters: 100,
        memory_bytes_f32: 400,
        num_layers: 2,
        hidden_dim: 64,
        vocab_size: 100,
        architecture: "test".to_string(),
    };
    let cloned = stats.clone();
    assert_eq!(cloned.total_parameters, 100);
    assert_eq!(cloned.architecture, "test");
}

#[test]
fn test_conversion_stats_debug() {
    let stats = ConversionStats {
        total_parameters: 100,
        memory_bytes_f32: 400,
        num_layers: 2,
        hidden_dim: 64,
        vocab_size: 100,
        architecture: "debug_test".to_string(),
    };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("ConversionStats"));
    assert!(debug.contains("debug_test"));
}

// ============================================================================
// RawTensor
// ============================================================================

#[test]
fn test_raw_tensor_construction() {
    let tensor = RawTensor {
        name: "blk.0.attn_q.weight".to_string(),
        data: vec![0u8; 144],
        shape: vec![256, 512],
        dtype: 12, // Q4_K
    };
    assert_eq!(tensor.name, "blk.0.attn_q.weight");
    assert_eq!(tensor.data.len(), 144);
    assert_eq!(tensor.shape, vec![256, 512]);
    assert_eq!(tensor.dtype, 12);
}

#[test]
fn test_raw_tensor_clone() {
    let tensor = RawTensor {
        name: "test".to_string(),
        data: vec![1, 2, 3],
        shape: vec![3],
        dtype: 0,
    };
    let cloned = tensor.clone();
    assert_eq!(cloned.name, "test");
    assert_eq!(cloned.data, vec![1, 2, 3]);
}

#[test]
fn test_raw_tensor_debug() {
    let tensor = RawTensor {
        name: "tensor_debug".to_string(),
        data: vec![0u8; 4],
        shape: vec![2, 2],
        dtype: 1,
    };
    let debug = format!("{:?}", tensor);
    assert!(debug.contains("RawTensor"));
    assert!(debug.contains("tensor_debug"));
}

#[test]
fn test_raw_tensor_all_dtypes() {
    // Test creating RawTensor with all supported GGML dtype codes
    let dtypes = [0u32, 1, 2, 3, 6, 7, 8, 12, 13, 14];
    for dtype in dtypes {
        let tensor = RawTensor {
            name: format!("dtype_{}", dtype),
            data: vec![0u8; 32],
            shape: vec![32],
            dtype,
        };
        assert_eq!(tensor.dtype, dtype);
    }
}

// ============================================================================
// Q4KConversionStats
// ============================================================================

#[test]
fn test_q4k_conversion_stats_construction() {
    let stats = Q4KConversionStats {
        tensor_count: 50,
        q4k_tensor_count: 40,
        total_bytes: 1_000_000,
        architecture: "qwen2".to_string(),
        num_layers: 24,
        hidden_size: 2048,
    };
    assert_eq!(stats.tensor_count, 50);
    assert_eq!(stats.q4k_tensor_count, 40);
    assert_eq!(stats.total_bytes, 1_000_000);
    assert_eq!(stats.architecture, "qwen2");
    assert_eq!(stats.num_layers, 24);
    assert_eq!(stats.hidden_size, 2048);
}

#[test]
fn test_q4k_conversion_stats_clone() {
    let stats = Q4KConversionStats {
        tensor_count: 10,
        q4k_tensor_count: 8,
        total_bytes: 500,
        architecture: "test".to_string(),
        num_layers: 2,
        hidden_size: 128,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.tensor_count, 10);
    assert_eq!(cloned.architecture, "test");
}

#[test]
fn test_q4k_conversion_stats_debug() {
    let stats = Q4KConversionStats {
        tensor_count: 10,
        q4k_tensor_count: 8,
        total_bytes: 500,
        architecture: "debug_q4k".to_string(),
        num_layers: 2,
        hidden_size: 128,
    };
    let debug = format!("{:?}", stats);
    assert!(debug.contains("Q4KConversionStats"));
    assert!(debug.contains("debug_q4k"));
}

// ============================================================================
// CRC32 and header checksum (via from_apr_bytes / to_apr_bytes round-trip)
// ============================================================================

#[test]
fn test_crc32_known_values() {
    // CRC32 of empty data
    let crc_empty = crc32(&[]);
    assert_eq!(crc_empty, 0x0000_0000, "CRC32 of empty should be 0");

    // CRC32 of "123456789" (known IEEE CRC32 test vector)
    let crc_test = crc32(b"123456789");
    assert_eq!(
        crc_test, 0xCBF4_3926,
        "CRC32 of '123456789' should be 0xCBF43926"
    );
}

#[test]
fn test_crc32_deterministic() {
    let data = b"Hello, World!";
    let crc1 = crc32(data);
    let crc2 = crc32(data);
    assert_eq!(crc1, crc2, "CRC32 should be deterministic");
}

#[test]
fn test_crc32_different_data() {
    let crc1 = crc32(b"abc");
    let crc2 = crc32(b"abd");
    assert_ne!(crc1, crc2, "Different data should produce different CRC32");
}

#[test]
fn test_compute_apr_header_checksum_deterministic() {
    let mut header = vec![0u8; 64];
    // Set some fields
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = 2;
    header[8..12].copy_from_slice(&1u32.to_le_bytes());

    let checksum1 = compute_apr_header_checksum(&header);
    let checksum2 = compute_apr_header_checksum(&header);
    assert_eq!(
        checksum1, checksum2,
        "Header checksum should be deterministic"
    );
}

#[test]
fn test_compute_apr_header_checksum_varies_with_content() {
    let mut header1 = vec![0u8; 64];
    header1[0..4].copy_from_slice(&MAGIC);
    header1[4] = 2;

    let mut header2 = vec![0u8; 64];
    header2[0..4].copy_from_slice(&MAGIC);
    header2[4] = 3; // Different version

    let checksum1 = compute_apr_header_checksum(&header1);
    let checksum2 = compute_apr_header_checksum(&header2);
    assert_ne!(
        checksum1, checksum2,
        "Different headers should produce different checksums"
    );
}

#[test]
fn test_compute_apr_header_checksum_ignores_checksum_field() {
    let mut header1 = vec![0u8; 64];
    header1[0..4].copy_from_slice(&MAGIC);
    header1[4] = 2;

    let mut header2 = header1.clone();
    // Modify bytes 40..44 (checksum field itself)
    header2[40] = 0xFF;
    header2[41] = 0xAA;
    header2[42] = 0x55;
    header2[43] = 0x01;

    let checksum1 = compute_apr_header_checksum(&header1);
    let checksum2 = compute_apr_header_checksum(&header2);
    assert_eq!(
        checksum1, checksum2,
        "Checksum field [40..44] should be excluded from computation"
    );
}

// ============================================================================
// from_apr_bytes error paths
// ============================================================================

#[test]
fn test_from_apr_bytes_too_small() {
    let result = GgufToAprConverter::from_apr_bytes(&[0u8; 10]);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_bad_magic() {
    let mut data = vec![0u8; 128];
    // Wrong magic
    data[0..4].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err());
}

#[test]
fn test_from_apr_bytes_truncated_tensor_index() {
    // Create a header that points to an index beyond the data
    let mut header = vec![0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = 2;
    header[5] = 0;
    // tensor_index_offset at 64, data_offset at 200 (beyond file)
    header[24..32].copy_from_slice(&64u64.to_le_bytes());
    header[32..40].copy_from_slice(&200u64.to_le_bytes());

    let checksum = compute_apr_header_checksum(&header);
    header[40..44].copy_from_slice(&checksum.to_le_bytes());

    // File is only 128 bytes, but data_offset says 200
    let mut data = header.clone();
    data.extend_from_slice(&[0u8; 64]); // metadata padding only

    let result = GgufToAprConverter::from_apr_bytes(&data);
    assert!(result.is_err(), "Should fail because file is truncated");
}

// ============================================================================
// GgufToAprConverter::stats
// ============================================================================

#[test]
fn test_gguf_to_apr_converter_stats() {
    use crate::apr_transformer::{AprTransformer, AprTransformerConfig};

    let config = AprTransformerConfig {
        architecture: "test_arch".to_string(),
        hidden_dim: 128,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 512,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
            eos_token_id: None,
    ..Default::default()
    };

    let transformer = AprTransformer {
        config,
        token_embedding: vec![0.0; 128 * 100], // vocab_size * hidden_dim
        layers: vec![],
        output_norm_weight: vec![0.0; 128],
        output_norm_bias: None,
        lm_head_weight: vec![0.0; 128 * 100],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    };

    let stats = GgufToAprConverter::stats(&transformer);
    assert_eq!(stats.architecture, "test_arch");
    assert_eq!(stats.num_layers, 2);
    assert_eq!(stats.hidden_dim, 128);
    assert_eq!(stats.vocab_size, 100);
    assert!(stats.total_parameters > 0);
    assert!(stats.memory_bytes_f32 > 0);
}

// ============================================================================
// infer_rope_type
// ============================================================================

#[test]
fn test_infer_rope_type_neox_architectures() {
    let empty_meta = std::collections::HashMap::new();

    // All these should return 2 (NEOX)
    let neox_archs = [
        "qwen2",
        "qwen3",
        "phi3",
        "gemma",
        "gemma2",
        "falcon",
        "starcoder2",
        "bert",
        "deepseek2",
        "internlm2",
        "nemotron",
        "exaone",
    ];

    for arch in neox_archs {
        let result = GgufToAprQ4KConverter::infer_rope_type(arch, &empty_meta);
        assert_eq!(
            result, 2,
            "Architecture '{}' should use NEOX rope_type (2)",
            arch
        );
    }
}

include!("tests_infer_rope_02.rs");
