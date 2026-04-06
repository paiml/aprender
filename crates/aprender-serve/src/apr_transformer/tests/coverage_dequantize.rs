
#[test]
fn test_dequantize_q4_k_single_block() {
    use super::super::dequant::dequantize_q4_k_apr;

    // 144 bytes for 256 values
    let mut data = vec![0u8; 144];

    // Set d (f16 scale) at offset 0-1
    // 1.0 in f16 = 0x3C00
    data[0] = 0x00;
    data[1] = 0x3C;

    // Set dmin (f16 min) at offset 2-3 to 0
    data[2] = 0x00;
    data[3] = 0x00;

    // Set scales (12 bytes at offset 4-15) to have scale=1 for all blocks
    for i in 0..4 {
        data[4 + i] = 1; // scale for blocks 0-3
        data[8 + i] = 0; // min for blocks 0-3
    }

    // Set qs (128 bytes at offset 16) to 0 (produces 0*scale values)
    // Each nibble=0, so output = d * scale * 0 - dmin * m = 0 - 0 = 0

    let result = dequantize_q4_k_apr(&data, 256);
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q4_k_truncation() {
    use super::super::dequant::dequantize_q4_k_apr;

    // Provide enough data for 1 block (256 values) but request only 100
    let data = vec![0u8; 144];
    let result = dequantize_q4_k_apr(&data, 100);

    assert_eq!(result.len(), 100);
}

#[test]
fn test_dequantize_q6_k_empty() {
    use super::super::dequant::dequantize_q6_k_apr;

    let result = dequantize_q6_k_apr(&[], 0);
    assert!(result.is_empty());
}

#[test]
fn test_dequantize_q6_k_insufficient_data() {
    use super::super::dequant::dequantize_q6_k_apr;

    // Request 256 elements but provide only 10 bytes (need 210)
    let data = vec![0u8; 10];
    let result = dequantize_q6_k_apr(&data, 256);

    // Should return zeros
    assert_eq!(result.len(), 256);
    assert!(result.iter().all(|&x| x == 0.0));
}

#[test]
fn test_dequantize_q6_k_single_block() {
    use super::super::dequant::dequantize_q6_k_apr;

    // 210 bytes for 256 values
    // Layout: ql (128) + qh (64) + scales (16) + d (2) = 210
    let mut data = vec![0u8; 210];

    // Set d (f16 scale) at offset 208-209
    // 1.0 in f16 = 0x3C00
    data[208] = 0x00;
    data[209] = 0x3C;

    let result = dequantize_q6_k_apr(&data, 256);
    assert_eq!(result.len(), 256);
}

#[test]
fn test_dequantize_q6_k_truncation() {
    use super::super::dequant::dequantize_q6_k_apr;

    // Provide enough data for 1 block but request only 100
    let data = vec![0u8; 210];
    let result = dequantize_q6_k_apr(&data, 100);

    assert_eq!(result.len(), 100);
}

// ============================================================================
// Part 9: SIMD Helpers Tests
// ============================================================================

#[test]
fn test_simd_dot_f32_empty() {
    use super::super::helpers::simd_dot_f32;

    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let result = simd_dot_f32(&a, &b);
    assert_eq!(result, 0.0);
}

#[test]
fn test_simd_dot_f32_small() {
    use super::super::helpers::simd_dot_f32;

    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    let result = simd_dot_f32(&a, &b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!((result - 32.0).abs() < 1e-6);
}

#[test]
fn test_simd_dot_f32_exact_8() {
    use super::super::helpers::simd_dot_f32;

    // Exactly 8 elements - exercises AVX2 path without remainder
    let a = [1.0f32; 8];
    let b = [2.0f32; 8];
    let result = simd_dot_f32(&a, &b);

    // 8 * (1.0 * 2.0) = 16.0
    assert!((result - 16.0).abs() < 1e-5);
}

#[test]
fn test_simd_dot_f32_large() {
    use super::super::helpers::simd_dot_f32;

    // Large enough to use AVX2 path with remainder
    let n = 100;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let result = simd_dot_f32(&a, &b);

    // Expected: sum of i * (n - i) for i in 0..n
    let expected: f32 = (0..n).map(|i| (i as f32) * ((n - i) as f32)).sum();
    assert!((result - expected).abs() < 1e-2);
}

#[test]
fn test_simd_add_weighted_empty() {
    use super::super::helpers::simd_add_weighted;

    let mut out: [f32; 0] = [];
    let val: [f32; 0] = [];
    simd_add_weighted(&mut out, &val, 2.0);
    // Should not panic
}

#[test]
fn test_simd_add_weighted_small() {
    use super::super::helpers::simd_add_weighted;

    let mut out = [1.0f32, 2.0, 3.0];
    let val = [1.0f32, 1.0, 1.0];
    simd_add_weighted(&mut out, &val, 2.0);

    // out[i] += 2.0 * val[i]
    assert!((out[0] - 3.0).abs() < 1e-6); // 1 + 2*1 = 3
    assert!((out[1] - 4.0).abs() < 1e-6); // 2 + 2*1 = 4
    assert!((out[2] - 5.0).abs() < 1e-6); // 3 + 2*1 = 5
}

#[test]
fn test_simd_add_weighted_exact_8() {
    use super::super::helpers::simd_add_weighted;

    let mut out = [0.0f32; 8];
    let val = [1.0f32; 8];
    simd_add_weighted(&mut out, &val, 3.0);

    assert!(out.iter().all(|&x| (x - 3.0).abs() < 1e-5));
}

#[test]
fn test_simd_add_weighted_large() {
    use super::super::helpers::simd_add_weighted;

    let n = 100;
    let mut out = vec![1.0f32; n];
    let val: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let weight = 0.5;

    simd_add_weighted(&mut out, &val, weight);

    for (i, &o) in out.iter().enumerate() {
        let expected = 1.0 + 0.5 * (i as f32);
        assert!((o - expected).abs() < 1e-5, "Mismatch at index {i}");
    }
}

#[test]
fn test_simd_add_weighted_negative_weight() {
    use super::super::helpers::simd_add_weighted;

    let mut out = [10.0f32, 20.0, 30.0, 40.0];
    let val = [1.0f32, 2.0, 3.0, 4.0];
    simd_add_weighted(&mut out, &val, -1.0);

    // out[i] -= val[i]
    assert!((out[0] - 9.0).abs() < 1e-6);
    assert!((out[1] - 18.0).abs() < 1e-6);
    assert!((out[2] - 27.0).abs() < 1e-6);
    assert!((out[3] - 36.0).abs() < 1e-6);
}

// ============================================================================
// Part 10: MmapAprTransformer Tests (Error Paths)
// ============================================================================

#[test]
fn test_mmap_transformer_file_not_found() {
    let result = MmapAprTransformer::from_file("/nonexistent/path/model.apr");
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_get_tensor_bytes_out_of_bounds() {
    // Create a minimal valid APR file in memory and test bounds
    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];

    // Set magic
    data[0..4].copy_from_slice(&MAGIC);

    // Set version to 1
    data[4..8].copy_from_slice(&1u32.to_le_bytes());

    // Set minimal config
    data[8..12].copy_from_slice(&64u32.to_le_bytes()); // hidden_dim
    data[12..16].copy_from_slice(&1u32.to_le_bytes()); // num_layers
    data[16..20].copy_from_slice(&4u32.to_le_bytes()); // num_heads
    data[20..24].copy_from_slice(&4u32.to_le_bytes()); // num_kv_heads
    data[24..28].copy_from_slice(&100u32.to_le_bytes()); // vocab_size
    data[28..32].copy_from_slice(&128u32.to_le_bytes()); // intermediate_dim
    data[32..36].copy_from_slice(&256u32.to_le_bytes()); // context_length
    data[36..40].copy_from_slice(&10000.0f32.to_le_bytes()); // rope_theta
    data[40..44].copy_from_slice(&1e-5f32.to_le_bytes()); // eps
    data[44..48].copy_from_slice(&(APR_TRANSFORMER_HEADER_SIZE as u32).to_le_bytes()); // tensor_data_offset

    // Write to temp file
    use std::io::Write;
    let mut temp_file = tempfile::NamedTempFile::new().expect("test value should be present");
    temp_file.write_all(&data).expect("test value should be present");

    let transformer = MmapAprTransformer::from_file(temp_file.path()).expect("test value should be present");

    // Try to read beyond file bounds
    let result = transformer.get_tensor_bytes(0, 1000);
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_invalid_magic() {
    use std::io::Write;

    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];
    data[0..4].copy_from_slice(b"GGUF"); // Wrong magic

    let mut temp_file = tempfile::NamedTempFile::new().expect("test value should be present");
    temp_file.write_all(&data).expect("test value should be present");

    let result = MmapAprTransformer::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_file_too_small() {
    use std::io::Write;

    let data = vec![0u8; 32]; // Too small

    let mut temp_file = tempfile::NamedTempFile::new().expect("test value should be present");
    temp_file.write_all(&data).expect("test value should be present");

    let result = MmapAprTransformer::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_unsupported_version() {
    use std::io::Write;

    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];
    data[0..4].copy_from_slice(&MAGIC);
    data[4..8].copy_from_slice(&99u32.to_le_bytes()); // Invalid version

    let mut temp_file = tempfile::NamedTempFile::new().expect("test value should be present");
    temp_file.write_all(&data).expect("test value should be present");

    let result = MmapAprTransformer::from_file(temp_file.path());
    assert!(result.is_err());
}

#[test]
fn test_mmap_transformer_accessors() {
    use std::io::Write;

    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];

    data[0..4].copy_from_slice(&MAGIC);
    data[4..8].copy_from_slice(&1u32.to_le_bytes());
    data[8..12].copy_from_slice(&64u32.to_le_bytes());
    data[12..16].copy_from_slice(&2u32.to_le_bytes());
    data[16..20].copy_from_slice(&4u32.to_le_bytes());
    data[20..24].copy_from_slice(&4u32.to_le_bytes());
    data[24..28].copy_from_slice(&100u32.to_le_bytes());
    data[28..32].copy_from_slice(&128u32.to_le_bytes());
    data[32..36].copy_from_slice(&256u32.to_le_bytes());
    data[36..40].copy_from_slice(&10000.0f32.to_le_bytes());
    data[40..44].copy_from_slice(&1e-5f32.to_le_bytes());
    data[44..48].copy_from_slice(&(APR_TRANSFORMER_HEADER_SIZE as u32).to_le_bytes());

    let mut temp_file = tempfile::NamedTempFile::new().expect("test value should be present");
    temp_file.write_all(&data).expect("test value should be present");

    let transformer = MmapAprTransformer::from_file(temp_file.path()).expect("test value should be present");

    assert!(transformer.is_mmap());
    assert_eq!(transformer.file_size(), data.len());
    assert!(transformer.num_parameters() > 0);
    assert_eq!(transformer.config.hidden_dim, 64);
    assert_eq!(transformer.config.num_layers, 2);
}

#[test]
fn test_mmap_transformer_get_tensor_f32() {
    use std::io::Write;

    let mut data = vec![0u8; APR_TRANSFORMER_HEADER_SIZE + 100];

    data[0..4].copy_from_slice(&MAGIC);
    data[4..8].copy_from_slice(&1u32.to_le_bytes());
    data[8..12].copy_from_slice(&64u32.to_le_bytes());
    data[12..16].copy_from_slice(&1u32.to_le_bytes());
    data[16..20].copy_from_slice(&4u32.to_le_bytes());
    data[20..24].copy_from_slice(&4u32.to_le_bytes());
    data[24..28].copy_from_slice(&100u32.to_le_bytes());
    data[28..32].copy_from_slice(&128u32.to_le_bytes());
    data[32..36].copy_from_slice(&256u32.to_le_bytes());
    data[36..40].copy_from_slice(&10000.0f32.to_le_bytes());
    data[40..44].copy_from_slice(&1e-5f32.to_le_bytes());
    data[44..48].copy_from_slice(&(APR_TRANSFORMER_HEADER_SIZE as u32).to_le_bytes());

    // Write some f32 values after header
    let test_values = [1.0f32, 2.0, 3.0, 4.0];
    for (i, &val) in test_values.iter().enumerate() {
        let bytes = val.to_le_bytes();
        let offset = APR_TRANSFORMER_HEADER_SIZE + i * 4;
        data[offset..offset + 4].copy_from_slice(&bytes);
    }

    let mut temp_file = tempfile::NamedTempFile::new().expect("test value should be present");
    temp_file.write_all(&data).expect("test value should be present");

    let transformer = MmapAprTransformer::from_file(temp_file.path()).expect("test value should be present");

    let floats = transformer.get_tensor_f32(0, 4).expect("test value should be present");
    assert_eq!(floats.len(), 4);
    assert!((floats[0] - 1.0).abs() < 1e-6);
    assert!((floats[1] - 2.0).abs() < 1e-6);
    assert!((floats[2] - 3.0).abs() < 1e-6);
    assert!((floats[3] - 4.0).abs() < 1e-6);
}

// ============================================================================
// Benchmark Infrastructure Tests (apr_transformer/benchmark.rs)
// ============================================================================

use crate::apr_transformer::benchmark::{
    AprBenchmarkResult, AprLoadResult, AprParityComparison, AprPrefillResult,
    APR_CPU_DECODE_THRESHOLD_TOK_S, APR_PARITY_THRESHOLD_PCT, APR_PREFILL_THRESHOLD_TOK_S,
};

#[test]
fn test_apr_benchmark_result_default() {
    let result = AprBenchmarkResult::default();
    assert_eq!(result.tokens_generated, 0);
    assert_eq!(result.total_time_ms, 0.0);
    assert_eq!(result.tokens_per_second, 0.0);
    assert_eq!(result.throughput_p50, 0.0);
    assert_eq!(result.throughput_p99, 0.0);
    assert_eq!(result.throughput_std_dev, 0.0);
    assert_eq!(result.peak_memory_mb, 0.0);
    assert_eq!(result.model_memory_mb, 0.0);
}

#[test]
fn test_apr_benchmark_result_meets_threshold_above() {
    let result = AprBenchmarkResult {
        tokens_per_second: 60.0,
        ..Default::default()
    };
    assert!(result.meets_threshold(APR_CPU_DECODE_THRESHOLD_TOK_S));
}

#[test]
fn test_apr_benchmark_result_meets_threshold_below() {
    let result = AprBenchmarkResult {
        tokens_per_second: 40.0,
        ..Default::default()
    };
    assert!(!result.meets_threshold(APR_CPU_DECODE_THRESHOLD_TOK_S));
}

#[test]
fn test_apr_benchmark_result_meets_threshold_exact() {
    let result = AprBenchmarkResult {
        tokens_per_second: APR_CPU_DECODE_THRESHOLD_TOK_S,
        ..Default::default()
    };
    assert!(result.meets_threshold(APR_CPU_DECODE_THRESHOLD_TOK_S));
}

#[test]
fn test_apr_benchmark_result_compare_to_baseline() {
    let result = AprBenchmarkResult {
        tokens_per_second: 95.0,
        peak_memory_mb: 100.0,
        ..Default::default()
    };
    let baseline = AprBenchmarkResult {
        tokens_per_second: 100.0,
        peak_memory_mb: 80.0,
        ..Default::default()
    };
    let comparison = result.compare_to_baseline(&baseline);
    assert!((comparison.throughput_ratio - 0.95).abs() < 1e-6);
    assert!((comparison.memory_ratio - 1.25).abs() < 1e-6);
    assert_eq!(comparison.parity_threshold_pct, APR_PARITY_THRESHOLD_PCT);
}
