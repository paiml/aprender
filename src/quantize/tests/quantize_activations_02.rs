
// =========================================================================
// Extended Coverage Tests for quantize_activations_q8k_into
// =========================================================================

#[test]
fn test_quantize_activations_q8k_into_success_ext_cov() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

#[test]
fn test_quantize_activations_q8k_into_not_multiple_ext_cov() {
    let activations = vec![1.0f32; 100]; // Not multiple of 256
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 256];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_scales_too_small_ext_cov() {
    let activations = vec![1.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 1]; // Only 1 scale, need 2
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_quants_too_small_ext_cov() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 100]; // Too small
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_err());
}

#[test]
fn test_quantize_activations_q8k_into_multi_superblock_ext_cov() {
    let activations = vec![1.0f32; 512]; // 2 superblocks
    let mut scales = vec![0.0f32; 2];
    let mut quants = vec![0i8; 512];
    let result = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    assert!(result.is_ok());
}

// =========================================================================
// Extended Coverage Tests for quantize_to_q8_blocks
// =========================================================================

#[test]
fn test_quantize_to_q8_blocks_success_ext_cov() {
    let values = vec![1.0f32; 64]; // 2 blocks
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 2);
}

#[test]
fn test_quantize_to_q8_blocks_not_multiple_ext_cov() {
    let values = vec![1.0f32; 50]; // Not multiple of 32
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_err());
}

#[test]
fn test_quantize_to_q8_blocks_single_block_ext_cov() {
    let values = vec![1.0f32; 32];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 1);
}

#[test]
fn test_quantize_to_q8_blocks_empty_ext_cov() {
    let values: Vec<f32> = vec![];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 0);
}

// =========================================================================
// Extended Coverage Tests for dequantize_q8_blocks
// =========================================================================

#[test]
fn test_dequantize_q8_blocks_ext_cov() {
    let blocks = vec![
        Q8_0Block {
            scale: 0.1,
            quants: [10i8; 32],
        },
        Q8_0Block {
            scale: 0.2,
            quants: [5i8; 32],
        },
    ];
    let result = dequantize_q8_blocks(&blocks);
    assert_eq!(result.len(), 64);
}

#[test]
fn test_dequantize_q8_blocks_empty_ext_cov() {
    let blocks: Vec<Q8_0Block> = vec![];
    let result = dequantize_q8_blocks(&blocks);
    assert!(result.is_empty());
}

// =========================================================================
// Extended Coverage Tests for InterleavedQ4K
// =========================================================================

#[test]
fn test_interleaved_q4k_from_q4k_success_ext_cov() {
    let data = vec![0u8; 144]; // 1 superblock
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_ok());
    let interleaved = result.expect("quantization failed");
    assert_eq!(interleaved.num_super_blocks, 1);
}

#[test]
fn test_interleaved_q4k_from_q4k_invalid_ext_cov() {
    let data = vec![0u8; 143]; // Not multiple of 144
    let result = InterleavedQ4K::from_q4k(&data);
    assert!(result.is_err());
}

#[test]
fn test_interleaved_q4k_num_values_ext_cov() {
    let data = vec![0u8; 288]; // 2 superblocks
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    assert_eq!(interleaved.num_values(), 512);
}

#[test]
fn test_interleaved_q4k_clone_ext_cov() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let cloned = interleaved.clone();
    assert_eq!(interleaved.num_super_blocks, cloned.num_super_blocks);
}

#[test]
fn test_interleaved_q4k_debug_ext_cov() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let debug_str = format!("{:?}", interleaved);
    assert!(debug_str.contains("num_super_blocks"));
}

#[test]
fn test_interleaved_q4k_dot_success_ext_cov() {
    let data = vec![0u8; 144]; // 1 superblock
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 256];
    let result = interleaved.dot(&activations);
    assert!(result.is_ok());
}

#[test]
fn test_interleaved_q4k_dot_wrong_size_ext_cov() {
    let data = vec![0u8; 144];
    let interleaved = InterleavedQ4K::from_q4k(&data).expect("quantization failed");
    let activations = vec![1.0f32; 100]; // Wrong size
    let result = interleaved.dot(&activations);
    assert!(result.is_err());
}

// =========================================================================
// Extended Coverage Tests for extract_scale_min_from_slice
// =========================================================================

#[test]
fn test_extract_scale_min_from_slice_even_idx_ext_cov() {
    let scales = [31u8, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0];
    let (scale, min) = extract_scale_min_from_slice(&scales, 0);
    assert!((scale - 31.0).abs() < 1e-6);
    assert!((min - 15.0).abs() < 1e-6);
}

#[test]
fn test_extract_scale_min_from_slice_odd_idx_ext_cov() {
    let mut scales = [0u8; 12];
    scales[0] = 0b11_000000; // high 2 bits contribute to scale[1]
    scales[2] = 0b0000_0011; // low 4 bits contribute to scale[1]
    let (scale, _) = extract_scale_min_from_slice(&scales, 1);
    // scale = (scales[0] >> 6) | ((scales[2] & 0x0F) << 2)
    // = (0b11_000000 >> 6) | ((0x03 & 0x0F) << 2) = 3 | 12 = 15
    assert!((scale - 15.0).abs() < 1e-6);
}

// =========================================================================
// Extended Coverage Tests for Q4_KBlock/Q5_KBlock/Q6_KBlock Clone/Debug
// =========================================================================

#[test]
fn test_q4_k_block_clone_debug_ext_cov() {
    let block = Q4_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qs: [0u8; 128],
    };
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q4_KBlock"));
}

#[test]
fn test_q5_k_block_clone_debug_ext_cov() {
    let block = Q5_KBlock {
        d: 1.0,
        dmin: 0.5,
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q5_KBlock"));
}

#[test]
fn test_q6_k_block_clone_debug_ext_cov() {
    let block = Q6_KBlock {
        d: 1.0,
        scales: [0i8; 16],
        qh: [0u8; 64],
        qs: [0u8; 128],
    };
    let cloned = block.clone();
    assert!((block.d - cloned.d).abs() < 1e-6);
    let debug_str = format!("{:?}", block);
    assert!(debug_str.contains("Q6_KBlock"));
}

// =========================================================================
// Extended Coverage Tests for constants
// =========================================================================

#[test]
fn test_block_size_constant_ext_cov() {
    assert_eq!(BLOCK_SIZE, 32);
}

#[test]
fn test_qk_k_constant_ext_cov() {
    assert_eq!(QK_K, 256);
}

// =========================================================================
// Extended Coverage Tests: DequantStats and SimdBackend
// =========================================================================

#[test]
fn test_dequant_stats_default_ext_cov() {
    let stats = DequantStats::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
    assert_eq!(stats.simd_backend, SimdBackend::Scalar);
}

#[test]
fn test_dequant_stats_clone_ext_cov() {
    let stats = DequantStats {
        blocks_processed: 100,
        bytes_processed: 5000,
        simd_backend: SimdBackend::Avx2,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.blocks_processed, 100);
    assert_eq!(cloned.bytes_processed, 5000);
}

#[test]
fn test_dequant_stats_debug_ext_cov() {
    let stats = DequantStats {
        blocks_processed: 42,
        bytes_processed: 1024,
        simd_backend: SimdBackend::Sse2,
    };
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("DequantStats"));
    assert!(debug_str.contains("42"));
    assert!(debug_str.contains("1024"));
}

#[test]
fn test_simd_backend_all_variants_ext_cov() {
    let variants = [
        SimdBackend::Avx2,
        SimdBackend::Sse2,
        SimdBackend::Neon,
        SimdBackend::Scalar,
    ];
    for v in variants {
        let _ = format!("{:?}", v);
    }
}

#[test]
fn test_simd_backend_display_all_ext_cov() {
    assert_eq!(format!("{}", SimdBackend::Avx2), "AVX2");
    assert_eq!(format!("{}", SimdBackend::Sse2), "SSE2");
    assert_eq!(format!("{}", SimdBackend::Neon), "NEON");
    assert_eq!(format!("{}", SimdBackend::Scalar), "Scalar");
}

#[test]
fn test_simd_backend_clone_ext_cov() {
    let backend = SimdBackend::Avx2;
    let cloned = backend;
    assert_eq!(backend, cloned);
}

#[test]
fn test_simd_backend_eq_ext_cov() {
    assert_eq!(SimdBackend::Scalar, SimdBackend::Scalar);
    assert_ne!(SimdBackend::Scalar, SimdBackend::Avx2);
    assert_ne!(SimdBackend::Sse2, SimdBackend::Neon);
}

#[test]
fn test_simd_backend_default_ext_cov() {
    let backend: SimdBackend = SimdBackend::default();
    assert_eq!(backend, SimdBackend::Scalar);
}

#[test]
fn test_simd_backend_copy_ext_cov() {
    let backend = SimdBackend::Neon;
    let copied = backend;
    assert_eq!(copied, SimdBackend::Neon);
}

// =========================================================================
// Extended Coverage Tests: Error paths for dequantize functions
// =========================================================================

#[test]
fn test_dequantize_q4_0_empty_input_ext_cov() {
    let result = dequantize_q4_0(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q8_0_empty_input_ext_cov() {
    let result = dequantize_q8_0(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_f16_empty_input_ext_cov() {
    let result = dequantize_f16(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_1_empty_input_ext_cov() {
    let result = dequantize_q4_1(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q5_0_empty_input_ext_cov() {
    let result = dequantize_q5_0(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q5_1_empty_input_ext_cov() {
    let result = dequantize_q5_1(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q4_k_empty_input_ext_cov() {
    let result = dequantize_q4_k(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q5_k_empty_input_ext_cov() {
    let result = dequantize_q5_k(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

#[test]
fn test_dequantize_q6_k_empty_input_ext_cov() {
    let result = dequantize_q6_k(&[]);
    assert!(result.is_ok());
    assert!(result.expect("quantization failed").is_empty());
}

// =========================================================================
// Extended Coverage Tests: fused matvec error paths
// =========================================================================

#[test]
fn test_fused_q4k_parallel_matvec_into_invalid_output_ext_cov() {
    let q4k_data = vec![0u8; 144]; // 1 super-block
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let mut output = vec![0.0f32; 0]; // Wrong size
    let result =
        fused_q4k_q8k_parallel_matvec_into(&q4k_data, &scales, &quants, 1, 256, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q4k_parallel_matvec_into_invalid_input_ext_cov() {
    let q4k_data = vec![0u8; 100]; // Invalid size
    let scales = vec![1.0f32; 1];
    let quants = vec![0i8; 256];
    let mut output = vec![0.0f32; 1];
    let result =
        fused_q4k_q8k_parallel_matvec_into(&q4k_data, &scales, &quants, 1, 256, &mut output);
    assert!(result.is_err());
}

#[test]
fn test_fused_q5k_parallel_matvec_into_invalid_input_ext_cov() {
    let data = vec![0u8; 100]; // Invalid
    let activations = vec![0.0f32; 256];
    let mut output = vec![0.0f32; 1];
    let result = fused_q5k_parallel_matvec_into(&data, &activations, 1, 256, &mut output);
    assert!(result.is_err());
}
