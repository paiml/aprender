
#[test]
fn test_cov95_fused_q6k_dot_basic() {
    // Q6_K block: 210 bytes for 256 values
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot(&q6k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q6k_dot_simd_basic() {
    let q6k_data = vec![0u8; 210];
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_dot_simd(&q6k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q5k_dot_basic() {
    // Q5_K block: 176 bytes for 256 values
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot(&q5k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q5k_dot_simd_basic() {
    let q5k_data = vec![0u8; 176];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_dot_simd(&q5k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q4k_tiled_matvec_small() {
    // Small matrix for tiled matvec
    let in_dim = 256;
    let out_dim = 64;
    let bytes_per_row = 144; // Q4_K block size
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_tiled_matvec(&weight_data, &activations, in_dim, out_dim, None);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

#[test]
fn test_cov95_fused_q4k_parallel_matvec_basic() {
    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 144;
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

#[test]
fn test_cov95_fused_q5k_parallel_matvec_basic() {
    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 176; // Q5_K block size
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

#[test]
fn test_cov95_fused_q6k_parallel_matvec_basic() {
    let in_dim = 256;
    let out_dim = 32;
    let bytes_per_row = 210; // Q6_K block size
    let weight_data = vec![0u8; bytes_per_row * out_dim];
    let activations = vec![1.0f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weight_data, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), out_dim);
}

#[test]
fn test_cov95_fused_q4k_q8_dot_basic() {
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let q8_blocks = vec![
        Q8_0Block {
            scale: 1.0,
            quants: [0i8; 32],
        };
        8
    ]; // 256 values

    let result = fused_q4k_q8_dot(&q4k_data, &q8_blocks);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q4k_q8k_dot_basic() {
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 8]; // 8 blocks for 256 values
    let q8k_quants = vec![0i8; 256];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_fused_q4k_q8k_dot_simd_basic() {
    // Q4_K super-block: 144 bytes for 256 values
    let q4k_data = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 8];
    let q8k_quants = vec![0i8; 256];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_cov95_quantize_rmsnorm_q8_0_basic() {
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);
    assert!(!scales.is_empty());
    assert!(!quants.is_empty());
}

#[test]
fn test_cov95_quantize_rmsnorm_q8_0_into_basic() {
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];
    let eps = 1e-5;
    let mut scales = vec![0.0f32; 8]; // 8 blocks
    let mut quants = vec![0i8; 256];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);
    // At least some scales should be non-zero
    let has_nonzero = scales.iter().any(|&s| s.abs() > 1e-10);
    assert!(has_nonzero);
}

#[test]
fn test_cov95_fused_swiglu_simd_basic() {
    let mut gate = vec![0.5f32; 256];
    let up = vec![1.0f32; 256];

    fused_swiglu_simd(&mut gate, &up);
    // Gate should be modified
    let modified = gate.iter().any(|&g| (g - 0.5).abs() > 1e-6);
    assert!(modified);
}

#[test]
fn test_cov95_softmax_simd_basic() {
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    softmax_simd(&mut x);

    // Sum should be ~1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_cov95_softmax_simd_large() {
    let mut x: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

#[test]
fn test_cov95_dequantize_q4_1_basic() {
    // Q4_1 block: 20 bytes for 32 values
    let data = vec![0u8; 20];
    let result = dequantize_q4_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 32);
}

#[test]
fn test_cov95_dequantize_q5_0_basic() {
    // Q5_0 block: 22 bytes for 32 values
    let data = vec![0u8; 22];
    let result = dequantize_q5_0(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 32);
}

#[test]
fn test_cov95_dequantize_q5_1_basic() {
    // Q5_1 block: 24 bytes for 32 values
    let data = vec![0u8; 24];
    let result = dequantize_q5_1(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 32);
}

#[test]
fn test_cov95_dequantize_q2_k_basic() {
    // Q2_K block: 84 bytes for 256 values
    let data = vec![0u8; 84];
    let result = dequantize_q2_k(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 256);
}

#[test]
fn test_cov95_dequantize_f16_basic() {
    // Two f16 values (4 bytes)
    let data = vec![0u8; 4];
    let result = dequantize_f16(&data);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 2);
}

#[test]
fn test_cov95_f16_to_f32_special_values() {
    // Test special f16 values
    let zero = f16_to_f32(0x0000);
    assert_eq!(zero, 0.0);

    let neg_zero = f16_to_f32(0x8000);
    assert_eq!(neg_zero, 0.0);

    let one = f16_to_f32(0x3C00); // 1.0 in f16
    assert!((one - 1.0).abs() < 1e-3);
}

// LAYOUT-002: Alias tests DELETED (2026-02-03)
// ONE WAY ONLY: Use fused_q{4,5,6}k_parallel_matvec* functions directly

#[test]
fn test_cov95_quantize_activations_q8k_into_basic() {
    let activations = vec![1.0f32; 256];
    let mut scales = vec![0.0f32; 8];
    let mut quants = vec![0i8; 256];

    let _ = quantize_activations_q8k_into(&activations, &mut scales, &mut quants);
    let has_nonzero = scales.iter().any(|&s| s.abs() > 1e-10);
    assert!(has_nonzero);
}

#[test]
fn test_cov95_quantize_to_q8_blocks_basic() {
    let values = vec![1.0f32; 32];
    let result = quantize_to_q8_blocks(&values);
    assert!(result.is_ok());
    assert_eq!(result.expect("quantization failed").len(), 1);
}

#[test]
fn test_cov95_dequantize_q8_blocks_basic() {
    let blocks = vec![Q8_0Block {
        scale: 1.0,
        quants: [64i8; 32],
    }];
    let result = dequantize_q8_blocks(&blocks);
    assert_eq!(result.len(), 32);
}
