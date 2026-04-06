use crate::quantize::*;
#[test]
fn test_quantize_activations_q8_0_valid_cov() {
    let activations: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 2); // 64/32 = 2 blocks
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_activations_q8_0_empty_cov() {
    let activations: Vec<f32> = vec![];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert!(scales.is_empty());
    assert!(quants.is_empty());
}

// =========================================================================
// Deep Coverage Tests: apply_rope_rotation_simd
// =========================================================================

#[test]
fn test_apply_rope_rotation_simd_identity_cov() {
    let mut x1 = vec![1.0, 2.0, 3.0, 4.0];
    let mut x2 = vec![5.0, 6.0, 7.0, 8.0];
    let cos_vals = vec![1.0, 1.0, 1.0, 1.0]; // cos(0) = 1
    let sin_vals = vec![0.0, 0.0, 0.0, 0.0]; // sin(0) = 0

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=1, sin=0: x1' = x1, x2' = x2
    assert_eq!(x1, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(x2, vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_apply_rope_rotation_simd_ninety_deg_cov() {
    let mut x1 = vec![1.0, 0.0];
    let mut x2 = vec![0.0, 1.0];
    let cos_vals = vec![0.0, 0.0]; // cos(90) = 0
    let sin_vals = vec![1.0, 1.0]; // sin(90) = 1

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // x1' = x1*0 - x2*1 = -x2
    // x2' = x1*1 + x2*0 = x1
    assert!((x1[0] - 0.0).abs() < 1e-6);
    assert!((x2[0] - 1.0).abs() < 1e-6);
}

// =========================================================================
// Deep Coverage Tests: apply_rope_rotation_simd
// =========================================================================

#[test]
fn test_apply_rope_rotation_simd_basic_cov() {
    let mut x1: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut x2: Vec<f32> = (0..16).map(|i| (i + 16) as f32).collect();
    let cos_vals: Vec<f32> = vec![1.0; 16];
    let sin_vals: Vec<f32> = vec![0.0; 16];

    apply_rope_rotation_simd(&mut x1, &mut x2, &cos_vals, &sin_vals);

    // With cos=1, sin=0: values unchanged
    assert_eq!(x1[0], 0.0);
    assert_eq!(x2[0], 16.0);
}

// =========================================================================
// Deep Coverage Tests: softmax_simd
// =========================================================================

#[test]
fn test_softmax_simd_basic_deep2() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    softmax_simd(&mut x);

    // Sum should be 1.0
    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Values should be ordered (higher input -> higher probability)
    assert!(x[3] > x[2]);
    assert!(x[2] > x[1]);
    assert!(x[1] > x[0]);
}

#[test]
fn test_softmax_simd_large_values_deep2() {
    let mut x = vec![100.0, 101.0, 102.0, 103.0];
    softmax_simd(&mut x);

    let sum: f32 = x.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_simd_uniform_deep2() {
    let mut x = vec![5.0; 4];
    softmax_simd(&mut x);

    // Uniform input -> uniform output
    for &val in &x {
        assert!((val - 0.25).abs() < 1e-5);
    }
}

// =========================================================================
// Deep Coverage Tests: fused_swiglu_simd
// =========================================================================

#[test]
fn test_fused_swiglu_simd_basic_deep2() {
    let mut gate = vec![1.0, 2.0, 3.0, 4.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];

    fused_swiglu_simd(&mut gate, &up);

    // gate should be modified
    assert!(gate[0] > 0.0);
}

#[test]
fn test_fused_swiglu_simd_zeros_deep2() {
    let mut gate = vec![0.0; 8];
    let up = vec![1.0; 8];

    fused_swiglu_simd(&mut gate, &up);

    // sigmoid(0) = 0.5, so gate[i] = 0 * 0.5 * 1 = 0
    for &val in &gate {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

// =========================================================================
// Deep Coverage Tests: DequantStats and SimdBackend
// =========================================================================

#[test]
fn test_simd_backend_display_cov() {
    let backend = detect_simd_backend();
    let display = format!("{backend}");
    assert!(!display.is_empty());
}

#[test]
fn test_dequant_stats_default_cov() {
    let stats = DequantStats::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
}

// =========================================================================
// Deep Coverage Tests: fused matvec functions
// =========================================================================

#[test]
fn test_fused_q4k_parallel_matvec_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 144 bytes per row
    let weights = vec![0u8; 288];
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_parallel_matvec(&weights, &activations, 256, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_fused_q5k_parallel_matvec_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 176 bytes per row
    let weights = vec![0u8; 352];
    let activations = vec![1.0f32; 256];
    let result = fused_q5k_parallel_matvec(&weights, &activations, 256, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_fused_q6k_parallel_matvec_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 210 bytes per row
    let weights = vec![0u8; 420];
    let activations = vec![1.0f32; 256];
    let result = fused_q6k_parallel_matvec(&weights, &activations, 256, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

// =========================================================================
// Deep Coverage Tests: fused_q4_0_q8_0 functions
// =========================================================================

#[test]
fn test_fused_q4_0_q8_0_parallel_matvec_valid_cov() {
    // Q4_0: 18 bytes per 32 values
    // in_dim=32, out_dim=2 -> 2 rows * 18 bytes per row
    let mut weights = vec![0u8; 36];
    // Set scales to 1.0 as f16
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale[0];
    weights[1] = scale[1];
    weights[18] = scale[0];
    weights[19] = scale[1];

    let activations = vec![1.0f32; 32];
    let result = fused_q4_0_q8_0_parallel_matvec(&weights, &activations, 32, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

// =========================================================================
// Deep Coverage Tests: parallel dequantization
// =========================================================================

#[test]
fn test_dequantize_q4_k_parallel_valid_cov() {
    let data = vec![0u8; 144];
    let result = dequantize_q4_k_parallel(&data);
    assert!(result.is_ok());
    let values = result.expect("quantization failed");
    assert_eq!(values.len(), 256);
}

#[test]
fn test_dequantize_q4_k_simd_valid_cov() {
    let data = vec![0u8; 144];
    let result = dequantize_q4_k_simd(&data);
    assert!(result.is_ok());
    let values = result.expect("quantization failed");
    assert_eq!(values.len(), 256);
}

#[test]
fn test_dequantize_q8_0_parallel_valid_cov() {
    // Q8_0: 34 bytes per 32 values
    let data = vec![0u8; 34];
    let result = dequantize_q8_0_parallel(&data);
    assert!(result.is_ok());
    let values = result.expect("quantization failed");
    assert_eq!(values.len(), 32);
}

#[test]
fn test_dequantize_q8_0_simd_valid_cov() {
    let data = vec![0u8; 34];
    let result = dequantize_q8_0_simd(&data);
    assert!(result.is_ok());
    let values = result.expect("quantization failed");
    assert_eq!(values.len(), 32);
}

// =========================================================================
// Deep Coverage Tests: fused_rmsnorm functions
// =========================================================================

#[test]
fn test_quantize_rmsnorm_q8_0_basic_cov() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
    let norm_weight = vec![1.0f32; 64];
    let eps = 1e-5;

    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, eps);
    assert_eq!(scales.len(), 2); // 64/32 = 2 blocks
    assert_eq!(quants.len(), 64);
}

#[test]
fn test_quantize_rmsnorm_q8_0_into_basic_cov() {
    let input: Vec<f32> = (0..32).map(|i| i as f32 / 10.0).collect();
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;
    let mut scales = vec![0.0f32; 1];
    let mut quants = vec![0i8; 32];

    quantize_rmsnorm_q8_0_into(&input, &norm_weight, eps, &mut scales, &mut quants);
    assert!(scales[0] > 0.0);
}

#[test]
fn test_fused_rmsnorm_q4_0_matmul_cov() {
    // in_dim=32, out_dim=2 -> 2 rows * 18 bytes per row
    let mut weights = vec![0u8; 36];
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale[0];
    weights[1] = scale[1];
    weights[18] = scale[0];
    weights[19] = scale[1];

    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let eps = 1e-5;

    // Signature: (input, norm_weight, eps, weight_data, in_dim, out_dim)
    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, eps, &weights, 32, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_cov() {
    // For Q4_K: 144 bytes per 256 values
    // in_dim=256, out_dim=1
    let up_weights = vec![0u8; 144];
    let gate_weights = vec![0u8; 144];
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];
    let eps = 1e-5;

    // Signature: (input, norm_weight, eps, up_weight_data, gate_weight_data, in_dim, out_dim)
    let result = fused_rmsnorm_ffn_up_gate(
        &input,
        &norm_weight,
        eps,
        &up_weights,
        &gate_weights,
        256,
        1,
    );
    assert!(result.is_ok());
    let (up_out, gate_out) = result.expect("quantization failed");
    assert_eq!(up_out.len(), 1);
    assert_eq!(gate_out.len(), 1);
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_tiled_matvec
// =========================================================================

#[test]
fn test_fused_q4k_tiled_matvec_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 144 bytes per row
    let weights = vec![0u8; 288];
    let activations = vec![1.0f32; 256];
    let result = fused_q4k_tiled_matvec(&weights, &activations, 256, 2, None);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

// =========================================================================
// Deep Coverage Tests: fused_q8_0_q8_0 functions
// =========================================================================

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_valid_cov() {
    // Q8_0: 34 bytes per 32 values
    // in_dim=32, out_dim=2 -> 2 rows * 34 bytes per row
    let mut weights = vec![0u8; 68];
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale[0];
    weights[1] = scale[1];
    weights[34] = scale[0];
    weights[35] = scale[1];

    let activations = vec![1.0f32; 32];
    let result = fused_q8_0_q8_0_parallel_matvec(&weights, &activations, 32, 2);
    assert!(result.is_ok());
    let output = result.expect("quantization failed");
    assert_eq!(output.len(), 2);
}

#[test]
fn test_fused_q8_0_q8_0_parallel_matvec_into_valid_cov() {
    let mut weights = vec![0u8; 68];
    let scale = half::f16::from_f32(1.0).to_le_bytes();
    weights[0] = scale[0];
    weights[1] = scale[1];
    weights[34] = scale[0];
    weights[35] = scale[1];

    let activations = vec![1.0f32; 32];
    let mut output = vec![0.0f32; 2];
    let result = fused_q8_0_q8_0_parallel_matvec_into(&weights, &activations, 32, 2, &mut output);
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_q8k_parallel_matvec_into
// =========================================================================

#[test]
fn test_fused_q4k_q8k_parallel_matvec_into_valid_cov() {
    // in_dim=256, out_dim=2 -> 2 rows * 144 bytes per row
    let weights = vec![0u8; 288];
    let q8k_scales = vec![1.0f32; 1]; // One scale for 256 values
    let q8k_quants = vec![0i8; 256];
    let mut output = vec![0.0f32; 2];

    let result =
        fused_q4k_q8k_parallel_matvec_into(&weights, &q8k_scales, &q8k_quants, 256, 2, &mut output);
    assert!(result.is_ok());
}

// =========================================================================
// Deep Coverage Tests: fused_q4k_q8k_ffn_up_gate_into
// =========================================================================

#[test]
fn test_fused_q4k_q8k_ffn_up_gate_into_valid_cov() {
    // Signature: (up_weight, gate_weight, q8k_scales, q8k_quants, in_dim, out_dim, up_output, gate_output)
    let up_weights = vec![0u8; 144];
    let gate_weights = vec![0u8; 144];
    let q8k_scales = vec![1.0f32; 1];
    let q8k_quants = vec![0i8; 256];
    let mut up_output = vec![0.0f32; 1];
    let mut gate_output = vec![0.0f32; 1];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        256,
        1,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());
}

include!("interleaved_q4k_03.rs");
include!("fused.rs");
include!("cov95_fused.rs");
