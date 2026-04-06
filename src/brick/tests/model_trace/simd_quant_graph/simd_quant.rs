use super::super::super::super::*;

// =========================================================================
// SIMD-EXP: Tests for SIMD-accelerated softmax
// =========================================================================

/// SIMD-EXP-001: SoftmaxOp produces correct results with SIMD backend
#[test]
fn test_simd_exp_001_softmax_simd_correctness() {
    let op = SoftmaxOp::new(8);
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Test with AVX2 backend
    let result = op.execute(input.clone(), Backend::Avx2).unwrap();

    // Verify sum is 1.0
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax sum should be 1.0, got {}", sum);

    // Verify monotonicity (larger inputs -> larger outputs)
    for i in 1..result.len() {
        assert!(
            result[i] > result[i - 1],
            "Softmax should be monotonic: result[{}]={} <= result[{}]={}",
            i,
            result[i],
            i - 1,
            result[i - 1]
        );
    }
}

/// SIMD-EXP-002: SoftmaxOp SIMD matches scalar
#[test]
fn test_simd_exp_002_simd_matches_scalar() {
    let op = SoftmaxOp::new(16);
    let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 4.0).collect();

    let scalar_result = op.execute(input.clone(), Backend::Scalar).unwrap();
    let simd_result = op.execute(input.clone(), Backend::Avx2).unwrap();

    // Results should match within floating point tolerance
    for (i, (s, a)) in scalar_result.iter().zip(simd_result.iter()).enumerate() {
        assert!((s - a).abs() < 1e-5, "Mismatch at index {}: scalar={}, simd={}", i, s, a);
    }
}

/// SIMD-EXP-003: SoftmaxOp handles negative values
#[test]
fn test_simd_exp_003_negative_values() {
    let op = SoftmaxOp::new(4);
    let input = vec![-10.0, -5.0, 0.0, 5.0];

    let result = op.execute(input, Backend::Auto).unwrap();

    // Sum should be 1.0
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Largest input should have largest probability
    assert!(result[3] > result[2] && result[2] > result[1] && result[1] > result[0]);
}

/// SIMD-EXP-004: SoftmaxOp numerical stability with large values
#[test]
fn test_simd_exp_004_numerical_stability() {
    let op = SoftmaxOp::new(3);
    // Large values that would overflow without max subtraction
    let input = vec![1000.0, 1001.0, 1002.0];

    let result = op.execute(input, Backend::Avx2).unwrap();

    // Should not produce NaN or Inf
    for &v in &result {
        assert!(!v.is_nan(), "Softmax produced NaN");
        assert!(!v.is_infinite(), "Softmax produced Inf");
    }

    // Sum should still be 1.0
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// =========================================================================
// QUANT-Q5K: Tests for Q5_K and Q6_K quantization
// =========================================================================

/// QUANT-Q5K-001: BlockQ5K dequantization basic test
#[test]
fn test_quant_q5k_001_basic_dequant() {
    let block = BlockQ5K {
        d: 1.0,
        dmin: 0.0,
        scales: [32; 12], // Zero scale (after -32 adjustment)
        qh: [0; 32],
        qs: [0; 128],
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // With zero scales and zero values, output should be related to dmin and d
    // The dequant formula is: d * scale * (q5 - 16) + dmin
    // With scale=0 (32-32) and q5=0, we get: d * 0 * (0-16) + dmin = dmin
    for &v in &output {
        assert!((v - 0.0).abs() < 1e-3, "Expected near zero with zero scale, got {}", v);
    }
}

/// QUANT-Q5K-002: DotQ5KOp empty input
#[test]
fn test_quant_q5k_002_empty_input() {
    let op = DotQ5KOp::new(256);
    let result = op.execute((vec![], vec![]), Backend::Scalar).unwrap();
    assert_eq!(result, 0.0);
}

/// QUANT-Q5K-003: BlockQ6K dequantization basic test
#[test]
fn test_quant_q6k_001_basic_dequant() {
    let block = BlockQ6K {
        ql: [0; 128],
        qh: [0; 64],
        scales: [0; 16], // Zero scales
        d: 1.0,
    };

    let mut output = [0.0f32; 256];
    block.dequantize(&mut output);

    // With zero scales and zero values, output should be:
    // d * scale * (q6 - 32) = 1.0 * 0 * (0 - 32) = 0
    for &v in &output {
        assert!((v - 0.0).abs() < 1e-3, "Expected near zero with zero scale, got {}", v);
    }
}

/// QUANT-Q5K-004: DotQ6KOp empty input
#[test]
fn test_quant_q6k_002_empty_input() {
    let op = DotQ6KOp::new(256);
    let result = op.execute((vec![], vec![]), Backend::Scalar).unwrap();
    assert_eq!(result, 0.0);
}

/// QUANT-Q5K-005: Block sizes are correct
#[test]
fn test_quant_block_sizes() {
    assert_eq!(BlockQ5K::BLOCK_SIZE, 256);
    assert_eq!(BlockQ6K::BLOCK_SIZE, 256);
}

/// QUANT-Q5K-006: DotQ5KOp name method
#[test]
fn test_quant_q5k_op_name() {
    let op = DotQ5KOp::new(256);
    assert_eq!(op.name(), "dot_q5k");
}

/// QUANT-Q5K-007: DotQ6KOp name method
#[test]
fn test_quant_q6k_op_name() {
    let op = DotQ6KOp::new(256);
    assert_eq!(op.name(), "dot_q6k");
}

/// QUANT-Q5K-008: DotQ5KOp tokens method
#[test]
fn test_quant_q5k_tokens() {
    let op = DotQ5KOp::new(512);
    let block = BlockQ5K { d: 1.0, dmin: 0.0, scales: [32; 12], qh: [0; 32], qs: [0; 128] };
    let input = (vec![block.clone(), block], vec![0.0f32; 512]);
    assert_eq!(op.tokens(&input), 512); // 2 blocks * 256
}

/// QUANT-Q5K-009: DotQ6KOp tokens method
#[test]
fn test_quant_q6k_tokens() {
    let op = DotQ6KOp::new(512);
    let block = BlockQ6K { ql: [0; 128], qh: [0; 64], scales: [0; 16], d: 1.0 };
    let input = (vec![block.clone(), block], vec![0.0f32; 512]);
    assert_eq!(op.tokens(&input), 512); // 2 blocks * 256
}

/// SIMD-EXP-005: SoftmaxOp is_simd_backend check
#[test]
fn test_simd_exp_005_backend_check() {
    assert!(SoftmaxOp::is_simd_backend(Backend::Avx2));
    assert!(SoftmaxOp::is_simd_backend(Backend::Avx512));
    assert!(SoftmaxOp::is_simd_backend(Backend::Sse2));
    assert!(SoftmaxOp::is_simd_backend(Backend::Neon));
    assert!(SoftmaxOp::is_simd_backend(Backend::Auto));
    assert!(!SoftmaxOp::is_simd_backend(Backend::Scalar));
    assert!(!SoftmaxOp::is_simd_backend(Backend::Wasm));
}
