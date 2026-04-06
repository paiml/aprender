
#[test]
fn test_simd_bf16_to_f32_exact_8_values() {
    // Test exactly 8 values (one SIMD chunk on AVX2)
    let values: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut bf16_bytes = Vec::with_capacity(16);
    for &v in &values {
        bf16_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }

    let result = simd_bf16_to_f32(&bf16_bytes);
    assert_eq!(result.len(), 8);

    for (i, (&expected, &actual)) in values.iter().zip(result.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 0.01,
            "Index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_simd_f16_to_f32_empty() {
    let result = simd_f16_to_f32(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_simd_f16_to_f32_special_values() {
    let zero_bytes = half::f16::from_f32(0.0).to_le_bytes();
    let result = simd_f16_to_f32(&zero_bytes);
    assert!((result[0]).abs() < 1e-10);

    let inf_bytes = half::f16::from_f32(f32::INFINITY).to_le_bytes();
    let result = simd_f16_to_f32(&inf_bytes);
    assert!(result[0].is_infinite());
}

#[test]
fn test_simd_f16_to_f32_subnormal() {
    // Test a very small F16 value (subnormal)
    let small = half::f16::from_f32(1e-7);
    let bytes = small.to_le_bytes();
    let result = simd_f16_to_f32(&bytes);
    assert!(result[0] >= 0.0);
    assert!(result[0] < 1e-3);
}

// ============================================================================
// BF16 Dot Product Edge Cases
// ============================================================================

#[test]
fn test_simd_bf16_dot_empty() {
    let result = simd_bf16_dot(&[], &[]);
    assert!((result).abs() < 1e-5);
}

#[test]
fn test_simd_bf16_dot_single_value() {
    let a_bytes = half::bf16::from_f32(3.0).to_le_bytes().to_vec();
    let b_bytes = half::bf16::from_f32(4.0).to_le_bytes().to_vec();
    let result = simd_bf16_dot(&a_bytes, &b_bytes);
    assert!((result - 12.0).abs() < 0.1);
}

#[test]
fn test_simd_bf16_dot_mismatched_lengths() {
    let mut a_bytes = Vec::new();
    let mut b_bytes = Vec::new();

    // a has 3 values, b has 2 values
    for v in [1.0, 2.0, 3.0] {
        a_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }
    for v in [1.0, 1.0] {
        b_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
    }

    // Should use minimum length
    let result = simd_bf16_dot(&a_bytes, &b_bytes);
    // Only first 2 values used: 1*1 + 2*1 = 3
    assert!((result - 3.0).abs() < 0.1);
}

#[test]
fn test_simd_bf16_dot_chunk_boundary() {
    // Test with exactly 64 values (one chunk)
    let n = 64;
    let mut a_bytes = Vec::with_capacity(n * 2);
    let mut b_bytes = Vec::with_capacity(n * 2);

    for i in 0..n {
        let v = ((i % 5) as f32) * 0.2;
        a_bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        b_bytes.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
    }

    let result = simd_bf16_dot(&a_bytes, &b_bytes);
    // Sum of 0.0, 0.2, 0.4, 0.6, 0.8, 0.0, ... (13 complete cycles = 2.0 each, + 3 values)
    // 64/5 = 12 complete + 4 remainder (0.0, 0.2, 0.4, 0.6 = 1.2)
    let expected = 12.0 * 2.0 + 1.2;
    assert!(
        (result - expected).abs() < 1.0,
        "Expected ~{}, got {}",
        expected,
        result
    );
}

// ============================================================================
// BF16 Matmul Edge Cases
// ============================================================================

#[test]
fn test_simd_bf16_matmul_single_element() {
    let input = vec![2.0];
    let weight_bytes = half::bf16::from_f32(3.0).to_le_bytes().to_vec();

    let output = simd_bf16_matmul(&input, &weight_bytes, 1, 1);
    assert_eq!(output.len(), 1);
    assert!((output[0] - 6.0).abs() < 0.1);
}

#[test]
fn test_simd_bf16_matmul_zeros() {
    let input = vec![0.0; 4];
    let mut weight_bytes = Vec::new();
    for _ in 0..8 {
        weight_bytes.extend_from_slice(&half::bf16::from_f32(1.0).to_le_bytes());
    }

    let output = simd_bf16_matmul(&input, &weight_bytes, 4, 2);
    for &v in &output {
        assert!(v.abs() < 0.01);
    }
}

#[test]
fn test_simd_bf16_matmul_large() {
    let in_dim = 128;
    let out_dim = 64;
    let input: Vec<f32> = (0..in_dim).map(|i| (i % 10) as f32 * 0.1).collect();

    // Create identity-like weight in BF16
    let mut weight_bytes = Vec::with_capacity(out_dim * in_dim * 2);
    for row in 0..out_dim {
        for col in 0..in_dim {
            let val = if row == col { 1.0 } else { 0.0 };
            weight_bytes.extend_from_slice(&half::bf16::from_f32(val).to_le_bytes());
        }
    }

    let output = simd_bf16_matmul(&input, &weight_bytes, in_dim, out_dim);
    assert_eq!(output.len(), out_dim);

    // First out_dim elements should approximately match input
    for i in 0..out_dim {
        let expected = (i % 10) as f32 * 0.1;
        assert!(
            (output[i] - expected).abs() < 0.05,
            "Index {}: expected {}, got {}",
            i,
            expected,
            output[i]
        );
    }
}

// ============================================================================
// Integration: Activation Chains
// ============================================================================

#[test]
fn test_matmul_silu_chain() {
    let input = vec![1.0, 2.0];
    let weight = vec![
        1.0, 1.0, // sum: 3
        1.0, -1.0, // diff: -1
    ];
    let mut output = simd_matmul(&input, &weight, 2, 2);
    simd_silu(&mut output);

    // silu(3) and silu(-1)
    assert!(output[0] > 2.5); // silu(3) > 2.5
    assert!(output[1] < 0.0); // silu(-1) < 0
}

#[test]
fn test_matmul_gelu_softmax_chain() {
    let input = vec![0.5, 0.5, 0.5, 0.5];
    let weight = vec![
        1.0, 1.0, 1.0, 1.0, // sum: 2
        1.0, 0.0, 0.0, 0.0, // just first: 0.5
        0.0, 0.0, 0.0, 1.0, // just last: 0.5
    ];
    let mut output = simd_matmul(&input, &weight, 4, 3);
    simd_gelu(&mut output);
    simd_softmax(&mut output);

    // Output should sum to 1
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // First element (gelu(2)) should have highest probability
    assert!(output[0] > output[1]);
    assert!(output[0] > output[2]);
}

#[test]
fn test_residual_with_activation() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![
        0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1,
    ];

    let proj = simd_matmul(&input, &weight, 4, 4);
    let mut proj_activated = proj.clone();
    simd_gelu(&mut proj_activated);

    let mut residual = input.clone();
    simd_add(&mut residual, &proj_activated);

    // Should be approximately input + gelu(0.1 * input)
    for (&r, &i) in residual.iter().zip(input.iter()) {
        assert!(r > i); // Residual should be larger than input
        assert!(r < i * 1.5); // But not too much larger
    }
}
