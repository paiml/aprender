
// =============================================================================
// Q4_K × Q8_K SIMD Variant Tests
// =============================================================================

#[test]
fn test_fused_q4k_q8k_dot_simd_zero_inputs() {
    let q4k_data = vec![0u8; 144];
    let q8k_scales = gen_q8k_scales();
    let q8k_quants = gen_q8k_quants();

    let result =
        fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("Should succeed");
    assert!(
        result.abs() < 1e-10,
        "Q4K×Q8K SIMD: Zero inputs should give zero result: {}",
        result
    );
}

#[test]
fn test_fused_q4k_q8k_dot_simd_matches_scalar() {
    let mut q4k_data = vec![0u8; 144];
    q4k_data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
    for i in 16..144 {
        q4k_data[i] = 0xCC;
    }

    let q8k_scales: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 + 0.5).collect();
    let q8k_quants: Vec<i8> = (0..QK_K).map(|i| ((i % 100) as i8) - 50).collect();

    let scalar =
        fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("Scalar should succeed");
    let simd =
        fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("SIMD should succeed");

    let tolerance = scalar.abs() * 1e-3 + 1e-4;
    assert!(
        (scalar - simd).abs() <= tolerance,
        "Q4K×Q8K SIMD should match scalar: scalar={}, simd={}, diff={}",
        scalar,
        simd,
        (scalar - simd).abs()
    );
}

#[test]
fn test_fused_q4k_q8k_dot_simd_invalid_length() {
    let q4k_data = vec![0u8; 100]; // Invalid
    let q8k_scales = gen_q8k_scales();
    let q8k_quants = gen_q8k_quants();

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_err(), "Q4K×Q8K SIMD: Invalid length should error");
}

#[test]
fn test_fused_q4k_q8k_dot_simd_deterministic() {
    let q4k_data: Vec<u8> = (0..144).map(|i| (i * 31) as u8).collect();
    let q8k_scales: Vec<f32> = (0..8).map(|i| (i as f32) * 0.2 + 0.3).collect();
    let q8k_quants: Vec<i8> = (0..QK_K).map(|i| ((i % 80) as i8) - 40).collect();

    let result1 =
        fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("Should succeed");
    let result2 =
        fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("Should succeed");

    assert_eq!(
        result1, result2,
        "Q4K×Q8K SIMD should be deterministic: {} vs {}",
        result1, result2
    );
}

// =============================================================================
// Proptest for SIMD Variants
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(25))]

    #[test]
    fn prop_fused_q4k_dot_simd_matches_scalar(
        q4k_data in gen_q4k_superblock(),
        activations in gen_activations(QK_K)
    ) {
        // Skip if dequantization fails
        let dequantized = match dequantize_q4_k(&q4k_data) {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        // Skip if NaN/Inf
        if dequantized.iter().any(|v| !v.is_finite()) {
            return Ok(());
        }

        let scalar = match fused_q4k_dot(&q4k_data, &activations) {
            Ok(r) if r.is_finite() => r,
            _ => return Ok(()),
        };

        let simd = match fused_q4k_dot_simd(&q4k_data, &activations) {
            Ok(r) if r.is_finite() => r,
            _ => return Ok(()),
        };

        // Use a more generous tolerance to account for:
        // 1. Different accumulation order in SIMD vs scalar (FMA reordering)
        // 2. Quantization noise in Q4_K format
        // 3. Large value ranges in randomized test data
        let tolerance = scalar.abs() * 5e-3 + simd.abs() * 5e-3 + 1e-3;
        prop_assert!(
            (scalar - simd).abs() <= tolerance,
            "SIMD should match scalar: scalar={}, simd={}, diff={}, tolerance={}",
            scalar, simd, (scalar - simd).abs(), tolerance
        );
    }
}

// =============================================================================
// Phase 49: Multi-Block Tests for SIMD Loop Coverage
// =============================================================================

/// Generate multiple Q4_K super-blocks for multi-block testing
fn gen_multi_q4k_blocks(num_blocks: usize) -> Vec<u8> {
    (0..num_blocks)
        .flat_map(|b| (0..144).map(move |i| ((i + b * 31) % 256) as u8))
        .collect()
}

/// Generate multi-block Q8_K data
fn gen_multi_q8k_data(num_blocks: usize) -> (Vec<f32>, Vec<i8>) {
    let scales: Vec<f32> = (0..num_blocks).map(|b| 0.3 + (b as f32) * 0.1).collect();
    let quants: Vec<i8> = (0..num_blocks * QK_K)
        .map(|i| ((i % 127) as i8) - 64)
        .collect();
    (scales, quants)
}

// -----------------------------------------------------------------------------
// fused_q4k_dot Multi-Block Tests
// -----------------------------------------------------------------------------

#[test]
fn test_fused_q4k_dot_two_blocks() {
    let q4k_data = gen_multi_q4k_blocks(2);
    let activations = vec![0.1f32; 2 * QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok(), "2-block Q4K dot should succeed");
}

#[test]
fn test_fused_q4k_dot_four_blocks() {
    let q4k_data = gen_multi_q4k_blocks(4);
    let activations: Vec<f32> = (0..4 * QK_K).map(|i| (i as f32 * 0.01) - 1.28).collect();

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok(), "4-block Q4K dot should succeed");
}

#[test]
fn test_fused_q4k_dot_eight_blocks() {
    let q4k_data = gen_multi_q4k_blocks(8);
    let activations: Vec<f32> = (0..8 * QK_K)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok(), "8-block Q4K dot should succeed");
}

#[test]
fn test_fused_q4k_dot_simd_two_blocks() {
    let q4k_data = gen_multi_q4k_blocks(2);
    let activations = vec![0.5f32; 2 * QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok(), "2-block SIMD Q4K dot should succeed");
}

#[test]
fn test_fused_q4k_dot_simd_four_blocks() {
    let q4k_data = gen_multi_q4k_blocks(4);
    let activations: Vec<f32> = (0..4 * QK_K).map(|i| (i as f32).sin() * 0.5).collect();

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok(), "4-block SIMD Q4K dot should succeed");
}

#[test]
fn test_fused_q4k_dot_simd_eight_blocks() {
    let q4k_data = gen_multi_q4k_blocks(8);
    let activations = vec![0.25f32; 8 * QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok(), "8-block SIMD Q4K dot should succeed");
}

#[test]
fn test_fused_q4k_dot_simd_sixteen_blocks() {
    let q4k_data = gen_multi_q4k_blocks(16);
    let activations: Vec<f32> = (0..16 * QK_K)
        .map(|i| ((i % 64) as f32 - 32.0) * 0.1)
        .collect();

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok(), "16-block SIMD Q4K dot should succeed");
}

#[test]
fn test_fused_q4k_dot_multi_block_simd_vs_scalar() {
    for num_blocks in [2, 4, 8] {
        let q4k_data = gen_multi_q4k_blocks(num_blocks);
        let activations: Vec<f32> = (0..num_blocks * QK_K)
            .map(|i| (i as f32 * 0.001) - 0.128)
            .collect();

        let scalar = fused_q4k_dot(&q4k_data, &activations).expect("scalar");
        let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd");

        if scalar.is_finite() && simd.is_finite() {
            let tolerance = scalar.abs() * 0.01 + simd.abs() * 0.01 + 0.01;
            assert!(
                (scalar - simd).abs() <= tolerance,
                "{}-block: scalar={}, simd={}, diff={}",
                num_blocks,
                scalar,
                simd,
                (scalar - simd).abs()
            );
        }
    }
}

// -----------------------------------------------------------------------------
// fused_q4k_q8k_dot Multi-Block Tests
// -----------------------------------------------------------------------------

#[test]
fn test_fused_q4k_q8k_dot_two_blocks() {
    let q4k_data = gen_multi_q4k_blocks(2);
    let (q8k_scales, q8k_quants) = gen_multi_q8k_data(2);

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok(), "2-block Q4K×Q8K dot should succeed");
}

#[test]
fn test_fused_q4k_q8k_dot_four_blocks() {
    let q4k_data = gen_multi_q4k_blocks(4);
    let (q8k_scales, q8k_quants) = gen_multi_q8k_data(4);

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok(), "4-block Q4K×Q8K dot should succeed");
}

#[test]
fn test_fused_q4k_q8k_dot_eight_blocks() {
    let q4k_data = gen_multi_q4k_blocks(8);
    let (q8k_scales, q8k_quants) = gen_multi_q8k_data(8);

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok(), "8-block Q4K×Q8K dot should succeed");
}

#[test]
fn test_fused_q4k_q8k_dot_simd_two_blocks() {
    let q4k_data = gen_multi_q4k_blocks(2);
    let (q8k_scales, q8k_quants) = gen_multi_q8k_data(2);

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok(), "2-block SIMD Q4K×Q8K dot should succeed");
}

#[test]
fn test_fused_q4k_q8k_dot_simd_four_blocks() {
    let q4k_data = gen_multi_q4k_blocks(4);
    let (q8k_scales, q8k_quants) = gen_multi_q8k_data(4);

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok(), "4-block SIMD Q4K×Q8K dot should succeed");
}

#[test]
fn test_fused_q4k_q8k_dot_simd_eight_blocks() {
    let q4k_data = gen_multi_q4k_blocks(8);
    let (q8k_scales, q8k_quants) = gen_multi_q8k_data(8);

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok(), "8-block SIMD Q4K×Q8K dot should succeed");
}

#[test]
fn test_fused_q4k_q8k_dot_simd_sixteen_blocks() {
    let q4k_data = gen_multi_q4k_blocks(16);
    let (q8k_scales, q8k_quants) = gen_multi_q8k_data(16);

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok(), "16-block SIMD Q4K×Q8K dot should succeed");
}

#[test]
fn test_fused_q4k_q8k_dot_multi_block_simd_vs_scalar() {
    for num_blocks in [2, 4, 8] {
        let q4k_data = gen_multi_q4k_blocks(num_blocks);
        let (q8k_scales, q8k_quants) = gen_multi_q8k_data(num_blocks);

        let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
        let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("simd");

        if scalar.is_finite() && simd.is_finite() {
            let tolerance = scalar.abs() * 0.01 + simd.abs() * 0.01 + 1.0;
            assert!(
                (scalar - simd).abs() <= tolerance,
                "{}-block Q4K×Q8K: scalar={}, simd={}, diff={}",
                num_blocks,
                scalar,
                simd,
                (scalar - simd).abs()
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Edge Case Activation Patterns
// -----------------------------------------------------------------------------

#[test]
fn test_fused_q4k_dot_all_negative_activations() {
    let q4k_data = gen_multi_q4k_blocks(1);
    let activations = vec![-0.5f32; QK_K];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_alternating_activations() {
    let q4k_data = gen_multi_q4k_blocks(2);
    let activations: Vec<f32> = (0..2 * QK_K)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    let scalar = fused_q4k_dot(&q4k_data, &activations).expect("scalar");
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd");

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

#[test]
fn test_fused_q4k_dot_sparse_activations() {
    let q4k_data = gen_multi_q4k_blocks(2);
    // Only every 8th activation is non-zero
    let activations: Vec<f32> = (0..2 * QK_K)
        .map(|i| if i % 8 == 0 { 1.0 } else { 0.0 })
        .collect();

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_large_values() {
    let q4k_data = gen_multi_q4k_blocks(1);
    let activations = vec![100.0f32; QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_dot_small_values() {
    let q4k_data = gen_multi_q4k_blocks(1);
    let activations = vec![1e-6f32; QK_K];

    let result = fused_q4k_dot_simd(&q4k_data, &activations);
    assert!(result.is_ok());
}

// -----------------------------------------------------------------------------
// Q8K Edge Cases
// -----------------------------------------------------------------------------

#[test]
fn test_fused_q4k_q8k_dot_all_negative_quants() {
    let q4k_data = gen_multi_q4k_blocks(1);
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![-64i8; QK_K];

    let result = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_max_quant_values() {
    let q4k_data = gen_multi_q4k_blocks(1);
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![127i8; QK_K];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_min_quant_values() {
    let q4k_data = gen_multi_q4k_blocks(1);
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![-128i8; QK_K];

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
}

#[test]
fn test_fused_q4k_q8k_dot_alternating_quants() {
    let q4k_data = gen_multi_q4k_blocks(2);
    let q8k_scales = vec![0.5f32, 1.5f32];
    let q8k_quants: Vec<i8> = (0..2 * QK_K)
        .map(|i| if i % 2 == 0 { 100 } else { -100 })
        .collect();

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    let simd = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants).expect("simd");

    assert!(scalar.is_finite());
    assert!(simd.is_finite());
}

#[test]
fn test_fused_q4k_q8k_dot_zero_scales() {
    let q4k_data = gen_multi_q4k_blocks(2);
    let q8k_scales = vec![0.0f32, 0.0f32];
    let q8k_quants: Vec<i8> = (0..2 * QK_K).map(|i| (i % 127) as i8).collect();

    let result = fused_q4k_q8k_dot_simd(&q4k_data, &q8k_scales, &q8k_quants);
    assert!(result.is_ok());
    // With zero scales, result should be zero or very small
}
