
#[test]
fn test_fused_q4k_dot_multiple_super_blocks() {
    // RED: Test with multiple super-blocks (realistic model tensor)
    //
    // 4 super-blocks = 1024 values (small but representative)
    let num_super_blocks = 4;
    let mut q4k_data = Vec::with_capacity(num_super_blocks * 144);

    for sb_idx in 0..num_super_blocks {
        // Varied d values
        let d = 0.5 + (sb_idx as f32) * 0.1;
        q4k_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());

        // dmin = small value
        q4k_data.extend_from_slice(&half::f16::from_f32(0.1).to_bits().to_le_bytes());

        // scales: 12 bytes with varied patterns
        for i in 0..12 {
            q4k_data.push(((sb_idx * 7 + i) % 64) as u8);
        }

        // qs: 128 bytes with varied patterns
        for i in 0..128 {
            q4k_data.push(((sb_idx * 13 + i) % 256) as u8);
        }
    }

    // Activations: random-ish pattern
    let activations: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.017).sin() * 2.0).collect();

    // Reference
    let dequantized = dequantize_q4_k(&q4k_data).expect("test");
    let reference = naive_dot_product(&dequantized, &activations);

    // Fused
    let fused = fused_q4k_dot(&q4k_data, &activations).expect("test");

    assert_ulp_eq(fused, reference, 4, "fused_q4k_dot multiple super-blocks");
}

#[test]
fn test_fused_q4k_dot_edge_values() {
    // RED: Test edge cases per Goldberg [9]
    // - All zeros
    // - Maximum quantized values
    // - Negative activations

    // Test 1: All zeros
    let mut q4k_zeros = Vec::new();
    q4k_zeros.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
    q4k_zeros.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
    q4k_zeros.extend_from_slice(&[0u8; 12]); // scales
    q4k_zeros.extend_from_slice(&[0u8; 128]); // qs

    let activations_zeros: Vec<f32> = vec![1.0; 256];
    let fused_zeros = fused_q4k_dot(&q4k_zeros, &activations_zeros).expect("test");
    assert!(
        fused_zeros.abs() < 1e-6,
        "Zero weights should produce zero dot product"
    );

    // Test 2: Maximum scale values
    let mut q4k_max = Vec::new();
    q4k_max.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    q4k_max.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
    q4k_max.extend_from_slice(&[0xFF; 12]); // max scales
    q4k_max.extend_from_slice(&[0xFF; 128]); // max qs (all 15s)

    let activations_ones: Vec<f32> = vec![1.0; 256];
    let dequantized_max = dequantize_q4_k(&q4k_max).expect("test");
    let reference_max = naive_dot_product(&dequantized_max, &activations_ones);
    let fused_max = fused_q4k_dot(&q4k_max, &activations_ones).expect("test");

    assert_ulp_eq(fused_max, reference_max, 4, "fused_q4k_dot max values");

    // Test 3: Negative activations
    let activations_neg: Vec<f32> = (0..256).map(|i| -((i as f32) * 0.01)).collect();
    let dequantized_neg = dequantize_q4_k(&q4k_max).expect("test");
    let reference_neg = naive_dot_product(&dequantized_neg, &activations_neg);
    let fused_neg = fused_q4k_dot(&q4k_max, &activations_neg).expect("test");

    assert_ulp_eq(
        fused_neg,
        reference_neg,
        4,
        "fused_q4k_dot negative activations",
    );
}

#[test]
fn test_fused_q4k_dot_length_mismatch() {
    // RED: Error handling for mismatched lengths
    let q4k_data = vec![0u8; 144]; // 1 super-block = 256 values
    let activations = vec![0.0f32; 128]; // Wrong length!

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(
        result.is_err(),
        "Should error on activation length mismatch"
    );
}

#[test]
fn test_fused_q4k_dot_invalid_data_length() {
    // RED: Error handling for invalid quantized data
    let q4k_data = vec![0u8; 143]; // Not a multiple of 144
    let activations = vec![0.0f32; 256];

    let result = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_err(), "Should error on invalid Q4_K data length");
}

#[test]
fn test_fused_q4k_dot_no_intermediate_allocation() {
    // RED: Verify fused operation doesn't allocate intermediate f32 buffer
    //
    // This is a performance contract test - the fused function signature
    // should NOT return a Vec<f32> intermediate, only the final scalar.
    //
    // We verify by checking the function returns f32 directly, not a tuple
    // or struct containing intermediate results.

    let q4k_data = vec![0u8; 144];
    let activations = vec![0.0f32; 256];

    // Type assertion: fused_q4k_dot returns Result<f32>, not Result<(Vec<f32>, f32)>
    let result: Result<f32> = fused_q4k_dot(&q4k_data, &activations);
    assert!(result.is_ok());

    // The function signature enforces no intermediate - this test documents the contract
}

// -------------------------------------------------------------------------
// Q6_K Fused Dequant+Dot Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q6k_dot_basic() {
    // RED: Test fused Q6_K dequant+dot against reference implementation
    //
    // Q6_K layout: ql (128) + qh (64) + scales (16) + d (2) = 210 bytes
    let mut q6k_data = Vec::new();

    // ql: 128 bytes (low 4 bits)
    for i in 0..128 {
        q6k_data.push((i % 16) as u8 | (((i + 1) % 16) as u8) << 4);
    }

    // qh: 64 bytes (high 2 bits)
    for i in 0..64 {
        q6k_data.push((i % 4) as u8 | (((i + 1) % 4) as u8) << 2);
    }

    // scales: 16 bytes (i8)
    for i in 0..16 {
        q6k_data.push((i as i8 - 8) as u8);
    }

    // d = 1.0 (f16)
    q6k_data.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());

    // Activations
    let activations: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

    // Reference
    let dequantized = dequantize_q6_k(&q6k_data).expect("test");
    let reference = naive_dot_product(&dequantized, &activations);

    // Fused
    let fused = fused_q6k_dot(&q6k_data, &activations).expect("test");

    assert_ulp_eq(fused, reference, 4, "fused_q6k_dot basic");
}

#[test]
fn test_fused_q6k_dot_multiple_super_blocks() {
    // RED: Test with multiple super-blocks
    let num_super_blocks = 4;
    let mut q6k_data = Vec::with_capacity(num_super_blocks * 210);

    for sb_idx in 0..num_super_blocks {
        // ql: 128 bytes
        for i in 0..128 {
            q6k_data.push(((sb_idx * 7 + i) % 256) as u8);
        }

        // qh: 64 bytes
        for i in 0..64 {
            q6k_data.push(((sb_idx * 11 + i) % 256) as u8);
        }

        // scales: 16 bytes (i8)
        for i in 0..16 {
            #[allow(clippy::cast_possible_wrap)]
            let scale = ((sb_idx * 3 + i) % 128) as i8;
            q6k_data.push(scale as u8);
        }

        // d with variation
        let d = 0.5 + (sb_idx as f32) * 0.2;
        q6k_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    }

    // Activations
    let activations: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.023).cos() * 1.5).collect();

    // Reference
    let dequantized = dequantize_q6_k(&q6k_data).expect("test");
    let reference = naive_dot_product(&dequantized, &activations);

    // Fused
    let fused = fused_q6k_dot(&q6k_data, &activations).expect("test");

    assert_ulp_eq(fused, reference, 4, "fused_q6k_dot multiple super-blocks");
}

#[test]
fn test_fused_q6k_dot_length_mismatch() {
    // RED: Error handling
    let q6k_data = vec![0u8; 210]; // 1 super-block = 256 values
    let activations = vec![0.0f32; 128]; // Wrong length!

    let result = fused_q6k_dot(&q6k_data, &activations);
    assert!(
        result.is_err(),
        "Should error on activation length mismatch"
    );
}

// -------------------------------------------------------------------------
// SIMD-Accelerated Fused Operations Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_dot_simd_matches_scalar() {
    // Test that SIMD version produces same results as scalar within 4 ULPs
    // This verifies correctness of AVX2 implementation (or fallback to scalar)

    // Generate varied test data
    let num_super_blocks = 4;
    let mut q4k_data = Vec::with_capacity(num_super_blocks * 144);

    for sb_idx in 0..num_super_blocks {
        // Varied d values
        let d = 0.5 + (sb_idx as f32) * 0.1;
        q4k_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());

        // dmin
        q4k_data.extend_from_slice(&half::f16::from_f32(0.1).to_bits().to_le_bytes());

        // scales: 12 bytes with varied patterns
        for i in 0..12 {
            q4k_data.push(((sb_idx * 7 + i) % 64) as u8);
        }

        // qs: 128 bytes with varied patterns
        for i in 0..128 {
            q4k_data.push(((sb_idx * 13 + i) % 256) as u8);
        }
    }

    // Activations
    let activations: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.017).sin() * 2.0).collect();

    // Get scalar result (reference)
    let scalar_result = fused_q4k_dot(&q4k_data, &activations).expect("test");

    // Get SIMD result (may use AVX2 or fall back to scalar)
    let simd_result = fused_q4k_dot_simd(&q4k_data, &activations).expect("test");

    // Should match within 8 ULPs (allowing for FMA reassociation in SIMD)
    // Per Goldberg [9], SIMD accumulation reordering can cause slightly more divergence
    assert_ulp_eq(
        simd_result,
        scalar_result,
        8,
        "SIMD result should match scalar within 8 ULPs",
    );
}

#[test]
fn test_fused_q4k_dot_simd_error_handling() {
    // Verify SIMD version has same error handling as scalar

    // Invalid data length
    let bad_data = vec![0u8; 143]; // Not multiple of 144
    let activations = vec![0.0f32; 256];
    assert!(fused_q4k_dot_simd(&bad_data, &activations).is_err());

    // Mismatched activation length
    let good_data = vec![0u8; 144];
    let bad_activations = vec![0.0f32; 128];
    assert!(fused_q4k_dot_simd(&good_data, &bad_activations).is_err());
}

#[test]
fn test_fused_q4k_dot_simd_large_input() {
    // Test with larger input to stress SIMD path
    // 16 super-blocks = 4096 values (2304 bytes)

    let num_super_blocks = 16;
    let mut q4k_data = Vec::with_capacity(num_super_blocks * 144);

    for sb_idx in 0..num_super_blocks {
        // d with variation
        let d = 1.0 + (sb_idx as f32) * 0.05;
        q4k_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());

        // dmin = 0.0
        q4k_data.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());

        // scales
        for i in 0..12 {
            q4k_data.push(((sb_idx + i) % 64) as u8);
        }

        // qs with varied patterns
        for i in 0..128 {
            q4k_data.push(((sb_idx * 17 + i * 3) % 256) as u8);
        }
    }

    // Large activation vector
    let activations: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.001).cos()).collect();

    // Get reference from dequantize + naive dot
    let dequantized = dequantize_q4_k(&q4k_data).expect("test");
    let reference = naive_dot_product(&dequantized, &activations);

    // SIMD result
    let simd_result = fused_q4k_dot_simd(&q4k_data, &activations).expect("test");

    // Allow slightly more ULP tolerance for larger accumulations
    // due to floating-point associativity differences
    let ulp_d = ulp_diff(simd_result, reference);
    assert!(
        ulp_d <= 16,
        "Large input SIMD result should match reference: simd={}, ref={}, ulp_diff={}",
        simd_result,
        reference,
        ulp_d
    );
}

// -------------------------------------------------------------------------
// Phase 2: L2-Aware Tiled Matrix-Vector Multiplication Tests
// -------------------------------------------------------------------------

#[test]
fn test_fused_q4k_tiled_matvec_basic() {
    // RED: Test tiled matvec produces same results as sequential dot products
    use crate::quantize::fused_q4k_tiled_matvec;

    // Setup: 4 output dimensions, 256 input dimensions (1 super-block per row)
    let in_dim = 256;
    let out_dim = 4;

    // Create weight data: 4 rows × 144 bytes = 576 bytes
    let mut weight_data = Vec::with_capacity(out_dim * 144);
    for row in 0..out_dim {
        // d with variation
        let d = 0.5 + (row as f32) * 0.1;
        weight_data.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        // dmin
        weight_data.extend_from_slice(&half::f16::from_f32(0.05).to_bits().to_le_bytes());
        // scales
        for i in 0..12 {
            weight_data.push(((row * 7 + i) % 64) as u8);
        }
        // qs
        for i in 0..128 {
            weight_data.push(((row * 13 + i) % 256) as u8);
        }
    }

    // Activations
    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Reference: compute each output using individual dot products
    let mut reference = Vec::with_capacity(out_dim);
    for row in 0..out_dim {
        let row_start = row * 144;
        let row_data = &weight_data[row_start..row_start + 144];
        let dot = fused_q4k_dot_simd(row_data, &activations).expect("test");
        reference.push(dot);
    }

    // Tiled result
    let tiled =
        fused_q4k_tiled_matvec(&weight_data, &activations, in_dim, out_dim, None).expect("test");

    // Compare
    assert_eq!(tiled.len(), out_dim);
    for i in 0..out_dim {
        assert_ulp_eq(
            tiled[i],
            reference[i],
            4,
            &format!("tiled_matvec output {}", i),
        );
    }
}
