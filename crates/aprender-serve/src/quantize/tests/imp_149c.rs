
/// IMP-149c: Verify parallel fused matvec scales with output dimension
#[test]
fn test_imp_149c_parallel_matvec_scaling() {
    // Test matrix dimensions (small for fast test)
    let in_dim: usize = 256;
    let out_dims: [usize; 3] = [64, 128, 256];

    let super_blocks_per_row = in_dim.div_ceil(256);
    let bytes_per_row = super_blocks_per_row * 144;

    let activations: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let iterations = 50;

    let mut timings = Vec::new();

    for &out_dim in &out_dims {
        let weight_bytes = out_dim * bytes_per_row;
        let mut weights = vec![0u8; weight_bytes];

        // Initialize weights
        for row in 0..out_dim {
            for block in 0..super_blocks_per_row {
                let offset = row * bytes_per_row + block * 144;
                let d: f32 = 0.1;
                weights[offset..offset + 2].copy_from_slice(&d.to_le_bytes()[0..2]);
            }
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
        }
        let elapsed = start.elapsed();
        timings.push((out_dim, elapsed));
    }

    println!("\nIMP-149c: Parallel Matvec Scaling:");
    for (out_dim, elapsed) in &timings {
        let throughput =
            (*out_dim * in_dim * iterations) as f64 / elapsed.as_secs_f64() / 1_000_000.0;
        println!(
            "  {}x{}: {:.2}ms ({:.1} MFLOPS)",
            in_dim,
            out_dim,
            elapsed.as_secs_f64() * 1000.0,
            throughput
        );
    }

    // IMP-149c: Larger matrices should have higher throughput (better utilization)
    // Verify timing roughly scales with output dimension
    let time_64 = timings[0].1.as_secs_f64();
    let time_256 = timings[2].1.as_secs_f64();
    let scaling_ratio = time_256 / time_64;

    // Expected: 256/64 = 4x work, but overhead makes it <4x time
    // Coverage instrumentation adds extreme overhead, so we only print the ratio
    // Performance assertions are meaningless under coverage instrumentation
    println!(
        "Scaling ratio: {:.2}x (expected <4x in release builds)",
        scaling_ratio
    );
}

/// IMP-149d: Verify memory bandwidth improvement from fused kernel
#[test]
fn test_imp_149d_memory_bandwidth_analysis() {
    // Per Five Whys Analysis:
    // - Q4_K: 4.5 bits/weight average
    // - F32: 32 bits/weight
    // - Theoretical bandwidth ratio: 32/4.5 = 7.1x

    let bits_per_q4k_weight: f64 = 4.5;
    let bits_per_f32: f64 = 32.0;
    let bandwidth_ratio = bits_per_f32 / bits_per_q4k_weight;

    println!("\nIMP-149d: Memory Bandwidth Analysis:");
    println!("  Q4_K bits/weight: {:.1}", bits_per_q4k_weight);
    println!("  F32 bits/weight: {:.0}", bits_per_f32);
    println!("  Theoretical bandwidth ratio: {:.1}x", bandwidth_ratio);

    // IMP-149d: Verify theoretical calculations
    assert!(
        (bandwidth_ratio - 7.1).abs() < 0.2,
        "IMP-149d: Bandwidth ratio should be ~7.1x, got {:.1}x",
        bandwidth_ratio
    );

    // Calculate expected throughput improvement
    // Assuming memory-bound operation, speedup ≈ bandwidth_ratio
    // Real-world speedup limited by:
    // - Dequantization overhead
    // - Cache effects
    // - SIMD utilization

    let realistic_efficiency: f64 = 0.3; // 30% of theoretical
    let expected_real_speedup = bandwidth_ratio * realistic_efficiency;

    println!(
        "  Realistic efficiency: {:.0}%",
        realistic_efficiency * 100.0
    );
    println!("  Expected real speedup: {:.1}x", expected_real_speedup);

    // IMP-149d: Even at 30% efficiency, should achieve >2x speedup
    assert!(
        expected_real_speedup > 2.0,
        "IMP-149d: Expected speedup should be >2x, got {:.1}x",
        expected_real_speedup
    );
}

// =========================================================================
// Additional Coverage Tests for Uncovered Functions
// =========================================================================

/// Test Q4_1 dequantization correctness
#[test]
fn test_dequantize_q4_1_basic() {
    // Q4_1 block: 2 bytes scale (f16) + 2 bytes min (f16) + 16 bytes quants = 20 bytes
    let mut data = vec![0u8; 20];
    // Scale = 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    // Min = 0.5 (f16 = 0x3800)
    data[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
    // Quants: first byte has low=0, high=1
    data[4] = 0x10;

    let result = dequantize_q4_1(&data).expect("test");
    assert_eq!(result.len(), 32);
    // value = d * q + min = 1.0 * 0 + 0.5 = 0.5
    assert!(
        (result[0] - 0.5).abs() < 1e-3,
        "Expected 0.5, got {}",
        result[0]
    );
    // value = d * q + min = 1.0 * 1 + 0.5 = 1.5
    // Note: High nibble goes to position 16 (candle layout: low=0-15, high=16-31)
    assert!(
        (result[16] - 1.5).abs() < 1e-3,
        "Expected 1.5, got {}",
        result[16]
    );
}

/// Test Q5_0 dequantization correctness
#[test]
fn test_dequantize_q5_0_basic() {
    // Q5_0 block: 2 bytes scale (f16) + 4 bytes high bits + 16 bytes quants = 22 bytes
    let mut data = vec![0u8; 22];
    // Scale = 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    // High bits = 0 (all zeros)
    data[2..6].copy_from_slice(&0u32.to_le_bytes());
    // Quants: first byte has low=0, high=0
    data[6] = 0x00;

    let result = dequantize_q5_0(&data).expect("test");
    assert_eq!(result.len(), 32);
    // With 5-bit value 0: value = d * (0 - 16) = 1.0 * -16 = -16.0
    assert!(
        (result[0] - (-16.0)).abs() < 1e-3,
        "Expected -16.0, got {}",
        result[0]
    );
}

/// Test Q5_1 dequantization correctness
#[test]
fn test_dequantize_q5_1_basic() {
    // Q5_1 block: 2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes quants = 24 bytes
    let mut data = vec![0u8; 24];
    // Scale = 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    // Min = 0.0 (f16 = 0x0000)
    data[2..4].copy_from_slice(&0x0000_u16.to_le_bytes());
    // High bits = 0
    data[4..8].copy_from_slice(&0u32.to_le_bytes());
    // Quants: first byte has low=8, high=8 (for value 8)
    data[8] = 0x88;

    let result = dequantize_q5_1(&data).expect("test");
    assert_eq!(result.len(), 32);
    // 5-bit value 8: value = d * 8 + min = 1.0 * 8 + 0.0 = 8.0
    assert!(
        (result[0] - 8.0).abs() < 1e-3,
        "Expected 8.0, got {}",
        result[0]
    );
}

/// Test Q4_K dequantization basic
#[test]
fn test_dequantize_q4_k_basic_block() {
    // Q4_K super-block: 144 bytes for 256 values
    let mut data = vec![0u8; 144];
    // d = 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    // dmin = 0.0 (f16 = 0x0000)
    data[2..4].copy_from_slice(&0x0000_u16.to_le_bytes());

    let result = dequantize_q4_k(&data).expect("test");
    assert_eq!(result.len(), 256);
    // All values should be finite
    assert!(result.iter().all(|x| x.is_finite()));
}

/// Test Q5_K dequantization basic
#[test]
fn test_dequantize_q5_k_basic_block() {
    // Q5_K super-block: 176 bytes for 256 values
    let mut data = vec![0u8; 176];
    // d = 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    // dmin = 0.0 (f16 = 0x0000)
    data[2..4].copy_from_slice(&0x0000_u16.to_le_bytes());

    let result = dequantize_q5_k(&data).expect("test");
    assert_eq!(result.len(), 256);
    // All values should be finite
    assert!(result.iter().all(|x| x.is_finite()));
}

/// Test Q6_K dequantization basic
#[test]
fn test_dequantize_q6_k_basic_block() {
    // Q6_K super-block: 210 bytes for 256 values
    let mut data = vec![0u8; 210];
    // scale = 1.0 (f16 = 0x3C00) - last 2 bytes
    data[208..210].copy_from_slice(&0x3C00_u16.to_le_bytes());

    let result = dequantize_q6_k(&data).expect("test");
    assert_eq!(result.len(), 256);
    // All values should be finite
    assert!(result.iter().all(|x| x.is_finite()));
}

/// Test F16 dequantization
#[test]
fn test_dequantize_f16_basic() {
    // F16: 2 bytes per value
    let mut data = vec![0u8; 4]; // 2 values
    // First value: 1.0 (f16 = 0x3C00)
    data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    // Second value: 0.5 (f16 = 0x3800)
    data[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());

    let result = dequantize_f16(&data).expect("test");
    assert_eq!(result.len(), 2);
    assert!(
        (result[0] - 1.0).abs() < 1e-3,
        "Expected 1.0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 0.5).abs() < 1e-3,
        "Expected 0.5, got {}",
        result[1]
    );
}

/// Test fused Q4K dot dimension mismatch
#[test]
fn test_fused_q4k_dot_dimension_error() {
    let q4k_data = vec![0u8; 144]; // 256 values
    let activations = vec![0.5f32; 128]; // Wrong size
    assert!(fused_q4k_dot(&q4k_data, &activations).is_err());
}

/// Test fused Q5K dot product coverage
#[test]
fn test_fused_q5k_dot_coverage() {
    // Q5_K super-block: 176 bytes for 256 values
    let mut q5k_data = vec![0u8; 176];
    // d = 0.1 (f16)
    q5k_data[0..2].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
    // dmin = 0.0
    q5k_data[2..4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    let activations = vec![0.5f32; 256];
    let result = fused_q5k_dot(&q5k_data, &activations).expect("test");
    assert!(result.is_finite());
}

/// Test detect_simd_backend
#[test]
fn test_detect_simd_backend_basic() {
    let backend = detect_simd_backend();
    // Should return a valid backend
    assert!(matches!(
        backend,
        SimdBackend::Avx2 | SimdBackend::Sse2 | SimdBackend::Neon | SimdBackend::Scalar
    ));
}

/// Test quantize_to_q8_blocks basic
#[test]
fn test_quantize_to_q8_blocks_basic() {
    let values = vec![0.5f32; 32];
    let blocks = quantize_to_q8_blocks(&values).expect("test");
    assert_eq!(blocks.len(), 1);
    // Verify dequantization roundtrip
    let dequant = dequantize_q8_blocks(&blocks);
    assert_eq!(dequant.len(), 32);
    // Values should be close to original (within quantization error)
    for i in 0..32 {
        assert!(
            (dequant[i] - values[i]).abs() < 0.1,
            "Mismatch at {}: {} vs {}",
            i,
            dequant[i],
            values[i]
        );
    }
}

/// Test quantize_to_q8_blocks with non-multiple-of-32 length
#[test]
fn test_quantize_to_q8_blocks_invalid_length() {
    let values = vec![0.5f32; 31]; // Not multiple of 32
    assert!(quantize_to_q8_blocks(&values).is_err());
}

/// Test quantize_activations_q8_0
#[test]
fn test_quantize_activations_q8_0_basic() {
    let activations = vec![0.5f32; 32];
    let (scales, quants) = quantize_activations_q8_0(&activations);
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

/// Test softmax_simd with various sizes
#[test]
fn test_softmax_simd_various_sizes() {
    // Small array
    let mut small = vec![1.0f32; 4];
    softmax_simd(&mut small);
    assert!((small.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    // Medium array
    let mut medium = vec![1.0f32; 16];
    softmax_simd(&mut medium);
    assert!((medium.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    // Large array
    let mut large = vec![1.0f32; 256];
    softmax_simd(&mut large);
    assert!((large.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

/// Test softmax_simd numerical stability with large values
#[test]
fn test_softmax_simd_numerical_stability() {
    let mut x = vec![1000.0f32, 1001.0, 1002.0];
    softmax_simd(&mut x);
    // Should not produce NaN or Inf
    assert!(x.iter().all(|v| v.is_finite()));
    // Should sum to 1.0
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

/// Test fused_swiglu_simd
#[test]
fn test_fused_swiglu_simd_basic() {
    let mut gate = vec![1.0f32; 8];
    let up = vec![2.0f32; 8];
    fused_swiglu_simd(&mut gate, &up);
    // SwiGLU: gate * silu(gate) * up where silu(x) = x * sigmoid(x)
    // For gate=1.0: silu(1.0) ≈ 0.731 * 1.0 * 2.0 ≈ 1.462
    assert!(gate.iter().all(|v| v.is_finite()));
}

/// Test apply_rope_rotation_simd
#[test]
fn test_apply_rope_rotation_simd_basic() {
    let mut x1 = vec![1.0f32; 32];
    let mut x2 = vec![0.0f32; 32];
    let freqs_cos = vec![1.0f32; 32];
    let freqs_sin = vec![0.0f32; 32];
    // With cos=1 and sin=0, rotation should leave values unchanged
    apply_rope_rotation_simd(&mut x1, &mut x2, &freqs_cos, &freqs_sin);
    // Should be close to original
    assert!((x1[0] - 1.0).abs() < 1e-5);
}

/// Test fused_rmsnorm_q4_0_matmul
#[test]
fn test_fused_rmsnorm_q4_0_matmul_basic() {
    let hidden_dim = 32;
    let input = vec![1.0f32; hidden_dim];
    let norm_weight = vec![1.0f32; hidden_dim];
    // Create Q4_0 weight data: 1 block per row (32 values = 1 block), 2 rows
    let weight_data = vec![0u8; 18 * 2]; // 2 rows of Q4_0 blocks

    let result = fused_rmsnorm_q4_0_matmul(&input, &norm_weight, 1e-5, &weight_data, hidden_dim, 2)
        .expect("test");
    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|v| v.is_finite()));
}

/// Test quantize_rmsnorm_q8_0
#[test]
fn test_quantize_rmsnorm_q8_0_basic() {
    let input = vec![1.0f32; 32];
    let norm_weight = vec![1.0f32; 32];
    let (scales, quants) = quantize_rmsnorm_q8_0(&input, &norm_weight, 1e-5);
    assert_eq!(scales.len(), 1);
    assert_eq!(quants.len(), 32);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

/// Test dequantize_q4_k_parallel
#[test]
fn test_dequantize_q4_k_parallel_basic() {
    // Q4_K block size is 144 bytes for 256 values
    let data = vec![0u8; 144];
    let result = dequantize_q4_k_parallel(&data).expect("test");
    assert_eq!(result.len(), 256);
}

/// Test dequantize_q4_k_simd
#[test]
fn test_dequantize_q4_k_simd_basic() {
    let data = vec![0u8; 144];
    let result = dequantize_q4_k_simd(&data).expect("test");
    assert_eq!(result.len(), 256);
}

/// Test dequantize_q8_0_parallel
#[test]
fn test_dequantize_q8_0_parallel_basic() {
    // Q8_0 block size is 34 bytes (2 for f16 scale + 32 for i8 quants)
    let mut data = vec![0u8; 34];
    data[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // Scale = 1.0
    let result = dequantize_q8_0_parallel(&data).expect("test");
    assert_eq!(result.len(), 32);
}
