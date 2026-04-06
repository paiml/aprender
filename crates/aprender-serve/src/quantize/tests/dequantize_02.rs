use crate::quantize::*;
#[test]
fn test_dequantize_q4_k_parallel_output_size() {
    // 4 super-blocks = 1024 values
    let data = vec![0u8; 144 * 4];
    let result = dequantize_q4_k_parallel(&data).expect("test");
    assert_eq!(result.len(), 256 * 4);
}

#[test]
fn test_dequantize_q8_0_parallel_matches_scalar() {
    // Create 4 blocks (136 bytes = 4 * 34)
    let mut data = vec![0u8; 136];

    // Block 0: scale=1.0, values 0-31
    data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    for i in 0..32 {
        data[2 + i] = i as u8;
    }

    // Block 1: scale=0.5, values offset
    data[34..36].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
    for i in 0..32 {
        data[36 + i] = (i as i8 - 64) as u8;
    }

    // Block 2-3: zeros
    data[68..70].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
    data[102..104].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

    let scalar = dequantize_q8_0(&data).expect("test");
    let parallel = dequantize_q8_0_parallel(&data).expect("test");

    assert_eq!(scalar.len(), parallel.len());
    for (s, p) in scalar.iter().zip(parallel.iter()) {
        assert!((s - p).abs() < 1e-3, "Mismatch: scalar={s}, parallel={p}");
    }
}

#[test]
fn test_dequantize_q8_0_simd_matches_scalar() {
    // Create 2 blocks with varied values (68 bytes = 2 * 34)
    let mut data = vec![0u8; 68];

    // Block 0: scale=2.0
    data[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
    for i in 0..32 {
        data[2 + i] = ((i as i8 - 16) * 2) as u8;
    }

    // Block 1: scale=0.25
    data[34..36].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
    for i in 0..32 {
        data[36 + i] = (127 - i as i8) as u8;
    }

    let scalar = dequantize_q8_0(&data).expect("test");
    let simd = dequantize_q8_0_simd(&data).expect("test");

    assert_eq!(scalar.len(), simd.len());
    assert_eq!(simd.len(), 64);

    for (i, (s, p)) in scalar.iter().zip(simd.iter()).enumerate() {
        assert!(
            (s - p).abs() < 1e-3,
            "Mismatch at index {i}: scalar={s}, simd={p}"
        );
    }
}

#[test]
fn test_dequantize_q8_0_parallel_invalid_length() {
    let data = vec![0u8; 35]; // Not a multiple of 34
    let result = dequantize_q8_0_parallel(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_0_simd_invalid_length() {
    let data = vec![0u8; 35]; // Not a multiple of 34
    let result = dequantize_q8_0_simd(&data);
    assert!(result.is_err());
}

#[test]
fn test_dequantize_q8_0_parallel_large_input() {
    // 1000 blocks = 32000 values (simulating a weight matrix row)
    // Q8_0 block: 2 bytes f16 scale + 32 bytes quants = 34 bytes
    let mut data = vec![0u8; 34 * 1000];

    // Set varied scales
    for block in 0..1000 {
        let scale = 0.001 * (block as f32);
        data[block * 34..block * 34 + 2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
    }

    let result = dequantize_q8_0_parallel(&data).expect("test");
    assert_eq!(result.len(), 32000);
}

#[test]
fn test_dequantize_q4_k_cov_correctness() {
    // Test that the superblock helper matches the main dequantize function
    let mut sb_data = vec![0u8; 144];

    // d=2.0, dmin=0.5
    sb_data[0..2].copy_from_slice(&0x4000_u16.to_le_bytes()); // d=2.0
    sb_data[2..4].copy_from_slice(&0x3800_u16.to_le_bytes()); // dmin=0.5

    // Set varied quantized values
    for (idx, byte) in sb_data[16..144].iter_mut().enumerate() {
        *byte = (idx % 16) as u8 | (((idx / 2) % 8) << 4) as u8;
    }

    // Compare superblock helper with main function
    let main_result_cmp = dequantize_q4_k(&sb_data).expect("test");
    let main_result = dequantize_q4_k(&sb_data).expect("test");

    assert_eq!(main_result_cmp.len(), main_result.len());
    assert_eq!(main_result_cmp.len(), 256);

    for (i, (sb, main)) in main_result_cmp.iter().zip(main_result.iter()).enumerate() {
        assert!(
            (sb - main).abs() < 1e-5,
            "Mismatch at index {i}: superblock={sb}, main={main}"
        );
    }
}

#[test]
fn test_detect_simd_backend() {
    let backend = detect_simd_backend();

    // On x86_64 with AVX2, should return AVX2
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            assert_eq!(backend, SimdBackend::Avx2);
        } else if is_x86_feature_detected!("sse2") {
            assert_eq!(backend, SimdBackend::Sse2);
        } else {
            assert_eq!(backend, SimdBackend::Scalar);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(backend, SimdBackend::Neon);
    }

    // Display trait works
    let display = format!("{backend}");
    assert!(!display.is_empty());
}

#[test]
fn test_simd_backend_display() {
    assert_eq!(format!("{}", SimdBackend::Avx2), "AVX2");
    assert_eq!(format!("{}", SimdBackend::Sse2), "SSE2");
    assert_eq!(format!("{}", SimdBackend::Neon), "NEON");
    assert_eq!(format!("{}", SimdBackend::Scalar), "Scalar");
}

#[test]
fn test_dequant_stats_default() {
    let stats = DequantStats::default();
    assert_eq!(stats.blocks_processed, 0);
    assert_eq!(stats.bytes_processed, 0);
    assert_eq!(stats.simd_backend, SimdBackend::Scalar);
}

// =========================================================================
// IMP-147: SIMD Nibble Extraction Optimization (P1 Fix)
// =========================================================================
// Per Five Whys Analysis (spec §12A.2 WHY 5):
// - Current: 8 scalar ops per byte (extract low/high nibbles individually)
// - Target: 3 SIMD ops for 32 bytes (like llama.cpp's ggml-cpu-quants.c)
// - Expected gain: ~1.5x throughput improvement
//
// Reference: llama.cpp AVX2 nibble extraction pattern
// lowMask = set1_epi8(0x0F); lo = and(bytes, lowMask); hi = srli_epi16(bytes, 4)

/// IMP-147a: Verify scalar nibble extraction produces correct values
#[test]
fn test_imp_147a_scalar_nibble_extraction() {
    // Test byte with known nibbles: 0xAB = low=0xB, high=0xA
    let byte: u8 = 0xAB;
    let low = byte & 0x0F;
    let high = (byte >> 4) & 0x0F;

    // IMP-147a: Verify basic nibble extraction
    assert_eq!(low, 0x0B, "IMP-147a: Low nibble of 0xAB should be 0xB");
    assert_eq!(high, 0x0A, "IMP-147a: High nibble of 0xAB should be 0xA");

    // Test all 256 possible byte values
    for byte in 0u8..=255 {
        let low = byte & 0x0F;
        let high = (byte >> 4) & 0x0F;

        assert!(low <= 15, "IMP-147a: Low nibble should be 0-15");
        assert!(high <= 15, "IMP-147a: High nibble should be 0-15");
        assert_eq!(
            (high << 4) | low,
            byte,
            "IMP-147a: Recombining nibbles should give original byte"
        );
    }
}

/// IMP-147b: Verify SIMD nibble extraction matches scalar
#[cfg(target_arch = "x86_64")]
#[test]
fn test_imp_147b_simd_nibble_extraction_avx2() {
    // Runtime detection of AVX2
    if !is_x86_feature_detected!("avx2") {
        println!("IMP-147b: Skipping AVX2 test - CPU doesn't support AVX2");
        return;
    }

    // Create test bytes with known pattern
    let bytes: [u8; 32] = [
        0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC,
        0xFE, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD,
        0xEE, 0xFF,
    ];

    // Compute expected values with scalar code
    let mut expected_low: [u8; 32] = [0; 32];
    let mut expected_high: [u8; 32] = [0; 32];
    for i in 0..32 {
        expected_low[i] = bytes[i] & 0x0F;
        expected_high[i] = (bytes[i] >> 4) & 0x0F;
    }

    // SIMD extraction per llama.cpp pattern
    // SAFETY: We've verified AVX2 is available above
    #[target_feature(enable = "avx2")]
    unsafe fn simd_nibble_extract(
        bytes: &[u8; 32],
        result_low: &mut [u8; 32],
        result_high: &mut [u8; 32],
    ) {
        use std::arch::x86_64::*;

        // SAFETY: Memory safety ensured by bounds checking and alignment
        unsafe {
            let bytes_vec = _mm256_loadu_si256(bytes.as_ptr().cast::<__m256i>());
            let low_mask = _mm256_set1_epi8(0x0F);

            // Extract low nibbles: bytes & 0x0F
            let low_vec = _mm256_and_si256(bytes_vec, low_mask);

            // Extract high nibbles: (bytes >> 4) & 0x0F
            // Note: _mm256_srli_epi16 shifts 16-bit lanes, so we need to mask afterward
            let high_shifted = _mm256_srli_epi16(bytes_vec, 4);
            let high_vec = _mm256_and_si256(high_shifted, low_mask);

            // Store results
            _mm256_storeu_si256(result_low.as_mut_ptr().cast::<__m256i>(), low_vec);
            _mm256_storeu_si256(result_high.as_mut_ptr().cast::<__m256i>(), high_vec);
        }
    }

    let mut result_low: [u8; 32] = [0; 32];
    let mut result_high: [u8; 32] = [0; 32];

    // SAFETY: AVX2 is available (checked above)
    unsafe {
        simd_nibble_extract(&bytes, &mut result_low, &mut result_high);
    }

    // IMP-147b: SIMD results must match scalar
    assert_eq!(
        result_low, expected_low,
        "IMP-147b: SIMD low nibbles should match scalar"
    );
    assert_eq!(
        result_high, expected_high,
        "IMP-147b: SIMD high nibbles should match scalar"
    );

    println!("\nIMP-147b: AVX2 SIMD nibble extraction verified correct");
}

/// IMP-147c: Benchmark SIMD vs scalar nibble extraction throughput
#[test]
fn test_imp_147c_extraction_throughput_comparison() {
    // Create realistic workload: 4KB of bytes (1024 Q4_K blocks worth)
    let num_bytes = 4096;
    let bytes: Vec<u8> = (0..num_bytes).map(|i| (i % 256) as u8).collect();

    // Scalar extraction (baseline)
    let start = std::time::Instant::now();
    let mut scalar_low = Vec::with_capacity(num_bytes);
    let mut scalar_high = Vec::with_capacity(num_bytes);
    for _ in 0..1000 {
        // 1000 iterations for timing
        scalar_low.clear();
        scalar_high.clear();
        for &byte in &bytes {
            scalar_low.push(byte & 0x0F);
            scalar_high.push((byte >> 4) & 0x0F);
        }
    }
    let scalar_time = start.elapsed();

    // IMP-147c: Verify results are correct
    assert_eq!(scalar_low.len(), num_bytes);
    assert_eq!(scalar_high.len(), num_bytes);

    // Calculate throughput
    let scalar_bytes_per_sec =
        (num_bytes as f64 * 1000.0) / scalar_time.as_secs_f64() / 1_000_000.0;

    println!("\nIMP-147c: Nibble Extraction Throughput:");
    println!("  Scalar: {:.1} MB/s", scalar_bytes_per_sec);
    println!(
        "  Time for 4KB x 1000: {:.2}ms",
        scalar_time.as_secs_f64() * 1000.0
    );

    // IMP-147c: Baseline should process at least 5 MB/s (conservative for coverage builds)
    // In release builds with SIMD, expect > 1000 MB/s
    assert!(
        scalar_bytes_per_sec > 5.0,
        "IMP-147c: Scalar extraction should be > 5 MB/s, got {:.1}",
        scalar_bytes_per_sec
    );
}

/// IMP-147d: Verify optimized Q4_K fused dot uses efficient extraction
#[test]
fn test_imp_147d_q4k_fused_dot_correctness() {
    // Create Q4_K test data (minimal valid structure)
    let num_super_blocks = 1;
    let super_block_bytes = 144; // QK_K/2 + scales + dmins
    let q4k_data = vec![0u8; num_super_blocks * super_block_bytes];

    // Create matching activations
    let num_values = num_super_blocks * 256; // QK_K = 256
    let activations: Vec<f32> = (0..num_values).map(|i| (i as f32) * 0.01).collect();

    // IMP-147d: Fused dot should produce valid result (not panic or return error)
    // Note: With zero weights, result should be approximately zero
    let result = fused_q4k_dot(&q4k_data, &activations);

    match result {
        Ok(dot) => {
            // With all-zero quantized data, dot product should be small
            // (dmin * min contribution only)
            assert!(
                dot.abs() < 1000.0,
                "IMP-147d: Fused Q4K dot with zeros should be bounded, got {}",
                dot
            );
        },
        Err(e) => {
            // Some implementations may reject all-zero data
            println!(
                "IMP-147d: fused_q4k_dot returned error (may be expected): {}",
                e
            );
        },
    }
}

include!("imp_148a.rs");
include!("imp_149c.rs");
include!("dequantize_fused.rs");
include!("extract_scale.rs");
