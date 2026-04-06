
/// SIMD backend detected at runtime
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SimdBackend {
    /// AVX2 (256-bit)
    Avx2,
    /// SSE2 (128-bit)
    Sse2,
    /// ARM NEON (128-bit)
    Neon,
    /// Scalar fallback
    #[default]
    Scalar,
}

impl std::fmt::Display for SimdBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimdBackend::Avx2 => write!(f, "AVX2"),
            SimdBackend::Sse2 => write!(f, "SSE2"),
            SimdBackend::Neon => write!(f, "NEON"),
            SimdBackend::Scalar => write!(f, "Scalar"),
        }
    }
}

/// Detect available SIMD backend
pub fn detect_simd_backend() -> SimdBackend {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return SimdBackend::Avx2;
        }
        // pmat-ignore: hardware-path (SSE2 fallback never reached when AVX2 available)
        if is_x86_feature_detected!("sse2") {
            return SimdBackend::Sse2;
        }
    }

    // pmat-ignore: hardware-path (NEON path only on aarch64)
    #[cfg(target_arch = "aarch64")]
    {
        return SimdBackend::Neon;
    }

    // pmat-ignore: hardware-path (scalar fallback never reached when SIMD available)
    SimdBackend::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============= Q4_0Block tests =============

    #[test]
    fn test_q4_0_block_construction() {
        let block = Q4_0Block {
            scale: 1.0,
            quants: [0x55; 16], // All 0101 patterns
        };
        assert_eq!(block.scale, 1.0);
        assert_eq!(block.quants.len(), 16);
    }

    // ============= Q8_0Block tests =============

    #[test]
    fn test_q8_0_block_quantize_zeros() {
        let values = [0.0f32; 32];
        let block = Q8_0Block::quantize(&values);

        // Near-zero values should use minimal scale
        assert!(block.scale > 0.0);
        for q in &block.quants {
            assert_eq!(*q, 0);
        }
    }

    #[test]
    fn test_q8_0_block_quantize_max() {
        let values = [127.0f32; 32];
        let block = Q8_0Block::quantize(&values);

        assert!((block.scale - 1.0).abs() < 0.01);
        for q in &block.quants {
            assert_eq!(*q, 127);
        }
    }

    #[test]
    fn test_q8_0_block_quantize_negative() {
        let values = [-127.0f32; 32];
        let block = Q8_0Block::quantize(&values);

        for q in &block.quants {
            assert_eq!(*q, -127);
        }
    }

    #[test]
    fn test_q8_0_block_quantize_mixed() {
        let mut values = [0.0f32; 32];
        for i in 0..32 {
            values[i] = (i as f32 - 16.0) * 8.0;
        }

        let block = Q8_0Block::quantize(&values);
        let dequantized = block.dequantize();

        // Verify roundtrip is approximate
        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            let diff = (orig - deq).abs();
            assert!(
                diff < block.scale * 2.0,
                "diff={} scale={}",
                diff,
                block.scale
            );
        }
    }

    #[test]
    fn test_q8_0_block_dequantize() {
        let block = Q8_0Block {
            scale: 2.0,
            quants: [10i8; 32],
        };

        let values = block.dequantize();
        for val in &values {
            assert!((val - 20.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_q8_0_block_quantization_error() {
        let values = [50.0f32; 32];
        let block = Q8_0Block::quantize(&values);
        let error = block.quantization_error(&values);

        // Error should be less than scale
        assert!(error <= block.scale);
    }

    #[test]
    fn test_q8_0_block_relative_error() {
        let values = [100.0f32; 32];
        let block = Q8_0Block::quantize(&values);
        let rel_error = block.relative_error(&values);

        // Relative error should be small for large values
        assert!(rel_error < 0.01);
    }

    #[test]
    fn test_q8_0_block_relative_error_near_zero() {
        let values = [0.00001f32; 32];
        let block = Q8_0Block::quantize(&values);
        let rel_error = block.relative_error(&values);

        // Should handle near-zero gracefully
        assert!(rel_error >= 0.0);
    }

    // ============= Q8KSuperBlock tests =============

    #[test]
    fn test_q8k_super_block_quantize_zeros() {
        let values = [0.0f32; 256];
        let block = Q8KSuperBlock::quantize(&values);

        assert!(block.scale > 0.0);
        for q in &block.quants {
            assert_eq!(*q, 0);
        }
    }

    #[test]
    fn test_q8k_super_block_quantize_max() {
        let values = [127.0f32; 256];
        let block = Q8KSuperBlock::quantize(&values);

        assert!((block.scale - 1.0).abs() < 0.01);
        for q in &block.quants {
            assert_eq!(*q, 127);
        }
    }

    #[test]
    fn test_q8k_super_block_quantize_into() {
        let values = [64.0f32; 256];
        let mut scale = 0.0f32;
        let mut quants = [0i8; 256];

        Q8KSuperBlock::quantize_into(&values, &mut scale, &mut quants);

        assert!(scale > 0.0);
        // All quants should be equal since input is uniform
        let first_q = quants[0];
        for q in &quants {
            assert_eq!(*q, first_q);
        }
    }

    #[test]
    fn test_q8k_super_block_dequantize() {
        let block = Q8KSuperBlock {
            scale: 0.5,
            quants: [50i8; 256],
        };

        let values = block.dequantize();
        for val in &values {
            assert!((val - 25.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_q8k_super_block_roundtrip() {
        let mut values = [0.0f32; 256];
        for i in 0..256 {
            values[i] = (i as f32 - 128.0) / 2.0;
        }

        let block = Q8KSuperBlock::quantize(&values);
        let dequant = block.dequantize();

        // Verify roundtrip within tolerance
        for (orig, deq) in values.iter().zip(dequant.iter()) {
            let diff = (orig - deq).abs();
            assert!(diff < block.scale * 2.0);
        }
    }

    // ============= InterleavedQ4K tests =============

    #[test]
    fn test_interleaved_q4k_invalid_size() {
        let data = vec![0u8; 100]; // Not multiple of 144
        let result = InterleavedQ4K::from_q4k(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_interleaved_q4k_empty() {
        let data = vec![];
        let result = InterleavedQ4K::from_q4k(&data).expect("result");
        assert_eq!(result.num_super_blocks, 0);
        assert_eq!(result.num_values(), 0);
    }

    #[test]
    fn test_interleaved_q4k_single_block() {
        let mut data = vec![0u8; 144];
        // Set d = 1.0 (f16 = 0x3C00)
        data[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        // Set dmin = 0.5 (f16 = 0x3800)
        data[2..4].copy_from_slice(&0x3800u16.to_le_bytes());

        let result = InterleavedQ4K::from_q4k(&data).expect("result");

        assert_eq!(result.num_super_blocks, 1);
        assert_eq!(result.num_values(), 256);
        assert_eq!(result.d.len(), 1);
        assert_eq!(result.dmin.len(), 1);
        assert_eq!(result.scales.len(), 12);
        assert_eq!(result.qs.len(), 128);
    }

    #[test]
    fn test_interleaved_q4k_multiple_blocks() {
        let data = vec![0u8; 288]; // 2 super-blocks
        let result = InterleavedQ4K::from_q4k(&data).expect("result");

        assert_eq!(result.num_super_blocks, 2);
        assert_eq!(result.num_values(), 512);
        assert_eq!(result.d.len(), 2);
        assert_eq!(result.dmin.len(), 2);
        assert_eq!(result.scales.len(), 24);
        assert_eq!(result.qs.len(), 256);
    }

    // ============= DequantStats tests =============

    #[test]
    fn test_dequant_stats_default() {
        let stats = DequantStats::default();
        assert_eq!(stats.blocks_processed, 0);
        assert_eq!(stats.bytes_processed, 0);
        assert_eq!(stats.simd_backend, SimdBackend::Scalar);
    }

    // ============= SimdBackend tests =============

    #[test]
    fn test_simd_backend_display() {
        assert_eq!(format!("{}", SimdBackend::Avx2), "AVX2");
        assert_eq!(format!("{}", SimdBackend::Sse2), "SSE2");
        assert_eq!(format!("{}", SimdBackend::Neon), "NEON");
        assert_eq!(format!("{}", SimdBackend::Scalar), "Scalar");
    }

    #[test]
    fn test_simd_backend_default() {
        assert_eq!(SimdBackend::default(), SimdBackend::Scalar);
    }

    #[test]
    fn test_simd_backend_equality() {
        assert_eq!(SimdBackend::Avx2, SimdBackend::Avx2);
        assert_ne!(SimdBackend::Avx2, SimdBackend::Scalar);
    }

    #[test]
    fn test_detect_simd_backend() {
        let backend = detect_simd_backend();
        // On x86_64, should detect AVX2 or SSE2
        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                backend == SimdBackend::Avx2 || backend == SimdBackend::Sse2,
                "expected AVX2 or SSE2, got {:?}",
                backend
            );
        }
        // On other architectures, just verify it returns something
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = backend; // Just verify it compiles and runs
        }
    }

    // ============= Block struct tests =============

    #[test]
    fn test_q4_k_block_fields() {
        let block = Q4_KBlock {
            d: 1.0,
            dmin: 0.5,
            scales: [0; 12],
            qs: [0; 128],
        };
        assert_eq!(block.d, 1.0);
        assert_eq!(block.dmin, 0.5);
        assert_eq!(block.scales.len(), 12);
        assert_eq!(block.qs.len(), 128);
    }

    #[test]
    fn test_q5_k_block_fields() {
        let block = Q5_KBlock {
            d: 1.0,
            dmin: 0.5,
            scales: [0; 12],
            qh: [0; 32],
            qs: [0; 128],
        };
        assert_eq!(block.d, 1.0);
        assert_eq!(block.qh.len(), 32);
        assert_eq!(block.qs.len(), 128);
    }

    #[test]
    fn test_q6_k_block_fields() {
        let block = Q6_KBlock {
            d: 1.0,
            scales: [0; 16],
            qh: [0; 64],
            qs: [0; 128],
        };
        assert_eq!(block.d, 1.0);
        assert_eq!(block.scales.len(), 16);
        assert_eq!(block.qh.len(), 64);
        assert_eq!(block.qs.len(), 128);
    }

    // ============= Constants tests =============

    #[test]
    fn test_constants() {
        assert_eq!(BLOCK_SIZE, 32);
        assert_eq!(QK_K, 256);
    }
}
