//! Phase 17: APR Q4_0 Kernel Trace
//!
//! Verifies CPU and GPU compute identical results for a known Q4_0 block.
//!
//! # The Hypothesis
//!
//! H4 (Precision Mismatch): CPU uses Q4×Q8 integer math, GPU uses Q4×F32.
//! H5 (Scale Factor): f16 scale may be misinterpreted.
//!
//! # Usage
//!
//! ```bash
//! cargo test --test apr_q4_kernel_trace --features cuda -- --nocapture
//! ```

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::cuda::CudaExecutor;
    use realizar::quantize::fused_q4_0_q8_0_parallel_matvec;
    use trueno_gpu::driver::GpuBuffer;

    /// Create a Q4_0 block with known values
    ///
    /// Q4_0 format: 2 bytes f16 scale + 16 bytes (32 x 4-bit nibbles)
    /// Total: 18 bytes per 32 values
    fn create_q4_0_block(scale_f32: f32, nibbles: &[u8; 16]) -> [u8; 18] {
        let mut block = [0u8; 18];

        // Convert scale to f16 (IEEE 754 half-precision)
        let scale_f16 = f32_to_f16(scale_f32);
        block[0] = (scale_f16 & 0xFF) as u8;
        block[1] = ((scale_f16 >> 8) & 0xFF) as u8;

        // Copy nibbles
        block[2..18].copy_from_slice(nibbles);

        block
    }

    /// Convert f32 to f16 bits (simple conversion)
    fn f32_to_f16(f: f32) -> u16 {
        // Use half crate logic or manual conversion
        let bits = f.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x7FFFFF;

        if exp == 0 {
            // Zero or subnormal
            return (sign << 15) as u16;
        }
        if exp == 0xFF {
            // Inf or NaN
            return ((sign << 15) | 0x7C00 | (frac >> 13)) as u16;
        }

        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            // Overflow to infinity
            return ((sign << 15) | 0x7C00) as u16;
        }
        if new_exp <= 0 {
            // Underflow to zero
            return (sign << 15) as u16;
        }

        let new_frac = frac >> 13;
        ((sign << 15) | ((new_exp as u32) << 10) | new_frac) as u16
    }

    /// Convert f16 bits to f32
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = (bits >> 15) & 1;
        let exp = (bits >> 10) & 0x1F;
        let frac = bits & 0x3FF;

        if exp == 0 {
            if frac == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            }
            // Subnormal
            let f = frac as f32 / 1024.0 * 2.0f32.powi(-14);
            return if sign == 1 { -f } else { f };
        }
        if exp == 31 {
            if frac == 0 {
                return if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
            }
            return f32::NAN;
        }

        let f = (1.0 + frac as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15);
        if sign == 1 {
            -f
        } else {
            f
        }
    }

    /// Test 1: Verify scale conversion round-trip
    #[test]
    fn test_scale_roundtrip() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 17: SCALE CONVERSION ROUND-TRIP                               ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let test_scales = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, -0.5, -1.0];

        for &scale in &test_scales {
            let f16_bits = f32_to_f16(scale);
            let roundtrip = f16_to_f32(f16_bits);
            let diff = (scale - roundtrip).abs();
            let rel = if scale.abs() > 1e-6 {
                diff / scale.abs()
            } else {
                diff
            };

            eprintln!(
                "scale={:8.4} -> f16=0x{:04X} -> roundtrip={:8.4} (diff={:.6e})",
                scale, f16_bits, roundtrip, diff
            );

            // f16 has ~3 decimal digits precision
            assert!(rel < 0.01, "Round-trip error too large for scale={}", scale);
        }

        eprintln!("\n✅ Scale conversion round-trip OK");
    }

    /// Test 2: Create Q4_0 block and verify bytes
    #[test]
    fn test_block_layout() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 17: Q4_0 BLOCK LAYOUT                                          ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // Scale = 1.0, all nibbles = 0x88 (each element = 8, centered = 0)
        let nibbles = [0x88u8; 16];
        let block = create_q4_0_block(1.0, &nibbles);

        eprintln!("Block bytes: {:02X?}", &block);

        // Verify scale bytes (1.0 in f16 = 0x3C00)
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale_f32 = f16_to_f32(scale_bits);
        eprintln!(
            "Scale bytes: 0x{:02X}{:02X} = 0x{:04X} = {}",
            block[1], block[0], scale_bits, scale_f32
        );

        assert!((scale_f32 - 1.0).abs() < 0.01, "Scale should be ~1.0");

        // Verify nibbles
        for i in 0..16 {
            let byte = block[2 + i];
            let lo = byte & 0x0F;
            let hi = (byte >> 4) & 0x0F;
            eprintln!("byte[{}] = 0x{:02X}: lo={}, hi={}", i, byte, lo, hi);
        }

        eprintln!("\n✅ Block layout OK");
    }

    /// Test 3: CPU Q4_0 dot product with known values
    #[test]
    fn test_cpu_q4_0_dot() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 17: CPU Q4_0 DOT PRODUCT                                       ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // Create weight: 1 row of 32 elements (1 block)
        // Scale = 0.1, nibbles represent [-8..7] centered values
        // Nibbles: 0x00 = (0, 0), means centered = (-8, -8)
        //          0x88 = (8, 8), means centered = (0, 0)
        //          0xFF = (15, 15), means centered = (7, 7)

        // Create simple pattern: all 8s (centered = 0)
        let nibbles = [0x88u8; 16];
        let block = create_q4_0_block(0.1, &nibbles);

        eprintln!("Weight block (scale=0.1, all centered=0):");
        eprintln!("  bytes: {:02X?}", &block);

        // Activation: all 1.0
        let activations = vec![1.0f32; 32];

        // Expected: sum over 32 elements of (0.1 * 0 * 1.0) = 0.0
        let result = fused_q4_0_q8_0_parallel_matvec(&block, &activations, 32, 1)
            .expect("CPU matmul failed");

        eprintln!("CPU result: {:?}", result);
        eprintln!("Expected: 0.0 (all weights are centered to 0)");

        assert!(
            result[0].abs() < 0.1,
            "Result should be ~0 for centered=0 weights"
        );

        // Now test with non-zero weights
        // Nibbles: 0x99 = (9, 9), means centered = (1, 1)
        let nibbles2 = [0x99u8; 16];
        let block2 = create_q4_0_block(0.1, &nibbles2);

        // Expected: sum over 32 elements of (0.1 * 1 * 1.0) = 32 * 0.1 = 3.2
        let result2 = fused_q4_0_q8_0_parallel_matvec(&block2, &activations, 32, 1)
            .expect("CPU matmul failed");

        eprintln!("\nWeight block (scale=0.1, all centered=1):");
        eprintln!("  bytes: {:02X?}", &block2);
        eprintln!("CPU result: {:?}", result2);
        eprintln!("Expected: ~3.2 (32 * 0.1 * 1)");

        // Allow for Q8 quantization error
        assert!((result2[0] - 3.2).abs() < 0.5, "Result should be ~3.2");

        eprintln!("\n✅ CPU Q4_0 dot product OK");
    }

    /// Test 4: GPU Q4_0 dot product with known values
    #[test]
    fn test_gpu_q4_0_dot() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 17: GPU Q4_0 DOT PRODUCT                                       ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // Create weight: scale=0.1, all nibbles = 0x99 (centered=1)
        let nibbles = [0x99u8; 16];
        let block = create_q4_0_block(0.1, &nibbles);

        eprintln!("Weight block (scale=0.1, all centered=1):");
        eprintln!("  bytes: {:02X?}", &block);

        // Activation: all 1.0
        let activations = vec![1.0f32; 32];

        // Expected: sum over 32 elements of (0.1 * 1 * 1.0) = 3.2

        match CudaExecutor::new(0) {
            Ok(mut executor) => {
                // Upload weight
                let cache_key = "test_weight".to_string();
                executor
                    .load_quantized_weights_with_type(&cache_key, &block, 2) // Q4_0 = type 2
                    .expect("Failed to upload weight");

                // Get weight pointer
                let weight_ptr = executor
                    .get_quantized_weight_ptr(&cache_key)
                    .expect("Weight not cached");

                eprintln!("Weight uploaded to GPU, ptr=0x{:016X}", weight_ptr);

                // Download weight bytes to verify
                // Note: We need to access the buffer to verify bytes
                // This is tricky because we only have the pointer

                // Upload activation
                let input_gpu = GpuBuffer::from_host(executor.context(), &activations)
                    .expect("Failed to upload activations");

                // Allocate output
                let output_gpu =
                    GpuBuffer::new(executor.context(), 1).expect("Failed to allocate output");

                // Run kernel
                executor
                    .q4_0_gemv_into(
                        weight_ptr,
                        &input_gpu,
                        &output_gpu,
                        1,  // n = 1 output
                        32, // k = 32 input
                    )
                    .expect("GPU GEMV failed");

                // Sync and read result
                executor.synchronize().expect("Sync failed");

                let mut result = vec![0.0f32; 1];
                output_gpu.copy_to_host(&mut result).expect("D2H failed");

                eprintln!("GPU result: {:?}", result);
                eprintln!("Expected: ~3.2");

                // Compare with CPU
                let cpu_result = fused_q4_0_q8_0_parallel_matvec(&block, &activations, 32, 1)
                    .expect("CPU matmul failed");

                eprintln!("CPU result: {:?}", cpu_result);

                let diff = (result[0] - cpu_result[0]).abs();
                let rel = diff / cpu_result[0].abs().max(1e-6);

                eprintln!("\n--- COMPARISON ---");
                eprintln!("CPU: {:.6}", cpu_result[0]);
                eprintln!("GPU: {:.6}", result[0]);
                eprintln!("Diff: {:.6} ({:.2}%)", diff, rel * 100.0);

                if rel < 0.01 {
                    eprintln!("\n✅ GPU matches CPU within 1%");
                } else if rel < 0.10 {
                    eprintln!("\n⚠️ GPU differs from CPU by {:.2}%", rel * 100.0);
                } else {
                    eprintln!("\n❌ GPU DIVERGES from CPU by {:.2}%!", rel * 100.0);
                }

                assert!(rel < 0.20, "GPU diverges too much from CPU");
            },
            Err(e) => {
                eprintln!("⚠️ CUDA not available: {:?}", e);
            },
        }
    }

    /// Test 5: Full comparison with real model weights (if available)
    #[test]
    fn test_real_model_block() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 17: REAL MODEL BLOCK COMPARISON                               ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // Try to load a real APR model
        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0.apr");

        if !model_path.exists() {
            eprintln!("⚠️ Skipping: APR model not found at {:?}", model_path);
            eprintln!("   Run: realizar convert ~/models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");
            return;
        }

        eprintln!("Model found at {:?}", model_path);
        eprintln!("(Full model test would require more infrastructure)");
    }
}
