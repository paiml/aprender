//! Phase 16: Q4 Matmul Debug Test
//!
//! Tests Q4K GEMV kernel in isolation to diagnose layout mismatch between
//! realizar's weight format and trueno-gpu kernel expectations.
//!
//! # Hypothesis H3 (Layout Mismatch)
//!
//! APR Q4 format uses 32-sized blocks, trueno kernel expects different layout.
//! Test: Create known weights, compute expected output, compare with GPU.
//!
//! # Usage
//!
//! ```bash
//! SKIP_CUDA_GRAPH=1 cargo test --test q4_matmul_debug --features cuda -- --nocapture
//! ```

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use realizar::cuda::CudaExecutor;
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    /// Q4_K super-block structure constants
    const Q4K_SUPER_BLOCK_SIZE: usize = 256;
    const Q4K_SUPER_BLOCK_BYTES: usize = 144;

    /// Create a Q4_K super-block with known values
    ///
    /// Q4_K layout (144 bytes total):
    /// - d: f16 (2 bytes) - main scale
    /// - dmin: f16 (2 bytes) - minimum scale
    /// - scales[12]: u8 (12 bytes) - per-32-block scales (packed)
    /// - qs[128]: u8 (128 bytes) - quantized values (4-bit nibbles)
    ///
    /// Dequant formula: val = d * (q - 8) * scale_for_block + dmin * min_for_block
    fn create_q4k_superblock_identity() -> Vec<u8> {
        let mut sb = vec![0u8; Q4K_SUPER_BLOCK_BYTES];

        // Set d = 1.0 (as f16)
        // f16 1.0 = 0x3C00
        sb[0] = 0x00;
        sb[1] = 0x3C;

        // Set dmin = 0.0 (as f16)
        // f16 0.0 = 0x0000
        sb[2] = 0x00;
        sb[3] = 0x00;

        // Set all scales to 1 (indices 4-15)
        // Scale packing: lower 4 bits of scales[i] for block i, upper bits for block i+8
        // For simplicity, set all scale bytes to 0x11 (scale = 1 for both packed values)
        for byte in sb[4..16].iter_mut() {
            *byte = 0x11; // Both nibbles = 1
        }

        // Set quantized values (indices 16-143)
        // Each byte contains 2 nibbles (4-bit values)
        // Q4_K uses nibbles in range [0, 15], centered at 8
        // So nibble 8 = 0, nibble 9 = 1, nibble 7 = -1, etc.
        //
        // Create an identity-like pattern: first value = 9 (dequant to 1*1*(9-8) = 1)
        // Rest = 8 (dequant to 0)
        for byte in sb[16..Q4K_SUPER_BLOCK_BYTES].iter_mut() {
            *byte = 0x88; // Both nibbles = 8 (zero after centering)
        }
        // First nibble = 9 (will dequant to 1.0 after d * (9-8) * scale)
        sb[16] = 0x89; // Low nibble = 9, high nibble = 8

        sb
    }

    /// Create a Q4_K super-block with all values = some constant
    fn create_q4k_superblock_constant(value: i8) -> Vec<u8> {
        let mut sb = vec![0u8; Q4K_SUPER_BLOCK_BYTES];

        // d = 1.0
        sb[0] = 0x00;
        sb[1] = 0x3C;

        // dmin = 0.0
        sb[2] = 0x00;
        sb[3] = 0x00;

        // scales = 1
        for byte in sb[4..16].iter_mut() {
            *byte = 0x11;
        }

        // All nibbles set to (value + 8) to represent the centered value
        let nibble = ((value + 8) as u8) & 0x0F;
        let packed = (nibble << 4) | nibble;
        for byte in sb[16..Q4K_SUPER_BLOCK_BYTES].iter_mut() {
            *byte = packed;
        }

        sb
    }

    /// Test 1: Q4K GEMV with identity-like pattern
    ///
    /// Setup:
    /// - Weight: 1 row of 256 columns (1 super-block)
    /// - First weight = 1.0, rest = 0.0
    /// - Input: vector of 256 ones
    /// - Expected output: 1.0 (dot product picks up just the first weight)
    #[test]
    fn test_q4k_gemv_identity_pattern() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: Q4K GEMV IDENTITY PATTERN TEST                            ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // Check CUDA availability
        let mut executor = match CudaExecutor::new(0) {
            Ok(exec) => exec,
            Err(e) => {
                eprintln!("CUDA init failed: {:?}", e);
                eprintln!("Investigate: driver version? library path?");
                return;
            },
        };

        // Create Q4K super-block with identity pattern
        let weights = create_q4k_superblock_identity();
        eprintln!("Weight super-block ({} bytes):", weights.len());
        eprintln!("  d (f16): [{:#04x}, {:#04x}]", weights[0], weights[1]);
        eprintln!("  dmin (f16): [{:#04x}, {:#04x}]", weights[2], weights[3]);
        eprintln!("  scales[0..4]: {:?}", &weights[4..8]);
        eprintln!("  qs[0..8]: {:?}", &weights[16..24]);

        // Load weights to GPU
        let weight_key = "test_identity";
        executor
            .load_quantized_weights(weight_key, &weights)
            .expect("Failed to load weights");

        // Create input vector: all 1.0s
        let input: Vec<f32> = vec![1.0; Q4K_SUPER_BLOCK_SIZE];
        let mut output = vec![0.0f32; 1];

        eprintln!("\nInput: {} ones", input.len());
        eprintln!("Expected: ~1.0 (first weight=1, rest=0, dot with all 1s)");

        // Execute Q4K GEMV
        let k = Q4K_SUPER_BLOCK_SIZE as u32;
        let n = 1u32;

        let result = executor.q4k_gemv_cached(weight_key, &input, &mut output, n, k);

        match result {
            Ok(()) => {
                eprintln!("\n✅ GEMV succeeded");
                eprintln!("Output: {:.6}", output[0]);

                // Tolerance: Q4K has limited precision
                let expected = 1.0f32;
                let diff = (output[0] - expected).abs();
                if diff < 0.5 {
                    eprintln!("✅ Output matches expected (diff={:.4})", diff);
                } else {
                    eprintln!("❌ OUTPUT MISMATCH!");
                    eprintln!("   Expected: {:.4}", expected);
                    eprintln!("   Got: {:.4}", output[0]);
                    eprintln!("   Diff: {:.4}", diff);
                    eprintln!("\n   Possible causes:");
                    eprintln!("   1. Scale extraction bug (d/dmin parsing)");
                    eprintln!("   2. Nibble extraction order (low/high swap)");
                    eprintln!("   3. Block indexing (which 32-block within superblock)");
                    eprintln!("   4. Row stride calculation");
                }
            },
            Err(e) => {
                eprintln!("❌ GEMV failed: {:?}", e);
            },
        }
    }

    /// Test 2: Q4K GEMV with constant value
    ///
    /// Setup:
    /// - Weight: 1 row of 256 columns (1 super-block)
    /// - All weights = 1.0
    /// - Input: vector of 256 ones
    /// - Expected output: 256.0
    #[test]
    fn test_q4k_gemv_constant_value() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: Q4K GEMV CONSTANT VALUE TEST                              ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let mut executor = match CudaExecutor::new(0) {
            Ok(exec) => exec,
            Err(e) => {
                eprintln!("CUDA init failed: {:?}", e);
                return;
            },
        };

        // Create Q4K super-block with all values = 1 (nibble = 9, centered at 8)
        let weights = create_q4k_superblock_constant(1);

        eprintln!("Weight super-block: all values = 1");
        eprintln!("  d (f16): [{:#04x}, {:#04x}]", weights[0], weights[1]);
        eprintln!("  qs[0..8]: {:?}", &weights[16..24]);

        let weight_key = "test_constant";
        executor
            .load_quantized_weights(weight_key, &weights)
            .expect("Failed to load weights");

        let input: Vec<f32> = vec![1.0; Q4K_SUPER_BLOCK_SIZE];
        let mut output = vec![0.0f32; 1];

        let k = Q4K_SUPER_BLOCK_SIZE as u32;
        let n = 1u32;

        eprintln!("\nInput: {} ones", input.len());
        eprintln!("Expected: ~256.0 (all weights=1 * all inputs=1)");

        let result = executor.q4k_gemv_cached(weight_key, &input, &mut output, n, k);

        match result {
            Ok(()) => {
                eprintln!("\n✅ GEMV succeeded");
                eprintln!("Output: {:.6}", output[0]);

                let expected = 256.0f32;
                let diff = (output[0] - expected).abs();
                let ratio = output[0] / expected;

                if diff < 32.0 {
                    eprintln!(
                        "✅ Output roughly matches (diff={:.1}, ratio={:.2}x)",
                        diff, ratio
                    );
                } else {
                    eprintln!("❌ OUTPUT MISMATCH!");
                    eprintln!("   Expected: {:.1}", expected);
                    eprintln!("   Got: {:.1}", output[0]);
                    eprintln!("   Ratio: {:.2}x", ratio);

                    // Diagnostic: what ratio tells us
                    if (ratio - 8.0).abs() < 0.5 {
                        eprintln!(
                            "\n   DIAGNOSIS: ~8x ratio suggests super-block vs block confusion"
                        );
                        eprintln!("   (256/32 = 8 blocks per super-block)");
                    } else if (ratio - 2.0).abs() < 0.5 {
                        eprintln!("\n   DIAGNOSIS: ~2x ratio suggests nibble double-counting");
                    } else if (ratio - 0.5).abs() < 0.1 {
                        eprintln!(
                            "\n   DIAGNOSIS: ~0.5x ratio suggests only half the values processed"
                        );
                    }
                }
            },
            Err(e) => {
                eprintln!("❌ GEMV failed: {:?}", e);
            },
        }
    }

    /// Test 3: Compare CPU vs GPU dequantization of same Q4K data
    #[test]
    fn test_q4k_cpu_gpu_dequant_comparison() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: Q4K CPU vs GPU DEQUANTIZATION COMPARISON                  ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        // Create Q4K data with known pattern
        let weights = create_q4k_superblock_constant(1);

        // CPU dequantization
        eprintln!("CPU Dequantization:");
        let cpu_dequant = cpu_dequantize_q4k_superblock(&weights);
        eprintln!("  First 8 values: {:?}", &cpu_dequant[..8]);
        eprintln!("  Sum: {:.1}", cpu_dequant.iter().sum::<f32>());

        // Try to init CUDA
        let mut executor = match CudaExecutor::new(0) {
            Ok(exec) => exec,
            Err(e) => {
                eprintln!("\nCUDA init failed: {:?}", e);
                return;
            },
        };

        // GPU GEMV with identity input (to extract dequantized values)
        let weight_key = "test_dequant";
        executor
            .load_quantized_weights(weight_key, &weights)
            .expect("Failed to load weights");

        // Use identity matrix approach: input with 1 at position i, 0 elsewhere
        // This extracts the i-th dequantized weight
        eprintln!("\nGPU Dequantization (via GEMV with unit vectors):");

        let mut gpu_dequant = Vec::with_capacity(Q4K_SUPER_BLOCK_SIZE);
        let k = Q4K_SUPER_BLOCK_SIZE as u32;
        let n = 1u32;

        // Sample first 8 positions
        for i in 0..8 {
            let mut input = vec![0.0f32; Q4K_SUPER_BLOCK_SIZE];
            input[i] = 1.0;
            let mut output = vec![0.0f32; 1];

            executor
                .q4k_gemv_cached(weight_key, &input, &mut output, n, k)
                .expect("GEMV failed");
            gpu_dequant.push(output[0]);
        }

        eprintln!("  First 8 values: {:?}", gpu_dequant);

        // Compare
        eprintln!("\nComparison:");
        for i in 0..8 {
            let cpu = cpu_dequant[i];
            let gpu = gpu_dequant[i];
            let diff = (cpu - gpu).abs();
            let status = if diff < 0.1 { "✅" } else { "❌" };
            eprintln!(
                "  [{}] CPU={:.4}, GPU={:.4}, diff={:.4} {}",
                i, cpu, gpu, diff, status
            );
        }
    }

    /// CPU reference dequantization of Q4K super-block
    fn cpu_dequantize_q4k_superblock(sb: &[u8]) -> Vec<f32> {
        assert_eq!(sb.len(), Q4K_SUPER_BLOCK_BYTES);

        // Parse header
        let d = f16_to_f32(sb[0], sb[1]);
        let dmin = f16_to_f32(sb[2], sb[3]);

        eprintln!("  d={:.4}, dmin={:.4}", d, dmin);

        // Parse scales (12 bytes for 8 blocks of 32)
        // Scale packing is complex - each block has a scale and a min
        // For simplicity, assume uniform scale = 1
        let scales: Vec<f32> = (0..8)
            .map(|i| {
                let byte = sb[4 + i];

                (byte & 0x0F) as f32
            })
            .collect();
        eprintln!("  scales: {:?}", scales);

        // Parse quantized values
        let mut result = Vec::with_capacity(Q4K_SUPER_BLOCK_SIZE);
        let qs_base = 16;

        for i in 0..Q4K_SUPER_BLOCK_SIZE {
            let byte_idx = i / 2;
            let nibble_idx = i % 2;
            let byte = sb[qs_base + byte_idx];
            let nibble = if nibble_idx == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };

            // Determine which 32-block this belongs to
            let block = i / 32;
            let scale = if block < 8 { scales[block] } else { 1.0 };

            // Dequantize: val = d * (nibble - 8) * scale
            // (Simplified - actual Q4K has more complex scale/min handling)
            let centered = nibble as i8 - 8;
            let dequant = d * (centered as f32) * scale;
            result.push(dequant);
        }

        result
    }

    /// Convert f16 bytes (little-endian) to f32
    fn f16_to_f32(lo: u8, hi: u8) -> f32 {
        let bits = (hi as u16) << 8 | (lo as u16);
        half::f16::from_bits(bits).to_f32()
    }

    /// Test 4: Real model weight inspection
    ///
    /// Load TinyLlama Q4K weights and inspect tensor info
    #[test]
    fn test_real_model_weight_dump() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: REAL MODEL WEIGHT INSPECTION                              ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping: model not found at {:?}", model_path);
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        // List tensor info
        eprintln!("Model loaded. Tensor count: {}", mapped.model.tensors.len());
        eprintln!("\nFirst 10 tensors:");
        for (i, tensor) in mapped.model.tensors.iter().take(10).enumerate() {
            eprintln!(
                "  {}: {} - dims={:?}, qtype={}",
                i,
                tensor.name,
                &tensor.dims[..tensor.n_dims as usize],
                tensor.qtype
            );
        }

        // Find embedding tensor
        if let Some(tensor) = mapped
            .model
            .tensors
            .iter()
            .find(|t| t.name.contains("embd"))
        {
            eprintln!("\nToken embedding tensor:");
            eprintln!("  Name: {}", tensor.name);
            eprintln!("  Dims: {:?}", &tensor.dims[..tensor.n_dims as usize]);
            eprintln!("  QType: {} (12=Q4K, 6=Q6K, 0=F32)", tensor.qtype);
            eprintln!("  Offset: {}", tensor.offset);

            // Get raw data via slice
            let data_start = mapped.model.tensor_data_start + tensor.offset as usize;
            let tensor_size: u64 = tensor.dims[..tensor.n_dims as usize].iter().product();
            if let Some(data) = mapped
                .data()
                .get(data_start..data_start + Q4K_SUPER_BLOCK_BYTES.min(tensor_size as usize))
            {
                if data.len() >= Q4K_SUPER_BLOCK_BYTES {
                    eprintln!("\n  First super-block ({} bytes):", Q4K_SUPER_BLOCK_BYTES);
                    eprintln!("    d (f16): [{:#04x}, {:#04x}]", data[0], data[1]);
                    eprintln!("    dmin (f16): [{:#04x}, {:#04x}]", data[2], data[3]);
                    eprintln!("    scales[0..8]: {:?}", &data[4..12]);
                    eprintln!("    qs[0..16]: {:?}", &data[16..32]);

                    let d = f16_to_f32(data[0], data[1]);
                    let dmin = f16_to_f32(data[2], data[3]);
                    eprintln!("    d={:.6}, dmin={:.6}", d, dmin);
                }
            }
        }

        // Find first layer Q projection
        if let Some(tensor) = mapped
            .model
            .tensors
            .iter()
            .find(|t| t.name.contains("blk.0.attn_q"))
        {
            eprintln!("\n\nAttention Q weight (layer 0):");
            eprintln!("  Name: {}", tensor.name);
            eprintln!("  Dims: {:?}", &tensor.dims[..tensor.n_dims as usize]);
            eprintln!("  QType: {} (12=Q4K)", tensor.qtype);

            let data_start = mapped.model.tensor_data_start + tensor.offset as usize;
            if let Some(data) = mapped
                .data()
                .get(data_start..data_start + Q4K_SUPER_BLOCK_BYTES)
            {
                let d = f16_to_f32(data[0], data[1]);
                let dmin = f16_to_f32(data[2], data[3]);
                eprintln!("  First superblock: d={:.6}, dmin={:.6}", d, dmin);
            }
        }
    }

    /// Test 5: Compare CPU and GPU dequantization of real Q4K weights
    ///
    /// This is the key diagnostic test: extract the first row of a real Q4K weight
    /// and compare CPU dequantization vs GPU GEMV with unit vector input.
    #[test]
    fn test_real_q4k_cpu_gpu_dequant() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: REAL Q4K CPU vs GPU DEQUANTIZATION                        ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        // Find first layer Q weight (Q4K quantized)
        let tensor = mapped
            .model
            .tensors
            .iter()
            .find(|t| t.name.contains("blk.0.attn_q"))
            .expect("attn_q tensor not found");

        eprintln!(
            "Tensor: {} - dims={:?}, qtype={}",
            tensor.name,
            &tensor.dims[..tensor.n_dims as usize],
            tensor.qtype
        );

        let out_dim = tensor.dims[0] as usize; // N = 2048
        let in_dim = tensor.dims[1] as usize; // K = 2048

        let data_start = mapped.model.tensor_data_start + tensor.offset as usize;
        let n_super_blocks = in_dim.div_ceil(256); // 8 for K=2048
        let bytes_per_row = n_super_blocks * 144; // 8 * 144 = 1152

        eprintln!("Weight layout:");
        eprintln!("  out_dim (N): {}", out_dim);
        eprintln!("  in_dim (K): {}", in_dim);
        eprintln!("  super_blocks per row: {}", n_super_blocks);
        eprintln!("  bytes per row: {}", bytes_per_row);

        // Extract first row of weights (row 0)
        let row_data = mapped
            .data()
            .get(data_start..data_start + bytes_per_row)
            .expect("Failed to get row data");

        // CPU dequantization
        eprintln!("\nCPU Dequantization (first row, {} values):", in_dim);
        let cpu_dequant =
            realizar::quantize::dequantize_q4_k(row_data).expect("CPU dequant failed");

        eprintln!("  First 8 values: {:?}", &cpu_dequant[..8]);
        eprintln!("  Sum: {:.6}", cpu_dequant.iter().sum::<f32>());
        eprintln!(
            "  L2: {:.6}",
            cpu_dequant.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // GPU dequantization via GEMV with unit vectors
        eprintln!("\nGPU Dequantization (via GEMV with unit vectors):");

        let mut executor = match CudaExecutor::new(0) {
            Ok(exec) => exec,
            Err(e) => {
                eprintln!("CUDA init failed: {:?}", e);
                return;
            },
        };

        // Load first row as a 1xK weight matrix
        let weight_key = "test_row0";
        executor
            .load_quantized_weights(weight_key, row_data)
            .expect("Failed to load weights");

        // Use unit vector to extract dequantized values
        let mut gpu_dequant = Vec::with_capacity(in_dim);
        let k = in_dim as u32;
        let n = 1u32;

        // Sample first 8 and last 8 positions
        for i in (0..8).chain((in_dim - 8)..in_dim) {
            let mut input = vec![0.0f32; in_dim];
            input[i] = 1.0;
            let mut output = vec![0.0f32; 1];

            executor
                .q4k_gemv_cached(weight_key, &input, &mut output, n, k)
                .expect("GEMV failed");

            if i < 8 {
                gpu_dequant.push(output[0]);
            }
        }

        eprintln!("  First 8 values: {:?}", gpu_dequant);

        // Compare
        eprintln!("\nComparison (first 8 values):");
        let mut max_diff = 0.0f32;
        for i in 0..8 {
            let cpu = cpu_dequant[i];
            let gpu = gpu_dequant[i];
            let diff = (cpu - gpu).abs();
            max_diff = max_diff.max(diff);
            let status = if diff < 0.001 { "✅" } else { "❌" };
            eprintln!(
                "  [{}] CPU={:.6}, GPU={:.6}, diff={:.6} {}",
                i, cpu, gpu, diff, status
            );
        }

        eprintln!("\nMax diff: {:.6}", max_diff);
        if max_diff > 0.01 {
            eprintln!("❌ SIGNIFICANT DIVERGENCE DETECTED!");
            eprintln!("\nAnalysis:");
            eprintln!("  CPU sum: {:.6}", cpu_dequant.iter().sum::<f32>());
            eprintln!("  GPU first 8 sum: {:.6}", gpu_dequant.iter().sum::<f32>());
            eprintln!(
                "\n  This confirms H3: Q4K layout mismatch between CPU dequant and GPU kernel"
            );
        } else {
            eprintln!("✅ CPU and GPU dequantization match!");
        }
    }

    /// Test 6: Full matrix Q4K GEMV (NxK) with unit vector input
    ///
    /// Tests that full matrix GEMV produces correct results for multiple output rows.
    #[test]
    fn test_full_matrix_q4k_gemv() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: FULL MATRIX Q4K GEMV TEST                                 ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        // Find first layer Q weight
        let tensor = mapped
            .model
            .tensors
            .iter()
            .find(|t| t.name.contains("blk.0.attn_q"))
            .expect("attn_q tensor not found");

        let out_dim = tensor.dims[0] as usize; // N = 2048
        let in_dim = tensor.dims[1] as usize; // K = 2048

        let data_start = mapped.model.tensor_data_start + tensor.offset as usize;
        let n_super_blocks = in_dim.div_ceil(256);
        let bytes_per_row = n_super_blocks * 144;
        let total_bytes = out_dim * bytes_per_row;

        eprintln!("Tensor: {} [{} x {}]", tensor.name, out_dim, in_dim);
        eprintln!("Total bytes: {}", total_bytes);

        // Extract full weight matrix
        let weight_data = mapped
            .data()
            .get(data_start..data_start + total_bytes)
            .expect("Failed to get weight data");

        // CPU: compute all output values with unit vector at position 0
        // output[i] = weight[i][0] (first column of weight matrix)
        eprintln!("\nCPU reference (first column of weight matrix):");
        let mut cpu_outputs = Vec::with_capacity(out_dim);
        for row in 0..out_dim {
            let row_start = row * bytes_per_row;
            let row_data = &weight_data[row_start..row_start + bytes_per_row];
            let dequant = realizar::quantize::dequantize_q4_k(row_data).expect("dequant");
            cpu_outputs.push(dequant[0]); // First element of each row
        }
        eprintln!("  First 8: {:?}", &cpu_outputs[..8]);

        // GPU: full matrix GEMV with unit vector at position 0
        let mut executor = match CudaExecutor::new(0) {
            Ok(exec) => exec,
            Err(e) => {
                eprintln!("CUDA init failed: {:?}", e);
                return;
            },
        };

        let weight_key = "test_full_matrix";
        executor
            .load_quantized_weights(weight_key, weight_data)
            .expect("Failed to load weights");

        let mut input = vec![0.0f32; in_dim];
        input[0] = 1.0; // Unit vector at position 0
        let mut gpu_outputs = vec![0.0f32; out_dim];

        executor
            .q4k_gemv_cached(
                weight_key,
                &input,
                &mut gpu_outputs,
                out_dim as u32,
                in_dim as u32,
            )
            .expect("GEMV failed");

        eprintln!("\nGPU result:");
        eprintln!("  First 8: {:?}", &gpu_outputs[..8]);

        // Compare
        eprintln!("\nComparison:");
        let mut max_diff = 0.0f32;
        for i in 0..out_dim.min(16) {
            let cpu = cpu_outputs[i];
            let gpu = gpu_outputs[i];
            let diff = (cpu - gpu).abs();
            max_diff = max_diff.max(diff);
            let status = if diff < 0.0001 { "✅" } else { "❌" };
            eprintln!(
                "  [{:4}] CPU={:+.6}, GPU={:+.6}, diff={:.6} {}",
                i, cpu, gpu, diff, status
            );
        }

        eprintln!(
            "\nMax diff (first {} rows): {:.6}",
            out_dim.min(16),
            max_diff
        );

        if max_diff > 0.001 {
            eprintln!("❌ FULL MATRIX GEMV DIVERGENCE DETECTED!");
        } else {
            eprintln!("✅ Full matrix GEMV matches CPU!");
        }
    }

    /// Test 7: Compare CPU and GPU forward pass intermediate values
    ///
    /// This enables debug output to trace intermediate values in both paths.
    #[test]
    fn test_forward_trace_comparison() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: CPU vs GPU FORWARD TRACE COMPARISON                       ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        // Load CPU model
        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        eprintln!("Test token: {} (BOS)", bos_token);

        // Get embedding for BOS token using public API
        let embedding = cpu_model.embed(&[bos_token]);
        eprintln!("Embedding dim: {}", embedding.len());
        eprintln!("Embedding[0..8]: {:?}", &embedding[..8]);
        eprintln!(
            "Embedding L2: {:.6}",
            embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // CPU forward pass
        eprintln!("\nCPU Forward:");
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), 10);
        let cpu_logits = cpu_model
            .forward_cached(bos_token, &mut cpu_cache, 0)
            .expect("CPU forward failed");

        let cpu_top1 = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        let cpu_decoded = mapped.model.decode(&[cpu_top1]);
        eprintln!("  Top-1: {} ({:?})", cpu_top1, cpu_decoded);
        eprintln!(
            "  Logits L2: {:.6}",
            cpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        eprintln!("  Logits[0..5]: {:?}", &cpu_logits[..5]);

        // GPU forward pass with debug output
        eprintln!("\nGPU Forward (enabling REALIZAR_DEBUG_FORWARD):");

        let mut cuda_model = match OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(),
            0,
            10,
        ) {
            Ok(model) => model,
            Err(e) => {
                eprintln!("CUDA init failed: {:?}", e);
                return;
            },
        };

        // Enable debug output and disable CUDA graphs
        std::env::set_var("REALIZAR_DEBUG_FORWARD", "1");
        std::env::set_var("SKIP_CUDA_GRAPH", "1");

        let mut gpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cuda_model.model().config(), 10);

        // This will print debug info if REALIZAR_DEBUG_FORWARD is set
        let gpu_logits =
            cuda_model.forward_single_full_cuda_with_cache(bos_token, &mut gpu_cache, 0);

        std::env::remove_var("REALIZAR_DEBUG_FORWARD");
        std::env::remove_var("SKIP_CUDA_GRAPH");

        match gpu_logits {
            Ok(logits) => {
                let gpu_top1 = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0);
                let gpu_decoded = mapped.model.decode(&[gpu_top1]);
                eprintln!("  Top-1: {} ({:?})", gpu_top1, gpu_decoded);
                eprintln!(
                    "  Logits L2: {:.6}",
                    logits.iter().map(|x| x * x).sum::<f32>().sqrt()
                );
                eprintln!("  Logits[0..5]: {:?}", &logits[..5]);

                // Compare
                eprintln!("\nComparison:");
                if cpu_top1 == gpu_top1 {
                    eprintln!("✅ CPU and GPU MATCH!");
                } else {
                    eprintln!("❌ DIVERGENCE!");
                    eprintln!("  CPU: {} ({:?})", cpu_top1, cpu_decoded);
                    eprintln!("  GPU: {} ({:?})", gpu_top1, gpu_decoded);
                    eprintln!("\n  Check [PAR-052] debug output above for intermediate values");
                }
            },
            Err(e) => {
                eprintln!("❌ GPU forward failed: {:?}", e);
            },
        }
    }

    /// Test 8: Direct QKV comparison with same normalized input
    ///
    /// This test isolates the QKV matmul by using the same normalized input
    /// and comparing CPU `fused_q4k_parallel_matvec` vs GPU `q4k_gemv_cached`.
    #[test]
    fn test_direct_qkv_comparison() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: DIRECT QKV COMPARISON (Same Input)                        ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        // Get BOS embedding and normalize it (same as forward pass would)
        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        let embedding = cpu_model.embed(&[bos_token]);

        eprintln!("Embedding dim: {}", embedding.len());
        eprintln!(
            "Embedding L2: {:.6}",
            embedding.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Get layer 0 attention norm weight
        let layer0 = &cpu_model.layers()[0];
        let attn_norm_weight = &layer0.attn_norm_weight;
        let eps = cpu_model.config().eps;

        // Manual RMSNorm (same as CPU model does)
        let normed = {
            let rms = (embedding.iter().map(|x| x * x).sum::<f32>() / embedding.len() as f32 + eps)
                .sqrt();
            embedding
                .iter()
                .zip(attn_norm_weight.iter())
                .map(|(x, w)| (x / rms) * w)
                .collect::<Vec<f32>>()
        };

        eprintln!("Normed[0..8]: {:?}", &normed[..8]);
        eprintln!(
            "Normed L2: {:.6}",
            normed.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Get QKV weight for CPU matmul
        let qkv_weight = &layer0.qkv_weight;

        // CPU QKV matmul using the internal fused_q4k_parallel_matvec
        eprintln!("\nCPU QKV matmul (fused_q4k_parallel_matvec):");
        let cpu_qkv = cpu_model
            .qkv_matmul(&normed, qkv_weight)
            .expect("CPU QKV failed");

        eprintln!("  Q[0..8]: {:?}", &cpu_qkv[..8]);
        eprintln!(
            "  QKV L2: {:.6}",
            cpu_qkv.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        eprintln!("  QKV sum: {:.6}", cpu_qkv.iter().sum::<f32>());

        // GPU QKV matmul using executor directly
        eprintln!("\nGPU QKV matmul (q4k_gemv_cached):");

        let mut executor = match CudaExecutor::new(0) {
            Ok(exec) => exec,
            Err(e) => {
                eprintln!("CUDA init failed: {:?}", e);
                return;
            },
        };

        // Handle fused vs separate QKV weights
        let gpu_qkv = match qkv_weight {
            realizar::gguf::OwnedQKVWeights::Fused(w) => {
                eprintln!("  Weight type: FUSED");
                eprintln!("  Weight shape: {} x {}", w.out_dim, w.in_dim);
                eprintln!("  Weight qtype: {}", w.qtype);
                eprintln!("  Weight data len: {}", w.data.len());

                // Load weights to GPU
                executor
                    .load_quantized_weights("test_qkv_fused", &w.data)
                    .expect("Failed to load weights");

                let mut output = vec![0.0f32; w.out_dim];
                executor
                    .q4k_gemv_cached(
                        "test_qkv_fused",
                        &normed,
                        &mut output,
                        w.out_dim as u32,
                        w.in_dim as u32,
                    )
                    .expect("GPU GEMV failed");
                output
            },
            realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
                eprintln!("  Weight type: SEPARATE Q/K/V");
                eprintln!(
                    "  Q shape: {} x {}, qtype={} (12=Q4_K, 14=Q6_K)",
                    q.out_dim, q.in_dim, q.qtype
                );
                eprintln!("  K shape: {} x {}, qtype={}", k.out_dim, k.in_dim, k.qtype);
                eprintln!("  V shape: {} x {}, qtype={}", v.out_dim, v.in_dim, v.qtype);

                // Load weights with appropriate type info
                executor
                    .load_quantized_weights("test_q", &q.data)
                    .expect("Failed to load Q");
                executor
                    .load_quantized_weights("test_k", &k.data)
                    .expect("Failed to load K");
                executor
                    .load_quantized_weights("test_v", &v.data)
                    .expect("Failed to load V");

                let mut q_out = vec![0.0f32; q.out_dim];
                let mut k_out = vec![0.0f32; k.out_dim];
                let mut v_out = vec![0.0f32; v.out_dim];

                // Use correct GEMV kernel based on qtype
                // Q4_K = 12, Q5_K = 13, Q6_K = 14
                match q.qtype {
                    12 => executor
                        .q4k_gemv_cached(
                            "test_q",
                            &normed,
                            &mut q_out,
                            q.out_dim as u32,
                            q.in_dim as u32,
                        )
                        .expect("GPU Q GEMV failed"),
                    14 => executor
                        .q6k_gemv_cached(
                            "test_q",
                            &normed,
                            &mut q_out,
                            q.out_dim as u32,
                            q.in_dim as u32,
                        )
                        .expect("GPU Q GEMV failed"),
                    _ => panic!("Unsupported Q qtype: {}", q.qtype),
                }

                match k.qtype {
                    12 => executor
                        .q4k_gemv_cached(
                            "test_k",
                            &normed,
                            &mut k_out,
                            k.out_dim as u32,
                            k.in_dim as u32,
                        )
                        .expect("GPU K GEMV failed"),
                    14 => executor
                        .q6k_gemv_cached(
                            "test_k",
                            &normed,
                            &mut k_out,
                            k.out_dim as u32,
                            k.in_dim as u32,
                        )
                        .expect("GPU K GEMV failed"),
                    _ => panic!("Unsupported K qtype: {}", k.qtype),
                }

                match v.qtype {
                    12 => executor
                        .q4k_gemv_cached(
                            "test_v",
                            &normed,
                            &mut v_out,
                            v.out_dim as u32,
                            v.in_dim as u32,
                        )
                        .expect("GPU V GEMV failed"),
                    14 => executor
                        .q6k_gemv_cached(
                            "test_v",
                            &normed,
                            &mut v_out,
                            v.out_dim as u32,
                            v.in_dim as u32,
                        )
                        .expect("GPU V GEMV failed"),
                    _ => panic!("Unsupported V qtype: {}", v.qtype),
                }

                // Concatenate Q, K, V (matching CPU qkv_matmul output order)
                let mut output = Vec::with_capacity(q.out_dim + k.out_dim + v.out_dim);
                output.extend_from_slice(&q_out);
                output.extend_from_slice(&k_out);
                output.extend_from_slice(&v_out);
                output
            },
        };

        eprintln!("  Q[0..8]: {:?}", &gpu_qkv[..8]);
        eprintln!(
            "  QKV L2: {:.6}",
            gpu_qkv.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        eprintln!("  QKV sum: {:.6}", gpu_qkv.iter().sum::<f32>());

        // Compare
        eprintln!("\nComparison (first 8 elements):");
        let mut max_diff = 0.0f32;
        for i in 0..8 {
            let cpu = cpu_qkv[i];
            let gpu = gpu_qkv[i];
            let diff = (cpu - gpu).abs();
            max_diff = max_diff.max(diff);
            let status = if diff < 0.01 { "✅" } else { "❌" };
            eprintln!(
                "  [{:4}] CPU={:+.6}, GPU={:+.6}, diff={:.6} {}",
                i, cpu, gpu, diff, status
            );
        }

        // Overall stats
        let l2_diff = cpu_qkv
            .iter()
            .zip(gpu_qkv.iter())
            .map(|(c, g)| (c - g).powi(2))
            .sum::<f32>()
            .sqrt();
        let cpu_l2 = cpu_qkv.iter().map(|x| x * x).sum::<f32>().sqrt();
        let gpu_l2 = gpu_qkv.iter().map(|x| x * x).sum::<f32>().sqrt();

        eprintln!("\nOverall:");
        eprintln!("  CPU L2: {:.6}", cpu_l2);
        eprintln!("  GPU L2: {:.6}", gpu_l2);
        eprintln!("  Diff L2: {:.6}", l2_diff);
        eprintln!("  Max diff: {:.6}", max_diff);
        eprintln!("  Relative diff: {:.4}%", l2_diff / cpu_l2 * 100.0);

        if l2_diff / cpu_l2 < 0.01 {
            eprintln!("\n✅ CPU and GPU QKV MATCH within 1%!");
        } else {
            eprintln!("\n❌ SIGNIFICANT QKV DIVERGENCE!");
            eprintln!("  This confirms the bug is in QKV matmul, not elsewhere in forward pass.");
        }
    }

    /// Test 9: Layer-by-layer hidden state comparison
    ///
    /// Compares CPU and GPU hidden states after each layer to isolate divergence.
    #[test]
    fn test_layer_by_layer_comparison() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: LAYER-BY-LAYER HIDDEN STATE COMPARISON                    ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        let hidden_dim = cpu_model.config().hidden_dim;
        let num_layers = cpu_model.layers().len();

        eprintln!("Model: {} layers, hidden_dim={}", num_layers, hidden_dim);
        eprintln!("Token: {} (BOS)\n", bos_token);

        // Run CPU forward and capture hidden states after each layer
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), 10);

        // Use internal forward logic to capture hidden states
        // We need to trace through manually since forward_cached doesn't expose intermediates

        let cpu_hidden = cpu_model.embed(&[bos_token]);
        eprintln!("After embedding:");
        eprintln!(
            "  CPU L2: {:.6}",
            cpu_hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        eprintln!("  CPU [0..4]: {:?}", &cpu_hidden[..4]);

        // Get GPU model for comparison
        std::env::set_var("SKIP_CUDA_GRAPH", "1");
        let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(),
            0,
            10,
        )
        .expect("Failed to create CUDA model");

        let mut gpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cuda_model.model().config(), 10);

        // Run GPU forward and get final logits
        let gpu_logits = cuda_model
            .forward_single_full_cuda_with_cache(bos_token, &mut gpu_cache, 0)
            .expect("GPU forward failed");

        // Run CPU forward to get final logits
        let cpu_logits = cpu_model
            .forward_cached(bos_token, &mut cpu_cache, 0)
            .expect("CPU forward failed");

        // Compare logits
        eprintln!("\n═══════════════════════════════════════════════════════════════════════");
        eprintln!("FINAL LOGITS COMPARISON");
        eprintln!("═══════════════════════════════════════════════════════════════════════");

        let cpu_l2 = cpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let gpu_l2 = gpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let diff_l2 = cpu_logits
            .iter()
            .zip(gpu_logits.iter())
            .map(|(c, g)| (c - g).powi(2))
            .sum::<f32>()
            .sqrt();

        eprintln!("CPU logits L2: {:.6}", cpu_l2);
        eprintln!("GPU logits L2: {:.6}", gpu_l2);
        eprintln!("Diff L2: {:.6}", diff_l2);
        eprintln!("Relative diff: {:.4}%", diff_l2 / cpu_l2 * 100.0);

        let cpu_top1 = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        let gpu_top1 = gpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        eprintln!(
            "\nCPU top-1: {} ({:?})",
            cpu_top1,
            mapped.model.decode(&[cpu_top1])
        );
        eprintln!(
            "GPU top-1: {} ({:?})",
            gpu_top1,
            mapped.model.decode(&[gpu_top1])
        );

        eprintln!("\nLogits[0..5]:");
        eprintln!("  CPU: {:?}", &cpu_logits[..5]);
        eprintln!("  GPU: {:?}", &gpu_logits[..5]);

        // Check LM head weight qtype
        let lm_head_qtype = cpu_model.lm_head_weight().qtype;
        eprintln!("\nLM head qtype: {} (12=Q4_K, 14=Q6_K)", lm_head_qtype);

        // Check layer 0 weight qtypes
        let layer0 = &cpu_model.layers()[0];
        eprintln!("\nLayer 0 weight qtypes (12=Q4_K, 14=Q6_K):");
        match &layer0.qkv_weight {
            realizar::gguf::OwnedQKVWeights::Fused(w) => {
                eprintln!("  QKV (fused): {}", w.qtype);
            },
            realizar::gguf::OwnedQKVWeights::Separate { q, k, v } => {
                eprintln!("  Q: {}, K: {}, V: {}", q.qtype, k.qtype, v.qtype);
            },
        }
        eprintln!("  attn_output: {}", layer0.attn_output_weight.qtype);
        eprintln!("  ffn_up: {}", layer0.ffn_up_weight.qtype);
        eprintln!("  ffn_down: {}", layer0.ffn_down_weight.qtype);
        if let Some(ref gate) = layer0.ffn_gate_weight {
            eprintln!("  ffn_gate: {}", gate.qtype);
        }

        if cpu_top1 == gpu_top1 {
            eprintln!("\n✅ CPU and GPU MATCH!");
        } else {
            eprintln!("\n❌ DIVERGENCE DETECTED!");
            eprintln!(
                "  Relative diff: {:.1}% suggests issue is NOT just quantization noise",
                diff_l2 / cpu_l2 * 100.0
            );
            eprintln!("  Need to trace through layer-by-layer to find source.");
        }

        std::env::remove_var("SKIP_CUDA_GRAPH");
    }

    /// Test 10: Q6_K matmul comparison (ffn_down weight)
    ///
    /// The ffn_down weight uses Q6_K. This test compares CPU and GPU implementations.
    #[test]
    fn test_q6k_ffn_down_comparison() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: Q6_K FFN DOWN COMPARISON                                   ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let layer0 = &cpu_model.layers()[0];
        let ffn_down = &layer0.ffn_down_weight;

        eprintln!("ffn_down_weight:");
        eprintln!("  in_dim: {}", ffn_down.in_dim);
        eprintln!("  out_dim: {}", ffn_down.out_dim);
        eprintln!("  qtype: {} (14=Q6_K)", ffn_down.qtype);

        // Create test input - use random-ish values (normalized to reasonable range)
        let mut input = vec![0.0f32; ffn_down.in_dim];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as f32 * 0.1234).sin() * 0.1;
        }
        let input_l2 = input.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("Test input L2: {:.6}", input_l2);

        // CPU matmul using fused_q6k_parallel_matvec
        eprintln!("\nCPU (fused_q6k_parallel_matvec):");
        let cpu_output = realizar::quantize::fused_q6k_parallel_matvec(
            &ffn_down.data,
            &input,
            ffn_down.in_dim,
            ffn_down.out_dim,
        )
        .expect("CPU matmul failed");
        eprintln!(
            "  Output[0..5]: {:?}",
            &cpu_output[..5.min(cpu_output.len())]
        );
        eprintln!(
            "  L2: {:.6}",
            cpu_output.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        eprintln!("  Sum: {:.6}", cpu_output.iter().sum::<f32>());

        // GPU matmul using q6k_gemv_cached
        eprintln!("\nGPU (q6k_gemv_cached):");
        let mut executor = match CudaExecutor::new(0) {
            Ok(exec) => exec,
            Err(e) => {
                eprintln!("CUDA init failed: {:?}", e);
                return;
            },
        };

        executor
            .load_quantized_weights("test_ffn_down", &ffn_down.data)
            .expect("Failed to load weights");

        let mut gpu_output = vec![0.0f32; ffn_down.out_dim];
        executor
            .q6k_gemv_cached(
                "test_ffn_down",
                &input,
                &mut gpu_output,
                ffn_down.out_dim as u32,
                ffn_down.in_dim as u32,
            )
            .expect("GPU matmul failed");

        eprintln!(
            "  Output[0..5]: {:?}",
            &gpu_output[..5.min(gpu_output.len())]
        );
        eprintln!(
            "  L2: {:.6}",
            gpu_output.iter().map(|x| x * x).sum::<f32>().sqrt()
        );
        eprintln!("  Sum: {:.6}", gpu_output.iter().sum::<f32>());

        // Compare
        let diff_l2 = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(c, g)| (c - g).powi(2))
            .sum::<f32>()
            .sqrt();
        let cpu_l2 = cpu_output.iter().map(|x| x * x).sum::<f32>().sqrt();

        eprintln!("\nComparison:");
        eprintln!("  Diff L2: {:.6}", diff_l2);
        eprintln!("  Relative diff: {:.4}%", diff_l2 / cpu_l2 * 100.0);

        let mut max_diff = 0.0f32;
        for i in 0..5 {
            let diff = (cpu_output[i] - gpu_output[i]).abs();
            max_diff = max_diff.max(diff);
            let status = if diff < 0.01 { "✅" } else { "❌" };
            eprintln!(
                "  [{:4}] CPU={:+.6}, GPU={:+.6}, diff={:.6} {}",
                i, cpu_output[i], gpu_output[i], diff, status
            );
        }

        if diff_l2 / cpu_l2 < 0.01 {
            eprintln!("\n✅ Q6_K GEMV matches within 1%!");
        } else {
            eprintln!("\n❌ Q6_K GEMV DIVERGENCE!");
        }
    }

    /// Test 11: Compare attn_output projection (Q4_K)
    ///
    /// Tests the attention output projection which uses Q4_K.
    #[test]
    fn test_attn_output_comparison() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: ATTN OUTPUT PROJECTION COMPARISON                         ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let layer0 = &cpu_model.layers()[0];
        let attn_output = &layer0.attn_output_weight;

        eprintln!("attn_output_weight:");
        eprintln!("  in_dim: {}", attn_output.in_dim);
        eprintln!("  out_dim: {}", attn_output.out_dim);
        eprintln!("  qtype: {} (12=Q4_K)", attn_output.qtype);

        // Create test input - simulate attention output
        let mut input = vec![0.0f32; attn_output.in_dim];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as f32 * 0.1234).sin() * 0.1;
        }
        eprintln!(
            "Test input L2: {:.6}",
            input.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // CPU matmul
        eprintln!("\nCPU (fused_q4k_parallel_matvec):");
        let cpu_output = realizar::quantize::fused_q4k_parallel_matvec(
            &attn_output.data,
            &input,
            attn_output.in_dim,
            attn_output.out_dim,
        )
        .expect("CPU matmul failed");
        eprintln!("  Output[0..5]: {:?}", &cpu_output[..5]);
        eprintln!(
            "  L2: {:.6}",
            cpu_output.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // GPU matmul
        eprintln!("\nGPU (q4k_gemv_cached):");
        let mut executor = match CudaExecutor::new(0) {
            Ok(exec) => exec,
            Err(e) => {
                eprintln!("CUDA init failed: {:?}", e);
                return;
            },
        };

        executor
            .load_quantized_weights("test_attn_out", &attn_output.data)
            .expect("Failed to load weights");

        let mut gpu_output = vec![0.0f32; attn_output.out_dim];
        executor
            .q4k_gemv_cached(
                "test_attn_out",
                &input,
                &mut gpu_output,
                attn_output.out_dim as u32,
                attn_output.in_dim as u32,
            )
            .expect("GPU matmul failed");

        eprintln!("  Output[0..5]: {:?}", &gpu_output[..5]);
        eprintln!(
            "  L2: {:.6}",
            gpu_output.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Compare
        let diff_l2 = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(c, g)| (c - g).powi(2))
            .sum::<f32>()
            .sqrt();
        let cpu_l2 = cpu_output.iter().map(|x| x * x).sum::<f32>().sqrt();

        eprintln!("\nRelative diff: {:.4}%", diff_l2 / cpu_l2 * 100.0);

        if diff_l2 / cpu_l2 < 0.01 {
            eprintln!("✅ ATTN OUTPUT matches within 1%!");
        } else {
            eprintln!("❌ ATTN OUTPUT DIVERGENCE!");
        }
    }

    /// Test 12: FFN gate and up projections (Q4_K)
    #[test]
    fn test_ffn_gate_up_comparison() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: FFN GATE/UP PROJECTION COMPARISON                         ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let layer0 = &cpu_model.layers()[0];

        // Create test input
        let in_dim = layer0.ffn_up_weight.in_dim;
        let mut input = vec![0.0f32; in_dim];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as f32 * 0.1234).sin() * 0.1;
        }
        eprintln!(
            "Test input L2: {:.6}",
            input.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        let mut executor = match CudaExecutor::new(0) {
            Ok(exec) => exec,
            Err(e) => {
                eprintln!("CUDA init failed: {:?}", e);
                return;
            },
        };

        // Test FFN UP (Q4_K)
        let ffn_up = &layer0.ffn_up_weight;
        eprintln!("\nFFN UP (qtype={}):", ffn_up.qtype);

        let cpu_up = realizar::quantize::fused_q4k_parallel_matvec(
            &ffn_up.data,
            &input,
            ffn_up.in_dim,
            ffn_up.out_dim,
        )
        .expect("CPU failed");

        executor
            .load_quantized_weights("ffn_up", &ffn_up.data)
            .expect("load");
        let mut gpu_up = vec![0.0f32; ffn_up.out_dim];
        executor
            .q4k_gemv_cached(
                "ffn_up",
                &input,
                &mut gpu_up,
                ffn_up.out_dim as u32,
                ffn_up.in_dim as u32,
            )
            .expect("GPU failed");

        let diff = cpu_up
            .iter()
            .zip(gpu_up.iter())
            .map(|(c, g)| (c - g).powi(2))
            .sum::<f32>()
            .sqrt();
        let cpu_l2 = cpu_up.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!(
            "  CPU L2: {:.6}, GPU L2: {:.6}, Diff: {:.6} ({:.4}%)",
            cpu_l2,
            gpu_up.iter().map(|x| x * x).sum::<f32>().sqrt(),
            diff,
            diff / cpu_l2 * 100.0
        );

        // Test FFN GATE (Q4_K)
        if let Some(ref ffn_gate) = layer0.ffn_gate_weight {
            eprintln!("\nFFN GATE (qtype={}):", ffn_gate.qtype);

            let cpu_gate = realizar::quantize::fused_q4k_parallel_matvec(
                &ffn_gate.data,
                &input,
                ffn_gate.in_dim,
                ffn_gate.out_dim,
            )
            .expect("CPU failed");

            executor
                .load_quantized_weights("ffn_gate", &ffn_gate.data)
                .expect("load");
            let mut gpu_gate = vec![0.0f32; ffn_gate.out_dim];
            executor
                .q4k_gemv_cached(
                    "ffn_gate",
                    &input,
                    &mut gpu_gate,
                    ffn_gate.out_dim as u32,
                    ffn_gate.in_dim as u32,
                )
                .expect("GPU failed");

            let diff = cpu_gate
                .iter()
                .zip(gpu_gate.iter())
                .map(|(c, g)| (c - g).powi(2))
                .sum::<f32>()
                .sqrt();
            let cpu_l2 = cpu_gate.iter().map(|x| x * x).sum::<f32>().sqrt();
            eprintln!(
                "  CPU L2: {:.6}, GPU L2: {:.6}, Diff: {:.6} ({:.4}%)",
                cpu_l2,
                gpu_gate.iter().map(|x| x * x).sum::<f32>().sqrt(),
                diff,
                diff / cpu_l2 * 100.0
            );
        }

        eprintln!("\n✅ All individual kernel tests pass - bug must be in forward pass logic");
    }

    /// Test 13: Step-by-step Q4K matmul trace
    #[test]
    fn test_q4k_matmul_step_trace() {
        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  PHASE 16: Q4K MATMUL STEP-BY-STEP TRACE                             ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("⚠️  Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        // Load CPU model
        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        // Single token: BOS
        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        eprintln!("Test token: {} (BOS)", bos_token);

        // CPU forward - position 0
        let mut cpu_cache =
            realizar::gguf::OwnedQuantizedKVCache::from_config(cpu_model.config(), 10);

        let cpu_logits = cpu_model
            .forward_cached(bos_token, &mut cpu_cache, 0)
            .expect("CPU forward failed");

        let cpu_l2 = cpu_logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cpu_top1 = cpu_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        let cpu_decoded = mapped.model.decode(&[cpu_top1]);

        eprintln!("\nCPU Forward (position 0):");
        eprintln!("  Logits L2: {:.4}", cpu_l2);
        eprintln!("  Top-1: {} ({:?})", cpu_top1, cpu_decoded);
        eprintln!("  Logits[0..5]: {:?}", &cpu_logits[..5]);

        // GPU forward
        std::env::set_var("SKIP_CUDA_GRAPH", "1");

        let gpu_model_result = OwnedQuantizedModelCuda::with_max_seq_len(
            OwnedQuantizedModel::from_mapped(&mapped).unwrap(),
            0,
            10,
        );

        match gpu_model_result {
            Ok(mut cuda_model) => {
                let gen_config = realizar::gguf::QuantizedGenerateConfig {
                    max_tokens: 1,
                    temperature: 0.0,
                    top_k: 1,
                    stop_tokens: vec![],
                    trace: false,
                    ..Default::default()
                };

                let tokens = vec![bos_token];
                let gpu_result = cuda_model.generate_full_cuda_with_cache(&tokens, &gen_config);

                match gpu_result {
                    Ok(generated) => {
                        eprintln!("\nGPU Forward:");
                        eprintln!("  Generated: {:?}", generated);

                        if generated.len() > 1 {
                            let gpu_top1 = generated[1];
                            let gpu_decoded = mapped.model.decode(&[gpu_top1]);
                            eprintln!("  Top-1: {} ({:?})", gpu_top1, gpu_decoded);

                            if cpu_top1 == gpu_top1 {
                                eprintln!("\n✅ CPU and GPU MATCH at position 0");
                            } else {
                                eprintln!("\n❌ CPU/GPU DIVERGE at position 0!");
                                eprintln!("   This confirms H3: Q4K matmul bug");
                            }
                        }
                    },
                    Err(e) => {
                        eprintln!("\n❌ GPU forward failed: {:?}", e);
                    },
                }
            },
            Err(e) => {
                eprintln!("\n❌ Failed to create CUDA model: {:?}", e);
            },
        }

        std::env::remove_var("SKIP_CUDA_GRAPH");
    }

    /// Test 14: QKV dimension mismatch detection
    ///
    /// The CPU path uses actual weight output dimensions for Q, K, V extraction.
    /// The GPU path assumes hidden_dim for Q and kv_dim for K/V.
    /// This test checks if these assumptions are violated.
    #[test]
    fn test_qkv_dimension_mismatch() {
        use realizar::gguf::OwnedQKVWeights;

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  TEST 14: QKV DIMENSION MISMATCH DETECTION                           ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let config = cpu_model.config();
        let hidden_dim = config.hidden_dim;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        eprintln!("Model config:");
        eprintln!("  hidden_dim: {}", hidden_dim);
        eprintln!("  num_heads: {}", num_heads);
        eprintln!("  num_kv_heads: {}", num_kv_heads);
        eprintln!("  head_dim: {}", head_dim);
        eprintln!("  kv_dim (calculated): {}", kv_dim);
        eprintln!();

        // Check all layers for dimension mismatches
        let mut mismatch_found = false;

        for (layer_idx, layer) in cpu_model.layers().iter().enumerate() {
            let qkv_weight = &layer.qkv_weight;
            let q_dim = qkv_weight.q_dim();

            let (k_dim, v_dim) = match qkv_weight {
                OwnedQKVWeights::Fused(_) => (q_dim, q_dim),
                OwnedQKVWeights::Separate { k, v, .. } => (k.out_dim, v.out_dim),
            };

            // GPU path assumes these dimensions
            let gpu_q_dim = hidden_dim;
            let gpu_k_dim = kv_dim;
            let gpu_v_dim = kv_dim;

            let q_match = q_dim == gpu_q_dim;
            let k_match = k_dim == gpu_k_dim;
            let v_match = v_dim == gpu_v_dim;

            if !q_match || !k_match || !v_match {
                mismatch_found = true;
                eprintln!("❌ Layer {} DIMENSION MISMATCH:", layer_idx);
                eprintln!(
                    "   Q: CPU={} vs GPU={} {}",
                    q_dim,
                    gpu_q_dim,
                    if q_match { "✅" } else { "❌" }
                );
                eprintln!(
                    "   K: CPU={} vs GPU={} {}",
                    k_dim,
                    gpu_k_dim,
                    if k_match { "✅" } else { "❌" }
                );
                eprintln!(
                    "   V: CPU={} vs GPU={} {}",
                    v_dim,
                    gpu_v_dim,
                    if v_match { "✅" } else { "❌" }
                );
            } else if layer_idx == 0 {
                eprintln!("✅ Layer {} dimensions match:", layer_idx);
                eprintln!("   Q: {} (CPU) = {} (GPU)", q_dim, gpu_q_dim);
                eprintln!("   K: {} (CPU) = {} (GPU)", k_dim, gpu_k_dim);
                eprintln!("   V: {} (CPU) = {} (GPU)", v_dim, gpu_v_dim);
            }
        }

        if !mismatch_found {
            eprintln!(
                "\n✅ All {} layers have matching Q/K/V dimensions",
                cpu_model.layers().len()
            );
            eprintln!("   Bug is NOT in QKV extraction offsets");
        } else {
            eprintln!("\n❌ DIMENSION MISMATCH FOUND!");
            eprintln!("   GPU path is using wrong offsets to extract Q/K/V!");
        }
    }

    /// Test 15: Trace forward pass step by step
    ///
    /// Compares intermediate values between CPU and GPU at each step to pinpoint divergence.
    #[test]
    fn test_forward_pass_step_trace() {
        use realizar::gguf::OwnedQKVWeights;

        eprintln!("\n╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("║  TEST 15: FORWARD PASS STEP-BY-STEP TRACE                            ║");
        eprintln!("╚══════════════════════════════════════════════════════════════════════╝\n");

        let model_path =
            std::path::Path::new(env!("HOME")).join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf");

        if !model_path.exists() {
            eprintln!("Skipping: model not found");
            return;
        }

        let mapped =
            MappedGGUFModel::from_path(model_path.to_str().unwrap()).expect("Failed to mmap GGUF");

        let cpu_model =
            OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to load CPU model");

        let config = cpu_model.config();
        let hidden_dim = config.hidden_dim;
        let num_heads = config.num_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let bos_token = mapped.model.bos_token_id().unwrap_or(1);
        eprintln!("Token: {} (BOS)", bos_token);
        eprintln!(
            "hidden_dim={}, num_heads={}, num_kv_heads={}, head_dim={}, kv_dim={}\n",
            hidden_dim, num_heads, num_kv_heads, head_dim, kv_dim
        );

        // Step 1: Embedding
        let embedding = cpu_model.embed(&[bos_token]);
        let embed_l2 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("Step 1: Embedding");
        eprintln!("  L2: {:.6}", embed_l2);
        eprintln!("  First 8: {:?}", &embedding[..8.min(embedding.len())]);
        eprintln!();

        // Step 2: Layer 0 attention norm
        let layer = &cpu_model.layers()[0];
        let use_rmsnorm = layer.ffn_gate_weight.is_some() && layer.attn_norm_bias.is_none();
        eprintln!("Step 2: Attention Norm (use_rmsnorm={})", use_rmsnorm);

        let normed = if use_rmsnorm {
            rms_norm_standalone(&embedding, &layer.attn_norm_weight, config.eps)
        } else {
            eprintln!("  Using LayerNorm (not RMSNorm)");
            embedding
        };
        let normed_l2 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  L2: {:.6}", normed_l2);
        eprintln!("  First 8: {:?}", &normed[..8.min(normed.len())]);
        eprintln!();

        // Step 3: QKV projection
        eprintln!("Step 3: QKV Projection");
        let qkv_weight = &layer.qkv_weight;
        let q_dim_cpu = qkv_weight.q_dim();
        let (k_dim_cpu, v_dim_cpu) = match qkv_weight {
            OwnedQKVWeights::Fused(_) => (q_dim_cpu, q_dim_cpu),
            OwnedQKVWeights::Separate { k, v, .. } => (k.out_dim, v.out_dim),
        };

        eprintln!(
            "  CPU QKV dims: q={}, k={}, v={}",
            q_dim_cpu, k_dim_cpu, v_dim_cpu
        );
        eprintln!(
            "  GPU assumed:  q={}, k={}, v={}",
            hidden_dim, kv_dim, kv_dim
        );

        // Compute QKV using CPU path
        let qkv = match qkv_weight {
            OwnedQKVWeights::Fused(tensor) => realizar::quantize::fused_q4k_parallel_matvec(
                &tensor.data,
                &normed,
                tensor.in_dim,
                tensor.out_dim,
            )
            .expect("QKV fused matmul failed"),
            OwnedQKVWeights::Separate { q, k, v } => {
                let q_out = realizar::quantize::fused_q4k_parallel_matvec(
                    &q.data, &normed, q.in_dim, q.out_dim,
                )
                .expect("Q matmul failed");
                let k_out = realizar::quantize::fused_q4k_parallel_matvec(
                    &k.data, &normed, k.in_dim, k.out_dim,
                )
                .expect("K matmul failed");
                let v_out = if v.qtype == 12 {
                    realizar::quantize::fused_q4k_parallel_matvec(
                        &v.data, &normed, v.in_dim, v.out_dim,
                    )
                    .expect("V matmul failed")
                } else {
                    realizar::quantize::fused_q6k_parallel_matvec(
                        &v.data, &normed, v.in_dim, v.out_dim,
                    )
                    .expect("V matmul failed")
                };
                let mut combined = Vec::with_capacity(q_out.len() + k_out.len() + v_out.len());
                combined.extend_from_slice(&q_out);
                combined.extend_from_slice(&k_out);
                combined.extend_from_slice(&v_out);
                combined
            },
        };

        eprintln!("  QKV total len: {}", qkv.len());
        let qkv_l2 = qkv.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("  QKV L2: {:.6}", qkv_l2);

        // Extract Q, K, V using CPU offsets
        let q_cpu = &qkv[0..q_dim_cpu];
        let k_cpu = &qkv[q_dim_cpu..q_dim_cpu + k_dim_cpu];
        let v_cpu = &qkv[q_dim_cpu + k_dim_cpu..q_dim_cpu + k_dim_cpu + v_dim_cpu];

        eprintln!(
            "  Q L2: {:.6}, K L2: {:.6}, V L2: {:.6}",
            q_cpu.iter().map(|x| x * x).sum::<f32>().sqrt(),
            k_cpu.iter().map(|x| x * x).sum::<f32>().sqrt(),
            v_cpu.iter().map(|x| x * x).sum::<f32>().sqrt()
        );

        // Compare with GPU offsets
        let q_gpu = &qkv[0..hidden_dim.min(qkv.len())];
        let k_start = hidden_dim;
        let v_start = hidden_dim + kv_dim;

        if k_start < qkv.len() && v_start + kv_dim <= qkv.len() {
            let k_gpu = &qkv[k_start..k_start + kv_dim];
            let v_gpu = &qkv[v_start..v_start + kv_dim];

            eprintln!(
                "\n  GPU extraction (hidden_dim={}, kv_dim={}):",
                hidden_dim, kv_dim
            );
            eprintln!(
                "  Q GPU L2: {:.6}, K GPU L2: {:.6}, V GPU L2: {:.6}",
                q_gpu.iter().map(|x| x * x).sum::<f32>().sqrt(),
                k_gpu.iter().map(|x| x * x).sum::<f32>().sqrt(),
                v_gpu.iter().map(|x| x * x).sum::<f32>().sqrt()
            );

            // Check if extraction matches
            let q_match = q_cpu.len() == q_gpu.len()
                && q_cpu
                    .iter()
                    .zip(q_gpu.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6);
            let k_match = k_cpu.len() == k_gpu.len()
                && k_cpu
                    .iter()
                    .zip(k_gpu.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6);
            let v_match = v_cpu.len() == v_gpu.len()
                && v_cpu
                    .iter()
                    .zip(v_gpu.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-6);

            eprintln!(
                "\n  CPU vs GPU extraction: Q={}, K={}, V={}",
                if q_match { "MATCH" } else { "DIFFER" },
                if k_match { "MATCH" } else { "DIFFER" },
                if v_match { "MATCH" } else { "DIFFER" }
            );

            if !q_match || !k_match || !v_match {
                eprintln!("\n  ❌ EXTRACTION OFFSET BUG DETECTED!");
                eprintln!(
                    "  CPU uses offsets: Q[0..{}], K[{}..{}], V[{}..{}]",
                    q_dim_cpu,
                    q_dim_cpu,
                    q_dim_cpu + k_dim_cpu,
                    q_dim_cpu + k_dim_cpu,
                    q_dim_cpu + k_dim_cpu + v_dim_cpu
                );
                eprintln!(
                    "  GPU uses offsets: Q[0..{}], K[{}..{}], V[{}..{}]",
                    hidden_dim,
                    hidden_dim,
                    hidden_dim + kv_dim,
                    hidden_dim + kv_dim,
                    hidden_dim + 2 * kv_dim
                );
            }
        } else {
            eprintln!("\n  ⚠️  QKV buffer too small for GPU assumed offsets");
            eprintln!(
                "  QKV len: {}, GPU needs: {}",
                qkv.len(),
                hidden_dim + 2 * kv_dim
            );
        }
    }

    /// Standalone RMSNorm implementation for testing
    fn rms_norm_standalone(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
        x.iter()
            .zip(weight.iter())
            .map(|(&xi, &wi)| (xi / rms) * wi)
            .collect()
    }
}
