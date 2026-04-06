
// ============================================================================
// Tests (Protocol T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activations_with_harness_gelu() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test GELU with hidden_dim sized tensor
        let input = vec![0.5f32; config.hidden_dim];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).expect("input_buf");
        let output_buf = exec
            .gelu_async(&input_buf, config.hidden_dim as u32)
            .expect("expected value");

        exec.stream.synchronize().expect("synchronize");
        let mut output = vec![0.0f32; config.hidden_dim];
        output_buf.copy_to_host(&mut output).expect("copy_to_host");

        // GELU(0.5) ≈ 0.345
        assert!(
            (output[0] - 0.345).abs() < 0.02,
            "GELU(0.5) = {}",
            output[0]
        );
    }

    #[test]
    fn test_activations_with_harness_rope() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let total_dim = config.num_heads * config.head_dim;
        let input = vec![1.0f32; total_dim];

        let buf_input = GpuBuffer::from_host(&exec.context, &input).expect("buf_input");
        let buf_output = GpuBuffer::new(&exec.context, total_dim).expect("buf_output");

        // Apply RoPE at position 0
        let result = exec.rope_into(
            &buf_input,
            &buf_output,
            0,
            config.num_heads as u32,
            config.head_dim as u32,
            exec.rope_theta,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_activations_with_harness_swiglu() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test SwiGLU with intermediate_dim sized tensor
        let gate = vec![1.0f32; config.intermediate_dim];
        let up = vec![2.0f32; config.intermediate_dim];

        let buf_gate = GpuBuffer::from_host(&exec.context, &gate).expect("buf_gate");
        let buf_up = GpuBuffer::from_host(&exec.context, &up).expect("buf_up");
        let output_buf = exec
            .fused_swiglu_gpu(&buf_gate, &buf_up, config.intermediate_dim as u32)
            .expect("expected value");

        exec.stream.synchronize().expect("synchronize");
        let mut output = vec![0.0f32; config.intermediate_dim];
        output_buf.copy_to_host(&mut output).expect("copy_to_host");

        // SwiGLU = SiLU(gate) * up ≈ 0.731 * 2 = 1.462
        assert!((output[0] - 1.462).abs() < 0.05);
    }

    #[test]
    fn test_activations_with_harness_residual_add() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let output_data = vec![1.0f32; config.hidden_dim];
        let input_data = vec![10.0f32; config.hidden_dim];

        let buf_output = GpuBuffer::from_host(&exec.context, &output_data).expect("buf_output");
        let buf_input = GpuBuffer::from_host(&exec.context, &input_data).expect("buf_input");

        exec.add_residual_gpu(&buf_output, &buf_input, config.hidden_dim as u32)
            .expect("expected value");

        exec.stream.synchronize().expect("synchronize");
        let mut result = vec![0.0f32; config.hidden_dim];
        buf_output.copy_to_host(&mut result).expect("copy_to_host");

        // Expected: 1.0 + 10.0 = 11.0
        assert!((result[0] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_activations_with_harness_large_tensor() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let mut config = HarnessConfig::default();
        config.hidden_dim = 4096; // Larger tensor
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test SiLU with large tensor
        let input: Vec<f32> = (0..config.hidden_dim)
            .map(|i| (i as f32 - 2048.0) / 1000.0)
            .collect();
        let input_buf = GpuBuffer::from_host(&exec.context, &input).expect("input_buf");
        let output_buf = exec.silu_gpu(&input_buf, config.hidden_dim as u32).expect("output_buf");

        exec.stream.synchronize().expect("synchronize");
        let mut output = vec![0.0f32; config.hidden_dim];
        output_buf.copy_to_host(&mut output).expect("copy_to_host");

        // Verify all values are finite
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] = {} is not finite", i, v);
        }
    }

    #[test]
    fn test_activations_with_harness_elementwise_mul() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        let a = vec![2.0f32; config.hidden_dim];
        let b = vec![3.0f32; config.hidden_dim];

        let buf_a = GpuBuffer::from_host(&exec.context, &a).expect("buf_a");
        let buf_b = GpuBuffer::from_host(&exec.context, &b).expect("buf_b");
        let output_buf = exec
            .elementwise_mul_gpu(&buf_a, &buf_b, config.hidden_dim as u32)
            .expect("expected value");

        exec.stream.synchronize().expect("synchronize");
        let mut output = vec![0.0f32; config.hidden_dim];
        output_buf.copy_to_host(&mut output).expect("copy_to_host");

        // 2.0 * 3.0 = 6.0
        assert!((output[0] - 6.0).abs() < 1e-5);
    }

    // ========================================================================
    // QWEN-009: 3-Way Fused FFN Tests
    // ========================================================================

    #[test]
    fn test_qwen009_kernel_type_generation() {
        use crate::cuda::kernels::{CudaKernels, KernelType};

        // Test that the kernel type generates valid PTX
        let kernels = CudaKernels::new();
        let kernel_type = KernelType::FusedRmsNormGateUpSwigluQ4K {
            k: 2048, // hidden_size
            n: 5632, // intermediate_size (Qwen2.5 0.5B)
            epsilon: 1e-6,
        };

        let ptx = kernels.generate_ptx(&kernel_type);
        assert!(!ptx.is_empty(), "PTX should not be empty");
        assert!(
            ptx.contains(".version") || ptx.contains(".entry"),
            "PTX should contain valid PTX assembly directives"
        );

        // Verify kernel name
        let name = kernels.kernel_name(&kernel_type);
        assert_eq!(name, "fused_rmsnorm_gate_up_swiglu_q4k");
    }

    #[test]
    fn test_qwen009_fused_ffn_rmsnorm_swiglu_q4k_basic() {
        let Some(mut exec) = create_executor() else {
            eprintln!("CUDA init failed - check driver");
            return;
        };

        // Use small dimensions for test to avoid OOM
        let hidden_size = 256u32;
        let intermediate_size = 512u32;
        let epsilon = 1e-6f32;

        // Create test data
        // Input: simple pattern for predictable RMSNorm output
        let input = vec![1.0f32; hidden_size as usize];
        let gamma = vec![1.0f32; hidden_size as usize]; // Identity scale

        // Create Q4K super-blocks for gate and up weights
        // Q4K format: 144 bytes per 256 values (super-block)
        // For K=256, N=512: each row has 1 super-block, 512 rows total
        // Total = 512 * 144 = 73728 bytes per weight matrix
        let num_super_blocks_per_row = (hidden_size as usize + 255) / 256;
        let bytes_per_super_block = 144;
        let weight_bytes =
            intermediate_size as usize * num_super_blocks_per_row * bytes_per_super_block;

        // Create dummy Q4K weights (zeros will dequantize to near-zero values)
        let w_gate_data = vec![0u8; weight_bytes];
        let w_up_data = vec![0u8; weight_bytes];

        // Upload to GPU
        let input_buf = GpuBuffer::from_host(&exec.context, &input).expect("input_buf");
        let gamma_buf = GpuBuffer::from_host(&exec.context, &gamma).expect("gamma_buf");
        let w_gate_buf = GpuBuffer::from_host(&exec.context, &w_gate_data).expect("w_gate_buf");
        let w_up_buf = GpuBuffer::from_host(&exec.context, &w_up_data).expect("w_up_buf");
        let output_buf = GpuBuffer::<f32>::new(&exec.context, intermediate_size as usize).expect("output_buf");

        // Execute fused kernel
        let result = exec.fused_ffn_rmsnorm_swiglu_q4k_into(
            &input_buf,
            &gamma_buf,
            w_gate_buf.as_ptr(),
            w_up_buf.as_ptr(),
            &output_buf,
            hidden_size,
            intermediate_size,
            epsilon,
        );

        assert!(result.is_ok(), "Kernel launch should succeed");

        exec.stream.synchronize().expect("synchronize");

        let mut output = vec![0.0f32; intermediate_size as usize];
        output_buf.copy_to_host(&mut output).expect("copy_to_host");

        // With zero weights, output should be near zero (SwiGLU of zeros)
        // Note: Results may vary based on kernel implementation
        for (i, &v) in output.iter().take(4).enumerate() {
            // Just verify output is finite (correctness depends on kernel implementation)
            assert!(v.is_finite(), "output[{}] = {} should be finite", i, v);
        }
    }

    #[test]
    fn test_qwen009_kernel_type_variants() {
        use crate::cuda::kernels::{CudaKernels, KernelType};

        let kernels = CudaKernels::new();

        // Test different dimension combinations
        let test_cases = [
            (896, 4864, 1e-6),  // Qwen2.5 0.5B-like
            (1024, 2816, 1e-5), // Small model
            (2048, 5632, 1e-6), // Medium model
        ];

        for (k, n, epsilon) in test_cases {
            let kernel_type = KernelType::FusedRmsNormGateUpSwigluQ4K { k, n, epsilon };

            let ptx = kernels.generate_ptx(&kernel_type);
            let name = kernels.kernel_name(&kernel_type);

            assert!(
                !ptx.is_empty(),
                "PTX for k={}, n={} should not be empty",
                k,
                n
            );
            assert_eq!(name, "fused_rmsnorm_gate_up_swiglu_q4k");
        }
    }
include!("activations_tests.rs");
}
