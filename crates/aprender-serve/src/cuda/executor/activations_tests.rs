
    /// Helper to create CudaExecutor for tests
    fn create_executor() -> Option<CudaExecutor> {
        CudaExecutor::new(0).ok()
    }

    // === SiLU activation tests ===

    #[test]
    fn test_silu_gpu_basic() {
        let Some(mut exec) = create_executor() else {
            eprintln!("CUDA init failed - check driver");
            return;
        };

        let input = vec![0.0f32, 1.0, -1.0, 2.0];
        let n = input.len() as u32;

        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = exec.silu_gpu(&input_buf, n).unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; input.len()];
        output_buf.copy_to_host(&mut output).unwrap();

        // SiLU(x) = x * sigmoid(x)
        // SiLU(0) = 0
        assert!(output[0].abs() < 1e-5, "SiLU(0) = {}", output[0]);
        // SiLU(1) ≈ 0.731
        assert!((output[1] - 0.731).abs() < 0.01, "SiLU(1) = {}", output[1]);
        // SiLU(-1) ≈ -0.269
        assert!(
            (output[2] - (-0.269)).abs() < 0.01,
            "SiLU(-1) = {}",
            output[2]
        );
    }

    #[test]
    fn test_silu_gpu_large() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) / 100.0).collect();

        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = exec.silu_gpu(&input_buf, n as u32).unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; n];
        output_buf.copy_to_host(&mut output).unwrap();

        // Verify output is finite
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] = {} is not finite", i, v);
        }
    }

    // === GELU activation tests ===

    #[test]
    fn test_gelu_async_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let input = vec![0.0f32, 1.0, -1.0, 2.0];
        let n = input.len() as u32;

        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = exec.gelu_async(&input_buf, n).unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; input.len()];
        output_buf.copy_to_host(&mut output).unwrap();

        // GELU(0) = 0
        assert!(output[0].abs() < 1e-5, "GELU(0) = {}", output[0]);
        // GELU(1) ≈ 0.841
        assert!((output[1] - 0.841).abs() < 0.02, "GELU(1) = {}", output[1]);
        // GELU(-1) ≈ -0.159
        assert!(
            (output[2] - (-0.159)).abs() < 0.02,
            "GELU(-1) = {}",
            output[2]
        );
    }

    #[test]
    fn test_gelu_async_large() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let n = 2048;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 1024.0) / 200.0).collect();

        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = exec.gelu_async(&input_buf, n as u32).unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; n];
        output_buf.copy_to_host(&mut output).unwrap();

        // Verify GELU properties: GELU(x) ≈ x for large positive x
        let last_input = input[n - 1]; // ~5.1
        let last_output = output[n - 1];
        assert!(
            (last_output - last_input).abs() < 0.1,
            "GELU({}) = {} should be ~{}",
            last_input,
            last_output,
            last_input
        );
    }

    // === Elementwise multiply tests ===

    #[test]
    fn test_elementwise_mul_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0];
        let n = a.len() as u32;

        let buf_a = GpuBuffer::from_host(&exec.context, &a).unwrap();
        let buf_b = GpuBuffer::from_host(&exec.context, &b).unwrap();
        let output_buf = exec.elementwise_mul_gpu(&buf_a, &buf_b, n).unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; a.len()];
        output_buf.copy_to_host(&mut output).unwrap();

        // Expected: [2, 6, 12, 20]
        assert!((output[0] - 2.0).abs() < 1e-5);
        assert!((output[1] - 6.0).abs() < 1e-5);
        assert!((output[2] - 12.0).abs() < 1e-5);
        assert!((output[3] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_elementwise_mul_zeros() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![0.0f32; 4];
        let n = a.len() as u32;

        let buf_a = GpuBuffer::from_host(&exec.context, &a).unwrap();
        let buf_b = GpuBuffer::from_host(&exec.context, &b).unwrap();
        let output_buf = exec.elementwise_mul_gpu(&buf_a, &buf_b, n).unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; a.len()];
        output_buf.copy_to_host(&mut output).unwrap();

        // All zeros
        for &v in &output {
            assert!(v.abs() < 1e-10, "expected 0, got {}", v);
        }
    }

    // === RoPE tests ===

    #[test]
    fn test_rope_into_position_zero() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        // Test with single head to match kernel behavior
        let num_heads = 1;
        let head_dim = 4;
        let n = num_heads * head_dim;
        let input = vec![1.0f32; n];
        let theta = 10000.0f32;

        let buf_input = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let buf_output = GpuBuffer::new(&exec.context, n).unwrap();

        exec.rope_into(
            &buf_input,
            &buf_output,
            0, // position 0
            num_heads as u32,
            head_dim as u32,
            theta,
        )
        .unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; n];
        buf_output.copy_to_host(&mut output).unwrap();

        // At position 0, cos(0)=1, sin(0)=0, so output should equal input
        // Only check the first head
        for (i, (&out, &inp)) in output.iter().zip(input.iter()).enumerate() {
            assert!(
                (out - inp).abs() < 1e-4,
                "RoPE pos=0 output[{}] = {}, expected {}",
                i,
                out,
                inp
            );
        }
    }

    #[test]
    fn test_rope_into_position_nonzero() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let num_heads = 1;
        let head_dim = 4;
        let n = num_heads * head_dim;
        let input = vec![1.0f32, 0.0, 0.0, 1.0];
        let theta = 10000.0f32;

        let buf_input = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let buf_output = GpuBuffer::new(&exec.context, n).unwrap();

        exec.rope_into(
            &buf_input,
            &buf_output,
            1, // position 1
            num_heads as u32,
            head_dim as u32,
            theta,
        )
        .unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; n];
        buf_output.copy_to_host(&mut output).unwrap();

        // At position 1, rotation should occur
        // Output should differ from input
        let diff: f32 = output
            .iter()
            .zip(input.iter())
            .map(|(o, i)| (o - i).abs())
            .sum();
        // With theta=10000, freq for dim 0 is ~1, so angle = 1 rad
        // cos(1)≈0.54, sin(1)≈0.84, so some rotation occurs
        assert!(diff > 0.1, "RoPE pos=1 should rotate, diff = {}", diff);
    }

    // === Fused SwiGLU tests ===

    #[test]
    fn test_fused_swiglu_gpu_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let n = 8;
        let gate = vec![1.0f32; n]; // SiLU(1) ≈ 0.731
        let up = vec![2.0f32; n];

        let buf_gate = GpuBuffer::from_host(&exec.context, &gate).unwrap();
        let buf_up = GpuBuffer::from_host(&exec.context, &up).unwrap();
        let output_buf = exec.fused_swiglu_gpu(&buf_gate, &buf_up, n as u32).unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; n];
        output_buf.copy_to_host(&mut output).unwrap();

        // SwiGLU = SiLU(gate) * up ≈ 0.731 * 2 = 1.462
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 1.462).abs() < 0.05,
                "SwiGLU output[{}] = {}, expected ~1.462",
                i,
                v
            );
        }
    }

    #[test]
    fn test_fused_swiglu_gpu_zero_gate() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let n = 4;
        let gate = vec![0.0f32; n]; // SiLU(0) = 0
        let up = vec![100.0f32; n];

        let buf_gate = GpuBuffer::from_host(&exec.context, &gate).unwrap();
        let buf_up = GpuBuffer::from_host(&exec.context, &up).unwrap();
        let output_buf = exec.fused_swiglu_gpu(&buf_gate, &buf_up, n as u32).unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; n];
        output_buf.copy_to_host(&mut output).unwrap();

        // SwiGLU with gate=0 should be ~0
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-5, "SwiGLU output[{}] = {}, expected ~0", i, v);
        }
    }

    // === Residual add tests ===

    #[test]
    fn test_add_residual_gpu_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let n = 4;
        // add_residual_gpu adds input to output in place: output += input
        let output_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input_data = vec![10.0f32, 20.0, 30.0, 40.0];

        let buf_output = GpuBuffer::from_host(&exec.context, &output_data).unwrap();
        let buf_input = GpuBuffer::from_host(&exec.context, &input_data).unwrap();

        exec.add_residual_gpu(&buf_output, &buf_input, n as u32)
            .unwrap();

        exec.stream.synchronize().unwrap();
        let mut result = vec![0.0f32; n];
        buf_output.copy_to_host(&mut result).unwrap();

        // Expected: [11, 22, 33, 44]
        assert!((result[0] - 11.0).abs() < 1e-5);
        assert!((result[1] - 22.0).abs() < 1e-5);
        assert!((result[2] - 33.0).abs() < 1e-5);
        assert!((result[3] - 44.0).abs() < 1e-5);
    }

    // === Host wrapper tests ===

    #[test]
    fn test_silu_host_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let input = vec![0.0f32, 1.0, -1.0, 2.0];
        let mut output = vec![0.0f32; 4];

        exec.silu_host(&input, &mut output).unwrap();

        // SiLU(0) = 0
        assert!(output[0].abs() < 1e-5);
        // SiLU(1) ≈ 0.731
        assert!((output[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_gelu_host_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let input = vec![0.0f32, 1.0, -1.0, 2.0];
        let mut output = vec![0.0f32; 4];

        exec.gelu_host(&input, &mut output).unwrap();

        // GELU(0) = 0
        assert!(output[0].abs() < 1e-5);
        // GELU(1) ≈ 0.841
        assert!((output[1] - 0.841).abs() < 0.02);
    }

    #[test]
    fn test_elementwise_mul_host_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];

        exec.elementwise_mul_host(&a, &b, &mut output).unwrap();

        assert!((output[0] - 2.0).abs() < 1e-5);
        assert!((output[1] - 6.0).abs() < 1e-5);
        assert!((output[2] - 12.0).abs() < 1e-5);
        assert!((output[3] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn test_fused_swiglu_host_basic() {
        let Some(mut exec) = create_executor() else {
            return;
        };

        let gate = vec![1.0f32; 4]; // SiLU(1) ≈ 0.731
        let up = vec![2.0f32; 4];
        let mut output = vec![0.0f32; 4];

        exec.fused_swiglu_host(&gate, &up, &mut output).unwrap();

        // SwiGLU = SiLU(gate) * up ≈ 0.731 * 2 = 1.462
        for &v in &output {
            assert!((v - 1.462).abs() < 0.05);
        }
    }

    // ========================================================================
    // Harness-Based Integration Tests
    // ========================================================================

    #[test]
    fn test_activations_with_harness_silu() {
        use crate::cuda::executor::test_fixtures::{setup_executor_harness, HarnessConfig};
        let Some(mut exec) = create_executor() else {
            return;
        };
        let config = HarnessConfig::default();
        if setup_executor_harness(&mut exec, &config).is_err() {
            return;
        }

        // Test SiLU with hidden_dim sized tensor
        let input = vec![0.5f32; config.hidden_dim];
        let input_buf = GpuBuffer::from_host(&exec.context, &input).unwrap();
        let output_buf = exec.silu_gpu(&input_buf, config.hidden_dim as u32).unwrap();

        exec.stream.synchronize().unwrap();
        let mut output = vec![0.0f32; config.hidden_dim];
        output_buf.copy_to_host(&mut output).unwrap();

        // SiLU(0.5) ≈ 0.311
        assert!(
            (output[0] - 0.311).abs() < 0.02,
            "SiLU(0.5) = {}",
            output[0]
        );
    }
