
#[test]
#[serial]
fn test_cov007_elementwise_mul_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 128u32;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|_| 2.0f32).collect();

    let a = GpuBuffer::from_host(&executor.context, &a_data).expect("a buffer");
    let b = GpuBuffer::from_host(&executor.context, &b_data).expect("b buffer");

    let result = executor.elementwise_mul_gpu(&a, &b, n);
    assert!(
        result.is_ok(),
        "elementwise_mul_gpu failed: {:?}",
        result.err()
    );

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // output[i] = a[i] * b[i] = i * 2 = 2i
    for (idx, &val) in output.iter().enumerate() {
        let expected = (idx as f32) * 2.0;
        assert!(
            (val - expected).abs() < 1e-4,
            "mul mismatch at {}: {} vs {}",
            idx,
            val,
            expected
        );
    }
}

#[test]
#[serial]
fn test_cov007_elementwise_mul_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 4096u32;
    let a = GpuBuffer::from_host(&executor.context, &vec![3.0f32; n as usize]).expect("a");
    let b = GpuBuffer::from_host(&executor.context, &vec![4.0f32; n as usize]).expect("b");

    let result = executor.elementwise_mul_gpu(&a, &b, n);
    assert!(
        result.is_ok(),
        "elementwise_mul_gpu large failed: {:?}",
        result.err()
    );

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // All should be 12.0
    assert!(
        output.iter().all(|&x| (x - 12.0).abs() < 1e-4),
        "All outputs should be 12.0"
    );
}

#[test]
#[serial]
fn test_cov007_silu_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) / 16.0).collect();
    let mut output = vec![0.0f32; n];

    let result = executor.silu_host(&input, &mut output);
    assert!(result.is_ok(), "silu_host failed: {:?}", result.err());

    // SiLU(0) should be near 0
    let mid_idx = 32;
    assert!(output[mid_idx].abs() < 0.1, "SiLU(0) should be near 0");
}

#[test]
#[serial]
fn test_cov007_silu_host_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 512usize;
    let input = vec![1.0f32; n];
    let mut output = vec![0.0f32; n];

    let result = executor.silu_host(&input, &mut output);
    assert!(result.is_ok(), "silu_host large failed: {:?}", result.err());

    // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
    assert!(
        output[0] > 0.7 && output[0] < 0.8,
        "SiLU(1) should be ~0.731, got {}",
        output[0]
    );
}

#[test]
#[serial]
fn test_cov007_gelu_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) / 16.0).collect();
    let mut output = vec![0.0f32; n];

    let result = executor.gelu_host(&input, &mut output);
    assert!(result.is_ok(), "gelu_host failed: {:?}", result.err());

    // GELU(0) should be near 0
    let mid_idx = 32;
    assert!(output[mid_idx].abs() < 0.1, "GELU(0) should be near 0");
}

#[test]
#[serial]
fn test_cov007_gelu_host_positive() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let input = vec![2.0f32; n]; // All 2.0
    let mut output = vec![0.0f32; n];

    let result = executor.gelu_host(&input, &mut output);
    assert!(
        result.is_ok(),
        "gelu_host positive failed: {:?}",
        result.err()
    );

    // GELU(2) should be close to 2 (slightly less)
    assert!(
        output[0] > 1.9 && output[0] < 2.1,
        "GELU(2) should be ~2.0, got {}",
        output[0]
    );
}

#[test]
#[serial]
fn test_cov007_elementwise_mul_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let mut output = vec![0.0f32; n];

    let result = executor.elementwise_mul_host(&a, &b, &mut output);
    assert!(
        result.is_ok(),
        "elementwise_mul_host failed: {:?}",
        result.err()
    );

    // output[i] = i * (n - i)
    for (idx, &val) in output.iter().enumerate() {
        let expected = (idx as f32) * ((n - idx) as f32);
        assert!(
            (val - expected).abs() < 1e-4,
            "mul_host mismatch at {}: {} vs {}",
            idx,
            val,
            expected
        );
    }
}

#[test]
#[serial]
fn test_cov007_fused_swiglu_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 64usize;
    let gate = vec![1.0f32; n];
    let up = vec![2.0f32; n];
    let mut output = vec![0.0f32; n];

    let result = executor.fused_swiglu_host(&gate, &up, &mut output);
    assert!(
        result.is_ok(),
        "fused_swiglu_host failed: {:?}",
        result.err()
    );

    // SwiGLU(gate, up) = silu(gate) * up = silu(1) * 2 ≈ 0.731 * 2 ≈ 1.462
    assert!(
        output[0] > 1.4 && output[0] < 1.6,
        "SwiGLU(1,2) should be ~1.46, got {}",
        output[0]
    );
}

#[test]
#[serial]
fn test_cov007_fused_swiglu_host_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256usize;
    let gate: Vec<f32> = (0..n).map(|i| (i as f32) / 128.0).collect();
    let up = vec![1.0f32; n];
    let mut output = vec![0.0f32; n];

    let result = executor.fused_swiglu_host(&gate, &up, &mut output);
    assert!(
        result.is_ok(),
        "fused_swiglu_host large failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov007_add_residual_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 128u32;

    // Output starts with values, input is what to add
    let output_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input_data: Vec<f32> = (0..n).map(|_| 10.0f32).collect();

    let output_buf = GpuBuffer::from_host(&executor.context, &output_data).expect("output buffer");
    let input_buf = GpuBuffer::from_host(&executor.context, &input_data).expect("input buffer");

    let result = executor.add_residual_gpu(&output_buf, &input_buf, n);
    assert!(
        result.is_ok(),
        "add_residual_gpu failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buf.copy_to_host(&mut output).expect("copy");

    // output[i] += 10, so output[i] = i + 10
    for (idx, &val) in output.iter().enumerate() {
        let expected = idx as f32 + 10.0;
        assert!(
            (val - expected).abs() < 1e-4,
            "add_residual mismatch at {}: {} vs {}",
            idx,
            val,
            expected
        );
    }
}

#[test]
#[serial]
fn test_cov007_add_residual_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 4096u32;

    let output_buf =
        GpuBuffer::from_host(&executor.context, &vec![5.0f32; n as usize]).expect("output");
    let input_buf =
        GpuBuffer::from_host(&executor.context, &vec![3.0f32; n as usize]).expect("input");

    let result = executor.add_residual_gpu(&output_buf, &input_buf, n);
    assert!(
        result.is_ok(),
        "add_residual_gpu large failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buf.copy_to_host(&mut output).expect("copy");

    // All should be 8.0 (5 + 3)
    assert!(
        output.iter().all(|&x| (x - 8.0).abs() < 1e-4),
        "All outputs should be 8.0"
    );
}

#[test]
#[serial]
fn test_cov007_fused_swiglu_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let gate_data = vec![1.0f32; n as usize];
    let up_data = vec![2.0f32; n as usize];

    let gate = GpuBuffer::from_host(&executor.context, &gate_data).expect("gate buffer");
    let up = GpuBuffer::from_host(&executor.context, &up_data).expect("up buffer");

    let result = executor.fused_swiglu_gpu(&gate, &up, n);
    assert!(
        result.is_ok(),
        "fused_swiglu_gpu failed: {:?}",
        result.err()
    );

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // SwiGLU(1,2) = silu(1) * 2 ≈ 1.46
    assert!(
        output[0] > 1.4 && output[0] < 1.6,
        "SwiGLU(1,2) should be ~1.46, got {}",
        output[0]
    );
}

#[test]
#[serial]
fn test_cov007_fused_swiglu_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 2048u32;

    let gate = GpuBuffer::from_host(&executor.context, &vec![0.5f32; n as usize]).expect("gate");
    let up = GpuBuffer::from_host(&executor.context, &vec![1.0f32; n as usize]).expect("up");

    let result = executor.fused_swiglu_gpu(&gate, &up, n);
    assert!(
        result.is_ok(),
        "fused_swiglu_gpu large failed: {:?}",
        result.err()
    );
}

