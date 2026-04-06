
#[test]
#[serial]
fn test_cov006_residual_add_host_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 1024usize;

    let input1 = vec![1.0f32; n];
    let input2 = vec![2.0f32; n];
    let mut output = vec![0.0f32; n];

    let result = executor.residual_add_host(&input1, &input2, &mut output);
    assert!(
        result.is_ok(),
        "residual_add_host large failed: {:?}",
        result.err()
    );

    // Verify all outputs are 3.0
    assert!(
        output.iter().all(|&x| (x - 3.0).abs() < 1e-5),
        "residual_add outputs should all be 3.0"
    );
}

#[test]
#[serial]
fn test_cov006_fused_residual_rmsnorm_host_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 32usize;
    let epsilon = 1e-5f32;

    let residual: Vec<f32> = (0..hidden_size).map(|i| i as f32 / 10.0).collect();
    let input: Vec<f32> = (0..hidden_size)
        .map(|i| (hidden_size - i) as f32 / 10.0)
        .collect();
    let gamma = vec![1.0f32; hidden_size];
    let mut output = vec![0.0f32; hidden_size];

    let result =
        executor.fused_residual_rmsnorm_host(&residual, &input, &gamma, &mut output, epsilon);
    assert!(
        result.is_ok(),
        "fused_residual_rmsnorm_host failed: {:?}",
        result.err()
    );

    // Output should be normalized version of (residual + input)
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "Fused output should not be all zeros");
}

#[test]
#[serial]
fn test_cov006_fused_residual_rmsnorm_host_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 256usize;
    let epsilon = 1e-5f32;

    let residual = vec![0.5f32; hidden_size];
    let input = vec![0.3f32; hidden_size];
    let gamma = vec![1.0f32; hidden_size];
    let mut output = vec![0.0f32; hidden_size];

    let result =
        executor.fused_residual_rmsnorm_host(&residual, &input, &gamma, &mut output, epsilon);
    assert!(
        result.is_ok(),
        "fused_residual_rmsnorm_host large failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov006_residual_add_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;

    let input1_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input2_data: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let input1 = GpuBuffer::from_host(&executor.context, &input1_data).expect("input1 buffer");
    let input2 = GpuBuffer::from_host(&executor.context, &input2_data).expect("input2 buffer");

    let result = executor.residual_add_gpu(&input1, &input2, n);
    assert!(
        result.is_ok(),
        "residual_add_gpu failed: {:?}",
        result.err()
    );

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // Verify: output[i] = i + (n - i) = n
    for (idx, &val) in output.iter().enumerate() {
        let expected = n as f32;
        assert!(
            (val - expected).abs() < 1e-4,
            "residual_add_gpu mismatch at {}: {} vs {}",
            idx,
            val,
            expected
        );
    }
}

#[test]
#[serial]
fn test_cov006_residual_add_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 4096u32;

    let input1 =
        GpuBuffer::from_host(&executor.context, &vec![1.5f32; n as usize]).expect("input1");
    let input2 =
        GpuBuffer::from_host(&executor.context, &vec![2.5f32; n as usize]).expect("input2");

    let result = executor.residual_add_gpu(&input1, &input2, n);
    assert!(
        result.is_ok(),
        "residual_add_gpu large failed: {:?}",
        result.err()
    );

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // All should be 4.0
    assert!(
        output.iter().all(|&x| (x - 4.0).abs() < 1e-4),
        "All outputs should be 4.0"
    );
}

#[test]
#[serial]
fn test_cov006_rmsnorm_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let epsilon = 1e-5f32;

    let input_data: Vec<f32> = (0..hidden_size).map(|i| (i as f32 + 1.0) / 10.0).collect();
    let gamma_data = vec![1.0f32; hidden_size as usize];

    let input = GpuBuffer::from_host(&executor.context, &input_data).expect("input");
    let gamma = GpuBuffer::from_host(&executor.context, &gamma_data).expect("gamma");

    let result = executor.rmsnorm_gpu(&input, &gamma, hidden_size, epsilon);
    assert!(result.is_ok(), "rmsnorm_gpu failed: {:?}", result.err());

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; hidden_size as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // Output should be normalized
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "RMSNorm GPU output should not be all zeros");
}

#[test]
#[serial]
fn test_cov006_rmsnorm_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 512u32;
    let epsilon = 1e-6f32;

    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| ((i as f32) - 256.0) / 128.0)
        .collect();
    let gamma_data: Vec<f32> = (0..hidden_size)
        .map(|i| 0.5 + (i as f32) / 1024.0)
        .collect();

    let input = GpuBuffer::from_host(&executor.context, &input_data).expect("input");
    let gamma = GpuBuffer::from_host(&executor.context, &gamma_data).expect("gamma");

    let result = executor.rmsnorm_gpu(&input, &gamma, hidden_size, epsilon);
    assert!(
        result.is_ok(),
        "rmsnorm_gpu large failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov006_fused_residual_rmsnorm_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 64u32;
    let epsilon = 1e-5f32;

    let residual_data: Vec<f32> = (0..hidden_size).map(|i| i as f32 / 20.0).collect();
    let input_data: Vec<f32> = (0..hidden_size)
        .map(|i| (hidden_size - i) as f32 / 20.0)
        .collect();
    let gamma_data = vec![1.0f32; hidden_size as usize];

    let residual = GpuBuffer::from_host(&executor.context, &residual_data).expect("residual");
    let input = GpuBuffer::from_host(&executor.context, &input_data).expect("input");
    let gamma = GpuBuffer::from_host(&executor.context, &gamma_data).expect("gamma");

    let result =
        executor.fused_residual_rmsnorm_gpu(&residual, &input, &gamma, hidden_size, epsilon);
    assert!(
        result.is_ok(),
        "fused_residual_rmsnorm_gpu failed: {:?}",
        result.err()
    );

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; hidden_size as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // Output should be normalized version of (residual + input)
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "Fused GPU output should not be all zeros");
}

#[test]
#[serial]
fn test_cov006_fused_residual_rmsnorm_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 256u32;
    let epsilon = 1e-5f32;

    let residual = GpuBuffer::from_host(&executor.context, &vec![0.5f32; hidden_size as usize])
        .expect("residual");
    let input = GpuBuffer::from_host(&executor.context, &vec![0.3f32; hidden_size as usize])
        .expect("input");
    let gamma = GpuBuffer::from_host(&executor.context, &vec![1.0f32; hidden_size as usize])
        .expect("gamma");

    let result =
        executor.fused_residual_rmsnorm_gpu(&residual, &input, &gamma, hidden_size, epsilon);
    assert!(
        result.is_ok(),
        "fused_residual_rmsnorm_gpu large failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov006_gelu_gpu_edge_values() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Test with edge values: very negative, zero, very positive
    let data = vec![-10.0f32, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 100.0];
    let n = data.len() as u32;

    let buffer = GpuBuffer::from_host(&executor.context, &data).expect("GPU buffer");
    let result = executor.gelu_gpu(&buffer, n);
    assert!(
        result.is_ok(),
        "gelu_gpu edge values failed: {:?}",
        result.err()
    );

    executor.stream.synchronize().expect("sync");
    let mut output = vec![0.0f32; n as usize];
    buffer.copy_to_host(&mut output).expect("copy");

    // GELU(-10) should be very small (near 0)
    // GELU(10) should be close to 10
    assert!(output[0].abs() < 0.01, "GELU(-10) should be near 0");
    assert!(
        (output[7] - 100.0).abs() < 1.0,
        "GELU(100) should be close to 100"
    );
}

// ============================================================================
// COV-007: activations.rs coverage tests
// Target: Increase coverage from 24.03% to 50%+
// Focus: silu_gpu, gelu_async, elementwise_mul_gpu, silu_host, gelu_host,
//        elementwise_mul_host, fused_swiglu_host, add_residual_gpu
// ============================================================================

#[test]
#[serial]
fn test_cov007_silu_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 64.0).collect();

    let input = GpuBuffer::from_host(&executor.context, &data).expect("input buffer");
    let result = executor.silu_gpu(&input, n);
    assert!(result.is_ok(), "silu_gpu failed: {:?}", result.err());

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    let mid_idx = 128;
    assert!(
        output[mid_idx].abs() < 0.1,
        "SiLU(0) should be near 0, got {}",
        output[mid_idx]
    );
}

#[test]
#[serial]
fn test_cov007_silu_gpu_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 1024u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) / 256.0).collect();

    let input = GpuBuffer::from_host(&executor.context, &data).expect("input buffer");
    let result = executor.silu_gpu(&input, n);
    assert!(result.is_ok(), "silu_gpu large failed: {:?}", result.err());
}

#[test]
#[serial]
fn test_cov007_gelu_async_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) / 64.0).collect();

    let input = GpuBuffer::from_host(&executor.context, &data).expect("input buffer");
    let result = executor.gelu_async(&input, n);
    assert!(result.is_ok(), "gelu_async failed: {:?}", result.err());

    let output_buffer = result.unwrap();
    executor.stream.synchronize().expect("sync");

    let mut output = vec![0.0f32; n as usize];
    output_buffer.copy_to_host(&mut output).expect("copy");

    // GELU(0) should be near 0
    let mid_idx = 128;
    assert!(
        output[mid_idx].abs() < 0.1,
        "GELU(0) should be near 0, got {}",
        output[mid_idx]
    );
}

#[test]
#[serial]
fn test_cov007_gelu_async_large() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 2048u32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 - 1024.0) / 512.0).collect();

    let input = GpuBuffer::from_host(&executor.context, &data).expect("input buffer");
    let result = executor.gelu_async(&input, n);
    assert!(
        result.is_ok(),
        "gelu_async large failed: {:?}",
        result.err()
    );
}
