
/// Test Q5_0 GEMV parity: CPU dequantize+matmul vs GPU Q5_0 GEMV
/// This test catches the candle layout bug in Q5_0 that causes GQA models to fail.
#[test]
#[serial]
fn test_q5_0_gemv_parity() {
    if !CudaExecutor::is_available() {
        eprintln!("[SKIP] CUDA not available");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Small test: 4 blocks = 128 elements
    let num_blocks = 4usize;
    let k = num_blocks * 32; // 128 input elements
    let n = 1usize; // Single output row (GEMV)

    let weights_q5_0 = generate_q5_0_weights(num_blocks);

    // CPU path: dequantize then matmul
    let weights_f32 = dequantize_q5_0(&weights_q5_0).expect("dequantize Q5_0");
    assert_eq!(weights_f32.len(), k, "Dequantized length mismatch");

    // Input vector
    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();

    // CPU matmul: dot product for single row
    let cpu_output: f32 = weights_f32
        .iter()
        .zip(input.iter())
        .map(|(w, x)| w * x)
        .sum();

    // GPU path - upload weights as bytes, get raw device pointer
    let weights_buf =
        GpuBuffer::from_host(&executor.context, &weights_q5_0).expect("upload weights");
    let input_buf = GpuBuffer::from_host(&executor.context, &input).expect("upload input");
    let output_buf = GpuBuffer::<f32>::new(&executor.context, n).expect("output buffer");

    // Execute Q5_0 GEMV using _into variant with raw device pointer
    let weight_ptr = weights_buf.as_ptr();
    executor
        .q5_0_gemv_into(weight_ptr, &input_buf, &output_buf, n as u32, k as u32)
        .expect("Q5_0 GEMV");

    executor.stream.synchronize().expect("sync");

    let mut gpu_output = vec![0.0f32; n];
    output_buf.copy_to_host(&mut gpu_output).expect("download");

    // Compare
    let diff = (cpu_output - gpu_output[0]).abs();
    let rel_diff = diff / cpu_output.abs().max(1e-6);

    println!("=== Q5_0 GEMV Parity Test ===");
    println!("CPU output: {:.6}", cpu_output);
    println!("GPU output: {:.6}", gpu_output[0]);
    println!("Absolute diff: {:.6}", diff);
    println!("Relative diff: {:.4}%", rel_diff * 100.0);

    // Should be within 1% for quantized GEMV
    // NOTE: This test will FAIL until Q5_0 kernel is fixed for candle layout
    assert!(
        rel_diff < 0.01,
        "Q5_0 GEMV parity failed: CPU={:.6}, GPU={:.6}, diff={:.4}%\n\
         This indicates Q5_0 kernel uses wrong layout (trueno interleaved vs candle)",
        cpu_output,
        gpu_output[0],
        rel_diff * 100.0
    );

    println!("Q5_0 GEMV parity VERIFIED");
}

/// Test larger Q5_0 GEMV with dimensions matching Qwen K projection
#[test]
#[serial]
fn test_q5_0_gemv_qwen_k_dimensions() {
    if !CudaExecutor::is_available() {
        eprintln!("[SKIP] CUDA not available");
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Qwen 0.5B K projection: hidden_dim=896 → kv_dim=128 (2 heads * 64 head_dim)
    let hidden_dim = 896usize;
    let kv_dim = 128usize;

    // For Q5_0: hidden_dim elements per row, kv_dim rows
    let num_blocks_per_row = (hidden_dim + 31) / 32; // ceil(896/32) = 28
    let total_blocks = num_blocks_per_row * kv_dim;

    let weights_q5_0 = generate_q5_0_weights(total_blocks);

    // CPU path: dequantize then matmul each row
    let weights_f32 = dequantize_q5_0(&weights_q5_0).expect("dequantize Q5_0");

    // Input vector (simulated normed hidden state)
    let input: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32 * 0.01) - 4.0).sin())
        .collect();

    // CPU matmul: for each output row, compute dot product
    let mut cpu_output = vec![0.0f32; kv_dim];
    let row_len = num_blocks_per_row * 32;
    for row in 0..kv_dim {
        let row_start = row * row_len;
        let row_end = row_start + hidden_dim.min(row_len);
        let row_weights = &weights_f32[row_start..row_end];
        cpu_output[row] = row_weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum();
    }

    // GPU path - upload weights as bytes, get raw device pointer
    let weights_buf =
        GpuBuffer::from_host(&executor.context, &weights_q5_0).expect("upload weights");
    let input_buf = GpuBuffer::from_host(&executor.context, &input).expect("upload input");
    let output_buf = GpuBuffer::<f32>::new(&executor.context, kv_dim).expect("output buffer");

    // Execute Q5_0 GEMV using _into variant with raw device pointer
    let weight_ptr = weights_buf.as_ptr();
    executor
        .q5_0_gemv_into(
            weight_ptr,
            &input_buf,
            &output_buf,
            kv_dim as u32,
            hidden_dim as u32,
        )
        .expect("Q5_0 GEMV");

    executor.stream.synchronize().expect("sync");

    let mut gpu_output = vec![0.0f32; kv_dim];
    output_buf.copy_to_host(&mut gpu_output).expect("download");

    // Compare element-wise
    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    let cpu_sum: f32 = cpu_output.iter().sum();
    let gpu_sum: f32 = gpu_output.iter().sum();

    let sum_rel_diff = (cpu_sum - gpu_sum).abs() / cpu_sum.abs().max(1e-6);

    println!("=== Q5_0 GEMV Qwen K Dimensions Test ===");
    println!("Dimensions: {}x{}", hidden_dim, kv_dim);
    println!("CPU first 5: {:?}", &cpu_output[..5]);
    println!("GPU first 5: {:?}", &gpu_output[..5]);
    println!("CPU sum: {:.6}", cpu_sum);
    println!("GPU sum: {:.6}", gpu_sum);
    println!("Max element diff: {:.6}", max_diff);
    println!("Sum relative diff: {:.4}%", sum_rel_diff * 100.0);

    assert!(
        sum_rel_diff < 0.05,
        "Q5_0 GEMV Qwen K dimensions failed: sum diff {:.4}%",
        sum_rel_diff * 100.0
    );

    println!("Q5_0 GEMV Qwen K dimensions VERIFIED");
}
