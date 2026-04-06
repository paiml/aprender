//! Long Row Softmax Test (WAPR-PERF-004)

/// Test: Long row softmax produces correct row sums (should be 1.0)
///
/// This tests the LongRowSoftmaxKernel with rows > 32 elements.
/// Critical for attention softmax where rows have seq_len (e.g., 1500) elements.
#[test]
#[cfg(feature = "cuda")]
fn test_long_row_softmax_correctness() {
    use trueno_gpu::driver::CudaContext;
    use trueno_gpu::memory::resident::GpuResidentTensor;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("CUDA not available, skipping test");
            return;
        }
    };

    // Test with row_size = 64 first (simpler case)
    let n_rows = 4;
    let row_size = 64;
    let total_size = n_rows * row_size;

    println!("Testing softmax with {} rows x {} elements...", n_rows, row_size);

    // Create simple input data
    let input_data: Vec<f32> = (0..total_size).map(|i| (i % row_size) as f32 * 0.1).collect();

    println!("Input first row: {:?}", &input_data[0..8]);

    let input = GpuResidentTensor::from_host(&ctx, &input_data).expect("input upload");
    println!("Input uploaded");

    // Run softmax
    let mut output = input.softmax(&ctx, n_rows as u32).expect("softmax");
    println!("Softmax completed");

    // Download result
    let result = output.to_host().expect("download");
    println!("Result downloaded, len={}", result.len());

    println!("Output first row (first 8): {:?}", &result[0..8]);
    println!("Output first row (last 4):  {:?}", &result[row_size - 4..row_size]);

    // FULL SOFTMAX TEST: Verify row sums to 1.0 and values match expected
    for row in 0..n_rows {
        let start = row * row_size;
        let end = start + row_size;
        let row_output = &result[start..end];
        let row_input = &input_data[start..end];

        // Compute expected softmax on CPU
        let row_max = row_input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_shifted: Vec<f32> = row_input.iter().map(|&x| (x - row_max).exp()).collect();
        let exp_sum: f32 = exp_shifted.iter().sum();
        let expected_softmax: Vec<f32> = exp_shifted.iter().map(|&e| e / exp_sum).collect();

        // Check row sums to 1.0
        let row_sum: f32 = row_output.iter().sum();
        let sum_diff = (row_sum - 1.0).abs();
        println!("Row {}: sum = {:.6} (diff from 1.0: {:.6})", row, row_sum, sum_diff);
        if sum_diff > 0.01 {
            panic!("Row {}: sum={:.6} does not equal 1.0 (diff={:.6})", row, row_sum, sum_diff);
        }

        // Check individual values
        for col in 0..row_size {
            let got = row_output[col];
            let expected = expected_softmax[col];
            let diff = (got - expected).abs();
            // 1% relative tolerance for individual values
            if diff > expected.max(1e-6) * 0.02 {
                panic!(
                    "Row {} col {}: expected {:.6}, got {:.6} (diff={:.6})",
                    row, col, expected, got, diff
                );
            }
        }
    }

    println!("✓ Full softmax test PASSED!");
    println!("  - {} rows x {} elements", n_rows, row_size);
    println!("  - All rows sum to 1.0");
    println!("  - All values match expected softmax within 2% tolerance");

    // ==== Test with 1500 elements (attention matrix row size) ====
    println!("\n=== Testing with 1500 elements (attention size) ===");
    let n_rows_large = 6; // 6 attention heads
    let row_size_large = 1500;
    let total_size_large = n_rows_large * row_size_large;

    // Create input with varying values
    let input_large: Vec<f32> = (0..total_size_large)
        .map(|i| ((i % row_size_large) as f32 - 750.0) * 0.01) // Range -7.5 to 7.49
        .collect();

    let input_gpu = GpuResidentTensor::from_host(&ctx, &input_large).expect("upload");
    let mut output_gpu = input_gpu.softmax(&ctx, n_rows_large as u32).expect("softmax");
    let result_large = output_gpu.to_host().expect("download");

    for row in 0..n_rows_large {
        let start = row * row_size_large;
        let end = start + row_size_large;
        let row_output = &result_large[start..end];
        let row_sum: f32 = row_output.iter().sum();
        let sum_diff = (row_sum - 1.0).abs();
        println!("Row {}: sum = {:.6} (diff from 1.0: {:.6})", row, row_sum, sum_diff);
        if sum_diff > 0.01 {
            panic!("Row {}: sum={:.6} does not equal 1.0", row, row_sum);
        }
    }

    println!("✓ Attention-sized softmax test PASSED!");
    println!("  - {} rows x {} elements", n_rows_large, row_size_large);
}
