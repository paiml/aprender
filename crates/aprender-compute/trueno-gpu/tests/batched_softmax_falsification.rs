//! Falsification Tests for BatchedSoftmaxKernel (WAPR-PERF-008)
//!
//! Karl Popper's Mandate: Isolate and verify the softmax before re-enabling batched attention.
//!
//! ## The "Catley" Bug
//!
//! BatchedSoftmaxKernel produces incorrect output, causing transcription to hallucinate
//! "[Catley]" instead of "The birds can use".
//!
//! ## Hypothesis to Falsify
//!
//! H0: BatchedSoftmaxKernel correctly computes softmax for 1500-element rows
//!
//! ## Test Strategy
//!
//! 1. Create known input: row of 1500 values
//! 2. Compute CPU reference softmax
//! 3. Run GPU BatchedSoftmaxKernel
//! 4. Compare: sum must equal 1.0, individual values must match within epsilon

#![allow(unused_imports)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{BatchedSoftmaxKernel, Kernel};

/// CPU reference softmax implementation
fn cpu_softmax(input: &[f32]) -> Vec<f32> {
    // Find max for numerical stability
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let exp_vals: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    // Normalize
    exp_vals.iter().map(|&x| x / sum).collect()
}

/// Test 1: Verify CPU softmax reference on small input
#[test]
fn test_cpu_softmax_sanity() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = cpu_softmax(&input);

    // Sum must be 1.0
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax sum should be 1.0, got {}", sum);

    // Values should be in (0, 1)
    for (i, &v) in output.iter().enumerate() {
        assert!(v > 0.0 && v < 1.0, "Softmax[{}] = {} should be in (0, 1)", i, v);
    }

    // Larger input should have larger softmax
    assert!(output[3] > output[2] && output[2] > output[1] && output[1] > output[0]);
}

/// Test 2: GPU BatchedSoftmaxKernel on short row (32 elements, single warp)
#[test]
#[cfg(feature = "cuda")]
fn test_batched_softmax_short_row() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip if no CUDA
    };

    let row_size = 32u32;
    let total_rows = 1u32;

    // Create input: [1, 2, 3, ..., 32]
    let input: Vec<f32> = (1..=row_size).map(|x| x as f32).collect();
    let expected = cpu_softmax(&input);

    // Upload to GPU
    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, row_size as usize).expect("Alloc failed");

    // Compile and run kernel
    let kernel = BatchedSoftmaxKernel::new(total_rows, row_size);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let config = LaunchConfig { grid: (total_rows, 1, 1), block: (32, 1, 1), shared_mem: 72 };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(total_rows) as *mut _,
        std::ptr::addr_of!(row_size) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    // Download result
    let mut output = vec![0.0f32; row_size as usize];
    output_buf.copy_to_host(&mut output).expect("Download failed");

    // Verify sum = 1.0
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Short row: softmax sum should be 1.0, got {} (delta={})",
        sum,
        (sum - 1.0).abs()
    );

    // Verify individual values match CPU reference
    for (i, (&gpu, &cpu)) in output.iter().zip(expected.iter()).enumerate() {
        let delta: f32 = (gpu - cpu).abs();
        assert!(delta < 1e-5, "Short row [{}]: GPU={} vs CPU={}, delta={}", i, gpu, cpu, delta);
    }

    eprintln!("✓ Short row (32 elements) softmax PASSED");
}

/// Test 3: GPU BatchedSoftmaxKernel on LONG row (1500 elements - the bug case!)
#[test]
#[cfg(feature = "cuda")]
fn test_batched_softmax_long_row_1500() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip if no CUDA
    };

    let row_size = 1500u32;
    let total_rows = 1u32;

    // Create input: use small values to avoid overflow
    // Values from -5 to +5 spread across 1500 elements
    let input: Vec<f32> =
        (0..row_size).map(|i| -5.0 + 10.0 * (i as f32 / (row_size - 1) as f32)).collect();
    let expected = cpu_softmax(&input);

    eprintln!("Input: first 5 = {:?}, last 5 = {:?}", &input[..5], &input[row_size as usize - 5..]);
    eprintln!(
        "CPU expected: first 5 = {:?}, last 5 = {:?}",
        &expected[..5],
        &expected[row_size as usize - 5..]
    );
    eprintln!("CPU expected sum = {}", expected.iter().sum::<f32>());

    // Upload to GPU
    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, row_size as usize).expect("Alloc failed");

    // Compile and run kernel
    let kernel = BatchedSoftmaxKernel::new(total_rows, row_size);
    let ptx = kernel.emit_ptx();

    // Debug: print PTX structure
    eprintln!("PTX has {} lines", ptx.lines().count());
    eprintln!("PTX contains max_loop: {}", ptx.contains("max_loop:"));
    eprintln!("PTX contains sum_loop: {}", ptx.contains("sum_loop:"));
    eprintln!("PTX contains write_loop: {}", ptx.contains("write_loop:"));

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let config = LaunchConfig { grid: (total_rows, 1, 1), block: (32, 1, 1), shared_mem: 72 };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(total_rows) as *mut _,
        std::ptr::addr_of!(row_size) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    // Download result
    let mut output = vec![0.0f32; row_size as usize];
    output_buf.copy_to_host(&mut output).expect("Download failed");

    eprintln!(
        "GPU output: first 5 = {:?}, last 5 = {:?}",
        &output[..5],
        &output[row_size as usize - 5..]
    );

    // Verify sum = 1.0 (THE CRITICAL CHECK)
    let sum: f32 = output.iter().sum();
    eprintln!("GPU sum = {}", sum);

    assert!(
        (sum - 1.0).abs() < 1e-4,
        "LONG ROW BUG: softmax sum should be 1.0, got {} (delta={})",
        sum,
        (sum - 1.0).abs()
    );

    // Verify max and min are reasonable
    let gpu_max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gpu_min = output.iter().cloned().filter(|&x| x > 0.0).fold(f32::INFINITY, f32::min);
    let cpu_max = expected.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cpu_min = expected.iter().cloned().filter(|&x| x > 0.0).fold(f32::INFINITY, f32::min);

    eprintln!("GPU max={}, min={}", gpu_max, gpu_min);
    eprintln!("CPU max={}, min={}", cpu_max, cpu_min);

    // Verify the distribution shape roughly matches
    let gpu_argmax =
        output.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    let cpu_argmax =
        expected.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

    assert_eq!(gpu_argmax, cpu_argmax, "Argmax mismatch: GPU={} vs CPU={}", gpu_argmax, cpu_argmax);

    // Sample comparison at specific indices
    let test_indices = [0, 32, 100, 500, 1000, 1499];
    for &i in &test_indices {
        let delta = (output[i] - expected[i]).abs();
        let rel_delta = if expected[i].abs() > 1e-10 { delta / expected[i].abs() } else { delta };
        assert!(
            rel_delta < 0.1 || delta < 1e-6,
            "Long row [{}]: GPU={} vs CPU={}, delta={}, rel_delta={}",
            i,
            output[i],
            expected[i],
            delta,
            rel_delta
        );
    }

    eprintln!("✓ Long row (1500 elements) softmax PASSED");
}

/// Test 4: Multiple rows (the actual batch case)
#[test]
#[cfg(feature = "cuda")]
fn test_batched_softmax_6_rows_of_1500() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip if no CUDA
    };

    let row_size = 1500u32;
    let total_rows = 6u32; // Simulating 6 attention heads

    // Create input: different values for each row
    let mut input: Vec<f32> = Vec::with_capacity((total_rows * row_size) as usize);
    for row in 0..total_rows {
        for i in 0..row_size {
            // Each row has slightly different distribution
            let base = -5.0 + 10.0 * (i as f32 / (row_size - 1) as f32);
            input.push(base + 0.1 * row as f32);
        }
    }

    // Compute CPU expected for each row
    let mut expected: Vec<f32> = Vec::with_capacity((total_rows * row_size) as usize);
    for row in 0..total_rows {
        let start = (row * row_size) as usize;
        let end = start + row_size as usize;
        let row_softmax = cpu_softmax(&input[start..end]);
        expected.extend(row_softmax);
    }

    // Upload to GPU
    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, (total_rows * row_size) as usize).expect("Alloc failed");

    // Compile and run kernel
    let kernel = BatchedSoftmaxKernel::new(total_rows, row_size);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let config = LaunchConfig { grid: (total_rows, 1, 1), block: (32, 1, 1), shared_mem: 72 };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(total_rows) as *mut _,
        std::ptr::addr_of!(row_size) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    // Download result
    let mut output = vec![0.0f32; (total_rows * row_size) as usize];
    output_buf.copy_to_host(&mut output).expect("Download failed");

    // Verify each row sums to 1.0
    for row in 0..total_rows {
        let start = (row * row_size) as usize;
        let end = start + row_size as usize;
        let row_sum: f32 = output[start..end].iter().sum();

        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "Row {}: softmax sum should be 1.0, got {} (delta={})",
            row,
            row_sum,
            (row_sum - 1.0).abs()
        );
    }

    eprintln!("✓ Batched softmax (6 rows × 1500 elements) PASSED");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_softmax_short_row() {
    // Skip without CUDA
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_softmax_long_row_1500() {
    // Skip without CUDA
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_softmax_6_rows_of_1500() {
    // Skip without CUDA
}
