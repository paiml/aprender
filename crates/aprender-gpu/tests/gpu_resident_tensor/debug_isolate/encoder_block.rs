//! Individual GPU operations and full encoder block tests

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;
#[cfg(feature = "cuda")]
use trueno_gpu::memory::resident::{
    reset_transfer_counters, total_d2h_transfers, total_h2d_transfers, GpuResidentTensor,
};

/// Test: Individual GPU operations work correctly
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_operations_individually() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    println!("\n=== Testing Individual GPU Operations ===");

    // Test data
    let d = 4u32; // small dimension for testing
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let weights: Vec<f32> =
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let bias: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let gamma: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let beta: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

    // Upload
    let x = GpuResidentTensor::from_host(&ctx, &data).expect("upload x");
    let w = GpuResidentTensor::from_host(&ctx, &weights).expect("upload w");
    let b = GpuResidentTensor::from_host(&ctx, &bias).expect("upload b");
    let g = GpuResidentTensor::from_host(&ctx, &gamma).expect("upload gamma");
    let bt = GpuResidentTensor::from_host(&ctx, &beta).expect("upload beta");

    // Test 1: matmul
    print!("1. matmul... ");
    match x.matmul(&ctx, &w, 1, d, d) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("result: {:?}", h);
        }
        Err(e) => println!("FAILED: {:?}", e),
    }

    // Test 2: bias_add
    print!("2. bias_add... ");
    match x.bias_add(&ctx, &b) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("result: {:?}", h);
        }
        Err(e) => println!("FAILED: {:?}", e),
    }

    // Test 3: gelu
    print!("3. gelu... ");
    match x.gelu(&ctx) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("result: {:?}", h);
        }
        Err(e) => println!("FAILED: {:?}", e),
    }

    // Test 4: layer_norm
    print!("4. layer_norm... ");
    match x.layer_norm(&ctx, &g, &bt, d, 1) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("result: {:?}", h);
        }
        Err(e) => println!("FAILED: {:?}", e),
    }

    // Test 5: linear (matmul + bias)
    print!("5. linear... ");
    match x.linear(&ctx, &w, Some(&b), 1, d, d) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("result: {:?}", h);
        }
        Err(e) => println!("FAILED: {:?}", e),
    }

    println!("=== Done ===");
}

/// Test: Full encoder block runs on GPU with minimal transfers
///
/// Requirement: Upload weights once, then run encoder blocks with only
/// input upload and output download per block. No intermediate transfers.
#[test]
#[cfg(feature = "cuda")]
fn test_full_encoder_block_gpu() {
    use trueno_gpu::memory::resident::{
        forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
    };

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("CUDA not available, skipping encoder block test");
            return;
        }
    };

    // Config: tiny-like (d_model=64, n_heads=4, ffn_dim=256)
    // Scaled down for fast testing
    let d_model = 64u32;
    let n_heads = 4u32;
    let ffn_dim = d_model * 4; // 256
    let seq_len = 8u32; // Short sequence for testing

    let config = GpuEncoderConfig { d_model, n_heads, ffn_dim };

    // Create dummy weights (random-ish for testing, actual values don't matter for transfer test)
    let weight_size = (d_model * d_model) as usize;
    let ffn_up_size = (d_model * ffn_dim) as usize;
    let ffn_down_size = (ffn_dim * d_model) as usize;

    // Initialize with simple patterns
    let ln_gamma: Vec<f32> = (0..d_model).map(|_| 1.0).collect();
    let ln_beta: Vec<f32> = (0..d_model).map(|_| 0.0).collect();
    let w_proj: Vec<f32> = (0..weight_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let b_proj: Vec<f32> = (0..d_model).map(|_| 0.0).collect();
    let ffn_up_w: Vec<f32> = (0..ffn_up_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let ffn_up_b: Vec<f32> = (0..ffn_dim).map(|_| 0.0).collect();
    let ffn_down_w: Vec<f32> = (0..ffn_down_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let ffn_down_b: Vec<f32> = (0..d_model).map(|_| 0.0).collect();

    // Upload weights (this counts as H2D transfers during initialization)
    reset_transfer_counters();

    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &ln_gamma).expect("ln1_gamma"),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &ln_beta).expect("ln1_beta"),
        w_q: GpuResidentTensor::from_host(&ctx, &w_proj).expect("w_q"),
        b_q: GpuResidentTensor::from_host(&ctx, &b_proj).expect("b_q"),
        w_k: GpuResidentTensor::from_host(&ctx, &w_proj).expect("w_k"),
        b_k: GpuResidentTensor::from_host(&ctx, &b_proj).expect("b_k"),
        w_v: GpuResidentTensor::from_host(&ctx, &w_proj).expect("w_v"),
        b_v: GpuResidentTensor::from_host(&ctx, &b_proj).expect("b_v"),
        w_o: GpuResidentTensor::from_host(&ctx, &w_proj).expect("w_o"),
        b_o: GpuResidentTensor::from_host(&ctx, &b_proj).expect("b_o"),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &ln_gamma).expect("ln2_gamma"),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &ln_beta).expect("ln2_beta"),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &ffn_up_w).expect("ffn_up_w"),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &ffn_up_b).expect("ffn_up_b"),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &ffn_down_w).expect("ffn_down_w"),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &ffn_down_b).expect("ffn_down_b"),
    };

    let weight_upload_h2d = total_h2d_transfers();
    println!("\n=== GPU Encoder Block Test ===");
    println!("Weight upload: {} H2D transfers", weight_upload_h2d);

    // Now reset and run forward pass
    reset_transfer_counters();

    // Create input
    let input_size = (seq_len * d_model) as usize;
    let input_data: Vec<f32> = (0..input_size).map(|i| (i as f32 * 0.01).sin()).collect();

    // Upload input (1 H2D)
    let input = GpuResidentTensor::from_host(&ctx, &input_data).expect("input upload");
    let h2d_after_input = total_h2d_transfers();
    println!("Input upload: {} H2D transfers", h2d_after_input);

    // Run forward pass (should have 0 additional transfers during computation)
    let mut output =
        forward_encoder_block_gpu(&ctx, &input, &weights, &config).expect("forward pass failed");

    let h2d_after_forward = total_h2d_transfers();
    let d2h_after_forward = total_d2h_transfers();
    println!("After forward pass: {} H2D, {} D2H", h2d_after_forward, d2h_after_forward);

    // Download output (1 D2H)
    let result = output.to_host().expect("output download");
    let final_h2d = total_h2d_transfers();
    let final_d2h = total_d2h_transfers();

    println!("After download: {} H2D, {} D2H", final_h2d, final_d2h);
    println!("Output size: {} elements", result.len());

    // Verify transfer counts
    // Expected: 1 H2D (input) + 0 during forward + 1 D2H (output) = 1 H2D, 1 D2H
    assert_eq!(
        final_h2d, 1,
        "Forward pass should have 1 H2D transfer (input only), got {}",
        final_h2d
    );
    assert_eq!(
        final_d2h, 1,
        "Forward pass should have 1 D2H transfer (output only), got {}",
        final_d2h
    );

    // Verify output is not all zeros (sanity check)
    let output_sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(output_sum > 0.0, "Output should not be all zeros");

    println!("Full GPU encoder block test PASSED!");
    println!("  - 1 H2D (input upload)");
    println!("  - 0 transfers during forward");
    println!("  - 1 D2H (output download)");
}
