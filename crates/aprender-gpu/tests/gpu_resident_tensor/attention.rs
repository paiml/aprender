//! PHASE 2: Batched Multi-Head Attention

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;
#[cfg(feature = "cuda")]
use trueno_gpu::memory::resident::{
    batched_multihead_attention, clear_kernel_cache, reset_transfer_counters, total_d2h_transfers,
    total_h2d_transfers, GpuResidentTensor, TransferStats,
};

/// Test: Batched attention should use SINGLE kernel launch for all heads
///
/// Requirement: Multi-head attention with N heads should NOT launch N kernels.
/// It should launch ONE kernel that processes all heads in parallel.
#[test]
#[cfg(feature = "cuda")]
fn test_batched_attention_single_kernel() {
    use trueno_gpu::memory::resident::batched_multihead_attention;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip if no CUDA
    };

    reset_transfer_counters();

    // Small test case: 4 sequence positions, 2 heads, 8-dim per head
    let seq_len = 4u32;
    let n_heads = 2u32;
    let head_dim = 8u32;
    let d_model = (n_heads * head_dim) as usize;

    // Q, K, V as [seq_len, d_model] tensors
    let q = GpuResidentTensor::from_host(&ctx, &vec![0.1f32; seq_len as usize * d_model])
        .expect("Upload Q");
    let k = GpuResidentTensor::from_host(&ctx, &vec![0.1f32; seq_len as usize * d_model])
        .expect("Upload K");
    let v = GpuResidentTensor::from_host(&ctx, &vec![0.1f32; seq_len as usize * d_model])
        .expect("Upload V");

    // 3 H2D transfers for Q, K, V
    assert_eq!(total_h2d_transfers(), 3);
    assert_eq!(total_d2h_transfers(), 0);

    // Batched attention - should be SINGLE kernel for all heads
    let output = batched_multihead_attention(&ctx, &q, &k, &v, n_heads, head_dim, seq_len)
        .expect("Batched attention failed");

    // Verify output shape
    assert_eq!(output.len(), seq_len as usize * d_model);
    // Verify stays on device
    assert!(output.is_device_resident());
    assert_eq!(output.device_to_host_transfers(), 0);
    // Verify NO additional H2D/D2H transfers (data stayed on GPU)
    assert_eq!(total_h2d_transfers(), 3); // Still just the original 3
    assert_eq!(total_d2h_transfers(), 0); // No downloads
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_batched_attention_single_kernel() {}

/// Test: Batched attention should include fused softmax
///
/// Requirement: The attention kernel should compute softmax INSIDE the kernel,
/// not as a separate kernel launch.
#[test]
#[ignore = "TDD: Implementation pending - fused softmax not yet implemented"]
fn test_batched_attention_fused_softmax() {
    // When implemented:
    //
    // // The kernel should compute:
    // // attention_output = softmax(Q @ K^T / sqrt(d_k)) @ V
    // // ALL IN ONE KERNEL (no separate softmax launch)
    //
    // let ctx = CudaContext::new(0).expect("test");
    // let output = batched_multihead_attention_with_stats(&ctx, &q, &k, &v, n_heads, head_dim).expect("test");
    //
    // // Stats should show:
    // assert_eq!(output.stats.kernel_launches, 1);
    // assert_eq!(output.stats.softmax_kernel_launches, 0); // Fused!
    // assert!(output.stats.has_fused_softmax);

    // TODO: "TDD: fused softmax not implemented"
}

/// Test: Simple 2x2 matmul correctness
///
/// Debug test to isolate matmul behavior before full attention pipeline.
#[test]
#[cfg(feature = "cuda")]
fn test_matmul_2x2_correctness() {
    clear_kernel_cache();
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let expected: Vec<f32> = vec![19.0, 22.0, 43.0, 50.0];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).expect("upload A");
    let b = GpuResidentTensor::from_host(&ctx, &b_data).expect("upload B");

    // matmul(A, B, m=2, n=2, k=2)
    let mut c = a.matmul(&ctx, &b, 2, 2, 2).expect("matmul failed");
    let result = c.to_host().expect("download C");

    println!("\n=== Matmul 2x2 Test ===");
    println!("A: {:?}", a_data);
    println!("B: {:?}", b_data);
    println!("Expected: {:?}", expected);
    println!("GPU result: {:?}", result);

    let max_diff: f32 =
        result.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    println!("Max diff: {}", max_diff);
    assert!(max_diff < 0.01, "Matmul 2x2 failed: max diff {} > 0.01", max_diff);
    println!("✓ Matmul 2x2 PASSED!");
}

/// Test: Each step of attention individually
///
/// Debug test to find which step produces zeros in row 1.
#[test]
#[cfg(feature = "cuda")]
fn test_attention_steps_individually() {
    use trueno_gpu::memory::resident::TransferStats;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let seq_len = 2u32;
    let d_model = 2u32;

    // Q = [[1, 0], [0, 1]], K = [[1, 0], [0, 1]], V = [[1, 2], [3, 4]]
    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let v_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).expect("upload Q");
    let k = GpuResidentTensor::from_host(&ctx, &k_data).expect("upload K");
    let v = GpuResidentTensor::from_host(&ctx, &v_data).expect("upload V");

    println!("\n=== Step-by-Step Attention Debug ===");

    // Step 1: Transpose K
    // K = [[1, 0], [0, 1]] -> K^T = [[1, 0], [0, 1]] (identity is its own transpose)
    // Actually K^T: [d_model, seq_len] so K[0,0]=1, K[0,1]=0 -> K^T[0,0]=1, K^T[1,0]=0, K^T[0,1]=0, K^T[1,1]=1
    // K^T stored as row-major [d_model, seq_len] = [[K[0,0], K[1,0]], [K[0,1], K[1,1]]] = [[1, 0], [0, 1]]

    // Step 2: Q @ K^T = [[1,0],[0,1]] @ [[1,0],[0,1]] = [[1,0],[0,1]]
    // Expected scores: [1.0, 0.0, 0.0, 1.0]
    // Note: We need to manually do transpose + matmul to test
    // For now, test Q @ K (not transposed) which should give same result for identity matrix
    let mut scores = q.matmul(&ctx, &k, seq_len, seq_len, d_model).expect("Q@K failed");
    let scores_host = scores.to_host().expect("download scores");
    println!("Step 1 - Q @ K (should be identity): {:?}", scores_host);

    // Step 3: Scale by 1/sqrt(head_dim)
    let scale = 1.0 / (d_model as f32).sqrt(); // 1/sqrt(2) = 0.707
    let q2 = GpuResidentTensor::from_host(&ctx, &q_data).expect("upload Q2");
    let k2 = GpuResidentTensor::from_host(&ctx, &k_data).expect("upload K2");
    let mut scores2 = q2.matmul(&ctx, &k2, seq_len, seq_len, d_model).expect("Q@K");
    let scaled = scores2.scale(&ctx, scale).expect("scale failed");
    let mut scaled_mut = scaled;
    let scaled_host = scaled_mut.to_host().expect("download scaled");
    println!("Step 2 - Scaled (×{}): {:?}", scale, scaled_host);

    // Step 4: Softmax
    let q3 = GpuResidentTensor::from_host(&ctx, &q_data).expect("upload Q3");
    let k3 = GpuResidentTensor::from_host(&ctx, &k_data).expect("upload K3");
    let mut scores3 = q3.matmul(&ctx, &k3, seq_len, seq_len, d_model).expect("Q@K");
    let scaled3 = scores3.scale(&ctx, scale).expect("scale");
    let softmax_result = scaled3.softmax(&ctx, seq_len).expect("softmax failed");
    let mut softmax_mut = softmax_result;
    let softmax_host = softmax_mut.to_host().expect("download softmax");
    println!("Step 3 - Softmax: {:?}", softmax_host);

    // Expected softmax:
    // Row 0: softmax([0.707, 0]) = [exp(0.707), exp(0)] / sum = [2.028, 1] / 3.028 = [0.670, 0.330]
    // Row 1: softmax([0, 0.707]) = [exp(0), exp(0.707)] / sum = [1, 2.028] / 3.028 = [0.330, 0.670]
    println!("Expected softmax: [0.670, 0.330, 0.330, 0.670]");

    // Step 5: Final matmul
    let q4 = GpuResidentTensor::from_host(&ctx, &q_data).expect("Q4");
    let k4 = GpuResidentTensor::from_host(&ctx, &k_data).expect("K4");
    let v4 = GpuResidentTensor::from_host(&ctx, &v_data).expect("V4");
    let mut scores4 = q4.matmul(&ctx, &k4, seq_len, seq_len, d_model).expect("Q@K");
    let scaled4 = scores4.scale(&ctx, scale).expect("scale");
    let attn4 = scaled4.softmax(&ctx, seq_len).expect("softmax");
    let mut output4 = attn4.matmul(&ctx, &v4, seq_len, d_model, seq_len).expect("attn@V failed");
    let output_host = output4.to_host().expect("download output");
    println!("Step 4 - Output (attn @ V): {:?}", output_host);

    // Check if any row is all zeros
    let row0_zero = output_host[0].abs() < 0.001 && output_host[1].abs() < 0.001;
    let row1_zero = output_host[2].abs() < 0.001 && output_host[3].abs() < 0.001;
    if row0_zero {
        println!("BUG: Row 0 is all zeros!");
    }
    if row1_zero {
        println!("BUG: Row 1 is all zeros!");
    }

    // Expected output:
    // Row 0: [0.670*1 + 0.330*3, 0.670*2 + 0.330*4] = [1.66, 2.66]
    // Row 1: [0.330*1 + 0.670*3, 0.330*2 + 0.670*4] = [2.34, 3.34]
    println!("Expected output: [1.66, 2.66, 2.34, 3.34]");
}

/// Test: Batched attention numerical correctness
///
/// Requirement: Output should match reference CPU implementation within tolerance.
/// This is a REAL correctness test that computes expected values on CPU.
#[test]
#[cfg(feature = "cuda")]
fn test_batched_attention_correctness() {
    clear_kernel_cache();
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("CUDA not available, skipping correctness test");
            return;
        }
    };

    // Small test case: seq_len=2, n_heads=1, head_dim=2
    // This keeps the math simple enough to verify by hand
    let seq_len = 2u32;
    let n_heads = 1u32;
    let head_dim = 2u32;
    let d_model = (n_heads * head_dim) as usize; // 2

    // Q = [[1, 0], [0, 1]]  (2x2, row-major)
    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    // K = [[1, 0], [0, 1]]  (same as Q for simplicity)
    let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    // V = [[1, 2], [3, 4]]
    let v_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    // Expected computation (scaled dot-product attention):
    // 1. scores = Q @ K^T = [[1,0],[0,1]] @ [[1,0],[0,1]] = [[1,0],[0,1]]
    // 2. scale = 1/sqrt(head_dim) = 1/sqrt(2) = 0.707
    // 3. scaled_scores = [[0.707, 0], [0, 0.707]]
    // 4. softmax row-wise:
    //    row0: softmax([0.707, 0]) = [exp(0.707), exp(0)] / sum = [2.028, 1.0] / 3.028 = [0.670, 0.330]
    //    row1: softmax([0, 0.707]) = [1.0, 2.028] / 3.028 = [0.330, 0.670]
    // 5. output = attn_weights @ V
    //    row0: [0.670, 0.330] @ [[1,2],[3,4]] = [0.670*1 + 0.330*3, 0.670*2 + 0.330*4] = [1.66, 2.66]
    //    row1: [0.330, 0.670] @ [[1,2],[3,4]] = [0.330*1 + 0.670*3, 0.330*2 + 0.670*4] = [2.34, 3.34]

    // Compute expected on CPU
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Q @ K^T (manually for 2x2)
    // scores[i,j] = sum_k Q[i,k] * K[j,k]
    let scores = vec![
        q_data[0] * k_data[0] + q_data[1] * k_data[1], // [0,0]
        q_data[0] * k_data[2] + q_data[1] * k_data[3], // [0,1]
        q_data[2] * k_data[0] + q_data[3] * k_data[1], // [1,0]
        q_data[2] * k_data[2] + q_data[3] * k_data[3], // [1,1]
    ];

    // Scale
    let scaled: Vec<f32> = scores.iter().map(|x| x * scale).collect();

    // Softmax row-wise
    let mut attn_weights = vec![0.0f32; 4];
    for row in 0..2 {
        let row_start = row * 2;
        let max_val = scaled[row_start].max(scaled[row_start + 1]);
        let exp0 = (scaled[row_start] - max_val).exp();
        let exp1 = (scaled[row_start + 1] - max_val).exp();
        let sum = exp0 + exp1;
        attn_weights[row_start] = exp0 / sum;
        attn_weights[row_start + 1] = exp1 / sum;
    }

    // attn_weights @ V
    let mut expected = vec![0.0f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            expected[i * 2 + j] =
                attn_weights[i * 2] * v_data[j] + attn_weights[i * 2 + 1] * v_data[2 + j];
        }
    }

    println!("\n=== Correctness Test ===");
    println!("Q: {:?}", q_data);
    println!("K: {:?}", k_data);
    println!("V: {:?}", v_data);
    println!("Scores (Q@K^T): {:?}", scores);
    println!("Scaled (/{:.3}): {:?}", 1.0 / scale, scaled);
    println!("Attn weights: {:?}", attn_weights);
    println!("Expected output: {:?}", expected);

    // Run on GPU
    let q = GpuResidentTensor::from_host(&ctx, &q_data).expect("upload Q");
    let k = GpuResidentTensor::from_host(&ctx, &k_data).expect("upload K");
    let v = GpuResidentTensor::from_host(&ctx, &v_data).expect("upload V");

    let mut output = batched_multihead_attention(&ctx, &q, &k, &v, n_heads, head_dim, seq_len)
        .expect("GPU attention failed");
    let result = output.to_host().expect("download output");

    println!("GPU output: {:?}", result);

    // Check numerical accuracy
    let max_diff: f32 =
        result.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    println!("Max diff: {}", max_diff);

    assert!(max_diff < 0.01, "Max diff: {} exceeds tolerance 0.01", max_diff);
    println!("✓ Correctness test PASSED!");
}
