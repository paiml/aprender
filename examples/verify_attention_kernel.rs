//! CORRECTNESS-013: Verify multi-warp attention kernel with known inputs
//!
//! This test creates simple Q, K, V values and verifies the attention output.
#![allow(clippy::needless_range_loop)]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::{Kernel, MultiWarpIncrementalAttentionKernel};

    println!("CORRECTNESS-013: Multi-Warp Attention Kernel Verification");
    println!("==========================================================\n");

    // Simple test parameters
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = 128;
    let max_seq_len = 16;
    let seq_len = 2; // Test with 2 positions
    let num_warps = 4;

    // Create CUDA context
    let context = CudaContext::new(0)?;

    // Create Q, K, V buffers
    // Q: [num_heads, head_dim] = [2, 128]
    // K cache: [num_kv_heads, max_seq_len, head_dim] = [2, 16, 128]
    // V cache: same as K

    // Initialize Q with simple values: Q[h][i] = 1.0 for all h, i
    let q_data: Vec<f32> = vec![1.0; num_heads * head_dim];

    // Initialize K cache with:
    // - K[head=0][pos=0][i] = 1.0 for all i
    // - K[head=0][pos=1][i] = 2.0 for all i
    // - K[head=1][pos=0][i] = 1.0 for all i
    // - K[head=1][pos=1][i] = 2.0 for all i
    let mut k_data = vec![0.0f32; num_kv_heads * max_seq_len * head_dim];
    for h in 0..num_kv_heads {
        for pos in 0..seq_len {
            let value = (pos + 1) as f32; // pos=0 -> 1.0, pos=1 -> 2.0
            let head_offset = h * max_seq_len * head_dim;
            let pos_offset = pos * head_dim;
            for i in 0..head_dim {
                k_data[head_offset + pos_offset + i] = value;
            }
        }
    }

    // Initialize V cache with:
    // - V[head=0][pos=0][i] = 0.5 for all i
    // - V[head=0][pos=1][i] = 1.5 for all i
    let mut v_data = vec![0.0f32; num_kv_heads * max_seq_len * head_dim];
    for h in 0..num_kv_heads {
        for pos in 0..seq_len {
            let value = (pos as f32) + 0.5; // pos=0 -> 0.5, pos=1 -> 1.5
            let head_offset = h * max_seq_len * head_dim;
            let pos_offset = pos * head_dim;
            for i in 0..head_dim {
                v_data[head_offset + pos_offset + i] = value;
            }
        }
    }

    // Upload to GPU
    let mut q_buf = GpuBuffer::new(&context, q_data.len())?;
    q_buf.copy_from_host(&q_data)?;

    let mut k_buf = GpuBuffer::new(&context, k_data.len())?;
    k_buf.copy_from_host(&k_data)?;

    let mut v_buf = GpuBuffer::new(&context, v_data.len())?;
    v_buf.copy_from_host(&v_data)?;

    let out_buf = GpuBuffer::new(&context, num_heads * head_dim)?;

    // Create attention kernel
    let scale = 1.0 / (head_dim as f32).sqrt();
    println!("Parameters:");
    println!("  num_heads = {}", num_heads);
    println!("  num_kv_heads = {}", num_kv_heads);
    println!("  head_dim = {}", head_dim);
    println!("  max_seq_len = {}", max_seq_len);
    println!("  seq_len = {}", seq_len);
    println!("  scale = {:.6}", scale);
    println!();

    // Compute expected attention scores manually
    // Q·K = sum(Q[i] * K[i]) for i in 0..head_dim
    // With Q[i] = 1, K[pos=0][i] = 1, K[pos=1][i] = 2:
    // score_0 = sum(1 * 1) * scale = 128 * scale = 128 / sqrt(128) = sqrt(128) ≈ 11.31
    // score_1 = sum(1 * 2) * scale = 256 * scale = 256 / sqrt(128) = 2 * sqrt(128) ≈ 22.63
    let score_0 = (head_dim as f32) * scale;
    let score_1 = (head_dim as f32) * 2.0 * scale;
    println!("Expected scores (before softmax):");
    println!("  score_0 = {:.6}", score_0);
    println!("  score_1 = {:.6}", score_1);

    // Softmax
    let exp_0 = score_0.exp();
    let exp_1 = score_1.exp();
    let sum_exp = exp_0 + exp_1;
    let weight_0 = exp_0 / sum_exp;
    let weight_1 = exp_1 / sum_exp;
    println!("\nExpected softmax weights:");
    println!("  weight_0 = {:.6} (exp({:.2}))", weight_0, score_0);
    println!("  weight_1 = {:.6} (exp({:.2}))", weight_1, score_1);

    // Expected output = weight_0 * V[0] + weight_1 * V[1]
    // V[0][i] = 0.5, V[1][i] = 1.5
    let expected_output = weight_0 * 0.5 + weight_1 * 1.5;
    println!("\nExpected output[i] = {:.6}", expected_output);

    // Build and run kernel
    let kernel = MultiWarpIncrementalAttentionKernel::new(
        max_seq_len as u32,
        head_dim as u32,
        num_heads as u32,
        num_kv_heads as u32,
        num_warps as u32,
    );
    let ptx = kernel.emit_ptx();
    let kernel_name = kernel.name();

    let mut module = CudaModule::from_ptx(&context, &ptx)?;
    let stream = CudaStream::new(&context)?;

    // Launch config: num_heads blocks, 32 * num_warps threads per block
    let config = LaunchConfig::grid_2d(num_heads as u32, 1, 32 * num_warps as u32, 1);

    let mut ptr_q = q_buf.as_ptr();
    let mut ptr_k = k_buf.as_ptr();
    let mut ptr_v = v_buf.as_ptr();
    let mut ptr_out = out_buf.as_ptr();
    let mut seq_len_val = seq_len as u32;

    unsafe {
        stream.launch_kernel(
            &mut module,
            kernel_name,
            &config,
            &mut [
                std::ptr::from_mut(&mut ptr_q) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut ptr_k) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut ptr_v) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut seq_len_val) as *mut std::ffi::c_void,
            ],
        )?;
    }

    stream.synchronize()?;

    // Read back output
    let mut output = vec![0.0f32; num_heads * head_dim];
    out_buf.copy_to_host(&mut output)?;

    println!("\n=== GPU Output ===");
    println!("Head 0 first 5 elements: {:?}", &output[..5]);
    println!("Head 0 element 0: {:.6}", output[0]);
    println!(
        "Head 1 first 5 elements: {:?}",
        &output[head_dim..head_dim + 5]
    );

    // Verify
    let tolerance = 0.001;
    let mut pass = true;
    for i in 0..num_heads * head_dim {
        let diff = (output[i] - expected_output).abs();
        if diff > tolerance {
            println!(
                "MISMATCH at index {}: expected {:.6}, got {:.6}, diff {:.6}",
                i, expected_output, output[i], diff
            );
            pass = false;
            if i > 10 {
                println!("... (showing first 10 mismatches only)");
                break;
            }
        }
    }

    if pass {
        println!(
            "\n✅ All elements match expected value within tolerance {}",
            tolerance
        );
    } else {
        println!("\n❌ VERIFICATION FAILED");
        println!("This indicates a bug in the multi-warp attention kernel.");
    }

    Ok(())
}
