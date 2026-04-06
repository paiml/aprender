//! CORRECTNESS-002: Controlled Q6K test with known weights
//!
//! Creates Q6K weights where expected output is easy to compute by hand.
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_q6k_controlled

use realizar::quantize::fused_q6k_dot;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CORRECTNESS-002: Controlled Q6K test\n");

    // Test 1: Simple case (256 values, 1 super-block)
    test_simple_case()?;

    // Test 2: Multi-super-block case (1536 values, 6 super-blocks - like LM head)
    test_multi_superblock()?;

    // Test 3: Varying scales (positive and negative)
    test_varying_scales()?;

    Ok(())
}

fn test_simple_case() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Test 1: Simple case (256 values) ===");

    let out_dim = 2;
    let in_dim = 256;
    let bytes_per_row = 210;

    let mut q6k_data = vec![0u8; out_dim * bytes_per_row];

    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        for i in 0..128 {
            q6k_data[row_start + i] = 0x11;
        }
        for i in 128..192 {
            q6k_data[row_start + i] = 0x00;
        }
        for i in 192..208 {
            q6k_data[row_start + i] = 1;
        }
        let d_f16 = half::f16::from_f32(1.0);
        q6k_data[row_start + 208..row_start + 210].copy_from_slice(&d_f16.to_bits().to_le_bytes());
    }

    let input: Vec<f32> = vec![1.0; in_dim];
    let expected = -7936.0;

    run_test(
        &q6k_data,
        &input,
        out_dim,
        in_dim,
        bytes_per_row,
        expected,
        "simple",
    )?;
    Ok(())
}

fn test_multi_superblock() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("\n=== Test 2: Multi-super-block (1536 values, 6 super-blocks) ===");

    let out_dim = 2;
    let in_dim = 1536; // 6 super-blocks
    let num_sb = 6;
    let bytes_per_row = num_sb * 210;

    let mut q6k_data = vec![0u8; out_dim * bytes_per_row];

    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        for sb in 0..num_sb {
            let sb_start = row_start + sb * 210;
            // Same pattern as simple case
            for i in 0..128 {
                q6k_data[sb_start + i] = 0x11;
            }
            for i in 128..192 {
                q6k_data[sb_start + i] = 0x00;
            }
            for i in 192..208 {
                q6k_data[sb_start + i] = 1;
            }
            let d_f16 = half::f16::from_f32(1.0);
            q6k_data[sb_start + 208..sb_start + 210]
                .copy_from_slice(&d_f16.to_bits().to_le_bytes());
        }
    }

    let input: Vec<f32> = vec![1.0; in_dim];
    // Expected: 1536 * (-31) = -47616
    let expected = -47616.0;

    run_test(
        &q6k_data,
        &input,
        out_dim,
        in_dim,
        bytes_per_row,
        expected,
        "multi-sb",
    )?;
    Ok(())
}

fn test_varying_scales() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("\n=== Test 3: Varying scales (positive and negative) ===");

    let out_dim = 2;
    let in_dim = 256;
    let bytes_per_row = 210;

    let mut q6k_data = vec![0u8; out_dim * bytes_per_row];

    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        // All quant values = 32 (which dequants to 32 - 32 = 0)
        // So the dot product should be 0 regardless of scales
        for i in 0..128 {
            q6k_data[row_start + i] = 0x00; // low nibble = 0, high nibble = 0
        }
        for i in 128..192 {
            q6k_data[row_start + i] = 0x22; // qh = 2 in each group -> quant = 0 | (2 << 4) = 32
        }
        // Varying scales: mix of positive and negative
        let scales: [u8; 16] = [
            1, 255, 2, 254, 127, 128, 64, 192, 10, 246, 50, 206, 100, 156, 30, 226,
        ];
        for (i, &s) in scales.iter().enumerate() {
            q6k_data[row_start + 192 + i] = s;
        }
        let d_f16 = half::f16::from_f32(0.5);
        q6k_data[row_start + 208..row_start + 210].copy_from_slice(&d_f16.to_bits().to_le_bytes());
    }

    let input: Vec<f32> = vec![1.0; in_dim];
    // Expected: all quants are 32, which minus 32 = 0, so result should be 0
    let expected = 0.0;

    run_test(
        &q6k_data,
        &input,
        out_dim,
        in_dim,
        bytes_per_row,
        expected,
        "varying-scales",
    )?;
    Ok(())
}

fn run_test(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
    bytes_per_row: usize,
    expected: f32,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // CPU reference
    let mut cpu_results = Vec::new();
    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        let row_data = &q6k_data[row_start..row_start + bytes_per_row];
        let cpu_result = fused_q6k_dot(row_data, input)?;
        cpu_results.push(cpu_result);
    }

    eprintln!(
        "CPU row 0: {:.4}, expected: {:.4}, match: {}",
        cpu_results[0],
        expected,
        (cpu_results[0] - expected).abs() < 1.0
    );

    // GPU Q6K
    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;
        use trueno_gpu::driver::GpuBuffer;

        let mut executor = CudaExecutor::new(0)?;
        let context = executor.context();

        let weight_buf = GpuBuffer::<u8>::from_host(context, q6k_data)?;
        let weight_ptr = weight_buf.as_ptr();
        let input_buf = GpuBuffer::<f32>::from_host(context, input)?;
        let output_buf = GpuBuffer::<f32>::new(context, out_dim)?;

        executor.q6k_gemv_into(
            weight_ptr,
            &input_buf,
            &output_buf,
            out_dim as u32,
            in_dim as u32,
        )?;
        executor.synchronize()?;

        let mut gpu_output = vec![0.0f32; out_dim];
        output_buf.copy_to_host(&mut gpu_output)?;

        let _cpu_match = (cpu_results[0] - expected).abs() < 1.0;
        let _gpu_match = (gpu_output[0] - expected).abs() < 1.0;
        let cpu_gpu_match = (cpu_results[0] - gpu_output[0]).abs() < 0.01;

        eprintln!(
            "GPU row 0: {:.4}, CPU-GPU diff: {:.6}",
            gpu_output[0],
            cpu_results[0] - gpu_output[0]
        );

        if cpu_gpu_match {
            eprintln!("[{}] PASS - GPU matches CPU", name);
        } else {
            eprintln!("[{}] FAIL - GPU diverges from CPU!", name);
            eprintln!("  CPU: {:.4}", cpu_results[0]);
            eprintln!("  GPU: {:.4}", gpu_output[0]);
            eprintln!("  Expected: {:.4}", expected);
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("[{}] SKIP - CUDA not enabled", name);
    }

    Ok(())
}
