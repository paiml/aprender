//! CORRECTNESS-002: Controlled Q4K test with known weights
//!
//! Tests Q4K GEMV kernel which is producing wrong output for Q/K projections
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_q4k_controlled

use realizar::quantize::fused_q4k_dot;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CORRECTNESS-002: Controlled Q4K test\n");

    // Test 1: Simple case (256 values, 1 super-block)
    test_simple_case()?;

    // Test 2: Multi-super-block case (1536 values, 6 super-blocks - like Q/K weights)
    test_multi_superblock()?;

    // Test 3: Real Q weight from model
    test_real_q_weight()?;

    Ok(())
}

fn test_simple_case() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("=== Test 1: Simple case (256 values) ===");

    // Q4_K block layout (144 bytes per 256 elements):
    // - d (f16): 2 bytes - scale
    // - dmin (f16): 2 bytes - min value
    // - scales (12 bytes): packed scales for 8 sub-blocks
    // - qs (128 bytes): quantized values, 4 bits each

    let out_dim = 2;
    let in_dim = 256;
    let bytes_per_row = 144; // Q4K uses 144 bytes per 256 values

    let mut q4k_data = vec![0u8; out_dim * bytes_per_row];

    for row in 0..out_dim {
        let row_start = row * bytes_per_row;

        // d = 1.0 in f16 (at offset 0-1)
        let d_f16 = half::f16::from_f32(1.0);
        q4k_data[row_start..row_start + 2].copy_from_slice(&d_f16.to_bits().to_le_bytes());

        // dmin = 0.0 in f16 (at offset 2-3) - so dequant = d * (q & 0xF) * scale
        let dmin_f16 = half::f16::from_f32(0.0);
        q4k_data[row_start + 2..row_start + 4].copy_from_slice(&dmin_f16.to_bits().to_le_bytes());

        // scales (offset 4-15): all 1's - scale=1 for each sub-block
        // Note: scales are packed, 6 bits each
        for i in 4..16 {
            q4k_data[row_start + i] = 0x41; // Simple pattern
        }

        // qs (offset 16-143): quantized values
        // Each byte holds 2 4-bit values. Let's set all to 0x11 (both nibbles = 1)
        for i in 16..144 {
            q4k_data[row_start + i] = 0x11;
        }
    }

    let input: Vec<f32> = vec![1.0; in_dim];

    // CPU reference
    let cpu_result = fused_q4k_dot(&q4k_data[..bytes_per_row], &input)?;
    eprintln!("CPU fused_q4k_dot result: {:.4}", cpu_result);

    // GPU Q4K
    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;
        use trueno_gpu::driver::GpuBuffer;

        let mut executor = CudaExecutor::new(0)?;
        let context = executor.context();

        let weight_buf = GpuBuffer::<u8>::from_host(context, &q4k_data)?;
        let weight_ptr = weight_buf.as_ptr();
        let input_buf = GpuBuffer::<f32>::from_host(context, &input)?;
        let output_buf = GpuBuffer::<f32>::new(context, out_dim)?;

        executor.q4k_gemv_into(
            weight_ptr,
            &input_buf,
            &output_buf,
            out_dim as u32,
            in_dim as u32,
        )?;
        executor.synchronize()?;

        let mut gpu_output = vec![0.0f32; out_dim];
        output_buf.copy_to_host(&mut gpu_output)?;

        eprintln!(
            "GPU row 0: {:.4}, CPU row 0: {:.4}, diff: {:.6}",
            gpu_output[0],
            cpu_result,
            gpu_output[0] - cpu_result
        );

        let match_result = (cpu_result - gpu_output[0]).abs() < 0.1;
        if match_result {
            eprintln!("[simple] PASS");
        } else {
            eprintln!("[simple] FAIL - GPU diverges from CPU!");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("[simple] SKIP - CUDA not enabled");
    }

    Ok(())
}

fn test_multi_superblock() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("\n=== Test 2: Multi-super-block (1536 values, 6 super-blocks) ===");

    let out_dim = 2;
    let in_dim = 1536;
    let num_sb = 6;
    let bytes_per_row = num_sb * 144;

    let mut q4k_data = vec![0u8; out_dim * bytes_per_row];

    for row in 0..out_dim {
        let row_start = row * bytes_per_row;
        for sb in 0..num_sb {
            let sb_start = row_start + sb * 144;

            // d = 1.0
            let d_f16 = half::f16::from_f32(1.0);
            q4k_data[sb_start..sb_start + 2].copy_from_slice(&d_f16.to_bits().to_le_bytes());

            // dmin = 0.0
            let dmin_f16 = half::f16::from_f32(0.0);
            q4k_data[sb_start + 2..sb_start + 4].copy_from_slice(&dmin_f16.to_bits().to_le_bytes());

            // scales: all 1's pattern
            for i in 4..16 {
                q4k_data[sb_start + i] = 0x41;
            }

            // qs: all 0x11
            for i in 16..144 {
                q4k_data[sb_start + i] = 0x11;
            }
        }
    }

    let input: Vec<f32> = vec![1.0; in_dim];

    // CPU reference
    let cpu_result = fused_q4k_dot(&q4k_data[..bytes_per_row], &input)?;
    eprintln!("CPU fused_q4k_dot result: {:.4}", cpu_result);

    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;
        use trueno_gpu::driver::GpuBuffer;

        let mut executor = CudaExecutor::new(0)?;
        let context = executor.context();

        let weight_buf = GpuBuffer::<u8>::from_host(context, &q4k_data)?;
        let weight_ptr = weight_buf.as_ptr();
        let input_buf = GpuBuffer::<f32>::from_host(context, &input)?;
        let output_buf = GpuBuffer::<f32>::new(context, out_dim)?;

        executor.q4k_gemv_into(
            weight_ptr,
            &input_buf,
            &output_buf,
            out_dim as u32,
            in_dim as u32,
        )?;
        executor.synchronize()?;

        let mut gpu_output = vec![0.0f32; out_dim];
        output_buf.copy_to_host(&mut gpu_output)?;

        eprintln!(
            "GPU row 0: {:.4}, CPU row 0: {:.4}, diff: {:.6}",
            gpu_output[0],
            cpu_result,
            gpu_output[0] - cpu_result
        );

        let match_result = (cpu_result - gpu_output[0]).abs() < 0.1;
        if match_result {
            eprintln!("[multi-sb] PASS");
        } else {
            eprintln!("[multi-sb] FAIL");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("[multi-sb] SKIP");
    }

    Ok(())
}

fn test_real_q_weight() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("\n=== Test 3: Real Q weight from model ===");

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Model not found, skipping");
        return Ok(());
    }

    use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let num_heads = model.config().num_heads;
    let q_dim = num_heads * (hidden_dim / num_heads);

    eprintln!("hidden_dim={}, q_dim={}", hidden_dim, q_dim);

    // Get Q weight data
    let layer = &model.layers()[0];
    let (q_data, q_in_dim, q_out_dim) = match &layer.qkv_weight {
        OwnedQKVWeights::Separate { q, .. } => (&q.data, q.in_dim, q.out_dim),
        OwnedQKVWeights::Fused(_) => {
            eprintln!("Fused QKV - cannot test separately");
            return Ok(());
        },
    };

    eprintln!(
        "Q weight: in_dim={}, out_dim={}, data_len={}",
        q_in_dim,
        q_out_dim,
        q_data.len()
    );

    // Compute expected bytes per row for Q4K
    let sb_per_row = q_in_dim.div_ceil(256);
    let bytes_per_row = sb_per_row * 144;
    eprintln!(
        "sb_per_row={}, expected_bytes_per_row={}, actual_bytes_per_row={}",
        sb_per_row,
        bytes_per_row,
        q_data.len() / q_out_dim
    );

    // Test with all-ones input
    let test_input: Vec<f32> = vec![1.0; q_in_dim];

    // CPU Q projection
    let cpu_q: Vec<f32> = (0..q_out_dim)
        .map(|row| {
            let row_start = row * bytes_per_row;
            let row_data = &q_data[row_start..row_start + bytes_per_row];
            fused_q4k_dot(row_data, &test_input).unwrap_or(f32::NAN)
        })
        .collect();

    eprintln!(
        "[CPU] Q first 5: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        cpu_q[0], cpu_q[1], cpu_q[2], cpu_q[3], cpu_q[4]
    );

    // GPU Q projection
    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;
        use trueno_gpu::driver::GpuBuffer;

        let mut executor = CudaExecutor::new(0)?;
        let context = executor.context();

        let weight_buf = GpuBuffer::<u8>::from_host(context, q_data)?;
        let weight_ptr = weight_buf.as_ptr();
        let input_buf = GpuBuffer::<f32>::from_host(context, &test_input)?;
        let output_buf = GpuBuffer::<f32>::new(context, q_out_dim)?;

        executor.q4k_gemv_into(
            weight_ptr,
            &input_buf,
            &output_buf,
            q_out_dim as u32,
            q_in_dim as u32,
        )?;
        executor.synchronize()?;

        let mut gpu_q = vec![0.0f32; q_out_dim];
        output_buf.copy_to_host(&mut gpu_q)?;

        eprintln!(
            "[GPU] Q first 5: [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            gpu_q[0], gpu_q[1], gpu_q[2], gpu_q[3], gpu_q[4]
        );

        // Correlation
        let mut dot = 0.0f64;
        let mut cpu_sq = 0.0f64;
        let mut gpu_sq = 0.0f64;
        for i in 0..q_out_dim {
            let c = cpu_q[i] as f64;
            let g = gpu_q[i] as f64;
            dot += c * g;
            cpu_sq += c * c;
            gpu_sq += g * g;
        }
        let corr = dot / (cpu_sq.sqrt() * gpu_sq.sqrt());
        eprintln!("\nCorrelation: {:.6}", corr);

        if corr > 0.99 {
            eprintln!("[real-Q] PASS");
        } else {
            eprintln!(
                "[real-Q] FAIL - GPU Q4K diverges from CPU (corr={:.4})",
                corr
            );
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("[real-Q] SKIP");
    }

    Ok(())
}
