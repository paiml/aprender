//! CORRECTNESS-002: Direct LM head comparison - isolate kernel vs data issue
//!
//! Tests GPU Q6K kernel with actual LM head weights and controlled input
//!
//! Run with: CUDA_GRAPH_DISABLE=1 cargo run --release --features cuda --example debug_lm_head_direct

use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::fused_q6k_parallel_matvec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("CORRECTNESS-002: Direct LM head GPU vs CPU comparison\n");

    let model_path =
        "/home/noah/src/single-shot-eval/models/raw/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    eprintln!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let hidden_dim = model.config().hidden_dim;
    let vocab_size = model.config().vocab_size;

    eprintln!("hidden_dim={}, vocab_size={}", hidden_dim, vocab_size);

    // Test with all-ones input
    let ones_input: Vec<f32> = vec![1.0; hidden_dim];

    // CPU LM head
    eprintln!("\n=== CPU LM head with all-ones ===");
    let cpu_result = fused_q6k_parallel_matvec(
        &model.lm_head_weight().data,
        &ones_input,
        hidden_dim,
        vocab_size,
    )?;

    let cpu_sum: f32 = cpu_result.iter().sum();
    let cpu_argmax = cpu_result
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "[CPU] sum={:.2}, argmax={}, first 10={:?}",
        cpu_sum,
        cpu_argmax,
        &cpu_result[..10]
    );

    // GPU LM head with same input
    #[cfg(feature = "cuda")]
    {
        use realizar::cuda::CudaExecutor;
        use trueno_gpu::driver::GpuBuffer;

        eprintln!("\n=== GPU LM head with all-ones ===");

        let mut executor = CudaExecutor::new(0)?;
        let context = executor.context();

        // Upload LM head weights
        let weight_buf = GpuBuffer::<u8>::from_host(context, &model.lm_head_weight().data)?;
        let weight_ptr = weight_buf.as_ptr();

        // Upload input
        let input_buf = GpuBuffer::<f32>::from_host(context, &ones_input)?;

        // Output buffer
        let output_buf = GpuBuffer::<f32>::new(context, vocab_size)?;

        // Run Q6K GEMV
        executor.q6k_gemv_into(
            weight_ptr,
            &input_buf,
            &output_buf,
            vocab_size as u32,
            hidden_dim as u32,
        )?;
        executor.synchronize()?;

        let mut gpu_result = vec![0.0f32; vocab_size];
        output_buf.copy_to_host(&mut gpu_result)?;

        let gpu_sum: f32 = gpu_result.iter().sum();
        let gpu_argmax = gpu_result
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        eprintln!(
            "[GPU] sum={:.2}, argmax={}, first 10={:?}",
            gpu_sum,
            gpu_argmax,
            &gpu_result[..10]
        );

        // Compare
        eprintln!("\n=== Comparison ===");
        let mut dot = 0.0f64;
        let mut cpu_sq = 0.0f64;
        let mut gpu_sq = 0.0f64;
        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0;

        for i in 0..vocab_size {
            let c = cpu_result[i] as f64;
            let g = gpu_result[i] as f64;
            dot += c * g;
            cpu_sq += c * c;
            gpu_sq += g * g;
            let diff = (c - g).abs() as f32;
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
        }

        let corr = dot / (cpu_sq.sqrt() * gpu_sq.sqrt());
        eprintln!("Correlation: {:.6}", corr);
        eprintln!(
            "Max diff: {:.4} at idx {} (CPU={:.4}, GPU={:.4})",
            max_diff, max_diff_idx, cpu_result[max_diff_idx], gpu_result[max_diff_idx]
        );

        // Sample comparisons
        eprintln!("\nSample comparisons:");
        for i in [0, 1, 2, 10, 100, 1000, 10000, 50000, 100000, vocab_size - 1] {
            if i < vocab_size {
                let diff = cpu_result[i] - gpu_result[i];
                eprintln!(
                    "  [{}]: CPU={:.4}, GPU={:.4}, diff={:.4}",
                    i, cpu_result[i], gpu_result[i], diff
                );
            }
        }

        // Check if it's a systematic offset
        let total_diff: f32 = (0..vocab_size).map(|i| cpu_result[i] - gpu_result[i]).sum();
        let avg_diff = total_diff / vocab_size as f32;
        eprintln!("\nAverage diff: {:.4}", avg_diff);

        if corr > 0.99 {
            eprintln!("\n[OK] GPU LM head matches CPU");
        } else {
            eprintln!("\n[FAIL] GPU LM head diverges from CPU (corr={:.4})", corr);

            // Try to identify pattern
            eprintln!("\n=== Debugging: Row-by-row comparison ===");
            let sb_per_row = hidden_dim.div_ceil(256);
            let bytes_per_row = sb_per_row * 210;

            eprintln!("sb_per_row={}, bytes_per_row={}", sb_per_row, bytes_per_row);

            // Check first few rows manually
            for row in [0, 1, 2, 1000, 10000] {
                if row < vocab_size {
                    let row_start = row * bytes_per_row;
                    let row_end = row_start + bytes_per_row;
                    let row_data = &model.lm_head_weight().data[row_start..row_end];

                    // CPU single row
                    let cpu_dot = realizar::quantize::fused_q6k_dot(row_data, &ones_input)?;

                    eprintln!(
                        "Row {}: CPU_parallel={:.4}, CPU_single={:.4}, GPU={:.4}",
                        row, cpu_result[row], cpu_dot, gpu_result[row]
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CUDA not enabled");
    }

    Ok(())
}
