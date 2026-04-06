//! Q6K GEMV correctness test: CPU vs GPU
//!
//! Tests the Q6K GEMV kernel correctness by comparing CPU and GPU results
//! on the actual model weights.

use realizar::cuda::CudaExecutor;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
use realizar::quantize::fused_q6k_parallel_matvec;

fn main() {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let model_path = "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("load");
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped).expect("cpu model");

    // Get layer 0 FFN down weight (Q6K)
    let layer0 = &cpu_model.layers()[0];
    let down_weight = &layer0.ffn_down_weight;

    println!("\nFFN down_proj weight:");
    println!("  in_dim (K): {}", down_weight.in_dim);
    println!("  out_dim (N): {}", down_weight.out_dim);
    println!("  data len: {} bytes", down_weight.data.len());

    // Expected Q6K size: 210 bytes per 256-element super-block
    let num_super_blocks = down_weight.in_dim.div_ceil(256);
    let expected_size = down_weight.out_dim * num_super_blocks * 210;
    println!("  expected Q6K size: {} bytes", expected_size);
    println!(
        "  actual / expected ratio: {:.4}",
        down_weight.data.len() as f64 / expected_size as f64
    );

    // Create a test input (all 1.0)
    let k = down_weight.in_dim;
    let n = down_weight.out_dim;
    let input: Vec<f32> = vec![1.0f32; k];

    println!("\n=== CPU Q6K GEMV ===");
    let cpu_output = fused_q6k_parallel_matvec(&down_weight.data, &input, k, n).expect("cpu q6k");
    println!("CPU output[0..8]: {:?}", &cpu_output[..8]);
    println!("CPU output sum: {:.6}", cpu_output.iter().sum::<f32>());

    println!("\n=== GPU Q6K GEMV ===");
    let mut executor = CudaExecutor::new(0).expect("cuda executor");

    // Cache the weight
    let cache_key = "test_down";
    executor
        .load_quantized_weights(cache_key, &down_weight.data)
        .expect("upload weight");

    // Run GPU GEMV
    let mut gpu_output = vec![0.0f32; n];
    executor
        .q6k_gemv_cached(cache_key, &input, &mut gpu_output, n as u32, k as u32)
        .expect("gpu q6k");
    println!("GPU output[0..8]: {:?}", &gpu_output[..8]);
    println!("GPU output sum: {:.6}", gpu_output.iter().sum::<f32>());

    // Compare
    println!("\n=== Comparison ===");
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut total_sq_diff = 0.0f64;

    for i in 0..n {
        let diff = (cpu_output[i] - gpu_output[i]).abs();
        total_sq_diff += (diff as f64).powi(2);
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    let rmse = (total_sq_diff / n as f64).sqrt();
    println!("Max diff: {:.6} at index {}", max_diff, max_diff_idx);
    println!("RMSE: {:.6}", rmse);

    // Show worst cases
    println!("\nTop 10 divergences:");
    let mut diffs: Vec<(usize, f32, f32, f32)> = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .enumerate()
        .map(|(i, (c, g))| (i, *c, *g, (c - g).abs()))
        .collect();
    diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    for &(idx, cpu, gpu, diff) in diffs.iter().take(10) {
        println!(
            "  [{:4}]: CPU={:10.4}, GPU={:10.4}, diff={:10.6}",
            idx, cpu, gpu, diff
        );
    }

    // Correlation
    let cpu_mean: f32 = cpu_output.iter().sum::<f32>() / n as f32;
    let gpu_mean: f32 = gpu_output.iter().sum::<f32>() / n as f32;

    let mut cov = 0.0f64;
    let mut var_cpu = 0.0f64;
    let mut var_gpu = 0.0f64;

    for i in 0..n {
        let dc = (cpu_output[i] - cpu_mean) as f64;
        let dg = (gpu_output[i] - gpu_mean) as f64;
        cov += dc * dg;
        var_cpu += dc * dc;
        var_gpu += dg * dg;
    }

    let correlation = cov / (var_cpu.sqrt() * var_gpu.sqrt());
    println!("\nCorrelation: {:.6}", correlation);

    if max_diff < 1.0 && correlation > 0.99 {
        println!("\n✓ Q6K GEMV: CPU and GPU match closely");
    } else {
        println!("\n✗ Q6K GEMV: Significant divergence detected");
    }
}
