//! Debug LM head projection - compare CPU vs GPU with same input
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

fn main() {
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let model_path = "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("load");
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped).expect("cpu model");

    println!("Creating CUDA model...");
    let mut cuda_model = OwnedQuantizedModelCuda::new(cpu_model.clone(), 0).expect("cuda model");
    cuda_model.preload_weights_gpu().expect("preload");

    // Run a single forward pass on both to compare final hidden state
    let token_id = 9707u32;

    // CPU forward to get hidden state before output norm
    let cpu_logits = cpu_model.forward(&[token_id]).expect("cpu forward");

    // Create a known input vector (simple pattern)
    let hidden_dim = cpu_model.config().hidden_dim;
    let vocab_size = cpu_model.lm_head_weight().out_dim;

    println!("\nModel dimensions:");
    println!("  hidden_dim: {}", hidden_dim);
    println!("  vocab_size: {}", vocab_size);

    println!("\n=== Running GPU forward ===");
    let kv_dim = cpu_model.config().num_kv_heads
        * (cpu_model.config().hidden_dim / cpu_model.config().num_heads);
    let mut cache =
        realizar::gguf::OwnedQuantizedKVCache::new(cpu_model.config().num_layers, kv_dim, 16);

    let gpu_logits = cuda_model
        .forward_gpu_resident(token_id, &mut cache, 0)
        .expect("gpu forward");

    // Compare logits
    println!("\n=== Final logits comparison ===");
    println!("CPU logits[0..10]: {:?}", &cpu_logits[..10]);
    println!("GPU logits[0..10]: {:?}", &gpu_logits[..10]);

    let cpu_top: (usize, f32) = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();
    let gpu_top: (usize, f32) = gpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, v)| (i, *v))
        .unwrap();

    println!(
        "\nCPU full forward argmax: {} (logit {:.4})",
        cpu_top.0, cpu_top.1
    );
    println!(
        "GPU full forward argmax: {} (logit {:.4})",
        gpu_top.0, gpu_top.1
    );

    // Compare at specific indices
    println!("\nComparison at CPU top indices:");
    let mut cpu_top_indices: Vec<(usize, f32)> = cpu_logits
        .iter()
        .enumerate()
        .map(|(i, v)| (i, *v))
        .collect();
    cpu_top_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for &(idx, cpu_val) in cpu_top_indices.iter().take(10) {
        let gpu_val = gpu_logits[idx];
        println!(
            "  idx {:6}: CPU={:8.4}, GPU={:8.4}, diff={:8.4}",
            idx,
            cpu_val,
            gpu_val,
            cpu_val - gpu_val
        );
    }

    // Also check where GPU gets high values
    println!("\nComparison at GPU top indices:");
    let mut gpu_top_indices: Vec<(usize, f32)> = gpu_logits
        .iter()
        .enumerate()
        .map(|(i, v)| (i, *v))
        .collect();
    gpu_top_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for &(idx, gpu_val) in gpu_top_indices.iter().take(10) {
        let cpu_val = cpu_logits[idx];
        println!(
            "  idx {:6}: CPU={:8.4}, GPU={:8.4}, diff={:8.4}",
            idx,
            cpu_val,
            gpu_val,
            cpu_val - gpu_val
        );
    }

    // Calculate correlation
    let mut sum_cpu = 0.0f64;
    let mut sum_gpu = 0.0f64;
    let mut sum_cpu_sq = 0.0f64;
    let mut sum_gpu_sq = 0.0f64;
    let mut sum_cross = 0.0f64;
    let n = cpu_logits.len();

    for i in 0..n {
        let c = cpu_logits[i] as f64;
        let g = gpu_logits[i] as f64;
        sum_cpu += c;
        sum_gpu += g;
        sum_cpu_sq += c * c;
        sum_gpu_sq += g * g;
        sum_cross += c * g;
    }

    let mean_cpu = sum_cpu / n as f64;
    let mean_gpu = sum_gpu / n as f64;
    let var_cpu = sum_cpu_sq / n as f64 - mean_cpu * mean_cpu;
    let var_gpu = sum_gpu_sq / n as f64 - mean_gpu * mean_gpu;
    let cov = sum_cross / n as f64 - mean_cpu * mean_gpu;
    let correlation = cov / (var_cpu.sqrt() * var_gpu.sqrt());

    println!("\nStatistics:");
    println!("  CPU mean: {:.4}", mean_cpu);
    println!("  GPU mean: {:.4}", mean_gpu);
    println!("  CPU std: {:.4}", var_cpu.sqrt());
    println!("  GPU std: {:.4}", var_gpu.sqrt());
    println!("  Correlation: {:.6}", correlation);
}
