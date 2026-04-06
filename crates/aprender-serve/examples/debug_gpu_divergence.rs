//! Debug GPU vs CPU divergence - find where outputs differ
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

fn main() {
    // PAR-069: Set CUDA_GRAPH_DISABLE BEFORE any CUDA operations to avoid OnceLock caching
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let model_path = "/home/noah/src/aprender/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

    println!("Loading model...");
    let mapped = MappedGGUFModel::from_path(model_path).expect("load");
    let cpu_model = OwnedQuantizedModel::from_mapped(&mapped).expect("cpu model");

    println!("Model config:");
    println!("  hidden_dim: {}", cpu_model.config().hidden_dim);
    println!("  num_heads: {}", cpu_model.config().num_heads);
    println!("  num_kv_heads: {}", cpu_model.config().num_kv_heads);
    println!("  num_layers: {}", cpu_model.config().num_layers);
    println!("  rope_theta: {}", cpu_model.config().rope_theta);

    println!("\nCreating CUDA model...");
    let mut cuda_model = OwnedQuantizedModelCuda::new(cpu_model.clone(), 0).expect("cuda model");

    // Single token test
    let token_id = 9707u32; // "Hello"
    println!("\nTesting with token {} ('Hello')", token_id);

    // 1. Compare embeddings
    let cpu_embed = cpu_model.embed(&[token_id]);
    println!("CPU embedding[0..5]: {:?}", &cpu_embed[..5]);
    println!("CPU embedding sum: {:.6}", cpu_embed.iter().sum::<f32>());

    // 2. Get CPU logits
    let cpu_logits = cpu_model.forward(&[token_id]).expect("cpu forward");
    let cpu_top5: Vec<(usize, f32)> = {
        let mut indexed: Vec<_> = cpu_logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(5);
        indexed
    };
    println!("\nCPU top 5 predictions:");
    for (idx, logit) in &cpu_top5 {
        println!("  Token {}: {:.4}", idx, logit);
    }

    // 3. Get GPU logits (non-graphed path, CUDA_GRAPH_DISABLE set at start)
    cuda_model.preload_weights_gpu().expect("preload");

    // Create cache for GPU path
    let kv_dim = cpu_model.config().num_kv_heads
        * (cpu_model.config().hidden_dim / cpu_model.config().num_heads);
    let mut cache =
        realizar::gguf::OwnedQuantizedKVCache::new(cpu_model.config().num_layers, kv_dim, 16);

    let gpu_logits = cuda_model
        .forward_gpu_resident(token_id, &mut cache, 0)
        .expect("gpu forward");
    let gpu_top5: Vec<(usize, f32)> = {
        let mut indexed: Vec<_> = gpu_logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(5);
        indexed
    };
    println!("\nGPU top 5 predictions:");
    for (idx, logit) in &gpu_top5 {
        println!("  Token {}: {:.4}", idx, logit);
    }

    // 4. Compare
    println!("\n=== COMPARISON ===");
    let cpu_argmax = cpu_top5[0].0;
    let gpu_argmax = gpu_top5[0].0;
    println!("CPU argmax: {} (logit {:.4})", cpu_argmax, cpu_top5[0].1);
    println!("GPU argmax: {} (logit {:.4})", gpu_argmax, gpu_top5[0].1);

    if cpu_argmax == gpu_argmax {
        println!("✅ CPU and GPU agree on top prediction");
    } else {
        println!("❌ CPU and GPU DISAGREE - bug in GPU path!");
    }

    // Check logit correlation
    let mut diff_sum = 0.0f32;
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for i in 0..cpu_logits.len().min(gpu_logits.len()) {
        let diff = (cpu_logits[i] - gpu_logits[i]).abs();
        diff_sum += diff;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    let mean_diff = diff_sum / cpu_logits.len() as f32;
    println!("\nLogit differences:");
    println!("  Mean absolute diff: {:.6}", mean_diff);
    println!("  Max diff: {:.6} at index {}", max_diff, max_diff_idx);
    println!(
        "  CPU[{}] = {:.4}, GPU[{}] = {:.4}",
        max_diff_idx, cpu_logits[max_diff_idx], max_diff_idx, gpu_logits[max_diff_idx]
    );
}
