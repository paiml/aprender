//! Debug layer 0 divergence between CPU and GPU paths
//!
//! This compares intermediate outputs step-by-step to find exact divergence point.
//!
//! Run with: cargo run --example debug_layer0_divergence --release --features cuda

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the 'cuda' feature. Run with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;

    let token_id = 791u32; // Single token

    // Get embedding (should be identical for both paths)
    let embedding = model.embed(&[token_id]);
    println!(
        "Embedding (first 5): {:?}",
        &embedding[..5.min(embedding.len())]
    );
    println!("Embedding sum: {:.6}", embedding.iter().sum::<f32>());

    // CPU forward first for baseline
    let cpu_logits = model.forward(&[token_id])?;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));
    println!("CPU argmax: {:?}", cpu_argmax);
    if cpu_logits.len() > 16 {
        println!("CPU logit[16]: {:.6}", cpu_logits[16]);
    }

    // Now set up GPU and compare
    println!("\n=== Setting up GPU ===");
    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;

    // Clear any previous decode graph
    cuda_model.clear_decode_graph();

    // GPU forward with detailed debug output
    std::env::set_var("GPU_DEBUG", "1");
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    println!("\n=== GPU Forward ===");
    let mut dummy_cache = realizar::gguf::OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        100,
    );
    let gpu_logits = cuda_model.forward_gpu_resident(token_id, &mut dummy_cache, 0)?;

    println!("\n=== Final Comparison ===");
    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v));

    println!("CPU argmax: {:?}", cpu_argmax);
    println!("GPU argmax: {:?}", gpu_argmax);

    // Compare logits at expected position (token 16)
    if cpu_logits.len() > 16 && gpu_logits.len() > 16 {
        println!("CPU logit[16]: {:.6}", cpu_logits[16]);
        println!("GPU logit[16]: {:.6}", gpu_logits[16]);
    }

    // Correlation
    if cpu_logits.len() == gpu_logits.len() {
        let mean_cpu: f32 = cpu_logits.iter().sum::<f32>() / cpu_logits.len() as f32;
        let mean_gpu: f32 = gpu_logits.iter().sum::<f32>() / gpu_logits.len() as f32;
        let mut cov = 0.0f32;
        let mut var_cpu = 0.0f32;
        let mut var_gpu = 0.0f32;
        for (c, g) in cpu_logits.iter().zip(gpu_logits.iter()) {
            let dc = c - mean_cpu;
            let dg = g - mean_gpu;
            cov += dc * dg;
            var_cpu += dc * dc;
            var_gpu += dg * dg;
        }
        let corr = cov / (var_cpu.sqrt() * var_gpu.sqrt() + 1e-10);
        println!("Correlation: {:.6}", corr);
    }

    // Show first 20 logits
    println!(
        "\nCPU logits[0..20]: {:?}",
        &cpu_logits[..20.min(cpu_logits.len())]
    );
    println!(
        "GPU logits[0..20]: {:?}",
        &gpu_logits[..20.min(gpu_logits.len())]
    );

    if cpu_argmax.map(|(i, _)| i) == gpu_argmax.map(|(i, _)| i) {
        println!("\n✓ CPU and GPU argmax MATCH!");
    } else {
        println!("\n✗ CPU and GPU argmax DIFFER!");
    }

    Ok(())
}
