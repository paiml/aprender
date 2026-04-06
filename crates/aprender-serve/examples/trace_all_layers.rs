//! Trace hidden state through all layers to find divergence point
//!
//! Run with: cargo run --example trace_all_layers --release --features cuda

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

    let token_id = 791u32;

    // Get CPU forward result
    let cpu_logits = model.forward(&[token_id])?;
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);

    println!("CPU argmax: {:?}", cpu_argmax);

    // Set up GPU with trace mode
    let mut cuda_model = OwnedQuantizedModelCuda::new(model.clone(), 0)?;
    cuda_model.preload_weights_gpu()?;
    cuda_model.clear_decode_graph();

    // Enable debug for ALL layers
    std::env::set_var("GPU_DEBUG_ALL_LAYERS", "1");
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let mut dummy_cache = realizar::gguf::OwnedQuantizedKVCache::new(
        model.config().num_layers,
        model.config().num_kv_heads * (model.config().hidden_dim / model.config().num_heads),
        100,
    );
    let gpu_logits = cuda_model.forward_gpu_resident(token_id, &mut dummy_cache, 0)?;

    let gpu_argmax = gpu_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);

    println!("\nGPU argmax: {:?}", gpu_argmax);

    // Compare specific logit positions
    println!("\nLogit comparison at key positions:");
    for pos in [0, 16, 74403].iter().filter(|&&p| p < gpu_logits.len()) {
        println!(
            "  pos {}: CPU={:.4}, GPU={:.4}, diff={:.4}",
            pos,
            cpu_logits.get(*pos).unwrap_or(&0.0),
            gpu_logits.get(*pos).unwrap_or(&0.0),
            cpu_logits.get(*pos).unwrap_or(&0.0) - gpu_logits.get(*pos).unwrap_or(&0.0)
        );
    }

    // Compute statistics
    let cpu_mean: f32 = cpu_logits.iter().sum::<f32>() / cpu_logits.len() as f32;
    let gpu_mean: f32 = gpu_logits.iter().sum::<f32>() / gpu_logits.len() as f32;
    let cpu_var: f32 = cpu_logits
        .iter()
        .map(|x| (x - cpu_mean).powi(2))
        .sum::<f32>()
        / cpu_logits.len() as f32;
    let gpu_var: f32 = gpu_logits
        .iter()
        .map(|x| (x - gpu_mean).powi(2))
        .sum::<f32>()
        / gpu_logits.len() as f32;

    println!("\nLogit statistics:");
    println!("  CPU: mean={:.4}, std={:.4}", cpu_mean, cpu_var.sqrt());
    println!("  GPU: mean={:.4}, std={:.4}", gpu_mean, gpu_var.sqrt());
    println!("  Mean diff: {:.4}", cpu_mean - gpu_mean);

    // Check if it's a simple linear transform
    // If GPU = a*CPU + b, then we can estimate a and b
    let mut sum_xy = 0.0f32;
    let mut sum_xx = 0.0f32;
    for (c, g) in cpu_logits.iter().zip(gpu_logits.iter()) {
        let cx = c - cpu_mean;
        let gx = g - gpu_mean;
        sum_xy += cx * gx;
        sum_xx += cx * cx;
    }
    let slope = sum_xy / sum_xx;
    let intercept = gpu_mean - slope * cpu_mean;

    println!(
        "\nLinear regression GPU ≈ {:.4}*CPU + {:.4}",
        slope, intercept
    );

    // Check residuals after linear correction
    let mut max_residual = 0.0f32;
    let mut residual_idx = 0;
    for (i, (c, g)) in cpu_logits.iter().zip(gpu_logits.iter()).enumerate() {
        let predicted = slope * c + intercept;
        let residual = (g - predicted).abs();
        if residual > max_residual {
            max_residual = residual;
            residual_idx = i;
        }
    }
    println!(
        "Max residual: {:.4} at index {}",
        max_residual, residual_idx
    );

    Ok(())
}
